"""
suzume-dsa: 最小 pretrain ループ (a)

data -> model -> loss(次トークン CE + MTP 補助) -> step -> checkpoint -> (export)
を通す最小構成。最初から配線しておくもの:

  * MTP 補助損失（モデルが mtp_depth>0 のとき自動で出す logits を使う）
  * aux-free MoE bias の commit（毎ステップ後）
  * 非有限勾配ガード（suzume-muon の nan 事件の教訓。loss/grad が壊れたら step を捨てる）
  * checkpoint 保存 + --resume

系列長カリキュラム・μP・Muon・動的データ選別などは次段で足す（docs/training-efficiency.md）。
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import TINY, GlmDsaConfig
from .data import (
    ByteTokenizer, TokenWindows, block_size_at, load_corpus_tokens,
    parse_block_size_schedule,
)
from .fim import build_fim_batch
from .model import SuzumeGlmDsa
from .optim import build_optimizer, build_scheduler


def compute_loss(logits: torch.Tensor, info: dict, targets: torch.Tensor,
                 mtp_coef: float, select_topp: float = 1.0) -> tuple[torch.Tensor, dict]:
    """次トークン CE（メイン）+ MTP 補助（あれば）。

    logits: (B, T, V)、targets: (B, T)（= 入力を 1 つずらしたもの）。
    メインは位置 i で targets[i] を予測。MTP モジュール m(1始まり) は更に m 個先を予測し、
    logits 長 T-m を targets[:, m:] に合わせて損失を取る。

    select_topp<1.0 で **選択的 backprop**: トークンごとの損失の上位 topp 割合だけを
    メイン損失に残す（易しいトークンに勾配を使わない。Selective Backprop 系）。
    """
    V = logits.size(-1)
    tok_loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), reduction="none")
    if select_topp < 1.0:
        k = max(1, int(select_topp * tok_loss.numel()))
        keep = torch.topk(tok_loss, k).values
        main = keep.mean()
    else:
        main = tok_loss.mean()

    mtp = torch.zeros((), device=logits.device)
    for m, ml in enumerate(info.get("mtp_logits", []), start=1):
        tgt = targets[:, m:]                       # m 個先へずらした教師
        pred = ml[:, : tgt.size(1)]                # 長さを教師に合わせて切る
        if tgt.numel() > 0:
            mtp = mtp + F.cross_entropy(pred.reshape(-1, V), tgt.reshape(-1))

    loss = main + mtp_coef * mtp
    return loss, {"main": float(main.detach()), "mtp": float(mtp.detach())}


def load_init_weights(model: SuzumeGlmDsa, path: str) -> None:
    """深さ成長（progressive stacking）: 層数の異なる checkpoint を継承する。

    共有ベース・埋め込み等の一致キーだけ strict=False で読み込み、増えた層は
    fresh のまま（delta が zero-init で恒等に近い接続なので、そのまま学習を継続できる）。
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    own = model.state_dict()
    loadable = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    missing = model.load_state_dict(loadable, strict=False)
    print(f"init-from {path}: {len(loadable)}/{len(own)} テンソル継承 "
          f"（欠け {len(missing.missing_keys)} は fresh）")


def save_checkpoint(path: Path, model, opt, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "step": step, "cfg": model.cfg}, path)


def load_checkpoint(path: Path, model, opt) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])
    return ckpt["step"]


def _mup_lr(lr: float, cfg: GlmDsaConfig, mup: bool, base_width: int) -> float:
    """μP（μTransfer）簡易版: 最適 LR は行列パラメータで概ね 1/width に比例する。

    基準幅 base_width で調整した lr を、実際の n_embd に合わせてスケールする。
    厳密な per-layer μP ではなく、幅を跨いだ探索コスト削減のための実用近似。
    """
    return lr * (base_width / cfg.n_embd) if mup else lr


def train(cfg: GlmDsaConfig, corpus, *, steps: int, batch_size: int, block_size: int,
          lr: float, out_dir: str, tokenizer=None, resume: str | None = None,
          init_from: str | None = None, block_size_schedule: str | None = None,
          optimizer: str = "adamw", muon_lr: float = 0.02,
          lr_schedule: str = "cosine", warmup: int = 0, wsd_decay_frac: float = 0.2,
          mup: bool = False, mup_base_width: int = 256,
          fim_rate: float = 0.0, fim_spm_prob: float = 0.5, select_topp: float = 1.0,
          weight_decay: float = 0.1, grad_clip: float = 1.0,
          log_every: int = 10, ckpt_every: int = 200,
          device: str = "cpu", seed: int = 0) -> SuzumeGlmDsa:
    torch.manual_seed(seed)
    tokenizer = tokenizer or ByteTokenizer()
    tokens = load_corpus_tokens(corpus, tokenizer)
    data = TokenWindows(tokens)

    # 系列長カリキュラム: 指定があれば step ごとに block を決める純関数、無ければ固定。
    schedule = parse_block_size_schedule(block_size_schedule, steps) if block_size_schedule else None

    # FIM: tokenizer が FIM センチネルを持つときだけ有効化。
    fim_ids = None
    if fim_rate > 0.0 and hasattr(tokenizer, "piece_id"):
        from .tokenizer import SPECIAL_TOKENS
        fim_ids = tuple(tokenizer.piece_id(t) for t in SPECIAL_TOKENS[:3])

    model = SuzumeGlmDsa(cfg).to(device).train()
    if init_from:                                    # 深さ成長: 前段 checkpoint を継承
        load_init_weights(model, init_from)

    eff_lr = _mup_lr(lr, cfg, mup, mup_base_width)
    opt = build_optimizer(model, eff_lr, weight_decay, optimizer, muon_lr)
    sched = build_scheduler(opt, lr_schedule, warmup, steps, wsd_decay_frac)

    start = load_checkpoint(Path(resume), model, opt) if resume else 0
    gen = torch.Generator().manual_seed(seed)
    out = Path(out_dir)

    n_skipped = 0
    for step in range(start, steps):
        cur_block = block_size_at(step, schedule) if schedule else block_size
        if fim_ids is not None:
            x, y = build_fim_batch(tokens, batch_size, cur_block, fim_ids, gen,
                                   fim_rate, fim_spm_prob)
        else:
            x, y = data.sample(batch_size, cur_block, generator=gen)
        x, y = x.to(device), y.to(device)
        logits, info = model(x)
        loss, parts = compute_loss(logits, info, y, cfg.mtp_loss_coef, select_topp)

        opt.zero_grad(set_to_none=True)
        loss.backward()

        # 非有限ガード: loss か勾配が壊れていたら、その step は捨てる（汚染を広げない）
        finite = torch.isfinite(loss) and all(
            torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if not finite:
            n_skipped += 1
            opt.zero_grad(set_to_none=True)
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        sched.step()
        model.commit_router_bias_updates()          # aux-free ロードバランス

        if step % log_every == 0:
            print(f"step {step:>6} | block {cur_block:>4} | loss {parts['main']:.4f} "
                  f"| mtp {parts['mtp']:.4f} | lr {sched.get_last_lr()[0]:.2e} "
                  f"| skipped {n_skipped}")
        if ckpt_every and step > start and step % ckpt_every == 0:
            save_checkpoint(out / f"checkpoint_step{step}.pt", model, opt, step)

    save_checkpoint(out / "model.pt", model, opt, steps)
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description="suzume-dsa 最小 pretrain")
    ap.add_argument("--corpus", required=True, help="テキストファイル / 文字列")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--block-size-schedule", default=None,
                    help='系列長カリキュラム。例 "0%%:64,50%%:128,80%%:256"（絶対step混在可）')
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="output")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--init-from", default=None, help="深さ成長: 前段 checkpoint を継承")
    ap.add_argument("--device", default="cpu")
    # 学習効率化
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "muon"])
    ap.add_argument("--muon-lr", type=float, default=0.02)
    ap.add_argument("--lr-schedule", default="cosine", choices=["cosine", "wsd"])
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--wsd-decay-frac", type=float, default=0.2)
    ap.add_argument("--mup", action="store_true", help="μP 幅転移（LR を基準幅で調整）")
    ap.add_argument("--mup-base-width", type=int, default=256)
    ap.add_argument("--fim", type=float, default=0.0, dest="fim_rate",
                    help="FIM 化する窓の割合（SentencePiece 語彙のみ有効）")
    ap.add_argument("--select-topp", type=float, default=1.0,
                    help="選択的 backprop: 損失上位この割合のトークンだけ学習（<1.0 で有効）")
    ap.add_argument("--sp-model", default=None, help="SentencePiece .model（本番語彙）")
    args = ap.parse_args()

    # tokenizer: --sp-model 指定時は本番 SentencePiece、無ければ疎通用バイト単位。
    cfg, tok = TINY, None
    if args.sp_model:
        from .config import SUZUME_4B
        from .tokenizer import SPTokenizer
        tok = SPTokenizer(args.sp_model)
        cfg = replace(SUZUME_4B, vocab_size=tok.vocab_size)

    train(cfg, args.corpus, steps=args.steps, batch_size=args.batch_size,
          block_size=args.block_size, block_size_schedule=args.block_size_schedule,
          lr=args.lr, out_dir=args.out, resume=args.resume, init_from=args.init_from,
          tokenizer=tok, optimizer=args.optimizer, muon_lr=args.muon_lr,
          lr_schedule=args.lr_schedule, warmup=args.warmup,
          wsd_decay_frac=args.wsd_decay_frac, mup=args.mup,
          mup_base_width=args.mup_base_width, fim_rate=args.fim_rate,
          select_topp=args.select_topp, device=args.device)


if __name__ == "__main__":
    main()
