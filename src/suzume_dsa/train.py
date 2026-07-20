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
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import TINY, GlmDsaConfig
from .data import (
    ByteTokenizer, TokenWindows, block_size_at, load_corpus_tokens,
    parse_block_size_schedule,
)
from .model import SuzumeGlmDsa


def compute_loss(logits: torch.Tensor, info: dict, targets: torch.Tensor,
                 mtp_coef: float) -> tuple[torch.Tensor, dict]:
    """次トークン CE（メイン）+ MTP 補助（あれば）。

    logits: (B, T, V)、targets: (B, T)（= 入力を 1 つずらしたもの）。
    メインは位置 i で targets[i] を予測。MTP モジュール m(1始まり) は更に m 個先を予測し、
    logits 長 T-m を targets[:, m:] に合わせて損失を取る。
    """
    V = logits.size(-1)
    main = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))

    mtp = torch.zeros((), device=logits.device)
    for m, ml in enumerate(info.get("mtp_logits", []), start=1):
        tgt = targets[:, m:]                       # m 個先へずらした教師
        pred = ml[:, : tgt.size(1)]                # 長さを教師に合わせて切る
        if tgt.numel() > 0:
            mtp = mtp + F.cross_entropy(pred.reshape(-1, V), tgt.reshape(-1))

    loss = main + mtp_coef * mtp
    return loss, {"main": float(main.detach()), "mtp": float(mtp.detach())}


def save_checkpoint(path: Path, model, opt, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "step": step, "cfg": model.cfg}, path)


def load_checkpoint(path: Path, model, opt) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])
    return ckpt["step"]


def train(cfg: GlmDsaConfig, corpus, *, steps: int, batch_size: int, block_size: int,
          lr: float, out_dir: str, tokenizer=None, resume: str | None = None,
          block_size_schedule: str | None = None,
          grad_clip: float = 1.0, log_every: int = 10, ckpt_every: int = 200,
          device: str = "cpu", seed: int = 0) -> SuzumeGlmDsa:
    torch.manual_seed(seed)
    tokenizer = tokenizer or ByteTokenizer()
    tokens = load_corpus_tokens(corpus, tokenizer)
    data = TokenWindows(tokens)

    # 系列長カリキュラム: 指定があれば step ごとに block を決める純関数、無ければ固定。
    schedule = parse_block_size_schedule(block_size_schedule, steps) if block_size_schedule else None

    model = SuzumeGlmDsa(cfg).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)

    start = load_checkpoint(Path(resume), model, opt) if resume else 0
    gen = torch.Generator().manual_seed(seed)
    out = Path(out_dir)

    n_skipped = 0
    for step in range(start, steps):
        cur_block = block_size_at(step, schedule) if schedule else block_size
        x, y = data.sample(batch_size, cur_block, generator=gen)
        x, y = x.to(device), y.to(device)
        logits, info = model(x)
        loss, parts = compute_loss(logits, info, y, cfg.mtp_loss_coef)

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
        model.commit_router_bias_updates()          # aux-free ロードバランス

        if step % log_every == 0:
            print(f"step {step:>6} | block {cur_block:>4} | loss {parts['main']:.4f} "
                  f"| mtp {parts['mtp']:.4f} | skipped {n_skipped}")
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
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    # 既定は TINY（バイト単位トークナイザ、vocab=256）。本番は cfg と tokenizer を差し替える。
    train(TINY, args.corpus, steps=args.steps, batch_size=args.batch_size,
          block_size=args.block_size, block_size_schedule=args.block_size_schedule,
          lr=args.lr, out_dir=args.out, resume=args.resume, device=args.device)


if __name__ == "__main__":
    main()
