"""
suzume-dsa: ORPO（Odds Ratio Preference Optimization）ステージ

SFT 済みモデルを選好データ（chosen / rejected ペア）で磨く。DPO と違い
**参照モデル不要**（メモリ 2 倍にならない＝単一GPU向き）で、SFT の NLL と
オッズ比ベースの選好項を 1 段で最適化する。

損失: L = NLL(chosen) + λ · L_OR
    L_OR = -log σ( log-odds(chosen) - log-odds(rejected) )
    log-odds(y) = logP(y) - log(1 - P(y)),  logP(y) = 補完トークンの平均対数尤度
（P(y) は幾何平均確率。assistant トークンのみで測る＝chat の完了部分だけ）。

事前学習・SFT と同じ効率化（bf16 / Muon / WSD / 非有限ガード）を共有する。
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from .chat import IGNORE, build_sft_example
from .config import GlmDsaConfig
from .model import SuzumeGlmDsa
from .optim import build_optimizer, build_scheduler
from .train import _mup_lr, load_checkpoint, make_amp, save_checkpoint


def _log1mexp(x: torch.Tensor) -> torch.Tensor:
    """log(1 - exp(x)) を数値安定に（x<0 前提）。P→1 で発散しないよう上側をクランプ。"""
    x = x.clamp_max(-1e-6)
    return torch.where(x > -0.6931471805599453,
                       torch.log(-torch.expm1(x)),
                       torch.log1p(-torch.exp(x)))


def _seq_logp(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """補完（assistant）トークンの平均対数尤度 (B,) を返す。labels=IGNORE は無視。"""
    logp = F.log_softmax(logits[:, :-1].float(), dim=-1)     # (B,T-1,V)
    tgt = labels[:, 1:]                                      # (B,T-1)
    mask = tgt != IGNORE
    tok_logp = logp.gather(-1, tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    tok_logp = tok_logp * mask
    cnt = mask.sum(-1).clamp_min(1)
    return tok_logp.sum(-1) / cnt


def orpo_loss(logits_c, labels_c, logits_r, labels_r, lam: float = 0.1):
    """ORPO 損失 = NLL(chosen) + λ·L_OR。parts に nll/or/acc を返す。"""
    lp_c = _seq_logp(logits_c, labels_c)                    # (B,)
    lp_r = _seq_logp(logits_r, labels_r)
    nll = -lp_c.mean()
    ratio = (lp_c - _log1mexp(lp_c)) - (lp_r - _log1mexp(lp_r))
    l_or = F.softplus(-ratio).mean()                        # -log σ(ratio)
    loss = nll + lam * l_or
    acc = (lp_c > lp_r).float().mean()                      # chosen を選好できている割合
    return loss, {"nll": float(nll.detach()), "or": float(l_or.detach()),
                  "acc": float(acc.detach())}


class ORPODataset:
    """(chosen_conv, rejected_conv) ペアを (ids,labels) 化し、両側を pad してバッチ化。"""

    def __init__(self, pairs: list[tuple[list[dict], list[dict]]], tokenizer,
                 max_len: int, pad_id: int = 0):
        self.ex: list[tuple[list[int], list[int], list[int], list[int]]] = []
        for cc, rc in pairs:
            ic, lc = build_sft_example(cc, tokenizer)
            ir, lr = build_sft_example(rc, tokenizer)
            ic, lc, ir, lr = ic[:max_len], lc[:max_len], ir[:max_len], lr[:max_len]
            if any(l != IGNORE for l in lc) and any(l != IGNORE for l in lr):
                self.ex.append((ic, lc, ir, lr))
        assert self.ex, "教師化できる選好ペアがありません"
        self.pad_id = pad_id

    def __len__(self) -> int:
        return len(self.ex)

    def _pad(self, seqs, fill):
        w = max(len(s) for s in seqs)
        return torch.tensor([s + [fill] * (w - len(s)) for s in seqs])

    def batches(self, batch_size: int, generator: torch.Generator | None = None):
        bs = min(batch_size, len(self))
        while True:
            order = torch.randperm(len(self), generator=generator)
            for i in range(0, len(self) - bs + 1, bs):
                idx = order[i : i + bs]
                ic = [self.ex[j][0] for j in idx]
                lc = [self.ex[j][1] for j in idx]
                ir = [self.ex[j][2] for j in idx]
                lr = [self.ex[j][3] for j in idx]
                yield (self._pad(ic, self.pad_id), self._pad(lc, IGNORE),
                       self._pad(ir, self.pad_id), self._pad(lr, IGNORE))


def _pair_from_row(row: dict):
    """HF 行 → (chosen_conv, rejected_conv)。chosen/rejected が会話 list でも文字列でも可。"""
    ch, rj = row.get("chosen"), row.get("rejected")
    if ch is None or rj is None:
        return None
    if isinstance(ch, list) and isinstance(rj, list):
        return list(ch), list(rj)
    prompt = row.get("prompt") or row.get("question") or row.get("instruction") or ""
    pre = [{"role": "user", "content": str(prompt)}] if prompt else []
    return (pre + [{"role": "assistant", "content": str(ch)}],
            pre + [{"role": "assistant", "content": str(rj)}])


def load_preference_pairs(spec: str, *, max_samples: int | None = None,
                          hf_token: str | None = None, stream: bool = True):
    """HF 選好データセットから (chosen_conv, rejected_conv) ペアを取り込む。"""
    from datasets import load_dataset

    from .data import _parse_hf_spec

    path, config, split, _ = _parse_hf_spec(spec)
    ds = load_dataset(path, config, split=split or "train", streaming=stream, token=hf_token)
    pairs = []
    for i, row in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break
        pair = _pair_from_row(row)
        if pair:
            pairs.append(pair)
    return pairs


def orpo_train(cfg: GlmDsaConfig, pairs, tokenizer, *, steps: int, batch_size: int,
               max_len: int, lr: float, out_dir: str, init_from: str | None = None,
               lam: float = 0.1, optimizer: str = "adamw", muon_lr: float = 0.02,
               lr_schedule: str = "cosine", warmup: int = 0, weight_decay: float = 0.1,
               grad_clip: float = 1.0, mup: bool = False, mup_base_width: int = 256,
               precision: str = "bf16", grad_accum: int = 1,
               log_every: int = 10, ckpt_every: int = 0,
               device: str = "cpu", seed: int = 0) -> SuzumeGlmDsa:
    torch.manual_seed(seed)
    data = ORPODataset(pairs, tokenizer, max_len)

    model = SuzumeGlmDsa(cfg).to(device).train()
    if init_from:                                   # 通常は SFT 済みから継続
        model.load_state_dict(
            torch.load(init_from, map_location="cpu", weights_only=False)["model"])

    eff_lr = _mup_lr(lr, cfg, mup, mup_base_width)
    opt = build_optimizer(model, eff_lr, weight_decay, optimizer, muon_lr)
    sched = build_scheduler(opt, lr_schedule, warmup, steps)
    gen = torch.Generator().manual_seed(seed)
    stream = data.batches(batch_size, generator=gen)
    out = Path(out_dir)
    dev_type, amp_dtype, use_amp, scaler = make_amp(precision, device)

    n_skipped = 0
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        acc = {"nll": 0.0, "or": 0.0, "acc": 0.0}
        for _ in range(grad_accum):
            xc, yc, xr, yr = next(stream)
            xc, yc, xr, yr = xc.to(device), yc.to(device), xr.to(device), yr.to(device)
            with torch.autocast(device_type=dev_type, dtype=amp_dtype, enabled=use_amp):
                lc, _ = model(xc)
                lr_, _ = model(xr)
                loss, parts = orpo_loss(lc, yc, lr_, yr, lam=lam)
            scaler.scale(loss / grad_accum).backward()
            for k in acc:
                acc[k] += parts[k] / grad_accum

        if scaler.is_enabled():
            scaler.unscale_(opt)
        finite = all(torch.isfinite(p.grad).all()
                     for p in model.parameters() if p.grad is not None)
        if not finite:
            n_skipped += 1
            scaler.update()
            opt.zero_grad(set_to_none=True)
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(opt)
        scaler.update()
        sched.step()
        model.commit_router_bias_updates()

        if step % log_every == 0:
            print(f"orpo {step:>6} | nll {acc['nll']:.4f} | or {acc['or']:.4f} "
                  f"| pref_acc {acc['acc']:.2f} | skipped {n_skipped}")
        if ckpt_every and step > 0 and step % ckpt_every == 0:
            save_checkpoint(out / f"orpo_step{step}.pt", model, opt, step)

    save_checkpoint(out / "orpo_model.pt", model, opt, steps)
    return model


def main() -> None:
    import argparse

    from .tokenizer import SPTokenizer

    ap = argparse.ArgumentParser(description="suzume-dsa ORPO（選好最適化・参照モデル不要）")
    ap.add_argument("--sp-model", required=True)
    ap.add_argument("--init-from", required=True, help="SFT 済み checkpoint(.pt)")
    ap.add_argument("--hf-pref-dataset", nargs="+", required=True,
                    help="選好データ（chosen/rejected 列）。複数可・読めない spec はスキップ")
    ap.add_argument("--hf-max-samples", type=int, default=None)
    ap.add_argument("--lam", type=float, default=0.1, help="選好項 L_OR の重み λ")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "muon"])
    ap.add_argument("--muon-lr", type=float, default=0.01)
    ap.add_argument("--lr-schedule", default="cosine", choices=["cosine", "wsd"])
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--precision", default="bf16", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--out", default="output_orpo")
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    tok = SPTokenizer(args.sp_model)
    ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    assert cfg.vocab_size == tok.vocab_size, "語彙不一致（SFT と同じ SP を渡すこと）"

    pairs = []
    for spec in args.hf_pref_dataset:
        try:
            part = load_preference_pairs(spec, max_samples=args.hf_max_samples)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] スキップ（読込失敗）: {spec}  ({type(e).__name__}: {e})")
            continue
        print(f"[orpo] {spec}: {len(part)} ペア")
        pairs.extend(part)
    assert pairs, "有効な選好ペアが 0 件でした。spec を確認してください。"
    print(f"選好ペア 合計 {len(pairs)} 件を読込")

    orpo_train(cfg, pairs, tok, steps=args.steps, batch_size=args.batch_size,
               max_len=args.max_len, lr=args.lr, out_dir=args.out,
               init_from=args.init_from, lam=args.lam, optimizer=args.optimizer,
               muon_lr=args.muon_lr, lr_schedule=args.lr_schedule, warmup=args.warmup,
               precision=args.precision, grad_accum=args.grad_accum,
               ckpt_every=args.ckpt_every, device=args.device)


__all__ = ["orpo_loss", "orpo_train", "ORPODataset", "load_preference_pairs"]
