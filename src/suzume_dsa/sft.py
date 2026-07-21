"""
suzume-dsa: SFT（教師ありファインチューニング）ステージ

事前学習済みモデルを会話データで微調整する。損失は **assistant トークンのみ**
（chat.build_sft_example が labels=-100 でマスク）。事前学習ループと同じ
非有限ガード・checkpoint・optimizer/スケジュールを流用する。

会話データは list[list[turn]]（各 turn = {"role","content"}）、または HF データセット
（messages / conversations カラム）を load_sft_conversations で取り込む。
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from .chat import IGNORE, build_sft_example
from .config import GlmDsaConfig
from .model import SuzumeGlmDsa
from .optim import build_optimizer, build_scheduler
from .train import load_checkpoint, save_checkpoint


class SFTDataset:
    """会話を (input_ids, labels) 例へ変換し、pad してバッチにする。"""

    def __init__(self, conversations: list[list[dict]], tokenizer, max_len: int,
                 pad_id: int = 0):
        self.examples = []
        for conv in conversations:
            ids, labels = build_sft_example(conv, tokenizer)
            ids, labels = ids[:max_len], labels[:max_len]
            if any(l != IGNORE for l in labels):        # 教師トークンが 1 つ以上あるものだけ
                self.examples.append((ids, labels))
        self.max_len = max_len
        self.pad_id = pad_id

    def __len__(self) -> int:
        return len(self.examples)

    def batches(self, batch_size: int, generator: torch.Generator | None = None):
        while True:
            order = torch.randperm(len(self), generator=generator)
            for i in range(0, len(self) - batch_size + 1, batch_size):
                idx = order[i : i + batch_size]
                width = max(len(self.examples[j][0]) for j in idx)
                xb, yb = [], []
                for j in idx:
                    ids, labels = self.examples[j]
                    pad = width - len(ids)
                    xb.append(ids + [self.pad_id] * pad)
                    yb.append(labels + [IGNORE] * pad)
                yield torch.tensor(xb), torch.tensor(yb)


def sft_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """assistant のみ教師化した次トークン CE（labels=-100 は無視）。"""
    shift_logits = logits[:, :-1].reshape(-1, logits.size(-1))
    shift_labels = labels[:, 1:].reshape(-1)
    return F.cross_entropy(shift_logits, shift_labels, ignore_index=IGNORE)


def load_sft_conversations(spec: str, *, split: str | None = None,
                           max_samples: int | None = None, hf_token: str | None = None,
                           stream: bool = True) -> list[list[dict]]:
    """HF データセットから会話（messages / conversations カラム）を取り込む。"""
    from datasets import load_dataset

    from .data import _parse_hf_spec

    path, config, spec_split, _ = _parse_hf_spec(spec)
    ds = load_dataset(path, config, split=split or spec_split or "train",
                      streaming=stream, token=hf_token)
    convs = []
    for i, row in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break
        turns = row.get("messages") or row.get("conversations")
        if turns:
            convs.append(list(turns))
    return convs


def sft_train(cfg: GlmDsaConfig, conversations: list[list[dict]], tokenizer, *,
              steps: int, batch_size: int, max_len: int, lr: float, out_dir: str,
              init_from: str | None = None, optimizer: str = "adamw",
              muon_lr: float = 0.02, lr_schedule: str = "cosine", warmup: int = 0,
              weight_decay: float = 0.1, grad_clip: float = 1.0,
              log_every: int = 10, ckpt_every: int = 0,
              device: str = "cpu", seed: int = 0) -> SuzumeGlmDsa:
    torch.manual_seed(seed)
    data = SFTDataset(conversations, tokenizer, max_len)
    assert len(data) > 0, "教師化できる会話がありません"

    model = SuzumeGlmDsa(cfg).to(device).train()
    if init_from:                                   # 事前学習済みから継続するのが通常
        model.load_state_dict(
            torch.load(init_from, map_location="cpu", weights_only=False)["model"])

    opt = build_optimizer(model, lr, weight_decay, optimizer, muon_lr)
    sched = build_scheduler(opt, lr_schedule, warmup, steps)
    gen = torch.Generator().manual_seed(seed)
    stream = data.batches(batch_size, generator=gen)
    out = Path(out_dir)

    n_skipped = 0
    for step in range(steps):
        x, y = next(stream)
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss = sft_loss(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        finite = torch.isfinite(loss) and all(
            torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if not finite:
            n_skipped += 1
            opt.zero_grad(set_to_none=True)
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        sched.step()
        model.commit_router_bias_updates()

        if step % log_every == 0:
            print(f"sft {step:>6} | loss {float(loss.detach()):.4f} | skipped {n_skipped}")
        if ckpt_every and step > 0 and step % ckpt_every == 0:
            save_checkpoint(out / f"sft_step{step}.pt", model, opt, step)

    save_checkpoint(out / "sft_model.pt", model, opt, steps)
    return model


def main() -> None:
    import argparse

    from .tokenizer import SPTokenizer

    ap = argparse.ArgumentParser(description="suzume-dsa SFT")
    ap.add_argument("--sp-model", required=True, help="SentencePiece .model")
    ap.add_argument("--init-from", required=True, help="事前学習済み checkpoint(.pt)")
    ap.add_argument("--hf-sft-dataset", required=True,
                    help='会話データ "path[:config][:split]"（messages/conversations カラム）')
    ap.add_argument("--hf-max-samples", type=int, default=None)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "muon"])
    ap.add_argument("--lr-schedule", default="cosine", choices=["cosine", "wsd"])
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--out", default="output_sft")
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    tok = SPTokenizer(args.sp_model)
    # 事前学習 checkpoint に保存された cfg をそのまま使う（プリセット非依存で
    # 0.5B でも 4B でも --init-from の寸法に一致させる。4B ハードコードは壊れる）。
    ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    assert cfg.vocab_size == tok.vocab_size, (
        f"語彙不一致: checkpoint {cfg.vocab_size} vs sp-model {tok.vocab_size}"
        "（事前学習と同じ SentencePiece を渡すこと）")
    convs = load_sft_conversations(args.hf_sft_dataset, max_samples=args.hf_max_samples)
    print(f"会話 {len(convs)} 件を読込")
    sft_train(cfg, convs, tok, steps=args.steps, batch_size=args.batch_size,
              max_len=args.max_len, lr=args.lr, out_dir=args.out,
              init_from=args.init_from, optimizer=args.optimizer,
              lr_schedule=args.lr_schedule, warmup=args.warmup,
              ckpt_every=args.ckpt_every, device=args.device)


__all__ = ["SFTDataset", "sft_loss", "sft_train", "load_sft_conversations"]


if __name__ == "__main__":
    main()
