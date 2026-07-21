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
from .train import _mup_lr, compute_loss, load_checkpoint, make_amp, save_checkpoint


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


def _parse_sft_field(field: str | None):
    """spec 4 番目のフィールドを解釈して (mode, arg) を返す。

    - None                                 → ("auto", None)   messages/conversations 自動
    - "conversations" / "messages" 等の名前 → ("column", 名前)  その列を会話リストとして使う
    - "format=harmony"                     → ("harmony", None) harmony 文字列をパース
    - "instruction=question,output=answer" → ("mapping", {..}) 別カラムを 1 会話へ組み立て
    """
    if not field:
        return "auto", None
    if "=" not in field:
        return "column", field
    pairs = {}
    for kv in field.split(","):
        k, _, v = kv.partition("=")
        pairs[k.strip()] = v.strip()
    if pairs.get("format") == "harmony":
        return "harmony", None
    return "mapping", pairs


def _row_to_turns(row: dict, mode: str, arg, *, normalize):
    from .chat import conversation_from_columns, parse_harmony

    if mode == "mapping":
        return conversation_from_columns(row, arg)
    if mode == "harmony":
        raw = (row.get("messages") or row.get("conversations")
               or row.get("text") or row.get("output") or "")
        if isinstance(raw, str):
            return parse_harmony(raw)
        return [normalize(t) for t in raw] if raw else []
    if mode == "column":
        turns = row.get(arg)
        return list(turns) if turns else []
    return list(row.get("messages") or row.get("conversations") or [])


def load_sft_conversations(spec: str, *, split: str | None = None,
                           max_samples: int | None = None, hf_token: str | None = None,
                           stream: bool = True) -> list[list[dict]]:
    """HF データセットから会話を取り込む。spec 書式 "path[:config][:split][:field]"。

    field で 3 形式に対応: 会話列名 / "format=harmony" / 列マッピング DSL
    （"instruction=question,reasoning=reasoning,output=answer"）。省略時は
    messages / conversations 列を自動検出（従来互換）。
    """
    from datasets import load_dataset

    from .chat import normalize_turn
    from .data import _parse_hf_spec

    path, config, spec_split, field = _parse_hf_spec(spec)
    mode, arg = _parse_sft_field(field)
    ds = load_dataset(path, config, split=split or spec_split or "train",
                      streaming=stream, token=hf_token)
    convs = []
    for i, row in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break
        turns = _row_to_turns(row, mode, arg, normalize=normalize_turn)
        if turns:
            convs.append(list(turns))
    return convs


def sft_train(cfg: GlmDsaConfig, conversations: list[list[dict]], tokenizer, *,
              steps: int, batch_size: int, max_len: int, lr: float, out_dir: str,
              init_from: str | None = None, optimizer: str = "adamw",
              muon_lr: float = 0.02, lr_schedule: str = "cosine", warmup: int = 0,
              weight_decay: float = 0.1, grad_clip: float = 1.0,
              select_topp: float = 1.0, mup: bool = False, mup_base_width: int = 256,
              precision: str = "bf16", grad_accum: int = 1,
              log_every: int = 10, ckpt_every: int = 0,
              device: str = "cpu", seed: int = 0) -> SuzumeGlmDsa:
    """SFT ループ。事前学習と同じ効率化を共有する: MTP 補助損失（cfg.mtp_depth>0 なら
    checkpoint から継承した MTP を assistant マスク付きで学習）、選択的 backprop
    （select_topp<1.0）、μP 幅転移（mup）、Muon/WSD、非有限ガード。"""
    torch.manual_seed(seed)
    data = SFTDataset(conversations, tokenizer, max_len)
    assert len(data) > 0, "教師化できる会話がありません"

    model = SuzumeGlmDsa(cfg).to(device).train()
    if init_from:                                   # 事前学習済みから継続するのが通常
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
        main_acc = mtp_acc = 0.0
        for _ in range(grad_accum):
            x, y = next(stream)
            x, y = x.to(device), y.to(device)
            # メイン/MTP を「位置 i で token i+1 を予測」に揃える（labels を左シフト）。
            # 非教師位置は IGNORE のままなので compute_loss がマスクする（assistant のみ学習）。
            targets = torch.full_like(y, IGNORE)
            targets[:, :-1] = y[:, 1:]
            with torch.autocast(device_type=dev_type, dtype=amp_dtype, enabled=use_amp):
                logits, info = model(x)
                loss, parts = compute_loss(logits, info, targets, cfg.mtp_loss_coef,
                                           select_topp=select_topp, ignore_index=IGNORE)
            scaler.scale(loss / grad_accum).backward()
            main_acc += parts["main"] / grad_accum
            mtp_acc += parts["mtp"] / grad_accum

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
            print(f"sft {step:>6} | loss {main_acc:.4f} | mtp {mtp_acc:.4f} "
                  f"| skipped {n_skipped}")
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
    ap.add_argument("--hf-sft-dataset", nargs="+", required=True,
                    help='会話データ "path[:config][:split]"（messages/conversations カラム）。'
                         '複数指定で全部を連結。読めない spec は警告してスキップ')
    ap.add_argument("--hf-max-samples", type=int, default=None)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "muon"])
    ap.add_argument("--muon-lr", type=float, default=0.02)
    ap.add_argument("--lr-schedule", default="cosine", choices=["cosine", "wsd"])
    ap.add_argument("--warmup", type=int, default=100)
    # 事前学習と共有する効率化レバー（MTP は checkpoint の cfg.mtp_depth から自動継承）
    ap.add_argument("--select-topp", type=float, default=1.0,
                    help="選択的 backprop: assistant トークンの損失上位この割合だけ学習（<1.0 で有効）")
    ap.add_argument("--mup", action="store_true", help="μP 幅転移（LR を基準幅で調整）")
    ap.add_argument("--mup-base-width", type=int, default=256)
    ap.add_argument("--precision", default="bf16", choices=["fp32", "bf16", "fp16"],
                    help="混合精度（CUDA のみ有効。bf16 推奨）")
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="勾配累積のマイクロバッチ数（実効 batch = batch×accum）")
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
    specs = args.hf_sft_dataset
    if isinstance(specs, str):
        specs = [specs]
    convs = []
    for spec in specs:
        try:
            part = load_sft_conversations(spec, max_samples=args.hf_max_samples)
        except Exception as e:  # noqa: BLE001 - 1 件ダメでも継続
            print(f"[warn] スキップ（読込失敗）: {spec}  ({type(e).__name__}: {e})")
            continue
        if not part:
            print(f"[warn] スキップ（会話0＝messages/conversations 列なし？）: {spec}")
            continue
        print(f"[sft] {spec}: {len(part)} 会話")
        convs.extend(part)
    assert convs, "有効な会話データが 0 件でした。spec を確認してください。"
    print(f"会話 合計 {len(convs)} 件を読込")
    if cfg.mtp_depth > 0:
        print(f"MTP 有効（depth={cfg.mtp_depth}）: 事前学習から継承した MTP を SFT でも学習")
    sft_train(cfg, convs, tok, steps=args.steps, batch_size=args.batch_size,
              max_len=args.max_len, lr=args.lr, out_dir=args.out,
              init_from=args.init_from, optimizer=args.optimizer,
              muon_lr=args.muon_lr, lr_schedule=args.lr_schedule, warmup=args.warmup,
              select_topp=args.select_topp, mup=args.mup, mup_base_width=args.mup_base_width,
              precision=args.precision, grad_accum=args.grad_accum,
              ckpt_every=args.ckpt_every, device=args.device)


__all__ = ["SFTDataset", "sft_loss", "sft_train", "load_sft_conversations"]


if __name__ == "__main__":
    main()
