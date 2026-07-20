"""
suzume-dsa: VRAM 実測キャリブレーション

学習 VRAM は `固定分 A(重み+勾配+optimizer+CUDA) + batch × B(1サンプルの活性化)` と
ほぼ直線。batch=1,2 の 2 点で実測ピークを測り、直線を復元して「目標 VRAM に収まる
最大 batch」を逆算する。

原理:  V(1)=A+B, V(2)=A+2B  ⇒  B=V(2)-V(1), A=2V(1)-V(2)  ⇒  b_max=floor((T-A)/B)

注意:
  * 活性化は block_size に比例 → カリキュラム末尾の最大 block で測ること（既定 4096）。
  * 測るのは 1 ステップ全体(fwd+bwd+optimizer)の peak（backward 中がピーク）。
  * precision / bf16-weights / optimizer は本番と同じ設定で。

使い方（count_params.py と同じ config 引数 + 追加フラグ）:
    uv run tools/vram_calibrate.py --n-embd 2048 --n-layer 24 --n-expert 24 \
        --precision bf16 --block-size 4096
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from suzume_dsa import SuzumeGlmDsa  # noqa: E402
from suzume_dsa.train import compute_loss  # noqa: E402

# count_params.py の config パーサを再利用
sys.path.insert(0, str(Path(__file__).resolve().parent))
from count_params import build_parser as _model_parser, config_from_args  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = _model_parser()
    g = p.add_argument_group("VRAM キャリブレーション")
    g.add_argument("--block-size", type=int, default=4096,
                   help="実測系列長。カリキュラム末尾の最大 block に合わせる")
    g.add_argument("--probe-batches", type=int, nargs="+", default=[1, 2])
    g.add_argument("--target-gb", type=float, default=None)
    g.add_argument("--safety-frac", type=float, default=0.90)
    g.add_argument("--precision", default="bf16", choices=["fp32", "bf16", "fp16"])
    g.add_argument("--lr", type=float, default=1.5e-4)
    return p


def _one_step(model, opt, vocab, block, batch, device, use_amp, amp_dtype, mtp_coef):
    x = torch.randint(0, vocab, (batch, block), device=device)
    y = torch.randint(0, vocab, (batch, block), device=device)
    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        logits, info = model(x)
        loss, _ = compute_loss(logits, info, y, mtp_coef)
    loss.backward()
    model.commit_router_bias_updates()
    opt.step()


def measure(model, opt, vocab, block, batch, device, use_amp, amp_dtype, mtp_coef) -> float:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    _one_step(model, opt, vocab, block, batch, device, use_amp, amp_dtype, mtp_coef)
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_reserved(device) / 1e9


def _fit_line(xs, ys) -> tuple[float, float]:
    n = len(xs); sx = sum(xs); sy = sum(ys)
    sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    B = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    A = (sy - B * sx) / n
    return A, B


def main() -> None:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("VRAM 実測には CUDA が必要です（GPU 上で実行してください）。")
    device = torch.device("cuda")
    cfg = config_from_args(args)

    model = SuzumeGlmDsa(cfg).to(device)
    if args.bf16_weights:
        model.to(dtype=torch.bfloat16)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    use_amp = args.precision in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16

    total_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
    target_gb = args.target_gb or total_gb * args.safety_frac
    probes = sorted(set(int(b) for b in args.probe_batches))
    print(f"GPU: {torch.cuda.get_device_name(device)}  総VRAM {total_gb:.1f} GB")
    print(f"構成: block={args.block_size} precision={args.precision} "
          f"bf16_weights={args.bf16_weights}  目標 {target_gb:.1f} GB\n")

    # warmup（optimizer state + CUDA context を固定分 A に含める）
    _one_step(model, opt, cfg.vocab_size, args.block_size, probes[0],
              device, use_amp, amp_dtype, cfg.mtp_loss_coef)
    torch.cuda.synchronize(device)

    xs, ys = [], []
    for b in probes:
        try:
            gb = measure(model, opt, cfg.vocab_size, args.block_size, b,
                         device, use_amp, amp_dtype, cfg.mtp_loss_coef)
        except torch.cuda.OutOfMemoryError:
            print(f"  batch={b:>3}: OOM"); torch.cuda.empty_cache(); continue
        xs.append(b); ys.append(gb)
        print(f"  batch={b:>3}: peak {gb:6.2f} GB")

    if len(xs) < 2:
        raise SystemExit("\n2 点以上測れませんでした。probe-batches を小さくして再実行を。")
    A, B = _fit_line(xs, ys)
    b_max = int((target_gb - A) // B)
    print(f"\nVRAM(batch) ≈ {A:.2f} + {B:.2f} × batch (GB)")
    print(f"  固定分 A={A:.2f} GB / 1サンプル B={B:.2f} GB (block={args.block_size})")
    if b_max < 1:
        print("\n⚠ 目標では batch=1 も収まりません。block を下げるか目標を上げてください。")
    else:
        print(f"\n★ 目標 {target_gb:.1f} GB に収まる最大 batch = {b_max} "
              f"(予測 {A + B * b_max:.1f} GB)")


if __name__ == "__main__":
    main()
