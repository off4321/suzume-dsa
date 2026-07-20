"""
suzume-dsa: パラメータ数の見積もりツール（学習前のサイズ確認用）

meta デバイスで構築するので 4B / 10B 構成でもメモリをほぼ使わずに
total / active を数えられる。本番の n_embd / n_layer / n_expert を決める前のサイズ設計に。

使い方:
    uv run tools/count_params.py                          # SUZUME_4B 既定
    uv run tools/count_params.py --n-embd 2560 --n-expert 32 --n-expert-used 6
"""

import argparse
import sys
from dataclasses import fields, replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from suzume_dsa import SUZUME_4B, GlmDsaConfig, SuzumeGlmDsa  # noqa: E402

# CLI で上書きできる config レバー（GlmDsaConfig のフィールド名）
_LEVERS = [f.name for f in fields(GlmDsaConfig)]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="glm-dsa モデルの total/active パラメータ数を見積もる（メモリ不要）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    for name in _LEVERS:
        default = getattr(SUZUME_4B, name)
        if isinstance(default, bool):
            p.add_argument(f"--{name.replace('_', '-')}", dest=name,
                           action=argparse.BooleanOptionalAction, default=None)
        else:
            p.add_argument(f"--{name.replace('_', '-')}", dest=name,
                           type=type(default), default=None)
    # 学習メモリ見積り用
    p.add_argument("--bf16-weights", action="store_true", help="重み・勾配を bf16 で見積る")
    p.add_argument("--optim8bit", action="store_true", help="8bit AdamW で見積る")
    return p


def config_from_args(args) -> GlmDsaConfig:
    overrides = {name: getattr(args, name) for name in _LEVERS
                 if getattr(args, name, None) is not None}
    return replace(SUZUME_4B, **overrides)


def print_train_memory(total: int, bf16_weights: bool, optim8bit: bool) -> None:
    def gb(bpp):
        return total * bpp / 1e9
    w = 2 if bf16_weights else 4
    g = w
    o = 2 if optim8bit else 8
    print("\n学習メモリ概算（重み+勾配+optimizer、活性化は別途）:")
    print(f"  選択設定 (bf16_weights={bf16_weights}, optim8bit={optim8bit}): 約 {gb(w+g+o):.1f} GB")
    print(f"    fp32 + fp32 Adam      (16 B/param): 約 {gb(16):.1f} GB")
    print(f"    bf16重み + fp32 Adam  (12 B/param): 約 {gb(12):.1f} GB")
    print(f"    bf16重み + 8bit Adam   (6 B/param): 約 {gb(6):.1f} GB  ← 最省メモリ")
    print("  ※ これに活性化メモリが加算（--grad-checkpoint で削減、batch/seq で増減）")


def main() -> None:
    args = build_parser().parse_args()
    cfg = config_from_args(args)
    with torch.device("meta"):
        model = SuzumeGlmDsa(cfg)
    p = model.count_parameters()
    print(f"config        : n_embd={cfg.n_embd} n_layer={cfg.n_layer} n_head={cfg.n_head} "
          f"E={cfg.n_expert}(used={cfg.n_expert_used},shared={cfg.n_expert_shared}) "
          f"ff_exp={cfg.n_ff_exp} kv_lora={cfg.kv_lora_rank}")
    print(f"total params  : {p['total']:,}  ({p['total']/1e9:.2f}B)")
    print(f"active params : {p['active']:,}  ({p['active']/1e9:.2f}B)  ← 1トークンあたり実計算量")
    print(f"重みメモリ(fp16): 約 {p['total']*2/1e9:.1f} GB")
    print_train_memory(p["total"], args.bf16_weights, args.optim8bit)


if __name__ == "__main__":
    main()
