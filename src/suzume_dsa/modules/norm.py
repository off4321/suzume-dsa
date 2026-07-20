"""
suzume-dsa: 正規化レイヤー

RMSNorm（Llama / Qwen / DeepSeek 系の標準）と LayerNorm を切り替えられるようにする。
RMSNorm は平均引き算とバイアスを省くため LayerNorm より軽く、大規模でも安定する。
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    x / sqrt(mean(x^2) + eps) * weight
    LayerNorm と違い平均中心化・バイアスを持たない。

    Args:
        dim: 正規化する最終次元
        eps: 数値安定化項
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # float32 で分散を計算してから元の dtype に戻す（低精度学習での安定性）
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


def build_norm(norm_type: str, dim: int) -> nn.Module:
    """norm_type から正規化レイヤーを生成する。"""
    if norm_type == "rmsnorm":
        return RMSNorm(dim)
    if norm_type == "layernorm":
        return nn.LayerNorm(dim)
    raise ValueError(f"unknown norm_type: {norm_type} (choose 'rmsnorm' or 'layernorm')")


__all__ = ["RMSNorm", "build_norm"]
