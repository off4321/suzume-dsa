"""
suzume-dsa: 密 FFN（SwiGLU）

glm-dsa の先頭 dense 層（n_layer_dense_lead）で使う通常の SwiGLU FFN。
テンソル対応: ffn_gate {D, n_ff} / ffn_up {D, n_ff} / ffn_down {n_ff, D}。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import GlmDsaConfig


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class DenseFFN(SwiGLU):
    """先頭 dense 層用の SwiGLU（config から中間次元を決めるだけの薄いラッパ）。"""

    def __init__(self, cfg: GlmDsaConfig):
        super().__init__(cfg.n_embd, cfg.n_ff)


__all__ = ["SwiGLU", "DenseFFN"]
