"""
suzume-dsa: Transformer ブロック

Pre-norm 残差接続:  x = x + MLA(attn_norm(x)) ; x = x + FFN(ffn_norm(x))
FFN は層の位置で密 SwiGLU（先頭 dense 層）か MoE（それ以降）に切り替わる。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import GlmDsaConfig
from .ffn import DenseFFN
from .mla import MultiHeadLatentAttention
from .moe import SharedExpertMoE
from .norm import RMSNorm


class Block(nn.Module):
    def __init__(self, cfg: GlmDsaConfig, layer_idx: int):
        super().__init__()
        self.is_dense = layer_idx < cfg.n_layer_dense_lead

        self.attn_norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.attn = MultiHeadLatentAttention(cfg)

        self.ffn_norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.ffn = DenseFFN(cfg) if self.is_dense else SharedExpertMoE(cfg)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), attn_mask=attn_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


__all__ = ["Block"]
