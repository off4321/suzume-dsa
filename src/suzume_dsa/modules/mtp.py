"""
suzume-dsa: Multi-Token Prediction (MTP) — 学習効率のための train-only 補助

DeepSeek-V3 方式。最終隠れ状態 h と「1つ先のトークン埋め込み e」を結合して
更に先のトークンを予測する補助タスクを足し、学習信号を濃くする。

**推論・export には一切関与しない**（glm-dsa の NextN 枠は llama.cpp 側で
"preserved but unused" のため、export では捨てて n_layer_nextn=0 とする）。
モジュール k（1 始まり）は位置 i で t_{i+k+1} を予測する。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import GlmDsaConfig
from .block import Block
from .norm import RMSNorm


class MTPModule(nn.Module):
    def __init__(self, cfg: GlmDsaConfig, layer_idx: int):
        super().__init__()
        self.hnorm = RMSNorm(cfg.n_embd, cfg.norm_eps)   # 隠れ状態側 norm
        self.enorm = RMSNorm(cfg.n_embd, cfg.norm_eps)   # 埋め込み側 norm
        self.eh_proj = nn.Linear(2 * cfg.n_embd, cfg.n_embd, bias=False)
        self.block = Block(cfg, layer_idx)               # 本体と同じ構造を 1 段

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """h, e: (B, T', D) -> 次段の隠れ状態 (B, T', D)。"""
        x = self.eh_proj(torch.cat([self.hnorm(h), self.enorm(e)], dim=-1))
        return self.block(x)


__all__ = ["MTPModule"]
