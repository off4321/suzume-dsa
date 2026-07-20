"""
suzume-dsa: glm-dsa 準拠モデル（v1 = 素の MLA + MoE）

llama.cpp `GLM_DSA` が要求するテンソル集合（docs/glm-dsa-tensors.md）に一致する
構造を持つ。DSA indexer と NextN は v1 では持たない（indexer は NOT_REQUIRED なので
これで GGUF は読める）。MTP は学習効率のための train-only 補助で、export では無視する。

forward は (logits, info) を返す。info["mtp_logits"] は学習時に MTP が有効なときだけ入る。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import SUZUME_4B, GlmDsaConfig
from .modules.block import Block
from .modules.moe import SharedExpertMoE
from .modules.mtp import MTPModule
from .modules.norm import RMSNorm


class SuzumeGlmDsa(nn.Module):
    def __init__(self, cfg: GlmDsaConfig = SUZUME_4B):
        super().__init__()
        self.cfg = cfg

        self.token_embd = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layer))
        self.output_norm = RMSNorm(cfg.n_embd, cfg.norm_eps)

        # tie 時は lm_head を持たず token_embd を転置流用する
        if cfg.tie_embeddings:
            self.output = None
        else:
            self.output = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # 学習効率用 MTP（export 非対象）
        if cfg.mtp_depth > 0:
            self.mtp = nn.ModuleList(
                MTPModule(cfg, cfg.n_layer) for _ in range(cfg.mtp_depth))
        else:
            self.mtp = None

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        if self.output is None:
            return x @ self.token_embd.weight.t()
        return self.output(x)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Args:
            input_ids: (B, T)
        Returns:
            logits: (B, T, vocab_size)
            info:   {"mtp_logits": [ (B, T-k, V), ... ]}（学習時 MTP 有効時のみ）
        """
        x = self.token_embd(input_ids)
        for block in self.blocks:
            x = block(x)

        h = x                      # final norm 前（MTP が使う）
        logits = self._logits(self.output_norm(x))

        info: dict = {}
        if self.mtp is not None and self.training and input_ids.size(1) > self.cfg.mtp_depth:
            mtp_logits = []
            h_k = h
            for k, mod in enumerate(self.mtp, start=1):
                e = self.token_embd(input_ids[:, k:])
                h_k = mod(h_k[:, :-1], e)
                mtp_logits.append(self._logits(self.output_norm(h_k)))
            info["mtp_logits"] = mtp_logits
        return logits, info

    def commit_router_bias_updates(self) -> None:
        """全 MoE 層の aux-free bias を更新する（学習ループが毎ステップ後に呼ぶ）。"""
        for module in self.modules():
            if isinstance(module, SharedExpertMoE):
                module.commit_router_bias_updates()

    def count_parameters(self) -> dict[str, int]:
        """total と active（1トークンあたり実際に計算されるパラメータ数）。"""
        total = sum(p.numel() for p in self.parameters())
        # MTP は学習専用なので active/total の会計からは除く
        mtp = sum(p.numel() for p in self.mtp.parameters()) if self.mtp is not None else 0
        inactive = 0
        for block in self.blocks:
            if isinstance(block.ffn, SharedExpertMoE):
                inactive += block.ffn.inactive_parameter_count()
        return {"total": total - mtp, "active": total - mtp - inactive}


__all__ = ["SuzumeGlmDsa"]
