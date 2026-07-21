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
        # NEFTune: 学習時に埋め込みへ一様ノイズを足す（SFT の指示追従を安価に底上げ）。
        # 0.0=無効。学習ループが必要に応じて設定する（推論・export では常に 0）。
        self.neftune_alpha = 0.0

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

    def forward(self, input_ids: torch.Tensor,
                attn_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        """
        Args:
            input_ids: (B, T)
            attn_mask: 省略時は素の因果マスク（高速パス）。シーケンスパッキング時は
                       (B,1,T,T) の bool（causal＋同一セグメント、True=attend）を渡す。
        Returns:
            logits: (B, T, vocab_size)
            info:   {"mtp_logits": [ (B, T-k, V), ... ]}（学習時 MTP 有効時のみ）
        """
        # router z-loss を今 forward 分だけ集めるため、全 MoE の保持値をリセット
        moes = [m for m in self.modules() if isinstance(m, SharedExpertMoE)]
        for m in moes:
            m._z_loss = None

        x = self.token_embd(input_ids)
        if self.training and self.neftune_alpha > 0.0:      # NEFTune 埋め込みノイズ
            d = x.size(-1) * x.size(1)
            noise = (torch.rand_like(x) * 2 - 1) * (self.neftune_alpha / (d ** 0.5))
            x = x + noise
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        h = x                      # final norm 前（MTP が使う）
        logits = self._logits(self.output_norm(x))

        info: dict = {}
        if self.mtp is not None and self.training and input_ids.size(1) > self.cfg.mtp_depth:
            mtp_logits = []
            h_k = h
            for k, mod in enumerate(self.mtp, start=1):
                e = self.token_embd(input_ids[:, k:])
                hk_in = h_k[:, :-1]
                # パッキング時は先頭 L×L の部分マスクが hk_in の位置に対応（セグメントIDは不変）
                m = attn_mask[..., :hk_in.size(1), :hk_in.size(1)] if attn_mask is not None else None
                h_k = mod(hk_in, e, attn_mask=m)
                mtp_logits.append(self._logits(self.output_norm(h_k)))
            info["mtp_logits"] = mtp_logits

        if self.training:                              # 全 MoE 層の router z-loss を平均
            zs = [m._z_loss for m in moes if m._z_loss is not None]
            if zs:
                info["router_z"] = torch.stack(zs).mean()
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
