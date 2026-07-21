"""
suzume-dsa: Multi-head Latent Attention (MLA)

DeepSeek / GLM 系の MLA。K/V を低ランク潜在 c_kv に圧縮して持つことで
KV キャッシュを小さくする。glm-dsa は **分割形式**（wk_b / wv_b が別テンソル）を
要求するため、本実装もその形で重みを持つ。

v1 は DSA indexer 無しの素の因果アテンション（indexer は llama.cpp 側で
NOT_REQUIRED なので、これだけで GGUF は読める）。学習では潜在から毎回 K/V を
復元する「非吸収」形で計算する（数式的に最も素直で、重みの意味と一致）。

重みと llama.cpp テンソルの対応:
    wq_a        -> attn_q_a.weight        {D, Lr_q}
    q_a_norm    -> attn_q_a_norm.weight   {Lr_q}
    wq_b        -> attn_q_b.weight        {Lr_q, H*hk}
    wkv_a_mqa   -> attn_kv_a_mqa.weight   {D, Lr_kv + rope}
    kv_a_norm   -> attn_kv_a_norm.weight  {Lr_kv}
    wk_b        -> attn_k_b.weight        {nope, Lr_kv, H}
    wv_b        -> attn_v_b.weight        {Lr_kv, hv, H}
    wo          -> attn_output.weight     {H*hv, D}
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import GlmDsaConfig
from .norm import RMSNorm
from .rope import RotaryEmbedding


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, cfg: GlmDsaConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.nope = cfg.head_dim_nope
        self.rope = cfg.head_dim_rope
        self.hk = cfg.head_dim_k          # nope + rope
        self.hv = cfg.head_dim_v
        self.kv_lora = cfg.kv_lora_rank
        self.scale = self.hk ** -0.5

        # Q 側: D -> Lr_q -(norm)-> H*hk
        self.wq_a = nn.Linear(cfg.n_embd, cfg.q_lora_rank, bias=False)
        self.q_a_norm = RMSNorm(cfg.q_lora_rank, cfg.norm_eps)
        self.wq_b = nn.Linear(cfg.q_lora_rank, self.n_head * self.hk, bias=False)

        # KV 側: D -> [c_kv (Lr_kv) | k_rope (rope)]。k_rope は全ヘッド共有 (MQA)。
        self.wkv_a_mqa = nn.Linear(cfg.n_embd, self.kv_lora + self.rope, bias=False)
        self.kv_a_norm = RMSNorm(self.kv_lora, cfg.norm_eps)

        # 低ランク因子（3D なので Linear でなく Parameter で持つ）
        # wk_b: c_kv -> ヘッド別 k_nope,  wv_b: c_kv -> ヘッド別 v
        self.wk_b = nn.Parameter(torch.empty(self.nope, self.kv_lora, self.n_head))
        self.wv_b = nn.Parameter(torch.empty(self.kv_lora, self.hv, self.n_head))
        nn.init.normal_(self.wk_b, std=0.02)
        nn.init.normal_(self.wv_b, std=0.02)

        self.wo = nn.Linear(self.n_head * self.hv, cfg.n_embd, bias=False)

        # RoPE は rope 次元にだけ掛ける
        self.rotary = RotaryEmbedding(self.rope, base=cfg.rope_base)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape

        # --- Q ---
        q = self.wq_b(self.q_a_norm(self.wq_a(x)))          # (B, T, H*hk)
        q = q.view(B, T, self.n_head, self.hk).transpose(1, 2)   # (B, H, T, hk)
        q_nope, q_rope = q.split([self.nope, self.rope], dim=-1)

        # --- KV: 潜在 c_kv と共有 k_rope を作り、ヘッド別 K/V を復元 ---
        kv = self.wkv_a_mqa(x)                              # (B, T, Lr_kv+rope)
        c_kv, k_rope = kv.split([self.kv_lora, self.rope], dim=-1)
        c_kv = self.kv_a_norm(c_kv)                         # (B, T, Lr_kv)
        k_rope = k_rope.unsqueeze(1)                        # (B, 1, T, rope)

        # c_kv から低ランク因子でヘッド別に復元
        k_nope = torch.einsum("btl,nlh->bhtn", c_kv, self.wk_b)   # (B, H, T, nope)
        v = torch.einsum("btl,lvh->bhtv", c_kv, self.wv_b)        # (B, H, T, hv)

        # RoPE（rope 部分のみ）。k_rope は 1 ヘッドを全ヘッドへ展開
        q_rope, k_rope = self.rotary(q_rope, k_rope)
        k_rope = k_rope.expand(B, self.n_head, T, self.rope)

        q = torch.cat([q_nope, q_rope], dim=-1)            # (B, H, T, hk)
        k = torch.cat([k_nope, k_rope], dim=-1)            # (B, H, T, hk)

        # 因果アテンション。attn_mask 指定時（シーケンスパッキング）はそれを使う
        # （causal＋同一セグメントを bool で内包。is_causal と併用不可なので False）。
        if attn_mask is None:
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)
        else:
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                                  scale=self.scale)
        attn = attn.transpose(1, 2).reshape(B, T, self.n_head * self.hv)
        return self.wo(attn)


__all__ = ["MultiHeadLatentAttention"]
