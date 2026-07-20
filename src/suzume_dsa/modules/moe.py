"""
suzume-dsa: Sigmoid-gated MoE + 共有 Expert（GLM-4.5 / DeepSeek-V3 系）

ルーティングは sigmoid ゲート。ロードバランスは **aux-free**：補助損失を足す代わりに
Expert ごとの bias（exp_probs_b）を「選択にだけ」加算し、負荷の偏りを見て bias を
少しずつ更新する（DeepSeek-V3 の loss-free balancing）。合成重みには bias を含まない
素の sigmoid 値を使う。

テンソル対応:
    gate_inp     -> ffn_gate_inp.weight    {D, E}
    exp_probs_b  -> ffn_exp_probs_b.bias   {E}
    w_gate/w_up/w_down -> ffn_{gate,up,down}_exps.weight
    共有 Expert   -> ffn_{gate,up,down}_shexp.weight

学習ループは各ステップ後に `commit_router_bias_updates()` を呼ぶ想定
（呼ばなくても学習は進むが、負荷が偏りやすくなる）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import GlmDsaConfig
from .ffn import SwiGLU


class SharedExpertMoE(nn.Module):
    def __init__(self, cfg: GlmDsaConfig):
        super().__init__()
        self.n_embd = cfg.n_embd
        self.num_experts = cfg.n_expert
        self.k = cfg.n_expert_used
        self.ff = cfg.n_ff_exp
        self.weights_scale = cfg.expert_weights_scale
        self.weights_norm = cfg.expert_weights_norm
        self.bias_update_rate = cfg.router_bias_update_rate

        # ルーター
        self.gate_inp = nn.Linear(cfg.n_embd, self.num_experts, bias=False)
        # aux-free の負荷バランス bias（勾配は流さない）
        self.register_buffer("exp_probs_b", torch.zeros(self.num_experts))
        # 直近ステップの Expert 選択回数を溜めて commit 時に bias 更新へ使う
        self.register_buffer("_load", torch.zeros(self.num_experts), persistent=False)

        # ルーティング Expert（バッチ化した重み。expert 次元を先頭に持つ）
        self.w_gate = nn.Parameter(torch.empty(self.num_experts, self.ff, cfg.n_embd))
        self.w_up = nn.Parameter(torch.empty(self.num_experts, self.ff, cfg.n_embd))
        self.w_down = nn.Parameter(torch.empty(self.num_experts, cfg.n_embd, self.ff))
        for w in (self.w_gate, self.w_up, self.w_down):
            nn.init.normal_(w, std=0.02)

        # 共有 Expert（常時通る。中間次元は ff * n_expert_shared）
        self.shared = SwiGLU(cfg.n_embd, self.ff * cfg.n_expert_shared)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        flat = x.reshape(-1, D)                          # (N, D)
        N = flat.size(0)

        scores = torch.sigmoid(self.gate_inp(flat))      # (N, E) 合成重み用（bias 無し）
        routing = scores + self.exp_probs_b              # 選択用（aux-free bias 込み）
        topk_idx = routing.topk(self.k, dim=-1).indices  # (N, k)

        weights = scores.gather(-1, topk_idx)            # (N, k) 合成重みは素の sigmoid
        if self.weights_norm:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        weights = weights * self.weights_scale

        # 負荷を記録（bias 更新用、勾配は無し）
        if self.training:
            self._load += torch.bincount(
                topk_idx.reshape(-1), minlength=self.num_experts).to(self._load.dtype)

        # Expert ごとに、その Expert へ振られたトークンだけ計算して散らし戻す
        out = torch.zeros_like(flat)
        for e in range(self.num_experts):
            hit = topk_idx == e                          # (N, k)
            token_mask = hit.any(dim=-1)                 # (N,)
            if not torch.any(token_mask):
                continue
            xe = flat[token_mask]                        # (Ne, D)
            he = F.silu(xe @ self.w_gate[e].t()) * (xe @ self.w_up[e].t())
            oe = he @ self.w_down[e].t()                 # (Ne, D)
            # このトークンが e に与えた重み（top-k 内の該当位置）
            we = (weights * hit).sum(dim=-1)[token_mask].unsqueeze(-1)
            out[token_mask] += we * oe

        out = out + self.shared(flat)                    # 共有 Expert を加算
        return out.view(B, T, D)

    @torch.no_grad()
    def commit_router_bias_updates(self) -> None:
        """溜めた負荷を見て aux-free bias を更新する（過負荷は下げ、過小は上げる）。"""
        load = self._load
        if load.sum() == 0:
            return
        target = load.mean()
        # 過負荷(+)なら bias を下げ、過小(-)なら上げる
        self.exp_probs_b += self.bias_update_rate * torch.sign(target - load)
        self._load.zero_()

    def inactive_parameter_count(self) -> int:
        """1トークンあたり計算されない Expert 分（active 集計用）。"""
        per_expert = (self.w_gate[0].numel() + self.w_up[0].numel() + self.w_down[0].numel())
        return per_expert * (self.num_experts - self.k)


__all__ = ["SharedExpertMoE"]
