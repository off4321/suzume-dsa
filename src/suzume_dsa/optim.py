"""
suzume-dsa: Muon optimizer + 学習率スケジュール

Muon（MomentUm Orthogonalized by Newton-Schulz、Keller Jordan 2024 / suzume-muon 由来）:
隠れ層の 2 次元重みだけモーメンタム勾配を近似直交化して更新し、Embedding / output /
1 次元パラメータ / バッチ化 expert 重み(ndim>=3) は AdamW で更新するハイブリッド。
モデル構造を変えずに frontier の学習手法へ追従できる。

LR スケジュール: cosine（線形warmup+cosine減衰）と WSD（Warmup-Stable-Decay、
stable 相が一定なので継続学習・安価な decay 分岐に向く、MiniCPM 由来）。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz 5 次反復で近似直交化（特異値をほぼ 1 に潰す）。"""
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon + 内蔵 AdamW のハイブリッド（1 インスタンスで既存の学習ループと互換）。"""

    def __init__(self, muon_params, adamw_params, lr: float = 0.02,
                 momentum: float = 0.95, nesterov: bool = True, ns_steps: int = 5,
                 adamw_lr: float = 3e-4, betas: tuple[float, float] = (0.9, 0.95),
                 eps: float = 1e-8, weight_decay: float = 0.01):
        groups = [
            {"params": list(muon_params), "lr": lr, "momentum": momentum,
             "nesterov": nesterov, "ns_steps": ns_steps,
             "weight_decay": weight_decay, "use_muon": True},
            {"params": list(adamw_params), "lr": adamw_lr, "betas": betas,
             "eps": eps, "weight_decay": weight_decay, "use_muon": False},
        ]
        super().__init__(groups, defaults={})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            (self._muon_step if group["use_muon"] else self._adamw_step)(group)
        return loss

    def _muon_step(self, group: dict) -> None:
        lr, momentum = group["lr"], group["momentum"]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            st = self.state[p]
            buf = st.setdefault("momentum_buffer", torch.zeros_like(g))
            buf.mul_(momentum).add_(g)
            g = g.add(buf, alpha=momentum) if group["nesterov"] else buf
            u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
            if group["weight_decay"]:
                p.mul_(1 - lr * group["weight_decay"])
            scale = max(1.0, g.size(0) / g.size(1)) ** 0.5   # 縦長補正（原著）
            p.add_(u, alpha=-lr * scale)

    def _adamw_step(self, group: dict) -> None:
        lr, (b1, b2), eps = group["lr"], group["betas"], group["eps"]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            st = self.state[p]
            if "exp_avg" not in st:
                st["step"] = 0
                st["exp_avg"] = torch.zeros_like(g)
                st["exp_avg_sq"] = torch.zeros_like(g)
            st["step"] += 1
            t = st["step"]
            if group["weight_decay"]:
                p.mul_(1 - lr * group["weight_decay"])
            st["exp_avg"].lerp_(g, 1 - b1)
            st["exp_avg_sq"].mul_(b2).addcmul_(g, g, value=1 - b2)
            denom = (st["exp_avg_sq"] / (1 - b2 ** t)).sqrt_().add_(eps)
            p.addcdiv_(st["exp_avg"], denom, value=-lr / (1 - b1 ** t))


def split_muon_params(model: nn.Module):
    """(Muon 対象=隠れ層の 2 次元重み, AdamW 対象=それ以外) に分割。

    Embedding / output(lm_head) / 1 次元(norm・bias) / バッチ化 expert 重み(ndim>=3、
    wk_b/wv_b や w_gate/up/down) は AdamW 側へ（直交化はプレーンな行列にのみ意味を持つ）。
    """
    embed_ids = {id(p) for m in model.modules() if isinstance(m, nn.Embedding)
                 for p in m.parameters(recurse=False)}
    io_ids = {id(p) for name, p in model.named_parameters()
              if "output" in name or "token_embd" in name}
    muon, adamw, seen = [], [], set()
    for p in model.parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if p.ndim == 2 and id(p) not in embed_ids and id(p) not in io_ids:
            muon.append(p)
        else:
            adamw.append(p)
    return muon, adamw


def build_optimizer(model, lr: float, weight_decay: float = 0.1,
                    optimizer: str = "adamw", muon_lr: float = 0.02):
    if optimizer == "muon":
        muon_p, adamw_p = split_muon_params(model)
        return Muon(muon_p, adamw_p, lr=muon_lr, adamw_lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                             weight_decay=weight_decay)


def lr_lambda_cosine(step: int, warmup: int, max_steps: int, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    prog = min((step - warmup) / max(1, max_steps - warmup), 1.0)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * prog))


def lr_lambda_wsd(step: int, warmup: int, max_steps: int,
                  decay_frac: float = 0.2, min_ratio: float = 0.0) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    decay_steps = max(1, round(decay_frac * max_steps))
    decay_start = max(warmup, max_steps - decay_steps)
    if step < decay_start:
        return 1.0
    prog = min((step - decay_start) / max(1, max_steps - decay_start), 1.0)
    return min_ratio + (1.0 - min_ratio) * (1.0 - math.sqrt(prog))   # 1-sqrt 減衰


def build_scheduler(optimizer, kind: str, warmup: int, max_steps: int,
                    wsd_decay_frac: float = 0.2):
    if kind == "wsd":
        fn = lambda s: lr_lambda_wsd(s, warmup, max_steps, wsd_decay_frac)
    else:
        fn = lambda s: lr_lambda_cosine(s, warmup, max_steps)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


__all__ = [
    "Muon", "split_muon_params", "build_optimizer",
    "lr_lambda_cosine", "lr_lambda_wsd", "build_scheduler",
]
