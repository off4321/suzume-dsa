"""
suzume-dsa: FIM（Fill-in-the-Middle）— 脱・純next-token の学習目的（suzume-muon 由来）

系列を prefix / middle / suffix に分割し「両側の文脈から中間を埋める」形に並べ替えて
学習する（Bavarian et al. 2022）。同じデータからより密な信号を引き出せ、MTP とは直交。
センチネル: <|fim_prefix|> / <|fim_suffix|> / <|fim_middle|>（tokenizer.SPECIAL_TOKENS）。
  - PSM: <PRE> prefix <SUF> suffix <MID> middle
  - SPM: <PRE> <SUF> suffix <MID> prefix middle   （spm_prob の確率で採用）
"""

from __future__ import annotations

import random

import torch


def fim_transform(content: torch.Tensor, pre_id: int, suf_id: int, mid_id: int,
                  rng: random.Random, spm_prob: float = 0.5) -> torch.Tensor:
    """1 本のトークン列を FIM 形式へ並べ替える（長さ +3 = センチネル分）。"""
    L = content.numel()
    a = rng.randint(0, L)
    b = rng.randint(a, L)
    prefix, middle, suffix = content[:a], content[a:b], content[b:]
    dev, dt = content.device, content.dtype
    P = torch.tensor([pre_id], dtype=dt, device=dev)
    S = torch.tensor([suf_id], dtype=dt, device=dev)
    M = torch.tensor([mid_id], dtype=dt, device=dev)
    if rng.random() < spm_prob:
        return torch.cat([P, S, suffix, M, prefix, middle])
    return torch.cat([P, prefix, S, suffix, M, middle])


def build_fim_batch(tokens: torch.Tensor, batch_size: int, block_size: int,
                    fim_ids: tuple[int, int, int], generator: torch.Generator,
                    rate: float = 0.5, spm_prob: float = 0.5):
    """ランダムウィンドウを一定割合 FIM 化して (x, y) を返す（長さ block_size を保つ）。"""
    pre_id, suf_id, mid_id = fim_ids
    n = tokens.numel() - block_size - 1
    assert n > 0, "corpus が block_size に対して短い"
    xs, ys = [], []
    for _ in range(batch_size):
        s = int(torch.randint(0, n + 1, (1,), generator=generator).item())
        w = tokens[s : s + block_size + 1]                 # 長さ block+1
        r = random.Random(int(torch.randint(0, 2**31, (1,), generator=generator).item()))
        if block_size >= 4 and r.random() < rate:
            content = w[: block_size + 1 - 3]              # センチネル 3 個分の余地
            seq = fim_transform(content, pre_id, suf_id, mid_id, r, spm_prob)
        else:
            seq = w
        xs.append(seq[:-1]); ys.append(seq[1:])
    return torch.stack(xs), torch.stack(ys)


__all__ = ["fim_transform", "build_fim_batch"]
