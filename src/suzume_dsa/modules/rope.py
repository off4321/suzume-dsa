"""
suzume-dsa: RoPE (Rotary Position Embedding)

Q/K に回転操作を適用して相対位置関係を表現する。
追加パラメータはゼロで、長文脈での性能向上が期待できる。
"""

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """後半半分を符号反転して前半と入れ替える（Llama/NeoX 方式）。"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """
    RoPE を Q/K に適用するモジュール。

    Args:
        head_dim: 1ヘッドあたりの次元（偶数であること）
        base: 周波数の基数（通常 10000）
    """

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0,
                positions: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        # q, k: (B, num_heads, T, head_dim)
        # offset: 先頭トークンの絶対位置。KV キャッシュのインクリメンタルデコードでは
        # 「これまで処理済みのトークン数」を渡す（0 始まり固定だと位置情報が壊れる）
        # positions: 各行の絶対位置を明示する (T,) テンソル。moa の疎実行のように
        # 連続しないトークン行をギャザーして処理するとき用（指定時は offset を無視）
        T = q.shape[-2]
        if positions is not None:
            t = positions.to(device=q.device, dtype=torch.float32)
        else:
            t = torch.arange(offset, offset + T, device=q.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)      # (T, head_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)    # (T, head_dim)
        cos = emb.cos().to(q.dtype)
        sin = emb.sin().to(q.dtype)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        return q, k


__all__ = ["RotaryEmbedding", "rotate_half"]
