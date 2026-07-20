"""
suzume-dsa: 最小データパイプライン

(a) 最小 pretrain ループ用。外部依存なしで回せるよう、バイト単位トークナイザ
（vocab=256）を用意する。本番は SentencePiece 語彙に差し替える（TINY は 256 なので
バイト単位とちょうど一致し、疎通確認に使える）。

corpus（長いトークン列）を block_size ごとに詰めてバッチを供給するだけの素朴な実装。
系列長カリキュラム等の高度な供給は学習ループ移植の次段で足す。
"""

from __future__ import annotations

from pathlib import Path

import torch


class ByteTokenizer:
    """UTF-8 バイトをそのまま id にする最小トークナイザ（vocab=256）。"""

    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(int(i) % 256 for i in ids).decode("utf-8", errors="replace")


def load_corpus_tokens(source, tokenizer) -> torch.Tensor:
    """テキスト source を 1 次元の LongTensor トークン列にする。

    source: テキストファイルのパス / 生文字列 / 文字列のリスト。
    """
    if isinstance(source, (list, tuple)):
        text = "\n".join(source)
    elif _is_existing_file(source):
        text = Path(source).read_text(encoding="utf-8")
    else:
        text = str(source)
    return torch.tensor(tokenizer.encode(text), dtype=torch.long)


def _is_existing_file(source) -> bool:
    """source をファイルパスとして扱えるか（長い生文字列で例外にならないよう保護）。"""
    try:
        return Path(str(source)).is_file()
    except OSError:
        return False


class PackedDataset:
    """トークン列を block_size の連続チャンクに詰めてランダムバッチを返す。"""

    def __init__(self, tokens: torch.Tensor, block_size: int):
        assert tokens.numel() > block_size, "corpus が block_size より短い"
        n = (tokens.numel() - 1) // block_size          # 末尾 1 は次トークン用に残す
        self.block_size = block_size
        # (n, block_size) の入力と、1 つずらした (n, block_size) の教師
        self.x = tokens[: n * block_size].view(n, block_size)
        self.y = tokens[1 : n * block_size + 1].view(n, block_size)

    def __len__(self) -> int:
        return self.x.size(0)

    def batches(self, batch_size: int, generator: torch.Generator | None = None):
        """1 エポック分のバッチを無限に生成し続けるイテレータ。"""
        while True:
            order = torch.randperm(len(self), generator=generator)
            for i in range(0, len(self) - batch_size + 1, batch_size):
                idx = order[i : i + batch_size]
                yield self.x[idx], self.y[idx]


__all__ = ["ByteTokenizer", "load_corpus_tokens", "PackedDataset"]
