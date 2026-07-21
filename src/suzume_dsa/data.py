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

    source: 既にトークン化済みの Tensor / テキストファイルのパス / 生文字列 / 文字列のリスト。
    """
    if isinstance(source, torch.Tensor):
        return source
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


def _parse_hf_spec(spec: str) -> tuple[str, str | None, str | None, str | None]:
    """"path[:config][:split][:column]" を分解（path は namespace/name）。"""
    parts = spec.split(":")
    path = parts[0]
    config = parts[1] if len(parts) > 1 and parts[1] else None
    split = parts[2] if len(parts) > 2 and parts[2] else None
    column = parts[3] if len(parts) > 3 and parts[3] else None
    return path, config, split, column


def _guess_text_column(features) -> str | None:
    for cand in ("text", "content", "body", "document", "raw", "ja", "japanese"):
        if cand in features:
            return cand
    # 最初の文字列カラムにフォールバック
    for name, feat in features.items():
        if getattr(feat, "dtype", None) == "string":
            return name
    return None


def preview_hf_dataset(spec: str, *, column: str | None = None, split: str | None = None,
                       max_samples: int = 200, n: int = 3, hf_token: str | None = None,
                       stream: bool = False, verbose: bool = True) -> dict:
    """HuggingFace データセットを下見する（check_dataset ツールが使う）。

    読めるか・どのカラムがテキストか・中身の例を確認する。`datasets` が必要。
    """
    try:
        from datasets import load_dataset
    except ImportError:
        if verbose:
            print("`datasets` が未インストール。`uv sync` で入れてください。")
        return {"ok": False, "usable": False}

    path, config, spec_split, spec_col = _parse_hf_spec(spec)
    split = split or spec_split or "train"
    column = column or spec_col
    try:
        ds = load_dataset(path, config, split=split, streaming=stream, token=hf_token)
    except Exception as e:  # データセット固有の多様なエラーをまとめて拾う
        if verbose:
            print(f"読み込み失敗: {e}")
        return {"ok": False, "usable": False}

    rows = []
    for i, row in enumerate(ds):
        if i >= max_samples:
            break
        rows.append(row)
    if not rows:
        return {"ok": True, "usable": False}

    col = column or _guess_text_column(rows[0].keys() if stream else ds.features)
    if verbose:
        print(f"path={path} config={config} split={split} → {len(rows)} 行読込")
        print(f"カラム: {list(rows[0].keys())}  / テキスト列: {col}")
        for r in rows[:n]:
            sample = str(r.get(col, r))[:200].replace("\n", " ")
            print(f"  - {sample}")
    return {"ok": True, "usable": col is not None, "column": col, "rows": len(rows)}


def load_hf_tokens(spec: str, tokenizer, *, column: str | None = None,
                   split: str | None = None, max_samples: int | None = None,
                   hf_token: str | None = None, stream: bool = True) -> torch.Tensor:
    """HF データセットのテキストを連結・トークン化して 1 次元 LongTensor にする。

    巨大コーパスは stream=True で先頭 max_samples 行だけ読む。本番学習の入口。
    """
    from datasets import load_dataset

    path, config, spec_split, spec_col = _parse_hf_spec(spec)
    split = split or spec_split or "train"
    column = column or spec_col
    ds = load_dataset(path, config, split=split, streaming=stream, token=hf_token)

    ids: list[int] = []
    for i, row in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break
        col = column or _guess_text_column(row.keys())
        text = row.get(col)
        if text:
            ids.extend(tokenizer.encode(text))
            ids.append(tokenizer.encode("\n")[0] if tokenizer.encode("\n") else 10)
    return torch.tensor(ids, dtype=torch.long)


class TokenWindows:
    """トークン列からランダムな長さ block のウィンドウをサンプルする。

    PackedDataset と違い block を毎回変えられるので、系列長カリキュラム
    （--block-size-schedule）で 1-run のまま block を段階的に伸ばせる。
    """

    def __init__(self, tokens: torch.Tensor):
        assert tokens.numel() > 2, "corpus が短すぎる"
        self.tokens = tokens

    def sample(self, batch_size: int, block_size: int,
               generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        n = self.tokens.numel() - block_size - 1
        assert n > 0, f"corpus が block_size={block_size} に対して短い"
        starts = torch.randint(0, n + 1, (batch_size,), generator=generator)
        x = torch.stack([self.tokens[s : s + block_size] for s in starts])
        y = torch.stack([self.tokens[s + 1 : s + block_size + 1] for s in starts])
        return x, y


def parse_block_size_schedule(spec: str, max_steps: int) -> list[tuple[int, int]]:
    """"0:64,100:128" や "0%:64,50%:128,80%:256" を [(step, block), ...] に。

    - step は絶対値か、--steps 比の "%"（混在可）。
    - step0（または 0%）が必須。step は昇順。block は正。
    """
    items: list[tuple[int, int]] = []
    for part in spec.split(","):
        key, _, val = part.strip().partition(":")
        key = key.strip()
        if key.endswith("%"):
            step = round(float(key[:-1]) / 100.0 * max_steps)
        else:
            step = int(key)
        block = int(val)
        assert block > 0, f"block は正: {part}"
        items.append((step, block))
    items.sort(key=lambda kv: kv[0])
    assert items[0][0] == 0, "スケジュールは step0（または 0%）から始めること"
    steps_only = [s for s, _ in items]
    assert steps_only == sorted(set(steps_only)), "step は昇順かつ重複なし"
    return items


def block_size_at(step: int, schedule: list[tuple[int, int]]) -> int:
    """step 時点で有効な block_size（step 以下で最大の閾値の値）。"""
    block = schedule[0][1]
    for thr, blk in schedule:
        if step >= thr:
            block = blk
        else:
            break
    return block


__all__ = [
    "ByteTokenizer", "load_corpus_tokens", "PackedDataset", "TokenWindows",
    "parse_block_size_schedule", "block_size_at",
    "preview_hf_dataset", "load_hf_tokens",
]
