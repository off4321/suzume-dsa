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


# 明示的なテキスト列（優先的に採用）
_TEXT_CANDIDATES = ("text", "content", "body", "document", "raw", "ja", "japanese",
                    "markdown", "story")
# 会話列（list）。テキスト列が無ければ平坦化して本文として使う
_CONV_CANDIDATES = ("conversations", "messages", "conversation", "dialog",
                    "dialogue", "turns")
# 本文でないメタ列（フォールバックの「最長文字列列」選びから除外する）
_META_COLS = {
    "id", "uid", "pageid", "revid", "sha", "source", "source_dataset", "category",
    "subcategory", "task", "url", "task_url", "language_url", "timestamp", "date",
    "date_created", "date_modified", "dump", "score", "split", "language",
    "language_score", "language_script", "region", "script", "lang", "resource",
    "session", "func_name", "top_langs", "file_path", "minhash_cluster_size",
    "repo", "path", "language_name", "is_disambiguation_page", "is_sexual_page",
    "is_violent_page", "templates",
}


def _content_parts_to_text(content) -> str:
    """会話ターンの content（文字列 / パーツ list / dict）を平坦なテキストへ。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for p in content:
            if isinstance(p, dict):
                t = p.get("text") or p.get("value")
                if t:
                    out.append(str(t))
            elif isinstance(p, str):
                out.append(p)
        return " ".join(out)
    if isinstance(content, dict):
        return str(content.get("text") or content.get("value") or "")
    return ""


def _cell_to_text(value) -> str:
    """セルの値をテキスト化する。文字列はそのまま、会話 list は各ターンを連結。"""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for turn in value:
            if isinstance(turn, dict):
                c = turn.get("content")
                if c is None:
                    c = turn.get("value") or turn.get("text") or ""
                parts.append(_content_parts_to_text(c))
            elif isinstance(turn, str):
                parts.append(turn)
        return "\n".join(p for p in parts if p)
    return ""


def _guess_text_column(sample: dict) -> str | None:
    """サンプル行（1件の dict）から本文列名を推測する。

    優先順: 明示テキスト列 → 会話列(list, 後段で平坦化) → メタ列以外で最長の文字列列。
    id/source/category のようなメタ文字列列を誤って選ばないようにする。
    """
    for cand in _TEXT_CANDIDATES:
        if isinstance(sample.get(cand), str):
            return cand
    for cand in _CONV_CANDIDATES:
        if isinstance(sample.get(cand), list) and sample[cand]:
            return cand
    # メタ列を避けつつ、サンプル値が最も長い文字列列を選ぶ
    best, best_len = None, 0
    for name, value in sample.items():
        if str(name).lower() in _META_COLS or not isinstance(value, str):
            continue
        if len(value) > best_len:
            best, best_len = name, len(value)
    return best


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

    col = column or _guess_text_column(rows[0])
    usable = col is not None and any(_cell_to_text(r.get(col)).strip() for r in rows[:n])
    if verbose:
        print(f"path={path} config={config} split={split} → {len(rows)} 行読込")
        print(f"カラム: {list(rows[0].keys())}  / テキスト列: {col}")
        for r in rows[:n]:
            sample = _cell_to_text(r.get(col))[:200].replace("\n", " ")
            print(f"  - {sample}")
    return {"ok": True, "usable": usable, "column": col, "rows": len(rows)}


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
        col = column or _guess_text_column(row)
        text = _cell_to_text(row.get(col)) if col else ""
        if text:
            ids.extend(tokenizer.encode(text))
            ids.append(tokenizer.encode("\n")[0] if tokenizer.encode("\n") else 10)
    return torch.tensor(ids, dtype=torch.long)


def load_hf_corpus(specs, tokenizer, *, max_samples: int | None = None,
                   hf_token: str | None = None, stream: bool = True) -> torch.Tensor:
    """複数の HF spec を順に読み、トークン列を 1 本に連結する（各 spec に max_samples 適用）。

    「data_check したデータセットを全部投入する」用。config 必須・カラム不明などで
    読めない spec は警告してスキップし、残りで学習を続ける（1 件の失敗で全体を落とさない）。
    単一 spec（str）も受け付ける。
    """
    if isinstance(specs, str):
        specs = [specs]
    chunks: list[torch.Tensor] = []
    for spec in specs:
        try:
            toks = load_hf_tokens(spec, tokenizer, max_samples=max_samples,
                                  hf_token=hf_token, stream=stream)
        except Exception as e:  # noqa: BLE001 - 1 件ダメでも継続したい
            print(f"[warn] スキップ（読込失敗）: {spec}  ({type(e).__name__}: {e})")
            continue
        if toks.numel() == 0:
            print(f"[warn] スキップ（トークン0＝テキスト列不明？ :column 明示を検討）: {spec}")
            continue
        print(f"[data] {spec}: {toks.numel():,} tokens")
        chunks.append(toks)
    if not chunks:
        raise SystemExit("有効な学習データが 0 件でした。spec / --column を確認してください。")
    corpus = torch.cat(chunks)
    print(f"[data] 合計 {len(chunks)} データセット / {corpus.numel():,} tokens")
    return corpus


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


def schedule_value_at(step: int, schedule: list[tuple[int, int]]) -> int:
    """step 時点で有効な値（step 以下で最大の閾値のもの）。block/batch 共通の探索。"""
    value = schedule[0][1]
    for thr, val in schedule:
        if step >= thr:
            value = val
        else:
            break
    return value


def block_size_at(step: int, schedule: list[tuple[int, int]]) -> int:
    """step 時点で有効な block_size。"""
    return schedule_value_at(step, schedule)


def parse_batch_size_schedule(spec: str, max_steps: int) -> list[tuple[int, int]]:
    """"0:16,50%:8" を [(step, batch), ...] に。バッチカリキュラム。

    系列長カリキュラム（--block-size-schedule）と対で使い、block を伸ばす step で
    batch を下げて VRAM をほぼ一定に保つ（VRAM ≈ 固定分 + batch×block×係数）。
    block と違い batch は増減どちらの向きも許可。step は絶対値か --steps 比の "%"、
    step0（または 0%）必須・昇順。step の純関数なので --resume と整合する。
    """
    items: list[tuple[int, int]] = []
    for part in spec.split(","):
        key, _, val = part.strip().partition(":")
        key = key.strip()
        if key.endswith("%"):
            step = round(float(key[:-1]) / 100.0 * max_steps)
        else:
            step = int(key)
        batch = int(val)
        assert batch >= 1, f"batch は 1 以上: {part}"
        items.append((step, batch))
    items.sort(key=lambda kv: kv[0])
    assert items[0][0] == 0, "スケジュールは step0（または 0%）から始めること"
    steps_only = [s for s, _ in items]
    assert steps_only == sorted(set(steps_only)), "step は昇順かつ重複なし"
    return items


def batch_size_at(step: int, schedule: list[tuple[int, int]]) -> int:
    """step 時点で有効な batch_size。"""
    return schedule_value_at(step, schedule)


__all__ = [
    "ByteTokenizer", "load_corpus_tokens", "PackedDataset", "TokenWindows",
    "parse_block_size_schedule", "block_size_at", "schedule_value_at",
    "parse_batch_size_schedule", "batch_size_at",
    "preview_hf_dataset", "load_hf_tokens",
]
