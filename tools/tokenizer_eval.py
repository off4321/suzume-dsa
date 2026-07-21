"""
suzume-dsa: トークナイザ評価（日本語での“勝負”用）

同じテキストに対する **fertility（1文字あたりトークン数）** を測って複数トークナイザを
横並び比較する。JA は fertility が低い＝1トークンで多くの文字を運べる＝実効コンテキストと
コストで有利。GGUF に焼けるトークナイザは製品の差別化点なので、数値で勝ちを示すためのツール。

指標:
    fertility        = tokens / chars   （小さいほど良い＝日本語を短く表せる）
    chars/token      = chars / tokens
    bytes/token      = utf8bytes / tokens
    unk%             = 未知トークン割合（byte-fallback があれば概ね 0）

使い方:
    # 自分の SP と、比較したい他の SP を並べて日本語コーパスで評価
    uv run tools/tokenizer_eval.py --sp tokenizer/sp.model other/sp.model \
        --hf-dataset "range3/wikipedia-ja-20230101" --max-samples 2000

    # HF のトークナイザ（transformers があれば）とも比較
    uv run tools/tokenizer_eval.py --sp tokenizer/sp.model \
        --hf-tokenizer meta-llama/Llama-2-7b-hf --corpus sample_ja.txt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def measure(encode_fn, text: str, unk_id: int | None = None) -> dict:
    """encode_fn(text)->list[int] を使ってテキスト全体の fertility 等を測る。"""
    n_chars = len(text)
    n_bytes = len(text.encode("utf-8"))
    ids = encode_fn(text)
    n_tokens = len(ids)
    n_unk = sum(1 for i in ids if i == unk_id) if unk_id is not None else 0
    n_tokens = max(n_tokens, 1)
    return {
        "chars": n_chars,
        "bytes": n_bytes,
        "tokens": n_tokens,
        "fertility": n_tokens / max(n_chars, 1),
        "chars_per_token": n_chars / n_tokens,
        "bytes_per_token": n_bytes / n_tokens,
        "unk_pct": 100.0 * n_unk / n_tokens,
    }


def measure_sp(sp_model_path: str, text: str) -> dict:
    """SentencePiece .model を SPTokenizer で読み、fertility 等を測る。"""
    from suzume_dsa.tokenizer import SPTokenizer

    tok = SPTokenizer(sp_model_path)
    unk_id = tok.sp.unk_id()
    m = measure(tok.encode, text, unk_id=unk_id)
    m["vocab"] = tok.vocab_size
    return m


def measure_hf(name: str, text: str) -> dict:
    """HuggingFace の AutoTokenizer で測る（transformers が必要）。"""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    m = measure(lambda t: tok.encode(t, add_special_tokens=False), text,
                unk_id=tok.unk_token_id)
    m["vocab"] = tok.vocab_size
    return m


def _gather_text(corpus: str | None, hf_dataset: str | None,
                 max_samples: int, max_chars: int, hf_token: str | None) -> str:
    """評価用テキストを 1 本の文字列に集める（--corpus か --hf-dataset）。"""
    if corpus:
        return Path(corpus).read_text(encoding="utf-8")[:max_chars]
    assert hf_dataset, "--corpus か --hf-dataset のどちらかが必要です"
    from datasets import load_dataset

    from suzume_dsa.data import _cell_to_text, _guess_text_column, _parse_hf_spec

    path, config, split, column = _parse_hf_spec(hf_dataset)
    ds = load_dataset(path, config, split=split or "train", streaming=True, token=hf_token)
    parts: list[str] = []
    total = 0
    for i, row in enumerate(ds):
        if i >= max_samples or total >= max_chars:
            break
        col = column or _guess_text_column(row)
        text = _cell_to_text(row.get(col)) if col else ""
        if text:
            parts.append(text)
            total += len(text)
    return "\n".join(parts)[:max_chars]


def main() -> None:
    p = argparse.ArgumentParser(description="トークナイザの fertility 比較")
    p.add_argument("--sp", nargs="*", default=[], help="SentencePiece .model（複数可）")
    p.add_argument("--hf-tokenizer", nargs="*", default=[],
                   help="HF トークナイザ名（transformers 必須。複数可）")
    p.add_argument("--corpus", default=None, help="評価テキストファイル")
    p.add_argument("--hf-dataset", default=None, help='HF "path[:config][:split][:column]"')
    p.add_argument("--max-samples", type=int, default=2000)
    p.add_argument("--max-chars", type=int, default=2_000_000)
    p.add_argument("--hf-token", default=None)
    args = p.parse_args()

    assert args.sp or args.hf_tokenizer, "--sp か --hf-tokenizer を 1 つ以上指定してください"
    text = _gather_text(args.corpus, args.hf_dataset, args.max_samples,
                        args.max_chars, args.hf_token)
    assert text, "評価テキストが空です"
    print(f"評価テキスト: {len(text):,} 文字 / {len(text.encode('utf-8')):,} bytes\n")

    rows = []
    for path in args.sp:
        try:
            rows.append((Path(path).stem, measure_sp(path, text)))
        except Exception as e:  # noqa: BLE001
            print(f"[warn] SP スキップ {path}: {type(e).__name__}: {e}")
    for name in args.hf_tokenizer:
        try:
            rows.append((name, measure_hf(name, text)))
        except Exception as e:  # noqa: BLE001
            print(f"[warn] HF スキップ {name}: {type(e).__name__}: {e}")

    if not rows:
        raise SystemExit("評価できたトークナイザがありません。")

    # fertility 昇順（=日本語を短く表せる順）で表示
    rows.sort(key=lambda kv: kv[1]["fertility"])
    header = f"{'tokenizer':28} {'vocab':>7} {'tokens':>10} {'fertility':>10} " \
             f"{'chars/tok':>10} {'bytes/tok':>10} {'unk%':>6}"
    print(header)
    print("-" * len(header))
    best = rows[0][1]["fertility"]
    for name, m in rows:
        rel = m["fertility"] / best
        mark = "  ← best" if rel == 1.0 else f"  (x{rel:.2f})"
        print(f"{name[:28]:28} {m['vocab']:>7,} {m['tokens']:>10,} "
              f"{m['fertility']:>10.4f} {m['chars_per_token']:>10.3f} "
              f"{m['bytes_per_token']:>10.3f} {m['unk_pct']:>6.2f}{mark}")
    print("\n※ fertility が小さいほど日本語を短いトークン列で表せる（実効コンテキスト・コスト有利）。")


if __name__ == "__main__":
    main()
