"""
suzume-dsa: SentencePiece トークナイザ学習 CLI

テキストファイル、または HF データセットから unigram トークナイザを学習して .model を書く。
FIM / chat の特殊トークンは自動で語彙に含める。

使い方:
    uv run tools/train_tokenizer.py --corpus corpus.txt --vocab-size 32000 --out tokenizer/sp
    uv run tools/train_tokenizer.py --hf-dataset "range3/wikipedia-ja-20230101" \
        --hf-max-samples 500000 --vocab-size 32000 --out tokenizer/sp
"""

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from suzume_dsa.data import ByteTokenizer, load_hf_tokens, preview_hf_dataset  # noqa: E402
from suzume_dsa.tokenizer import train_spm  # noqa: E402


def _dump_hf_to_text(specs, out_txt: Path, max_samples: int, hf_token: str | None) -> None:
    """複数 HF データセットのテキスト列を 1 行ずつ 1 ファイルへ追記（SP 学習の入力用）。

    読めない spec（config 必須・カラム不明など）は警告してスキップし、残りで続行する。
    """
    from datasets import load_dataset

    from suzume_dsa.data import _cell_to_text, _guess_text_column, _parse_hf_spec

    if isinstance(specs, str):
        specs = [specs]
    total = 0
    with out_txt.open("w", encoding="utf-8") as f:
        for spec in specs:
            path, config, split, column = _parse_hf_spec(spec)
            try:
                ds = load_dataset(path, config, split=split or "train",
                                  streaming=True, token=hf_token)
                n = 0
                for i, row in enumerate(ds):
                    if i >= max_samples:
                        break
                    col = column or _guess_text_column(row)
                    text = _cell_to_text(row.get(col)) if col else ""
                    if text:
                        f.write(text.replace("\n", " ") + "\n")
                        n += 1
            except Exception as e:  # noqa: BLE001 - 1 件ダメでも継続
                print(f"[warn] スキップ（読込失敗）: {spec}  ({type(e).__name__}: {e})")
                continue
            print(f"[tok] {spec}: {n} 行")
            total += n
    print(f"HF → {out_txt}: 合計 {total} 行書き出し")


def main() -> None:
    ap = argparse.ArgumentParser(description="SentencePiece トークナイザ学習")
    ap.add_argument("--corpus", default=None, help="テキストファイル")
    ap.add_argument("--hf-dataset", nargs="+", default=None,
                    help='HF "path[:config][:split][:column]"。複数指定で全部を語彙学習に使う')
    ap.add_argument("--hf-max-samples", type=int, default=1_000_000)
    ap.add_argument("--hf-token", default=None)
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--character-coverage", type=float, default=0.9995)
    ap.add_argument("--out", default="tokenizer/sp", help="出力 model_prefix（→ <out>.model）")
    args = ap.parse_args()

    corpus = args.corpus
    tmp = None
    if args.hf_dataset:
        tmp = Path(tempfile.mkdtemp()) / "hf_corpus.txt"
        _dump_hf_to_text(args.hf_dataset, tmp, args.hf_max_samples, args.hf_token)
        corpus = str(tmp)
    assert corpus, "--corpus か --hf-dataset が必要です"

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model_path = train_spm(corpus, args.out, vocab_size=args.vocab_size,
                           character_coverage=args.character_coverage)
    print(f"wrote {model_path}")


if __name__ == "__main__":
    main()
