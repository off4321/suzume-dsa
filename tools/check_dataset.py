"""
suzume-dsa: HuggingFace データセットの下見ツール

学習に使う前に、読めるか・どのカラムがテキストか・中身の例を確認する。
`datasets` が必要（uv sync で入る）。

spec 書式: "path[:config][:split][:column]"（path は namespace/name）

使い方:
    uv run tools/check_dataset.py "range3/wikipedia-ja-20230101" --split "train[:5]"
    uv run tools/check_dataset.py "Salesforce/wikitext:wikitext-2-raw-v1" --column text --n 3
    uv run tools/check_dataset.py "allenai/c4:en" --stream --max-samples 200   # 巨大データ
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from suzume_dsa.data import preview_hf_dataset  # noqa: E402


def _preview_sft(spec: str, split, max_samples: int, n: int, hf_token) -> int:
    """SFT spec を実際の loader で会話へ変換して先頭 n 件を表示（DSL/harmony も検証）。"""
    from suzume_dsa.sft import load_sft_conversations

    convs = load_sft_conversations(spec, split=split, max_samples=max_samples,
                                   hf_token=hf_token, stream=True)
    if not convs:
        print("会話 0 件（messages/conversations 列なし、または DSL/harmony が空）。"
              "spec の 4 番目フィールド（列名 / key=col,... / format=harmony）を確認。")
        return 1
    print(f"会話 {len(convs)} 件（先頭 {min(n, len(convs))} 件を表示）:")
    for conv in convs[:n]:
        print("-" * 60)
        for turn in conv:
            role = turn.get("role") or turn.get("from") or "?"
            content = turn.get("content") or turn.get("value") or ""
            print(f"[{role}] {str(content)[:200]}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="HuggingFace データセットの下見")
    p.add_argument("spec", help='"path[:config][:split][:column]"')
    p.add_argument("--column", default=None, help="テキストカラム名（省略時は自動判定）")
    p.add_argument("--split", default=None, help="スプリット（例 train, 'train[:100]'）")
    p.add_argument("--max-samples", type=int, default=200, help="読み込む上限行数")
    p.add_argument("--n", type=int, default=3, help="表示するサンプル数")
    p.add_argument("--hf-token", default=None, help="gated/private 用（無ければ環境変数 HF_TOKEN）")
    p.add_argument("--stream", action="store_true", help="streaming で先頭だけ読む（c4 等）")
    p.add_argument("--sft", action="store_true",
                   help="会話データとして下見（messages/conversations/列マッピング/harmony）")
    args = p.parse_args()

    if args.sft:
        return _preview_sft(args.spec, args.split, args.max_samples, args.n, args.hf_token)

    info = preview_hf_dataset(
        args.spec, column=args.column, split=args.split, max_samples=args.max_samples,
        n=args.n, hf_token=args.hf_token, stream=args.stream, verbose=True)
    if not info.get("ok"):
        print("ヒント: path は 'namespace/name' か / config 名が要らないか / "
              "gated なら --hf-token（or 環境変数 HF_TOKEN）を確認。")
        return 1
    if not info.get("usable"):
        print("テキストカラムを特定できませんでした。--column で明示してください。")
        return 1
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # HF の streaming 後片付けスレッドが interpreter finalize 時に
    # "Fatal Python error: PyGILState_Release" で落ちるのを避けるため、
    # Python のファイナライザを走らせずに即終了する（出力は上で flush 済み）。
    os._exit(code)
