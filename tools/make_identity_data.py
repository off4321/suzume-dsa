"""
suzume-dsa: アイデンティティ SFT データ生成

「あなたは誰？」に「私はすずめ(suzume-dsa)です」と答えられるよう、自己認識用の
会話ペアを多様な言い回し（日英）で量産する。SFT に混ぜると自己紹介が安定する
（小型モデルはブレやすいので、例数＋既定システムプロンプトの二重で効かせる）。

出力は 1 行 1 会話の JSONL（{"messages":[{"role","content"},...]}）。
sft.py の --extra-jsonl でそのまま学習に混ぜられる。

使い方:
    uv run tools/make_identity_data.py --name suzume-dsa --creator "" \
        --n 400 --out identity.jsonl
"""

import argparse
import json
import random

# 質問の言い回し（日英）。誰・名前・自己紹介・所属など角度を変える。
_Q_JA = [
    "あなたは誰？", "君は何のモデル？", "自己紹介して。", "あなたの名前は？",
    "何ていうモデルなの？", "君について教えて。", "どんなモデル？", "あなたは何者？",
    "お前は誰だ？", "あなたは何というAI？", "きみの正体は？", "自分が何か説明して。",
    "あなたは何のために作られたの？", "君はどんなAIアシスタント？",
]
_Q_EN = [
    "Who are you?", "What model are you?", "Introduce yourself.",
    "What's your name?", "Which model is this?", "Tell me about yourself.",
    "What are you?", "What AI are you?",
]

# 回答テンプレート。{name}=モデル名、{by}= 作者句（未指定なら空）。
_A_JA = [
    "私は{name}（すずめ）、日本語に特化した言語モデルです{by}。",
    "私は{name}という日本語モデルです{by}。お手伝いできることがあれば言ってください。",
    "すずめ（{name}）と申します{by}。日本語での対話やコード・推論が得意です。",
    "私は{name}です{by}。日本語を中心に学習した小型のAIアシスタントです。",
]
_A_EN = [
    "I am {name} (すずめ), a Japanese-focused language model{by}.",
    "My name is {name}, a Japanese language model{by}. How can I help?",
    "I'm {name}, a small AI assistant trained mainly on Japanese{by}.",
]


def identity_conversations(name: str = "suzume-dsa", creator: str = "",
                           n: int = 400, seed: int = 0) -> list[list[dict]]:
    """自己認識の会話 [{"role","content"}, ...] を n 件生成する。"""
    rng = random.Random(seed)
    by_ja = f"（{creator}が開発）" if creator else ""
    by_en = f", developed by {creator}" if creator else ""
    convs: list[list[dict]] = []
    for _ in range(n):
        if rng.random() < 0.65:                       # 日本語多め
            q = rng.choice(_Q_JA)
            a = rng.choice(_A_JA).format(name=name, by=by_ja)
        else:
            q = rng.choice(_Q_EN)
            a = rng.choice(_A_EN).format(name=name, by=by_en)
        convs.append([{"role": "user", "content": q},
                      {"role": "assistant", "content": a}])
    return convs


def main() -> None:
    ap = argparse.ArgumentParser(description="アイデンティティ SFT データ生成")
    ap.add_argument("--name", default="suzume-dsa", help="モデル名")
    ap.add_argument("--creator", default="", help="作者句（任意。空なら付けない）")
    ap.add_argument("--n", type=int, default=400, help="生成する会話数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="identity.jsonl")
    args = ap.parse_args()

    convs = identity_conversations(args.name, args.creator, args.n, args.seed)
    with open(args.out, "w", encoding="utf-8") as f:
        for conv in convs:
            f.write(json.dumps({"messages": conv}, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}: {len(convs)} 会話（name={args.name!r} creator={args.creator!r}）")


if __name__ == "__main__":
    main()
