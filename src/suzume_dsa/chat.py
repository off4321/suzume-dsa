"""
suzume-dsa: チャットテンプレート（SFT 用）

ChatML 風テンプレートで会話を平坦化し、**assistant の発話だけを教師化**する
（user/system やヘッダは labels=-100 で損失から除外）。特殊トークン
<|im_start|> / <|im_end|> は tokenizer.SPECIAL_TOKENS に含まれる（SP の
user_defined_symbols として 1 トークンに保たれる）。
"""

from __future__ import annotations

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
IGNORE = -100


def normalize_turn(turn: dict) -> dict:
    """{"role","content"} へ正規化（from/value や reasoning 形式にも軽く対応）。"""
    role = turn.get("role") or turn.get("from") or "user"
    role = {"human": "user", "gpt": "assistant", "system": "system"}.get(role, role)
    content = turn.get("content")
    if content is None:
        content = turn.get("value", "")
    return {"role": role, "content": str(content)}


def build_sft_example(messages: list[dict], tokenizer) -> tuple[list[int], list[int]]:
    """会話を (input_ids, labels) にする。labels は assistant 発話のみ実 id、他は -100。"""
    ids: list[int] = []
    labels: list[int] = []

    def add(text: str, supervise: bool) -> None:
        toks = tokenizer.encode(text)
        ids.extend(toks)
        labels.extend(toks if supervise else [IGNORE] * len(toks))

    for turn in messages:
        m = normalize_turn(turn)
        is_asst = m["role"] == "assistant"
        add(f"{IM_START}{m['role']}\n", supervise=False)   # ヘッダは教師化しない
        add(m["content"], supervise=is_asst)
        add(f"{IM_END}\n", supervise=is_asst)              # 終端も assistant のみ教師化
    return ids, labels


__all__ = ["build_sft_example", "normalize_turn", "IM_START", "IM_END", "IGNORE"]
