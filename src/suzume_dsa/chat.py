"""
suzume-dsa: チャットテンプレート（SFT 用）

ChatML 風テンプレートで会話を平坦化し、**assistant の発話だけを教師化**する
（user/system やヘッダは labels=-100 で損失から除外）。特殊トークン
<|im_start|> / <|im_end|> は tokenizer.SPECIAL_TOKENS に含まれる（SP の
user_defined_symbols として 1 トークンに保たれる）。
"""

from __future__ import annotations

import re

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
IGNORE = -100

# 列マッピング DSL（"instruction=question,output=answer" 等）の論理キー同義語。
# 別カラムに分かれた instruct データを 1 会話（user→assistant）へ組み立てるため。
_SYSTEM_KEYS = {"system"}
_USER_KEYS = {"instruction", "input", "question", "prompt", "problem", "user"}
_REASON_KEYS = {"reasoning", "thinking", "thought", "cot", "rationale"}
_ASST_KEYS = {"output", "answer", "response", "completion", "assistant"}


def _wrap_reasoning(reason: str, answer: str) -> str:
    """reasoning があれば <think>…</think> で包んで answer の前に付ける。"""
    return f"<think>\n{reason.strip()}\n</think>\n\n{answer}" if reason.strip() else answer


def conversation_from_columns(row: dict, mapping: dict) -> list[dict]:
    """別カラムに分かれた 1 行を単一ターン会話 [{"role","content"}, ...] へ。

    mapping は {論理キー: カラム名}（例 {"instruction":"question","output":"answer"}）。
    論理キーは _USER_KEYS / _ASST_KEYS / _REASON_KEYS / _SYSTEM_KEYS の同義語で解釈。
    """
    system, user_parts, reason, answer = "", [], "", ""
    for logical, col in mapping.items():
        val = row.get(col)
        val = "" if val is None else str(val)
        lk = logical.strip().lower()
        if lk in _SYSTEM_KEYS:
            system = val
        elif lk in _REASON_KEYS:
            reason = val
        elif lk in _ASST_KEYS:
            answer = val
        else:                       # user 系 or 未知キーは user 入力として連結
            user_parts.append(val)
    turns: list[dict] = []
    if system.strip():
        turns.append({"role": "system", "content": system})
    user = "\n\n".join(p for p in user_parts if p.strip())
    if user.strip():
        turns.append({"role": "user", "content": user})
    asst = _wrap_reasoning(reason, answer)
    if asst.strip():
        turns.append({"role": "assistant", "content": asst})
    return turns


_HARMONY_RE = re.compile(
    r"<\|start\|>(?P<role>[^<]*?)(?:<\|channel\|>(?P<channel>[^<]*?))?"
    r"<\|message\|>(?P<content>.*?)(?:<\|end\|>|<\|return\|>)",
    re.S,
)


def parse_harmony(text: str) -> list[dict]:
    """harmony 形式文字列 → [{"role","content"}]。

    assistant の analysis チャネル（reasoning）は直後の final と統合し <think> で包む。
    マーカーが 1 つも無ければ [] を返す（呼び出し側でスキップ）。
    """
    turns: list[dict] = []
    pending_think = None
    for m in _HARMONY_RE.finditer(text):
        role = m.group("role").strip()
        role = {"human": "user", "gpt": "assistant"}.get(role, role)
        channel = (m.group("channel") or "").strip()
        content = m.group("content").strip()
        if role == "assistant" and channel in ("analysis", "reasoning", "think"):
            pending_think = content
            continue
        if role == "assistant":
            content = _wrap_reasoning(pending_think or "", content)
            pending_think = None
        turns.append({"role": role, "content": content})
    return turns


def build_chatml_template(default_system: str) -> str:
    """ChatML の Jinja チャットテンプレート（GGUF の tokenizer.chat_template 用）。

    system メッセージが無いときは default_system を自動注入する。これにより
    llama.cpp 側で「名乗り」の既定システムプロンプトが常に効く（アイデンティティ）。
    """
    ds = default_system.replace('"', '\\"')
    return (
        "{% if messages[0]['role'] == 'system' %}"
        "{{ '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}"
        "{% set loop_messages = messages[1:] %}"
        "{% else %}"
        f"{{{{ '<|im_start|>system\\n{ds}<|im_end|>\\n' }}}}"
        "{% set loop_messages = messages %}"
        "{% endif %}"
        "{% for message in loop_messages %}"
        "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    )


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


__all__ = ["build_sft_example", "normalize_turn", "conversation_from_columns",
           "parse_harmony", "build_chatml_template", "IM_START", "IM_END", "IGNORE"]
