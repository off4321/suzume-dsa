"""
suzume-dsa: SentencePiece トークナイザ（本番用）

ByteTokenizer(vocab=256, 疎通用) と同じ encode/decode インターフェースを持ち、
学習・GGUF エクスポートの本番経路で使う。GGUF へは llama(SPM) 形式で
語彙（piece / score / type）をそのまま書き出せる（export_gguf が利用）。

FIM 等の特殊トークンは学習前に user_defined_symbols として語彙へ含める。
"""

from __future__ import annotations

from pathlib import Path

# FIM / チャット用の特殊トークン（語彙構築時に user_defined で埋め込む）
SPECIAL_TOKENS = [
    "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>",
    "<|im_start|>", "<|im_end|>",
]


class SPTokenizer:
    """SentencePiece モデルをラップする。"""

    def __init__(self, model_path: str):
        import sentencepiece as spm
        self.model_path = str(model_path)
        self.sp = spm.SentencePieceProcessor(model_file=self.model_path)

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    def encode(self, text: str) -> list[int]:
        return self.sp.encode(text, out_type=int)

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode([int(i) for i in ids])

    def piece_id(self, piece: str) -> int:
        return self.sp.piece_to_id(piece)

    # ---- GGUF 書き出し用（piece / score / gguf TokenType）----
    def gguf_vocab(self):
        import gguf
        T = gguf.TokenType
        tokens, scores, types = [], [], []
        for i in range(self.vocab_size):
            tokens.append(self.sp.id_to_piece(i))
            scores.append(float(self.sp.get_score(i)))
            if self.sp.is_unknown(i):
                types.append(T.UNKNOWN)
            elif self.sp.is_control(i):
                types.append(T.CONTROL)
            elif self.sp.is_byte(i):
                types.append(T.BYTE)
            elif self.sp.is_unused(i):
                types.append(T.UNUSED)
            else:
                types.append(T.NORMAL)
        return tokens, scores, types


def train_spm(corpus_path: str, out_prefix: str, vocab_size: int = 32000,
              character_coverage: float = 0.9995,
              special_tokens: list[str] | None = None,
              input_sentence_size: int = 0,
              shuffle_input_sentence: bool = False,
              seed_sentencepiece_size: int = 1_000_000) -> str:
    """コーパスから SentencePiece(unigram) を学習し .model を書き出してパスを返す。

    input_sentence_size: 0以外を指定すると、学習前にその件数までランダムに間引く
    （SentencePiece自身が "Too many sentences are loaded" 警告と共に推奨する対処）。
    32k語彙程度なら数十万〜100万文で十分なことが多く、コーパスが大きすぎて
    suffix array構築が終わらない（＝チェックポイント不可のため丸ごとやり直しになる）
    事故を避けられる。shuffle_input_sentenceはinput_sentence_size指定時にTrue推奨
    （先頭からの偏った間引きを避ける）。

    seed_sentencepiece_size: EMで削っていく前の候補語彙数（既定100万、SentencePiece
    のライブラリ既定と同じ）。vocab_sizeをこれに近い/超える値（例: 250k）にする場合は、
    削る余地が無くなり質が落ちるため、vocab_sizeの3〜4倍程度まで引き上げること
    （例: vocab_size=250000 なら seed_sentencepiece_size=1_000_000〜2_000_000）。
    suffix array構築のコストはコーパスサイズで決まりvocab_size自体には依らないが、
    seed_sentencepiece_sizeを上げるとここだけ多少コストが増える。
    """
    import sentencepiece as spm
    special = SPECIAL_TOKENS if special_tokens is None else special_tokens
    spm.SentencePieceTrainer.train(
        input=corpus_path, model_prefix=out_prefix, vocab_size=vocab_size,
        model_type="unigram", character_coverage=character_coverage,
        user_defined_symbols=special,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        byte_fallback=True,
        max_sentence_length=1 << 20,      # 長い行も1文として扱う（既定4192で切られるのを回避）
        hard_vocab_limit=False,           # vocab_size を上限扱い（小コーパスでも学習を通す）
        # 大規模コーパス（総文字数が int32=約21億を超える）でも学習を通す。
        # 超えると "Input corpus too large" で落ちるため常時有効化（小コーパスでも安全）。
        train_extremely_large_corpus=True,
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=shuffle_input_sentence,
        seed_sentencepiece_size=seed_sentencepiece_size,
    )
    return str(Path(out_prefix).with_suffix(".model"))


__all__ = ["SPTokenizer", "train_spm", "SPECIAL_TOKENS"]
