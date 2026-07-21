"""
SentencePiece トークナイザ + 実語彙の GGUF 書き出しテスト。
"""

import random
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from suzume_dsa import TINY, SuzumeGlmDsa  # noqa: E402
from suzume_dsa.export_gguf import export  # noqa: E402
from suzume_dsa.tokenizer import SPTokenizer, train_spm  # noqa: E402
from tokenizer_eval import measure, measure_sp  # noqa: E402
import gguf  # noqa: E402


def _train_tokenizer():
    d = Path(tempfile.mkdtemp())
    random.seed(0)
    words = "すずめ 鳥 空 森 川 山 the quick brown fox lazy dog river".split()
    lines = [" ".join(random.choice(words) for _ in range(8)) for _ in range(3000)]
    (d / "c.txt").write_text("\n".join(lines), encoding="utf-8")
    mp = train_spm(str(d / "c.txt"), str(d / "sp"), vocab_size=512)
    return SPTokenizer(mp), d


def test_sp_roundtrip_and_specials():
    tok, _ = _train_tokenizer()
    assert tok.vocab_size > 256
    assert tok.piece_id("<|fim_prefix|>") >= 0        # FIM センチネルが語彙にある
    assert "fox" in tok.decode(tok.encode("the fox"))


def test_sp_vocab_written_to_gguf():
    tok, d = _train_tokenizer()
    cfg = replace(TINY, vocab_size=tok.vocab_size)
    path = str(d / "m.gguf")
    export(SuzumeGlmDsa(cfg).eval(), path, tokenizer=tok)
    r = gguf.GGUFReader(path)
    model = r.get_field("tokenizer.ggml.model")
    assert model.parts[model.data[0]].tobytes().decode() == "llama"
    assert len(r.get_field("tokenizer.ggml.tokens").data) == tok.vocab_size


def test_tokenizer_fertility_metrics():
    """fertility 等の指標が整合する（tokens/chars、chars/token の逆数関係など）。"""
    text = "すずめ すずめ 鳥 the fox " * 50
    # 純関数 measure: 既知の encode でチェック（1文字1トークンなら fertility=1）
    m = measure(lambda t: list(range(len(t))), text)
    assert abs(m["fertility"] - 1.0) < 1e-9
    assert abs(m["chars_per_token"] - 1.0) < 1e-9
    # 実 SP: fertility>0、chars/token = 1/fertility が成り立つ
    tok, _ = _train_tokenizer()
    ms = measure_sp(tok.model_path, text)
    assert ms["tokens"] > 0 and ms["fertility"] > 0
    assert abs(ms["chars_per_token"] - 1.0 / ms["fertility"]) < 1e-6
    assert ms["vocab"] == tok.vocab_size


if __name__ == "__main__":
    test_sp_roundtrip_and_specials()
    test_sp_vocab_written_to_gguf()
    test_tokenizer_fertility_metrics()
    print("all tokenizer/export tests passed")
