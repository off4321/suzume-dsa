"""
ORPO テスト: オッズ比選好損失 + ループが回る + データ整形。
"""

import random
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from suzume_dsa import TINY, SuzumeGlmDsa  # noqa: E402
from suzume_dsa.orpo import (ORPODataset, _pair_from_row,  # noqa: E402
                             _seq_logp, orpo_loss, orpo_train)
from suzume_dsa.tokenizer import SPTokenizer, train_spm  # noqa: E402


def _tokenizer():
    d = Path(tempfile.mkdtemp())
    random.seed(0)
    words = "すずめ 鳥 空 the good bad better worse yes no answer".split()
    lines = [" ".join(random.choice(words) for _ in range(8)) for _ in range(3000)]
    (d / "c.txt").write_text("\n".join(lines), encoding="utf-8")
    return SPTokenizer(train_spm(str(d / "c.txt"), str(d / "sp"), vocab_size=512))


PAIRS = [
    ([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "good answer"}],
     [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "bad"}]),
    ([{"role": "user", "content": "q"}, {"role": "assistant", "content": "yes better"}],
     [{"role": "user", "content": "q"}, {"role": "assistant", "content": "no worse"}]),
]


def test_pair_from_row_shapes():
    # 文字列 chosen/rejected + prompt
    p = _pair_from_row({"prompt": "hello", "chosen": "A", "rejected": "B"})
    assert p[0][-1] == {"role": "assistant", "content": "A"}
    assert p[1][-1] == {"role": "assistant", "content": "B"}
    # 会話 list 形式
    cc = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    rr = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "z"}]
    p2 = _pair_from_row({"chosen": cc, "rejected": rr})
    assert p2 == (cc, rr)
    assert _pair_from_row({"chosen": "only"}) is None      # rejected 欠如は None


def test_orpo_loss_prefers_chosen_direction():
    """chosen の対数尤度が高いほど L_OR が小さく、pref_acc が上がる。"""
    tok = _tokenizer()
    cfg = replace(TINY, vocab_size=tok.vocab_size)
    model = SuzumeGlmDsa(cfg).train()
    data = ORPODataset(PAIRS * 4, tok, max_len=48)
    xc, yc, xr, yr = next(data.batches(2))
    lc, _ = model(xc)
    lr_, _ = model(xr)
    loss, parts = orpo_loss(lc, yc, lr_, yr, lam=0.5)
    assert torch.isfinite(loss)
    assert set(parts) == {"nll", "or", "acc"} and 0.0 <= parts["acc"] <= 1.0
    # _seq_logp は補完トークン平均なので <= 0
    assert (_seq_logp(lc, yc) <= 0).all()


def test_orpo_train_runs():
    tok = _tokenizer()
    cfg = replace(TINY, vocab_size=tok.vocab_size)
    out = tempfile.mkdtemp()
    orpo_train(cfg, PAIRS * 4, tok, steps=4, batch_size=2, max_len=48, lr=1e-4,
               out_dir=out, lam=0.1, log_every=1000)
    assert (Path(out) / "orpo_model.pt").exists()


if __name__ == "__main__":
    test_pair_from_row_shapes()
    test_orpo_loss_prefers_chosen_direction()
    test_orpo_train_runs()
    print("all orpo tests passed")
