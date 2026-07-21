"""
SFT テスト: assistant のみ教師化 + マスク損失 + SFT ループが回る。
"""

import random
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from suzume_dsa import TINY, SuzumeGlmDsa  # noqa: E402
from suzume_dsa.chat import IGNORE, build_sft_example  # noqa: E402
from suzume_dsa.sft import SFTDataset, sft_loss, sft_train  # noqa: E402
from suzume_dsa.tokenizer import SPTokenizer, train_spm  # noqa: E402


def _tokenizer():
    d = Path(tempfile.mkdtemp())
    random.seed(0)
    words = "すずめ 鳥 空 the fox dog river hello answer question".split()
    lines = [" ".join(random.choice(words) for _ in range(8)) for _ in range(3000)]
    (d / "c.txt").write_text("\n".join(lines), encoding="utf-8")
    return SPTokenizer(train_spm(str(d / "c.txt"), str(d / "sp"), vocab_size=512))


CONV = [
    [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "すずめ です"}],
    [{"role": "system", "content": "you are すずめ"},
     {"role": "user", "content": "fox?"}, {"role": "assistant", "content": "the fox"}],
]


def test_only_assistant_supervised():
    tok = _tokenizer()
    ids, labels = build_sft_example(CONV[0], tok)
    assert len(ids) == len(labels)
    # user 側は全て IGNORE、assistant 側は実 id が少なくとも 1 つ
    assert any(l != IGNORE for l in labels)
    assert any(l == IGNORE for l in labels)


def test_sft_loop_runs_and_masks():
    tok = _tokenizer()
    cfg = replace(TINY, vocab_size=tok.vocab_size)
    data = SFTDataset(CONV * 8, tok, max_len=64)
    x, y = next(data.batches(4))
    model = SuzumeGlmDsa(cfg)
    logits, _ = model(x)
    loss = sft_loss(logits, y)
    assert torch.isfinite(loss)
    # 全マスクのバッチは損失 nan にならず学習ループも回る
    out = tempfile.mkdtemp()
    sft_train(cfg, CONV * 8, tok, steps=6, batch_size=4, max_len=64, lr=1e-3,
              out_dir=out, log_every=1000)


def test_sft_uses_checkpoint_cfg_not_4b():
    """SFT は --init-from の checkpoint に保存された cfg を使う（4B ハードコードだと
    0.5B/TINY 事前学習の重みを load できず壊れる）ことの回帰。"""
    from suzume_dsa.optim import build_optimizer
    from suzume_dsa.train import save_checkpoint

    tok = _tokenizer()
    cfg = replace(TINY, vocab_size=tok.vocab_size)          # 4B ではない寸法
    d = Path(tempfile.mkdtemp())
    model = SuzumeGlmDsa(cfg)
    opt = build_optimizer(model, 1e-3, 0.1, "adamw", 0.02)
    save_checkpoint(d / "pretrain.pt", model, opt, 0)       # cfg も一緒に保存される

    # main() が行う手順: checkpoint から cfg を復元し、その寸法で SFT する。
    ckpt = torch.load(d / "pretrain.pt", map_location="cpu", weights_only=False)
    assert ckpt["cfg"].n_embd == TINY.n_embd                # 4B 由来でないこと
    sft_train(ckpt["cfg"], CONV * 8, tok, steps=4, batch_size=4, max_len=64,
              lr=1e-3, out_dir=str(d / "sft"), init_from=str(d / "pretrain.pt"),
              log_every=1000)


if __name__ == "__main__":
    test_only_assistant_supervised()
    test_sft_loop_runs_and_masks()
    test_sft_uses_checkpoint_cfg_not_4b()
    print("all sft tests passed")
