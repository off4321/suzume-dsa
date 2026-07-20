"""
学習効率化機能のテスト: Muon / LR スケジュール / FIM / 選択backprop / 深さ成長。
"""

import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from suzume_dsa import TINY, SuzumeGlmDsa  # noqa: E402
from suzume_dsa.fim import fim_transform, build_fim_batch  # noqa: E402
from suzume_dsa.optim import (  # noqa: E402
    Muon, split_muon_params, lr_lambda_wsd, lr_lambda_cosine, build_optimizer,
)
from suzume_dsa.train import compute_loss, load_init_weights, train  # noqa: E402

CORPUS = ("すずめ dsa test " * 300)


def test_muon_splits_and_steps():
    model = SuzumeGlmDsa(TINY)
    muon_p, adamw_p = split_muon_params(model)
    # token_embd は AdamW 側、隠れ 2D 重みは Muon 側
    assert any(p.ndim == 2 for p in muon_p)
    assert all(p is not model.token_embd.weight for p in muon_p)
    opt = build_optimizer(model, lr=1e-3, optimizer="muon", muon_lr=1e-2)
    x = torch.randint(0, TINY.vocab_size, (2, 16))
    logits, info = model(x)
    loss, _ = compute_loss(logits, info, x, TINY.mtp_loss_coef)
    loss.backward()
    before = model.blocks[1].attn.wq_a.weight.detach().clone()
    opt.step()
    assert not torch.equal(before, model.blocks[1].attn.wq_a.weight)  # 更新された


def test_lr_schedules():
    # cosine: warmup 後に単調減衰
    assert lr_lambda_cosine(0, 10, 100) < lr_lambda_cosine(9, 10, 100)
    assert lr_lambda_cosine(10, 10, 100) > lr_lambda_cosine(99, 10, 100)
    # WSD: stable 相は 1.0、decay 相で下がる
    assert lr_lambda_wsd(50, 10, 100, decay_frac=0.2) == 1.0
    assert lr_lambda_wsd(95, 10, 100, decay_frac=0.2) < 1.0


def test_fim_transform_length():
    content = torch.arange(20)
    import random
    seq = fim_transform(content, 100, 101, 102, random.Random(0))
    assert seq.numel() == 23                          # +3 センチネル
    assert set([100, 101, 102]).issubset(set(seq.tolist()))


def test_fim_batch_shape():
    tokens = torch.arange(500) % 90
    gen = torch.Generator().manual_seed(0)
    x, y = build_fim_batch(tokens, batch_size=4, block_size=64,
                           fim_ids=(90, 91, 92), generator=gen, rate=1.0)
    assert x.shape == (4, 64) and y.shape == (4, 64)


def test_selective_backprop_reduces_tokens():
    model = SuzumeGlmDsa(TINY)
    x = torch.randint(0, TINY.vocab_size, (2, 16))
    logits, info = model(x)
    full, _ = compute_loss(logits, info, x, 0.0, select_topp=1.0)
    top, _ = compute_loss(logits, info, x, 0.0, select_topp=0.3)
    # 上位のみ平均するので選択損失のほうが大きい
    assert top.item() >= full.item()


def test_depth_growth_init_from():
    small = replace(TINY, n_layer=3)
    big = replace(TINY, n_layer=5)
    out = Path(tempfile.mkdtemp())
    train(small, CORPUS, steps=5, batch_size=4, block_size=32, lr=1e-3,
          out_dir=str(out), log_every=1000, ckpt_every=0)
    model = SuzumeGlmDsa(big)
    load_init_weights(model, str(out / "model.pt"))    # 3層→5層継承（例外なく通る）


if __name__ == "__main__":
    for fn in [test_muon_splits_and_steps, test_lr_schedules, test_fim_transform_length,
               test_fim_batch_shape, test_selective_backprop_reduces_tokens,
               test_depth_growth_init_from]:
        fn()
    print("all efficiency tests passed")
