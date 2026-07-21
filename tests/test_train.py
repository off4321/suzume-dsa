"""
最小 pretrain ループのスモークテスト（CPU、TINY）。

- 反復のある小コーパスで数十 step 回すと loss が下がる（学習が進む）
- MTP 有効時に補助損失が有限で出る
- checkpoint 保存 → resume で step が復元される
"""

import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from suzume_dsa import TINY  # noqa: E402
from suzume_dsa.data import (  # noqa: E402
    ByteTokenizer, PackedDataset, batch_size_at, block_size_at,
    load_corpus_tokens, parse_batch_size_schedule, parse_block_size_schedule,
)
from suzume_dsa.train import compute_loss, load_checkpoint, train  # noqa: E402
from suzume_dsa.model import SuzumeGlmDsa  # noqa: E402

# 反復のある学習しやすいコーパス
CORPUS = ("すずめの学習テスト。" * 200) + ("the quick brown fox. " * 200)


def _avg_loss(model, cfg):
    tok = ByteTokenizer()
    data = PackedDataset(load_corpus_tokens(CORPUS, tok), block_size=64)
    x, y = next(data.batches(8, generator=torch.Generator().manual_seed(1)))
    model.eval()
    with torch.no_grad():
        logits, info = model(x)
        loss, _ = compute_loss(logits, info, y, cfg.mtp_loss_coef)
    return float(loss)


def test_loss_decreases():
    cfg = TINY
    out = tempfile.mkdtemp()
    model = train(cfg, CORPUS, steps=60, batch_size=8, block_size=64,
                  lr=3e-3, out_dir=out, log_every=1000, ckpt_every=0, seed=0)
    before = _avg_loss(SuzumeGlmDsa(cfg), cfg)   # 初期化直後
    after = _avg_loss(model, cfg)
    assert after < before, f"loss が下がっていない: {before:.3f} -> {after:.3f}"


def test_mtp_loss_finite():
    cfg = replace(TINY, mtp_depth=2)
    model = SuzumeGlmDsa(cfg).train()
    tok = ByteTokenizer()
    data = PackedDataset(load_corpus_tokens(CORPUS, tok), block_size=64)
    x, y = next(data.batches(4))
    logits, info = model(x)
    loss, parts = compute_loss(logits, info, y, cfg.mtp_loss_coef)
    assert len(info["mtp_logits"]) == 2
    assert parts["mtp"] > 0 and torch.isfinite(loss)


def test_checkpoint_resume():
    cfg = TINY
    out = Path(tempfile.mkdtemp())
    train(cfg, CORPUS, steps=20, batch_size=8, block_size=64, lr=1e-3,
          out_dir=str(out), log_every=1000, ckpt_every=0, seed=0)
    model = SuzumeGlmDsa(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    step = load_checkpoint(out / "model.pt", model, opt)
    assert step == 20


def test_block_size_schedule_parse():
    # 絶対 step
    sched = parse_block_size_schedule("0:64,100:128,200:256", max_steps=300)
    assert sched == [(0, 64), (100, 128), (200, 256)]
    assert block_size_at(0, sched) == 64
    assert block_size_at(99, sched) == 64
    assert block_size_at(100, sched) == 128
    assert block_size_at(250, sched) == 256
    # % 指定（--steps 比）
    pct = parse_block_size_schedule("0%:64,50%:128,80%:256", max_steps=100)
    assert pct == [(0, 64), (50, 128), (80, 256)]


def test_curriculum_run():
    """カリキュラム有効で 1-run 完走し、loss が下がる。"""
    cfg = TINY
    out = tempfile.mkdtemp()
    model = train(cfg, CORPUS, steps=40, batch_size=8, block_size=64,
                  block_size_schedule="0%:32,50%:64", lr=3e-3, out_dir=out,
                  log_every=1000, ckpt_every=0, seed=0)
    before = _avg_loss(SuzumeGlmDsa(cfg), cfg)
    after = _avg_loss(model, cfg)
    assert after < before


def test_batch_size_schedule_parse():
    # 絶対 step。batch は減る向き（block 伸長で VRAM 一定化）。
    sched = parse_batch_size_schedule("0:16,100:8,200:4", max_steps=300)
    assert sched == [(0, 16), (100, 8), (200, 4)]
    assert batch_size_at(0, sched) == 16
    assert batch_size_at(99, sched) == 16
    assert batch_size_at(100, sched) == 8
    assert batch_size_at(250, sched) == 4
    # 増える向きも許可 + % 指定
    pct = parse_batch_size_schedule("0%:4,50%:8", max_steps=100)
    assert pct == [(0, 4), (50, 8)]
    assert batch_size_at(60, pct) == 8


def test_batch_curriculum_run():
    """batch カリキュラム有効で 1-run 完走し、loss が下がる。"""
    cfg = TINY
    out = tempfile.mkdtemp()
    model = train(cfg, CORPUS, steps=40, batch_size=8, block_size=64,
                  batch_size_schedule="0:8,50%:4", lr=3e-3, out_dir=out,
                  log_every=1000, ckpt_every=0, seed=0)
    before = _avg_loss(SuzumeGlmDsa(cfg), cfg)
    after = _avg_loss(model, cfg)
    assert after < before


if __name__ == "__main__":
    test_loss_decreases()
    test_mtp_loss_finite()
    test_checkpoint_resume()
    test_block_size_schedule_parse()
    test_curriculum_run()
    test_batch_size_schedule_parse()
    test_batch_curriculum_run()
    print("all train smoke tests passed")
