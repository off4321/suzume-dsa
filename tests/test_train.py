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
from suzume_dsa.data import ByteTokenizer, PackedDataset, load_corpus_tokens  # noqa: E402
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


if __name__ == "__main__":
    test_loss_decreases()
    test_mtp_loss_finite()
    test_checkpoint_resume()
    print("all train smoke tests passed")
