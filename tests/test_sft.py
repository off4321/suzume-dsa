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
from suzume_dsa.chat import (IGNORE, build_sft_example,  # noqa: E402
                             conversation_from_columns, normalize_turn, parse_harmony)
from suzume_dsa.sft import (SFTDataset, _parse_sft_field,  # noqa: E402
                            _row_to_turns, sft_loss, sft_train)
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


def test_sft_uses_mtp_and_select_topp():
    """SFT でも MTP 補助損失（mtp_depth>0）と選択的 backprop が事前学習と同じ経路で効く。"""
    tok = _tokenizer()
    cfg = replace(TINY, vocab_size=tok.vocab_size, mtp_depth=1)   # MTP 有効
    # 1 ステップの loss が有限で MTP 項が出ること（compute_loss 経路の疎通）
    from suzume_dsa.chat import IGNORE as IG
    from suzume_dsa.train import compute_loss
    data = SFTDataset(CONV * 8, tok, max_len=64)
    x, y = next(data.batches(4))
    model = SuzumeGlmDsa(cfg)
    targets = torch.full_like(y, IG)
    targets[:, :-1] = y[:, 1:]
    logits, info = model(x)
    assert "mtp_logits" in info and len(info["mtp_logits"]) == 1
    loss, parts = compute_loss(logits, info, targets, cfg.mtp_loss_coef,
                               select_topp=0.7, ignore_index=IG)
    assert torch.isfinite(loss) and parts["mtp"] > 0.0
    # ループも回る（MTP + select-topp + mup 同時）
    out = tempfile.mkdtemp()
    sft_train(cfg, CONV * 8, tok, steps=4, batch_size=4, max_len=64, lr=1e-3,
              out_dir=out, select_topp=0.7, mup=True, log_every=1000)


def test_sft_field_parsing_and_conversion():
    """spec 4 番目フィールドの各形式（列名 / mapping / harmony）を会話へ変換できる。"""
    # 列マッピング DSL: 別カラム → user→assistant（reasoning は <think> で包む）
    mode, arg = _parse_sft_field("instruction=question,reasoning=reasoning,output=answer")
    assert mode == "mapping"
    turns = _row_to_turns({"question": "2+2?", "reasoning": "足す", "answer": "4"},
                          mode, arg, normalize=normalize_turn)
    assert [t["role"] for t in turns] == ["user", "assistant"]
    assert "<think>" in turns[1]["content"] and "4" in turns[1]["content"]
    # system 付き・reasoning 無しは <think> を付けない
    plain = conversation_from_columns({"s": "sys", "i": "q", "o": "a"},
                                      {"system": "s", "instruction": "i", "output": "o"})
    assert [t["role"] for t in plain] == ["system", "user", "assistant"]
    assert "<think>" not in plain[-1]["content"]

    # harmony: analysis チャネルは直後の final と統合
    mode, arg = _parse_sft_field("format=harmony")
    assert mode == "harmony"
    h = ("<|start|>user<|message|>hi<|end|>"
         "<|start|>assistant<|channel|>analysis<|message|>think<|end|>"
         "<|start|>assistant<|channel|>final<|message|>yo<|return|>")
    turns = _row_to_turns({"text": h}, mode, arg, normalize=normalize_turn)
    assert [t["role"] for t in turns] == ["user", "assistant"]
    assert "<think>" in turns[1]["content"] and "yo" in turns[1]["content"]
    assert parse_harmony("no markers here") == []

    # 明示列名 / auto は従来どおり会話リストをそのまま使う
    assert _parse_sft_field("conversations") == ("column", "conversations")
    assert _parse_sft_field(None) == ("auto", None)


if __name__ == "__main__":
    test_only_assistant_supervised()
    test_sft_loop_runs_and_masks()
    test_sft_uses_checkpoint_cfg_not_4b()
    test_sft_uses_mtp_and_select_topp()
    test_sft_field_parsing_and_conversion()
    print("all sft tests passed")
