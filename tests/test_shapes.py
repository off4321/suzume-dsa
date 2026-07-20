"""
glm-dsa v1 モデルの形状・疎通テスト。

- forward が (B, T, V) の logits を返すこと
- MTP 有効時に補助 logits が出ること
- MoE の aux-free bias 更新が負荷の偏りを是正する向きに動くこと
- state_dict のテンソル形状が docs/glm-dsa-tensors.md の要求と一致すること
"""

import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from suzume_dsa import TINY, SuzumeGlmDsa  # noqa: E402


def test_forward_shape():
    model = SuzumeGlmDsa(TINY)
    ids = torch.randint(0, TINY.vocab_size, (2, 16))
    logits, info = model(ids)
    assert logits.shape == (2, 16, TINY.vocab_size)
    assert "mtp_logits" not in info


def test_mtp_train_only():
    cfg = replace(TINY, mtp_depth=2)
    model = SuzumeGlmDsa(cfg).train()
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    _, info = model(ids)
    assert len(info["mtp_logits"]) == 2
    assert info["mtp_logits"][0].shape == (2, 15, cfg.vocab_size)  # T-1
    # eval では MTP を回さない
    model.eval()
    _, info = model(ids)
    assert "mtp_logits" not in info


def test_router_bias_balances_load():
    model = SuzumeGlmDsa(TINY).train()
    moe = next(m for b in model.blocks if hasattr(b.ffn, "exp_probs_b") for m in [b.ffn])
    # 1つの Expert に偏らせて負荷を注入 → commit で bias が下がる向きに動く
    moe._load[0] = 100.0
    before = moe.exp_probs_b[0].item()
    moe.commit_router_bias_updates()
    assert moe.exp_probs_b[0].item() < before


def test_tensor_shapes_match_spec():
    """v1 が持つべき glm-dsa テンソルの形状を代表点で検証。"""
    cfg = TINY
    model = SuzumeGlmDsa(cfg)
    sd = dict(model.named_parameters())
    D, H = cfg.n_embd, cfg.n_head
    hk, hv, nope, rope = cfg.head_dim_k, cfg.head_dim_v, cfg.head_dim_nope, cfg.head_dim_rope
    Lq, Lkv = cfg.q_lora_rank, cfg.kv_lora_rank

    # グローバル（tie なので output は持たない）
    assert sd["token_embd.weight"].shape == (cfg.vocab_size, D)
    assert model.output is None

    a = "blocks.1.attn."      # 層1は MoE 層（dense_lead=1）
    assert sd[a + "wq_a.weight"].shape == (Lq, D)
    assert sd[a + "q_a_norm.weight"].shape == (Lq,)
    assert sd[a + "wq_b.weight"].shape == (H * hk, Lq)
    assert sd[a + "wkv_a_mqa.weight"].shape == (Lkv + rope, D)
    assert sd[a + "kv_a_norm.weight"].shape == (Lkv,)
    assert sd[a + "wk_b"].shape == (nope, Lkv, H)
    assert sd[a + "wv_b"].shape == (Lkv, hv, H)
    assert sd[a + "wo.weight"].shape == (D, H * hv)

    f = "blocks.1.ffn."
    assert sd[f + "gate_inp.weight"].shape == (cfg.n_expert, D)
    assert sd[f + "w_gate"].shape == (cfg.n_expert, cfg.n_ff_exp, D)
    assert sd[f + "w_down"].shape == (cfg.n_expert, D, cfg.n_ff_exp)
    assert sd[f + "shared.gate.weight"].shape == (cfg.n_ff_exp * cfg.n_expert_shared, D)

    # 先頭層は密 FFN
    assert "blocks.0.ffn.gate.weight" in sd


if __name__ == "__main__":
    test_forward_shape()
    test_mtp_train_only()
    test_router_bias_balances_load()
    test_tensor_shapes_match_spec()
    print("all shape tests passed")
