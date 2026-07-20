"""
GGUF エクスポートの契約テスト（llama.cpp 非依存、gguf リーダで検証）。

- general.architecture == "glm-dsa"
- 必須テンソルが揃い、ne 形状が llama.cpp の期待（docs/glm-dsa-tensors.md）と一致
- tie 時に output.weight を持たない

実機ロード（llama-cli）での疎通は別途確認済み。ここは形状契約の回帰防止。
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from suzume_dsa import TINY, SuzumeGlmDsa  # noqa: E402
from suzume_dsa.export_gguf import export  # noqa: E402
import gguf  # noqa: E402 （export 経由で gguf-py が sys.path に入る）


def _reader():
    model = SuzumeGlmDsa(TINY).eval()
    path = Path(tempfile.mkdtemp()) / "tiny.gguf"
    export(model, str(path))
    return gguf.GGUFReader(str(path)), TINY


def test_architecture_is_glm_dsa():
    r, _ = _reader()
    arch = r.get_field("general.architecture")
    assert arch.parts[arch.data[0]].tobytes().decode() == "glm-dsa"


def test_required_tensors_and_shapes():
    r, cfg = _reader()
    shapes = {t.name: list(t.shape) for t in r.tensors}
    D, H = cfg.n_embd, cfg.n_head
    hk, hv, nope, rope = cfg.head_dim_k, cfg.head_dim_v, cfg.head_dim_nope, cfg.head_dim_rope
    Lq, Lkv = cfg.q_lora_rank, cfg.kv_lora_rank

    # gguf の shape は llama.cpp の ne 順（= docs/glm-dsa-tensors.md の {..} と同順）
    assert shapes["token_embd.weight"] == [D, cfg.vocab_size]
    assert "output.weight" not in shapes            # tie
    assert shapes["blk.1.attn_q_a.weight"] == [D, Lq]
    assert shapes["blk.1.attn_q_b.weight"] == [Lq, H * hk]
    assert shapes["blk.1.attn_kv_a_mqa.weight"] == [D, Lkv + rope]
    assert shapes["blk.1.attn_k_b.weight"] == [nope, Lkv, H]
    assert shapes["blk.1.attn_v_b.weight"] == [Lkv, hv, H]
    assert shapes["blk.1.attn_output.weight"] == [H * hv, D]
    assert shapes["blk.1.ffn_gate_exps.weight"] == [D, cfg.n_ff_exp, cfg.n_expert]
    assert shapes["blk.1.ffn_down_exps.weight"] == [cfg.n_ff_exp, D, cfg.n_expert]
    assert shapes["blk.1.ffn_gate_shexp.weight"] == [D, cfg.n_ff_exp * cfg.n_expert_shared]
    # 先頭層は密 FFN
    assert shapes["blk.0.ffn_gate.weight"] == [D, cfg.n_ff]


if __name__ == "__main__":
    test_architecture_is_glm_dsa()
    test_required_tensors_and_shapes()
    print("all export tests passed")
