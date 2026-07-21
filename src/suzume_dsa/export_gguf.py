"""
suzume-dsa: GGUF エクスポート（llama.cpp `glm-dsa` 直書き）

HF 形式を経由せず、gguf-py で llama.cpp のテンソル名・メタデータを直接書き出す。
trainer 側の重みは手元にあるので、変換モジュール（conversion/glm.py）に合わせるより
直接書くほうが素直。

重要な約束:
  * GGUF の ne[] は numpy/torch の shape を **逆順** に格納する。
    torch Linear の weight (out, in) はそのまま書くと ne=[in, out] になり、
    llama.cpp の {in, out} 期待と一致する（2D は転置不要）。
  * 3D の MLA 低ランク因子 wk_b / wv_b だけは軸順が合わないので明示転置する。
  * MLA は MQA 化して格納するため n_head_kv=1、key_length=kv_lora+rope。

v1 は indexer / NextN テンソルを持たない（indexer はメタデータのみ、
llama.cpp 側 NOT_REQUIRED によりフォールバック）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from .config import GlmDsaConfig
from .model import SuzumeGlmDsa

# ローカルの gguf-py を優先的に使う（pip 版が無くても動くように）
_GGUF_PY = Path("/workspace/llama.cpp/gguf-py")
if _GGUF_PY.exists() and str(_GGUF_PY) not in sys.path:
    sys.path.insert(0, str(_GGUF_PY))
import gguf  # noqa: E402


def _np(t: torch.Tensor):
    return t.detach().to(torch.float32).contiguous().numpy()


def write_metadata(w: "gguf.GGUFWriter", cfg: GlmDsaConfig) -> None:
    w.add_context_length(cfg.n_ctx_train)
    w.add_embedding_length(cfg.n_embd)
    w.add_block_count(cfg.n_layer)
    w.add_feed_forward_length(cfg.n_ff)            # 密層の中間次元
    w.add_head_count(cfg.n_head)
    w.add_head_count_kv(1)                          # MLA→MQA
    w.add_layer_norm_rms_eps(cfg.norm_eps)
    w.add_rope_freq_base(cfg.rope_base)
    w.add_rope_dimension_count(cfg.head_dim_rope)
    w.add_vocab_size(cfg.vocab_size)

    # MLA
    w.add_q_lora_rank(cfg.q_lora_rank)
    w.add_kv_lora_rank(cfg.kv_lora_rank)
    w.add_key_length(cfg.kv_lora_rank + cfg.head_dim_rope)   # MQA 実ヘッド長
    w.add_value_length(cfg.kv_lora_rank)
    w.add_key_length_mla(cfg.head_dim_k)                     # nope + rope
    w.add_value_length_mla(cfg.head_dim_v)

    # MoE
    w.add_leading_dense_block_count(cfg.n_layer_dense_lead)
    w.add_expert_feed_forward_length(cfg.n_ff_exp)
    w.add_expert_count(cfg.n_expert)
    w.add_expert_used_count(cfg.n_expert_used)
    w.add_expert_shared_count(cfg.n_expert_shared)
    w.add_expert_weights_scale(cfg.expert_weights_scale)
    w.add_expert_weights_norm(cfg.expert_weights_norm)
    w.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)

    # DSA indexer（v1 は重み無し・メタデータのみ）
    w.add_indexer_head_count(cfg.indexer_n_head)
    w.add_indexer_key_length(cfg.indexer_head_size)
    w.add_indexer_top_k(cfg.indexer_top_k)


def write_tensors(w: "gguf.GGUFWriter", model: SuzumeGlmDsa) -> None:
    cfg = model.cfg
    w.add_tensor("token_embd.weight", _np(model.token_embd.weight))
    w.add_tensor("output_norm.weight", _np(model.output_norm.weight))
    if model.output is not None:                    # tie 時は書かない（token_embd 流用）
        w.add_tensor("output.weight", _np(model.output.weight))

    for i, blk in enumerate(model.blocks):
        p = f"blk.{i}."
        a = blk.attn
        w.add_tensor(p + "attn_norm.weight", _np(blk.attn_norm.weight))
        w.add_tensor(p + "attn_q_a.weight", _np(a.wq_a.weight))
        w.add_tensor(p + "attn_q_a_norm.weight", _np(a.q_a_norm.weight))
        w.add_tensor(p + "attn_q_b.weight", _np(a.wq_b.weight))
        w.add_tensor(p + "attn_kv_a_mqa.weight", _np(a.wkv_a_mqa.weight))
        w.add_tensor(p + "attn_kv_a_norm.weight", _np(a.kv_a_norm.weight))
        # 3D 因子は軸順を llama.cpp の ne に合わせて転置
        #   wk_b: (nope, Lr_kv, H) -> ne {nope, Lr_kv, H} には numpy (H, Lr_kv, nope)
        w.add_tensor(p + "attn_k_b.weight", _np(a.wk_b.permute(2, 1, 0)))
        #   wv_b: (Lr_kv, hv, H) -> ne {Lr_kv, hv, H} には numpy (H, hv, Lr_kv)
        w.add_tensor(p + "attn_v_b.weight", _np(a.wv_b.permute(2, 1, 0)))
        w.add_tensor(p + "attn_output.weight", _np(a.wo.weight))
        w.add_tensor(p + "ffn_norm.weight", _np(blk.ffn_norm.weight))

        ffn = blk.ffn
        if blk.is_dense:
            w.add_tensor(p + "ffn_gate.weight", _np(ffn.gate.weight))
            w.add_tensor(p + "ffn_up.weight", _np(ffn.up.weight))
            w.add_tensor(p + "ffn_down.weight", _np(ffn.down.weight))
        else:
            w.add_tensor(p + "ffn_gate_inp.weight", _np(ffn.gate_inp.weight))
            w.add_tensor(p + "exp_probs_b.bias", _np(ffn.exp_probs_b))
            # 3D expert 群は ne 逆順がそのまま llama.cpp の期待に一致（転置不要）
            w.add_tensor(p + "ffn_gate_exps.weight", _np(ffn.w_gate))
            w.add_tensor(p + "ffn_up_exps.weight", _np(ffn.w_up))
            w.add_tensor(p + "ffn_down_exps.weight", _np(ffn.w_down))
            w.add_tensor(p + "ffn_gate_shexp.weight", _np(ffn.shared.gate.weight))
            w.add_tensor(p + "ffn_up_shexp.weight", _np(ffn.shared.up.weight))
            w.add_tensor(p + "ffn_down_shexp.weight", _np(ffn.shared.down.weight))


def write_minimal_tokenizer(w: "gguf.GGUFWriter", cfg: GlmDsaConfig) -> None:
    """疎通確認用の最小 SPM トークナイザ（0=<unk>,1=<s>,2=</s>, 以降ダミー）。

    本番は SentencePiece 語彙（SPTokenizer）を渡す（write_sp_tokenizer）。
    """
    tokens, scores, toktypes = [], [], []
    T = gguf.TokenType
    for i in range(cfg.vocab_size):
        if i == 0:
            tokens.append("<unk>"); toktypes.append(T.UNKNOWN)
        elif i == 1:
            tokens.append("<s>"); toktypes.append(T.CONTROL)
        elif i == 2:
            tokens.append("</s>"); toktypes.append(T.CONTROL)
        else:
            tokens.append(f"<0x{i:02X}>"); toktypes.append(T.BYTE)
        scores.append(0.0)
    w.add_tokenizer_model("llama")
    w.add_token_list(tokens)
    w.add_token_scores(scores)
    w.add_token_types(toktypes)
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)
    w.add_unk_token_id(0)


def write_sp_tokenizer(w: "gguf.GGUFWriter", tokenizer) -> None:
    """SentencePiece(SPTokenizer) の実語彙を llama(SPM) 形式で書き出す。"""
    tokens, scores, types = tokenizer.gguf_vocab()
    w.add_tokenizer_model("llama")
    w.add_token_list(tokens)
    w.add_token_scores(scores)
    w.add_token_types(types)
    w.add_unk_token_id(1)
    w.add_bos_token_id(2)
    w.add_eos_token_id(3)


def export(model: SuzumeGlmDsa, path: str, with_tokenizer: bool = True,
           tokenizer=None, name: str | None = "suzume-dsa",
           chat_template: str | None = None,
           default_system: str | None = None) -> str:
    """GGUF を書き出す。name/chat_template を埋め込むとモデルが自己識別しやすくなる。

    chat_template 未指定でも default_system（既定の名乗りシステムプロンプト）を渡せば
    ChatML テンプレートを自動生成して埋め込む。どちらも None なら従来どおり書かない。
    """
    w = gguf.GGUFWriter(path, arch="glm-dsa")
    write_metadata(w, model.cfg)
    if name and hasattr(w, "add_name"):
        w.add_name(name)
    if chat_template is None and default_system:
        from .chat import build_chatml_template
        chat_template = build_chatml_template(default_system)
    if chat_template and hasattr(w, "add_chat_template"):
        w.add_chat_template(chat_template)
    if tokenizer is not None and hasattr(tokenizer, "gguf_vocab"):
        write_sp_tokenizer(w, tokenizer)          # 本番: 実 SentencePiece 語彙
    elif with_tokenizer:
        write_minimal_tokenizer(w, model.cfg)     # 疎通: バイト単位ダミー
    write_tensors(w, model)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="suzume-dsa → GGUF エクスポート")
    ap.add_argument("--checkpoint", help="学習済み checkpoint(.pt)。省略時は TINY を書き出し")
    ap.add_argument("--sp-model", help="SentencePiece .model（本番語彙を GGUF に埋め込む）")
    ap.add_argument("--out", default="suzume.gguf")
    ap.add_argument("--name", default="suzume-dsa", help="general.name（モデル名）")
    ap.add_argument("--default-system", default=None,
                    help="既定システムプロンプト（名乗り）。ChatML テンプレートに埋め込む")
    args = ap.parse_args()

    tok = None
    if args.checkpoint:
        import torch
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model = SuzumeGlmDsa(ckpt["cfg"])
        model.load_state_dict(ckpt["model"])
    else:
        from .config import TINY
        model = SuzumeGlmDsa(TINY)
    model.eval()

    if args.sp_model:
        from .tokenizer import SPTokenizer
        tok = SPTokenizer(args.sp_model)

    export(model, args.out, tokenizer=tok, name=args.name,
           default_system=args.default_system)
    print(f"wrote {args.out}  (vocab={'SP:'+args.sp_model if tok else 'byte-dummy'})")


if __name__ == "__main__":
    main()
