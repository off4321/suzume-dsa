# glm-dsa 要求テンソル仕様

llama.cpp `src/models/glm-dsa.cpp` の `load_arch_tensors` から抽出。
**学習モデルはこの名前・形状のテンソルを出力できないと GGUF が読み込めない**（省略不可、
ただし `NOT_REQUIRED` 印のものは任意）。HF エクスポートは `GlmMoeDsaForCausalLM` として
`conversion/glm.py` に食わせる。

記号: `D`=n_embd, `H`=n_head, `V`=n_vocab, `Lr_q`=q_lora_rank, `Lr_kv`=kv_lora_rank,
`rope`=n_rot, `hk`=n_embd_head_k_mla, `hv`=n_embd_head_v_mla, `nope`=hk-rope,
`ff`=n_ff(dense), `ff_e`=n_ff_exp, `E`=n_expert, `Es`=n_expert_shared,
`ih`=indexer_head_size, `inh`=indexer_n_head。

## グローバル
| tensor | 形状 | 備考 |
|---|---|---|
| `token_embd.weight` | {D, V} | |
| `output_norm.weight` | {D} | |
| `output.weight` | {D, V} | 省略可 → token_embd を流用（tied）|

## 各層（`i < n_layer` = 本体層）
### Attention（MLA、必須。**wk_b/wv_b は分割形式**）
| tensor | 形状 |
|---|---|
| `attn_norm.weight` | {D} |
| `attn_q_a_norm.weight` | {Lr_q} |
| `attn_kv_a_norm.weight` | {Lr_kv} |
| `attn_q_a.weight` (wq_a) | {D, Lr_q} |
| `attn_q_b.weight` (wq_b) | {Lr_q, H·hk} |
| `attn_kv_a_mqa.weight` | {D, Lr_kv + rope} |
| `attn_k_b.weight` (wk_b) | {nope, Lr_kv, H} |
| `attn_v_b.weight` (wv_b) | {Lr_kv, hv, H} |
| `attn_output.weight` (wo) | {H·hv, D} |
| `ffn_norm.weight` | {D} |

### DSA indexer（**すべて NOT_REQUIRED = v1 では省略して素の MLA で動かせる**）
| tensor | 形状 |
|---|---|
| `indexer_k_norm.weight` / `.bias` | {ih} |
| `indexer_proj.weight` | {D, inh} |
| `indexer_attn_k.weight` | {D, ih} |
| `indexer_attn_q_b.weight` | {Lr_q, inh·ih} |

### FFN
- `i < n_layer_dense_lead`（先頭密層）: `ffn_gate/down/up.weight` = {D,ff}/{ff,D}/{D,ff}
- それ以外（MoE層）:
  | tensor | 形状 |
  |---|---|
  | `ffn_gate_inp.weight` | {D, E} |
  | `ffn_exp_probs_b.bias` | {E}（NOT_REQUIRED, aux-free bias）|
  | `ffn_gate_exps.weight` | {D, ff_e, E} |
  | `ffn_down_exps.weight` | {ff_e, D, E} |
  | `ffn_up_exps.weight` | {D, ff_e, E} |
  | `ffn_gate_shexp.weight` | {D, ff_e·Es} |
  | `ffn_down_shexp.weight` | {ff_e·Es, D} |
  | `ffn_up_shexp.weight` | {D, ff_e·Es} |

  gating: sigmoid（GLM-4.5系）。`expert_weights_scale`/`expert_weights_norm` あり。

## NextN/MTP 層（`i >= n_layer`、任意）
`nextn.eh_proj`{2D,D} / `nextn.enorm`{D} / `nextn.hnorm`{D} と、
任意で `nextn.embed_tokens`/`shared_head_head`/`shared_head_norm`。
→ **v1 では作らない**（`n_layer_nextn=0`）。

## 実装マイルストーン（de-risking）
1. **v1 = 素のMLA + MoE のみ**（indexer なし・NextN なし）。indexer が NOT_REQUIRED なので
   これで llama.cpp は読める。train → HF export → GGUF → 推論の全経路を先に通す。
2. **v2 = DSA indexer を追加**（本来の "DSA"）。
3. **v3 = NextN/MTP**（任意、学習効率）。

## 主要ハイパラ（GGUFメタデータ, glm-dsa.cpp loader より）
`n_ff_exp`, `f_norm_rms_eps`, `rope_sections[4]`, `n_expert`/`n_expert_used`/`n_expert_shared`,
`n_layer_dense_lead`, `expert_weights_scale`/`_norm`, `n_lora_q`, `n_lora_kv`,
`n_embd_head_k/v_mla`, indexer(`n_head`/`key_length`/`top_k`), `expert_gating_func`(=sigmoid),
`n_layer_nextn`。
