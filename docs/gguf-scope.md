# GGUF スコープと可変レバー整理

対象 llama.cpp: このworkspaceのチェックアウト（`178a6c449` 時点）で確認。
「新しい層は追加できない」制約下で、**何が固定で・何を動かせるか**を実装本体から抽出した。

## 1. 検討アーキの対応状況

| 検討名 | llama.cpp arch | HF `architectures` | HF→GGUF 変換 | 状態 |
|---|---|---|---|---|
| qwen35 | `QWEN35` | `Qwen3_5ForCausalLM` | conversion/qwen.py | ✅ |
| qwen35_moe | `QWEN35MOE` | `Qwen3_5MoeForCausalLM` | conversion/qwen.py | ✅ |
| deepseek4 | `DEEPSEEK4` | `DeepseekV4ForCausalLM` | conversion/deepseek.py | ✅ |
| glm-dsa | `GLM_DSA` | `GlmMoeDsaForCausalLM` | conversion/glm.py | ✅ |
| ht_v3 | `HY_V3` | `HYV3ForCausalLM` | conversion/hunyuan.py | ✅ |
| **inkling** | — | — | — | ❌ enum・C++・変換すべて無し |

## 2. 大原則

llama.cpp の `build_graph` とテンソル要求は C++ で固定。
動かせるのは **loader が `get_key` で読む数値ハイパラだけ**。省略も不可
（archを名乗る以上、要求テンソルは全部そろえる必要がある）。

### ❌ 変えられない
- 層の種類・順序・結線、norm の位置、活性化関数
- アテンション機構そのもの（MLA/GQA、DSA indexer の有無、線形アテンションか）
- rope 種別: deepseek4/glm-dsa = NORM、ht_v3 = NEOX、qwen35 = IMROPE
- MoE の gating 方式、shared expert / NextN / hyper-connection 等の**存在有無**

### ✅ 変えられる（archごとの可変レバー）

| arch | 主な可変レバー |
|---|---|
| **deepseek4** | `n_layer`, `n_embd`, `n_head`, `n_lora_q`, `kv_lora_rank`, `n_ff_exp`, `n_expert`/`n_expert_used`/`n_expert_shared`, `expert_weights_scale/norm`, indexer(`n_head`/`key_length`/`top_k`), `dsv4_o_group_count`/`o_lora_rank`, `dsv4_hc_mult`/`sinkhorn_iters`, `dsv4_hash_layer_count`, `n_swa`, `n_layer_nextn` |
| **glm-dsa** | `n_layer`(+`n_layer_dense_lead`), `n_ff_exp`, `n_expert`/`used`/`shared`, `n_lora_q`/`kv_lora_rank`, MLA head長, indexer(`n_head`/`key_length`/`top_k`), `expert_weights_scale/norm`, gating func, `n_layer_nextn` |
| **qwen35(_moe)** | `n_layer`, `n_embd`, `n_head`/`n_head_kv`, SSM系(`ssm_d_conv`/`d_inner`/`d_state`/`dt_rank`/`n_group`), `full_attention_interval`, recurrent層割当, `n_layer_nextn`, (moe: `n_expert`系) |
| **ht_v3** | Hunyuan V3系（未精読、MoE/dense 寸法系が中心）|

## 3. 実装難易度（重要）

選ばれた候補は llama.cpp で**最重量級**ばかり。「小さくする」＝寸法が小さくなるだけで、
**実装量はフルスペック必要**。

- **qwen35 / qwen35_moe**: gated-delta-net の**線形アテンション（recurrent層）**+
  周期フルアテンション + IMROPE + NextN。`llama-memory-recurrent` を使う別系統。
- **deepseek4**（1199行）: MLA + DSA(indexer) + hyper-connection(Sinkhorn) + hash層 +
  MoE+shared expert + per-layer swiglu clamp + NextN。
- **glm-dsa**（152行）: MLA必須 + DSA(indexer) + MoE(sigmoid) + NextN。
  → **deepseek4 の背骨に相当し、追加難物がない最小構成。第一目標に推奨。**
- **ht_v3**: NEOX rope 系の Hunyuan V3。線形アテン/DSA なし、相対的に素直な可能性。

## 4. deepseek4 と glm-dsa の関係

パラメータ数はほぼ同一（同じ MLA+MoE 背骨が支配）。差は実装再現の重さ：

```
deepseek4 = glm-dsa の背骨(MLA + DSA indexer + MoE + NextN)
          + hyper-connection(Sinkhorn) + hash層 + o_group/o_lora + swiglu clamp
```

→ glm-dsa で GGUF エクスポート経路を先に通し、その上に deepseek4 の追加分を載せるのが低リスク。

## 5. パラメータ設計

`total ~= 3·D·ff_exp·n_expert·L`（MoE項が支配）。
active は `n_expert_used` で決まり、`n_expert` を増やしても不変。
→ **total を 4B に合わせる操作 = 主に `n_expert` の調整**。詳細は `tools/params.py`。
