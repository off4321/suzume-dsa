# v2「DSA indexer」パリティ調査の結論（2026-07-21）

## 結論: glm-dsa の DSA indexer は llama.cpp で**実行されない**

パリティ検証のため llama.cpp のグラフ実装を精査した結果、**glm-dsa 用に indexer を
実装する意味が無い**ことが判明した。実装せず、glm-dsa は密 MLA + MoE のまま進めるのが正解。

### 根拠（このチェックアウトの llama.cpp）

1. `src/models/models.h:1220`
   ```cpp
   struct llama_model_glm_dsa : public llama_model_base {
       using graph = llama_model_deepseek2::graph;   // ← DeepSeek-V2 のグラフを再利用
   };
   ```
   glm-dsa の forward グラフは **DeepSeek-V2 のもの**。

2. `src/models/deepseek2.cpp` には `indexer` / `top_k` / `sparse` / `lid_` の参照が**ゼロ**
   （唯一の arch 分岐は `DEEPSEEK2OCR` のみ）。DeepSeek-V2 は DSA を持たない密 MLA。

3. よって glm-dsa の `load_arch_tensors` が indexer テンソルを `TENSOR_NOT_REQUIRED` で
   受け付けても、**グラフはそれを一切使わない**（NextN と同じ「読むが使わない」状態）。

→ glm-dsa に indexer を学習・export しても、llama.cpp は**密 MLA で推論する**。
   むしろ「疎で学習・密で推論」の train/inference ミスマッチになり有害。

### 実際に DSA indexer を走らせる唯一の arch = deepseek4

`src/models/deepseek4.cpp` だけが `build_lid_top_k`(RoPE付き indexer ヘッド + compressor
+ top-k 選択) を実行する。ただし deepseek4 をロードするには**全層で以下が必須**（flag=0）:

- hyper-connection: `hc_attn_fn/base/scale`, `hc_ffn_base/scale`（+ Sinkhorn 反復）
- o_group / o_lora（出力射影の低ランク分割）
- 圧縮: `attn_comp_wkv/wgate/ape/norm`（`compress_ratio` 依存）
- indexer: `compress_ratio==4` の層のみ
- hash 層: 先頭 `hash_layer_count` 層は `ffn_gate_tid2eid`

= **frontier フルスタックの再現とパリティ検証**が必要。glm-dsa より桁違いに重い。

## 選択肢

| 方針 | 内容 | コスト |
|---|---|---|
| **A. glm-dsa=密MLA+MoE で確定（推奨）** | 現状のまま。indexer は作らない（llama.cpp が使わない）。実データ学習へ進む | 追加ゼロ・検証済み |
| B. deepseek4 へ retarget | 本物の推論時 DSA。ただし hyper-connection/hash/compression/o_group/indexer を全部パリティ実装 | 非常に大 |

**推奨は A**。glm-dsa の密 MLA + aux-free MoE は完成・実ロード検証済みで、それ自体が
妥当な小型 MoE モデル。推論時 DSA が本当に要るときだけ B（deepseek4）を別プロジェクトとして検討。
