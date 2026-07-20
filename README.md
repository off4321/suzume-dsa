# suzume-dsa

GGUF/llama.cpp ネイティブを**最初から前提**にした小型 LLM の学習リポジトリ。
最終モデル名は **`suzume`**。

## 方針転換の背景

前身 `suzume-muon` は独自層（moa / delta / MoD / scope / MLA 重み吸収 など）を
多数発明したが、これらは **llama.cpp のグラフに存在しないため GGUF に載らない**。
本リポジトリは逆に「llama.cpp が対応済みのアーキテクチャを**そのまま**再現し、
可変ハイパラ（寸法・カウント）だけを探索する」方針に振り切る。

- **できないこと**: 新しい層の追加、結線/norm位置/活性化/アテンション方式/rope種別の変更。
- **できること**: `n_layer` / `n_embd` / `n_head` / `n_expert(_used/_shared)` / `ff_exp` /
  MLA の `q_lora`・`kv_lora` / DSA indexer の `head/key_length/top_k` / `n_layer_nextn` などの数値。

詳細は [docs/gguf-scope.md](docs/gguf-scope.md) を参照。

## ターゲットアーキ

第一目標は **glm-dsa**（llama.cpp `GLM_DSA` / HF `GlmMoeDsaForCausalLM`）。
= MLA + DSA(indexer) + MoE(sigmoid gating) + NextN の背骨のみで完結する最小構成。

- **deepseek4**（`DEEPSEEK4` / `DeepseekV4ForCausalLM`）は glm-dsa の上位互換
  （hyper-connection + hash層 + o_group/o_lora を追加）。glm-dsa の GGUF 経路が
  通ってから追加分を載せる。
- `inkling` はこの llama.cpp チェックアウトに**存在しない**ため対象外。

## パラメータ目標

**total ~4B / active ~1.5B**（MoEなので total と active を独立に設計できる）。

| D | L | H | n_expert(used) | ff_exp | kv_lora | total | active |
|---|---|---|---|---|---|---|---|
| 2048 | 24 | 16 | 22〜24 (6) | 1024 | 512 | ~4.0B | ~1.5B |

`total` はほぼ `3·D·ff_exp·n_expert·L`（MoE項）で決まる。見積りは
`python tools/params.py` で再計算できる。

## 構成

```
src/suzume_dsa/
  config.py            GlmDsaConfig（可変レバーのみ）+ SUZUME_4B / TINY
  model.py             SuzumeGlmDsa（v1 = 素MLA + MoE）
  modules/
    mla.py             Multi-head Latent Attention（分割 wk_b/wv_b）
    moe.py             Sigmoid-gated MoE + 共有Expert + aux-free bias
    ffn.py             密 SwiGLU（先頭 dense 層）
    block.py           Pre-norm 残差ブロック
    mtp.py             Multi-Token Prediction（train-only 補助）
    norm.py / rope.py  RMSNorm / RoPE（suzume-muon から流用）
tests/test_shapes.py   形状・疎通・テンソル仕様一致テスト
tools/params.py        パラメータ概算
docs/                  gguf-scope / glm-dsa-tensors / training-efficiency
```

## 現状

- [x] GGUF スコープ調査・可変レバー整理（docs/gguf-scope.md）
- [x] glm-dsa 要求テンソル仕様（docs/glm-dsa-tensors.md）
- [x] 学習効率化の移植方針（docs/training-efficiency.md）
- [x] **v1 モデル定義（素MLA + MoE、SUZUME_4B = total 4.03B / active 1.43B 実測）**
- [x] 形状・疎通テスト（tests/test_shapes.py, 4件パス）
- [ ] 学習ループ移植（data / pipeline / sft を suzume-muon から。系列長カリキュラム等の効率化込み）
- [ ] GGUF エクスポート（HF形式 → conversion/glm.py）
- [ ] v2: DSA indexer / v3: NextN
