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

## 現状

- [x] GGUF スコープ調査・可変レバー整理（docs/gguf-scope.md）
- [x] パラメータ概算ツール（tools/params.py）
- [ ] 学習基盤の移植方針決定（suzume-muon 流用 vs 新規）
- [ ] glm-dsa モデル定義（llama.cpp テンソル集合に一致させる）
- [ ] GGUF エクスポート経路の疎通（HF形式 → conversion/glm.py）
