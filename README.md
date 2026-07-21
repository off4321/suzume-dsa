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
`uv run tools/count_params.py`（実モデルを meta 構築）で再計算できる。

## セットアップ（uv）

suzume-muon と同じく uv で動く。CPU/CUDA は torch のグループで切替。

```bash
uv sync                       # 開発コンテナ（CPU 版 torch）
# 本番 GPU: uv run --no-group cpu --group cu124 <cmd>
uv run suzume-dsa-train --corpus mytext.txt --steps 200 \
    --block-size-schedule "0%:1024,50%:2048,80%:4096"
uv run python tests/test_train.py           # スモークテスト
```

## 本番パイプライン（A: glm-dsa 密MLA+MoE）

GPU では各コマンドに `--no-group cpu --group cu124` を付け、`--device cuda` を渡す。

```bash
# 0. セットアップ（GPU 版 torch）
uv sync --no-group cpu --group cu124

# 1. SentencePiece トークナイザ学習（日本語コーパス等から。vocab 32k）
uv run tools/train_tokenizer.py \
    --hf-dataset "range3/wikipedia-ja-20230101" --hf-max-samples 1000000 \
    --vocab-size 32000 --out tokenizer/sp
#   → tokenizer/sp.model

# 2. 事前学習（SUZUME_4B = total 4B/active 1.4B、系列長カリキュラム + Muon + WSD）
uv run --no-group cpu --group cu124 suzume-dsa-train \
    --sp-model tokenizer/sp.model \
    --hf-dataset "range3/wikipedia-ja-20230101" \
    --steps 18000 --batch-size 16 \
    --block-size-schedule "0%:1024,50%:2048,80%:4096" \
    --optimizer muon --muon-lr 0.02 --lr 1.5e-4 \
    --lr-schedule wsd --warmup 400 \
    --out output --device cuda
#   → output/model.pt（途中も checkpoint_step*.pt、--resume で再開可）
#   FIM を混ぜるなら --fim 0.3、選択的backpropなら --select-topp 0.7

# 3. SFT（事前学習 checkpoint から継続、assistant のみ教師化）
uv run --no-group cpu --group cu124 suzume-dsa-sft \
    --sp-model tokenizer/sp.model --init-from output/model.pt \
    --hf-sft-dataset "llm-jp/magpie-sft-v1.0" \
    --steps 2000 --batch-size 8 --max-len 2048 --lr 1e-5 \
    --out output_sft --device cuda
#   → output_sft/sft_model.pt

# 4. GGUF エクスポート（実 SentencePiece 語彙を埋め込む）
uv run suzume-dsa-export \
    --checkpoint output_sft/sft_model.pt --sp-model tokenizer/sp.model \
    --out suzume.gguf

# 5. 量子化して実行（llama.cpp 側）
/workspace/llama.cpp/build/bin/llama-quantize suzume.gguf suzume-Q4_K_M.gguf Q4_K_M
/workspace/llama.cpp/build/bin/llama-cli -m suzume-Q4_K_M.gguf -p "すずめとは" -n 128
```

学習前のサイズ・VRAM 確認: `uv run tools/count_params.py` /
`uv run --no-group cpu --group cu124 tools/vram_calibrate.py --block-size 4096 --device cuda`。

## 本番パイプライン（B: 0.5B トライ一気通し）

いきなり 4B は 30〜60B トークン必要で単一 GPU では月単位。まず **0.5B（`SUZUME_05B`
= total 460M / active 201M）でパイプライン全体を安く検証**するのを推奨。同じ glm-dsa
アーキで寸法だけ縮小したもの。Chinchilla×20 ≈ 4B トークンなので手持ちコーパスでも回せる。
本番データの学習は Google Colab（RTX PRO 6000 Blackwell 96GB）で実施する前提。

`train.py` は `--preset {4b,05b,tiny}` と個別レバー（`--n-embd`/`--n-layer`/`--n-expert`
…= `GlmDsaConfig` 全フィールド）で寸法を指定できる。SFT/export は `--init-from` /
`--checkpoint` に保存された cfg を自動継承するので、寸法指定は事前学習の 1 回だけでよい。

```bash
# 0. GPU 版 torch
uv sync --no-group cpu --group cu124

# 1. トークナイザ（4B と共通。既にあれば流用）
uv run tools/train_tokenizer.py \
    --hf-dataset "range3/wikipedia-ja-20230101" --hf-max-samples 1000000 \
    --vocab-size 32000 --out tokenizer/sp

# 1.5 サイズと VRAM を事前確認（0.5B 構成で）
uv run tools/count_params.py --n-embd 1024 --n-layer 12 --n-head 8 \
    --q-lora-rank 768 --kv-lora-rank 384 --n-ff 2816 --n-ff-exp 640 \
    --n-expert 16 --n-expert-used 4        # → total 460M / active 201M
uv run --no-group cpu --group cu124 tools/vram_calibrate.py \
    --n-embd 1024 --n-layer 12 --block-size 4096 --device cuda   # 収まる最大 batch

# 2. 事前学習（--preset 05b、系列長 + バッチ カリキュラムで VRAM をほぼ一定に）
#    block を伸ばす step で batch を下げる（vram_calibrate の最大 batch を先頭に）。
uv run --no-group cpu --group cu124 suzume-dsa-train \
    --preset 05b --sp-model tokenizer/sp.model \
    --hf-dataset "range3/wikipedia-ja-20230101" \
    --steps 20000 \
    --block-size-schedule "0%:1024,50%:2048,80%:4096" \
    --batch-size-schedule "0%:48,50%:24,80%:12" \
    --optimizer muon --muon-lr 0.02 --lr 2e-4 \
    --lr-schedule wsd --warmup 400 \
    --out output_05b --device cuda
#   → output_05b/model.pt（checkpoint_step*.pt も。切断時は --resume output_05b/checkpoint_step*.pt）
#   Muon で nan が出たら --resume + --muon-lr 0.01。FIM は --fim 0.3、選択backprop は --select-topp 0.7

# 3. SFT（cfg は checkpoint から継承＝0.5B のまま。寸法指定不要）
uv run --no-group cpu --group cu124 suzume-dsa-sft \
    --sp-model tokenizer/sp.model --init-from output_05b/model.pt \
    --hf-sft-dataset "llm-jp/magpie-sft-v1.0" \
    --steps 2000 --batch-size 8 --max-len 2048 --lr 1e-5 \
    --out output_05b_sft --device cuda

# 4. GGUF エクスポート（cfg は checkpoint から、語彙は SP から）
uv run suzume-dsa-export \
    --checkpoint output_05b_sft/sft_model.pt --sp-model tokenizer/sp.model \
    --out suzume-05b.gguf

# 5. 量子化して実行
/workspace/llama.cpp/build/bin/llama-quantize suzume-05b.gguf suzume-05b-Q4_K_M.gguf Q4_K_M
/workspace/llama.cpp/build/bin/llama-cli -m suzume-05b-Q4_K_M.gguf -p "すずめとは" -n 128
```

4B へ上げるときは 2 で `--preset 4b` にし、カリキュラムの batch を下げるだけ（3〜5 は不変）。

## tools/

```bash
uv run tools/count_params.py --n-expert 24          # total/active + 学習メモリ見積り
uv run tools/vram_calibrate.py --block-size 4096    # 実測して収まる最大 batch を逆算(要CUDA)
uv run tools/check_dataset.py "range3/wikipedia-ja-20230101" --split "train[:5]"  # HFデータ下見
```

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
- [x] **GGUF エクスポート（gguf-py 直書き `export_gguf.py`）+ llama.cpp 実ロード疎通確認**
      （arch=glm-dsa 認識・全メタデータ一致・生成到達）+ 契約テスト（tests/test_export.py）
- [x] **(a) 最小 pretrain ループ**（`data.py` + `train.py`）: 次トークンCE + MTP補助 +
      aux-free bias commit + 非有限勾配ガード + checkpoint/resume
- [x] **系列長カリキュラム `--block-size-schedule`**（`"0%:1024,50%:2048,80%:4096"`、1-run）
- [x] **uv 対応**（CPU/CUDA torch グループ切替、`suzume-dsa-train` エントリポイント）
- [x] **tools/**（count_params / vram_calibrate / check_dataset を移植）
- [x] **HF datasets 対応**（`data.py` の preview_hf_dataset / load_hf_tokens）
- [x] **本番 SentencePiece トークナイザ**（`tokenizer.py`、train_spm + 実語彙の GGUF 書き出し）
- [x] **ループ側効率化フル**: Muon（`optim.py`）/ WSD・cosine スケジュール / μP `--mup` /
      FIM `--fim`（`fim.py`）/ 選択的 backprop `--select-topp` / 深さ成長 `--init-from`
- [x] **SFT ステージ**（`chat.py` + `sft.py`、assistant のみ教師化 + マスク損失）
- [ ] **v2: DSA indexer** — 要 llama.cpp パリティ検証（下記）
- [~] **v3: NextN** — llama.cpp では "preserved but unused"（推論で使われない）ため
      **export しない**のが正解。学習効率は MTP（train-only）で既に取得済み

### v2 / v3 の位置づけ（重要）

- **v3 NextN は export しない**: llama.cpp の glm-dsa は NextN テンソルを読み込むが
  推論では使わない。MTP の学習効率メリットは train-only 補助損失で既に得ているので、
  NextN を書いても無意味（`n_layer_nextn=0` のまま）。
- **v2 DSA indexer は専用の検証が必要**: llama.cpp の `build_lid_top_k` は RoPE 付き
  indexer ヘッド + compressor + top-k 選択という具体的アルゴリズム。学習側と llama.cpp
  側で**同一でないと export したモデルが壊れる**（v1 は indexer 無し＝正しい dense
  フォールバックで動作確認済み）。indexer はパリティ検証込みの独立タスクとして実装する。

## 最小 pretrain

```bash
cd src && python -m suzume_dsa.train --corpus mytext.txt --steps 200
# TINY(vocab=256, バイト単位) で回る。本番は cfg=SUZUME_4B + SentencePiece へ差し替え。
```

現状ループに配線済みの効率化: **MTP補助損失**・**aux-free MoEバランス**・**非有限勾配ガード**
（suzume-muon の nan 事件の教訓）。系列長カリキュラム等のループ側効率化は次段で追加。

## GGUF エクスポート

```bash
cd src && python -m suzume_dsa.export_gguf out.gguf   # TINY を書き出し
# llama.cpp で確認:
/workspace/llama.cpp/build/bin/llama-cli -m out.gguf -p "..." -n 8
```

HF 形式を経由せず、`export_gguf.py` が llama.cpp のテンソル名・メタデータを直接書く。
MLA は MQA 化して格納（n_head_kv=1）。3D 因子 wk_b/wv_b のみ ne 軸順に合わせて転置。
v1 は indexer/NextN の重みを持たず、indexer はメタデータのみ（llama.cpp がフォールバック）。
