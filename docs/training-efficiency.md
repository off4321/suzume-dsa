# 学習効率化機能の移植方針（suzume-muon → suzume-dsa）

判定基準はただ一つ: **エクスポートされる tensor グラフを変えるか否か**。
変えないもの（学習ループ・損失・データ・最適化のみに作用）は**丸ごと移植可**。
グラフを変えるもの／推論時の工夫は、GGUF では **llama.cpp が担うので対象外**。

## ✅ そのまま移植（学習時のみ・エクスポート結果に無影響）

| 機能 | フラグ | 効果 | 備考 |
|---|---|---|---|
| 系列長カリキュラム | `--block-size-schedule` | **CPU実測 seq1/2で3.5x, 1/4で8.6x/step** | 最大の低リスク時短。1-run完結、resume整合 |
| 深さ成長(progressive stacking) | `--init-from` | 壁時計1.5-2x | 新モデルが層数不一致ロードに対応必要 |
| MTP(多token予測) | `--mtp-depth` `--mtp-loss-coef` | 学習信号の増強 | **train-only 補助損失として移植・export時は破棄**（下記注） |
| μP(幅超えハイパラ転移) | `--mup` | 探索コスト削減 | |
| WSD 学習率 | `--lr-schedule wsd` | | |
| Muon optimizer | `--optimizer muon` | | nan対策(非有限勾配スキップ)も一緒に移植 |
| FIM(穴埋め目的) | `--fim` | 信号軸追加 | データ側 |
| 動的データ選別 | `--select-topp` `--rho-lite` | 無駄トークン削減 | データ/損失側 |
| dedup | (data) | | |
| 評価/再開インフラ | `--eval-interval` `--eval-max-batches` `--resume` + ローテーションckpt | サイレント切断対策 | 長時間run必須 |

これらは**新モデル定義とは独立**なので、`pipeline.py` / `settings.py` / `data.py` を流用すれば
基本そのまま効く。組合せも維持（例: MTP × 系列長カリキュラムは互換）。

### MTP の注意
glm-dsa には native な NextN/MTP tensor 枠があるが、llama.cpp のコメントは
"preserved but **unused**"（推論では使わない）。よって MTP を NextN として export しても
推論利得は無い。**MTP は純粋に学習効率のための train-only 補助損失として使い、export では捨てる**
のが正解（`n_layer_nextn=0` のまま）。MTPModule は新 glm-dsa ブロックを参照する形に要改修。
小型では利得が中立〜小幅の文献報告あり → **アブレーションで採否判断**。

## ❌ 移植しない（グラフを変える／arch非互換）

| 機能 | 理由 |
|---|---|
| Mixture-of-Depths `--mod-capacity` | per-token で FFN をスキップ＝**グラフを変える**。glm-dsa に無く export 不可 |
| BitNet `--bitnet` | 三値 BitLinear＝別 arch(`BITNET`)。glm-dsa と非互換。やるなら別プロジェクト |

## ⛔ 対象外（推論・キャッシュ最適化 = llama.cpp が担当）

suzume-muon の以下は**自前推論エンジンの最適化**で、GGUF では llama.cpp が同等機能を持つため
移植不要（捨てても損失なし。これが GGUF ネイティブ化の狙いそのもの）:

- KVキャッシュ Phase2 重み吸収 / CLA(層間KV共有) / prefix キャッシュ
- kv_quant(int8) / PQ量子化 / GrowBuffer → **量子化・キャッシュは llama.cpp / ollama 側**
- expert flash streaming / Mixture-of-Scopes / MoA

## 結論
「学習効率を上げる」中核（**系列長カリキュラム・深さ成長・MTP・μP・WSD・Muon・FIM・動的データ選別**）は
**もれなく移植できる**。失うのは推論時最適化だけで、それは llama.cpp が肩代わりする。
