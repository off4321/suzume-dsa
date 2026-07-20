"""
suzume-dsa: モデル設定

llama.cpp の `GLM_DSA` アーキが要求するテンソル構成に 1:1 で対応する
ハイパラだけを持つ dataclass。ここにある値はすべて「可変レバー」であり、
層の種類・結線は固定（docs/gguf-scope.md 参照）。

v1 は素の MLA + MoE のみ（DSA indexer と NextN/MTP export は無し）。
学習効率のための MTP は train-only 補助損失として別に持つ（export では捨てる）。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GlmDsaConfig:
    # --- 語彙・埋め込み ---
    vocab_size: int = 32000
    n_embd: int = 2048            # D: 隠れ次元
    tie_embeddings: bool = True   # output.weight を token_embd と共有（4B構成で埋め込み節約）

    # --- 層 ---
    n_layer: int = 24             # L: 本体層数
    n_layer_dense_lead: int = 1   # 先頭の密 FFN 層数（以降は MoE）

    # --- MLA アテンション ---
    n_head: int = 16              # H
    head_dim_nope: int = 128      # RoPE を掛けない部分の 1ヘッド次元
    head_dim_rope: int = 64       # RoPE を掛ける部分の 1ヘッド次元
    head_dim_v: int = 128         # value の 1ヘッド次元
    q_lora_rank: int = 1152       # Q 側の低ランク圧縮次元
    kv_lora_rank: int = 512       # K/V 共有の潜在次元 c_kv
    rope_base: float = 10000.0
    n_ctx_train: int = 4096       # 学習系列長（GGUF メタデータ用）

    # --- DSA indexer（v1 では重みを持たず、メタデータのみ書く。llama.cpp が
    #     indexer テンソル不在を検知して素の MLA にフォールバックする）---
    indexer_n_head: int = 8
    indexer_head_size: int = 64
    indexer_top_k: int = 2048

    # --- FFN / MoE ---
    n_ff: int = 4096              # 密 FFN の中間次元（先頭 dense 層用）
    n_ff_exp: int = 1024          # Expert 1個あたりの中間次元
    n_expert: int = 24            # E: ルーティング Expert 総数
    n_expert_used: int = 6        # top-k: 1トークンが使う Expert 数
    n_expert_shared: int = 1      # 常時通す共有 Expert 数
    expert_weights_scale: float = 1.0
    expert_weights_norm: bool = True   # top-k の重みを和で正規化するか
    router_bias_update_rate: float = 1e-3  # aux-free ロードバランスの bias 更新幅

    # --- 学習効率（train-only、export非対象）---
    mtp_depth: int = 0            # >0 で Multi-Token Prediction を有効化
    mtp_loss_coef: float = 0.3

    # --- 正規化 ---
    norm_eps: float = 1e-6

    # ---- 派生プロパティ ----
    @property
    def head_dim_k(self) -> int:
        """Q/K の 1ヘッド次元（nope + rope）。"""
        return self.head_dim_nope + self.head_dim_rope

    def __post_init__(self) -> None:
        assert self.head_dim_rope % 2 == 0, "head_dim_rope は RoPE のため偶数"
        assert self.n_expert_used <= self.n_expert, "used は expert 総数以下"
        assert self.n_layer_dense_lead < self.n_layer, "dense 先頭層は総層数未満"


# total ~4.1B / active ~1.5B を狙う既定構成（tools/params.py と一致）
SUZUME_4B = GlmDsaConfig()

# ユニットテスト・疎通用の極小構成
TINY = GlmDsaConfig(
    vocab_size=256, n_embd=64, n_layer=3, n_layer_dense_lead=1,
    n_head=4, head_dim_nope=16, head_dim_rope=8, head_dim_v=16,
    q_lora_rank=48, kv_lora_rank=32, n_ff=128, n_ff_exp=32,
    n_expert=8, n_expert_used=2, n_expert_shared=1,
)

__all__ = ["GlmDsaConfig", "SUZUME_4B", "TINY"]
