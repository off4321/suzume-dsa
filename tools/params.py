#!/usr/bin/env python3
"""DSA-MoE (glm-dsa / deepseek4 系) のパラメータ概算.

llama.cpp の create_tensor 形状に基づく要素数カウント。
total はほぼ MoE 項 3*D*ff_exp*n_expert*L で決まり、active は n_expert_used で決まる。
"""


def count(D, L, H, head_nope, head_rope, v_head,
          q_lora, kv_lora, ff_exp, n_expert, n_used, n_shared,
          idx_head, idx_hsize, V=32000, tie=False, dense_lead=1,
          o_lora=0):
    head_k = head_nope + head_rope
    emb = V * D * (1 if tie else 2)                        # 入力埋め込み + lm_head
    per = 0
    # --- attention (MLA) ---
    per += D * q_lora + q_lora + q_lora * (H * head_k)     # q_a, q_a_norm, q_b
    per += D * (kv_lora + head_rope) + kv_lora             # kv_a(+rope), kv_a_norm
    per += kv_lora * (H * (head_nope + v_head))            # kv_b
    if o_lora > 0:
        per += (H * v_head) * o_lora + o_lora * D          # o_proj low-rank
    else:
        per += (H * v_head) * D                            # o_proj full
    # --- DSA indexer (概算) ---
    per += D * idx_head + q_lora * (idx_head * idx_hsize) + D * (2 * idx_hsize) * 2 + idx_hsize
    per += 3 * D                                           # 各種 norm
    # --- MoE FFN ---
    moe = D * n_expert + 3 * D * ff_exp * n_expert + 3 * D * ff_exp * n_shared
    dense_ff = 3 * D * ff_exp * max(n_shared, 4)           # leading dense 層
    total = emb + (L - dense_lead) * (per + moe) + dense_lead * (per + dense_ff)
    act_moe = D * n_expert + 3 * D * ff_exp * n_used + 3 * D * ff_exp * n_shared
    active = emb + (L - dense_lead) * (per + act_moe) + dense_lead * (per + dense_ff)
    return total / 1e9, active / 1e9


# total ~4B / active ~1.5B を狙う既定構成。
SUZUME_4B = dict(D=2048, L=24, H=16, head_nope=128, head_rope=64, v_head=128,
                 q_lora=1152, kv_lora=512, ff_exp=1024, n_expert=24, n_used=6,
                 n_shared=1, idx_head=8, idx_hsize=64)


if __name__ == "__main__":
    t, a = count(**SUZUME_4B)
    print(f"suzume-4B  total {t:.2f}B  active {a:.2f}B  {SUZUME_4B}")
    print("\nn_expert 感度 (他固定):")
    for E in (16, 20, 22, 24, 26, 32):
        c = dict(SUZUME_4B); c["n_expert"] = E
        t, a = count(**c)
        print(f"  n_expert={E:>3}: total {t:.2f}B  active {a:.2f}B")
