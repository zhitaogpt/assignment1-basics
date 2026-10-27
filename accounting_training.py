"""CS336 A1 §4.3 adamw_accounting — 训练阶段的显存 / 算力账(纯 glue 脚本)。

用法:  cd assignment1-basics && .venv/bin/python accounting_training.py

复用 accounting.py 里的参数量公式 (§3.5) 与前向 FLOPs 公式,新增:
  (a)(b) 训练峰值显存 = a·batch_size + b
  (c)    AdamW 一步的 FLOPs(逐元素,≈ 常数·P)
  (d)    单卡 H100、50% MFU 下训练 400K 步的小时数
"""
from __future__ import annotations

from accounting import flops_breakdown, human, param_breakdown, round_to_multiple_of_64

BYTES_PER_FP32 = 4
GIB = 1024**3
GPU_MEM_GIB = 80.0

# ---- GPT-2 XL 配置(与 §3.5 一致) ----
VOCAB = 50_257
CTX = 1_024
D_MODEL = 1_600
NUM_LAYERS = 48
NUM_HEADS = 25
D_FF = round_to_multiple_of_64(8 / 3 * D_MODEL)  # 4288


# ------------------------------------------------------------ (a) 激活清单
def activation_elements_per_sample(T, d, d_ff, h, L, V):
    """handout 清单里的每个组件,B=1 时的元素数。"""
    per_layer = {
        "ln1 output": T * d,
        "ln2 output": T * d,
        "Q/K/V projections (x3)": 3 * T * d,
        "QK^T scores": h * T * T,          # ← T²
        "softmax": h * T * T,              # ← T²
        "attn weighted sum (A·V)": T * d,
        "output projection": T * d,
        "FFN W1 output": T * d_ff,
        "FFN SiLU gate": T * d_ff,
        "FFN elementwise product": T * d_ff,
        "FFN W3 output": T * d_ff,
        "FFN W2 output": T * d,
    }
    glob = {
        "final RMSNorm": T * d,
        "logits (output embedding)": T * V,     # ← V
        "cross-entropy on logits": T * V,       # ← V
    }
    return per_layer, glob, L * sum(per_layer.values()) + sum(glob.values())


def memory_accounting(T, d, d_ff, h, L, V, gpu_gib=GPU_MEM_GIB):
    P = param_breakdown(V, d, L, d_ff)["TOTAL"]
    b_bytes = 4 * P * BYTES_PER_FP32                 # params + grads + m + v
    per_layer, glob, act_per_sample = activation_elements_per_sample(T, d, d_ff, h, L, V)
    a_bytes = act_per_sample * BYTES_PER_FP32        # 每样本激活 bytes
    max_batch = int((gpu_gib * GIB - b_bytes) // a_bytes)
    return {
        "P": P,
        "b_bytes": b_bytes,
        "per_layer": per_layer,
        "glob": glob,
        "act_per_sample": act_per_sample,
        "a_bytes": a_bytes,
        "max_batch": max_batch,
    }


# ------------------------------------------------------------ (c) AdamW 一步 FLOPs
def adamw_step_flops(P):
    ops = {
        "weight decay (θ ← θ − αλθ)": 2,        # 1 mul + 1 sub(αλ 预计算)
        "m update (m ← β₁m + (1−β₁)g)": 3,      # 2 mul + 1 add
        "v update (v ← β₂v + (1−β₂)g²)": 4,     # g² + 2 mul + 1 add
        "param update (θ ← θ − αₜ m/(√v+ε))": 5,  # sqrt + add ε + div + mul + sub
    }
    per_elem = sum(ops.values())
    return ops, per_elem, per_elem * P


# ------------------------------------------------------------ (d) 训练时间
def train_time(T, d, d_ff, h, L, V, batch, steps, peak_tflops=495.0, mfu=0.5):
    fwd_per_sample = flops_breakdown(T, d, L, d_ff, V)[2]
    fwd_per_step = fwd_per_sample * batch
    total_flops = 3 * fwd_per_step * steps       # 前向 + 反向(2× 前向)
    eff = peak_tflops * 1e12 * mfu
    return fwd_per_sample, total_flops, total_flops / eff / 3600


def main():
    T, d, d_ff, h, L, V = CTX, D_MODEL, D_FF, NUM_HEADS, NUM_LAYERS, VOCAB

    print("=" * 72)
    print(f"GPT-2 XL 配置: T={T}, V={V}, L={L}, d={d}, h={h}, d_ff={d_ff}")
    print("=" * 72)

    m = memory_accounting(T, d, d_ff, h, L, V)
    print("\n########## (a)(b) 训练峰值显存 ##########")
    print(f"  参数量 P                                  {m['P']:>18,d}  ({human(m['P'])})")
    print(f"  b = 4P × 4B (参数+梯度+m+v)               {m['b_bytes'] / GIB:>15.2f} GiB")
    print("\n  每样本激活清单 (B=1):")
    for k, v in m["per_layer"].items():
        print(f"    [x{L:>3}] {k:28s} {v:>15,d}")
    for k, v in m["glob"].items():
        print(f"           {k:28s} {v:>15,d}")
    print(f"  每样本激活元素数合计                      {m['act_per_sample']:>18,d}  ({human(m['act_per_sample'])})")
    print(f"  a = 每样本激活 × 4B                       {m['a_bytes'] / GIB:>15.2f} GiB / 样本")
    print(f"\n  显存(B) ≈ a·B + b = {m['a_bytes'] / GIB:.2f}·B + {m['b_bytes'] / GIB:.2f}  (GiB)")
    print(f"  max batch (80 GiB)                        {m['max_batch']:>18d}")

    print("\n########## (c) AdamW 一步 FLOPs ##########")
    ops, per_elem, total = adamw_step_flops(m["P"])
    for k, v in ops.items():
        print(f"    {k:40s} {v:>3} FLOP/元素")
    print(f"    每元素                        {per_elem:>3} FLOP")
    print(f"    合计 (× P)                    {total:>18,d}  ({human(total)})")

    print("\n########## (d) 单卡 H100 / 50% MFU ##########")
    fwd_per_sample, total_flops, hours = train_time(T, d, d_ff, h, L, V, batch=1024, steps=400_000)
    print(f"    前向 FLOPs / 样本 (T={T})      {fwd_per_sample:>18,d}  ({human(fwd_per_sample)})")
    print(f"    总 FLOPs (×3 × batch1024 ×400K){total_flops:>18,d}  ({human(total_flops)})")
    print(f"    训练时长                      {hours:>18,.1f} 小时  ({hours / 24:,.1f} 天)")


if __name__ == "__main__":
    main()