"""CS336 A1 §3.5 transformer_accounting 辅助脚本(纯 glue,不属于交付物)。

用法:  cd assignment1-basics && .venv/bin/python accounting.py

做两件事:
  1. 用参数量公式  hand-count  各配置;
  2. 建真实 TransformerLM,  sum(p.numel()) 自查参数量是否对得上;
  3. 分项列 FLOPs(矩阵乘),给出占比 breakdown,覆盖 (a)(b)(c)(d)(e)。
"""
from __future__ import annotations

import torch

from cs336_basics.model import TransformerLM

ROPE_THETA = 10_000.0
BYTES_PER_PARAM_FP32 = 4


def round_to_multiple_of_64(x: float) -> int:
    return int(round(x / 64)) * 64


# ---------------------------------------------------------------- 参数量公式
def param_breakdown(vocab_size: int, d_model: int, num_layers: int, d_ff: int) -> dict[str, int]:
    """对应 TransformerLM docstring:
       2*V*d + L*(4*d^2 + 3*d*d_ff + 2*d) + d
    """
    emb_and_head = 2 * vocab_size * d_model
    per_layer = 4 * d_model * d_model + 3 * d_model * d_ff + 2 * d_model
    final_norm = d_model
    total = emb_and_head + num_layers * per_layer + final_norm
    return {
        "token_embeddings + lm_head (2*V*d)": emb_and_head,
        "per layer (4d^2 + 3*d*d_ff + 2d)": per_layer,
        "all layers": num_layers * per_layer,
        "ln_final (d)": final_norm,
        "TOTAL": total,
    }


def model_selfcheck(vocab_size, context_length, d_model, num_layers, num_heads, d_ff) -> int:
    model = TransformerLM(
        vocab_size, context_length, d_model, num_layers, num_heads, d_ff, ROPE_THETA
    )
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------- FLOPs 公式
def flops_breakdown(context_length: int, d_model: int, num_layers: int, d_ff: int, vocab_size: int):
    """每个矩阵乘 A(m*n)·B(n*p) = 2*m*n*p。忽略 causal mask(全量计数)。"""
    T, d, L, dff, V = context_length, d_model, num_layers, d_ff, vocab_size
    comp = {
        "Q/K/V 投影 (3 * 2Td²)": 3 * 2 * T * d * d,
        "输出投影 W_O (2Td²)": 2 * T * d * d,
        "注意力分数 QKᵀ (2T²d)": 2 * T * T * d,
        "注意力输出 AV (2T²d)": 2 * T * T * d,
        "FFN w1+w3 (4Td·d_ff)": 2 * (2 * T * d * dff),
        "FFN w2 (2T·d_ff·d)": 2 * T * dff * d,
    }
    per_layer = sum(comp.values())
    lm_head = 2 * T * d * V
    total = L * per_layer + lm_head
    return per_layer, lm_head, total, comp


# ---------------------------------------------------------------- 报告
def human(n: float) -> str:
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.3f}{unit}"
    return f"{n:.0f}"


def report_params(name: str, vocab_size, context_length, d_model, num_layers, num_heads, d_ff):
    pb = param_breakdown(vocab_size, d_model, num_layers, d_ff)
    check = model_selfcheck(vocab_size, context_length, d_model, num_layers, num_heads, d_ff)
    mem_gb = pb["TOTAL"] * BYTES_PER_PARAM_FP32 / 1024**3
    print(f"\n===== {name}  (L={num_layers}, d={d_model}, h={num_heads}, d_ff={d_ff}, V={vocab_size}) =====")
    for k, v in pb.items():
        print(f"  {k:40s} {v:>16,d}   ({human(v)})")
    print(f"  {'模型自查 sum(numel)':40s} {check:>16,d}")
    print(f"  一致? {check == pb['TOTAL']}   | fp32 显存 = {mem_gb:.2f} GB")


def report_flops(name: str, context_length, d_model, num_layers, d_ff, vocab_size):
    per_layer, lm_head, total, comp = flops_breakdown(context_length, d_model, num_layers, d_ff, vocab_size)
    print(f"\n===== {name}  FLOPs (T={context_length}, L={num_layers}, d={d_model}, d_ff={d_ff}) =====")
    print(f"  {'组件':30s} {'每层 FLOPs':>14s} {'占每层':>8s} {'总量 FLOPs':>14s} {'占总量':>8s}")
    for k, v in comp.items():
        print(f"  {k:30s} {human(v):>14s} {v / per_layer:>7.2%} {human(v * num_layers):>14s} {v * num_layers / total:>7.2%}")
    print(f"  {'lm_head (2TdV, 仅一次)':30s} {'-':>14s} {'-':>8s} {human(lm_head):>14s} {lm_head / total:>7.2%}")
    print(f"  {'TOTAL 前向':30s} {'-':>14s} {'-':>8s} {human(total):>14s} {total:.4e}")
    attn = (comp["注意力分数 QKᵀ (2T²d)"] + comp["注意力输出 AV (2T²d)"]) * num_layers
    proj = (comp["Q/K/V 投影 (3 * 2Td²)"] + comp["输出投影 W_O (2Td²)"]) * num_layers
    ffn = (comp["FFN w1+w3 (4Td·d_ff)"] + comp["FFN w2 (2T·d_ff·d)"]) * num_layers
    print(f"  [汇总] 注意力={attn / total:.2%}  投影={proj / total:.2%}  FFN={ffn / total:.2%}  lm_head={lm_head / total:.2%}")


def main():
    V = 50_257
    # (d) 各模型: d_ff 未给出 → 按 8/3*d 取最近 64 倍数(与 XL 一致)
    configs = [
        ("GPT-2 small",  1024, 768, 12, 12),
        ("GPT-2 medium", 1024, 1024, 24, 16),
        ("GPT-2 large",  1024, 1280, 36, 20),
        ("GPT-2 XL",     1024, 1600, 48, 25),
    ]
    print("################ (a) 参数量 + 模型自查 ################")
    for name, ctx, d, L, h in configs:
        report_params(name, V, ctx, d, L, h, round_to_multiple_of_64(8 / 3 * d))

    print("\n\n################ (b)(c)(d) FLOPs 分项 ################")
    for name, ctx, d, L, h in configs:
        report_flops(name, ctx, d, L, round_to_multiple_of_64(8 / 3 * d), V)

    print("\n\n################ (e) GPT-2 XL, context 16384 ################")
    report_flops("GPT-2 XL @ ctx=16384", 16_384, 1600, 48, round_to_multiple_of_64(8 / 3 * 1600), V)


if __name__ == "__main__":
    main()