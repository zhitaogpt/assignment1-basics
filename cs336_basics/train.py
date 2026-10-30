"""CS336 A1 §5.3 主训练脚本(training_together)。

用法:
    python -m cs336_basics.train --train-path data/owt_train.npy --valid-path data/owt_valid.npy ...

本文件只包含 **命令行参数解析** 与 **空的 main / evaluate 骨架**。
训练循环、评估、日志、续训逻辑请自行实现。
"""
from __future__ import annotations

import argparse
import math
import os

from .model import *
from .training import *
import numpy as np

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a Transformer LM (CS336 A1 §5.3)")

    # ---- 数据 ----
    p.add_argument("--train-path", required=True, help="训练集 .npy(token id;建议 np.memmap 按需加载)")
    p.add_argument("--valid-path", required=True, help="验证集 .npy")

    # ---- 模型 ----
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--d-ff", type=int, default=1344)
    p.add_argument("--rope-theta", type=float, default=10000.0)

    # ---- 优化器 ----
    p.add_argument("--lr-max", type=float, default=1e-3)
    p.add_argument("--lr-min", type=float, default=1e-4)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-iters", type=int, default=100)
    p.add_argument("--cosine-cycle-iters", type=int, default=5000)
    p.add_argument("--max-grad-norm", type=float, default=1.0)

    # ---- 训练 ----
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-steps", type=int, default=5000)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--checkpoint-path", type=str, default=None, help="checkpoint 路径;已存在则从它续训")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=20)

    # ---- 消融(--variant -> TransformerLM 的 norm/placement/pos_emb/ffn 组合) ----
    p.add_argument(
        "--variant", default="baseline",
        choices=["baseline", "no_rmsnorm", "post_norm", "nope", "silu"],
        help="baseline 与模型默认值完全一致;其它是 §7 消融。",
    )

    return p


# ---- variant → TransformerLM  kwargs 映射(配置接线,无算法逻辑) ----
VARIANT_KWARGS = {
    "baseline":   dict(norm="rmsnorm", norm_placement="pre",  pos_emb="rope", ffn="swiglu"),
    "no_rmsnorm": dict(norm="none",    norm_placement="pre",  pos_emb="rope", ffn="swiglu"),
    "post_norm":  dict(norm="rmsnorm", norm_placement="post", pos_emb="rope", ffn="swiglu"),
    "nope":       dict(norm="rmsnorm", norm_placement="pre",  pos_emb="nope", ffn="swiglu"),
    "silu":       dict(norm="rmsnorm", norm_placement="pre",  pos_emb="rope", ffn="silu"),
}


def resolve_variant(args):
    """--variant -> (实际 d_ff, TransformerLM 的消融 kwargs)。

    silu 只有 2 个 FFN 矩阵,要与 SwiGLU(3 个)对齐参数量。
    handout §7.3 指定:FFN_SiLU 用 d_ff = 4 * d_model(近似匹配 3 矩阵的 SwiGLU)。
    只在调用侧做这一次转换,model.py 不要重复处理。
    """
    kwargs = dict(VARIANT_KWARGS[args.variant])
    d_ff = args.d_ff
    if args.variant == "silu":
        d_ff = 4 * args.d_model
    return d_ff, kwargs


def evaluate(model, dataset, batch_size, context_length, device, num_batches) -> float:
    """在 dataset 上取 num_batches 个 batch,返回平均 cross-entropy loss(标量 float)。"""
    model.eval()
    loss = 0.0
    with torch.no_grad():
        for i in range(num_batches):
            x, y = get_batch(dataset, batch_size, context_length, device)
            loss += cross_entropy(model(x), y).item() / num_batches
    model.train()
    return loss

def main() -> None:
    args = build_arg_parser().parse_args()

    d_ff, variant_kwargs = resolve_variant(args)
    model = TransformerLM(args.vocab_size, args.context_length, args.d_model, args.num_layers, args.num_heads,
                          d_ff, args.rope_theta, args.device, **variant_kwargs)
    print(f"[variant] {args.variant}  d_ff={d_ff}  "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    optimizer = AdamW(model.parameters(), lr=args.lr_max, betas=(args.beta1, args.beta2),
                      eps=args.eps, weight_decay=args.weight_decay)
    dataset = np.load(args.train_path, mmap_mode='r')
    valid_data = np.load(args.valid_path, mmap_mode='r')

    start_step = 0
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        start_step = load_checkpoint(args.checkpoint_path, model, optimizer)

    # 冷启动从 step 0 开始;续训时 start_step=已完成的步数,从该处继续。
    for step in range(start_step, args.num_steps):
        lr = get_lr_cosine_schedule(step, args.lr_max, args.lr_min, args.warmup_iters, args.cosine_cycle_iters)
        for g in optimizer.param_groups:
            g['lr'] = lr
        x, y = get_batch(dataset, args.batch_size, args.context_length, args.device)
        logits = model(x)
        loss = cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), args.max_grad_norm)
        optimizer.step()

        if step % args.log_every == 0:
            print(f"step {step}, loss: {loss.item():.4f}, lr: {lr:.6g}")

        if step % args.eval_every == 0:
            val_loss = evaluate(model, valid_data, args.batch_size, args.context_length, args.device, args.eval_batches)
            perplexity = math.exp(val_loss)
            print(f"val_loss: {val_loss:.4f}, perplexity : {perplexity:.2f}")
            if args.checkpoint_path:
                save_checkpoint(model, optimizer, step, args.checkpoint_path)

if __name__ == "__main__":
    main()