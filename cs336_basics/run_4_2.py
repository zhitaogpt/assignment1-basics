"""CS336 A1 §4.2 learning_rate_tuning — toy SGD 学习率实验。

用法(在 assignment1-basics 目录下):
    .venv/bin/python -m cs336_basics.run_4_2

固定随机种子,用 handout 的 toy 设置(loss = (w**2).mean(),100 个参数)
跑 NUM_ITERS 步,对比 LEARNING_RATES 中各学习率下 loss 的走向。
"""
from __future__ import annotations

import torch

from cs336_basics.training import SGD

LEARNING_RATES = [1.0, 1e1, 1e2, 1e3]
NUM_ITERS = 10
SEED = 0


def run(lr: float, num_iters: int = NUM_ITERS, seed: int = SEED) -> list[float]:
    """跑一次 toy 训练,返回每步的 loss。同 seed → 所有 lr 共享同一初始权重。"""
    torch.manual_seed(seed)
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)

    losses: list[float] = []
    for _ in range(num_iters):
        opt.zero_grad()
        loss = (weights**2).mean()
        losses.append(loss.item())
        loss.backward()
        opt.step()
    return losses


def main() -> None:
    all_losses = [run(lr) for lr in LEARNING_RATES]

    header = f"{'step':>4} | " + " | ".join(f"{'lr=' + format(lr, 'g'):>12}" for lr in LEARNING_RATES)
    print(header)
    print("-" * len(header))
    for t in range(NUM_ITERS):
        row = " | ".join(f"{losses[t]:>12.4e}" for losses in all_losses)
        print(f"{t:>4} | {row}")


if __name__ == "__main__":
    main()