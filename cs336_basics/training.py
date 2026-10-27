"""CS336 A1 §4 训练组件:损失、优化器、学习率调度、梯度裁剪。"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Optional
import math

import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # print("logits.shape", logits.shape)
    # print("targets.shape", targets.shape)
    logits_m = logits.max(dim=-1, keepdim=True).values
    # print("logits_m.shape",logits_m.shape)
    logits_target = torch.gather(logits, -1, targets.unsqueeze(-1)).squeeze(-1)
    # print("logits_target.shape",logits_target.shape)
    logsumexp = torch.logsumexp(logits - logits_m, dim=-1)
    # print("logsumexp.shape",logsumexp.shape)
    logits_target_max = -logits_target + logits_m.squeeze(-1)
    # print("logits_target_max.shape",logits_target_max.shape)
    ce = logits_target_max + logsumexp
    # print("ce.shape",ce.shape)
    return ce.mean()


class SGD(torch.optim.Optimizer):

    def __init__(self, params, lr: float = 1e-3):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = p.grad.data
                t = state.get("t", 0 )
                p.data.add_(grad, alpha=-lr/math.sqrt(t + 1))
                state['t'] = t + 1
        return loss


class AdamW(torch.optim.Optimizer):
    """
    参数量   P = 2Vd + L(4d² + 3d·d_ff + 2d) + d
    梯度     P
    优化器   2P
            ────────────────────────────  固定部分 = 4P

    每层激活  = 8Td + 4T·d_ff + 2hT²          (norm×2 + QKV×3 + AV + 投影 + W2 = 8Td)
    全局激活  = Td + 2TV                       (final norm + logits + CE)

    总显存 = [ 4P + B·( L·(8Td + 4T·d_ff + 2hT²) + Td + 2TV ) ] × 4 bytes
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr" :lr,
            "beta1": betas[0],
            "beta2": betas[1],
            "weight_decay": weight_decay,
            "eps": eps
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            weight_decay = group['weight_decay']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get('t', 1)
                m = state.get('m', torch.zeros_like(p))
                v = state.get('v', torch.zeros_like(p))

                #weight decay
                # 2p
                p.data.add_(-weight_decay * lr * p.data)

                alpha_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                # 3p
                m = beta1 * m + (1 - beta1) * p.grad.data
                # 4p
                v = beta2 * v + (1 - beta2) * (p.grad.data * p.grad.data)
                # 5p
                p.data.add_(-alpha_t * m / (torch.sqrt(v) + eps))
                state['t'] = t + 1
                state['m'] = m
                state['v'] = v
        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate
    elif it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (1 + math.cos((it - warmup_iters) /
                                                        (cosine_cycle_iters - warmup_iters) * math.pi)) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    norm2 = 0
    for param in parameters:
        if param.grad is None:
            continue
        norm2 += torch.sum(param.grad.detach() ** 2)
    norm2 = math.sqrt(norm2)
    if norm2 > max_l2_norm:
        clip_coef = max_l2_norm / (norm2 + 1e-7)
        for param in parameters:
            if param.grad is None:
                continue
            param.grad.detach().mul_(clip_coef)


def get_batch(
    dataset: "np.ndarray",
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, len(dataset) - context_length, (batch_size,))
    idx = starts[:,None] + torch.arange(context_length)[None,:]
    x_np = dataset[idx]
    y_np = dataset[idx + 1]
    # for taken code
    x = torch.from_numpy(x_np).long()
    y = torch.from_numpy(y_np).long()
    return  x.to(device), y.to(device)
    


def save_checkpoint(model, optimizer, iteration: int, out) -> None:
    if out :
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iteration":iteration
            },
            out
        )
    else:
        print(f"out is none {out}")


def load_checkpoint(src, model, optimizer) -> int:
    ckpt = torch.load(src)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt['iteration']
    
         
    
    
