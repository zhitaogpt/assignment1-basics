"""CS336 A1 §6 文本生成:自回归采样解码(温度缩放 + top-p / nucleus)。

签名占位;实现留给学生。
"""
from __future__ import annotations

import torch


def softmax_with_temperature(
    logits: torch.Tensor,
    temperature: float = 1.0,
    dim: int = -1,
) -> torch.Tensor:
    """把 logits 除以 temperature 后做 softmax,返回概率分布。"""
    eps = 1e-8
    x_max = torch.max(logits, dim=dim, keepdim=True).values
    x_exp = torch.exp((logits - x_max)/(temperature + eps))
    x_sum = torch.sum(x_exp , dim=dim, keepdim=True)
    return x_exp / x_sum


def top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus(top-p)截断:只保留累积概率 >= top_p 的最小集合,其余置 0 并重新归一化。"""
    probs = probs.squeeze(0)
    if top_p >=1.0:
        return probs
    sorted_list = sorted(range(probs.size(0)), key=lambda i: probs[i],reverse=True)
    cump = 0
    idx = -1
    for i in range(probs.size(0)):
        cump += probs[sorted_list[i]]
        if cump >= top_p:
            idx = i
            break
    for i in range(0,idx+1):
        probs[sorted_list[i]] /= cump
    for i in range(idx+1, probs.size(0)):
        probs[sorted_list[i]] = 0
    return probs

def decode(
    model,
    prompt: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float | None = None,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """自回归生成:从 prompt 出发,逐 token 采样,直到 EOS 或达到 max_new_tokens。

    返回拼接 prompt 之后的【完整】token 序列。

    形状契约(采样段统一为一维,无 batch 维):
        prompt                  (1, T)      # 模型内部用 size(1),必须保留 batch 维
        model(prompt)           (1, T, V)
        [:, -1, :]              (1, V)
        squeeze(0) 之后         (V,)        # ← 采样段从这里开始一维
        probs                   (V,)
        multinomial(probs, 1)   (1,)
        unsqueeze(0) 之后       (1, 1)      # ← 拼之前加回 batch 维
        cat([prompt, ...], 1)   (1, T+1)
    返回值:prompt[:, prompt.size(1) - 新生成数:] 或直接返回 prompt(完整序列)。
    """
    # start_len = len(prompt)
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(prompt)[:,-1,:].squeeze(0)
        probs = softmax_with_temperature(logits, temperature)
        if top_p is not None:
            probs = top_p_filter(probs, top_p)
        next_id = torch.multinomial(probs, num_samples=1)
        prompt = torch.cat([prompt, next_id.unsqueeze(0)], dim=1)
        if next_id.item() == eos_token_id:
            break
    return prompt