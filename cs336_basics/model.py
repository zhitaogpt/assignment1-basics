"""Transformer language model building blocks."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class Linear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self._din = in_features
        self._dout = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))

        # init
        sigma = math.sqrt(2 / (self._din + self._dout))
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=sigma, a=-3 * sigma, b=3 * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x @ self.weight.T  # -> [batch, seq_len, out_features]


class Embedding(nn.Module):
    """
    参数量: num_embeddings * embedding_dim
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))

        torch.nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty((d_model,), device=device, dtype=dtype))
        torch.nn.init.ones_(self.weight)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        indtype = x.dtype
        x = x.to(torch.float32) # upcast to float32 for numerical stability
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        res =  x * (self.weight / rms)
        return res.to(torch.float16 if indtype == torch.float16 else indtype) # downcast back to original dtype


class SwiGLU(nn.Module):
    """
    参数量: 3 * d_model * d_ff
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device=None,
        dtype=None,
        mode="swiglu"
    ):
        super().__init__()
        self.mode = mode
        # 升维
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype) if mode== "swiglu" else None

        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)


    def _silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r1 = self.w1(x)
        if self.mode == "swiglu":
            r3 = self.w3(x)
            silu = self._silu(r1) * r3
        else:
            silu = self._silu(r1)
        return self.w2(silu)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device=None,
    ):
        super().__init__()
        pos = torch.arange(max_seq_len, device=device)
        angle = theta ** (-torch.arange(0, d_k, 2, device=device) / d_k) # cos shape (max_seq_len, d_k // 2)
        cos = torch.cos(pos[:, None] * angle[None, :])
        sin = torch.sin(pos[:, None] * angle[None, :])
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)

        return x_rotated


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    x_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp / x_sum

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))

    attention_weights = softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)

    return output


class MultiHeadSelfAttention(nn.Module):
    """
    参数量: d_model * d_model * 4
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta=None,
        max_seq_len=None,
        device=None,
        dtype=None,
        pos_emb="rope"
    ):
        super().__init__()
        self.dk = d_model // num_heads
        self.num_heads = num_heads
        if max_seq_len is not None and theta is not None and pos_emb == "rope":
            self.rope = RotaryPositionalEmbedding(theta, self.dk, max_seq_len, device=device)
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)


    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.dk).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.dk).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.dk).transpose(1, 2)

        if hasattr(self, 'rope') and token_positions is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
        attention_output = scaled_dot_product_attention(Q, K, V, mask)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        output = self.output_proj(attention_output)
        return output


class TransformerBlock(nn.Module):
    """
    参数量: 4 * d_model * d_model + 3 * d_model * d_ff + 2 * d_model
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device=None,
        dtype=None,
        norm="rmsnorm",#abalation
        norm_placement="pre",
        pos_emb="rope",
        ffn="swiglu"
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype) if norm == "rmsnorm" else None
        self.attn = MultiHeadSelfAttention(d_model, num_heads, theta, max_seq_len, device=device, dtype=dtype, pos_emb=pos_emb)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype) if norm == "rmsnorm" else None
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype,mode=ffn)
        self.abalation={
            "norm_placement":norm_placement
        }

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if token_positions is None:
            token_positions = torch.arange(x.size(1), device=x.device)
        if self.abalation.get("norm_placement","pre") == "pre":
            x_norm1 = self.ln1(x) if self.ln1 else x
            attention_output = self.attn(x_norm1, token_positions)
            x = x + attention_output

            x_norm2 = self.ln2(x) if self.ln2 else x
            ffn_output = self.ffn(x_norm2)
            x = x + ffn_output
        else:
            x = self.attn(x, token_positions) + x
            x = self.ln1(x) if self.ln1 else x

            x = self.ffn(x) + x
            x = self.ln2(x) if self.ln2 else x        

        return x


class TransformerLM(nn.Module):
    """
    参数量: vocab_size * d_model * 2 + num_layers * (4 * d_model * d_model + 3 * d_model * d_ff + 2 * d_model) + d_model 
    """
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None,
        norm="rmsnorm",#abalation
        norm_placement="pre",
        pos_emb="rope",
        ffn="swiglu"
    ):
        super().__init__()
        # vocab_size * d_model
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        # num_layers * (4 * d_model * d_model + 3 * d_model * d_ff + 2 * d_model)
        self.layers = nn.ModuleList(
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device=device, dtype=dtype,
                             norm=norm, norm_placement=norm_placement, pos_emb=pos_emb, ffn=ffn)
            for _ in range(num_layers)
        )
        # d_model
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype) if norm =="rmsnorm" else None
        # d_model * vocab_size
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)


    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        token_positions = torch.arange(in_indices.size(1), device=in_indices.device)
        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x, token_positions)
        x = self.ln_final(x) if self.ln_final else x
        logits = self.lm_head(x)
        return logits
