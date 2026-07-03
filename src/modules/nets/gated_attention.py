from functools import partial
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedMultiheadAttention(nn.Module):
    """Multi-head self-attention with per-head sigmoid gating on the SDPA output.

    The query projection is widened to produce both Q and per-head gate scores.
    Gate scores modulate the SDPA output via sigmoid before the output projection.
    cf. https://arxiv.org/abs/2505.06708
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.wq = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)
        self.wk = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.wv = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_dropout)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in (self.wq, self.wk, self.wv, self.proj):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.normal_(module.bias, std=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, _ = x.shape

        q_raw = self.wq(x).view(batch_size, seq_length, self.num_heads, -1)
        q, gate_score = torch.split(q_raw, [self.head_dim, self.head_dim], dim=-1)

        k = self.wk(x).reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        v = self.wv(x).reshape(batch_size, seq_length, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        dropout_p = self.attn_drop.p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        out = out * torch.sigmoid(gate_score)
        out = out.reshape(batch_size, seq_length, self.embed_dim)
        return self.proj_drop(self.proj(out))


class GatedEncoderBlock(nn.Module):
    """Transformer encoder block with gated self-attention."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ) -> None:
        super().__init__()

        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = GatedMultiheadAttention(
            hidden_dim,
            num_heads,
            attn_dropout=attention_dropout,
            proj_dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

        self.ln_2 = norm_layer(hidden_dim)
        from src.modules.nets.vision_transformer import MLPBlock

        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        torch._assert(
            input.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}",
        )
        x = self.ln_1(input)
        x = self.self_attention(x)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class GatedEncoder(nn.Module):
    """Transformer encoder that uses gated attention blocks."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ) -> None:
        super().__init__()
        self.pos_embedding = nn.Parameter(
            torch.empty(1, seq_length, hidden_dim).normal_(std=0.02)
        )
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.Sequential(
            *[
                GatedEncoderBlock(
                    num_heads,
                    hidden_dim,
                    mlp_dim,
                    dropout,
                    attention_dropout,
                    norm_layer,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        torch._assert(
            input.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}",
        )
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))
