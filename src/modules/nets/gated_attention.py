from functools import partial
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedMultiheadAttention(nn.Module):
    """Multi-head self-attention with a sigmoid gate on the attention output.

    The gate modulates how much of the attention output is passed through before
    the output projection::

        attn_out = SDPA(Q, K, V)
        output = out_proj(sigmoid(gate_proj(x)) * attn_out)

  This follows the gated-attention design used in recent transformer work, where
    element-wise gating gives the model finer control over information flow in
    each block.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        gate_bias: float = 2.0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.gate_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters(gate_bias)

    def _reset_parameters(self, gate_bias: float) -> None:
        for module in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, gate_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        dropout_p = self.dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_length, self.embed_dim)

        gate = torch.sigmoid(self.gate_proj(x))
        return self.out_proj(gate * attn_out)


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
        gate_bias: float = 2.0,
    ) -> None:
        super().__init__()

        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = GatedMultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=attention_dropout,
            gate_bias=gate_bias,
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
        x = self.dropout(self.self_attention(x))
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
        gate_bias: float = 2.0,
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
                    gate_bias=gate_bias,
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
