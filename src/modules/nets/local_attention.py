import math
from functools import partial
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.nets.vision_transformer import MLPBlock


class Attention(nn.Module):
    """Self-attention with optional attention mask support."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

        for module in (self.wq, self.wk, self.wv, self.proj):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.normal_(module.bias, std=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        kv: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        kv_input = x if kv is None else kv
        kv_len = kv_input.shape[1]

        q = self.wq(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.wk(kv_input).reshape(batch_size, kv_len, self.num_heads, self.head_dim)
        v = self.wv(kv_input).reshape(batch_size, kv_len, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        dropout_p = self.attn_drop.p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p
        )

        out = out.transpose(1, 2).contiguous().reshape(batch_size, seq_len, self.dim)
        return self.proj_drop(self.proj(out))


def infer_grid_size(seq_length: int) -> int:
    """Infer spatial grid side length from a sequence that starts with a CLS token."""
    spatial_tokens = seq_length - 1
    grid_size = int(math.sqrt(spatial_tokens))
    if grid_size * grid_size != spatial_tokens:
        raise ValueError(
            f"Expected seq_length - 1 to be a perfect square, got {spatial_tokens} "
            f"(seq_length={seq_length})."
        )
    return grid_size


def build_window_attention_mask(grid_size: int, window_size: int) -> torch.Tensor:
    """Build a boolean mask for non-overlapping windowed self-attention on a patch grid.

    Returns:
        Mask of shape (grid_size * grid_size, grid_size * grid_size) where True means
        the query/key pair is allowed to attend.
    """
    if grid_size % window_size != 0:
        raise ValueError(
            f"grid_size ({grid_size}) must be divisible by window_size ({window_size})."
        )

    num_spatial = grid_size * grid_size
    token_ids = torch.arange(num_spatial)
    rows = token_ids // grid_size
    cols = token_ids % grid_size
    windows_per_dim = grid_size // window_size
    window_ids = (rows // window_size) * windows_per_dim + (cols // window_size)
    return window_ids.unsqueeze(0) == window_ids.unsqueeze(1)


def build_local_attention_mask(grid_size: int, window_size: int) -> torch.Tensor:
    """Build a boolean mask for windowed attention with a leading CLS token.

    Layout: ``[cls, spatial_0, ..., spatial_{N-1}]``.

    - CLS attends to the full sequence.
    - Spatial tokens attend only within their window (no CLS key).
    """
    spatial_mask = build_window_attention_mask(grid_size, window_size)
    seq_len = spatial_mask.shape[0] + 1
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    mask[0, :] = True
    mask[1:, 1:] = spatial_mask
    return mask


def shift_tokens(x: torch.Tensor, grid_size: int, shift_size: int) -> torch.Tensor:
    """Cyclically shift spatial tokens on the 2D patch grid."""
    if shift_size == 0:
        return x
    batch_size, _, channels = x.shape
    x = x.view(batch_size, grid_size, grid_size, channels)
    x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
    return x.view(batch_size, grid_size * grid_size, channels)


def reverse_shift_tokens(x: torch.Tensor, grid_size: int, shift_size: int) -> torch.Tensor:
    """Reverse a cyclic shift applied by `shift_tokens`."""
    if shift_size == 0:
        return x
    batch_size, _, channels = x.shape
    x = x.view(batch_size, grid_size, grid_size, channels)
    x = torch.roll(x, shifts=(shift_size, shift_size), dims=(1, 2))
    return x.view(batch_size, grid_size * grid_size, channels)


class LocalEncoderBlock(nn.Module):
    """Transformer block with windowed local attention and a CLS token."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        grid_size: int,
        window_size: int,
        shift_size: int = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        if grid_size % window_size != 0:
            raise ValueError(
                f"grid_size ({grid_size}) must be divisible by window_size ({window_size})."
            )
        if not 0 <= shift_size < window_size:
            raise ValueError(
                f"shift_size ({shift_size}) must be in [0, window_size) "
                f"(window_size={window_size})."
            )

        self.grid_size = grid_size
        self.window_size = window_size
        self.shift_size = shift_size

        self.ln_1 = norm_layer(hidden_dim)
        self.attention = Attention(
            hidden_dim,
            num_heads,
            attn_dropout=attention_dropout,
            proj_dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

        local_attn_mask = build_local_attention_mask(grid_size, window_size)
        self.register_buffer("local_attn_mask", local_attn_mask, persistent=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        torch._assert(
            input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}"
        )

        x = self.ln_1(input)
        cls_token = x[:, :1]
        spatial = x[:, 1:]

        if self.shift_size > 0:
            spatial = shift_tokens(spatial, self.grid_size, self.shift_size)

        x = torch.cat([cls_token, spatial], dim=1)
        x = self.attention(x, attn_mask=self.local_attn_mask)

        if self.shift_size > 0:
            cls_token = x[:, :1]
            spatial = reverse_shift_tokens(x[:, 1:], self.grid_size, self.shift_size)
            x = torch.cat([cls_token, spatial], dim=1)

        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y
