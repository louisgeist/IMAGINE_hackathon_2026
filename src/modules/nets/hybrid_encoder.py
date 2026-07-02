from collections import OrderedDict
from functools import partial
from typing import Callable

import torch
import torch.nn as nn

from src.modules.nets.pom_encoder import PomEncoderBlock
from src.modules.nets.vision_transformer import EncoderBlock

_VALID_LAYER_TYPES = frozenset({"attention", "pom"})
_VALID_LAYER_TYPES_MSG = ", ".join(f"{t!r}" for t in sorted(_VALID_LAYER_TYPES))


def _validate_layer_types(layer_types: list[str], num_layers: int) -> list[str]:
    if len(layer_types) != num_layers:
        raise ValueError(
            f"layer_types length ({len(layer_types)}) must match num_layers ({num_layers})."
        )
    for layer_type in layer_types:
        if layer_type not in _VALID_LAYER_TYPES:
            raise ValueError(
                f"Unknown layer type {layer_type!r}. Expected {_VALID_LAYER_TYPES_MSG}."
            )
    return layer_types


class HybridEncoder(nn.Module):
    """Transformer encoder with per-layer attention or PoM blocks."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        layer_types: list[str],
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        n_sel_heads: int,
        n_groups: int,
        pom_degree: int,
        pom_expansion: int,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        resolved_layer_types = _validate_layer_types(layer_types, num_layers)

        self.pos_embedding = nn.Parameter(
            torch.empty(1, seq_length, hidden_dim).normal_(std=0.02)
        )
        self.dropout = nn.Dropout(dropout)

        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i, layer_type in enumerate(resolved_layer_types):
            if layer_type == "attention":
                layers[f"encoder_layer_{i}"] = EncoderBlock(
                    num_heads,
                    hidden_dim,
                    mlp_dim,
                    dropout,
                    attention_dropout,
                    norm_layer,
                )
            elif layer_type == "pom":
                layers[f"encoder_layer_{i}"] = PomEncoderBlock(
                    n_sel_heads=n_sel_heads,
                    n_groups=n_groups,
                    hidden_dim=hidden_dim,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    pom_degree=pom_degree,
                    pom_expansion=pom_expansion,
                    norm_layer=norm_layer,
                )
            else:
                raise ValueError(
                    f"Unknown layer type {layer_type!r} at index {i}. "
                    f"Expected {_VALID_LAYER_TYPES_MSG}."
                )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(
            input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}"
        )
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))
