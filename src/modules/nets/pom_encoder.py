import torch
import torch.nn as nn
from typing import Callable, OrderedDict, Optional, Union, Tuple, Any
from functools import partial

from src.modules.nets.vision_transformer import MLPBlock
from pom import PoM

class PomEncoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        n_sel_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        pom_degree: int,
        pom_expansion:int,
        n_groups: int,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(
            torch.empty(1, seq_length, hidden_dim).normal_(std=0.02)
        )  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
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
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(
            input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}"
        )
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))


class PomEncoderBlock(nn.Module):
    """Pom encoder block."""

    def __init__(
        self,
        n_sel_heads: int,
        n_groups: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        pom_degree: int,
        pom_expansion:int,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.n_sel_heads = n_sel_heads
        self.n_groups = n_groups

        # Attention block -> PoM
        self.ln_1 = norm_layer(hidden_dim)
        self.pom = PoM(hidden_dim, pom_degree, pom_expansion, n_groups, n_sel_heads, bias=False)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(
            input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}"
        )
        x = self.ln_1(input)
        x = self.pom(x)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y