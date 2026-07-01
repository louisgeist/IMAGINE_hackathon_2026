# Adapted from https://github.com/SHI-Labs/Compact-Transformers
# Paper: "Escaping the Big Data Paradigm with Compact Transformers" https://arxiv.org/abs/2104.05704


from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn

from src.modules.nets.vision_transformer import Encoder


class Tokenizer(nn.Module):
    """Convolutional tokenizer: replaces ViT's non-overlapping patchify stem with a small
    conv/ReLU/maxpool stack, giving the model convolutional inductive bias before the
    transformer encoder."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 256,
        num_conv_layers: int = 2,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pooling_kernel_size: int = 3,
        pooling_stride: int = 2,
        pooling_padding: int = 1,
        max_pool: bool = True,
        activation_layer: Callable[..., nn.Module] = nn.ReLU,
    ):
        super().__init__()
        self.in_channels = in_channels

        n_filter_list = [in_channels] + [hidden_dim] * num_conv_layers
        layers: list[nn.Module] = []
        for i in range(num_conv_layers):
            layers.append(
                nn.Conv2d(
                    n_filter_list[i],
                    n_filter_list[i + 1],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                )
            )
            layers.append(activation_layer(inplace=True))
            if max_pool:
                layers.append(
                    nn.MaxPool2d(
                        kernel_size=pooling_kernel_size,
                        stride=pooling_stride,
                        padding=pooling_padding,
                    )
                )
        self.conv_layers = nn.Sequential(*layers)

    def sequence_length(self, image_size: int) -> int:
        """Number of tokens produced for a given (square) input resolution."""
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, image_size, image_size)
            return self.forward(dummy).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (n, c, h, w) -> (n, hidden_dim, h', w')
        x = self.conv_layers(x)
        # (n, hidden_dim, h', w') -> (n, h' * w', hidden_dim)
        return x.flatten(2).transpose(1, 2)


class SeqPool(nn.Module):
    """Sequence pooling as per https://arxiv.org/abs/2104.05704.

    Replaces ViT's class token: a learned attention pool over the output sequence
    collapses it into a single vector for classification, so no extra token needs to be
    carried through every encoder layer.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention_pool = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (n, seq_length, hidden_dim)
        weights = self.attention_pool(x).softmax(dim=1)  # (n, seq_length, 1)
        return torch.bmm(weights.transpose(1, 2), x).squeeze(1)  # (n, hidden_dim)


class CompactConvolutionalTransformer(nn.Module):
    """Compact Convolutional Transformer (CCT) as per https://arxiv.org/abs/2104.05704.

    A ViT variant that swaps the patchify stem for a convolutional `Tokenizer` and the
    class token for `SeqPool`, which lowers the amount of data/compute needed to train a
    transformer from scratch. Reuses the same `Encoder`/`EncoderBlock` transformer
    backbone as `VisionTransformer`.
    """

    def __init__(
        self,
        image_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        num_classes: int = 1000,
        in_channels: int = 3,
        num_conv_layers: int = 2,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pooling_kernel_size: int = 3,
        pooling_stride: int = 2,
        pooling_padding: int = 1,
        max_pool: bool = True,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.image_size = image_size
        self.hidden_dim = hidden_dim

        self.tokenizer = Tokenizer(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_conv_layers=num_conv_layers,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            pooling_kernel_size=pooling_kernel_size,
            pooling_stride=pooling_stride,
            pooling_padding=pooling_padding,
            max_pool=max_pool,
        )
        seq_length = self.tokenizer.sequence_length(image_size)

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.seq_pool = SeqPool(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        torch._assert(
            h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!"
        )
        torch._assert(
            w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!"
        )

        x = self.tokenizer(x)
        x = self.encoder(x)
        x = self.seq_pool(x)
        return self.head(x)
