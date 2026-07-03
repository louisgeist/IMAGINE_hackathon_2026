# Adapted from https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py


import collections
import math
import warnings
from collections import OrderedDict
from functools import lru_cache, partial
from itertools import repeat
from typing import Any, Callable, NamedTuple, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@lru_cache(maxsize=None)
def _nested_order(n: int) -> tuple[int, ...]:
    """Bit-reversal order of [0, n): any prefix is evenly spread, and prefixes are nested
    (the coords for a smaller prefix are a subset of those for a larger one)."""
    bits = max(1, (n - 1).bit_length())
    return tuple(r for i in range(1 << bits) if (r := int(f"{i:0{bits}b}"[::-1], 2)) < n)


def _grid_keep_indices(n_side: int, num_keep: int, device: torch.device) -> torch.Tensor:
    """Flat indices of a nested n_side x n_side grid, kept count close to num_keep."""
    keep_side = max(1, min(n_side, round(num_keep**0.5)))
    coords = torch.tensor(_nested_order(n_side)[:keep_side], device=device)
    return (coords.unsqueeze(1) * n_side + coords.unsqueeze(0)).flatten()


class ConvStemConfig(NamedTuple):
    out_channels: int
    kernel_size: int
    stride: int
    norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d
    activation_layer: Callable[..., nn.Module] = nn.ReLU


class GMRConv2d(nn.Module):
    """Gaussian Mixture Ring convolution (GMR-Conv, arXiv:2504.02819): kernels are weighted
    sums of concentric Gaussian rings, hence radially symmetric and equivariant to rotations
    and reflections. Used by Equi-ViT (arXiv:2601.09130) as an equivariant patch stem."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int = 0,
        num_rings: Optional[int] = None,
        sigma: float = 0.5,
    ):
        super().__init__()
        self.stride = stride
        self.padding = padding
        num_rings = num_rings or (kernel_size + 1) // 2
        coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
        radius = (coords[None, :] ** 2 + coords[:, None] ** 2).sqrt()
        mus = torch.linspace(0, (kernel_size - 1) / 2, num_rings)
        basis = torch.exp(-((radius[None] - mus[:, None, None]) ** 2) / (2 * sigma**2))
        basis = basis / basis.sum(dim=(-2, -1), keepdim=True)
        self.register_buffer("basis", basis)  # (num_rings, k, k)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, num_rings))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.trunc_normal_(self.weight, std=math.sqrt(1 / (in_channels * num_rings)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel = torch.einsum("oir,rhw->oihw", self.weight, self.basis)
        return nn.functional.conv2d(x, kernel, self.bias, stride=self.stride, padding=self.padding)


class MirrorSymConv2d(nn.Module):
    """Conv2d with kernels symmetric (so equivariant) under horizontal flips."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        # store the left half (incl. center column if odd), mirror it to build the full kernel
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, (kernel_size + 1) // 2)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.trunc_normal_(self.weight, std=math.sqrt(1 / (in_channels * kernel_size**2)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel = torch.cat([self.weight, self.weight[..., : self.kernel_size // 2].flip(-1)], dim=-1)
        return nn.functional.conv2d(x, kernel, self.bias, stride=self.stride)


class VisionTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_dim: int,
        encoder: Callable[..., nn.Module],
        dropout: float = 0.0,
        num_classes: int = 1000,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        conv_stem_configs: Optional[list[ConvStemConfig]] = None,
        patch_drop_rate: float = 0.0,
        patch_drop_mode: str = "random",
        pos_embedding_type: str = "learned",
        patch_stem: str = "patchify",
    ):
        super().__init__()
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer
        self.num_registers = num_registers

        if conv_stem_configs is not None:
            # As per https://arxiv.org/abs/2106.14881
            seq_proj = nn.Sequential()
            prev_channels = 3
            for i, conv_stem_layer_config in enumerate(conv_stem_configs):
                seq_proj.add_module(
                    f"conv_bn_relu_{i}",
                    Conv2dNormActivation(
                        in_channels=prev_channels,
                        out_channels=conv_stem_layer_config.out_channels,
                        kernel_size=conv_stem_layer_config.kernel_size,
                        stride=conv_stem_layer_config.stride,
                        norm_layer=conv_stem_layer_config.norm_layer,
                        activation_layer=conv_stem_layer_config.activation_layer,
                    ),
                )
                prev_channels = conv_stem_layer_config.out_channels
            seq_proj.add_module(
                "conv_last",
                nn.Conv2d(in_channels=prev_channels, out_channels=hidden_dim, kernel_size=1),
            )
            self.conv_proj: nn.Module = seq_proj
        elif patch_stem == "gmr":
            # Equi-ViT's best config: two GMR convs [6, 11]; strides 4x4 = patch_size 16
            torch._assert(patch_size == 16, "GMR stem strides assume patch_size 16")
            self.conv_proj = nn.Sequential(
                GMRConv2d(3, hidden_dim // 4, kernel_size=6, stride=4, padding=1),
                nn.GELU(),
                GMRConv2d(hidden_dim // 4, hidden_dim, kernel_size=11, stride=4, padding=4),
            )
        elif patch_stem == "mirror":
            self.conv_proj = MirrorSymConv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)
        else:
            self.conv_proj = nn.Conv2d(
                in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
            )

        seq_length = (image_size // patch_size) ** 2

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1


        self.encoder = encoder(
            seq_length=seq_length,
            hidden_dim=hidden_dim,
        )
        self.seq_length = seq_length

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = nn.Linear(hidden_dim * (self.num_registers + 1), num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(hidden_dim * (self.num_registers + 1), representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, num_classes)

        self.heads = nn.Sequential(heads_layers)

        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = (
                self.conv_proj.in_channels
                * self.conv_proj.kernel_size[0]
                * self.conv_proj.kernel_size[1]
            )
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)
        elif isinstance(getattr(self.conv_proj, "conv_last", None), nn.Conv2d):
            # Init the last 1x1 conv of the conv stem
            nn.init.normal_(
                self.conv_proj.conv_last.weight,
                mean=0.0,
                std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels),
            )
            if self.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.conv_proj.conv_last.bias)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(
            h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!"
        )
        torch._assert(
            w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!"
        )
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        batch_class_token = self.class_token.expand(n, -1, -1)
        if self.num_registers > 0:
            batch_registers = self.registers.expand(n, self.num_registers, self.hidden_dim)
            x = torch.cat([batch_class_token, batch_registers, x], dim=1)
        else:
            x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x, n_side=self.image_size // self.patch_size)

        x = x[:, :self.num_registers + 1].flatten(1)

        x = self.heads(x)

        return x


def build_2d_sincos_pos_embedding(
    seq_length: int, hidden_dim: int, has_class_token: bool = True, temperature: float = 10000.0
) -> torch.Tensor:
    """Fixed 2D sine-cosine positional embedding (as in MAE / DeiT-III).

    Returns a (1, seq_length, hidden_dim) tensor. When ``has_class_token`` is True the first
    position is left as zeros to align with the prepended ``[CLS]`` token.
    """
    n_patches = seq_length - 1 if has_class_token else seq_length
    grid = int(math.isqrt(n_patches))
    torch._assert(grid * grid == n_patches, "sincos pos-embed requires a square patch grid")
    torch._assert(hidden_dim % 4 == 0, "hidden_dim must be divisible by 4 for 2D sincos")
    gy, gx = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    dim4 = hidden_dim // 4
    omega = 1.0 / (temperature ** (torch.arange(dim4) / dim4))
    ox = gx.flatten()[:, None] * omega[None, :]
    oy = gy.flatten()[:, None] * omega[None, :]
    pe = torch.cat([ox.sin(), ox.cos(), oy.sin(), oy.cos()], dim=1)  # (n_patches, hidden_dim)
    if has_class_token:
        pe = torch.cat([torch.zeros(1, hidden_dim), pe], dim=0)
    return pe.unsqueeze(0)  # (1, seq_length, hidden_dim)


class RotaryEmbedding(nn.Module):
    """1D Rotary Position Embedding (RoPE) from RoFormer (Su et al., 2021, arXiv:2104.09864).

    Precomputes cos/sin tables for a fixed sequence length and rotates query/key tensors of
    shape ``(batch, num_heads, seq_length, head_dim)``. Token 0 (the ``[CLS]`` token) sits at
    position 0, where the rotation angle is 0, i.e. it is left unrotated.

    This uses the half-split ("rotate_half") formulation popularised by GPT-NeoX/LLaMA, which
    is equivalent to the paper's block-diagonal rotation matrix up to a fixed permutation of
    the (learned) channel dimensions, and preserves RoPE's defining relative-position property
    ``<f(q, m), f(k, n)>`` depending only on ``m - n``.
    """

    def __init__(self, head_dim: int, seq_length: int, base: float = 10000.0):
        super().__init__()
        torch._assert(head_dim % 2 == 0, "RoPE requires an even head dimension")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))  # (head_dim/2,)
        positions = torch.arange(seq_length).float()  # (seq_length,)
        freqs = torch.outer(positions, inv_freq)  # (seq_length, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_length, head_dim)
        # (1, 1, seq_length, head_dim) so it broadcasts over batch and heads.
        self.register_buffer("cos", emb.cos()[None, None], persistent=False)
        self.register_buffer("sin", emb.sin()[None, None], persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Cast tables to the (possibly fp16 under AMP) dtype of q/k to keep SDPA inputs consistent.
        cos = self.cos.to(dtype=q.dtype)
        sin = self.sin.to(dtype=q.dtype)
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


class RoPESelfAttention(nn.Module):
    """Multi-head self-attention with rotary position embedding applied to Q and K.

    Mirrors ``nn.MultiheadAttention`` (fused QKV projection + output projection, batch_first)
    but exposes Q/K so RoPE can be injected before the scaled dot-product.
    """

    def __init__(
        self, hidden_dim: int, num_heads: int, attention_dropout: float, rotary: RotaryEmbedding
    ):
        super().__init__()
        torch._assert(hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.attention_dropout = attention_dropout
        self.rotary = rotary
        self.in_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        # (b, n, 3*c) -> (3, b, num_heads, n, head_dim)
        qkv = self.in_proj(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (b, num_heads, n, head_dim)
        q, k = self.rotary(q, k)
        dropout_p = self.attention_dropout if self.training else 0.0
        out = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        out = out.transpose(1, 2).reshape(b, n, c)  # (b, n, c)
        return self.out_proj(out)


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        use_qk_norm: bool = False,
        use_swiglu: bool = False,
        pos_embedding_type: str = "learned",
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.rotary: Optional[RotaryEmbedding] = None
        if pos_embedding_type == "learned":
            self.pos_embedding = nn.Parameter(
                torch.empty(1, seq_length, hidden_dim).normal_(std=0.02)
            )  # from BERT
        elif pos_embedding_type == "sincos":
            # Fixed (non-learnable) 2D sine-cosine embedding.
            self.register_buffer(
                "pos_embedding",
                build_2d_sincos_pos_embedding(seq_length, hidden_dim, has_class_token=True),
                persistent=False,
            )
        elif pos_embedding_type == "rope":
            # Rotary embedding is applied inside attention; no additive positional term is used.
            self.pos_embedding = None
            self.rotary = RotaryEmbedding(hidden_dim // num_heads, seq_length)
        else:
            raise ValueError(
                f"Unknown pos_embedding_type={pos_embedding_type!r}; "
                "expected 'learned', 'sincos' or 'rope'"
            )
        self.dropout = nn.Dropout(dropout)
        self.patch_drop_rate = patch_drop_rate
        self.patch_drop_mode = patch_drop_mode
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
                use_qk_norm,
                rotary=self.rotary,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor, n_side: Optional[int] = None):
        torch._assert(
            input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}"
        )
        if self.pos_embedding is not None:
            input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        use_qk_norm: bool = False,
        rotary: Optional[RotaryEmbedding] = None,
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block. With a rotary embedding we need access to Q/K, so we swap the stock
        # nn.MultiheadAttention for a rotary-aware implementation.
        self.ln_1 = norm_layer(hidden_dim)
        self.uses_rope = rotary is not None
        if self.uses_rope:
            self.self_attention = RoPESelfAttention(
                hidden_dim, num_heads, attention_dropout, rotary
            )
        else:
            self.self_attention = QKNormAttention(
                hidden_dim, num_heads, attention_dropout, use_qk_norm=use_qk_norm
            )
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = (
            SwiGLUMLPBlock(hidden_dim, mlp_dim, dropout)
            if use_swiglu
            else MLPBlock(hidden_dim, mlp_dim, dropout)
        )

    def forward(self, input: torch.Tensor):
        torch._assert(
            input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}"
        )
        x = self.ln_1(input)
        if self.uses_rope:
            x = self.self_attention(x)
        else:
            x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class MLP(torch.nn.Sequential):
    """This block implements the multi-layer perceptron (MLP) module.

    Args:
        in_channels (int): Number of channels of the input
        hidden_channels (List[int]): List of the hidden channel dimensions
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the linear layer. If ``None`` this layer won't be used. Default: ``None``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the linear layer. If ``None`` this layer won't be used. Default: ``torch.nn.ReLU``
        inplace (bool, optional): Parameter for the activation layer, which can optionally do the operation in-place.
            Default is ``None``, which uses the respective default values of the ``activation_layer`` and Dropout layer.
        bias (bool): Whether to use bias in the linear layer. Default ``True``
        dropout (float): The probability for the dropout layer. Default: 0.0
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: list[int],
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        inplace: Optional[bool] = None,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        # The addition of `norm_layer` is inspired from the implementation of TorchMultimodal:
        # https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(torch.nn.Dropout(dropout, **params))

        super().__init__(*layers)


class MLPBlock(MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(
            in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i + 1}.{type}"
                    new_key = f"{prefix}{3 * i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class SwiGLUMLPBlock(nn.Module):
    """Transformer MLP block with a SwiGLU gated activation.

    Replaces the standard two-matrix GELU MLP with a gated linear unit:
    ``out = W_down( SiLU(W_gate x) * W_up x )``. This uses three projection
    matrices instead of two, so to keep the parameter count comparable to the
    GELU MLP set ``mlp_dim`` to ~2/3 of the GELU hidden dim (e.g. 1024 vs 1536).
    """

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.w_gate = nn.Linear(in_dim, mlp_dim)
        self.w_up = nn.Linear(in_dim, mlp_dim)
        self.w_down = nn.Linear(mlp_dim, in_dim)
        self.dropout = nn.Dropout(dropout)

        # Match MLPBlock's initialization for comparability.
        for m in (self.w_gate, self.w_up, self.w_down):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.w_gate(x)) * self.w_up(x)
        x = self.dropout(x)
        x = self.w_down(x)
        return self.dropout(x)


class ConvNormActivation(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple[int, ...]] = 3,
        stride: Union[int, tuple[int, ...]] = 1,
        padding: Optional[Union[int, tuple[int, ...], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: Union[int, tuple[int, ...]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
        conv_layer: Callable[..., torch.nn.Module] = torch.nn.Conv2d,
    ) -> None:

        if padding is None:
            if isinstance(kernel_size, int) and isinstance(dilation, int):
                padding = (kernel_size - 1) // 2 * dilation
            else:
                _conv_dim = len(kernel_size) if isinstance(kernel_size, Sequence) else len(dilation)
                kernel_size = _make_ntuple(kernel_size, _conv_dim)
                dilation = _make_ntuple(dilation, _conv_dim)
                padding = tuple((kernel_size[i] - 1) // 2 * dilation[i] for i in range(_conv_dim))
        if bias is None:
            bias = norm_layer is None

        layers = [
            conv_layer(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels))

        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)
        self.out_channels = out_channels

        if self.__class__ == ConvNormActivation:
            warnings.warn(
                "Don't use ConvNormActivation directly, please use Conv2dNormActivation and Conv3dNormActivation instead."
            )


class Conv2dNormActivation(ConvNormActivation):
    """
    Configurable block used for Convolution2d-Normalization-Activation blocks.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the Convolution-Normalization-Activation block
        kernel_size: (int, optional): Size of the convolving kernel. Default: 3
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: None, in which case it will be calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer won't be used. Default: ``torch.nn.BatchNorm2d``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer won't be used. Default: ``torch.nn.ReLU``
        dilation (int): Spacing between kernel elements. Default: 1
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple[int, int]] = 3,
        stride: Union[int, tuple[int, int]] = 1,
        padding: Optional[Union[int, tuple[int, int], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: Union[int, tuple[int, int]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
    ) -> None:

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            norm_layer,
            activation_layer,
            dilation,
            inplace,
            bias,
            torch.nn.Conv2d,
        )


def _make_ntuple(x: Any, n: int) -> tuple[Any, ...]:
    """
    Make n-tuple from input x. If x is an iterable, then we just convert it to tuple.
    Otherwise, we will make a tuple of length n, all with value of x.
    reference: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/utils.py#L8

    Args:
        x (Any): input value
        n (int): length of the resulting tuple
    """
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return tuple(repeat(x, n))
