from functools import lru_cache

import torch

from src.modules.nets.vision_transformer import VisionTransformer


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


class PatchDropVisionTransformer(VisionTransformer):
    """VisionTransformer with patch dropping during training, kept out of the shared
    VisionTransformer on purpose (only the patch_drop experiment instantiates this class).

    `patch_drop_rate` is scheduled per-epoch by ImageNetModule. Dropping happens after the
    additive positional embedding, so this requires an encoder with `pos_embedding` set
    ('learned' or 'sincos', not 'rope' whose positions live inside attention and would
    desync with dropped tokens).
    """

    def __init__(self, *args, patch_drop_rate: float = 0.0, patch_drop_mode: str = "random", **kwargs):
        super().__init__(*args, **kwargs)
        if getattr(self.encoder, "pos_embedding", None) is None:
            raise ValueError("PatchDropVisionTransformer requires an additive pos embedding")
        self.patch_drop_rate = patch_drop_rate
        self.patch_drop_mode = patch_drop_mode

    def forward(self, x: torch.Tensor):
        x = self._process_input(x)
        n = x.shape[0]
        x = torch.cat([self.class_token.expand(n, -1, -1), x], dim=1)

        # mirrors Encoder.forward, with patch dropping inserted after the pos embedding
        enc = self.encoder
        x = x + enc.pos_embedding

        if self.training and self.patch_drop_rate > 0:
            cls_token, patches = x[:, :1], x[:, 1:]
            num_keep = max(1, int(patches.shape[1] * (1 - self.patch_drop_rate)))
            if self.patch_drop_mode == "grid":
                # fixed nested grid, like increasing resolution little by little
                n_side = self.image_size // self.patch_size
                keep_idx = _grid_keep_indices(n_side, num_keep, patches.device)
                keep_idx = keep_idx.unsqueeze(0).expand(patches.shape[0], -1)
            else:
                keep_idx = torch.rand(patches.shape[:2], device=patches.device).argsort(dim=1)[:, :num_keep]
            patches = torch.gather(patches, 1, keep_idx.unsqueeze(-1).expand(-1, -1, patches.shape[-1]))
            x = torch.cat([cls_token, patches], dim=1)

        x = enc.ln(enc.layers(enc.dropout(x)))

        return self.heads(x[:, 0])
