import torch.nn as nn
from timm.models.metaformer import (
    Attention,
    LayerNorm2dNoBias,
    LayerNormNoBias,
    MetaFormer,
    Pooling,
)


class MetaFormerHybridClassifier(nn.Module):
    """MetaFormer hybrid: pooling token mixer in early stages, attention in late stages.

    Matches the PoolFormer paper Table 5 variant [Pool, Pool, Attention, Attention]
    on the S12 skeleton (~16.5M params, vs ~22M ViT-S/16).
    """

    def __init__(
        self,
        num_classes: int = 1000,
        depths: tuple[int, ...] = (2, 2, 6, 2),
        dims: tuple[int, ...] = (64, 128, 320, 512),
        drop_path_rate: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model = MetaFormer(
            num_classes=num_classes,
            depths=depths,
            dims=dims,
            token_mixers=[Pooling, Pooling, Attention, Attention],
            norm_layers=[
                LayerNorm2dNoBias,
                LayerNorm2dNoBias,
                LayerNormNoBias,
                LayerNormNoBias,
            ],
            use_mlp_head=False,
            drop_path_rate=drop_path_rate,
            **kwargs,
        )

    def forward(self, x):
        return self.model(x)
