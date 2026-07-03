import torch.nn as nn
import timm


class TimmClassifier(nn.Module):
    """Thin wrapper around timm models for ImageNet classification."""

    def __init__(
        self,
        model_name: str,
        num_classes: int = 1000,
        pretrained: bool = False,
        **model_kwargs,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.model = timm.create_model(
            model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            **model_kwargs,
        )

    def forward(self, x):
        return self.model(x)
