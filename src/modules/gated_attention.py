from src.modules.imagenet_module import ImageNetModule


class GatedAttentionImageNetModule(ImageNetModule):
    """`ImageNetModule` for training a ViT with gated self-attention.

    The gating mechanism lives entirely inside the network
    (`src.modules.nets.gated_vision_transformer.GatedVisionTransformer`), so the
    training/optimization logic is identical to the base `ImageNetModule`. This
    subclass exists to give the experiment its own dedicated module type for
    clarity and to leave room for gating-specific behaviour (e.g. logging gate
    statistics) later on.
    """
