from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F

from src.modules.imagenet_module import ImageNetModule


class LabelSmoothing:
    r"""Turns hard (or soft) targets into label-smoothed targets.

    For a target ``y`` and ``K`` classes the smoothed target is::

        y_LS = (1 - alpha) * y + alpha / K

    where ``y`` is the (possibly one-hot) target distribution and ``alpha`` is the
    smoothing strength. The formula preserves a valid probability distribution
    (the entries still sum to one) for both one-hot targets and already-soft
    targets such as those produced by MixUp/CutMix.

    Smoothing is applied only within the epoch window
    ``[start_epoch, end_epoch)``, so it can be limited to e.g. the first or the
    final epochs. The default window ``[0.0, 1.0]`` spans the whole run:

    - whole run: ``start_epoch=0.0``, ``end_epoch=1.0``.
    - first epochs: ``start_epoch=0.0``, ``end_epoch=x``.
    - final epochs: ``start_epoch=x``, ``end_epoch=1.0``.

    Epoch bounds may be given as integers (absolute epoch indices) or as floats
    in ``[0, 1]`` (fractions of ``max_epochs``, resolved at call time).
    """

    def __init__(
        self,
        alpha: float = 0.1,
        num_classes: int = 1000,
        start_epoch: float = 0.0,
        end_epoch: float = 1.0,
    ) -> None:
        """Initialize the label smoother.

        :param alpha: Smoothing strength in ``[0, 1]``. ``0`` disables smoothing.
        :param num_classes: Number of classes ``K``.
        :param start_epoch: Epoch (int) or fraction of training (float in
            ``[0, 1]``) at which the active window starts (inclusive).
        :param end_epoch: Epoch (int) or fraction of training (float in
            ``[0, 1]``) at which the active window ends (exclusive).
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        self.alpha = alpha
        self.num_classes = num_classes
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch

    @staticmethod
    def _resolve_epoch(value: float, max_epochs: Optional[int]) -> float:
        """Resolve a fractional epoch bound to an absolute epoch index."""
        # Treat floats in [0, 1] as fractions of the total training length.
        if isinstance(value, float) and 0.0 <= value <= 1.0:
            if max_epochs is None:
                raise ValueError(
                    "Fractional epoch bounds require max_epochs to be provided."
                )
            return value * max_epochs
        return value

    def is_active(self, epoch: int, max_epochs: Optional[int] = None) -> bool:
        """Return whether smoothing should be applied at the given epoch."""
        if self.alpha == 0.0:
            return False

        start = self._resolve_epoch(self.start_epoch, max_epochs)
        end = self._resolve_epoch(self.end_epoch, max_epochs)
        return start <= epoch < end

    def current_alpha(self, epoch: int, max_epochs: Optional[int] = None) -> float:
        """Return the effective smoothing strength at the given epoch."""
        return self.alpha if self.is_active(epoch, max_epochs) else 0.0

    def __call__(
        self,
        targets: torch.Tensor,
        epoch: int = 0,
        max_epochs: Optional[int] = None,
    ) -> torch.Tensor:
        """Apply label smoothing to ``targets`` for the current ``epoch``.

        :param targets: Either class indices of shape ``(N,)`` or a target
            distribution of shape ``(N, K)`` (e.g. from MixUp/CutMix).
        :param epoch: The current training epoch (0-indexed).
        :param max_epochs: Total number of training epochs, required only when
            fractional epoch bounds are used.
        :return: Smoothed targets of shape ``(N, K)`` when active, otherwise the
            original ``targets`` untouched.
        """
        alpha = self.current_alpha(epoch, max_epochs)
        if alpha == 0.0:
            return targets

        if targets.dim() == 1:
            targets = F.one_hot(targets.long(), num_classes=self.num_classes).to(
                dtype=torch.get_default_dtype()
            )
        else:
            targets = targets.to(dtype=torch.get_default_dtype())

        return (1.0 - alpha) * targets + alpha / self.num_classes


class LabelSmoothingImageNetModule(ImageNetModule):
    """`ImageNetModule` variant that applies label smoothing to the targets.

    This subclass changes nothing about the base module except the target used
    to compute the training loss: it replaces the hard targets by the
    label-smoothed targets produced by a `LabelSmoothing` instance. Smoothing is
    only applied during training; validation and test keep using hard labels.
    """

    def __init__(
        self,
        *args: Any,
        label_smoothing: Optional[LabelSmoothing] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the module.

        :param label_smoothing: The `LabelSmoothing` used to smooth the training
            targets. If `None`, this behaves exactly like `ImageNetModule`.
        :param args: Positional arguments forwarded to `ImageNetModule`.
        :param kwargs: Keyword arguments forwarded to `ImageNetModule`.
        """
        super().__init__(*args, **kwargs)
        self.label_smoothing = label_smoothing

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Same as `ImageNetModule.model_step`, but smooths the training targets.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.
        :return: A tuple containing the loss, the logits and the (hard) target labels.
        """
        x, y = batch
        logits = self.forward(x)

        # Only smooth the targets while training; validation/test use hard labels.
        target = y
        if self.training and self.label_smoothing is not None:
            max_epochs = self.trainer.max_epochs if self.trainer is not None else None
            target = self.label_smoothing(
                y, epoch=self.current_epoch, max_epochs=max_epochs
            )

        loss = self.criterion(logits, target)
        if y.dim() > 1:
            y = y.argmax(dim=1)
        return loss, logits, y.long()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Apply label smoothing to a set of targets."
    )
    parser.add_argument(
        "--alpha", type=float, default=0.1, help="Smoothing strength in [0, 1]."
    )
    parser.add_argument(
        "--num-classes", type=int, default=5, help="Number of classes K."
    )
    parser.add_argument(
        "--start-epoch",
        type=float,
        default=0.0,
        help="Epoch (int) or fraction of training (float in [0, 1]) at which smoothing starts.",
    )
    parser.add_argument(
        "--end-epoch",
        type=float,
        default=1.0,
        help="Epoch (int) or fraction of training (float in [0, 1]) at which smoothing ends.",
    )
    parser.add_argument(
        "--targets",
        type=int,
        nargs="+",
        default=[0, 2],
        help="Class indices to smooth.",
    )
    parser.add_argument(
        "--epoch", type=int, default=0, help="Current training epoch (0-indexed)."
    )
    parser.add_argument(
        "--max-epochs", type=int, default=10, help="Total number of training epochs."
    )
    args = parser.parse_args()

    smoother = LabelSmoothing(
        alpha=args.alpha,
        num_classes=args.num_classes,
        start_epoch=args.start_epoch,
        end_epoch=args.end_epoch,
    )
    y = torch.tensor(args.targets)
    print(y)
    for epoch in range(args.max_epochs):
        print(smoother(y, epoch=epoch, max_epochs=args.max_epochs))
