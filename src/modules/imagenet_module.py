from typing import Any, Dict, Tuple

import kornia
import torch
from kornia import augmentation as aug
from lightning import LightningModule
from torchmetrics import MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchvision.transforms import v2 as T


class ImageNetModule(LightningModule):
    """`LightningModule` for ImageNet classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        compile: bool,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        main_scheduler: torch.optim.lr_scheduler,
        mixup_alpha: float,
        cutmix_alpha: float,
        num_classes: int = 1000,
        warmup_scheduler: torch.optim.lr_scheduler = None,
        uint8_pipeline: bool = True,
    ) -> None:
        """Initialize an `ImageNetModule`.

        :param net: The model to train.
        :param compile: Whether to use `torch.compile` on the model for training.
        :param optimizer: The optimizer to use for training.
        :param warmup_steps: The number of warmup steps to use for training. If 0, no warmup scheduler will be used.
        :param main_scheduler: The main learning rate scheduler to use for training.
        :param warmup_scheduler: The learning rate scheduler to use for warmup.
        :param uint8_pipeline: If `True`, batches arrive as uint8 tensors and normalization plus
            CutMix/MixUp are applied here, on GPU. If `False`, the datamodule already normalized
            the images and applied CutMix/MixUp on CPU, so this module leaves batches untouched.
            Must be kept in sync with `ImageNetDataModule`'s `uint8_pipeline` flag.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=True, ignore=["net"])

        self.uint8_pipeline = uint8_pipeline

        if uint8_pipeline:
            imagenet_mean = (0.485, 0.456, 0.406)
            imagenet_std = (0.229, 0.224, 0.225)

            mixup_cutmix = []
            if mixup_alpha > 0:
                mixup_cutmix.append(T.MixUp(alpha=mixup_alpha, num_classes=num_classes))
            if cutmix_alpha > 0:
                mixup_cutmix.append(T.CutMix(alpha=cutmix_alpha, num_classes=num_classes))

            self.cutmix_mixup = T.RandomChoice(mixup_cutmix)

            self.train_augmentations = aug.AugmentationSequential(
                aug.Normalize(mean=imagenet_mean, std=imagenet_std),
                # Add more here if necessary
            )
            self.valid_augmentations = aug.AugmentationSequential(
                aug.Normalize(mean=imagenet_mean, std=imagenet_std)
            )

        self.net = net

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc1 = Accuracy(task="multiclass", num_classes=1000)
        self.train_acc5 = Accuracy(task="multiclass", num_classes=1000, top_k=5)
        self.val_acc1 = Accuracy(task="multiclass", num_classes=1000)
        self.val_acc5 = Accuracy(task="multiclass", num_classes=1000, top_k=5)
        self.test_acc1 = Accuracy(task="multiclass", num_classes=1000)
        self.test_acc5 = Accuracy(task="multiclass", num_classes=1000, top_k=5)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc1.reset()
        self.val_acc5.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of logits.
            - A tensor of target labels.
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        if y.dim() > 1:
            y = y.argmax(dim=1)
        return loss, logits, y.long()

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        if self.uint8_pipeline:
            source, targets = batch
            source = source.float() / 255.0
            source = self.train_augmentations(source)
            batch = (source, targets)
            batch = self.cutmix_mixup(batch)

        loss, logits, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss.detach())
        self.train_acc1(logits.detach(), targets.detach())
        self.train_acc5(logits.detach(), targets.detach())
        self.log(
            "train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/acc1", self.train_acc1, on_step=True, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/acc5", self.train_acc5, on_step=True, on_epoch=True, prog_bar=True
        )

        # return loss or backpropagation will fail
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        if self.uint8_pipeline:
            source, targets = batch
            source = source.float() / 255.0
            source = self.valid_augmentations(source)
            batch = (source, targets)
        loss, logits, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss.detach())
        self.val_acc1(logits.detach(), targets.detach())
        self.val_acc5(logits.detach(), targets.detach())
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc1", self.val_acc1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc5", self.val_acc5, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        if self.uint8_pipeline:
            source, targets = batch
            source = source.float() / 255.0
            source = self.valid_augmentations(source)
            batch = (source, targets)

        _, logits, targets = self.model_step(batch)

        # update and log metrics
        self.test_acc1(logits.detach(), targets.detach())
        self.test_acc5(logits.detach(), targets.detach())
        self.log(
            "test/acc1", self.test_acc1, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "test/acc5", self.test_acc5, on_step=False, on_epoch=True, prog_bar=True
        )

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        main_scheduler = self.hparams.main_scheduler(optimizer=optimizer)
        if self.hparams.warmup_steps > 0:
            warmup_scheduler = self.hparams.warmup_scheduler(optimizer=optimizer)
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[self.hparams.warmup_steps],
            )
        else:
            scheduler = main_scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


if __name__ == "__main__":
    _ = ImageNetModule(None, None, None, None)
