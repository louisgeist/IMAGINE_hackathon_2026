from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torch.utils.data.dataloader import default_collate
from torchvision.datasets import ImageNet
import torchvision.transforms.v2 as T


class ImageNetDataModule(LightningDataModule):
    """`LightningDataModule` for the ImageNet dataset.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_split: Tuple[int, int] = (0.99, 0.01),
        eval_resize_size: int = 256,
        eval_crop_size: int = 224,
        train_crop_size: int = 224,
        interpolation: str = 'bilinear',
        hflip_prob: float = 0.5,
        auto_augment_policy: str = None,
        ra_magnitude: int = None,
        augmix_severity: int = None,
        cutmix_alpha: float = 0.0,
        mixup_alpha: float = 0.2,
        random_erase_prob: float = 0.0,
        batch_size: int = 64,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        pin_memory: bool = False,
    ) -> None:
        """Initialize an `ImageNetDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_split: The train and validation split. Defaults to `(0.99, 0.01)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        interpolation_mode = T.InterpolationMode(interpolation)
        imagenet_mean = (0.485, 0.456, 0.406)
        imagenet_std = (0.229, 0.224, 0.225)
        train_transforms = []
        train_transforms.append(T.RandomResizedCrop(train_crop_size,
                                                    interpolation=interpolation_mode))
        if hflip_prob > 0:
            train_transforms.append(T.RandomHorizontalFlip(hflip_prob))

        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                train_transforms.append(T.RandAugment(interpolation=interpolation_mode, magnitude=ra_magnitude))
            elif auto_augment_policy == "ta_wide":
                train_transforms.append(T.TrivialAugmentWide(interpolation=interpolation_mode))
            elif auto_augment_policy == "augmix":
                train_transforms.append(T.AugMix(interpolation=interpolation_mode, severity=augmix_severity))
            else:
                aa_policy = T.AutoAugmentPolicy(auto_augment_policy)
                train_transforms.append(T.AutoAugment(policy=aa_policy, interpolation=interpolation_mode))

        train_transforms.extend(
            [
                T.PILToTensor(),
                T.ToDtype(torch.float, scale=True),
                T.Normalize(mean=imagenet_mean, std=imagenet_std),
            ]
        )
        if random_erase_prob > 0:
            train_transforms.append(T.RandomErasing(p=random_erase_prob))
        train_transforms.append(T.ToPureTensor())
        self.train_transforms = T.Compose(train_transforms)
        
        self.eval_transforms = T.Compose([
            T.Resize(eval_resize_size, interpolation=interpolation_mode),
            T.CenterCrop(eval_crop_size),
            T.PILToTensor(),
            T.ToDtype(torch.float, scale=True),
            T.Normalize(mean=imagenet_mean, std=imagenet_std),
            T.ToPureTensor()
        ])
        
        
        if cutmix_alpha or mixup_alpha:
            mixup_cutmix = self._get_mixup_cutmix(
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
            )
            self.collate_fn = lambda batch: mixup_cutmix(*default_collate(batch))
        else:
            self.collate_fn = default_collate

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of ImageNet-1k classes (1000).
        """
        return 1000

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        ImageNet(root=self.hparams.data_dir, split='train')
        ImageNet(root=self.hparams.data_dir, split='val')

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = ImageNet(self.hparams.data_dir, split='train')
            
            # Split indices
            train_indices, val_indices = random_split(
                dataset=range(len(dataset)),
                lengths=self.hparams.train_val_split,
                generator=torch.Generator().manual_seed(42)
            )
            
            train_set = ImageNet(self.hparams.data_dir, split='train', transform=self.train_transforms)
            val_set = ImageNet(self.hparams.data_dir, split='train', transform=self.eval_transforms)
            self.data_test = ImageNet(self.hparams.data_dir, split='val', transform=self.eval_transforms)
            
            self.data_train = Subset(train_set, train_indices)
            self.data_val = Subset(val_set, val_indices)

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            prefetch_factor=self.hparams.prefetch_factor,
            collate_fn=self.collate_fn,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            prefetch_factor=self.hparams.prefetch_factor,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            prefetch_factor=self.hparams.prefetch_factor,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass

    def _get_mixup_cutmix(self, mixup_alpha, cutmix_alpha):
        mixup_cutmix = []
        if mixup_alpha > 0:
            mixup_cutmix.append(
                T.MixUp(alpha=mixup_alpha, num_classes=self.num_classes)
            )
        if cutmix_alpha > 0:
            mixup_cutmix.append(
                T.CutMix(alpha=cutmix_alpha, num_classes=self.num_classes)
            )
        if not mixup_cutmix:
            return None

        return T.RandomChoice(mixup_cutmix)


if __name__ == "__main__":
    _ = ImageNetDataModule()