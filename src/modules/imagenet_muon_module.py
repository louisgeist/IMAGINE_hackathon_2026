from typing import Any, Dict

import torch

from src.modules.imagenet_module import ImageNetModule


class ImageNetMuonModule(ImageNetModule):
    """`ImageNetModule` variant that optimizes the network with Muon.

    This subclass is identical to `ImageNetModule` except for
    `configure_optimizers`: Muon needs its parameters split into the hidden
    weight matrices (optimized with Muon) and everything else -- embeddings,
    the classifier head and all 1D gains/biases -- (optimized with an auxiliary
    AdamW). That split has to happen on the module itself rather than on a flat
    parameter iterable, so the configured optimizer is called with ``model=...``
    instead of ``params=...``.

    The configured ``optimizer`` is therefore expected to be a partial around
    :func:`src.modules.optim.muon.build_muon_optimizer` (or any callable that
    accepts a ``model`` keyword and returns a `torch.optim.Optimizer`). The
    learning-rate scheduler wiring is unchanged from the base module.
    """

    def configure_optimizers(self) -> Dict[str, Any]:
        """Build the Muon optimizer from the module and wrap it with the schedulers.

        :return: A dict containing the configured optimizer and learning-rate scheduler.
        """
        optimizer = self.hparams.optimizer(model=self.trainer.model)
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
