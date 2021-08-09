"""

    Lightening Model

"""
from typing import Any, Optional, Tuple

import pytorch_lightning as pl
import torch

from wav2rec.nn.audionets import AudioImageNetwork, AudioResnet50
from wav2rec.nn.simsam import SimSam


class Wav2RecNet(pl.LightningModule):
    """Unified (SimSam with Encoder) network.

    Args:
        lr (float): learning rate for the model
        encoder (AudioImageNetwork, optional): a model which inherits from
            ``AudioImageNetwork``, to be used as the encoder in ``SimSam``.
            If ``None``, ``AudioResnet50`` will be used.
        **kwargs (Keyword Arguments): Keyword arguments to pass to ``SimSam``.

    """

    def __init__(
        self,
        encoder: Optional[AudioImageNetwork] = None,
        lr: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.encoder = encoder or AudioResnet50()

        self.learner = SimSam(self.encoder, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.learner.wrapped_encoder(x)

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        _, x = batch
        loss = self.learner(x)
        self.log("loss", loss)
        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        _, x = batch
        loss = self.learner(x)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
