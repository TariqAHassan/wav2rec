"""

    SimSam Model

    Notes:
        * Code adapted from https://github.com/lucidrains/byol-pytorch

    References:
        * https://arxiv.org/abs/2006.07733
        * https://arxiv.org/abs/2011.10566
        * https://github.com/lucidrains/byol-pytorch

"""
from typing import Callable, Optional

import torch
from kornia.augmentation import RandomErasing
from torch import nn
from torch.nn import functional as F

from wav2rec.data import transforms
from wav2rec.nn.audionets import AudioImageNetwork
from wav2rec.signal.dsp import MelSpectrogram


def _loss_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


def _default_aug_factory(encoder: AudioImageNetwork) -> nn.Sequential:
    return nn.Sequential(
        transforms.RandomNoise(p=0.25),
        transforms.RandomReplaceMean(p=0.25),
        transforms.RandomReplaceZero(p=0.25),
        MelSpectrogram(
            sr=encoder.sr,
            n_mels=encoder.n_mels,
            as_db=True,
            normalize_db=True,
        ),
        transforms.Resize(size=encoder.image_size, mode="nearest"),
        RandomErasing(p=0.25),
    )


class _MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        out_features: int,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.out_features = out_features

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _NetWrapper(nn.Module):
    def __init__(
        self,
        encoder: AudioImageNetwork,
        projection_size: int,
        projection_hidden_size: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.projector = _MLP(
            in_features=self.encoder.hidden_features,
            hidden_size=self.projection_hidden_size,
            out_features=self.projection_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        representation = self.encoder(x)
        return self.projector(representation)


class SimSam(nn.Module):
    """Simple Siamese Neural Network for self-supervised
    representation learning.

    Args:
        encoder (AudioImageNetwork): a model which inherits from ``AudioImageNetwork``,
        projection_size (int): dimensionality of vectors to be compared
        projection_hidden_size (int): number of units in Multilayer Perceptron (MLP) networks
        augment1 (callable, optional): First augmentation (yields ``x1). If ``None``,
            the default augmentation will be used.
        augment2 (callable, optional): Second augmentation (yield ``x2``). If ``None``,
            ``augment1`` will be used.

    References:
        * https://arxiv.org/abs/2006.07733
        * https://arxiv.org/abs/2011.10566
        * https://github.com/lucidrains/byol-pytorch

    """

    def __init__(
        self,
        encoder: AudioImageNetwork,
        projection_size: int = 256,
        projection_hidden_size: int = 4096,
        augment1: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        augment2: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size
        self.augment1 = augment1 or _default_aug_factory(encoder)
        self.augment2 = augment2 or self.augment1

        self.wrapped_encoder = _NetWrapper(
            encoder=encoder,
            projection_size=projection_size,
            projection_hidden_size=projection_hidden_size,
        )
        self.predictor = _MLP(
            in_features=projection_size,
            hidden_size=projection_hidden_size,
            out_features=projection_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass of the learner and the
        combined loss.

        Args:
            x (torch.Tensor): a tensor of shape ``[BATCH, ...]``

        Returns:
            loss (torch.Tensor): combined, average loss of the operation

        """
        x1, x2 = self.augment1(x), self.augment2(x)

        online_pred_1 = self.predictor(self.wrapped_encoder(x1))
        online_pred_2 = self.predictor(self.wrapped_encoder(x2))

        with torch.no_grad():
            target_proj_1 = self.wrapped_encoder(x1).detach_()
            target_proj_2 = self.wrapped_encoder(x2).detach_()

        loss_1 = _loss_fn(online_pred_1, target_proj_2)
        loss_2 = _loss_fn(online_pred_2, target_proj_1)

        loss = loss_1 + loss_2
        return loss.mean()
