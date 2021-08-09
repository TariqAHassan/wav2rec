"""

    Audio-Image Networks

"""
from typing import Any, Tuple, Union

import torch
from torch import nn
from torchvision import models
from vit_pytorch import ViT

from wav2rec.data import transforms
from wav2rec.signal.dsp import MelSpectrogram


class AudioImageNetwork(torch.nn.Module):
    """Class of networks which handle 1D waveforms by making them image-like.

    Args:
        sr (int): sample rate of the audio files
        n_mels (int): number of mel bands to construct for raw audio.
        image_size (int): size to reshape the "images" (Melspectrograms) to.
        **kwargs (Keyword Args): Keyword arguments to pass to ``MelSpectrogram()``

    """

    def __init__(
        self,
        sr: int = 20_050,
        n_mels: int = 256,
        image_size: Union[int, Tuple[int, ...]] = 256,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.sr = sr
        self.n_mels = n_mels
        self.image_size = image_size

        self.bn = nn.BatchNorm2d(1)
        self.wav2spec = torch.nn.Sequential(
            MelSpectrogram(
                sr=self.sr,
                n_mels=self.n_mels,
                as_db=kwargs.pop("as_db", True),
                normalize_db=kwargs.pop("normalize_db", True),
                **kwargs,
            ),
            transforms.Resize(size=self.image_size, mode="nearest"),
        )

    @property
    def hidden_features(self) -> int:
        """Number of features emitted by the network."""
        raise NotImplementedError()


class AudioResnet50(AudioImageNetwork):
    """Resnet50-Based Audio network.

    This network is designed to generate features against
    melspectrogram input, using a Resnet50 model as the encoder.

    Args:
        sr (int): sample rate of the audio files
        n_mels (int): number of mel bands to construct for raw audio.
        image_size (int): size to reshape the "images" (Melspectrograms) to.
        **kwargs (Keyword Arguments): keyword arguments to pass to
            the parent class.

    Notes:
        * Batches are normalized prior to being fed to the network in order to
          stabilize training.

    """

    def __init__(
        self,
        sr: int = 20_050,
        n_mels: int = 256,
        image_size: Union[int, Tuple[int, ...]] = 224,
        **kwargs: Any,
    ) -> None:
        super().__init__(sr=sr, n_mels=n_mels, image_size=image_size, **kwargs)
        self.net = models.resnet50(pretrained=False)
        self.net.conv1 = torch.nn.Conv2d(  # override the first layer
            in_channels=1,
            out_channels=self.net.conv1.out_channels,
            kernel_size=self.net.conv1.kernel_size,
            stride=self.net.conv1.stride,
            padding=self.net.conv1.padding,
            bias=self.net.conv1.bias,
        )
        self._hidden_features: int = self.net.fc.in_features
        self.net.fc = torch.nn.Identity()

    @property
    def hidden_features(self) -> int:
        """Number of features emitted by the network."""
        return self._hidden_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass of the network.

        Args:
            x (torch.Tensor): an input tensor

        Returns:
            torch.Tensor

        """
        if x.ndim == 2:  # assume waveforms
            x = self.wav2spec(x)
        return self.net(self.bn(x))


class AudioVit(AudioImageNetwork):
    """ViT-Based Audio network.

    This network is designed to generate features against
    melspectrogram input, using a ViT model as the encoder.

    Args:
        sr (int): sample rate of the audio files
        n_mels (int): number of mel bands to construct for raw audio.
        image_size (int): size to reshape the "images" (Melspectrograms) to.
        patch_size (int): size of each patch. Must be square.
        dim (int): dimension of output following ``nn.Linear()``
        depth (int): number of transformer blocks.
        heads (int): number of multi-head Attention layers
        mlp_dim (int): dimensions of the multi-layer perceptron (MLP)
            in the feed forward layer of the transformer(s).
        dim_head (int): dimensions in the head of the attention block(s)
        dropout (float): dropout rate to use. Must be on ``[0, 1]``.
        emb_dropout (float): dropout of the embedding layer. Must be
            on ``[0, 1]``.
        **kwargs (Keyword Arguments): keyword arguments to pass to
            the parent class.

    Notes:
        * Batches are normalized prior to being fed to the network in order to
          stabilize training.

    """

    def __init__(
        self,
        sr: int = 20_050,
        n_mels: int = 256,
        image_size: Union[int, Tuple[int, ...]] = 256,
        patch_size: int = 32,
        dim: int = 1024,
        depth: int = 6,
        heads: int = 8,
        mlp_dim: int = 2048,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(sr=sr, n_mels=n_mels, image_size=image_size, **kwargs)
        self.patch_size = patch_size
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.dim_head = dim_head
        self.dropout = dropout
        self.emb_dropout = emb_dropout

        self.net = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=1,  # overridden below
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            channels=1,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout,
        )
        self._hidden_features: int = self.net.mlp_head[0].normalized_shape[0]
        self.net.mlp_head = torch.nn.Identity()

    @property
    def hidden_features(self) -> int:
        """Number of features emitted by the network."""
        return self._hidden_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass of the network.

        Args:
            x (torch.Tensor): an input tensor

        Returns:
            torch.Tensor

        """
        if x.ndim == 2:  # assume waveforms
            x = self.wav2spec(x)
        return self.net(self.bn(x))
