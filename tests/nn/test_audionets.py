"""

    Test Audio Nets

"""
from typing import Type

import pytest
import torch

from wav2rec.nn.audionets import AudioImageNetwork, AudioResnet50, AudioVit


@pytest.mark.parametrize("network", [AudioResnet50, AudioVit])
def test_audionets(network: Type[AudioImageNetwork]) -> None:
    net = network()

    image = torch.randn((1, 1, net.image_size, net.image_size))
    waveform = torch.randn((1, net.sr))
    assert net(image).shape == net(waveform).shape == (1, net.hidden_features)
