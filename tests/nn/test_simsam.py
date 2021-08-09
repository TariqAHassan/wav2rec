"""

    Test SimSam

"""
from typing import Type

import pytest
import torch

from wav2rec.nn.audionets import AudioImageNetwork, AudioResnet50, AudioVit
from wav2rec.nn.simsam import SimSam


@pytest.mark.parametrize("network", [AudioResnet50, AudioVit])
def test_simsam(network: Type[AudioImageNetwork]) -> None:
    net = network()

    x = torch.randn((2, 22_050))
    simsam = SimSam(net)
    loss = simsam(x)
    assert isinstance(loss.item(), float)
