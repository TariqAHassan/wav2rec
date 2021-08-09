"""

    Test Transforms

"""
from typing import Tuple, Union

import pytest
import torch

from wav2rec.data.transforms import (
    RandomNoise,
    RandomReplaceMean,
    RandomReplaceZero,
    Resize,
)


def test_random_replace_zero() -> None:
    torch.manual_seed(99)

    x = torch.ones((2, 1000), dtype=torch.float32)
    actual = RandomReplaceZero(p=1)(x)
    assert (actual == 0).any()


def test_random_replace_mean() -> None:
    torch.manual_seed(99)

    x = torch.arange(500, dtype=torch.float32).unsqueeze(0)
    actual = RandomReplaceMean(p=1)(x.clone())
    assert ((actual == x.mean()).sum() > 1).item()


def test_random_noise() -> None:
    torch.manual_seed(99)

    x = torch.ones((2, 1000), dtype=torch.float32)
    actual = RandomNoise(p=1)(x.clone())
    assert (actual != x).float().mean().item() > 0.5


@pytest.mark.parametrize(
    "shape,size,mode,expected",
    [
        ((1, 3, 64, 64), 32, "nearest", (1, 3, 32, 32)),
        ((1, 3, 64, 64), 32, "bicubic", (1, 3, 32, 32)),
        ((1, 3, 128, 128), 32, "bicubic", (1, 3, 32, 32)),
    ],
)
def test_resize(
    shape: torch.Tensor,
    size: Union[int, Tuple[int, ...]],
    mode: str,
    expected: Tuple[int, ...],
) -> None:
    torch.manual_seed(99)

    x = torch.randn(*shape)
    actual = Resize(size, mode=mode)(x.clone())
    assert actual.shape == expected
