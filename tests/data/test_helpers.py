"""

    Test Helpers

"""
import numpy as np
import pytest

from wav2rec.data._helpers import zero_pad1d


@pytest.mark.parametrize(
    "x,target_length",
    [
        (np.array([1, 2]), 3),
        (np.array([1, 2, 3]), 3),
    ],
)
def test_zero_pad1d(
    x: np.ndarray,
    target_length: int,
) -> None:
    actual = zero_pad1d(x, target_length=target_length)
    assert actual.ndim == 1
    assert actual.shape[-1] == target_length


def test_zero_pad1d_invalid_dims() -> None:
    with pytest.raises(IndexError):
        zero_pad1d(np.array([[1]]), target_length=2)


def test_zero_pad1d_x_too_long() -> None:
    with pytest.raises(IndexError):
        zero_pad1d(np.array([[1, 2, 3]]), target_length=2)
