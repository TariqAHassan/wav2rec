"""

    Test Engine

"""
from typing import Union

import numpy as np
import pytest
import torch

from wav2rec.core.engine import Wav2Rec, _l2_normalize, _standardize_input
from wav2rec.data.dataset import Wav2RecDataset


@pytest.mark.parametrize(
    "array,expected",
    [
        (
            np.arange(1, dtype="float32"),
            np.array([0.0], dtype="float32"),
        ),
        (
            np.arange(2, dtype="float32"),
            np.array([0.0, 1.0], dtype="float32"),
        ),
        (
            np.arange(3, dtype="float32"),
            np.array([0.0, 0.4472136, 0.8944272], dtype="float32"),
        ),
    ],
)
def test_l2_normalize(
    array: np.ndarray,
    expected: np.ndarray,
) -> None:
    actual = _l2_normalize(array, axis=-1)
    assert np.isclose(actual, expected).all()


@pytest.mark.parametrize(
    "x",
    [
        # Torch
        torch.tensor([1]),
        torch.tensor([[1]]),
        # Numpy
        np.array([1]),
        np.array([[1]]),
    ],
)
def test_standardize_input(x: Union[torch.Tensor, np.ndarray]) -> None:
    actual = _standardize_input(x)
    assert isinstance(actual, torch.Tensor)
    assert actual.ndim == 2


@pytest.mark.parametrize(
    "x",
    [
        # Torch
        torch.tensor([[[1]]]),
        # Numpy
        np.array([[[1]]]),
    ],
)
def test_standardize_input_invalid(x: Union[torch.Tensor, np.ndarray]) -> None:
    with pytest.raises(IndexError):
        _standardize_input(x)


def test_wav2rec_get_projection(
    wav2rec: Wav2Rec,
    wav2rec_dataset: Wav2RecDataset,
) -> None:
    expected_dim: int = 2
    expected_features: int = wav2rec.net.learner.projection_size

    for _, audio in wav2rec_dataset:
        actual = wav2rec.get_projection(audio)
        assert actual.ndim == expected_dim
        assert actual.shape[-1] == expected_features


def test_wav2rec_fit(
    wav2rec: Wav2Rec,
    wav2rec_dataset: Wav2RecDataset,
) -> None:
    assert not wav2rec.fitted
    wav2rec.fit(wav2rec_dataset)
    assert wav2rec.fitted


@pytest.mark.parametrize("cancel_similarity_metric", [True, False])
def test_wav2rec_recommend(
    cancel_similarity_metric: bool,
    wav2rec: Wav2Rec,
    wav2rec_dataset: Wav2RecDataset,
) -> None:
    wav2rec.fit(wav2rec_dataset)
    if cancel_similarity_metric:
        wav2rec.similarity_metric = None

    for _, audio in wav2rec_dataset:
        metrics, paths = wav2rec.recommend(
            x=audio,
            n=len(wav2rec_dataset),
        )
        assert isinstance(metrics, np.ndarray)
        assert isinstance(paths, np.ndarray)
        assert metrics.shape == paths.shape == (1, len(wav2rec_dataset))


def test_wav2rec_recommend_n_too_large(
    wav2rec: Wav2Rec,
    wav2rec_dataset: Wav2RecDataset,
) -> None:
    wav2rec.fit(wav2rec_dataset)
    _, audio = wav2rec_dataset[0]
    with pytest.raises(ValueError):
        wav2rec.recommend(audio, n=len(wav2rec_dataset) + 1)
