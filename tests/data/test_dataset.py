"""

    Test Dataset

"""
from copy import deepcopy
from pathlib import Path

import torch

from wav2rec.data.dataset import Wav2RecDataset


def test_wav2rec_dataset_n_features(wav2rec_dataset: Wav2RecDataset) -> None:
    expected: int = wav2rec_dataset.duration * wav2rec_dataset.sr
    assert wav2rec_dataset.n_features == expected


def test_wav2rec_dataset_get_audio_files(wav2rec_dataset: Wav2RecDataset) -> None:
    actual = list(wav2rec_dataset.get_audio_files())
    assert all(isinstance(p, Path) for p in actual)


def test_wav2rec_dataset_scan(wav2rec_dataset: Wav2RecDataset) -> None:
    before = deepcopy(wav2rec_dataset.files)
    wav2rec_dataset.scan()
    assert wav2rec_dataset.files == before


def test_wav2rec_dataset_load_audio(wav2rec_dataset: Wav2RecDataset) -> None:
    for path, _ in wav2rec_dataset:
        actual = wav2rec_dataset.load_audio(Path(path))
        assert isinstance(actual, torch.Tensor)
        assert actual.ndim == 1


def test_wav2rec__len__(wav2rec_dataset: Wav2RecDataset) -> None:
    expected: int = len(wav2rec_dataset.files)
    assert wav2rec_dataset.__len__() == expected
    assert wav2rec_dataset.__len__() > 0


def test_wav2rec__getitem__(wav2rec_dataset: Wav2RecDataset) -> None:
    for item in range(len(wav2rec_dataset)):
        path, audio = wav2rec_dataset.__getitem__(item)
        assert isinstance(path, str)
        assert isinstance(audio, torch.Tensor)


def test_wav2rec__iter__(wav2rec_dataset: Wav2RecDataset) -> None:
    for path, audio in wav2rec_dataset:
        assert isinstance(path, str)
        assert isinstance(audio, torch.Tensor)
