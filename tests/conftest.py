"""

    Conftest

"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests import (
    GENERATED_WAVEFORMS_DURATION,
    GENERATED_WAVEFORMS_EXT,
    GENERATED_WAVEFORMS_PATH,
    GENERATED_WAVEFORMS_SR,
)
from wav2rec.core.engine import Wav2Rec
from wav2rec.data.dataset import Wav2RecDataset
from wav2rec.nn.lightening import Wav2RecNet


@pytest.fixture()
def wav2rec_dataset() -> Wav2RecDataset:
    return Wav2RecDataset(
        audio_path=GENERATED_WAVEFORMS_PATH,
        sr=GENERATED_WAVEFORMS_SR,
        offset=0,
        duration=GENERATED_WAVEFORMS_DURATION,
        ext=GENERATED_WAVEFORMS_EXT,
    ).scan()


@pytest.fixture()
def wav2recnet() -> Wav2RecNet:
    class MockWav2RecNet(Wav2RecNet):
        @classmethod
        def load_from_checkpoint(cls, *args: Any, **kwargs: Any) -> MockWav2RecNet:
            return cls()

    return MockWav2RecNet().eval()


@pytest.fixture()
def wav2rec(wav2recnet: Wav2RecNet, mocker) -> Wav2Rec:
    mocker.patch("wav2rec.core.engine.Wav2RecNet", wav2recnet)
    return Wav2Rec(Path(""))
