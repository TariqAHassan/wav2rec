"""

    Test DSP

"""
import librosa
import numpy as np
import torch

from wav2rec.signal.dsp import MelSpectrogram

TOLERANCE = 1e-5
SAMPLE_RATE = 22_050


def _mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b) ** 2).mean()


def test_mel_spectrogram_against_librosa() -> None:
    # Create data
    y = np.sin(np.linspace(0, 8 * np.pi, num=SAMPLE_RATE * 15)).astype(np.float32)
    # Use two to check robustness to batch sizes > 1
    y_torch_double = torch.stack((torch.as_tensor(y), torch.as_tensor(y)))

    mel_spec = MelSpectrogram(
        sr=SAMPLE_RATE,
        n_fft=2048,
        n_mels=128,
        ref="max",
        as_db=True,
        normalize_db=False,
    )

    # Compute the expected/reference array via numpy
    expected = librosa.power_to_db(
        librosa.feature.melspectrogram(y, sr=SAMPLE_RATE, n_fft=2048, n_mels=128),
        ref=np.max,
    )

    # Verify the results match the expected
    actual = mel_spec(y_torch_double)[0].squeeze(0).numpy().T
    assert _mse(actual, expected) < TOLERANCE
    assert np.isclose(((actual - actual) ** 2).mean(), 0)
