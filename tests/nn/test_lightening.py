"""

    Test Lightening

"""
import torch

from wav2rec.data.dataset import Wav2RecDataset
from wav2rec.nn.lightening import Wav2RecNet


def test_wav2recnet(
    wav2recnet: Wav2RecNet,
    wav2rec_dataset: Wav2RecDataset,
) -> None:
    with torch.inference_mode():
        for _, audio in wav2rec_dataset:
            actual = wav2recnet(audio.unsqueeze(0))
            assert actual.shape[-1] == wav2recnet.learner.projection_size
