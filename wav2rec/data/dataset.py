"""

    Dataset

"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable, List, Tuple, Union

import torch
from librosa import load
from torch.utils.data import Dataset
from tqdm.auto import tqdm, trange

from wav2rec.data._helpers import zero_pad1d


class Wav2RecDataset(Dataset):
    """Base Wav2Rec Dataset.

    Args:
        audio_path (Path): path to a directory of caches of type ``ext``
        sr (int): sample rate to use for each track
        offset (int): seconds to skip in each track
        duration (int): the duration of each track to use.
        ext (str, tuple): one or more file extensions in ``audio_path`` to
            filter for
        res_type (str): resampling algorithm
        zero_pad (bool): if ``True``, automatically zero pad waveforms
            shorter than ``n_features``.
        verbose (bool): if ``True`` display progress bars

    """

    def __init__(
        self,
        audio_path: Path,
        sr: int = 22050,
        offset: int = 0,
        duration: int = 15,
        ext: Union[str, Tuple[str, ...]] = "mp3",
        res_type: str = "kaiser_fast",
        zero_pad: bool = True,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self.audio_path = audio_path
        self.sr = sr
        self.offset = offset
        self.duration = duration
        self.ext = ext
        self.res_type = res_type
        self.zero_pad = zero_pad
        self.verbose = verbose

        self.files: List[Path] = list()

    @property
    def n_features(self) -> int:
        """Expected number of elements (samples) in each sample."""
        return self.duration * self.sr

    def _audio_path_iter(self) -> Iterable[Path]:
        if isinstance(self.ext, str):
            yield from self.audio_path.rglob(f"*.{self.ext.lstrip('.')}")
        else:
            for e in self.ext:
                yield from self.audio_path.rglob(f"*.{e.lstrip('.')}")

    def get_audio_files(self) -> Iterable[Path]:
        """Generate an iterable of all eligible files in ``audio_path``.

        Yields:
            path

        """
        yield from tqdm(
            self._audio_path_iter(),
            desc="Scanning for Audio",
            disable=not self.verbose,
            total=sum(1 for _ in self._audio_path_iter()),
            unit="file",
        )

    def scan(self) -> Wav2RecDataset:
        """Scan ``audio_path`` for audio files.

        Returns:
            Wav2RecDataset

        """
        files = list(self.get_audio_files())
        if files:
            self.files = files
        else:
            raise OSError(f"No files found in '{str(self.audio_path)}'")
        return self

    def load_audio(self, path: Path) -> torch.Tensor:
        """Load an audio file from ``path``.

        Args:
            path (Path): a file path to a piece of audio

        Returns:
            x (np.ndarray): a mono-signaled piece of audio.

        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="PySoundFile failed.*")
            x, _ = load(
                path=path,
                sr=self.sr,
                mono=True,
                offset=self.offset,
                duration=self.duration,
                res_type=self.res_type,
            )
        if self.zero_pad:
            x = zero_pad1d(x, target_length=self.n_features)
        return torch.as_tensor(x)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, item: int) -> Tuple[str, torch.Tensor]:
        path = self.files[item]
        return str(path), self.load_audio(path)

    def __iter__(self) -> Iterable[Tuple[str, torch.Tensor]]:
        for i in trange(
            len(self),
            disable=not self.verbose,
            desc="Processing Dataset",
        ):
            yield self[i]
