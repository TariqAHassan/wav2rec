"""

    Dataset

"""
import logging
import shutil
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple, Union

import torch
from audioread.exceptions import NoBackendError
from tqdm import tqdm

from experiments.fma.data.meta import FmaMetadata
from wav2rec.data.dataset import Wav2RecDataset

log = logging.getLogger(__name__)


class FmaDataset(Wav2RecDataset):
    """Dataset for FMA tracks.

    Args:
        audio_path (Path): path to a FMA track dataset
        metadata_path (Path): path to FMA metadata
        set_subset (str, optional): the subset of the full dataset that
            ``audio_path`` refers to. If ``None``, will be determined
            automatically.
        sr (int): sample rate to use for each track
        offset (int): seconds to skip in each track
        duration (int): the duration of each track to use.
         ext (str, tuple): one or more file extensions in ``audio_path`` to
            filter for
        res_type (str): resampling algorithm
        cache_dir (Path): path to a directory where processed tracks
            will be stored in pickle format.
        **kwargs (Keyword Args): Keyword arguments to pass to
            ``FmaMetadata``.

    References:
        * https://github.com/mdeff/fma

    """

    def __init__(
        self,
        audio_path: Path = Path("~/fma_small").expanduser(),
        metadata_path: Path = Path("~/fma_metadata").expanduser(),
        set_subset: Optional[str] = None,
        sr: int = 22050,
        offset: int = 0,
        duration: int = 15,
        ext: Union[str, Tuple[str, ...]] = "mp3",
        res_type: str = "kaiser_fast",
        cache_dir: Path = Path("~/wav2vec/cache").expanduser(),
        verbose: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            audio_path=audio_path,
            sr=sr,
            offset=offset,
            duration=duration,
            ext=ext,
            res_type=res_type,
            verbose=verbose,
        )
        self.metadata_path = metadata_path
        self.set_subset = set_subset or audio_path.stem.split("_")[-1]
        self.cache_dir = cache_dir

        self.cache_dir_complete = self.cache_dir.joinpath(f"{sr}/{offset}/{duration}")
        self._create_cache()
        self.metadata = FmaMetadata(metadata_path, set_subset=set_subset, **kwargs)

        self.cached_files: List[Path] = list()

    def _include(self, track_id: str) -> bool:
        try:
            duration = self.metadata.get_track_duration(track_id)
            return duration > (self.offset + self.duration)
        except KeyError:
            return False

    def get_audio_files(self) -> Iterable[Path]:
        for f in super().get_audio_files():
            if self._include(f.stem):
                yield f

    def _cache_path(self, file: Path) -> Path:
        return self.cache_dir_complete.joinpath(f"{file.stem}.p")

    def _create_cache(self) -> None:
        self.cache_dir_complete.mkdir(parents=True, exist_ok=True)

    def reset_cache(self) -> None:
        shutil.rmtree(self.cache_dir_complete, ignore_errors=True)
        self._create_cache()

    def collect_cached_files(self) -> None:
        self.cached_files = list(self.cache_dir_complete.rglob("*.p"))

    def cache_all(self, lazy: bool = False, limit: Optional[int] = None) -> None:
        for e, f in tqdm(
            zip(range(len(self.files)), self.files),
            desc="Caching",
            disable=not self.verbose,
            unit="file",
            total=min(len(self.files), limit or float("inf")),
        ):
            if lazy and self._cache_path(f).is_file():
                continue

            try:
                x = self.load_audio(f)
            except (NoBackendError, RuntimeError):
                log.info("Error loading track %r.", f.stem)
                continue

            if len(x) >= self.n_features:
                torch.save(x[: self.n_features], f=self._cache_path(f))
            else:
                log.info("Skipping track %r due to length.", f.stem)

            if limit and e > limit:
                break

        self.collect_cached_files()

    def __len__(self) -> None:
        return len(self.cached_files)

    def __getitem__(self, item: int) -> Tuple[str, torch.Tensor]:
        path = self.cached_files[item]
        return str(path), torch.load(path)
