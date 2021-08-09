"""

    Recommender

"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union, cast

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from wav2rec._utils.validation import check_is_fitted
from wav2rec.core.similarity import cosine_similarity, similarity_calculator
from wav2rec.data.dataset import Wav2RecDataset
from wav2rec.nn.lightening import Wav2RecNet


def _l2_normalize(array: np.ndarray, axis: int = -1) -> np.ndarray:
    norm = np.linalg.norm(array, ord=2, axis=axis, keepdims=True)
    array_norm: np.ndarray = array / np.maximum(norm, np.finfo(array.dtype).eps)
    return array_norm


def _standardize_input(x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    x = torch.as_tensor(x)
    if x.ndim == 1:
        return x.unsqueeze(0)
    elif x.ndim == 2:
        return x
    else:
        raise IndexError(f"Input must be 1D or 2D, got {x.ndim}D")


class Wav2Rec:
    """Waveform recommendation & matching engine.

    Args:
        model_path (Path): path to (training) checkpoint for ``Wav2RecNet``
        distance_metric (str): distance metric to use for nearest neighbours search
        normalize (bool): if ``True`` perform L2 normalization on all projections
        similarity (callable, optional): a callable which accepts two 1D arrays
            and returns a float. Must be compiled with ``numba.jit(nopython=True)``.
            If ``None`` distances will be returned instead (see ``distance_metric``).
        batch_size (int): number of audio files to send to the Wav2Rec neural network
            model for projection simultaneously.
        num_workers (int): number of subprocesses to use when loading data from the
            dataset. See ``torch.utils.data.dataloader.DataLoader``.
        pin_memory (bool): copy tensors to CUDA memory before the data loader
            returns them.
        prefetch_factor (int): Number of samples to load in advance of each worker.
            See ``torch.utils.data.dataloader.DataLoader``.
        device (torch.device, optional): device to run the model on.
            If ``None``, the device will be selected automatically.
        verbose (bool): if ``True`` display a progress bar while fitting.
        **kwargs (Keyword Arguments): Keyword arguments to pass to ``NearestNeighbors``.

    Warnings:
        * By default, this class uses ``distance_metric='euclidean'`` and ``normalize=True``.
          These settings have been purposefully chosen so that the distances computed
          for nearest neighbours search accord with the default similarity metric used:
          cosine similarity. (The euclidean distance between L2 normalized vectors is an
          effective proxy of cosine similarity, see reference below.)

    References:
        * https://en.wikipedia.org/wiki/Cosine_similarity

    """

    def __init__(
        self,
        model_path: Path,
        distance_metric: str = "euclidean",
        normalize: bool = True,
        similarity_metric: Optional[
            Callable[[np.ndarray, np.ndarray], float]
        ] = cosine_similarity,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
        device: Optional[torch.device] = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> None:
        self.model_path = Path
        self.normalize = normalize
        self.similarity_metric = similarity_metric
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.verbose = verbose

        self.net = Wav2RecNet.load_from_checkpoint(model_path).eval().to(self.device)
        self._nneighbours = NearestNeighbors(
            metric=kwargs.pop("metric", distance_metric),
            n_jobs=kwargs.pop("n_jobs", -1),
            **kwargs,
        )

        self.paths: np.ndarray = np.array([], dtype="str")
        self.fitted: bool = False

    @property
    def _X(self) -> np.ndarray:
        return cast(np.ndarray, self._nneighbours._fit_X)

    def _dataset2loader(self, dataset: Wav2RecDataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
        )

    def get_projection(self, x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Get the model's projection of a waveform ``x``.

        Args:
            x (np.ndarray, torch.Tensor): a 1D array or tensor with shape ``[FEATURES]``
                or a 2D array or tensor with shape ``[BATCH, FEATURES]``.

        Returns:
            proj (np.ndarray): a projection of ``x``.

        """
        with torch.inference_mode():
            proj: np.ndarray = (
                self.net(_standardize_input(x).to(self.device)).cpu().numpy()
            )
        return _l2_normalize(proj, axis=-1) if self.normalize else proj

    def fit(self, dataset: Wav2RecDataset) -> Wav2Rec:
        """Fit the recommender to a dataset.

        Fitting is composed of three steps:

            1. Iterating over all files in the dataset
            2. Computing `Wav2RecNet`` projections for each file
            3. Fitting the nearest neighbours algorithm against the projections

        Args:
            dataset (Wav2RecDataset): a dataset to fit against.

        Returns:
            Wav2Rec

        """
        all_paths, all_projections = list(), list()
        with tqdm(desc="Fitting", disable=not self.verbose, total=len(dataset)) as pbar:
            for paths, audio in self._dataset2loader(dataset):
                all_paths.extend(paths)
                all_projections.append(self.get_projection(audio))
                pbar.update(len(audio))

        self.paths = np.asarray(all_paths)
        self._nneighbours.fit(np.concatenate(all_projections))
        self.fitted = True
        return self

    def _get_neighbours(
        self,
        proj: np.ndarray,
        n: int,
        return_distance: bool,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if n > len(self._X):
            raise ValueError("`n` is larger than dataset")

        neighbors: np.ndarray = self._nneighbours.kneighbors(
            proj,
            n_neighbors=n,
            return_distance=return_distance,
        )
        return neighbors

    @check_is_fitted
    def recommend(
        self,
        x: Union[torch.Tensor, np.ndarray],
        n: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Recommend waveforms in ``dataset`` similar to ``x`.

        Args:
            x (np.ndarray, torch.Tensor): a 2D array or tensor
                Shape: ``[BATCH, FEATURES]``.
            n (int): number of recommendations to generate

        Returns:
            result (Tuple[np.ndarray, np.ndarray]): a tuple containing:

                * ``metrics``: a 2D array of either similarity or distance metrics.
                        Shape: ``[BATCH, NEIGHBOURS]``.
                * ``paths``: a 2D array of recommended file paths.
                        Shape: ``[BATCH, NEIGHBOURS]``.

        """
        proj = self.get_projection(x)
        if callable(self.similarity_metric):
            ix = self._get_neighbours(proj, n=n, return_distance=False)
            metrics = similarity_calculator(
                X_query=proj,
                X_neighbours=self._X[ix],
                metric=self.similarity_metric,
            )
        else:
            metrics, ix = self._get_neighbours(proj, n=n, return_distance=True)
        return metrics, self.paths[ix]
