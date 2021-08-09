"""

    Digital Signal Processing

"""
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torchlibrosa as tl

_REFERENCE_OPS: Tuple[str, ...] = ("max",)


class MelSpectrogram(torch.nn.Module):
    """Layer to compute the melspectrogram of a 1D audio waveform.

    This layer leverages the convolutional-based ``torchlibrosa``
    library to compute the melspectrogram of an audio waveform.
    The computation can be performed efficiently on a GPU.

    Args:
        sr (int): sample rate of audio
        n_fft (int): FFT window size
        win_length (int): length of the FFT window function
        hop_length (int): number of samples between frames
        f_min (float): lowest frequency (Hz)
        f_max (float, optional): highest frequency (Hz)
        n_mels (int): number of mel bands to create
        window (str): window function to use.
        power (float): exponent for the mel spectrogram.
        center (bool): if True, center the input signal
        pad_mode (str): padding to use at the edges of the signal.
            (Note: this only applies if ``center=True``)
        as_db (bool): if ``True``, convert the output from amplitude to
            decibels.
        ref (float, str): the reference point to use when converting
            to decibels. If a ``float``, the reference point will be
            used 'as is'. If a string, must be ``'max'`` (computed and
            applied individually for each waveform in the batch).
            (Note: this only applies if ``as_db=True``.)
        amin (float): minimum threshold when converting to decibels.
            (Note: this only applies if ``as_db=True``.)
        top_db (float): the maximum threshold value to use when converting
            to decibels. (Note: this only applies if ``as_db=True``.)
        normalize_db (bool): if ``True``, normalize the final output
            s.t. it is on [0, 1]. (Note: requires ``as_db=True``).

    """

    def __init__(
        self,
        sr: int = 20_050,
        n_fft: int = 2048,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        n_mels: int = 128,
        window: str = "hann",
        power: Optional[float] = 2.0,
        center: bool = True,
        pad_mode: str = "reflect",
        as_db: bool = True,
        ref: Union[float, str] = "max",
        amin: float = 1e-10,
        top_db: float = 80.0,
        normalize_db: bool = True,
    ) -> None:
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.n_mels = n_mels
        self.window = window
        self.power = power
        self.center = center
        self.pad_mode = pad_mode
        self.as_db = as_db
        self.ref = ref
        self.amin = amin
        self.top_db = top_db
        self.normalize_db = normalize_db

        if ref not in _REFERENCE_OPS and not isinstance(ref, float):
            raise ValueError(f"Unsupported `ref`: {ref}")

        if amin <= 0:
            raise ValueError("`amin` must > 0")
        elif top_db < 0:
            raise ValueError("`top_db` must be >= 0")
        elif normalize_db and not as_db:
            raise ValueError("`normalize_db` requires as_db to be `True`")

        self.meltransform = torch.nn.Sequential(
            tl.Spectrogram(
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=self.center,
                pad_mode=self.pad_mode,
                power=self.power,
                freeze_parameters=True,
            ),
            tl.LogmelFilterBank(
                sr=self.sr,
                n_mels=self.n_mels,
                fmin=self.f_min,
                fmax=self.f_max,
                is_log=False,  # see below
                freeze_parameters=True,
            ),
        )

    def _power_to_db(self, S: torch.Tensor) -> torch.Tensor:
        magnitude = abs(S) if S.dtype.is_complex else S
        if self.ref == "max":
            ref = torch.max(magnitude.flatten(1), dim=-1, keepdim=True).values
        else:
            ref = torch.as_tensor([self.ref], dtype=S.dtype).repeat(S.shape[0], 1)

        log_spec = 10.0 * torch.log10(torch.clamp(S, min=self.amin, max=np.inf))
        log_spec -= 10.0 * torch.log10(
            ref.clamp(min=self.amin, max=np.inf)
            .view(*ref.shape, 1, 1)
            .repeat(1, 1, *log_spec.shape[-2:])
        )

        if self.top_db is None:
            return log_spec
        else:
            maxes = log_spec.flatten(1).max(dim=-1, keepdim=True).values
            ceilings = (
                (maxes - self.top_db)
                .view(*maxes.shape, 1, 1)
                .repeat(1, 1, *log_spec.shape[-2:])
            )
            return torch.maximum(log_spec, ceilings)

    def _normalize_db(self, log_spec: torch.Tensor) -> torch.Tensor:
        return (log_spec + self.top_db) / self.top_db

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the melspectrogram of ``x``.

        Args:
            x (torch.Tensor): 2D tensor with shape ``[BATCH, TIME]``

        Returns:
            melspec (torch.Tensor): 3D tensor with shape ``[BATCH, CHANNEL, TIME, N_MELS]``.

        """
        S = self.meltransform(x)
        if self.as_db and self.normalize_db:
            return self._normalize_db(self._power_to_db(S))
        elif self.as_db:
            return self._power_to_db(S)
        else:
            return S
