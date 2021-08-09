"""

    Transforms

"""
from typing import Any, Tuple, Union

import numpy as np
import torch
from numpy.random import uniform
from torch import nn
from torch.nn import functional as F  # noqa

from wav2rec._utils.printing import auto_repr


class RandomOp(nn.Module):
    def __init__(self, p: float) -> None:
        """Base class for randomly applying an operation.

        Args:
            p (float): probability of performing the transformation

        """
        super().__init__()
        self.p = p

    def __repr__(self) -> str:
        return auto_repr(self, p=self.p)

    def op(self, x: torch.Tensor) -> torch.Tensor:
        """Operation to perform.

        Args:
            x (torch.Tensor): tensor to operate on

        Returns:
            torch.Tensor

        """
        raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform ``op()`` on ``x`` with probability ``p``.

        Args:
            x (torch.Tensor): tensor to operate on

        Returns:
            torch.Tensor

        """
        if np.random.uniform(0, 1) <= self.p:
            return self.op(x)
        else:
            return x


class _RandomReplace(RandomOp):
    def __init__(
        self,
        p: float = 0.3,
        min_chunk_width: float = 0.01,
        max_chunk_width: float = 0.10,
    ) -> None:
        super().__init__(p)
        self.min_chunk_width = min_chunk_width
        self.max_chunk_width = max_chunk_width

    def __repr__(self) -> str:
        return auto_repr(
            self,
            p=self.p,
            min_chunk_width=self.min_chunk_width,
            max_chunk_width=self.max_chunk_width,
        )

    def replacement(
        self,
        x: torch.Tensor,
        a: int,
        b: int,
    ) -> Union[float, torch.Tensor]:
        """Generate replacement.

        Args:
            x (torch.Tensor): tensor to operate on. Should be of the
                form ``[BATCH, FEATURES]``.
            a (float): start position in the tensor
            b (float): end position in the tensor

        Returns:
            torch.Tensor

        """
        raise NotImplementedError()

    def op(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly replace part of ``x`` using the output
        of ``replacement()``.

        Args:
            x (torch.Tensor): tensor to operate on.

        Returns:
            torch.Tensor

        """
        batch, length = x.shape

        def random_range() -> Tuple[int, int]:
            buffer = int(length * self.max_chunk_width)
            left = int(uniform(0, length - buffer))
            right = int(left + uniform(int(length * self.min_chunk_width), buffer))
            return left, min(right, length)

        # This is applied over the whole mini-batch, following torchvision
        a, b = random_range()
        x[..., a:b] = self.replacement(x, a=a, b=b)
        return x


class RandomReplaceZero(_RandomReplace):
    """Randomly replace part of a tensor with zero."""

    def replacement(
        self,
        x: torch.Tensor,
        a: int,
        b: int,
    ) -> Union[float, torch.Tensor]:
        """Generate replacement (zero).

        Args:
            x (torch.Tensor): tensor to operate on. Should be of the
                form ``[BATCH, FEATURES]``.
            a (float): start position in the tensor
            b (float): end position in the tensor

        Returns:
            torch.Tensor

        """
        return 0.0


class RandomReplaceMean(_RandomReplace):
    """Randomly replace part of a tensor with its mean."""

    def replacement(
        self,
        x: torch.Tensor,
        a: int,
        b: int,
    ) -> Union[float, torch.Tensor]:
        """Generate replacement (mean of each batch).

        Args:
            x (torch.Tensor): tensor to operate on. Should be of the
                form ``[BATCH, FEATURES]``.
            a (float): start position in the tensor
            b (float): end position in the tensor

        Returns:
            torch.Tensor

        """
        return x.mean(dim=-1).repeat_interleave(b - a).view(-1, b - a)


class RandomNoise(RandomOp):
    """Add random noise to a signal.

    Args:
        alpha (tuple): a tuple to characterize a uniform distribution.
            Values drawn from this distribution will determine the
            weight given to the random noise.
        **kwargs (Keyword Args): keyword arguments to pass to the
            parent class.

    """

    def __init__(self, alpha: Tuple[float, float] = (0.01, 0.1), **kwargs: Any):
        super().__init__(**kwargs)
        self.alpha = alpha

    def op(self, x: torch.Tensor) -> torch.Tensor:
        """Add random noise to ``x``

        Args:
            x (torch.Tensor): a tensor to operate on

        Returns:
            x_fuzzed (torch.Tensor): x + noise.

        """
        noise_weight = np.random.uniform(*self.alpha)
        return x + torch.rand_like(x) * noise_weight


class Resize(nn.Module):
    """Resize a tensor.

    Args:
        size (int, tuple): one or more integers
        mode (str): resizing algorithm to use

    """

    def __init__(
        self,
        size: Union[int, Tuple[int, ...]],
        mode: str = "nearest",
    ) -> None:
        super().__init__()
        self.size = size
        self.mode = mode

    def __repr__(self) -> str:
        return auto_repr(self, size=self.size, mode=self.mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Resize ``x`` to ``size``.

        Args:
            x (torch.Tensor): a tensor of the form ``[BATCH, ...]``.

        Returns:
            x_resized (torch.Tensor): ``x`` resized

        """
        return F.interpolate(x, size=self.size, mode=self.mode)
