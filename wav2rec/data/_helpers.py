"""

    Data Helpers

"""
import numpy as np


def zero_pad1d(x: np.ndarray, target_length: int) -> np.ndarray:
    """Pad ``x`` with zeros s.t. its length equals ``target_length``.

    Args:
        x (np.ndarray): the array to pad
        target_length (int): the desired length of ``x``

    Returns:
        x_padded (np.ndarray): ``x`` padded to ``target_length``

    """
    if x.ndim != 1:
        raise IndexError(f"`x` is not 1D, got {x.ndim}D")

    if len(x) == target_length:
        return x
    elif len(x) < target_length:
        padding = np.zeros((target_length - len(x)), dtype=x.dtype)
        x_padded: np.ndarray = np.concatenate((x, padding), axis=-1)
        return x_padded
    else:
        raise IndexError("`x` is longer than `target_length`")
