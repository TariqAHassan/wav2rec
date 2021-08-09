"""

    Similarity

"""
from typing import Callable

import numba
import numpy as np


@numba.jit(nopython=True)
def _clip(value: float, a_min: float, a_max: float) -> float:
    return max(min(value, a_max), a_min)


@numba.jit(nopython=True)
def cosine_similarity(x1: np.ndarray, x2: np.ndarray) -> float:
    """Compute cosine similarity between two 1D arrays.

    Args:
        x1 (np.ndarray): a 1D array with shape ``[FEATURES]``
        x2 (np.ndarray): a 1D array with shape ``[FEATURES]``

    Returns:
        similarity (float): a similarity score on [0, 1].

    Warning:
        * ``x1`` and ``x2`` must be normalized.

    """
    return float(_clip(x1 @ x2, a_min=0, a_max=1))


@numba.jit(nopython=True)
def similarity_calculator(
    X_query: np.ndarray,
    X_neighbours: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float] = cosine_similarity,
) -> np.ndarray:
    """Compute the similarity of ``X_query`` with all entries in ``X_neighbours``.

    Args:
        X_query (np.ndarray): a query 2D array with shape ``[N_QUERIES, FEATURES]``
        X_neighbours (np.ndarray): a reference 2D array with shape
            ``[N_QUERIES, N_NEIGHBOURS, FEATURES]``
        metric (callable): a callable which accepts two 1D arrays
            and returns a float. Must be compiled with ``numba.jit(nopython=True)``.

    Returns:
        sims (np.ndarray): a 2D array of similarities with shape ``[N_QUERIES, N_NEIGHBOURS]``.

    """
    n_queries = X_query.shape[0]
    n_neighbours = X_neighbours.shape[1]

    sims = np.zeros((n_queries, n_neighbours), dtype=X_neighbours.dtype)
    for i in range(n_queries):
        for j in range(n_neighbours):
            sims[i, j] = metric(X_query[i], X_neighbours[i, j])
    return sims
