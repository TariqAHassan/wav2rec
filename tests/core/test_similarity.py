"""

    Test Similarity

"""
import numpy as np
import pytest

from wav2rec.core.similarity import cosine_similarity, similarity_calculator


@pytest.mark.parametrize(
    "x1,x2,expected",
    [
        (np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0.0),
        (np.array([0.0, 0.5]), np.array([0.0, 1.0]), 0.5),
        (np.array([1.0, 0.0]), np.array([1.0, 0.0]), 1.0),
    ],
)
def test_cosine_similarity(x1: np.ndarray, x2: np.ndarray, expected: float) -> None:
    sim = cosine_similarity(x1, x2)
    assert np.isclose(sim, expected)


@pytest.mark.parametrize(
    "X_query,X_neighbours,expected",
    [
        (np.array([[0.0, 0.0]]), np.array([[[0.0, 0.0]]]), np.array([[0.0]])),
        (np.array([[0.0, 0.5]]), np.array([[[0.0, 1.0]]]), np.array([[0.5]])),
        (np.array([[1.0, 0.0]]), np.array([[[1.0, 0.0]]]), np.array([[1.0]])),
    ],
)
def test_similarity_calculator(
    X_query: np.ndarray,
    X_neighbours: np.ndarray,
    expected: np.ndarray,
) -> None:
    sims = similarity_calculator(X_query, X_neighbours)
    assert np.isclose(sims, expected).all()
