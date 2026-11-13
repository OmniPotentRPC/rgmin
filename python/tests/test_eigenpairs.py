"""Block LOBPCG matches a dense symmetric eigensolve."""

import numpy as np
import pytest

from rgmin import lowest_eigenpairs


def test_random_spd_matches_numpy_eigh():
    rng = np.random.default_rng(11)
    factor = rng.normal(size=(6, 6))
    matrix = factor.T @ factor
    seed = np.eye(6, 3)
    seed[1:, :] += 0.3
    values, vectors = lowest_eigenpairs(
        lambda _x, vector: matrix @ vector, seed, tol=1e-10, max_iter=40
    )
    evals, evecs = np.linalg.eigh(matrix)
    assert values.shape == (3,)
    assert vectors.shape == (6, 3)
    assert np.allclose(values, evals[:3], atol=1e-5)
    overlap = np.sum((evecs[:, :3].T @ vectors) ** 2)
    assert abs(overlap - 3.0) < 1e-4


def test_degenerate_subspace_matches_numpy_eigh():
    first = np.array([1.0, 1.0, 1.0])
    first /= np.linalg.norm(first)
    second = np.array([1.0, -1.0, 0.0])
    second -= first * np.dot(first, second)
    second /= np.linalg.norm(second)
    third = np.cross(first, second)
    matrix = np.eye(3) + 3.0 * np.outer(third, third)
    seed = np.array([[0.2, 0.9], [-0.4, 0.1], [0.7, -0.3]])
    values, vectors = lowest_eigenpairs(
        lambda _x, vector: matrix @ vector, seed, tol=1e-10, max_iter=30
    )
    evals, evecs = np.linalg.eigh(matrix)
    assert np.allclose(values, evals[:2], atol=1e-6)
    overlap = np.sum((evecs[:, :2].T @ vectors) ** 2)
    assert abs(overlap - 2.0) < 1e-5


def test_nonconvergence_raises():
    rng = np.random.default_rng(19)
    raw = rng.normal(size=(8, 8))
    matrix = 0.5 * (raw + raw.T)
    seed = np.eye(8, 3)
    with pytest.raises(RuntimeError, match="did not meet the residual tolerance"):
        lowest_eigenpairs(
            lambda _x, vector: matrix @ vector, seed, tol=1e-14, max_iter=1
        )


def test_bad_block_and_callback_errors_are_exceptions():
    with pytest.raises(ValueError):
        lowest_eigenpairs(lambda _x, vector: vector, np.zeros((3, 0)))
    with pytest.raises(ValueError):
        lowest_eigenpairs(lambda _x, vector: vector, np.array([[np.nan], [0.0], [1.0]]))

    def broken(_x, _vector):
        raise RuntimeError("hvp failed")

    with pytest.raises(RuntimeError, match="hvp failed"):
        lowest_eigenpairs(broken, np.eye(3, 1))
