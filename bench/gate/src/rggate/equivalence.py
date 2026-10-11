"""Same stationary point: energy window and aligned RMSD.

A free cluster is aligned with the Kabsch rotation. A periodic cell is
not rotated. After the minimum-image map only the mean displacement is
removed, because a rotation is a symmetry of an isolated cluster and
not of a crystal.

The energy window is ``e_tol`` when the atom count ``N`` is unknown.
``dim`` is the free coordinate count, so ``N = dim / 3`` (frozen atoms
are excluded, because ``fmax`` covers only moving atoms). ``N`` is also
the number of sites in an endpoint file. A split counted with unknown
``N`` is an upper bound: the force-based window is wider, so some of
those splits can be the same stationary point. When ``N`` is known the
window is

    e_tol + N * max(fmax_candidate, fmax_baseline) * rmsd_tol

from ``|dE| <= ||grad E||_2 ||dx||_2``, with ``||grad E||_2 <= sqrt(N)
fmax`` and ``||dx||_2 = sqrt(N) * RMSD``. ``fmax`` is the largest
absolute force component reported for the run.
"""

import math

import numpy as np


def energy_window(e_tol, rmsd_tol, n_atoms, fmax_a, fmax_b):
    """Absolute energy tolerance for one pair of runs."""
    if n_atoms is None or n_atoms <= 0:
        return e_tol
    fmax = max(_finite(fmax_a), _finite(fmax_b))
    return e_tol + float(n_atoms) * fmax * rmsd_tol


def atom_count(rows, positions=None):
    """``N`` from an endpoint, otherwise ``dim / 3`` when that divides."""
    if positions is not None:
        return int(np.asarray(positions).shape[0])
    for row in rows:
        if row.dim is not None and row.dim > 0 and row.dim % 3 == 0:
            return row.dim // 3
    return None


def kabsch_rmsd(reference, candidate, cell=None):
    """RMSD after the minimum-image map.

    Atoms are paired by index. ``cell`` is 3x3 with rows as lattice
    vectors. A missing cell is a free cluster: the candidate is rotated
    onto the reference. A cell suppresses that rotation.
    """
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if reference.shape != candidate.shape or reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("positions must be matching (N, 3) arrays")
    delta = candidate - reference
    if cell is not None:
        lattice = np.asarray(cell, dtype=float).reshape(3, 3)
        fractional = delta @ np.linalg.inv(lattice)
        fractional -= np.round(fractional)
        delta = fractional @ lattice
        delta -= delta.mean(axis=0)
        return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
    moved = reference + delta
    p = reference - reference.mean(axis=0)
    q = moved - moved.mean(axis=0)
    covariance = q.T @ p
    u, _s, vt = np.linalg.svd(covariance)
    if np.linalg.det(u) * np.linalg.det(vt) < 0.0:
        u = u.copy()
        u[:, -1] *= -1.0
    rotation = u @ vt
    aligned = q @ rotation
    residual = aligned - p
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number) or number < 0.0:
        return 0.0
    return number
