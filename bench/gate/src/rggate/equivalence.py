"""Same stationary point: energy window and Kabsch RMSD.

A converged run matches the per-system reference when
``|E - E_ref| <= e_tol`` (default ``1e-4`` eV). If both geometries are
present, the Kabsch RMSD after the minimum-image map must also be at
most ``rmsd_tol`` (default ``0.05`` Angstrom). A converged run that
fails either test reached a different stationary point. An unconverged
run is censored and is not counted as a different stationary point.
"""

import numpy as np

from rggate.contract import load_structure


def kabsch_rmsd(reference, candidate, cell=None):
    """RMSD after minimum-image wrapping and a proper rotation.

    Atoms are paired by index. ``cell`` is 3x3 with rows as lattice
    vectors. A missing cell is a free cluster. Translation is removed
    by centering, so a rigid shift of a free cluster has RMSD 0.
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
    moved = reference + delta
    p = reference - reference.mean(axis=0)
    q = moved - moved.mean(axis=0)
    # Rotate the candidate onto the reference. Rows are atoms.
    covariance = q.T @ p
    u, _s, vt = np.linalg.svd(covariance)
    if np.linalg.det(u) * np.linalg.det(vt) < 0.0:
        u = u.copy()
        u[:, -1] *= -1.0
    rotation = u @ vt
    aligned = q @ rotation
    residual = aligned - p
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


def _as_float(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def attach_references(rows, e_tol, rmsd_tol):
    """Return per-row status ``success``, ``different``, or ``censored``.

    Without a ``reference`` column the reference energy is the lowest
    converged energy on that system. A numeric reference replaces that
    energy for the row. A reference path supplies energy and coordinates.
    """
    best = {}
    for row in rows:
        if row.converged and np.isfinite(row.energy):
            current = best.get(row.system)
            if current is None or row.energy < current[0]:
                best[row.system] = (row.energy, row)

    cache = {}

    def structure(path):
        if path not in cache:
            cache[path] = load_structure(path)
        return cache[path]

    # Geometry of the run that holds the reference energy, when it wrote one.
    ref_geom = {}
    for system, (energy, row) in best.items():
        if row.endpoint:
            _e, cell, pos = structure(row.endpoint)
            ref_geom[system] = (pos, cell, energy)

    status = []
    for row in rows:
        if not row.converged or not np.isfinite(row.energy):
            status.append("censored")
            continue
        ref_energy = best[row.system][0] if row.system in best else np.nan
        ref_pos = None
        ref_cell = None
        if row.system in ref_geom:
            ref_pos, ref_cell, _ = ref_geom[row.system]
        if row.reference:
            numeric = _as_float(row.reference)
            if numeric is None:
                energy, cell, pos = structure(row.reference)
                if energy is not None:
                    ref_energy = energy
                ref_pos, ref_cell = pos, cell
            else:
                ref_energy = numeric
        if not np.isfinite(ref_energy) or abs(row.energy - ref_energy) > e_tol:
            status.append("different")
            continue
        if row.endpoint and ref_pos is not None:
            _e, cell, pos = structure(row.endpoint)
            # The row's own cell, if written, overrides the reference cell.
            use_cell = cell if cell is not None else ref_cell
            if kabsch_rmsd(ref_pos, pos, use_cell) > rmsd_tol:
                status.append("different")
                continue
        status.append("success")
    return status
