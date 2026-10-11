"""Per-start basin labels and Gehan scores.

Precedence for the reference on one ``(system, seed)``, inside a pair:
an explicit ``reference`` (energy or endpoint path), then the baseline
endpoint from that same start when the baseline converged, then the
lowest converged energy on the system when ``--reference lowest`` is
set. Otherwise the two runs are compared with each other.

Both runs converging in the same basin is an informative pair. Both
converging in different basins is a basin split: it is counted, given
a Clopper-Pearson interval by the caller, and left out of the call
comparison. It is not a loss for either side.

Gehan scoring uses the remaining starts. The candidate wins when it
reached the basin with fewer calls than the baseline's observed count,
whether that count is a later success or a later censoring. A loss is
the mirror image. Both unsolved, a tie, or a censoring earlier than the
other side's success is indeterminate and drops out of the walk.
"""

from dataclasses import dataclass
import math

from rggate.contract import load_structure
from rggate.equivalence import atom_count, energy_window, kabsch_rmsd


@dataclass
class StartRecord:
    system: str
    seed: int
    kind: str
    log_ratio: float | None
    energy_only: bool
    window_fallback: bool


def gehan(candidate_event, candidate_time, baseline_event, baseline_time):
    """Return ``win``, ``loss``, or ``indeterminate``."""
    if candidate_event and baseline_event:
        if candidate_time < baseline_time:
            return "win"
        if candidate_time > baseline_time:
            return "loss"
        return "indeterminate"
    if candidate_event and not baseline_event and candidate_time < baseline_time:
        return "win"
    if baseline_event and not candidate_event and baseline_time < candidate_time:
        return "loss"
    return "indeterminate"


def classify_pair(rows, candidate, baseline, e_tol, rmsd_tol, mode):
    """Label every start where both methods have a row."""
    left = {}
    right = {}
    lowest = {}
    for row in rows:
        key = (row.system, row.seed)
        if row.method == candidate:
            left[key] = row
        elif row.method == baseline:
            right[key] = row
        if row.converged and math.isfinite(row.energy):
            current = lowest.get(row.system)
            if current is None or row.energy < current:
                lowest[row.system] = row.energy
    cache = {}
    records = []
    for key in sorted(left):
        if key not in right:
            continue
        records.append(
            _one(
                left[key],
                right[key],
                e_tol,
                rmsd_tol,
                mode,
                lowest.get(key[0]),
                cache,
            )
        )
    return records


def _one(cand, base, e_tol, rmsd_tol, mode, lowest_energy, cache):
    target, energy_only = _target(cand, base, mode, lowest_energy, cache, e_tol, rmsd_tol)
    if target is None:
        return _pairwise(cand, base, e_tol, rmsd_tol, cache)
    cand_hit = _hit(cand, base, target, e_tol, rmsd_tol, cache)
    base_hit = _hit(base, cand, target, e_tol, rmsd_tol, cache)
    used_geometry = target["positions"] is not None and (
        cand.endpoint is not None or base.endpoint is not None
    )
    fallback = _uses_fallback(cand, base, target, cache) or _uses_fallback(base, cand, target, cache)
    if _converged(cand) and _converged(base) and not (cand_hit and base_hit):
        if cand_hit != base_hit:
            return _record(cand, "split", None, not used_geometry, fallback)
        same, geom, pair_fallback = _pairwise_same(cand, base, e_tol, rmsd_tol, cache)
        if not same:
            return _record(cand, "split", None, not geom, fallback or pair_fallback)
        return _record(cand, "indeterminate", None, not geom, fallback or pair_fallback)
    if cand_hit and base_hit:
        ratio = _ratio(cand, base)
        kind = gehan(True, cand.calls, True, base.calls)
        return _record(cand, kind, ratio, not used_geometry and energy_only, fallback)
    kind = gehan(cand_hit, cand.calls, base_hit, base.calls)
    return _record(cand, kind, None, True, fallback)


def _pairwise(cand, base, e_tol, rmsd_tol, cache):
    cand_event = _converged(cand)
    base_event = _converged(base)
    if cand_event and base_event:
        same, geom, fallback = _pairwise_same(cand, base, e_tol, rmsd_tol, cache)
        if not same:
            return _record(cand, "split", None, not geom, fallback)
        return _record(
            cand,
            gehan(True, cand.calls, True, base.calls),
            _ratio(cand, base),
            not geom,
            fallback,
        )
    return _record(
        cand,
        gehan(cand_event, cand.calls, base_event, base.calls),
        None,
        True,
        False,
    )


def _pairwise_same(cand, base, e_tol, rmsd_tol, cache):
    positions = _positions(cand, cache) or _positions(base, cache)
    n_atoms = atom_count((cand, base), positions[0] if positions else None)
    fallback = n_atoms is None
    window = energy_window(e_tol, rmsd_tol, n_atoms, cand.fmax, base.fmax)
    if abs(cand.energy - base.energy) > window:
        return False, False, fallback
    cand_xyz = _positions(cand, cache)
    base_xyz = _positions(base, cache)
    if cand_xyz is None or base_xyz is None:
        return True, False, fallback
    rmsd = kabsch_rmsd(base_xyz[0], cand_xyz[0], cand_xyz[1] if cand_xyz[1] is not None else base_xyz[1])
    return rmsd <= rmsd_tol, True, fallback


def _uses_fallback(row, other, target, cache):
    """True when a converged run was judged with the ``e_tol`` window."""
    if not _converged(row):
        return False
    own = _positions(row, cache)
    positions = own[0] if own is not None else None
    site = target["positions"] if target["positions"] is not None else positions
    return atom_count((row, other), site) is None


def _target(cand, base, mode, lowest_energy, cache, e_tol, rmsd_tol):
    del e_tol, rmsd_tol
    explicit = cand.reference or base.reference
    if explicit:
        numeric = _as_float(explicit)
        if numeric is None:
            energy, cell, positions = _load(explicit, cache)
            return {"energy": energy, "positions": positions, "cell": cell}, False
        return {"energy": numeric, "positions": None, "cell": None}, True
    if base.endpoint and _converged(base):
        energy, cell, positions = _load(base.endpoint, cache)
        if energy is None:
            energy = base.energy
        return {"energy": energy, "positions": positions, "cell": cell}, False
    if mode == "lowest" and lowest_energy is not None:
        return {"energy": lowest_energy, "positions": None, "cell": None}, True
    return None, True


def _hit(row, other, target, e_tol, rmsd_tol, cache):
    if not _converged(row):
        return False
    positions = None
    cell = target["cell"]
    own = _positions(row, cache)
    if own is not None:
        positions = own[0]
        if cell is None:
            cell = own[1]
    n_atoms = atom_count((row, other), target["positions"] if target["positions"] is not None else positions)
    window = energy_window(e_tol, rmsd_tol, n_atoms, row.fmax, other.fmax)
    if target["energy"] is not None and abs(row.energy - target["energy"]) > window:
        return False
    if target["positions"] is not None:
        if own is None:
            return target["energy"] is not None
        rmsd = kabsch_rmsd(target["positions"], own[0], cell)
        return rmsd <= rmsd_tol
    return target["energy"] is not None


def _positions(row, cache):
    if not row.endpoint:
        return None
    _energy, cell, positions = _load(row.endpoint, cache)
    return positions, cell


def _load(path, cache):
    if path not in cache:
        cache[path] = load_structure(path)
    return cache[path]


def _converged(row):
    return bool(row.converged) and math.isfinite(row.energy) and math.isfinite(row.calls)


def _ratio(cand, base):
    if cand.calls <= 0.0 or base.calls <= 0.0:
        return None
    return math.log(cand.calls / base.calls)


def _record(row, kind, log_ratio, energy_only, window_fallback):
    return StartRecord(row.system, row.seed, kind, log_ratio, energy_only, window_fallback)


def _as_float(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def row_succeeded(row, mode, lowest_energy, cache, e_tol, rmsd_tol):
    """Event indicator for per-method curves.

    Without an explicit reference and without ``--reference lowest``,
    success is the run's own convergence flag. An explicit reference,
    or the lowest-energy option, requires a match to that energy.
    """
    if not _converged(row):
        return False
    if row.reference:
        numeric = _as_float(row.reference)
        if numeric is None:
            energy, cell, positions = _load(row.reference, cache)
            target = {"energy": energy, "positions": positions, "cell": cell}
        else:
            target = {"energy": numeric, "positions": None, "cell": None}
        return _hit(row, row, target, e_tol, rmsd_tol, cache)
    if mode == "lowest":
        if lowest_energy is None:
            return False
        target = {"energy": lowest_energy, "positions": None, "cell": None}
        return _hit(row, row, target, e_tol, rmsd_tol, cache)
    return True
