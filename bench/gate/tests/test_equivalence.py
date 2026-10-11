"""Equivalence of a start, including a periodic cell and another basin."""

import numpy as np

from rggate.contract import parse_csv
from rggate.equivalence import energy_window, kabsch_rmsd
from rggate.pairing import classify_pair, gehan


def test_minimum_image_free_rotation_and_periodic_rotation():
    cell = np.diag([10.0, 10.0, 10.0])
    reference = np.array(
        [
            [0.1, 0.0, 0.0],
            [1.0, 2.0, 0.2],
            [0.2, -0.4, 0.5],
            [-0.3, 0.4, 1.2],
        ]
    )
    wrapped = reference + np.array([10.0, 0.0, 0.0])
    assert kabsch_rmsd(reference, wrapped, cell) < 1e-8
    theta = np.pi / 3.0
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated = reference @ rotation.T
    assert kabsch_rmsd(reference, rotated, None) < 1e-8
    assert kabsch_rmsd(reference, rotated, cell) > 0.05


def test_energy_window_uses_the_force_bound():
    # |dE| <= N * fmax * RMSD, added to e_tol.
    window = energy_window(1e-4, 0.05, 38, 0.01, 0.008)
    assert window == pytest_approx(1e-4 + 38 * 0.01 * 0.05)


def pytest_approx(value):
    import pytest

    return pytest.approx(value)


def test_gehan_censoring_and_split():
    assert gehan(False, 5.0, True, 40.0) == "indeterminate"
    assert gehan(True, 5.0, False, 40.0) == "win"
    assert gehan(False, 5.0, False, 40.0) == "indeterminate"
    text = "\n".join(
        [
            "system,method,seed,converged,iters,calls,energy,fmax,wall_s",
            "mol,cand,0,1,1,10,-1.0,1e-3,0.1",
            "mol,base,0,1,1,12,-1.0,1e-3,0.1",
            "mol,cand,1,1,1,10,-1.0,1e-3,0.1",
            "mol,base,1,1,1,11,-2.0,1e-3,0.1",
        ]
    )
    rows = parse_csv(text)
    records = classify_pair(rows, "cand", "base", 1e-4, 0.05, "auto")
    kinds = {(record.seed, record.kind) for record in records}
    assert (0, "win") in kinds
    assert (1, "split") in kinds
    assert all(record.energy_only for record in records)


def test_numeric_reference_overrides_the_pair():
    text = "\n".join(
        [
            "system,method,seed,converged,iters,calls,energy,fmax,wall_s,reference",
            "mol,cand,0,1,1,10,-9.5,1e-8,0.1,-9.5",
            "mol,base,0,1,1,12,-10.0,1e-8,0.1,",
        ]
    )
    rows = parse_csv(text)
    records = classify_pair(rows, "cand", "base", 1e-4, 0.05, "auto")
    assert len(records) == 1
    assert records[0].kind == "split"


def test_unknown_major_is_rejected():
    from rggate.contract import ContractError

    header = "system,method,seed,converged,iters,calls,energy,fmax,wall_s"
    try:
        parse_csv("# rggate-csv 2\n" + header + "\n")
    except ContractError as exc:
        assert "major 2" in str(exc)
    else:
        raise AssertionError("major 2 was accepted")
    rows = parse_csv("# rggate-csv 1.2\n" + header + "\nmol,a,0,1,1,10,-1,1e-4,0.1\n")
    assert len(rows) == 1
