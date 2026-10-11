"""Equivalence, including a converged run in another basin."""

import numpy as np

from rggate.contract import Row, parse_csv
from rggate.equivalence import attach_references, kabsch_rmsd


def _row(method, energy, converged=True, endpoint=None, reference=None):
    return Row(
        system="mol",
        method=method,
        seed=0,
        converged=converged,
        iters=1,
        calls=10.0,
        energy=energy,
        fmax=1e-3,
        wall_s=0.1,
        pot_s=None,
        overhead_us_per_iter=None,
        endpoint=endpoint,
        reference=reference,
        dim=None,
        threads=None,
        line=1,
    )


def test_minimum_image_and_rotation():
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
    assert kabsch_rmsd(reference, reference @ rotation.T, None) < 1e-8
    mirrored = reference.copy()
    mirrored[:, 0] *= -1.0
    assert kabsch_rmsd(reference, mirrored, None) > 0.05


def test_mislabelled_basin(tmp_path):
    reference = np.array([[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [0.1, 1.0, 0.3]])
    theta = 0.4
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated = reference @ rotation.T
    distorted = reference.copy()
    distorted[0, 0] += 1.0

    def write(path, energy, coords):
        body = [f"energy {energy}"]
        body.extend(" ".join(str(v) for v in row) for row in coords)
        path.write_text("\n".join(body) + "\n", encoding="utf-8")

    ref_path = tmp_path / "ref.xyz"
    rot_path = tmp_path / "rot.xyz"
    bad_path = tmp_path / "bad.xyz"
    write(ref_path, -10.0, reference)
    write(rot_path, -10.0, rotated)
    write(bad_path, -10.0, distorted)

    rows = [
        _row("deep", -10.0, endpoint=str(ref_path)),
        _row("rotated", -10.0, endpoint=str(rot_path)),
        _row("other-basin", -9.5, endpoint=str(ref_path)),
        _row("distorted", -10.0, endpoint=str(bad_path)),
        _row("budget", -8.0, converged=False, endpoint=str(bad_path)),
    ]
    status = attach_references(rows, e_tol=1e-4, rmsd_tol=0.05)
    assert status == ["success", "success", "different", "different", "censored"]


def test_numeric_reference_overrides_the_minimum():
    text = "\n".join(
        [
            "system,method,seed,converged,iters,calls,energy,fmax,wall_s,reference",
            "mol,deep,0,1,1,10,-10.0,1e-4,0.1,",
            "mol,labelled,0,1,1,10,-9.5,1e-4,0.1,-9.5",
            "mol,unlabelled,0,1,1,10,-9.5,1e-4,0.1,",
        ]
    )
    rows = parse_csv(text)
    assert attach_references(rows, 1e-4, 0.05) == ["success", "success", "different"]


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
