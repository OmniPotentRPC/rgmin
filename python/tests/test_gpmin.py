"""Tests for rgmin.optimize.GPMin and parity against ase.optimize.GPMin."""

from __future__ import annotations

import inspect
import time
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.optimize import GPMin as AseGPMin
from ase.optimize.gpmin.kernel import SquaredExponential
from ase.optimize.gpmin.prior import ConstantPrior

import rgmin
from rgmin._maturin import Session
from rgmin.optimize import GPMin


class CountingLJ(LennardJones):
    """Lennard-Jones calculator that counts force/energy evaluations."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = 0

    def calculate(self, atoms=None, properties=("energy",), system_changes=None):
        super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)
        self.calls += 1


def _lj7_cluster() -> Atoms:
    """Perturbed 7-atom pentagonal-bipyramid LJ cluster (sigma=2.5 A, eps=0.1 eV)."""
    sigma = 2.5
    r_eq = (2.0 ** (1.0 / 6.0)) * sigma
    ring_r = 0.85 * r_eq
    apex_z = 0.65 * r_eq
    positions = [[0.0, 0.0, apex_z], [0.0, 0.0, -apex_z]]
    for k in range(5):
        angle = 2.0 * np.pi * k / 5.0
        positions.append([ring_r * np.cos(angle), ring_r * np.sin(angle), 0.0])
    pos = np.asarray(positions, dtype=np.float64)
    # Deterministic asymmetric perturbation so the cluster starts away from D5h.
    perturb = np.array(
        [
            [0.09, -0.05, 0.08],
            [-0.06, 0.07, -0.09],
            [0.08, 0.04, 0.05],
            [-0.07, 0.06, -0.04],
            [0.05, -0.08, 0.06],
            [-0.04, -0.06, -0.05],
            [0.06, 0.05, -0.03],
        ],
        dtype=np.float64,
    )
    return Atoms("Ar7", positions=pos + perturb)


def test_gpmin_constructor_matches_ase_signature_and_defaults():
    ase_params = [
        name
        for name, param in inspect.signature(AseGPMin.__init__).parameters.items()
        if name != "self" and param.kind != inspect.Parameter.VAR_KEYWORD
    ]
    rgmin_params = [
        name
        for name, param in inspect.signature(GPMin.__init__).parameters.items()
        if name != "self"
        and param.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    assert rgmin_params == ase_params

    atoms = _lj7_cluster()
    atoms.calc = LennardJones(sigma=2.5, epsilon=0.1, rc=12.0)

    opt_fixed = GPMin(atoms, logfile=None)
    assert opt_fixed.scale == pytest.approx(0.4)
    assert opt_fixed.weight == pytest.approx(1.0)
    assert opt_fixed.noise == pytest.approx(0.001)
    assert opt_fixed.batch_size is None
    assert opt_fixed.bounds is None
    assert opt_fixed.update_prior_strategy == "maximum"
    assert opt_fixed.update_hyperparams is False

    opt_upd = GPMin(atoms, logfile=None, update_hyperparams=True)
    assert opt_upd.scale == pytest.approx(0.3)
    assert opt_upd.weight == pytest.approx(2.0)
    assert opt_upd.noise == pytest.approx(0.004)
    assert opt_upd.batch_size == 1
    assert opt_upd.bounds == pytest.approx(0.1)
    assert opt_upd.update_hyperparams is True

    assert rgmin.GPMin is GPMin


def test_gpmin_accepts_ase_prior_and_kernel_objects():
    atoms = _lj7_cluster()
    atoms.calc = LennardJones(sigma=2.5, epsilon=0.1, rc=12.0)
    kernel = SquaredExponential()
    kernel.set_params([1.5, 0.5])
    opt = GPMin(
        atoms,
        logfile=None,
        prior=ConstantPrior(0.0),
        kernel=kernel,
    )
    assert opt.weight == pytest.approx(1.5)
    assert opt.scale == pytest.approx(0.5)
    assert bool(opt.run(fmax=0.05, steps=40)) is True


def test_gpmin_warnings_and_fail_closed_validation():
    atoms = _lj7_cluster()
    atoms.calc = LennardJones(sigma=2.5, epsilon=0.1, rc=12.0)

    with pytest.warns(UserWarning, match="batch_size"):
        GPMin(atoms, logfile=None, batch_size=2, update_hyperparams=False)
    with pytest.warns(UserWarning, match="bounds"):
        GPMin(atoms, logfile=None, bounds=0.1, update_hyperparams=False)

    big = Atoms("Ar101", positions=np.zeros((101, 3)))
    with pytest.warns(UserWarning, match="100 atoms"):
        GPMin(big, logfile=None)

    with pytest.raises(ValueError, match="prior"):
        GPMin(atoms, logfile=None, prior="bad")
    with pytest.raises(ValueError, match="kernel"):
        GPMin(atoms, logfile=None, kernel="bad")
    with pytest.raises(ValueError, match="bounds"):
        GPMin(atoms, logfile=None, bounds=True, update_hyperparams=True)

    # Invalid hyperparameters fail closed when the native session opens.
    with pytest.raises(ValueError):
        GPMin(atoms, logfile=None, scale=-0.1).run(fmax=0.05, steps=1)
    with pytest.raises(ValueError):
        GPMin(atoms, logfile=None, weight=0.0).run(fmax=0.05, steps=1)
    with pytest.raises(ValueError):
        GPMin(atoms, logfile=None, noise=0.0).run(fmax=0.05, steps=1)
    with pytest.raises(ValueError):
        GPMin(atoms, logfile=None, update_prior_strategy="fit").run(fmax=0.05, steps=1)
    with pytest.raises(ValueError):
        GPMin(atoms, logfile=None, update_hyperparams=True, bounds=1.0).run(fmax=0.05, steps=1)

    # GPMin-specific options are refused on non-GPMin Session kinds.
    with pytest.raises(ValueError, match="gpmin"):
        Session("lbfgs", 3, scale=0.4)


def test_gpmin_run_returns_convergence_flag_and_restarts(tmp_path: Path):
    atoms = _lj7_cluster()
    atoms.calc = LennardJones(sigma=2.5, epsilon=0.1, rc=12.0)
    opt_short = GPMin(atoms.copy(), logfile=None)
    opt_short.atoms.calc = LennardJones(sigma=2.5, epsilon=0.1, rc=12.0)
    assert bool(opt_short.run(fmax=1e-4, steps=1)) is False

    # Uninterrupted run vs break-and-restart run agree identically.
    single_atoms = _lj7_cluster()
    single_atoms.calc = LennardJones(sigma=2.5, epsilon=0.1, rc=12.0)
    single_restart = tmp_path / "single.json"
    with GPMin(single_atoms, restart=single_restart, logfile=None) as single:
        assert bool(single.run(fmax=0.01, steps=60)) is True
    e_single = single_atoms.get_potential_energy()

    split_atoms = _lj7_cluster()
    split_atoms.calc = LennardJones(sigma=2.5, epsilon=0.1, rc=12.0)
    split_restart = tmp_path / "split.json"
    with GPMin(split_atoms, restart=split_restart, logfile=None) as part1:
        part1.run(fmax=0.01, steps=3)
    with GPMin(split_atoms, restart=split_restart, logfile=None) as part2:
        assert bool(part2.run(fmax=0.01, steps=60)) is True
    assert split_atoms.get_potential_energy() == pytest.approx(e_single, abs=1e-10)
    assert single.nsteps == part1.nsteps + part2.nsteps


@pytest.mark.parametrize("update_hyperparams", [False, True])
def test_gpmin_parity_against_ase_on_lj7(update_hyperparams: bool, capsys):
    """Parity test on a 7-atom LJ cluster with identical hyperparameters."""
    fmax = 1e-3
    lj_kwargs = {"sigma": 2.5, "epsilon": 0.1, "rc": 12.0}
    if update_hyperparams:
        gp_kwargs = {
            "scale": 0.3,
            "weight": 2.0,
            "noise": 0.004,
            "batch_size": 2,
            "bounds": 0.1,
            "update_prior_strategy": "maximum",
            "update_hyperparams": True,
        }
    else:
        gp_kwargs = {
            "scale": 0.4,
            "weight": 1.0,
            "noise": 0.001,
            "update_prior_strategy": "maximum",
            "update_hyperparams": False,
        }

    # 1. rgmin.optimize.GPMin (uncapped maxstep=0.0 to match ASE GPMin)
    atoms_rg = _lj7_cluster()
    calc_rg = CountingLJ(**lj_kwargs)
    atoms_rg.calc = calc_rg
    t0 = time.perf_counter()
    opt_rg = GPMin(atoms_rg, logfile=None, maxstep=0.0, **gp_kwargs)
    conv_rg = opt_rg.run(fmax=fmax, steps=120)
    wall_rg = time.perf_counter() - t0
    e_rg = float(atoms_rg.get_potential_energy())
    fmax_rg = float(np.max(np.linalg.norm(atoms_rg.get_forces(), axis=1)))

    # 2. ase.optimize.GPMin
    atoms_ase = _lj7_cluster()
    calc_ase = CountingLJ(**lj_kwargs)
    atoms_ase.calc = calc_ase
    t0 = time.perf_counter()
    opt_ase = AseGPMin(atoms_ase, logfile=None, **gp_kwargs)
    conv_ase = opt_ase.run(fmax=fmax, steps=120)
    wall_ase = time.perf_counter() - t0
    e_ase = float(atoms_ase.get_potential_energy())
    fmax_ase = float(np.max(np.linalg.norm(atoms_ase.get_forces(), axis=1)))

    assert bool(conv_rg) is True
    assert bool(conv_ase) is True
    assert fmax_rg <= fmax
    assert fmax_ase <= fmax

    # Energy agreement tolerance derived from fmax (quadratic near a minimum:
    # delta E = O(fmax^2 / k_min); use 10 * fmax^2 <= fmax for fmax = 1e-3).
    e_tol = max(10.0 * fmax * fmax, 1e-6 * fmax)
    assert abs(e_rg - e_ase) <= e_tol, (
        f"energy mismatch: rgmin={e_rg:.10f} vs ase={e_ase:.10f} "
        f"(diff={abs(e_rg - e_ase):.3e}, tol={e_tol:.3e})"
    )

    if update_hyperparams:
        w, s, n = opt_rg.hyperparameters
        assert np.isfinite(w) and w > 0.0
        assert np.isfinite(s) and s > 0.0
        assert np.isfinite(n) and n > 0.0

    with capsys.disabled():
        print(
            f"\n[GPMin LJ7 parity update_hyperparams={update_hyperparams}] "
            f"rgmin: calls={calc_rg.calls} steps={opt_rg.nsteps} wall={wall_rg * 1e3:.2f}ms E={e_rg:.10f} | "
            f"ASE: calls={calc_ase.calls} steps={opt_ase.nsteps} wall={wall_ase * 1e3:.2f}ms E={e_ase:.10f}"
        )


def test_gpmin_session_direct_and_error_paths():
    # Non-finite objective fails closed.
    session = Session("gpmin", 2, scale=0.4, weight=1.0, noise=0.001)

    def bad_obj(_x: list[float]) -> tuple[float, list[float]]:
        return float("nan"), [0.0, 0.0]

    with pytest.raises(ValueError):
        session.step([0.1, 0.2], bad_obj)

    # Non-SPD kernel (negligible noise with repeated near-identical curvature)
    # fails closed rather than silently adding jitter.
    tiny_noise = Session("gpmin", 2, scale=0.4, weight=1.0, noise=1e-12)

    def quad(x: list[float]) -> tuple[float, list[float]]:
        return x[0] ** 2 + x[1] ** 2, [2.0 * x[0], 2.0 * x[1]]

    with pytest.raises(ValueError, match="not positive definite"):
        x = [0.2, -0.1]
        for _ in range(10):
            x, _, _ = tiny_noise.step(x, quad)
