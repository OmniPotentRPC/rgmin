"""Tests for the GROMACS conjugate-gradient mode and mdp helper."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms, FixBondLength

import rgmin
from rgmin._maturin import Session
from rgmin.optimize import (
    GROMACS_KJMOL_NM_TO_EV_ANGSTROM,
    GROMACS_NM_TO_ANGSTROM,
    GromacsCg,
    gromacs_mdp_options,
)

DATA = Path(__file__).resolve().parents[2] / "tests" / "data" / "gromacs_steep_argon.txt"
KJMOL_TO_EV = GROMACS_KJMOL_NM_TO_EV_ANGSTROM * GROMACS_NM_TO_ANGSTROM


def _load_argon_fixture():
    section = ""
    meta = {}
    steps = []
    start = []
    final = []
    for raw in DATA.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line in {"steps", "start", "final"}:
            section = line
            continue
        if section == "steps":
            idx, step_size, epot, fmax = line.split()
            steps.append((int(idx), float(step_size), float(epot), float(fmax)))
        elif section == "start":
            start.extend(float(v) for v in line.split())
        elif section == "final":
            final.extend(float(v) for v in line.split())
        else:
            key, val = line.split()
            meta[key] = float(val)
    return meta, steps, np.array(start, dtype=np.float64), np.array(final, dtype=np.float64)


class ArgonLjGromacsUnits:
    """Shifted Lennard-Jones argon cluster in GROMACS units (nm, kJ/mol)."""

    def __init__(self, sigma: float, epsilon: float, cutoff: float):
        s6 = sigma**6
        self.c6 = 4.0 * epsilon * s6
        self.c12 = 4.0 * epsilon * s6 * s6
        self.cutoff2 = cutoff * cutoff

    def eval_and_grad(self, flat: list[float] | np.ndarray) -> tuple[float, list[float]]:
        x = np.asarray(flat, dtype=np.float64).reshape(-1, 3)
        n = len(x)
        energy = 0.0
        grad = np.zeros_like(x)
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                r2 = float(np.dot(dx, dx))
                if r2 >= self.cutoff2:
                    continue
                inv_r2 = 1.0 / r2
                inv_r6 = inv_r2**3
                inv_r12 = inv_r6 * inv_r6
                energy += self.c12 * inv_r12 - self.c6 * inv_r6
                f_over_r = (12.0 * self.c12 * inv_r12 - 6.0 * self.c6 * inv_r6) * inv_r2
                pair = f_over_r * dx
                # Gradient is -force.
                grad[i] -= pair
                grad[j] += pair
        return float(energy), grad.ravel().tolist()


class ArgonLjAse(Calculator):
    """Same argon cluster converted to ASE units (Angstrom, eV)."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, sigma_nm: float, epsilon_kjmol: float, cutoff_nm: float):
        super().__init__()
        self._gmx = ArgonLjGromacsUnits(sigma_nm, epsilon_kjmol, cutoff_nm)

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        pos_nm = self.atoms.get_positions() / GROMACS_NM_TO_ANGSTROM
        energy_kjmol, grad_kjmol_nm = self._gmx.eval_and_grad(pos_nm.ravel())
        forces_kjmol_nm = -np.asarray(grad_kjmol_nm, dtype=np.float64).reshape(-1, 3)
        self.results["energy"] = energy_kjmol * KJMOL_TO_EV
        self.results["forces"] = forces_kjmol_nm * GROMACS_KJMOL_NM_TO_EV_ANGSTROM


def _max_atom_norm(grad: list[float] | np.ndarray) -> float:
    arr = np.asarray(grad, dtype=np.float64).reshape(-1, 3)
    return float(np.linalg.norm(arr, axis=1).max())


def test_argon_session_and_ase_match_gromacs_cg_reference() -> None:
    meta, steps, start, _ = _load_argon_fixture()
    natoms = start.size // 3
    lj = ArgonLjGromacsUnits(meta["sigma"], meta["epsilon"], meta["cutoff"])

    # The initial Fmax and Epot match GROMACS; after 12 steps Epot matches gmx_d within 1e-5.
    session = Session("gromacs-cg", start.size, istep=meta["emstep"], max_atom=0.0, nstcgsteep=1000)
    assert session.nstcgsteep == 1000
    point = start.tolist()
    e0, g0 = lj.eval_and_grad(point)
    assert e0 == pytest.approx(steps[0][2], abs=5e-4)
    assert _max_atom_norm(g0) == pytest.approx(steps[0][3], abs=5e-4)

    for _ in range(12):
        point, value, _ = session.step(point, lj.eval_and_grad)
    _, grad_12 = lj.eval_and_grad(point)
    assert value == pytest.approx(-1.25489376279604e01, abs=1e-5)
    assert _max_atom_norm(grad_12) < 5e-3

    # ASE GromacsCg wrapper in Angstrom / eV units with emtol in kJ/mol/nm and emstep in nm.
    start_ang = start.reshape(natoms, 3) * GROMACS_NM_TO_ANGSTROM
    atoms_ase = Atoms(f"Ar{natoms}", positions=start_ang)
    atoms_ase.calc = ArgonLjAse(meta["sigma"], meta["epsilon"], meta["cutoff"])
    opt = GromacsCg(atoms_ase, emtol=1e-2, emstep=meta["emstep"], nsteps=100, nstcgsteep=1000, logfile=None)
    assert opt.run() is True
    assert opt.nsteps == 6
    epot_kjmol = atoms_ase.get_potential_energy() / KJMOL_TO_EV
    fmax_kjmol_nm = np.linalg.norm(atoms_ase.get_forces(), axis=1).max() / GROMACS_KJMOL_NM_TO_EV_ANGSTROM
    assert epot_kjmol == pytest.approx(-1.25489364597334e01, abs=1e-5)
    assert fmax_kjmol_nm < 1e-2

    atoms_tight = Atoms(f"Ar{natoms}", positions=start_ang)
    atoms_tight.calc = ArgonLjAse(meta["sigma"], meta["epsilon"], meta["cutoff"])
    opt_tight = GromacsCg(
        atoms_tight, emtol=1e-6, emstep=meta["emstep"], nsteps=100, nstcgsteep=1000, logfile=None
    )
    assert opt_tight.run() is True
    assert opt_tight.nsteps == 9
    epot_tight_kjmol = atoms_tight.get_potential_energy() / KJMOL_TO_EV
    fmax_tight_kjmol_nm = (
        np.linalg.norm(atoms_tight.get_forces(), axis=1).max() / GROMACS_KJMOL_NM_TO_EV_ANGSTROM
    )
    assert epot_tight_kjmol == pytest.approx(-1.25489377088304e01, abs=1e-12)
    assert fmax_tight_kjmol_nm < 1e-6


def _quad_restart(v: list[float]) -> tuple[float, list[float]]:
    x, y, _ = v
    return float(0.5 * (x * x + 4.0 * y * y)), [x, 4.0 * y, 0.0]


def _quad_path(nstcgsteep: int, steps: int) -> tuple[list[list[float]], float]:
    session = Session("gromacs-cg", 3, istep=0.5, max_atom=0.0, nstcgsteep=nstcgsteep)
    assert session.nstcgsteep == nstcgsteep
    point = [4.0, 1.0, 0.0]
    path = []
    gnorm = float("inf")
    for _ in range(steps):
        point, _val, gnorm = session.step(point, _quad_restart)
        path.append(list(point))
    return path, gnorm


def test_quadratic_polak_ribiere_path_and_nstcgsteep_restart() -> None:
    # f(x, y, z) = 0.5 * (x^2 + 4 * y^2) from (4, 1, 0). Step 1 is steepest
    # descent with alpha = 2/5 to (2.4, -0.6, 0). GROMACS do_cg tests
    # `nstcg && (step % nstcg) == 0` after step 0, so any positive nstcgsteep
    # makes step 2 steepest descent too, to (1.44, 0.36, 0).
    path, gnorm = _quad_path(1000, 3)
    np.testing.assert_allclose(path[0], [2.4, -0.6, 0.0], atol=1e-6)
    np.testing.assert_allclose(path[1], [1.44, 0.36, 0.0], atol=1e-5)
    # Step 3 is the first Polak-Ribiere step and reaches the origin.
    np.testing.assert_allclose(path[2], [0.0, 0.0, 0.0], atol=1e-7)
    assert gnorm < 1e-7

    # nstcgsteep = 0 turns the periodic restart off: step 2 reaches the origin.
    path_zero, gnorm_zero = _quad_path(0, 2)
    np.testing.assert_allclose(path_zero[0], [2.4, -0.6, 0.0], atol=1e-6)
    np.testing.assert_allclose(path_zero[1], [0.0, 0.0, 0.0], atol=1e-7)
    assert gnorm_zero < 1e-7

    # nstcgsteep = 1 restarts every step: step 3 is steepest descent again.
    path_sd, _ = _quad_path(1, 3)
    np.testing.assert_allclose(path_sd[1], [1.44, 0.36, 0.0], atol=1e-5)
    np.testing.assert_allclose(path_sd[2], [0.864, -0.216, 0.0], atol=1e-5)


def test_gromacs_cg_forget_restarts_the_nstcgsteep_cadence() -> None:
    used = Session("gromacs-cg", 3, istep=0.5, max_atom=0.0, nstcgsteep=3)
    point = [4.0, 1.0, 0.0]
    for _ in range(2):
        point, _, _ = used.step(point, _quad_restart)
    used.forget()
    fresh = Session("gromacs-cg", 3, istep=0.5, max_atom=0.0, nstcgsteep=3)
    other = list(point)
    for _ in range(3):
        point, _, gnorm = used.step(point, _quad_restart)
        other, _, _ = fresh.step(other, _quad_restart)
        assert list(point) == list(other)
        if gnorm < 1e-7:
            break


def test_rosenbrock_converges_with_gromacs_cg() -> None:
    def rosenbrock(v: list[float]) -> tuple[float, list[float]]:
        x, y, z = v
        dx = 1.0 - x
        dy = y - x * x
        val = dx * dx + 100.0 * dy * dy + 0.5 * z * z
        grad = [-2.0 * dx - 400.0 * x * dy, 200.0 * dy, z]
        return float(val), grad

    session = Session("gromacs-cg", 3, istep=0.01, max_atom=0.0, nstcgsteep=20)
    point = [-1.2, 1.0, 0.5]
    for _ in range(200):
        point, _val, gnorm = session.step(point, rosenbrock)
        if gnorm < 1e-8:
            break
    assert gnorm < 1e-8
    np.testing.assert_allclose(point, [1.0, 1.0, 0.0], atol=1e-6)


def test_gromacs_mdp_options_cg_and_validation() -> None:
    opts = gromacs_mdp_options(
        {"integrator": "cg", "emtol": 5.0, "emstep": 0.02, "nsteps": 12, "nstcgsteep": 50}
    )
    assert opts == {
        "method": "gromacs-cg",
        "gtol": 5.0 * GROMACS_KJMOL_NM_TO_EV_ANGSTROM,
        "istep": 0.02 * GROMACS_NM_TO_ANGSTROM,
        "maxiter": 12,
        "nstcgsteep": 50,
    }

    defaults = gromacs_mdp_options(integrator="CG", Constraints="None")
    assert defaults == {
        "method": "gromacs-cg",
        "gtol": 10.0 * GROMACS_KJMOL_NM_TO_EV_ANGSTROM,
        "istep": 0.01 * GROMACS_NM_TO_ANGSTROM,
        "maxiter": sys.maxsize,
        "nstcgsteep": 1000,
    }

    assert rgmin.GromacsCg is GromacsCg

    with pytest.raises(ValueError, match="constraints"):
        gromacs_mdp_options({"integrator": "cg", "constraints": "h-bonds"})

    with pytest.raises(ValueError, match="constraints"):
        gromacs_mdp_options({"integrator": "cg", "constraints": "all-bonds"})

    with pytest.raises(ValueError, match="constraints"):
        gromacs_mdp_options({"integrator": "cg", "constraints": True})

    assert gromacs_mdp_options({"integrator": "cg", "nstcgsteep": 0})["nstcgsteep"] == 0
    assert gromacs_mdp_options(integrator="cg", nstcgsteep="0")["nstcgsteep"] == 0

    with pytest.raises(ValueError, match="nstcgsteep"):
        gromacs_mdp_options({"integrator": "cg", "nstcgsteep": -5})

    with pytest.raises(ValueError, match="nstcgsteep"):
        gromacs_mdp_options({"integrator": "cg", "nstcgsteep": 2**32})

    with pytest.raises(ValueError, match="nstcgsteep"):
        gromacs_mdp_options({"integrator": "cg", "nstcgsteep": True})

    with pytest.raises(ValueError, match="nstcgsteep"):
        gromacs_mdp_options({"integrator": "steep", "nstcgsteep": 100})

    with pytest.raises(ValueError, match="nstcgsteep"):
        Session("gromacs-steep", 3, nstcgsteep=10)

    assert Session("gromacs-cg", 3, nstcgsteep=0).nstcgsteep == 0

    s_other = Session("gromacs-steep", 3)
    assert s_other.nstcgsteep is None


def test_gromacs_cg_ase_refuses_holonomic_constraints_and_allows_subspace_masks() -> None:
    meta, _, start, _ = _load_argon_fixture()
    natoms = start.size // 3
    start_ang = start.reshape(natoms, 3) * GROMACS_NM_TO_ANGSTROM

    atoms_bond = Atoms(f"Ar{natoms}", positions=start_ang.copy())
    atoms_bond.calc = ArgonLjAse(meta["sigma"], meta["epsilon"], meta["cutoff"])
    atoms_bond.set_constraint(FixBondLength(0, 1))
    with pytest.raises(ValueError, match="constraints"):
        GromacsCg(atoms_bond, logfile=None)

    atoms_clean = Atoms(f"Ar{natoms}", positions=start_ang.copy())
    atoms_clean.calc = ArgonLjAse(meta["sigma"], meta["epsilon"], meta["cutoff"])
    with pytest.raises(ValueError, match="constraints"):
        GromacsCg(atoms_clean, constraints="h-bonds", logfile=None)

    opt = GromacsCg(atoms_clean, constraints="none", logfile=None)
    atoms_clean.set_constraint(FixBondLength(0, 1))
    with pytest.raises(ValueError, match="constraints"):
        opt.step()

    # Subspace mask (FixAtoms, like GROMACS freezegrps) is allowed and keeps atom 0 fixed.
    atoms_frozen = Atoms(f"Ar{natoms}", positions=start_ang.copy())
    atoms_frozen.calc = ArgonLjAse(meta["sigma"], meta["epsilon"], meta["cutoff"])
    atoms_frozen.set_constraint(FixAtoms(indices=[0]))
    opt_frozen = GromacsCg(atoms_frozen, emtol=1.0, emstep=meta["emstep"], nsteps=5, logfile=None)
    opt_frozen.run()
    np.testing.assert_allclose(atoms_frozen.get_positions()[0], start_ang[0], atol=0.0)


def test_gromacs_cg_step_count_pins_nsteps_and_unlimited() -> None:
    class QuadAse(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(
            self, atoms=None, properties=("energy", "forces"), system_changes=all_changes
        ):
            super().calculate(atoms, properties, system_changes)
            r = self.atoms.get_positions()[0]
            # E = 0.5 * (x^2 + 4 y^2 + z^2) in eV, forces = -[x, 4 y, z] in eV/A
            self.results["energy"] = float(0.5 * (r[0] ** 2 + 4.0 * r[1] ** 2 + r[2] ** 2))
            self.results["forces"] = np.array([[-r[0], -4.0 * r[1], -r[2]]], dtype=np.float64)

    # Unlimited (nsteps = -1) maps to sys.maxsize.
    opts_unlim = gromacs_mdp_options(
        {"integrator": "cg", "emtol": 1e-6, "emstep": 0.05, "nsteps": -1}
    )
    assert opts_unlim["maxiter"] == sys.maxsize

    # ASE GromacsCg pins exact step count for nsteps = 0, 1, 2, -1.
    for budget, want_converged in ((0, False), (1, False), (2, False), (-1, True)):
        atoms = Atoms("Ar", positions=[[4.0, 1.0, 0.0]])
        atoms.calc = QuadAse()
        opt = GromacsCg(atoms, emtol=1e-4, emstep=0.05, nsteps=budget, logfile=None)
        assert opt.run() is want_converged
        assert opt.nsteps == (3 if budget == -1 else budget)


def test_gromacs_cg_stops_on_eval_budget_and_returns_false() -> None:
    meta, _, start, _ = _load_argon_fixture()
    natoms = start.size // 3
    start_ang = start.reshape(natoms, 3) * GROMACS_NM_TO_ANGSTROM

    atoms = Atoms(f"Ar{natoms}", positions=start_ang.copy())
    atoms.calc = ArgonLjAse(meta["sigma"], meta["epsilon"], meta["cutoff"])
    opt = GromacsCg(atoms, emtol=1e-6, emstep=meta["emstep"], nsteps=200, max_evals=3, logfile=None)
    assert opt.run() is False
    assert opt.stop == "EvalBudget"

    atoms_irun = Atoms(f"Ar{natoms}", positions=start_ang.copy())
    atoms_irun.calc = ArgonLjAse(meta["sigma"], meta["epsilon"], meta["cutoff"])
    opt_irun = GromacsCg(
        atoms_irun, emtol=1e-6, emstep=meta["emstep"], nsteps=200, max_evals=3, logfile=None
    )
    outcomes = list(opt_irun.irun())
    assert outcomes[-1] is False
    assert opt_irun.stop == "EvalBudget"
