"""Tests for the GROMACS L-BFGS mode and MDP helper."""

from __future__ import annotations

import sys

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixBondLength

import rgmin
from rgmin._maturin import Session
from rgmin.optimize import (
    GROMACS_KJMOL_NM_TO_EV_ANGSTROM,
    GROMACS_NM_TO_ANGSTROM,
    GromacsLbfgs,
    gromacs_mdp_options,
)


def test_session_converges_and_caps_nbfgscorr_memory() -> None:
    def rosenbrock(v: list[float]) -> tuple[float, list[float]]:
        x, y, z = v
        u = 1.0 - x
        w = y - x * x
        return (
            u * u + 100.0 * w * w + 0.5 * z * z,
            [-2.0 * u - 400.0 * x * w, 200.0 * w, z],
        )

    session = Session("gromacs-lbfgs", 3, istep=0.01, max_atom=0.0, nbfgscorr=3)
    assert session.nbfgscorr == 3
    point = [-1.2, 1.0, 0.5]
    for _ in range(10_000):
        point, _value, gnorm = session.step(point, rosenbrock)
        if gnorm < 1e-6:
            break
    assert gnorm < 1e-6
    assert session.pair_count() == 3
    assert session.steps > 0
    assert session.evals > session.steps
    np.testing.assert_allclose(point, [1.0, 1.0, 0.0], atol=1e-4)


def test_gromacs_mdp_options_lbfgs_and_validation() -> None:
    options = gromacs_mdp_options(
        {
            "integrator": "l-bfgs",
            "emtol": 5.0,
            "emstep": 0.02,
            "nsteps": 12,
            "nbfgscorr": 7,
        }
    )
    assert options == {
        "method": "gromacs-lbfgs",
        "gtol": 5.0 * GROMACS_KJMOL_NM_TO_EV_ANGSTROM,
        "istep": 0.02 * GROMACS_NM_TO_ANGSTROM,
        "maxiter": 12,
        "nbfgscorr": 7,
    }
    defaults = gromacs_mdp_options(integrator="L_BFGS")
    assert defaults == {
        "method": "gromacs-lbfgs",
        "gtol": 10.0 * GROMACS_KJMOL_NM_TO_EV_ANGSTROM,
        "istep": 0.01 * GROMACS_NM_TO_ANGSTROM,
        "maxiter": sys.maxsize,
        "nbfgscorr": 10,
    }
    assert rgmin.GromacsLbfgs is GromacsLbfgs

    for bad in (0, -1, True, 2.5):
        with pytest.raises(ValueError, match="nbfgscorr"):
            gromacs_mdp_options(integrator="l-bfgs", nbfgscorr=bad)
    with pytest.raises(ValueError, match="nbfgscorr"):
        gromacs_mdp_options(integrator="steep", nbfgscorr=10)
    with pytest.raises(ValueError, match="constraints"):
        gromacs_mdp_options(integrator="l-bfgs", constraints="h-bonds")
    with pytest.raises(ValueError, match="mystery_key"):
        gromacs_mdp_options({"integrator": "l-bfgs", "mystery_key": 1})
    with pytest.raises(ValueError, match="nbfgscorr"):
        Session("gromacs-lbfgs", 3, nbfgscorr=0)
    with pytest.raises(ValueError, match="nbfgscorr"):
        Session("gromacs-cg", 3, nbfgscorr=10)


class HarmonicCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = self.atoms.get_positions()
        self.results["energy"] = 0.5 * float(np.sum(positions * positions))
        self.results["forces"] = -positions


def test_ase_wrapper_uses_mdp_stops_and_rejects_holonomic_constraints() -> None:
    atoms = Atoms("Ar2", positions=[[0.5, 0.0, 0.0], [-0.5, 0.2, 0.0]])
    atoms.calc = HarmonicCalculator()
    optimizer = GromacsLbfgs(
        atoms,
        emtol=1e-3 / GROMACS_KJMOL_NM_TO_EV_ANGSTROM,
        emstep=0.01,
        nsteps=100,
        nbfgscorr=4,
        logfile=None,
    )
    assert optimizer.run() is True
    assert optimizer._native is not None
    assert optimizer._native.nbfgscorr == 4
    assert np.linalg.norm(atoms.get_forces(), axis=1).max() < 1e-3

    constrained = Atoms("Ar2", positions=[[0.5, 0.0, 0.0], [-0.5, 0.2, 0.0]])
    constrained.calc = HarmonicCalculator()
    constrained.set_constraint(FixBondLength(0, 1))
    with pytest.raises(ValueError, match="constraints"):
        GromacsLbfgs(constrained, logfile=None)
