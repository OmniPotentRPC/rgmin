import sys

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

import rgmin
from rgmin._maturin import Session
from rgmin.optimize import (
    GROMACS_KJMOL_NM_TO_EV_ANGSTROM,
    GROMACS_NM_TO_ANGSTROM,
    GromacsSteep,
    gromacs_mdp_options,
)


def test_gromacs_mdp_defaults_map_to_ase_units() -> None:
    options = gromacs_mdp_options()
    assert options == {
        "method": "gromacs-steep",
        "gtol": 10.0 * GROMACS_KJMOL_NM_TO_EV_ANGSTROM,
        "istep": 0.01 * GROMACS_NM_TO_ANGSTROM,
        "maxiter": sys.maxsize,
    }


def test_gromacs_mdp_accepts_text_values_and_finite_budget() -> None:
    options = gromacs_mdp_options(
        {"integrator": "steep", "emtol": "5", "emstep": "0.002", "nsteps": "7"}
    )
    assert options["gtol"] == pytest.approx(5.0 * GROMACS_KJMOL_NM_TO_EV_ANGSTROM)
    assert options["istep"] == pytest.approx(0.02)
    assert options["maxiter"] == 7


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("integrator", "md"),
        ("emtol", 0),
        ("emtol", True),
        ("emstep", float("nan")),
        ("emstep", False),
        ("nsteps", -2),
        ("nsteps", 1.5),
        ("nsteps", True),
    ],
)
def test_gromacs_mdp_rejects_invalid_values(key: str, value: object) -> None:
    with pytest.raises(ValueError):
        gromacs_mdp_options({key: value})


def test_top_level_gromacs_mdp_options_export() -> None:
    assert rgmin.gromacs_mdp_options is gromacs_mdp_options


def test_gromacs_steep_stops_at_machine_precision_when_unlimited() -> None:
    class _LinearSlope(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            assert atoms is not None
            pos = atoms.get_positions()
            self.results["energy"] = float(pos[0, 0])
            forces = np.zeros_like(pos)
            forces[0, 0] = -1.0
            self.results["forces"] = forces

    atoms = Atoms("H", positions=[[1.0, 0.0, 0.0]])
    atoms.calc = _LinearSlope()
    # emstep smaller than half-ulp(1.0) in Angstrom triggers MachinePrecision immediately.
    opt = GromacsSteep(atoms, logfile=None, emtol=1.0, emstep=1e-18, nsteps=-1)
    assert opt.run() is False
    assert atoms.positions[0, 0] == 1.0
    assert opt.nsteps == 1


def test_native_session_consumes_the_mapped_step() -> None:
    options = gromacs_mdp_options({"emtol": 1e-12, "emstep": 0.01, "nsteps": 1})
    method = options["method"]
    session = Session(method, 3, istep=options["istep"])

    def quadratic(point: list[float]) -> tuple[float, list[float]]:
        vector = np.asarray(point)
        return 0.5 * float(vector @ vector), vector.tolist()

    result, _value, _gnorm = session.step([1.0, 0.0, 0.0], quadratic)
    np.testing.assert_allclose(result, [0.9, 0.0, 0.0], rtol=0.0, atol=1e-15)
