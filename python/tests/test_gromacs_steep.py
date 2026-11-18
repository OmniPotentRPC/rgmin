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


def _quadratic(point: list[float]) -> tuple[float, list[float]]:
    vector = np.asarray(point, dtype=float)
    return 0.5 * float(vector @ vector), vector.tolist()


def _refused_then_accepted(shrink: float | None = None) -> float:
    """One accept, one refusal, one accept, from x=0.3 with h=0.5."""
    kwargs: dict[str, float] = {"istep": 0.5, "max_atom": 0.0}
    if shrink is not None:
        kwargs["shrink"] = shrink
    session = Session("gromacs-steep", 1, **kwargs)
    point = [0.3]
    for _ in range(3):
        point, _value, _gnorm = session.step(point, _quadratic)
    return point[0]


def test_session_shrink_selects_the_refusal_factor() -> None:
    def expected(shrink: float) -> float:
        step = 0.5
        point = 0.3
        point -= step
        step *= 1.2
        step *= shrink
        point += step
        return point

    assert _refused_then_accepted(0.5) == expected(0.5)
    assert _refused_then_accepted(0.2) == expected(0.2)
    assert _refused_then_accepted(0.5) != _refused_then_accepted(0.2)
    # Omitted shrink follows the GROMACS 2025 default (0.5). If the
    # implementation mutates to 0.2, this equality fails.
    assert _refused_then_accepted() == expected(0.5)
    assert _refused_then_accepted() != expected(0.2)
    with pytest.raises(ValueError):
        Session("steepest", 1, shrink=0.5)
    with pytest.raises(ValueError):
        Session("steepest", 1, max_constraint_shrinks=2)


def test_session_accepts_max_constraint_shrinks_for_gromacs_steep() -> None:
    session = Session("gromacs-steep", 1, max_constraint_shrinks=0)
    assert session.max_constraint_shrinks == 0
    point, _value, _gnorm = session.step([0.3], _quadratic)
    assert len(point) == 1


def _expected_shrinks(step_size: float, shrink: float) -> int:
    eps = float(np.finfo(np.float64).eps)
    count = 0
    h = step_size
    while h > eps:
        h *= shrink
        count += 1
    return count


def test_session_constraint_shrink_budget_follows_istep() -> None:
    small = Session("gromacs-steep", 3, istep=0.001)
    large = Session("gromacs-steep", 3, istep=0.1)
    assert small.max_constraint_shrinks == _expected_shrinks(0.001, 0.5)
    assert large.max_constraint_shrinks == _expected_shrinks(0.1, 0.5)
    assert small.max_constraint_shrinks < large.max_constraint_shrinks

    small_half = Session("gromacs-steep", 3, istep=0.001, shrink=0.5)
    large_half = Session("gromacs-steep", 3, istep=0.1, shrink=0.5)
    assert small_half.max_constraint_shrinks == _expected_shrinks(0.001, 0.5)
    assert large_half.max_constraint_shrinks == _expected_shrinks(0.1, 0.5)
    assert small_half.max_constraint_shrinks < large_half.max_constraint_shrinks

    small_manual = Session("gromacs-steep", 3, istep=0.001, shrink=0.2)
    large_manual = Session("gromacs-steep", 3, istep=0.1, shrink=0.2)
    assert small_manual.max_constraint_shrinks == _expected_shrinks(0.001, 0.2)
    assert large_manual.max_constraint_shrinks == _expected_shrinks(0.1, 0.2)
    assert small_manual.max_constraint_shrinks < large_manual.max_constraint_shrinks
    assert large_manual.max_constraint_shrinks < large_half.max_constraint_shrinks

    explicit = Session("gromacs-steep", 3, istep=0.1, shrink=0.2, max_constraint_shrinks=3)
    assert explicit.max_constraint_shrinks == 3
    assert Session("steepest", 3).max_constraint_shrinks is None


def test_gromacs_steep_session_does_not_install_the_generic_atom_cap() -> None:
    def walk(max_atom: float | None) -> float:
        kwargs: dict[str, float] = {"istep": 0.1}
        if max_atom is not None:
            kwargs["max_atom"] = max_atom
        session = Session("gromacs-steep", 1, **kwargs)
        point = [1.0]
        for _ in range(5):
            point, _value, _gnorm = session.step(point, _quadratic)
        return point[0]

    def recurrence(cap: float | None) -> float:
        point = 1.0
        step = 0.1
        for _ in range(5):
            move = step if cap is None or step <= cap else cap
            point -= move
            step *= 1.2
        return point

    assert walk(None) == recurrence(None)
    assert walk(0.0) == recurrence(None)
    assert walk(0.2) == recurrence(0.2)
    assert walk(None) != walk(0.2)

    plain = Session("lbfgs", 3)
    capped = Session("lbfgs", 3, max_atom=0.2)
    start = [1.0, 0.0, 0.0]
    assert plain.step(start, _quadratic)[0] == capped.step(start, _quadratic)[0]


def test_ase_gromacs_steep_emtol_convergence() -> None:
    class _FlatEnough(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            assert atoms is not None
            self.results["energy"] = 0.0
            forces = np.zeros((len(atoms), 3))
            forces[0, 0] = -0.4
            self.results["forces"] = forces

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.calc = _FlatEnough()
    emtol = 500.0  # kJ mol^-1 nm^-1
    opt = GromacsSteep(atoms, logfile=None, emtol=emtol, nsteps=8)
    assert opt.run() is True
    assert opt.nsteps == 0
