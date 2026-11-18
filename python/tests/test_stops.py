"""``etol`` and ``max_evals`` on ``Session`` and on the ASE optimizers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rgmin._core import Session

_STIFFNESS = np.array([1.0, 4.0, 16.0, 40.0])


def _quadratic(x: list[float]) -> tuple[float, list[float]]:
    gradient = _STIFFNESS * np.asarray(x)
    return 0.5 * float(np.asarray(x) @ gradient), gradient.tolist()


def _run(session: Session, objective, x: list[float], steps: int) -> list[float]:
    for _ in range(steps):
        x, _value, _gnorm = session.step(x, objective)
        if session.stop in ("Tolerance", "EnergyTolerance", "EvalBudget"):
            break
    return x


def test_stop_is_none_before_the_first_step() -> None:
    assert Session("lbfgs", 4).stop is None


def test_etol_stops_a_session_with_energy_tolerance() -> None:
    plain = Session("lbfgs", 4)
    _run(plain, _quadratic, [1.0, 1.0, 1.0, 1.0], 200)
    cut = Session("lbfgs", 4, etol=1e-3)
    _run(cut, _quadratic, [1.0, 1.0, 1.0, 1.0], 200)
    assert cut.stop == "EnergyTolerance"
    assert plain.stop != "EnergyTolerance"
    assert cut.steps < plain.steps


def test_max_evals_counts_every_objective_call() -> None:
    calls = [0]

    def objective(x: list[float]) -> tuple[float, list[float]]:
        calls[0] += 1
        return _quadratic(x)

    session = Session("lbfgs", 4, max_evals=7)
    _run(session, objective, [1.0, -1.0, 1.0, -1.0], 200)
    assert session.stop == "EvalBudget"
    assert session.evals == calls[0]
    assert session.evals >= 7


def test_setters_set_and_clear_the_stops() -> None:
    session = Session("lbfgs", 4)
    session.set_etol(1e-3)
    session.set_max_evals(1000)
    _run(session, _quadratic, [1.0, 1.0, 1.0, 1.0], 200)
    assert session.stop == "EnergyTolerance"
    other = Session("lbfgs", 4, etol=1e-3)
    other.set_etol(None)
    _run(other, _quadratic, [1.0, 1.0, 1.0, 1.0], 200)
    assert other.stop != "EnergyTolerance"


@pytest.mark.parametrize("etol", [-1.0, math.nan, math.inf])
def test_invalid_etol_is_a_value_error(etol: float) -> None:
    with pytest.raises(ValueError, match="etol"):
        Session("lbfgs", 4, etol=etol)
    with pytest.raises(ValueError, match="etol"):
        Session("lbfgs", 4).set_etol(etol)


def test_zero_max_evals_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="max_evals"):
        Session("lbfgs", 4, max_evals=0)
    with pytest.raises(ValueError, match="max_evals"):
        Session("lbfgs", 4).set_max_evals(0)


def _pt7():
    from ase import Atoms
    from ase.calculators.emt import EMT

    a = 2.7
    base = np.array(
        [[0, 0, 0], [a, 0, 0], [-a, 0, 0], [0, a, 0], [0, -a, 0], [0, 0, a], [0, 0, -a]],
        dtype=float,
    )
    rng = np.random.default_rng(7)
    atoms = Atoms("Pt7", positions=base + rng.normal(scale=0.1, size=base.shape))
    atoms.calc = EMT()
    return atoms


def test_ase_optimizer_stops_on_etol() -> None:
    pytest.importorskip("ase")
    from rgmin.optimize import LBFGS

    plain = LBFGS(_pt7(), logfile=None)
    plain.run(fmax=1e-9, steps=200)
    assert plain.stop != "EnergyTolerance"
    cut = LBFGS(_pt7(), logfile=None, etol=1e-4)
    assert cut.run(fmax=1e-9, steps=200) is True
    assert cut.stop == "EnergyTolerance"
    assert cut.nsteps < plain.nsteps


def test_ase_optimizer_stops_on_the_eval_budget() -> None:
    pytest.importorskip("ase")
    from rgmin.optimize import FIRE

    opt = FIRE(_pt7(), logfile=None, max_evals=6)
    assert opt.run(fmax=1e-9, steps=200) is False
    assert opt.stop == "EvalBudget"
