"""Drop-in ``rgmin.ase`` line-search optimizers against ASE 3.29.

The import and signature checks fail on a tree that does not export
``BFGSLineSearch``, ``QuasiNewton``, and ``LBFGSLineSearch``. Numerical
checks need the nanobind ``Solver`` and compare a finished relaxation
with ASE's own classes on an EMT copper cluster and a Lennard-Jones
cluster. The calculator supplies the derivatives.
"""

from __future__ import annotations

import inspect
import io
import re
import shutil
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
import rgmin
from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from ase.io.trajectory import Trajectory
from ase.optimize import BFGSLineSearch as AseBFGSLineSearch
from ase.optimize import LBFGSLineSearch as AseLBFGSLineSearch
from ase.optimize import QuasiNewton as AseQuasiNewton
from ase.optimize.optimize import RestartError
from rgmin.ase import LBFGS, BFGSLineSearch, LBFGSLineSearch, QuasiNewton

_FMAX = 0.05
_STEPS = 80


class Counted(Calculator):
    """Count geometries the inner calculator evaluates."""

    def __init__(self, inner: Calculator) -> None:
        super().__init__()
        self.inner = inner
        self.implemented_properties = list(inner.implemented_properties)
        self.calls = 0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.calls += 1
        self.inner.calculate(atoms, properties, system_changes)
        self.results = dict(self.inner.results)


class Field(Calculator):
    """One-atom analytic surface that records each geometry it evaluates."""

    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def __init__(self, surface) -> None:
        super().__init__()
        self.surface = surface
        self.trials: list[float] = []

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        positions = self.atoms.positions.copy()
        self.trials.append(float(positions[0, 0]))
        energy, gradient = self.surface(positions)
        self.results = {"energy": energy, "forces": -gradient}


def _bowl(positions: np.ndarray) -> tuple[float, np.ndarray]:
    """``(x - 1)^2 / 2`` along x, with its minimum one angstrom away."""
    shift = positions[0, 0] - 1.0
    gradient = np.zeros_like(positions)
    gradient[0, 0] = shift
    return 0.5 * shift * shift, gradient


def _slope(positions: np.ndarray) -> tuple[float, np.ndarray]:
    """``-x``, which descends at the same rate everywhere."""
    gradient = np.zeros_like(positions)
    gradient[0, 0] = -1.0
    return -float(positions[0, 0]), gradient


def _logged_force_calls(text: str) -> list[int]:
    return [int(item) for item in re.findall(r"\[\s*(\d+)\]", text)]


def _call_counts(cls: type, steps: int) -> tuple[int, int, int, list[int]]:
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.calc = Field(_bowl)
    log = io.StringIO()
    opt = cls(atoms, logfile=log, alpha=1.0, maxstep=0.2)
    opt.run(fmax=1e-8, steps=steps)
    return (
        opt.force_calls,
        opt.function_calls,
        opt.nsteps,
        _logged_force_calls(log.getvalue()),
    )


def _first_step(cls: type, surface, **kwargs: object) -> tuple[str, float, list[float]]:
    """Outcome, final x, and the evaluated x values of one optimizer step."""
    field = Field(surface)
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.calc = field
    opt = cls(atoms, logfile=None, **kwargs)
    try:
        opt.step()
        outcome = "ok"
    except RuntimeError as err:
        outcome = str(err)
    return outcome, float(atoms.positions[0, 0]), field.trials


def _have_solver() -> bool:
    return hasattr(rgmin, "Solver") and hasattr(rgmin, "SolverConfig")


def _parameters(fn: object) -> list[tuple[str, object, object]]:
    signature = inspect.signature(fn)  # type: ignore[arg-type]
    return [
        (parameter.name, parameter.kind, parameter.default)
        for parameter in signature.parameters.values()
        if parameter.name != "self"
    ]


def _cu_cluster() -> Atoms:
    atoms = bulk("Cu") * (2, 2, 2)
    atoms.rattle(stdev=0.05, seed=1)
    return atoms


def _lj_cluster() -> Atoms:
    """Four-atom Lennard-Jones molecule, rattled off its compact well."""
    atoms = Atoms(
        "Ar4",
        positions=[
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [0.6, 1.0, 0.0],
            [0.6, 0.35, 0.95],
        ],
    )
    atoms.rattle(stdev=0.05, seed=3)
    return atoms


def _copy(atoms: Atoms, calc: Calculator) -> Atoms:
    clone = atoms.copy()
    clone.calc = calc
    return clone


def _relax(cls: type, atoms: Atoms, **kwargs: object) -> Atoms:
    cls(atoms, logfile=None, **kwargs).run(fmax=_FMAX, steps=_STEPS)
    return atoms


@pytest.mark.parametrize(
    ("ours", "theirs"),
    [
        (BFGSLineSearch, AseBFGSLineSearch),
        (LBFGSLineSearch, AseLBFGSLineSearch),
    ],
)
def test_constructors_match_ase_3_29(ours: type, theirs: type) -> None:
    assert _parameters(ours.__init__) == _parameters(theirs.__init__)


def test_quasi_newton_is_the_bfgs_line_search() -> None:
    assert QuasiNewton is BFGSLineSearch
    assert AseQuasiNewton is AseBFGSLineSearch
    assert QuasiNewton.__name__ == "BFGSLineSearch"


def test_names_are_exported() -> None:
    assert rgmin.ase.BFGSLineSearch is BFGSLineSearch
    assert rgmin.ase.QuasiNewton is QuasiNewton
    assert rgmin.ase.LBFGSLineSearch is LBFGSLineSearch


@pytest.mark.skipif(not _have_solver(), reason="nanobind Solver is not in this build")
def test_lbfgs_line_search_memory_is_not_coerced() -> None:
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="memory must be an int"):
        LBFGSLineSearch(atoms, logfile=None, memory=1.5)
    with pytest.raises(ValueError, match="memory must be an int"):
        LBFGSLineSearch(atoms, logfile=None, memory=True)


def test_lbfgs_line_search_rejects_a_duplicate_flag_and_a_large_step() -> None:
    atoms = Atoms("Cu", positions=[[0.0, 0.0, 0.0]])
    with pytest.raises(TypeError, match="use_line_search"):
        LBFGSLineSearch(
            atoms, None, "-", None, None, 100, 1.0, 70.0, False, use_line_search=True
        )


def test_a_positional_use_line_search_is_a_duplicate_argument() -> None:
    # ASE sets the keyword before calling LBFGS, so the ninth positional
    # argument collides even when the caller passed no keyword.
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    message = "multiple values for argument 'use_line_search'"
    with pytest.raises(TypeError, match=re.escape(message)):
        AseLBFGSLineSearch(atoms, None, "-", None, None, 100, 1.0, 70.0, False)
    with pytest.raises(TypeError, match=re.escape(message)):
        LBFGSLineSearch(atoms, None, "-", None, None, 100, 1.0, 70.0, False)
    with pytest.raises(ValueError, match="maximum step size"):
        LBFGSLineSearch(atoms, logfile=None, maxstep=2.0)


@pytest.mark.skipif(not _have_solver(), reason="nanobind Solver is not in this build")
def test_lbfgs_line_search_forces_the_flag() -> None:
    atoms = Atoms("Cu", positions=[[0.0, 0.0, 0.0]])
    opt = LBFGSLineSearch(atoms, logfile=None, use_line_search=False)
    assert opt.use_line_search is True
    assert opt.damping == 1.0
    assert opt.memory == 100
    assert opt.alpha == 70.0


def test_other_ase_classes_still_refuse_restart(tmp_path: Path) -> None:
    atoms = Atoms("Cu", positions=[[0.0, 0.0, 0.0]])
    with pytest.raises(NotImplementedError):
        LBFGS(atoms, restart=str(tmp_path / "refused.json"), logfile=None)


@pytest.mark.skipif(not _have_solver(), reason="nanobind Solver is not in this build")
@pytest.mark.parametrize(
    ("ours", "theirs", "factory", "calculator"),
    [
        (BFGSLineSearch, AseBFGSLineSearch, _cu_cluster, EMT),
        (QuasiNewton, AseQuasiNewton, _cu_cluster, EMT),
        (LBFGSLineSearch, AseLBFGSLineSearch, _cu_cluster, EMT),
        (BFGSLineSearch, AseBFGSLineSearch, _lj_cluster, LennardJones),
        (LBFGSLineSearch, AseLBFGSLineSearch, _lj_cluster, LennardJones),
    ],
)
def test_relaxed_energy_and_positions_match_ase(
    ours: type, theirs: type, factory, calculator: type
) -> None:
    start = factory()
    left = _copy(start, calculator())
    right = _copy(start, calculator())
    _relax(ours, left)
    _relax(theirs, right)
    assert np.linalg.norm(left.get_forces(), axis=1).max() <= _FMAX
    assert np.linalg.norm(right.get_forces(), axis=1).max() <= _FMAX
    assert left.get_potential_energy() == pytest.approx(
        right.get_potential_energy(), rel=1e-4, abs=1e-4
    )
    assert np.allclose(left.positions, right.positions, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not _have_solver(), reason="nanobind Solver is not in this build")
@pytest.mark.parametrize(
    ("ours", "theirs"),
    [
        (BFGSLineSearch, AseBFGSLineSearch),
        (LBFGSLineSearch, AseLBFGSLineSearch),
    ],
)
def test_maxstep_limits_each_line_search_increment(ours: type, theirs: type) -> None:
    # ASE limits each increment from the previous trial to maxstep, so
    # one step reaches 0.2, 0.4, then the strong-Wolfe point 0.6.
    expected = _first_step(theirs, _bowl, alpha=1.0, maxstep=0.2)
    assert expected[0] == "ok"
    assert expected[1] == pytest.approx(0.6, abs=1e-12)
    outcome, x, trials = _first_step(ours, _bowl, alpha=1.0, maxstep=0.2)
    assert outcome == "ok"
    assert x == pytest.approx(expected[1], abs=1e-12)
    assert trials == pytest.approx(expected[2], abs=1e-12)


@pytest.mark.skipif(not _have_solver(), reason="nanobind Solver is not in this build")
@pytest.mark.parametrize(
    ("ours", "theirs"),
    [
        (BFGSLineSearch, AseBFGSLineSearch),
        (LBFGSLineSearch, AseLBFGSLineSearch),
    ],
)
def test_negative_maxstep_matches_ase(ours: type, theirs: type) -> None:
    expected = _first_step(theirs, _bowl, alpha=1.0, maxstep=-0.2)
    outcome, x, trials = _first_step(ours, _bowl, alpha=1.0, maxstep=-0.2)
    assert outcome == expected[0]
    assert x == pytest.approx(expected[1], abs=1e-8)
    assert trials == pytest.approx(expected[2], abs=1e-8)


@pytest.mark.skipif(not _have_solver(), reason="nanobind Solver is not in this build")
@pytest.mark.parametrize("cls", [BFGSLineSearch, LBFGSLineSearch])
def test_zero_maxstep_scales_the_increment_and_stops(cls: type) -> None:
    # ASE scales a zero increment by ``0/0`` and the search does not finish.
    # The port keeps the first trial at the start and stops before that NaN.
    field = Field(_bowl)
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.calc = field
    opt = cls(atoms, logfile=None, alpha=1.0, maxstep=0.0)
    with pytest.raises(RuntimeError, match="LineSearch failed!"):
        opt.step()
    assert atoms.positions[0, 0] == pytest.approx(0.0)
    assert field.trials
    assert field.trials[0] == pytest.approx(0.0)
    assert all(np.isfinite(trial) for trial in field.trials)


@pytest.mark.skipif(not _have_solver(), reason="nanobind Solver is not in this build")
@pytest.mark.parametrize("stpmax", [0.5, 0.0])
def test_an_opening_step_above_stpmax_fails_as_in_ase(stpmax: float) -> None:
    # ASE opens at 1 and rejects stpmax < 1 before any trial.
    expected = _first_step(
        AseBFGSLineSearch, _bowl, alpha=1.0, maxstep=10.0, stpmax=stpmax
    )
    assert expected == ("LineSearch failed!", 0.0, [0.0])
    assert (
        _first_step(BFGSLineSearch, _bowl, alpha=1.0, maxstep=10.0, stpmax=stpmax)
        == expected
    )


@pytest.mark.skipif(not _have_solver(), reason="nanobind Solver is not in this build")
@pytest.mark.parametrize(
    ("ours", "theirs"),
    [
        (BFGSLineSearch, AseBFGSLineSearch),
        (LBFGSLineSearch, AseLBFGSLineSearch),
    ],
)
def test_a_spent_line_search_budget_fails_without_a_fallback(
    ours: type, theirs: type
) -> None:
    # ASE needs three trials on this step. Two leave the search unfinished,
    # and the step fails instead of taking the best trial so far.
    unbounded = _first_step(theirs, _bowl, alpha=1.0, maxstep=0.2)
    assert len(unbounded[2]) - 1 > 2
    outcome, x, trials = _first_step(
        ours, _bowl, alpha=1.0, maxstep=0.2, linesearch_maxiter=2
    )
    assert outcome == "LineSearch failed!"
    assert x == 0.0
    assert trials == pytest.approx(unbounded[2][:3], abs=1e-12)


@pytest.mark.skipif(not _have_solver(), reason="nanobind Solver is not in this build")
def test_a_descending_trial_at_stpmax_is_taken_as_in_ase() -> None:
    # ASE ends on its "STP = maxstep" warning and keeps that trial.
    expected = _first_step(
        AseBFGSLineSearch, _slope, alpha=1.0, maxstep=10.0, stpmax=1.0
    )
    assert expected == ("ok", 1.0, [0.0, 1.0])
    assert (
        _first_step(BFGSLineSearch, _slope, alpha=1.0, maxstep=10.0, stpmax=1.0)
        == expected
    )


@pytest.mark.skipif(not _have_solver(), reason="nanobind Solver is not in this build")
def test_trajectory_logfile_and_restart(tmp_path: Path) -> None:
    start = _cu_cluster()
    counted = Counted(EMT())
    atoms = _copy(start, counted)
    trajectory = tmp_path / "cu.traj"
    logfile = tmp_path / "cu.log"
    restart = tmp_path / "cu.json"
    opt = BFGSLineSearch(
        atoms,
        restart=str(restart),
        logfile=str(logfile),
        trajectory=str(trajectory),
    )
    assert opt.run(fmax=_FMAX, steps=4) is False or counted.calls > 0
    assert counted.calls > 0
    assert restart.is_file()
    text = logfile.read_text()
    assert "FC" in text
    assert "Energy" in text
    with Trajectory(trajectory) as frames:
        assert len(frames) >= 1

    resumed = _copy(start, EMT())
    again = BFGSLineSearch(resumed, restart=str(restart), logfile=None)
    assert np.allclose(resumed.positions, atoms.positions, rtol=0.0, atol=0.0)
    again.run(fmax=_FMAX, steps=_STEPS)
    fresh = _copy(start, EMT())
    calls = Counted(EMT())
    fresh.calc = calls
    BFGSLineSearch(fresh, logfile=None).run(fmax=_FMAX, steps=_STEPS)
    assert fresh.get_potential_energy() == pytest.approx(
        resumed.get_potential_energy(), abs=1e-4
    )
    assert np.allclose(fresh.positions, resumed.positions, rtol=1e-3, atol=1e-3)

    with pytest.raises(RestartError):
        BFGSLineSearch(start, restart=str(logfile), logfile=None)


@pytest.mark.skipif(not _have_solver(), reason="nanobind Solver is not in this build")
@pytest.mark.parametrize(
    ("ours", "theirs"),
    [
        (BFGSLineSearch, AseBFGSLineSearch),
        (LBFGSLineSearch, AseLBFGSLineSearch),
    ],
)
def test_force_and_function_calls_match_ase(ours: type, theirs: type) -> None:
    # The opening value is ASE's ``func`` call. Line-search trials count
    # as both ``func`` and ``fprime``. The opening gradient does not.
    left = _call_counts(ours, 2)
    right = _call_counts(theirs, 2)
    assert left[2] == right[2]
    assert left[0] == right[0]
    assert left[1] == right[1]
    assert left[1] == left[0] + left[2]
    if ours is BFGSLineSearch:
        assert left[3] == right[3]


@pytest.mark.skipif(not _have_solver(), reason="nanobind Solver is not in this build")
@pytest.mark.parametrize("cls", [BFGSLineSearch, LBFGSLineSearch])
def test_restart_carries_the_force_call_count(tmp_path: Path, cls: type) -> None:
    start = _cu_cluster()
    restart = tmp_path / "count.json"
    first = cls(_copy(start, EMT()), restart=str(restart), logfile=None)
    first.run(fmax=1e-6, steps=3)
    recorded = first.force_calls
    recorded_values = first.function_calls
    assert recorded > 0
    assert recorded_values == recorded + first.nsteps
    snapshot = tmp_path / "snapshot.json"
    shutil.copyfile(restart, snapshot)
    first.run(fmax=1e-6, steps=2)
    assert first.force_calls > recorded

    logfile = tmp_path / "count.log"
    again = cls(_copy(start, EMT()), restart=str(snapshot), logfile=str(logfile))
    assert again.force_calls == recorded
    assert again.function_calls == recorded_values
    again.run(fmax=1e-6, steps=2)
    assert again.force_calls == first.force_calls
    assert again.function_calls == first.function_calls
    if cls is BFGSLineSearch:
        lines = logfile.read_text().splitlines()
        assert f"[{recorded:3d}]" in lines[1]
        assert f"[{first.force_calls:3d}]" in lines[-1]


@pytest.mark.skipif(not _have_solver(), reason="nanobind Solver is not in this build")
def test_h0_and_stpmax_are_refused_when_unused() -> None:
    with pytest.raises(ValueError, match="h0"):
        rgmin.SolverConfig("fire", h0=0.1)
    with pytest.raises(ValueError, match="stpmax"):
        rgmin.SolverConfig("fire", stpmax=50.0)
    with pytest.raises(ValueError, match="stpmax"):
        rgmin.SolverConfig("lbfgs", stpmax=50.0)
    with pytest.raises(ValueError, match="stpmax"):
        rgmin.SolverConfig("bfgs", linesearch="wolfe", stpmax=50.0)
    with pytest.raises(ValueError, match="ase"):
        rgmin.SolverConfig("fire", linesearch="ase")
    config = rgmin.SolverConfig(
        "bfgs", linesearch="ase", h0=0.1, stpmax=50.0, c1=0.23, c2=0.46
    )
    assert config.method_name == "bfgs"
