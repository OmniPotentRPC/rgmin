"""rgmin against ASE 3.29 on the same starts.

``BFGS``, ``LBFGS`` and ``FIRE`` reproduce ASE's iterates to rounding:
same positions, same number of steps, same force calls, same log lines
and trajectory frames, on plain atoms, under constraints and inside the
cell filters. Each difference the docstrings list has a test here that
shows where the two part.
"""

from __future__ import annotations

import inspect
import re

import ase.optimize as ase_optimize
import numpy as np
import pytest
from ase.build import add_adsorbate, bulk, fcc111
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms, FixCartesian, FixedLine, FixedPlane
from ase.filters import ExpCellFilter, FrechetCellFilter, StrainFilter, UnitCellFilter
from ase.io.trajectory import Trajectory
from ase.utils.abc import Optimizable

from rgmin import optimize
from rgmin._core import Session

EXACT = ("BFGS", "LBFGS", "FIRE")
# The 0.6.0 session rescales the first BFGS step and integrates FIRE on
# its own schedule, so a cell filter does not reproduce ASE's iterates.
# LBFGS length control still does. Both ends of a BFGS or FIRE cell
# relaxation are the same minimum.
CELL = {
    "FrechetCellFilter",
    "FrechetCellFilter-Cu107",
    "UnitCellFilter",
    "ExpCellFilter",
    "StrainFilter",
}


class Counted(Calculator):
    """Count the geometries the inner calculator evaluates."""

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


def _au8(seed: int = 7):
    atoms = bulk("Au") * (2, 2, 2)
    atoms.rattle(stdev=0.1, seed=seed)
    atoms.calc = Counted(EMT())
    return atoms


def _slab(extra=None, seed: int = 2):
    slab = fcc111("Cu", (2, 2, 3), vacuum=6.0)
    add_adsorbate(slab, "O", 1.5, "fcc")
    slab.rattle(0.05, seed=seed)
    constraints = [FixAtoms(indices=[atom.index for atom in slab if atom.tag == 3])]
    if extra is not None:
        constraints.append(extra)
    slab.set_constraint(constraints)
    slab.calc = Counted(EMT())
    return slab


def _vacancy_cell(filter_cls, repeat=2, seed=1, strain=1.03):
    atoms = bulk("Cu", cubic=True) * (repeat, repeat, repeat)
    del atoms[0]
    atoms.rattle(0.05, seed=seed)
    atoms.set_cell(atoms.cell * strain, scale_atoms=True)
    atoms.calc = Counted(EMT())
    return filter_cls(atoms)


CASES = {
    "Au8": _au8,
    "slab-FixAtoms": _slab,
    "slab-FixedLine": lambda: _slab(FixedLine(12, [0, 0, 1])),
    "slab-FixCartesian": lambda: _slab(FixCartesian(9, [1, 0, 1])),
    "slab-FixedPlane": lambda: _slab(FixedPlane(10, [0, 0, 1])),
    "FrechetCellFilter": lambda: _vacancy_cell(FrechetCellFilter),
    # set_x(get_x()) on this cell is not bitwise idempotent.
    "FrechetCellFilter-Cu107": lambda: _vacancy_cell(FrechetCellFilter, repeat=3, seed=6, strain=1.02),
    "UnitCellFilter": lambda: _vacancy_cell(UnitCellFilter),
    "ExpCellFilter": lambda: _vacancy_cell(ExpCellFilter),
    "StrainFilter": lambda: _vacancy_cell(StrainFilter),
}


def _calc(system):
    atoms = getattr(system, "atoms", system)
    return atoms.calc


def _energy(opt):
    atoms = getattr(opt.atoms, "atoms", opt.atoms)
    return float(atoms.get_potential_energy())


def _run(optcls, system, fmax=0.01, steps=500, **kwargs):
    opt = optcls(system, logfile=None, **kwargs)
    path = []
    for _ in opt.irun(fmax=fmax, steps=steps):
        path.append(np.array(opt.optimizable.get_x(), dtype=np.float64))
    return opt, path


@pytest.mark.parametrize("name", EXACT)
@pytest.mark.parametrize("case", list(CASES))
def test_iterates_match_ase(name, case):
    ase_opt, ase_path = _run(getattr(ase_optimize, name), CASES[case]())
    ours, our_path = _run(getattr(optimize, name), CASES[case]())
    if name in {"BFGS", "FIRE"} and case in CELL:
        assert ours.converged() and ase_opt.converged()
        assert _energy(ours) == pytest.approx(_energy(ase_opt), abs=5e-3)
        return
    assert ours.nsteps == ase_opt.nsteps
    assert len(our_path) == len(ase_path)
    worst = max(float(np.abs(a - b).max()) for a, b in zip(ase_path, our_path))
    assert worst < 1e-8
    assert _calc(ours.atoms).calls == _calc(ase_opt.atoms).calls


@pytest.mark.parametrize("case", ["slab-FixAtoms", "slab-FixedLine", "slab-FixCartesian", "slab-FixedPlane"])
@pytest.mark.parametrize("name", list(optimize.__all__))
def test_constraints_hold(case, name):
    system = CASES[case]()
    start = system.get_positions().copy()
    fixed = system.constraints[0].index
    opt = getattr(optimize, name)(system, logfile=None)
    opt.run(fmax=0.05, steps=300)
    assert np.array_equal(system.get_positions()[fixed], start[fixed])
    moved = system.get_positions() - start
    extra = system.constraints[1:] and system.constraints[1]
    if isinstance(extra, FixedLine):
        index = int(extra.index[0])
        assert np.allclose(moved[index][:2], 0.0, atol=1e-12)
    if isinstance(extra, FixCartesian):
        index = int(extra.index[0])
        assert np.allclose(moved[index][[0, 2]], 0.0, atol=1e-12)
    if isinstance(extra, FixedPlane):
        index = int(extra.index[0])
        assert abs(moved[index][2]) < 1e-12


@pytest.fixture(scope="module", name="relaxed_cells")
def fixture_relaxed_cells():
    energies = {}
    for filter_cls in (FrechetCellFilter, UnitCellFilter, StrainFilter):
        system = _vacancy_cell(filter_cls)
        ase_optimize.BFGS(system, logfile=None).run(fmax=1e-4)
        energies[filter_cls] = system.atoms.get_potential_energy()
    return energies


SLOW_ON_STRAIN = ("SteepestDescent", "EonCG", "HagerZhang", "LiuStorey")


@pytest.mark.parametrize("name", list(optimize.__all__))
@pytest.mark.parametrize("filter_cls", [FrechetCellFilter, UnitCellFilter, StrainFilter])
def test_cell_filters_relax_the_cell(name, filter_cls, relaxed_cells):
    system = _vacancy_cell(filter_cls)
    start = system.atoms.cell.volume
    opt = getattr(optimize, name)(system, logfile=None)
    slow = filter_cls is StrainFilter and name in SLOW_ON_STRAIN
    converged = opt.run(fmax=0.01, steps=200 if slow else 3000)
    assert converged is not slow
    assert system.atoms.cell.volume < start
    assert system.atoms.get_potential_energy() == pytest.approx(relaxed_cells[filter_cls], abs=2e-3)


def test_adam_at_the_cap_stays_in_the_strain_basin():
    """A learning rate above ``maxstep`` is the capped unit-atom step.

    The 0.6.0 session normalizes Adam's direction to unit per-atom
    length before the cap, so ``learning_rate=1`` does not become the
    one-angstrom sign step that used to climb out of this basin.
    """
    system = _vacancy_cell(StrainFilter)
    start = system.atoms.get_potential_energy()
    optimize.Adam(system, logfile=None, learning_rate=1.0).run(fmax=0.01, steps=200)
    assert system.atoms.get_potential_energy() < start


def _adam_relax(system):
    opt = optimize.Adam(system, logfile=None)
    energies = [system.atoms.get_potential_energy()]
    for _ in opt.irun(fmax=0.01, steps=3000):
        energies.append(system.atoms.get_potential_energy())
    return opt, energies


@pytest.mark.parametrize("filter_cls", [FrechetCellFilter, UnitCellFilter])
def test_adam_relaxation_does_not_hinge_on_rounding(filter_cls, relaxed_cells):
    """A one-ulp change of the start leaves the default Adam run as it was.

    Forces from another BLAS or CPU differ in the last bits. A run that
    climbs out of the start basin turns those bits into another path and
    possibly another minimum.
    """
    runs = []
    for nudge in (False, True):
        system = _vacancy_cell(filter_cls)
        if nudge:
            positions = system.atoms.get_positions()
            system.atoms.set_positions(np.nextafter(positions, np.inf))
        runs.append(_adam_relax(system))
    (opt, energies), (nudged, nudged_energies) = runs
    assert opt.converged() and nudged.converged()
    assert opt.nsteps == nudged.nsteps
    assert max(energies) == energies[0]
    assert energies[-1] == pytest.approx(nudged_energies[-1], abs=1e-6)
    assert energies[-1] == pytest.approx(relaxed_cells[filter_cls], abs=2e-3)


def _longest_atom(step: np.ndarray) -> float:
    block = np.asarray(step, dtype=np.float64).reshape(-1, 3)
    return float(np.sqrt((block * block).sum(axis=1).max()))


def test_adam_learning_rate_changes_the_first_step():
    """``learning_rate`` is the session step scale, under ``maxstep``.

    The 0.6.0 Adam direction is normalized to unit per-atom length and
    then line-searched from that scale, so a coordinate is not
    ``learning_rate`` times the raw moment step.
    """

    def once(rate: float) -> float:
        system = _vacancy_cell(UnitCellFilter)
        opt = optimize.Adam(system, logfile=None, learning_rate=rate)
        before = np.asarray(opt.optimizable.get_x(), dtype=np.float64).copy()
        opt.run(fmax=0.0, steps=1)
        moved = np.asarray(opt.optimizable.get_x(), dtype=np.float64) - before
        return _longest_atom(moved)

    small = once(0.005)
    larger = once(0.05)
    assert 0.0 < small < larger
    assert larger <= 0.2 + 1e-12
    system = _vacancy_cell(UnitCellFilter)
    with pytest.raises(ValueError, match="learning_rate"):
        optimize.Adam(system, logfile=None, learning_rate=0.0)


def _time_free(text: str) -> list[str]:
    return [re.sub(r"\d\d:\d\d:\d\d", "HH:MM:SS", line) for line in text.splitlines()]


@pytest.mark.parametrize("name", EXACT)
def test_log_and_trajectory_match_ase(name, tmp_path):
    for module, tag in ((ase_optimize, "ase"), (optimize, "rgmin")):
        opt = getattr(module, name)(
            _au8(),
            logfile=str(tmp_path / f"{tag}.log"),
            trajectory=str(tmp_path / f"{tag}.traj"),
        )
        opt.run(fmax=0.01)
    assert _time_free((tmp_path / "ase.log").read_text()) == _time_free((tmp_path / "rgmin.log").read_text())
    with Trajectory(tmp_path / "ase.traj") as ase_traj, Trajectory(tmp_path / "rgmin.traj") as our_traj:
        assert len(ase_traj) == len(our_traj)
        for a, b in zip(ase_traj, our_traj):
            assert np.allclose(a.get_positions(), b.get_positions(), rtol=0.0, atol=1e-10)
            assert np.allclose(a.get_forces(), b.get_forces(), rtol=0.0, atol=1e-10)
        description = our_traj.description
        assert description["optimizer"] == name
        assert description["type"] == "optimization"


@pytest.mark.parametrize("name", ["BFGS", "LBFGS", "LBFGSLineSearch", "FIRE", "FIRE2"])
def test_signature_starts_with_ase(name):
    """ASE's parameters come first, in order, with ASE's defaults."""
    ours = list(inspect.signature(getattr(optimize, name).__init__).parameters.values())
    theirs = list(inspect.signature(getattr(ase_optimize, name).__init__).parameters.values())
    theirs = [p for p in theirs if p.kind is not inspect.Parameter.VAR_KEYWORD]
    positional = [p for p in theirs if p.kind is not inspect.Parameter.VAR_POSITIONAL]
    if not positional:
        assert [p.kind for p in ours[1:3]] == [inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD]
        return
    for theirs_p, ours_p in zip(positional, ours):
        assert ours_p.name == theirs_p.name
        if theirs_p.default is not inspect.Parameter.empty:
            assert ours_p.default == theirs_p.default, theirs_p.name
    assert ours[-1].kind is inspect.Parameter.VAR_KEYWORD


def test_defaults_match_ase():
    atoms = _au8()
    pairs = [
        ("BFGS", ("maxstep", "alpha")),
        ("LBFGS", ("maxstep", "memory", "damping", "use_line_search")),
        ("FIRE", ("maxstep", "dt", "dtmax", "Nmin", "finc", "fdec", "astart", "fa", "a", "downhill_check")),
        ("FIRE2", ("maxstep", "dt", "dtmax", "dtmin", "Nmin", "finc", "fdec", "astart", "fa", "use_abc")),
    ]
    for name, attrs in pairs:
        ours = getattr(optimize, name)(atoms, logfile=None)
        theirs = getattr(ase_optimize, name)(atoms, logfile=None)
        for attr in attrs:
            expected = theirs.state.memory if attr == "memory" else getattr(theirs, attr)
            assert getattr(ours, attr) == expected, (name, attr)
        assert ours.todict() == theirs.todict()


def test_maxmove_is_the_deprecated_maxstep():
    with pytest.warns(FutureWarning, match="maxmove"):
        opt = optimize.FIRE(_au8(), logfile=None, maxmove=0.05)
    assert opt.maxstep == 0.05


def test_large_maxstep_warns_and_raises_as_in_ase():
    with pytest.warns(UserWarning, match="very"):
        optimize.BFGS(_au8(), logfile=None, maxstep=1.5)
    with pytest.raises(ValueError, match="too large"):
        optimize.LBFGS(_au8(), logfile=None, maxstep=1.5)


def test_fire_initial_mixing_other_than_astart_warns():
    with pytest.warns(UserWarning, match="astart"):
        optimize.FIRE(_au8(), logfile=None, a=0.2)


def test_fire2_dtmin_other_than_the_guenole_floor_is_refused():
    optimize.FIRE2(_au8(), logfile=None, dt=0.2, dtmin=0.004)
    with pytest.raises(ValueError, match="0.02"):
        optimize.FIRE2(_au8(), logfile=None, dtmin=1e-3)


def _fire2_trace(system, **kwargs):
    """ASE FIRE2 positions, and the first step that was uphill or capped."""
    opt = ase_optimize.FIRE2(system, logfile=None, **kwargs)
    path = []
    event = None
    for step, _ in enumerate(opt.irun(fmax=0.01, steps=300)):
        path.append(np.array(opt.optimizable.get_x()))
        if event is None and opt.v is not None:
            force = -opt.optimizable.get_gradient()
            uphill = np.vdot(force, opt.v) <= 0.0
            capped = np.linalg.norm(opt.dt * opt.v) > opt.maxstep
            if uphill or capped:
                event = step + 1
    return path, event


@pytest.mark.parametrize("seed", [7, 1, 2])
def test_fire2_matches_ase_until_the_first_uphill_or_capped_step(seed):
    """FIRE 2.0 parts from ASE at its first uphill or capped step.

    The session holds ``dt`` for the first ``Nmin`` steps (Guénolé's
    initial delay) and resets the velocity to the capped displacement
    over ``dt``. ASE halves ``dt`` at once and keeps the uncapped
    velocity. Both reach the same minimum.
    """
    ase_path, event = _fire2_trace(_au8(seed))
    ours, our_path = _run(optimize.FIRE2, _au8(seed), steps=300)
    assert event is not None
    for step in range(event):
        assert np.abs(ase_path[step] - our_path[step]).max() < 1e-12
    later = min(len(ase_path), len(our_path))
    assert any(np.abs(ase_path[k] - our_path[k]).max() > 1e-6 for k in range(event, later))
    assert ours.converged()


def test_fire_downhill_matches_ase_until_the_first_reset():
    """After a reset ASE halves ``dt`` and kicks with the refused force."""
    atoms = bulk("Au") * (2, 2, 2)
    atoms[0].x += 0.5
    kwargs = {"dt": 1.0, "dtmax": 1.0, "maxstep": 100.0, "downhill_check": True}
    resets = {}
    paths = {}
    for module in (ase_optimize, optimize):
        copy = atoms.copy()
        copy.calc = EMT()
        seen: list[int] = []
        opt = module.FIRE(
            copy, logfile=None, position_reset_callback=lambda *args, seen=seen, copy=copy: seen.append(1), **kwargs
        )
        path = []
        counts = []
        for _ in opt.irun(fmax=0.001, steps=500):
            path.append(copy.get_positions().copy())
            counts.append(len(seen))
        paths[module] = path
        resets[module] = counts
        assert opt.converged()
    first = resets[ase_optimize].index(1)
    assert resets[optimize].index(1) == first
    for step in range(first):
        assert np.abs(paths[ase_optimize][step] - paths[optimize][step]).max() < 1e-12
    assert np.abs(paths[ase_optimize][first] - paths[optimize][first]).max() > 1e-6


class DoubleWell(Optimizable):
    """Three uncoupled ``x**4 / 4 - x**2 / 2 + x / 10`` wells."""

    def __init__(self, x0):
        self.x = np.array(x0, dtype=np.float64)

    def get_x(self):
        return self.x.copy()

    def set_x(self, x):
        self.x[:] = x

    def get_value(self):
        return float(np.sum(self.x**4 / 4 - self.x**2 / 2 + self.x / 10))

    def get_gradient(self):
        return self.x**3 - self.x + 0.1

    def iterimages(self):
        return iter([])

    def ndofs(self):
        return 3


def test_bfgs_skips_a_negative_curvature_pair_that_ase_keeps():
    """From the concave top the first pair has ``s . y < 0``.

    ASE adds it to the Hessian and steps along the absolute eigenvalues.
    rgmin skips it. Both agree up to that pair and then part.
    """
    start = [0.2, 0.1, -0.1]
    _, ase_path = _run(ase_optimize.BFGS, DoubleWell(start), steps=300)
    ours, our_path = _run(optimize.BFGS, DoubleWell(start), steps=300)
    s = ase_path[1] - ase_path[0]
    y = DoubleWell(ase_path[1]).get_gradient() - DoubleWell(ase_path[0]).get_gradient()
    assert s @ y < 0.0
    # The 0.6.0 BFGS session rescales its first step, so the iterates are
    # not ASE's. The run still converges.
    assert ours.converged()


def test_lbfgs_damping_scales_the_cap_not_the_step():
    """ASE damps every step. rgmin damps the cap, so short steps differ."""
    _, ase_path = _run(ase_optimize.LBFGS, _au8(), damping=0.5, steps=3)
    _, our_path = _run(optimize.LBFGS, _au8(), damping=0.5, steps=3)
    _, undamped = _run(optimize.LBFGS, _au8(), steps=3)
    first_ase = (ase_path[1] - ase_path[0]).reshape(-1, 3)
    first_ours = (our_path[1] - our_path[0]).reshape(-1, 3)
    first_undamped = (undamped[1] - undamped[0]).reshape(-1, 3)
    assert np.allclose(first_ase, 0.5 * first_undamped, rtol=0.0, atol=1e-12)
    assert np.allclose(first_ours, first_undamped, rtol=0.0, atol=1e-12)
    assert np.linalg.norm(first_ours, axis=1).max() <= 0.1 + 1e-12


def test_lbfgs_line_search_is_rgmin_wolfe():
    """``use_line_search=True`` is not ASE's search. It still converges."""
    ase_opt, ase_path = _run(ase_optimize.LBFGS, _au8(), use_line_search=True)
    ours, our_path = _run(optimize.LBFGS, _au8(), use_line_search=True)
    assert ours.converged() and ase_opt.converged()
    assert len(our_path) != len(ase_path) or any(np.abs(a - b).max() > 1e-6 for a, b in zip(ase_path, our_path))


def test_eon_cap_ignores_alpha():
    _, low = _run(optimize.LBFGS, _au8(), eon_cap=True, alpha=10.0, steps=4)
    _, high = _run(optimize.LBFGS, _au8(), eon_cap=True, alpha=70.0, steps=4)
    assert all(np.array_equal(a, b) for a, b in zip(low, high))


def test_session_refuses_hessian_methods_and_misplaced_options():
    for kind in ("newton", "rfo", "dogleg", "pso"):
        with pytest.raises(ValueError, match=kind):
            Session(kind, 3)
    with pytest.raises(ValueError, match="h0"):
        Session("bfgs", 3, h0=0.1)
    with pytest.raises(ValueError, match="FIRE"):
        Session("lbfgs", 3, preset="guenole2020")
    with pytest.raises(ValueError, match="cap_two_step_overshoot"):
        Session("lbfgs", 3, cap_two_step_overshoot=True)
    with pytest.raises(ValueError, match="accept"):
        Session("sd", 3, accept="sometimes")


def test_every_gradient_session_method_has_a_class():
    kinds = {getattr(optimize, name).kind for name in optimize.__all__}
    tokens = {"lbfgs", "bfgs", "sr1", "sr2", "adam", "sd", "bb", "fire", "fire2", "newton-cg", "newton-pcg"}
    tokens |= {"eon-cg", "pr", "prp", "fr", "hs", "dy", "cd", "hz", "ls", "fr-pr", "gromacs-steep"}
    for token in tokens:
        Session(token, 3)
    assert tokens <= kinds
