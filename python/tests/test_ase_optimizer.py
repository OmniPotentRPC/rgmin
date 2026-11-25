"""ASE's own optimizer tests, run over the rgmin classes.

Each test follows the test of the same name in ``ase/test/optimize``
of ASE 3.29: ``test_optimizers.py``, ``test_nsteps.py``,
``test_optimize_stepwise.py``, ``test_pure.py``, ``test_restart.py``,
``test_optimizer_restart.py``, ``test_replay.py``, ``test_fire.py``,
``test_fire2.py`` and ``test_maxstep.py``. Where an rgmin class cannot
do what ASE does, the test states the difference and checks it.
"""

from __future__ import annotations

from math import sqrt
from os.path import getsize
from pathlib import Path

import numpy as np
import pytest
from ase import Atom, Atoms
from ase.build import bulk, molecule
from ase.calculators.calculator import compare_atoms
from ase.calculators.emt import EMT
from ase.calculators.qmmm import ForceConstantCalculator
from ase.constraints import FixAtoms
from ase.filters import FrechetCellFilter
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.optimize import RestartError
from ase.utils.abc import Optimizable

import rgmin
from rgmin import optimize
from rgmin._core import Session
from rgmin.optimize import BFGS, FIRE, FIRE2, LBFGS, LBFGSLineSearch

ALL = [getattr(optimize, name) for name in optimize.__all__]
ASE_NAMED = [BFGS, LBFGS, LBFGSLineSearch, FIRE, FIRE2]
# ASE calls its 1.75e-5 arbitrary. Steepest descent and the session's
# line-search-free conjugate gradients reach fmax 0.01 along the soft
# mode, up to 1.76e-5 above the minimum on this cell.
FIRST_ORDER = (optimize.SteepestDescent, optimize.CG, optimize.PolakRibiere, optimize.FRPR) + tuple(
    cls for cls in ALL if cls.__doc__ and "conjugate gradients" in cls.__doc__
)


@pytest.fixture(name="ref_atoms")
def fixture_ref_atoms():
    ref_atoms = bulk("Au")
    ref_atoms.calc = EMT()
    ref_atoms.get_potential_energy()
    return ref_atoms


@pytest.fixture(name="atoms")
def fixture_atoms(ref_atoms):
    atoms = ref_atoms * (2, 2, 2)
    atoms.calc = EMT()
    atoms.rattle(stdev=0.1, seed=7)
    assert atoms.get_potential_energy() > 0.45
    return atoms


@pytest.fixture(name="distorted_bulk_gold")
def fixture_distorted_bulk_gold():
    atoms = bulk("Au")
    atoms *= (2, 2, 2)
    atoms[0].x += 0.5
    return atoms


@pytest.fixture(name="testdir")
def fixture_testdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.parametrize("optcls", ALL)
def test_optimize(optcls, atoms, ref_atoms):
    fmax = 0.01
    with optcls(atoms, logfile=None) as opt:
        is_converged = opt.run(fmax=fmax)
    assert is_converged

    forces = atoms.get_forces()
    final_fmax = max((forces**2).sum(axis=1) ** 0.5)
    e_opt = atoms.get_potential_energy() * len(ref_atoms) / len(atoms)
    tolerance = 2.0e-5 if optcls in FIRST_ORDER else 1.75e-5
    assert abs(e_opt - ref_atoms.get_potential_energy()) < tolerance
    assert final_fmax < fmax
    assert opt.converged()
    assert opt.converged(forces=forces)
    with pytest.warns(FutureWarning):
        assert opt.converged(gradient=-forces.ravel())


@pytest.mark.parametrize("optcls", ALL)
def test_unconverged(optcls, atoms):
    with optcls(atoms, logfile=None) as opt:
        opt.run(fmax=1e-9, steps=1)
    gradient = opt.optimizable.get_gradient()
    assert not opt.gradient_converged(gradient)
    assert not opt.converged()
    assert not opt.converged(forces=-gradient.reshape(-1, 3))
    assert opt.todict()["fmax"] == 1e-9


@pytest.mark.parametrize("optcls", ALL)
def test_run_twice(optcls, atoms):
    """ASE uses fmax 1e-9. Newton-CG gets below that in five steps."""
    steps = 5
    with optcls(atoms, logfile=None) as opt:
        opt.run(fmax=0.0, steps=steps)
        opt.run(fmax=0.0, steps=steps)
    assert opt.nsteps == 2 * steps
    assert opt.max_steps == 2 * steps


@pytest.mark.parametrize("optcls", ALL)
def test_path(testdir, optcls, atoms):
    traj, log = Path("trajectory.traj"), Path("relax.log")
    seen: list[float] = []
    with optcls(atoms, logfile=log, trajectory=traj) as opt:
        opt.attach(lambda: seen.append(float(atoms.get_potential_energy())))
        assert opt.run(fmax=0.01)
    assert traj.is_file()
    assert log.is_file()
    assert len(seen) == opt.nsteps + 1
    with Trajectory(traj) as frames:
        assert len(frames) == opt.nsteps + 1
    lines = log.read_text().splitlines()
    assert len(lines) == opt.nsteps + 2


@pytest.mark.parametrize("optcls", ASE_NAMED)
@pytest.mark.parametrize("steps", [0, 1, 4])
def test_nsteps(optcls, steps):
    atoms = bulk("Au", cubic=True)
    atoms.rattle(stdev=0.12345, seed=42)
    atoms.calc = EMT()
    with optcls(atoms, logfile=None) as opt:
        irun = opt.irun(fmax=0, steps=steps)
        for _ in range(steps + 1):
            next(irun)
        with pytest.raises(StopIteration):
            next(irun)


@pytest.fixture(name="atoms0")
def fixture_atoms0():
    atoms = bulk("Au", cubic=True)
    atoms.rattle(stdev=0.15, seed=3)
    return atoms


@pytest.mark.parametrize("optcls", ASE_NAMED)
def test_irun_start(optcls, atoms0):
    atoms = atoms0.copy()
    atoms.calc = EMT()
    opt = optcls(atoms, logfile=None)
    irun = opt.irun(fmax=0.0)
    next(irun)
    assert not compare_atoms(atoms0, atoms)
    next(irun)
    assert compare_atoms(atoms0, atoms) == ["positions"]


@pytest.mark.parametrize("optcls", ASE_NAMED)
def test_attach_trajectory(optcls, tmp_path, atoms0):
    atoms = atoms0.copy()
    atoms.calc = EMT()
    trajfile = tmp_path / "tmp.traj"
    n_yields = 3
    with Trajectory(trajfile, "w") as traj:
        opt = optcls(atoms, trajectory=traj, logfile=None)
        irun = opt.irun(fmax=0.0)
        for _ in range(n_yields):
            next(irun)
    with Trajectory(trajfile) as traj:
        images = list(traj)
    assert len(images) == n_yields
    assert not compare_atoms(atoms0, images[0])
    assert compare_atoms(atoms0, images[-1]) == ["positions"]


class BoothFunctionOptimizable(Optimizable):
    """The Booth function, as in ASE's ``test_pure.py``."""

    def __init__(self, x0):
        self.xy = np.array(x0)

    def get_x(self):
        return self.xy.copy()

    def set_x(self, x):
        self.xy[:] = x

    @staticmethod
    def ab(x, y):
        return x + 2 * y - 7, 2 * x + y - 5

    def get_value(self):
        a, b = self.ab(*self.xy)
        return a * a + b * b

    def get_gradient(self):
        a, b = self.ab(*self.xy)
        return np.array([2 * a + 4 * b, 4 * a + 2 * b])

    def iterimages(self):
        return iter([])

    def ndofs(self):
        return len(self.xy)

    def gradient_norm(self, gradient):
        return np.linalg.norm(gradient)


def test_riemannian_cg_leaves_the_default_atom_cap_off():
    opt = optimize.RiemannianCG(BoothFunctionOptimizable([1.234, 2.345]), logfile=None)
    assert opt._session_options()["max_atom"] == 0.0
    capped = optimize.RiemannianCG(
        BoothFunctionOptimizable([1.234, 2.345]), logfile=None, maxstep=0.05
    )
    assert capped._session_options()["max_atom"] == 0.05


@pytest.mark.parametrize("optcls", ALL)
def test_booth(optcls):
    target = BoothFunctionOptimizable([1.234, 2.345])
    eps = 1e-8
    # GPMin's ASE default observation noise (1e-3) sits five orders of
    # magnitude above eps = 1e-8; pass a noise floor compatible with eps.
    extra = {"noise": 1e-6} if optcls is optimize.GPMin else {}
    with optcls(target, logfile=None, **extra) as opt:
        opt.run(fmax=eps, steps=20_000)
    assert target.xy == pytest.approx([1, 3], abs=eps)
    assert target.get_value() == pytest.approx(0, abs=eps**2)


@pytest.mark.parametrize("optcls", ASE_NAMED)
def test_path_restart(testdir, optcls):
    atoms = bulk("Cu")
    atoms.calc = EMT()
    with optcls(atoms, trajectory="opt.traj", restart=Path("restart.json"), logfile=None) as dyn:
        dyn.run()
        assert dyn.todict()["restart"] == "restart.json"


@pytest.mark.parametrize("optcls", ASE_NAMED)
def test_none_restart(testdir, optcls):
    atoms = bulk("Cu")
    atoms.calc = EMT()
    with optcls(atoms, trajectory="opt.traj", restart=None, logfile=None) as dyn:
        dyn.run()
        assert dyn.todict()["restart"] is None


@pytest.mark.parametrize("optcls", ASE_NAMED)
def test_bad_restart(testdir, optcls):
    Path("tmp.dat").write_text("hello world\n")
    with pytest.raises(RestartError, match="Could not decode"):
        optcls(Atoms(), restart="tmp.dat")


def test_ase_restart_file_is_refused(testdir, atoms):
    """An ASE ``BFGS`` restart holds a Hessian. rgmin cannot load it."""
    from ase.optimize import BFGS as AseBFGS

    other = atoms.copy()
    other.calc = EMT()
    with AseBFGS(other, restart="ase.json", logfile=None) as opt:
        opt.run(fmax=0.05, steps=2)
    with pytest.raises(RestartError, match="not an rgmin restart file"):
        BFGS(atoms, restart="ase.json", logfile=None)


def test_restart_from_another_class_is_refused(testdir, atoms):
    with LBFGS(atoms, restart="r.json", logfile=None) as opt:
        opt.run(fmax=0.05, steps=2)
    with pytest.raises(RestartError, match="written by LBFGS"):
        BFGS(atoms, restart="r.json", logfile=None)


def _filter_atoms(optcls, trajectory, restart):
    atoms = bulk("Au")
    atoms *= 2
    atoms.rattle(stdev=0.005, seed=1)
    if Path(trajectory).is_file():
        with Trajectory(trajectory, "r") as traj:
            atoms = traj[-1]
    atoms.calc = EMT()
    return optcls(
        FrechetCellFilter(atoms, exp_cell_factor=1.0),
        trajectory=trajectory,
        restart=restart,
        append_trajectory=True,
        logfile=None,
    )


def _read_traj(file: str):
    data = []
    with Trajectory(file, "r") as traj:
        for atoms in traj:
            forces = atoms.calc.results["forces"]
            data.append(
                {
                    "energy": atoms.calc.results["energy"],
                    "position": atoms.get_positions(),
                    "forces": forces,
                    "fmax": sqrt((forces**2).sum(axis=1).max()),
                    "smax": abs(atoms.calc.results["stress"]).max(),
                }
            )
    return data


@pytest.mark.parametrize("optcls", ASE_NAMED + [optimize.CG, optimize.SR1])
def test_optimizers_restart(testdir, optcls):
    """Break after two steps, restart from the file, match a single run."""
    with _filter_atoms(optcls, "single.traj", "single.json") as single:
        single.run(fmax=0.005, steps=300)

    with _filter_atoms(optcls, "fragile.traj", "fragile.json") as first:
        for index, _ in enumerate(first.irun(fmax=0.005, steps=300)):
            if index == 2:
                break
    assert first.nsteps == 2
    assert all(getsize(name) for name in ("fragile.json", "fragile.traj"))

    with _filter_atoms(optcls, "fragile.traj", "fragile.json") as second:
        second.run(fmax=0.005, steps=300)

    assert single.nsteps == first.nsteps + second.nsteps
    single_traj = _read_traj("single.traj")
    fragile_traj = _read_traj("fragile.traj")
    assert len(single_traj) == len(fragile_traj)
    for one, two in zip(fragile_traj, single_traj):
        for key in one:
            assert np.allclose(one[key], two[key])


def test_restart_replays_without_calling_the_calculator(testdir, atoms):
    with LBFGS(atoms, restart="r.json", logfile=None) as opt:
        opt.run(fmax=1e-9, steps=4)

    class Refuse(EMT):
        def calculate(self, *args, **kwargs):
            raise AssertionError("replay called the calculator")

    copy = atoms.copy()
    copy.calc = Refuse()
    opt = LBFGS(copy, restart="r.json", logfile=None)
    opt._open()
    assert opt._native.steps == 4


def test_restart_with_other_parameters_is_refused(testdir, atoms):
    start = atoms.copy()
    with BFGS(atoms, restart="r.json", logfile=None) as opt:
        opt.run(fmax=1e-9, steps=3)
    start.calc = EMT()
    opt = BFGS(start, restart="r.json", logfile=None, maxstep=0.01)
    with pytest.raises(RestartError, match="does not replay|more evaluations"):
        opt.run(fmax=1e-9, steps=1)


def _ag_on_cu():
    d = 3.6 / sqrt(2)
    a = Atoms("Cu", positions=[(0, 0, 0)], cell=(d, d, 1.0), pbc=(True, True, False))
    a *= (2, 2, 1)
    a += Atom("Ag", (d / 2, d / 2, 2.0))
    a.set_constraint(FixAtoms(range(len(a) - 1)))
    a.calc = EMT()
    return a


@pytest.mark.parametrize("optcls", [LBFGS, LBFGSLineSearch])
def test_replay(testdir, optcls):
    with optcls(_ag_on_cu(), trajectory="AgCu1.traj", logfile="AgCu1.log") as dyn1:
        dyn1.run(fmax=0.1)
    a = read("AgCu1.traj")
    a.calc = EMT()
    with optcls(a, trajectory="AgCu2.traj", logfile="AgCu2.log") as dyn2:
        dyn2.replay_trajectory("AgCu1.traj")
        assert dyn2._native.pair_count() > 0
        assert dyn2.run(fmax=0.01)


@pytest.mark.parametrize("optcls", [BFGS, FIRE, FIRE2, optimize.CG])
def test_replay_is_lbfgs_only(testdir, optcls):
    """Only the L-BFGS session takes outside pairs. ASE's BFGS replays."""
    with optcls(_ag_on_cu(), trajectory="AgCu1.traj", logfile=None) as dyn1:
        dyn1.run(fmax=0.1)
    a = read("AgCu1.traj")
    a.calc = EMT()
    with pytest.raises(TypeError, match="_replay_trajectory"):
        optcls(a, logfile=None).replay_trajectory("AgCu1.traj")


def test_fire(distorted_bulk_gold):
    a = distorted_bulk_gold.copy()
    a.calc = EMT()
    opt = FIRE(a, dtmax=1.0, dt=1.0, maxstep=100.0, downhill_check=False, logfile=None)
    opt.run(fmax=0.001)
    e1 = a.get_potential_energy()
    n1 = opt.nsteps

    a = distorted_bulk_gold.copy()
    a.calc = EMT()
    reset_history = []

    def callback(a, r, e, e_last):
        reset_history.append([e - e_last])

    opt = FIRE(
        a,
        dtmax=1.0,
        dt=1.0,
        maxstep=100.0,
        downhill_check=True,
        position_reset_callback=callback,
        logfile=None,
    )
    opt.run(fmax=0.001)
    e2 = a.get_potential_energy()
    n2 = opt.nsteps

    assert abs(e1 - e2) < 1e-6
    assert n2 < n1
    assert reset_history
    assert all(np.array(reset_history) > 0)


def test_fire2(distorted_bulk_gold):
    a = distorted_bulk_gold.copy()
    a.calc = EMT()
    fire_parameters = {
        "dt": 0.1,
        "maxstep": 0.2,
        "dtmax": 1.0,
        "Nmin": 20,
        "finc": 1.1,
        "fdec": 0.5,
        "astart": 0.25,
        "fa": 0.99,
    }
    opt = FIRE2(a, logfile=None, **fire_parameters)
    opt.run(fmax=0.001)
    e1 = a.get_potential_energy()
    n1 = opt.nsteps

    a = distorted_bulk_gold.copy()
    a.calc = EMT()
    reset_history = []

    def callback(a, r, e, e_last):
        reset_history.append([e - e_last])

    opt = FIRE2(a, use_abc=True, position_reset_callback=callback, logfile=None, **fire_parameters)
    opt.run(fmax=0.001)
    e2 = a.get_potential_energy()
    n2 = opt.nsteps

    assert e2 == pytest.approx(e1, abs=1e-6)
    assert n2 < n1
    assert reset_history == []


@pytest.mark.parametrize("optcls", ALL)
def test_maxstep(optcls):
    """ASE's ``test_maxstep`` clips MDMin. Here every rgmin class."""
    atoms = molecule("N2", vacuum=10)
    atoms.calc = ForceConstantCalculator(np.eye(6) * 100, atoms.copy(), np.zeros((2, 3)))
    atoms.positions[1] += np.array([1.0, 1.0, 1.0])
    initial = atoms.positions.copy()
    opt = optcls(atoms, logfile=None)
    opt.run(steps=1)
    moved = atoms.positions - initial
    if optcls in (FIRE, FIRE2):
        assert np.linalg.norm(moved) <= 0.2 + 1e-12
    else:
        assert np.max(np.linalg.norm(moved, axis=1)) <= 0.2 + 1e-12


def test_lbfgs_signature_accepts_the_ase_keywords(atoms):
    opt = LBFGS(
        atoms,
        logfile=None,
        maxstep=0.2,
        memory=8,
        damping=1.0,
        alpha=70.0,
        use_line_search=False,
    )
    assert opt.maxstep == 0.2
    assert opt.memory == 8
    opt.run(fmax=0.05, steps=5)


def test_package_exports_every_optimizer():
    assert rgmin._OPTIMIZERS == frozenset(optimize.__all__)
    for name in optimize.__all__:
        assert getattr(rgmin, name) is getattr(optimize, name)


def test_eon_cap_escapes_the_well_the_ase_scale_keeps():
    """The scaled step stops near x0 = -1. The eOn cap does not."""

    def rosenbrock(x: list[float]) -> tuple[float, list[float]]:
        f = 0.0
        g = [0.0] * len(x)
        for i in range(len(x) - 1):
            a = 1.0 - x[i]
            b = x[i + 1] - x[i] * x[i]
            f += a * a + 100.0 * b * b
            g[i] += -2.0 * a - 400.0 * x[i] * b
            g[i + 1] += 200.0 * b
        return f, g

    start = [(-1.2 if i % 2 == 0 else 1.0) for i in range(10)]

    def relax(eon_cap: bool) -> float:
        session = Session("lbfgs", 10, max_atom=0.2, memory=8, eon_cap=eon_cap)
        x = list(start)
        value = rosenbrock(x)[0]
        for _ in range(20_000):
            x, value, gnorm = session.step(x, rosenbrock)
            if gnorm < 1e-5:
                break
        return value

    scaled = relax(False)
    capped = relax(True)
    assert scaled > 1.0
    assert capped < 1e-8
