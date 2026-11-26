# Copyright (c) 2026 Rohit Goswami
# SPDX-License-Identifier: MIT
"""ASE 3.29.0 optimizer tests, parametrised over ``rgmin.ase``.

Adapted from ASE 3.29.0 (LGPL-2.1-or-later):

- ``ase/test/optimize/test_optimizers.py``
- ``ase/test/optimize/test_nsteps.py``
- ``ase/test/optimize/test_optimize_stepwise.py``
- ``ase/test/optimize/test_restart.py``
- ``ase/test/optimize/test_optimizer_restart.py``
- ``ase/test/optimize/test_replay.py``
- ``ase/test/optimize/test_fire.py``
- ``ase/test/optimize/test_fire2.py``
- ``ase/test/optimize/test_pure.py``
- ``ase/test/optimize/test_maxstep.py``

ASE is free software under the GNU Lesser General Public License,
version 2.1 or later. This adapted copy keeps that notice.

``ase/test/optimize/test_minimahop.py`` is not included. It drives
asap3 MinimaHopping, not an optimizer. ``test_climb_fix_internals.py``
and ``test_cellawarebfgs.py`` are not part of the optimizer contract.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import add_adsorbate, bulk, fcc100, molecule
from ase.calculators.calculator import compare_atoms
from ase.calculators.emt import EMT
from ase.calculators.qmmm import ForceConstantCalculator
from ase.constraints import FixAtoms, FixCartesian
from ase.filters import FrechetCellFilter, UnitCellFilter
from ase.io import read as ase_read
from ase.io.trajectory import Trajectory
from ase.optimize.optimize import RestartError
from ase.utils.abc import Optimizable

import rgmin.ase as rgmin_ase

# ASE's upstream contract suite is serial. Running its parametrized
# calculator and trajectory cases concurrently introduces races in ASE
# and NumPy before the adapter is exercised.  Repeating those serialized
# cases adds no concurrency coverage.
pytestmark = [pytest.mark.thread_unsafe, pytest.mark.iterations(1)]

# A local force relaxation reaches the rattled Au8 EMT minimum inside
# this budget. PSO does not: it is a swarm search, not a local step.
_RELAX_STEPS = 120
_NOT_LOCAL = frozenset({"PSO"})
_CELL_RESTART = ("BFGS", "LBFGS", "FIRE", "FIRE2", "CG", "QuickMin")
_HESSIAN_REQUIRED = frozenset({"Newton", "RFO", "Dogleg"})
_NO_UPSTREAM_HESSIAN = (
    "upstream ASE test supplies neither an analytic Hessian nor a Hessian-vector "
    "product; rgmin forbids an implicit finite-difference fallback"
)
_STRICT_FMAX = float.fromhex("0x1.12e0be826d695p-30")
_RESTART_POSITION_ATOL = float.fromhex("0x1.b7cdfd9d7bdbbp-34")
_RESTART_ENERGY_ATOL = float.fromhex("0x1.5798ee2308c3ap-27")
_STEP_BOUND_SLOP = float.fromhex("0x1.19799812dea11p-40")
_ENERGY_ATOL = float.fromhex("0x1.0c6f7a0b5ed8dp-20")
_FIRE2_DTMIN = float.fromhex("0x1.a36e2eb1c432dp-14")
_BOOTH_FMAX = float.fromhex("0x1.5798ee2308c3ap-27")
_BOOTH_ZERO_ATOL = float.fromhex("0x1.cd2b297d889bcp-54")
_MODERN_OPTIMIZABLE = hasattr(Optimizable, "get_x")
# replay_trajectory pushes L-BFGS pairs. Other classes have no ASE
# Hessian or velocity to rebuild, so replay_trajectory raises TypeError.
_REPLAY = "LBFGS"


def _classes() -> list[type]:
    return [getattr(rgmin_ase, name) for name in rgmin_ase.__all__]


def _local_params() -> list[object]:
    params: list[object] = []
    for name in rgmin_ase.__all__:
        if name in _NOT_LOCAL:
            params.append(
                pytest.param(
                    name,
                    marks=pytest.mark.xfail(
                        reason=(
                            "PSO is not a local force relaxation. "
                            "rgmin.ase.PSO documents that parity checks do not use it."
                        ),
                        strict=True,
                    ),
                )
            )
        else:
            params.append(name)
    return params


def _cls(name: str) -> type:
    return getattr(rgmin_ase, name)


def _xfail_without_hessian(name: str) -> None:
    if name in _HESSIAN_REQUIRED:
        pytest.xfail(_NO_UPSTREAM_HESSIAN)


def _au8() -> Atoms:
    atoms = bulk("Au") * (2, 2, 2)
    atoms.calc = EMT()
    atoms.rattle(stdev=0.1, seed=7)
    return atoms


def _ref_energy() -> float:
    ref = bulk("Au")
    ref.calc = EMT()
    return float(ref.get_potential_energy())


@pytest.fixture(name="atoms")
def fixture_atoms() -> Atoms:
    return _au8()


@pytest.mark.parametrize("name", _local_params())
def test_optimize(name: str, atoms: Atoms) -> None:
    """Forces converge, and the energy matches bulk gold per atom."""
    _xfail_without_hessian(name)
    fmax = 0.01
    opt = _cls(name)(atoms, logfile=None)
    assert opt.run(fmax=fmax, steps=_RELAX_STEPS) is True
    forces = atoms.get_forces()
    final = float(np.sqrt((forces**2).sum(axis=1).max()))
    per_atom = atoms.get_potential_energy() * 1 / len(atoms)
    assert abs(per_atom - _ref_energy()) < 1.75e-5
    assert final < fmax
    assert opt.converged()
    assert opt.converged(forces=forces)
    if hasattr(opt.optimizable, "get_gradient"):
        with pytest.warns(FutureWarning):
            assert opt.converged(gradient=-forces.ravel())


@pytest.mark.parametrize("name", rgmin_ase.__all__)
def test_unconverged(name: str, atoms: Atoms) -> None:
    _xfail_without_hessian(name)
    fmax = _STRICT_FMAX
    opt = _cls(name)(atoms, logfile=None)
    opt.run(fmax=fmax, steps=1)
    if hasattr(opt.optimizable, "get_gradient"):
        gradient = opt.optimizable.get_gradient()
    else:
        gradient = -atoms.get_forces().reshape(-1)
    assert not opt.gradient_converged(gradient)
    assert not opt.converged()
    assert not opt.converged(forces=-gradient.reshape(-1, 3))
    assert opt.todict()["fmax"] == fmax


@pytest.mark.parametrize("name", rgmin_ase.__all__)
def test_run_twice(name: str, atoms: Atoms) -> None:
    _xfail_without_hessian(name)
    steps = 5
    opt = _cls(name)(atoms, logfile=None)
    # fmax 0 does not count as converged, so both calls take every step.
    opt.run(fmax=0, steps=steps)
    opt.run(fmax=0, steps=steps)
    assert opt.nsteps == 2 * steps
    assert opt.max_steps == 2 * steps


@pytest.mark.parametrize("name", rgmin_ase.__all__)
def test_path_logfile_trajectory_and_observer(name: str, atoms: Atoms, tmp_path: Path) -> None:
    _xfail_without_hessian(name)
    traj = tmp_path / "trajectory.traj"
    log = tmp_path / "relax.log"
    opt = _cls(name)(atoms, logfile=str(log), trajectory=str(traj))
    seen: list[int] = []
    opt.attach(lambda: seen.append(opt.nsteps), interval=1)
    opt.run(fmax=0.05, steps=3)
    assert seen and seen[0] == 0
    assert log.is_file() and log.stat().st_size > 0
    images = ase_read(str(traj), ":")
    assert len(images) >= 2
    assert "Step" in log.read_text(encoding="utf-8")


@pytest.mark.parametrize("steps", [0, 1, 4])
@pytest.mark.parametrize("name", rgmin_ase.__all__)
def test_nsteps(name: str, steps: int) -> None:
    if steps:
        _xfail_without_hessian(name)
    atoms = bulk("Au", cubic=True)
    atoms.rattle(stdev=0.12345, seed=42)
    atoms.calc = EMT()
    opt = _cls(name)(atoms, logfile=None)
    irun = opt.irun(fmax=0, steps=steps)
    for _ in range(steps + 1):
        next(irun)
    with pytest.raises(StopIteration):
        next(irun)


@pytest.mark.parametrize("name", rgmin_ase.__all__)
def test_irun_starts_before_a_step(name: str) -> None:
    _xfail_without_hessian(name)
    atoms0 = bulk("Au", cubic=True)
    atoms0.rattle(stdev=0.15, seed=1)
    atoms = atoms0.copy()
    atoms.calc = EMT()
    opt = _cls(name)(atoms, logfile=None)
    irun = opt.irun(fmax=0.0, steps=2)
    next(irun)
    assert opt.nsteps == 0
    assert not compare_atoms(atoms0, atoms)
    next(irun)
    assert opt.nsteps == 1
    # A swarm step can keep the start when that point is still the best.
    if name not in _NOT_LOCAL:
        assert compare_atoms(atoms0, atoms) == ["positions"]


@pytest.mark.parametrize("name", rgmin_ase.__all__)
def test_attach_trajectory(name: str, tmp_path: Path) -> None:
    _xfail_without_hessian(name)
    atoms0 = bulk("Au", cubic=True)
    atoms0.rattle(stdev=0.15, seed=2)
    atoms = atoms0.copy()
    atoms.calc = EMT()
    path = tmp_path / "opt.traj"
    with Trajectory(path, "w") as traj:
        opt = _cls(name)(atoms, logfile=None, trajectory=traj)
        yielded = 0
        for _ in opt.irun(fmax=0.0, steps=2):
            yielded += 1
    with Trajectory(path) as stored:
        assert len(stored) == yielded
        assert not compare_atoms(atoms0, stored[0])


@pytest.mark.parametrize("name", rgmin_ase.__all__)
def test_restart_path_and_none(name: str, tmp_path: Path) -> None:
    _xfail_without_hessian(name)
    atoms = bulk("Cu")
    atoms.calc = EMT()
    restart = tmp_path / "restart.json"
    opt = _cls(name)(atoms, trajectory=str(tmp_path / "opt.traj"), restart=restart, logfile=None)
    opt.run(fmax=0.05, steps=2)
    assert str(opt.todict()["restart"]) == str(restart)
    bare = bulk("Cu")
    bare.calc = EMT()
    opt = _cls(name)(bare, restart=None, logfile=None)
    opt.run(fmax=0.05, steps=1)
    assert opt.todict()["restart"] is None


@pytest.mark.parametrize("name", rgmin_ase.__all__)
def test_bad_restart(name: str, tmp_path: Path) -> None:
    garbage = tmp_path / "tmp.dat"
    garbage.write_text("hello world\n", encoding="utf-8")
    with pytest.raises(RestartError, match="Could not decode"):
        _cls(name)(Atoms(), restart=garbage, logfile=None)
    shaped = tmp_path / "ase.json"
    shaped.write_text('{"optimizer": "BFGS"}\n', encoding="utf-8")
    with pytest.raises(RestartError, match="not an rgmin restart file"):
        _cls(name)(Atoms(), restart=shaped, logfile=None)


@pytest.mark.parametrize("name", rgmin_ase.__all__)
def test_restart_matches_a_continuous_run(name: str, tmp_path: Path) -> None:
    _xfail_without_hessian(name)
    restart = tmp_path / "restart.json"
    broken = _au8()
    whole = _au8()
    _cls(name)(broken, restart=restart, logfile=None).run(fmax=0, steps=2)
    _cls(name)(broken, restart=restart, logfile=None).run(fmax=0, steps=2)
    _cls(name)(whole, logfile=None).run(fmax=0, steps=4)
    assert np.allclose(broken.positions, whole.positions, rtol=0.0, atol=_RESTART_POSITION_ATOL)
    assert broken.get_potential_energy() == pytest.approx(
        whole.get_potential_energy(), abs=_RESTART_ENERGY_ATOL
    )


@pytest.mark.parametrize("name", _CELL_RESTART)
def test_frechet_restart_matches_a_continuous_run(name: str, tmp_path: Path) -> None:
    def fresh() -> FrechetCellFilter:
        atoms = bulk("Au") * (2, 2, 2)
        atoms.rattle(0.05, seed=3)
        atoms.calc = EMT()
        return FrechetCellFilter(atoms, exp_cell_factor=1.0)

    restart = tmp_path / "restart.json"
    broken = fresh()
    whole = fresh()
    kw: dict[str, object] = {"logfile": None, "restart": restart}
    if name == "BFGS":
        kw["alpha"] = 70.0
    _cls(name)(broken, **kw).run(fmax=0, steps=2)
    _cls(name)(broken, **kw).run(fmax=0, steps=2)
    whole_kw = dict(kw)
    whole_kw.pop("restart")
    _cls(name)(whole, **whole_kw).run(fmax=0, steps=4)
    assert np.allclose(
        broken.atoms.positions, whole.atoms.positions, rtol=0.0, atol=_RESTART_ENERGY_ATOL
    )
    assert np.allclose(broken.atoms.cell, whole.atoms.cell, rtol=0.0, atol=_RESTART_ENERGY_ATOL)
    assert broken.atoms.get_potential_energy() == pytest.approx(
        whole.atoms.get_potential_energy(), abs=_RESTART_ENERGY_ATOL
    )


def test_unitcellfilter_relaxes_the_cell() -> None:
    atoms = bulk("Cu", cubic=True)
    atoms.rattle(0.02, seed=2)
    atoms.calc = EMT()
    start = atoms.get_potential_energy()
    volume = atoms.get_volume()
    opt = rgmin_ase.LBFGS(UnitCellFilter(atoms), logfile=None)
    assert opt.run(fmax=0.05, steps=40) is True
    assert atoms.get_potential_energy() < start
    assert atoms.get_volume() != pytest.approx(volume)


def test_fixatoms_and_fixcartesian() -> None:
    slab = fcc100("Cu", size=(2, 2, 2), vacuum=5.0)
    add_adsorbate(slab, "Ag", 1.5, "hollow")
    mask = [atom.symbol == "Cu" for atom in slab]
    slab.set_constraint(FixAtoms(mask=mask))
    slab.calc = EMT()
    frozen = slab.positions[np.array(mask)].copy()
    assert rgmin_ase.LBFGS(slab, logfile=None).run(fmax=0.05, steps=30) is True
    assert np.allclose(slab.positions[np.array(mask)], frozen)

    atoms = _au8()
    atoms.set_constraint(FixCartesian(0, mask=[True, True, False]))
    held = atoms.positions[0, :2].copy()
    rgmin_ase.CG(atoms, logfile=None).run(fmax=0.05, steps=8)
    assert np.allclose(atoms.positions[0, :2], held)
    assert not np.allclose(atoms.positions[0, 2], _au8().positions[0, 2])


def _huge_dimer() -> tuple[Atoms, np.ndarray]:
    atoms = molecule("N2", vacuum=10.0)
    reference = atoms.copy()
    atoms.calc = ForceConstantCalculator(np.eye(6) * 100, reference, np.zeros((2, 3)))
    atoms.positions[1] += np.array([1.0, 1.0, 1.0])
    return atoms, atoms.positions.copy()


@pytest.mark.parametrize("name", rgmin_ase.__all__)
def test_maxstep(name: str) -> None:
    _xfail_without_hessian(name)
    atoms, initial = _huge_dimer()
    cls = _cls(name)
    if name == "QuickMin":
        opt = cls(atoms, logfile=None, dt=1.0)
    else:
        opt = cls(atoms, logfile=None)
    opt.run(steps=1)
    per_atom = np.linalg.norm(atoms.positions - initial, axis=1)
    if name in {"FIRE", "FIRE2"}:
        assert float(np.linalg.norm((atoms.positions - initial).ravel())) <= 0.2 + _STEP_BOUND_SLOP
    else:
        assert float(per_atom.max()) <= 0.2 + _STEP_BOUND_SLOP


def test_fire_downhill_check() -> None:
    def gold() -> Atoms:
        atoms = bulk("Au") * (2, 2, 2)
        atoms[0].x += 0.5
        atoms.calc = EMT()
        return atoms

    plain = gold()
    opt = rgmin_ase.FIRE(plain, dtmax=1.0, dt=1.0, maxstep=100.0, downhill_check=False, logfile=None)
    assert opt.run(fmax=0.001, steps=400) is True
    energy = plain.get_potential_energy()
    steps = opt.nsteps
    checked = gold()
    resets: list[float] = []

    def callback(_optimizable: object, _r: np.ndarray, value: float, previous: float) -> None:
        resets.append(value - previous)

    opt = rgmin_ase.FIRE(
        checked,
        dtmax=1.0,
        dt=1.0,
        maxstep=100.0,
        downhill_check=True,
        position_reset_callback=callback,
        logfile=None,
    )
    assert opt.run(fmax=0.001, steps=400) is True
    assert abs(checked.get_potential_energy() - energy) < _ENERGY_ATOL
    assert opt.nsteps < steps
    assert resets
    assert all(np.asarray(resets) > 0)


def test_fire2_use_abc_reduces_the_step_count() -> None:
    def gold() -> Atoms:
        atoms = bulk("Au") * (2, 2, 2)
        atoms[0].x += 0.5
        atoms.calc = EMT()
        return atoms

    parameters = {
        "dt": 0.1,
        "maxstep": 0.2,
        "dtmax": 1.0,
        "Nmin": 20,
        "finc": 1.1,
        "fdec": 0.5,
        "astart": 0.25,
        "fa": 0.99,
    }
    plain = gold()
    opt = rgmin_ase.FIRE2(plain, logfile=None, **parameters)
    assert opt.run(fmax=0.001, steps=400) is True
    energy = plain.get_potential_energy()
    steps = opt.nsteps
    resets: list[float] = []

    def callback(_optimizable: object, _r: np.ndarray, value: float, previous: float) -> None:
        resets.append(value - previous)

    abc = gold()
    opt = rgmin_ase.FIRE2(abc, logfile=None, use_abc=True, position_reset_callback=callback, **parameters)
    assert opt.run(fmax=0.001, steps=400) is True
    assert abc.get_potential_energy() == pytest.approx(energy, abs=_ENERGY_ATOL)
    assert opt.nsteps < steps
    # ASE's FIRE2.step never calls the callback, so the history stays empty.
    assert all(np.asarray(resets) > 0)


def test_fire2_refuses_a_dtmin_other_than_the_guenole_floor() -> None:
    atoms = bulk("Cu")
    atoms.calc = EMT()
    with pytest.raises(ValueError, match="0.02 \\* dt"):
        rgmin_ase.FIRE2(atoms, logfile=None, dt=0.1, dtmin=_FIRE2_DTMIN)


def test_lbfgs_refuses_maxstep_above_one_and_bfgs_warns() -> None:
    atoms = bulk("Cu")
    atoms.calc = EMT()
    with pytest.raises(ValueError, match="maximum step size"):
        rgmin_ase.LBFGS(atoms, logfile=None, maxstep=1.5)
    with pytest.warns(UserWarning, match="maximum step size"):
        rgmin_ase.BFGS(atoms, logfile=None, maxstep=1.5)


class _Booth(Optimizable):
    """Booth function, as in ASE's ``test_pure.py``."""

    def __init__(self, x0: list[float]) -> None:
        self.xy = np.array(x0, dtype=np.float64)

    def get_x(self) -> np.ndarray:
        return self.xy.copy()

    def set_x(self, x: np.ndarray) -> None:
        self.xy[:] = x

    def __len__(self) -> int:
        return 1

    def get_positions(self) -> np.ndarray:
        return self.xy.copy()

    def set_positions(self, x: np.ndarray) -> None:
        self.xy[:] = np.asarray(x, dtype=np.float64).reshape(-1)

    def get_value(self) -> float:
        x, y = self.xy
        a = x + 2 * y - 7
        b = 2 * x + y - 5
        return float(a * a + b * b)

    def get_gradient(self) -> np.ndarray:
        x, y = self.xy
        a = x + 2 * y - 7
        b = 2 * x + y - 5
        return np.array([2 * a + 4 * b, 4 * a + 2 * b])

    def get_potential_energy(self, force_consistent: bool = False) -> float:
        return self.get_value()

    def get_forces(self) -> np.ndarray:
        return -self.get_gradient()

    def iterimages(self) -> object:
        return iter(())

    def ndofs(self) -> int:
        return len(self.xy)

    def gradient_norm(self, gradient: np.ndarray) -> float:
        return float(np.linalg.norm(gradient))


@pytest.mark.skipif(
    not _MODERN_OPTIMIZABLE,
    reason="ASE before 3.29 restricts Optimizable to atom-shaped coordinates",
)
@pytest.mark.parametrize("name", rgmin_ase.__all__)
def test_booth(name: str) -> None:
    _xfail_without_hessian(name)
    target = _Booth([1.234, 2.345])
    opt = _cls(name)(target, logfile=None)
    assert opt.run(fmax=_BOOTH_FMAX, steps=300) is True
    assert target.xy == pytest.approx([1, 3], abs=_RESTART_ENERGY_ATOL)
    assert target.get_value() == pytest.approx(0, abs=_BOOTH_ZERO_ATOL)


def test_replay_trajectory_is_lbfgs_only(tmp_path: Path) -> None:
    atoms = _au8()
    path = tmp_path / "opt.traj"
    rgmin_ase.LBFGS(atoms, logfile=None, trajectory=str(path)).run(fmax=0.05, steps=6)
    fresh = _au8()
    opt = rgmin_ase.LBFGS(fresh, logfile=None)
    opt.replay_trajectory(str(path))
    assert opt._open().pair_count() > 0
    with pytest.raises(TypeError, match="_replay_trajectory"):
        rgmin_ase.BFGS(fresh, logfile=None).replay_trajectory(str(path))
    with pytest.raises(TypeError, match="_replay_trajectory"):
        rgmin_ase.FIRE(fresh, logfile=None).replay_trajectory(str(path))
