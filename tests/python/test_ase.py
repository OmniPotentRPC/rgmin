# Copyright (c) 2026 Rohit Goswami
# SPDX-License-Identifier: MIT
"""ASE Optimizer contract, cache behaviour, and a short parity check."""

from __future__ import annotations

import os
from io import StringIO
from typing import Any

import numpy as np
import pytest

ase = pytest.importorskip("ase")
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT
from ase.io import read as ase_read
from ase.optimize.optimize import Optimizer as ASEOptimizer

import rgmin
from rgmin.ase import BFGS, FIRE, LBFGS, Optimizer, SD

_SLOW = bool(os.environ.get("TSAN_OPTIONS"))


class CountingEMT(EMT):
    """EMT that counts ``calculate`` calls on this instance."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.calls = 0

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] | None = None,
    ) -> None:
        self.calls += 1
        if system_changes is None:
            system_changes = list(all_changes)
        super().calculate(atoms, properties, system_changes)


def _pt7(seed: int, scale: float = 0.002) -> Atoms:
    a = 2.7
    base = np.array(
        [
            [0.0, 0.0, 0.0],
            [a, 0.0, 0.0],
            [-a, 0.0, 0.0],
            [0.0, a, 0.0],
            [0.0, -a, 0.0],
            [0.0, 0.0, a],
            [0.0, 0.0, -a],
        ]
    )
    rng = np.random.default_rng(seed)
    atoms = Atoms("Pt7", positions=base + rng.normal(scale=scale, size=base.shape))
    atoms.calc = EMT()
    return atoms


def _rmsd(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=np.float64) - np.mean(a, axis=0)
    right = np.asarray(b, dtype=np.float64) - np.mean(b, axis=0)
    cross = left.T @ right
    u, _, vt = np.linalg.svd(cross)
    rot = u @ vt
    if np.linalg.det(rot) < 0.0:
        u = u.copy()
        u[:, -1] *= -1.0
        rot = u @ vt
    aligned = left @ rot
    delta = aligned - right
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def test_run_steps_zero() -> None:
    loose = _pt7(1, scale=0.2)
    opt = SD(loose, logfile=None, maxstep=0.05)
    assert opt.run(fmax=1e-8, steps=0) is False
    one = Atoms("Pt", positions=[[0.0, 0.0, 0.0]])
    one.calc = EMT()
    assert SD(one, logfile=None).run(fmax=0.05, steps=0) is True


def test_irun_yields_bool() -> None:
    atoms = _pt7(2)
    opt = SD(atoms, logfile=None, maxstep=0.05)
    flags = list(opt.irun(fmax=0.5, steps=2))
    assert flags
    assert all(isinstance(flag, (bool, np.bool_)) for flag in flags)


def test_attach_interval(tmp_path: Any) -> None:
    atoms = _pt7(3)
    seen: list[int] = []
    opt = SD(atoms, logfile=None, maxstep=0.05)
    opt.attach(lambda: seen.append(opt.nsteps), interval=2)
    opt.run(fmax=1e-6, steps=4)
    assert seen[0] == 0
    assert all(step % 2 == 0 for step in seen)


def test_logfile_columns_match_ase() -> None:
    atoms = _pt7(4)
    ours = StringIO()
    theirs = StringIO()
    LBFGS(atoms, logfile=ours, maxstep=0.05).run(fmax=0.5, steps=1)
    atoms2 = _pt7(4)
    from ase.optimize import LBFGS as AseLBFGS

    AseLBFGS(atoms2, logfile=theirs, maxstep=0.05).run(fmax=0.5, steps=1)
    our_header = ours.getvalue().splitlines()[0]
    ase_header = theirs.getvalue().splitlines()[0]
    assert "Step" in our_header and "Energy" in our_header and "fmax" in our_header
    assert our_header.split() == ase_header.split()


def test_trajectory_round_trip(tmp_path: Any) -> None:
    atoms = _pt7(5)
    path = tmp_path / "opt.traj"
    LBFGS(atoms, logfile=None, trajectory=str(path), maxstep=0.05).run(fmax=0.5, steps=2)
    images = ase_read(str(path), ":")
    assert len(images) >= 2
    for image in images:
        energy = image.get_potential_energy()
        forces = image.get_forces()
        assert np.isfinite(energy)
        assert forces.shape == (len(image), 3)
        assert np.isfinite(forces).all()


def test_first_evaluation_hits_calculator_cache() -> None:
    atoms = _pt7(6, scale=0.02)
    calc = CountingEMT()
    atoms.calc = calc
    atoms.get_forces()
    start = atoms.get_positions().copy()
    warmed = calc.calls
    positions: list[np.ndarray] = []
    original = calc.calculate

    def spy(
        atoms_arg: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] | None = None,
    ) -> None:
        if atoms_arg is not None:
            positions.append(atoms_arg.get_positions().copy())
        original(atoms_arg, properties, system_changes)

    calc.calculate = spy  # type: ignore[method-assign]
    FIRE(atoms, logfile=None, maxstep=0.05).step()
    assert positions, "the step made no new force call"
    assert not np.array_equal(positions[0], start)
    assert calc.calls > warmed


def test_todict_and_restart_refusal(tmp_path: Any) -> None:
    atoms = _pt7(7)
    opt = FIRE(atoms, logfile=None, trajectory=None, memory=5)
    data = opt.todict()
    assert data["type"] == "optimization"
    assert data["optimizer"] == "FIRE"
    assert data["method"] == "fire"
    with pytest.raises(NotImplementedError, match="FIRE"):
        opt.dump(data)
    restart = tmp_path / "restart.json"
    restart.write_text("{}\n", encoding="utf-8")
    with pytest.raises(NotImplementedError, match=r"FIRE\.read is not implemented"):
        FIRE(atoms, restart=str(restart), logfile=None)


def test_cell_filter_runs() -> None:
    pytest.importorskip("ase.filters")
    from ase.build import bulk
    from ase.filters import FrechetCellFilter

    atoms = bulk("Cu", cubic=True)
    atoms.rattle(0.01, seed=1)
    atoms.calc = EMT()
    opt = LBFGS(FrechetCellFilter(atoms), logfile=None, maxstep=0.05)
    assert isinstance(opt, ASEOptimizer)
    assert opt.run(fmax=0.05, steps=80) is True


@pytest.mark.skipif(_SLOW, reason="NEB is too slow under TSan")
def test_neb_au_on_al() -> None:
    from ase.build import add_adsorbate, fcc100
    from ase.constraints import FixAtoms
    from ase.mep import NEB

    def image(offset: float) -> Atoms:
        slab = fcc100("Al", size=(2, 2, 2), vacuum=5.0)
        add_adsorbate(slab, "Au", 1.7, "hollow")
        slab.positions[-1, 0] += offset
        mask = [atom.tag > 1 for atom in slab]
        slab.set_constraint(FixAtoms(mask=mask))
        slab.calc = EMT()
        return slab

    images = [image(0.0)]
    for _ in range(3):
        images.append(image(0.0))
    images.append(image(1.5))
    neb = NEB(images, method="improvedtangent")
    neb.interpolate()
    opt = FIRE(neb, logfile=None, maxstep=0.2)
    assert opt.run(fmax=0.2, steps=80) is True
    assert np.isfinite(images[2].get_potential_energy())


@pytest.mark.skipif(_SLOW, reason="min-mode is too slow under TSan")
@pytest.mark.thread_unsafe
def test_min_mode_atoms() -> None:
    from ase.mep import MinModeAtoms

    atoms = _pt7(8, scale=0.02)
    mode = MinModeAtoms(atoms, logfile=None, random_seed=1)
    opt = FIRE(mode, logfile=None, maxstep=0.1)
    assert opt.run(fmax=0.5, steps=80) is True


@pytest.mark.thread_unsafe
def test_autoneb_fire(tmp_path: Any) -> None:
    from ase.io import write as ase_write
    from ase.mep import AutoNEB

    # ASE 3.23 writes np.NaN. NumPy 2 removed that alias.
    if not hasattr(np, "NaN"):
        np.NaN = np.nan  # type: ignore[attr-defined]

    start = _pt7(9, scale=0.01)
    start.get_potential_energy()
    end = start.copy()
    end.positions[:, 0] += 0.4
    end.calc = EMT()
    end.get_potential_energy()
    prefix = tmp_path / "band"
    ase_write(str(prefix) + "000.traj", start)
    ase_write(str(prefix) + "001.traj", end)

    def attach(images: list[Atoms]) -> None:
        for image in images:
            image.calc = EMT()

    band = AutoNEB(
        attach,
        prefix=str(prefix),
        n_simul=1,
        n_max=3,
        fmax=1.0,
        maxsteps=4,
        optimizer=FIRE,
        parallel=False,
        interpolate_method="linear",
        climb=False,
    )
    band.run()
    assert len(band.all_images) >= 3


def _kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """RMSD after translation and a proper rotation, with atom order fixed.

    The parity specification asks for an IRA-corrected RMSD, which also
    assigns identical atoms. This check does not permute the seven Pt
    atoms. The threshold stays 0.05 angstrom on the fixed labeling.
    """
    return _rmsd(a, b)


@pytest.mark.skipif(_SLOW, reason="parity repeats are too slow under TSan")
@pytest.mark.parametrize("ours_cls, ase_name", [(FIRE, "FIRE"), (LBFGS, "LBFGS")])
def test_pt7_parity(ours_cls: type[Optimizer], ase_name: str) -> None:
    from ase.optimize import FIRE as AseFIRE
    from ase.optimize import LBFGS as AseLBFGS

    ase_cls = AseFIRE if ase_name == "FIRE" else AseLBFGS
    for seed in range(5):
        left = _pt7(100 + seed, scale=0.002)
        right = left.copy()
        right.calc = EMT()
        ours_cls(left, logfile=None, maxstep=0.2).run(fmax=0.02, steps=250)
        ase_cls(right, logfile=None, maxstep=0.2).run(fmax=0.02, steps=250)
        e_left = left.get_potential_energy()
        e_right = right.get_potential_energy()
        assert abs(e_left - e_right) < 1e-4
        assert _kabsch_rmsd(left.positions, right.positions) < 0.05


def test_counting_calculator_is_a_calculator() -> None:
    assert issubclass(CountingEMT, Calculator)


def _iteration_calls(optimizer: type[Optimizer], ase_optimizer: type[ASEOptimizer]) -> tuple[int, int, bool]:
    """New calculator calls for one step, then the force read that closes it."""

    def once(cls: type) -> tuple[int, bool]:
        atoms = _pt7(6, scale=0.02)
        calc = CountingEMT()
        atoms.calc = calc
        atoms.get_forces()
        start = atoms.get_positions().copy()
        warmed = calc.calls
        first_moved: list[bool] = []
        original = calc.calculate

        def spy(
            atoms_arg: Atoms | None = None,
            properties: list[str] | None = None,
            system_changes: list[str] | None = None,
        ) -> None:
            if atoms_arg is not None and not first_moved:
                first_moved.append(not np.array_equal(atoms_arg.get_positions(), start))
            original(atoms_arg, properties, system_changes)

        calc.calculate = spy  # type: ignore[method-assign]
        cls(atoms, logfile=None, maxstep=0.05).step()
        atoms.get_forces()
        moved = bool(first_moved and first_moved[0])
        return calc.calls - warmed, moved

    ours, ours_moved = once(optimizer)
    theirs, _theirs_moved = once(ase_optimizer)
    return ours, theirs, ours_moved


def test_fire_calculator_calls_match_ase() -> None:
    from ase.optimize import FIRE as AseFIRE

    ours, theirs, moved = _iteration_calls(FIRE, AseFIRE)
    assert moved
    assert ours == theirs


def test_lbfgs_cache_hit_and_line_search_calls() -> None:
    """The first new EMT call is not the start geometry.

    On this Pt7 start the Wolfe search adds no calculator call beyond
    ASE LBFGS. ``extra`` is that difference.
    """
    from ase.optimize import LBFGS as AseLBFGS

    ours, theirs, moved = _iteration_calls(LBFGS, AseLBFGS)
    extra = ours - theirs
    assert moved
    assert extra == 0


def test_accept_follows_optimizable_type() -> None:
    from ase.build import bulk
    from ase.filters import FrechetCellFilter
    from ase.mep import NEB, MinModeAtoms

    atoms = _pt7(12, scale=0.01)
    plain = FIRE(atoms, logfile=None)._config_options()
    assert "accept" not in plain
    filtered = LBFGS(FrechetCellFilter(bulk("Cu")), logfile=None)
    assert "accept" not in filtered._config_options()

    images = [_pt7(13, scale=0.0), _pt7(13, scale=0.0), _pt7(14, scale=0.01)]
    for image in images:
        image.calc = EMT()
    neb = NEB(images, method="improvedtangent")
    neb_opts = FIRE(neb, logfile=None)._config_options()
    assert neb_opts["accept"] == "step"
    assert neb_opts["neb_guards"] is True
    mode = MinModeAtoms(_pt7(15, scale=0.02), logfile=None)
    mode_opts = FIRE(mode, logfile=None)._config_options()
    assert mode_opts["accept"] == "step"
    assert mode_opts["neb_guards"] is True
    overridden = FIRE(neb, logfile=None, accept="energy")._config_options()
    assert overridden["accept"] == "energy"
    assert "neb_guards" not in overridden
    forced = FIRE(atoms, logfile=None, accept="step")._config_options()
    assert forced["accept"] == "step"


def test_hessian_methods_step() -> None:
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]])

    class Harmonic(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(
            self,
            atoms: Atoms | None = None,
            properties: list[str] | None = None,
            system_changes: list[str] | None = None,
        ) -> None:
            assert atoms is not None
            x = atoms.get_positions().ravel() - np.array([0.0, 0.0, 0.0, 0.7, 0.0, 0.0])
            self.results = {"energy": 0.5 * float(x @ x), "forces": -x.reshape(-1, 3)}

    def hessian(x: np.ndarray) -> np.ndarray:
        return np.eye(x.shape[0])

    atoms.calc = Harmonic()
    from rgmin.ase import Dogleg, Newton, RFO

    for cls in (Newton, RFO, Dogleg):
        sample = atoms.copy()
        sample.calc = Harmonic()
        opt = cls(sample, logfile=None, hessian=hessian, maxstep=0.2)
        assert opt.run(fmax=1e-6, steps=12) is True
    bare = atoms.copy()
    bare.calc = Harmonic()
    with pytest.raises(NotImplementedError, match=r"Newton\.step"):
        Newton(bare, logfile=None).step()


def test_pt7_emt_thread_hash() -> None:
    """Independent EMT relaxations of one start share a trajectory hash."""
    from concurrent.futures import ThreadPoolExecutor

    start = _pt7(11, scale=0.03).positions.copy()

    def once() -> bytes:
        atoms = Atoms("Pt7", positions=start.copy())
        atoms.calc = EMT()
        SD(atoms, logfile=None, maxstep=0.05).run(fmax=1.0, steps=3)
        return np.asarray(atoms.positions, dtype=np.float64).tobytes()

    reference = once()
    with ThreadPoolExecutor(max_workers=2) as pool:
        found = list(pool.map(lambda _i: once(), range(2)))
    assert found == [reference, reference]
