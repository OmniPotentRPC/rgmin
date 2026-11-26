"""Converged energy and force of the ASE classes still missing from rgmin.

Each pair starts from the same rattled geometry and stops at the same
``fmax``. The assertion is the converged energy and the final force, not
matching iterates: PreconLBFGS and PreconFIRE take rgmin's scaled L-BFGS
and FIRE steps. Force-call counts are printed; they are not required to
match.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes
from ase.filters import StrainFilter
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from ase.optimize import MDMin as ASEMDMin
from ase.optimize.ode import ODE12r as ASEODE12r
from ase.optimize.precon import Exp
from ase.optimize.precon import PreconFIRE as ASEPreconFIRE
from ase.optimize.precon import PreconLBFGS as ASEPreconLBFGS

from rgmin.optimize import MDMin, ODE12r, PreconFIRE, PreconLBFGS

# EMT bulk energies agree much more tightly than this. The bound is the
# largest gap the four pairs are allowed on these two systems.
_ENERGY_ABS = 1.0 / 1_000.0
_FMAX_ABS = 1.0 / 100.0
_STEP_ABS = 1.0 / 100_000_000.0


class _Counted(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, inner: Calculator) -> None:
        super().__init__()
        self.inner = inner
        self.calls = 0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.calls += 1
        self.inner.calculate(atoms, properties, system_changes)
        self.results = dict(self.inner.results)


def _counted(atoms, factory):
    copy = atoms.copy()
    copy.calc = _Counted(factory())
    return copy


def _cu():
    atoms = bulk("Cu", cubic=True)
    atoms.rattle(stdev=0.05, seed=1)
    return atoms, EMT


def _lj():
    """Four-atom Lennard-Jones cluster, rattled off the tetrahedron."""
    bond = 2.0 ** (1.0 / 6.0)
    height = bond * (2.0 / 3.0) ** 0.5
    positions = [
        [0.0, 0.0, 0.0],
        [bond, 0.0, 0.0],
        [0.5 * bond, bond * (0.75**0.5), 0.0],
        [0.5 * bond, bond * (0.75**0.5) / 3.0, height],
    ]
    atoms = Atoms("He4", positions=positions, cell=[10.0, 10.0, 10.0], pbc=False)
    atoms.center()
    atoms.rattle(stdev=0.05, seed=3)
    return atoms, LennardJones


def _fmax(atoms):
    forces = atoms.get_forces()
    return float(np.linalg.norm(forces, axis=1).max())


def _relax(optcls, atoms, factory, **kwargs):
    system = _counted(atoms, factory)
    opt = optcls(system, logfile=None, **kwargs)
    converged = bool(opt.run(fmax=0.02, steps=800))
    return system, system.calc.calls, _fmax(system), converged


def _pair_kwargs(ours):
    if ours is PreconLBFGS:
        return {"precon": "Exp"}, {"precon": "Exp"}
    if ours is PreconFIRE:
        return {"precon": "Exp"}, {"precon": Exp()}
    return {}, {}


@pytest.mark.parametrize("builder", [_cu, _lj])
@pytest.mark.parametrize(
    ("ours", "theirs"),
    [
        (MDMin, ASEMDMin),
        (ODE12r, ASEODE12r),
        (PreconLBFGS, ASEPreconLBFGS),
        (PreconFIRE, ASEPreconFIRE),
    ],
)
def test_converged_energy_matches_ase(builder, ours, theirs):
    atoms, calc = builder()
    our_kwargs, ase_kwargs = _pair_kwargs(ours)
    left, left_calls, left_f, left_ok = _relax(ours, atoms, calc, **our_kwargs)
    right, right_calls, right_f, right_ok = _relax(theirs, atoms, calc, **ase_kwargs)
    print(
        f"{ours.__name__} {builder.__name__} force_calls rgmin={left_calls} ase={right_calls} "
        f"converged rgmin={left_ok} ase={right_ok} fmax rgmin={left_f:.6g} ase={right_f:.6g}"
    )
    assert left_ok == right_ok
    if left_ok:
        assert left.get_potential_energy() == pytest.approx(right.get_potential_energy(), abs=_ENERGY_ABS)
        assert left_f == pytest.approx(right_f, abs=_FMAX_ABS)
        assert left_f < 0.05 and right_f < 0.05
    else:
        # ASE itself did not reach fmax on this start. The end points still match.
        assert left.get_potential_energy() == pytest.approx(right.get_potential_energy(), abs=_STEP_ABS)
        assert left_f == pytest.approx(right_f, abs=_STEP_ABS)


def test_mdmin_first_step_matches_ase():
    atoms, calc = _cu()
    left = _counted(atoms, calc)
    right = _counted(atoms, calc)
    MDMin(left, logfile=None, dt=0.2, maxstep=0.2).run(steps=1)
    ASEMDMin(right, logfile=None, dt=0.2, maxstep=0.2).run(steps=1)
    assert left.positions == pytest.approx(right.positions, abs=_STEP_ABS)
    print(f"MDMin first step force_calls rgmin={left.calc.calls} ase={right.calc.calls}")


def test_mdmin_strain_steps_match_ase():
    """Default ``dt`` does not relax a StrainFilter; the strain still matches ASE.

    ASE documents that this filter wants a timestep near ``0.01`` over a
    typical cell length. Both implementations therefore leave the vacancy
    short of the cell minimum, at the same strain.
    """
    atoms = bulk("Cu", cubic=True) * (2, 2, 2)
    del atoms[0]
    atoms.rattle(0.05, seed=1)
    atoms.set_cell(atoms.cell * 1.03, scale_atoms=True)
    left_atoms = atoms.copy()
    right_atoms = atoms.copy()
    left_atoms.calc = EMT()
    right_atoms.calc = EMT()
    left = StrainFilter(left_atoms)
    right = StrainFilter(right_atoms)
    MDMin(left, logfile=None).run(steps=8)
    ASEMDMin(right, logfile=None).run(steps=8)
    assert np.asarray(left.strain) == pytest.approx(np.asarray(right.strain), abs=_STEP_ABS)
    assert left.atoms.get_potential_energy() == pytest.approx(right.atoms.get_potential_energy(), abs=_STEP_ABS)


_RESTART_STEPS = 3


@pytest.mark.parametrize(
    ("optcls", "kwargs"),
    [
        (MDMin, {}),
        (ODE12r, {}),
        (ODE12r, {"precon": "Exp"}),
        (PreconLBFGS, {"precon": "Exp"}),
        (PreconLBFGS, {"precon": None}),
        (PreconFIRE, {}),
        (PreconFIRE, {"precon": "Exp"}),
    ],
    ids=["MDMin", "ODE12r", "ODE12r-Exp", "PreconLBFGS-Exp", "PreconLBFGS-None", "PreconFIRE", "PreconFIRE-Exp"],
)
def test_restart_round_trip_matches_one_run(tmp_path, optcls, kwargs):
    """Write a restart file, replay it in a new optimizer, and match one run."""
    atoms, calc = _cu()
    single = atoms.copy()
    single.calc = calc()
    optcls(single, logfile=None, **kwargs).run(fmax=1e-9, steps=2 * _RESTART_STEPS)

    restart = str(tmp_path / "restart.json")
    first = atoms.copy()
    first.calc = calc()
    optcls(first, restart=restart, logfile=None, **kwargs).run(fmax=1e-9, steps=_RESTART_STEPS)

    second = first.copy()
    second.calc = calc()
    opt = optcls(second, restart=restart, logfile=None, **kwargs)
    opt.run(fmax=1e-9, steps=_RESTART_STEPS)

    assert opt._native is not None
    assert opt._native.steps == 2 * _RESTART_STEPS
    assert second.positions == pytest.approx(single.positions, abs=_STEP_ABS)


def test_ase_preconditioner_names():
    """None and ID leave the metric off. Exp and an Exp instance select it."""
    atoms, calc = _lj()
    system = _counted(atoms, calc)
    for kwargs in ({"precon": None}, {"precon": "ID"}):
        assert PreconLBFGS(system, logfile=None, **kwargs)._precon_mode == "off"
        assert PreconFIRE(system, logfile=None, **kwargs)._precon_mode == "off"
    assert PreconLBFGS(system, logfile=None, precon="Exp")._precon_mode == "exp"
    assert PreconFIRE(system, logfile=None, precon=Exp())._precon_mode == "exp"
    with pytest.warns(UserWarning, match="too small"):
        auto = PreconLBFGS(system, logfile=None)
    assert auto._precon_mode == "off"


def test_unsupported_preconditioner_is_refused():
    atoms, calc = _lj()
    system = _counted(atoms, calc)
    with pytest.raises(ValueError, match="unsupported preconditioner"):
        PreconLBFGS(system, logfile=None, precon="C1")
    with pytest.raises(ValueError, match="unsupported preconditioner"):
        PreconFIRE(system, logfile=None, precon="FF")
    with pytest.raises(ValueError, match="unsupported ODE12r"):
        ODE12r(system, logfile=None, precon="C1")
    with pytest.raises(ValueError, match="rigid_units"):
        PreconLBFGS(system, logfile=None, rigid_units=[[0, 1]])


def test_ode12r_accepts_ase_preconditioner_instances():
    """ASE passes a Precon instance through. Exp and identity follow that."""
    from ase.optimize.precon import C1
    from ase.optimize.precon.precon import IdentityPrecon

    atoms, calc = _lj()
    system = _counted(atoms, calc)
    assert ODE12r(system, logfile=None, precon=Exp())._precon_mode == "exp"
    assert ODE12r(system, logfile=None, precon=IdentityPrecon())._precon_mode == "off"
    with pytest.raises(ValueError, match="unsupported ODE12r"):
        ODE12r(system, logfile=None, precon=C1())


def test_preconfire_builds_exp_after_a_downhill_reset(monkeypatch):
    """A rejected step must not be the geometry the Exp metric is built on."""
    import rgmin.optimize as optmod

    class _Outward(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            Calculator.calculate(self, atoms, properties, system_changes)
            pos = np.asarray(atoms.get_positions(), dtype=float)
            self.results["energy"] = float(np.sum(pos * pos))
            self.results["forces"] = pos.copy()

    bond = 1.2
    atoms = Atoms(
        "He3",
        positions=[[0.2, 0.1, 0.0], [bond, 0.0, 0.0], [0.4, bond, 0.0]],
        cell=[8.0, 8.0, 8.0],
        pbc=False,
    )
    atoms.calc = _Outward()
    installed: list[np.ndarray] = []
    rejected: list[np.ndarray] = []
    real_install = optmod._install_exp

    def spy(native, optimizable, x=None):
        installed.append(np.asarray(optimizable.get_x(), dtype=float).copy())
        return real_install(native, optimizable, x)

    def on_reset(optimizable, _r_last, _e, _e_last):
        rejected.append(np.asarray(optimizable.get_x(), dtype=float).copy())

    monkeypatch.setattr(optmod, "_install_exp", spy)
    PreconFIRE(
        atoms,
        logfile=None,
        precon="Exp",
        downhill_check=True,
        position_reset_callback=on_reset,
    ).run(fmax=0.0, steps=2)
    assert rejected, "the outward force should raise the energy and reset"
    for bad in rejected:
        assert not any(np.allclose(bad, got, rtol=0.0, atol=1e-8) for got in installed)


def test_exp_estimates_mu_once_while_the_matrix_holds():
    """A held Exp matrix pays for one mu probe, not one probe on every step."""
    atoms, calc = _lj()
    plain = _counted(atoms, calc)
    ODE12r(plain, logfile=None).run(fmax=1e-12, steps=6)
    metric = _counted(atoms, calc)
    ODE12r(metric, logfile=None, precon="Exp").run(fmax=1e-12, steps=6)
    assert metric.calc.calls == plain.calc.calls + 1
