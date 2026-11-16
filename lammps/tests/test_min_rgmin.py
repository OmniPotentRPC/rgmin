"""min_style rgmin against LAMMPS's own minimizers.

Needs ``RGMIN_LAMMPS_LMP`` (an lmp binary with the PLUGIN and MANYBODY
packages), ``RGMIN_LAMMPS_PLUGIN`` (the built ``rgminplugin.so``) and
``LAMMPS_POTENTIALS`` (the LAMMPS ``potentials`` directory).
``RGMIN_LAMMPS_MPIRUN`` overrides the ``mpirun`` launcher of the rank tests,
and ``RGMIN_LAMMPS_RUNNER`` prefixes every ``lmp`` run, such as with valgrind.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import systems  # noqa: E402

LMP = os.environ.get("RGMIN_LAMMPS_LMP", "")
PLUGIN = os.environ.get("RGMIN_LAMMPS_PLUGIN", "")

pytestmark = pytest.mark.skipif(
    not (LMP and PLUGIN), reason="RGMIN_LAMMPS_LMP and RGMIN_LAMMPS_PLUGIN are not set"
)

FTOL = 1e-6
# Both minimizers stop with |F| < 1e-6 in the same basin, where the
# energy is flat to ~|F|^2 / k; the rest is summation order.
ETOL_REL = 1e-10

_reference: dict[tuple, systems.Result] = {}


def rgmin(tmp_path: Path, system: str, modify: str = "", nprocs: int = 1, **kw):
    text = systems.deck(system, "rgmin", modify, plugin=PLUGIN, **kw)
    return systems.run(LMP, text, tmp_path / f"rgmin-{nprocs}", nprocs=nprocs)


def reference(tmp_path: Path, system: str, style: str = "cg", **kw) -> systems.Result:
    key = (system, style, tuple(sorted(kw.items())))
    if key not in _reference:
        text = systems.deck(system, style, **kw)
        _reference[key] = systems.run(LMP, text, tmp_path / style)
    return _reference[key]


def assert_converged(res: systems.Result, ftol: float = FTOL) -> None:
    assert res.returncode == 0, res.error or res.log[-2000:]
    assert res.stop == "force tolerance", res.stop
    assert res.fnorm < ftol * (1 + 1e-9), res.fnorm


@pytest.mark.parametrize("system", ["lj177", "cu-vac", "cu-sia-vac"])
@pytest.mark.parametrize(
    "modify",
    [
        "",
        "min_modify method fire",
        "min_modify method fire2",
        "min_modify method bb",
        "min_modify method lbfgs linesearch wolfe",
        "min_modify method lbfgs trust 0.1 memory 20",
    ],
)
def test_energy_matches_cg(tmp_path, system, modify):
    ref = reference(tmp_path, system)
    assert_converged(ref)
    res = rgmin(tmp_path, system, modify)
    assert_converged(res)
    assert abs(res.pe - ref.pe) <= ETOL_REL * abs(ref.pe), (res.pe, ref.pe)
    # Min::setup's forces seed the first step: one evaluation per
    # iteration at most for the line-search-free default.
    if modify == "":
        assert res.evaluations <= res.iterations + 1


@pytest.mark.parametrize("modify", ["", "min_modify method fire2"])
def test_rank_counts_agree(tmp_path, modify):
    runs = {n: rgmin(tmp_path, "cu-sia-vac", modify, nprocs=n, thermo=1) for n in (1, 2, 4)}
    base = runs[1]
    assert_converged(base)
    for n, res in runs.items():
        assert_converged(res)
        assert res.iterations == base.iterations, (n, res.iterations, base.iterations)
        assert res.evaluations == base.evaluations, (n, res.evaluations, base.evaluations)
        assert len(res.thermo) == len(base.thermo)
        for (s1, e1, f1), (s2, e2, f2) in zip(base.thermo, res.thermo):
            assert s1 == s2
            assert abs(e1 - e2) <= 1e-12 * abs(e1), (n, s1, e1, e2)
            # force sums differ in order between decompositions
            assert abs(f1 - f2) <= 1e-8 * f1 + 1e-3 * FTOL, (n, s1, f1, f2)


FIXES = """
region bottom block INF INF INF INF INF 1.2 units box
group bottom region bottom
fix hold bottom setforce 0.0 0.0 0.0
group top subtract all bottom
fix tether top spring/self 0.5
fix_modify tether energy yes
compute moved bottom displace/atom
compute maxmoved bottom reduce max c_moved[4]
"""


def test_fixes_modify_forces_and_energy(tmp_path):
    ref = reference(tmp_path, "cu-vac", extra=FIXES)
    assert_converged(ref)
    res = rgmin(tmp_path, "cu-vac", extra=FIXES)
    assert_converged(res)
    assert abs(res.pe - ref.pe) <= ETOL_REL * abs(ref.pe), (res.pe, ref.pe)
    plain = rgmin(tmp_path / "plain", "cu-vac")
    assert abs(plain.pe - res.pe) > 1e-6, "spring/self energy did not reach the minimizer"


def test_setforce_atoms_stay(tmp_path):
    text = systems.deck("cu-vac", "rgmin", plugin=PLUGIN, extra=FIXES)
    text += 'print "RGMIN_MOVED $(c_maxmoved:%.17g)"\n'
    res = systems.run(LMP, text, tmp_path)
    assert_converged(res)
    moved = float(res.log.split("RGMIN_MOVED ")[1].split()[0])
    assert moved == 0.0


def test_etol_is_the_lammps_relative_test(tmp_path):
    etol = 1e-9
    res = rgmin(tmp_path, "cu-sia-vac", etol=etol, ftol=0.0, thermo=1)
    assert res.returncode == 0, res.error
    assert res.stop == "energy tolerance"
    energies = [e for _, e, _ in res.thermo]

    def met(a, b):
        return abs(b - a) < etol * 0.5 * (abs(a) + abs(b) + 1e-8)

    assert met(energies[-2], energies[-1])
    assert not any(met(a, b) for a, b in zip(energies[:-2], energies[1:-1]) if a != b)


def test_norm_max_stops_on_the_largest_atom_force(tmp_path):
    ftol = 1e-4
    res = rgmin(tmp_path, "cu-sia-vac", "min_modify norm max", ftol=ftol)
    assert res.returncode == 0, res.error
    assert res.stop == "force tolerance"
    assert res.fmax < ftol * (1 + 1e-9)
    assert res.fnorm > ftol


def test_maxeval_and_maxiter(tmp_path):
    res = rgmin(tmp_path / "eval", "cu-sia-vac", "min_modify method fire2", maxeval=25)
    assert res.stop == "max force evaluations"
    assert res.evaluations >= 25
    res = rgmin(tmp_path / "iter", "cu-sia-vac", maxiter=5, thermo=1)
    assert res.stop == "max iterations"
    assert res.iterations == 5
    # the atoms hold rgmin's last iterate, whose energy thermo printed
    assert res.thermo[-1][0] == 5
    assert res.pe == res.thermo[-1][1]


def test_thermo_norm_yes(tmp_path):
    extra = "thermo_modify norm yes\n"
    ref = reference(tmp_path, "cu-vac", extra=extra)
    res = rgmin(tmp_path, "cu-vac", extra=extra)
    assert_converged(ref)
    assert_converged(res)
    assert abs(res.pe - ref.pe) <= ETOL_REL * abs(ref.pe), (res.pe, ref.pe)
    total = rgmin(tmp_path / "total", "cu-vac")
    assert abs(res.pe * 255 - total.pe) <= ETOL_REL * abs(total.pe)


def test_second_minimize_starts_converged(tmp_path):
    text = systems.deck("cu-vac", "rgmin", plugin=PLUGIN)
    text += f"minimize 0.0 {FTOL!r} 100 1000\n"
    res = systems.run(LMP, text, tmp_path)
    assert res.returncode == 0, res.error
    assert res.log.count("Stopping criterion = force tolerance") == 2
    assert "Iterations, force evaluations = 1 0" in res.log


@pytest.mark.parametrize(
    ("modify", "extra", "message"),
    [
        ("min_modify bogus 1", "", "Unknown min_modify keyword bogus for min_style rgmin"),
        ("min_modify method nonsense", "", "Unknown rgmin method nonsense"),
        ("min_modify method bfgs", "", "stores a dense 3N x 3N matrix"),
        ("min_modify method hz", "", "is not available with min_style rgmin"),
        ("min_modify memory 0", "", "min_modify memory must be at least 1"),
        ("min_modify linesearch armijo", "", "min_modify linesearch is none or wolfe"),
        ("min_modify method fire linesearch wolfe", "", "linesearch wolfe needs"),
        ("min_modify abcfire yes", "", "abcfire yes needs min_modify method fire2"),
        ("min_modify line backtrack", "", "does not take min_modify line"),
        ("min_modify method fire2 tmax 4.0", "", "keeps the Guenole 2020 FIRE parameters"),
        ("", "fix relax all box/relax iso 0.0\n", "fix box/relax"),
    ],
)
def test_errors(tmp_path, modify, extra, message):
    res = rgmin(tmp_path, "cu-vac", modify, extra=extra)
    assert res.returncode != 0
    assert message in res.log, res.error or res.log[-2000:]
