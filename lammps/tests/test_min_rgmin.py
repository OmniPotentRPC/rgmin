"""min_style rgmin against LAMMPS's own minimizers.

Needs ``RGMIN_LAMMPS_LMP`` (an lmp binary with the PLUGIN and MANYBODY
packages), ``RGMIN_LAMMPS_PLUGIN`` (the built ``rgminplugin.so``) and
``LAMMPS_POTENTIALS`` (the LAMMPS ``potentials`` directory).
``RGMIN_LAMMPS_MPIRUN`` overrides the ``mpirun`` launcher of the rank tests,
and ``RGMIN_LAMMPS_RUNNER`` prefixes every ``lmp`` run, such as with valgrind.
"""

from __future__ import annotations

import itertools
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


BOX_RELAX = "fix relax all box/relax iso 0.0\n"
BOX_VMAX = 0.001
BOX_DOF = {"iso": 1, "aniso": 3, "tri": 6}
BOX_PRINT = (
    "thermo_style custom step pe vol pxx pyy pzz pxy pxz pyz\nrun 0\n"
    'print "RGMIN_BOX vol $(vol:%.17g) pxx $(pxx:%.17g) pyy $(pyy:%.17g) pzz $(pzz:%.17g)'
    ' pxy $(pxy:%.17g) pxz $(pxz:%.17g) pyz $(pyz:%.17g)"\n'
)
# fextra is about 1e-3 eV per bar of residual stress in these 108-atom
# cells, so |F| < FTOL leaves every component under 1e-2 bar.
BOX_PRESS = 1e-2
# Both styles stop on the same strain-free crystal; the residual stress
# leaves the volume within B / |P| ~ 1e-8 of it.
BOX_VOL_REL = 1e-7


def box_relax(mode: str, target: str = "0.0") -> str:
    return f"fix relax all box/relax {mode} {target} vmax {BOX_VMAX!r}\n"


def box_state(res: systems.Result) -> dict[str, float]:
    words = res.log.split("RGMIN_BOX ")[1].split("\n")[0].split()
    return {k: float(v) for k, v in zip(words[::2], words[1::2])}


def box_run(tmp_path: Path, system: str, style: str, extra: str, modify: str = "", **kw):
    text = systems.deck(
        system, style, modify, extra=extra, plugin=PLUGIN if style == "rgmin" else None, **kw
    )
    return systems.run(LMP, text + BOX_PRINT, tmp_path)


def assert_same_cell(res: systems.Result, ref: systems.Result, press: dict[str, float]) -> None:
    got, want = box_state(res), box_state(ref)
    for key in ("pxx", "pyy", "pzz", "pxy", "pxz", "pyz"):
        assert abs(got[key] - press.get(key, 0.0)) < BOX_PRESS, (key, got[key])
    assert abs(got["vol"] - want["vol"]) <= BOX_VOL_REL * want["vol"], (got["vol"], want["vol"])
    assert abs(res.pe - ref.pe) <= ETOL_REL * abs(ref.pe), (res.pe, ref.pe)


@pytest.mark.parametrize(
    ("system", "mode"),
    [("cu-strained", "iso"), ("cu-strained", "aniso"), ("cu-strained-tri", "tri")],
)
@pytest.mark.parametrize(
    "modify", ["", "min_modify method bb", "min_modify method lbfgs linesearch wolfe"]
)
def test_box_relax_matches_cg(tmp_path, system, mode, modify):
    ref = box_run(tmp_path / "cg", system, "cg", box_relax(mode))
    assert_converged(ref)
    res = box_run(tmp_path / "rgmin", system, "rgmin", box_relax(mode), modify)
    assert_converged(res)
    assert f"rgmin: {BOX_DOF[mode]} box coordinates" in res.log
    assert_same_cell(res, ref, {})


def test_box_relax_on_three_ranks(tmp_path):
    extra = box_relax("tri")
    ref = box_run(tmp_path / "cg", "cu-strained-tri", "cg", extra)
    text = systems.deck("cu-strained-tri", "rgmin", extra=extra, plugin=PLUGIN) + BOX_PRINT
    res = systems.run(LMP, text, tmp_path / "rgmin", nprocs=3)
    assert_converged(ref)
    assert_converged(res)
    assert_same_cell(res, ref, {})


def test_box_relax_deviatoric_target_resets_the_reference(tmp_path):
    # A non-hydrostatic target with nreset makes fix box/relax move its
    # reference cell during the run (Modify::min_reset_ref).
    extra = "fix relax all box/relax x 0.0 y 0.0 z 5000.0 couple none nreset 20 vmax 0.001\n"
    ref = box_run(tmp_path / "cg", "cu-strained", "cg", extra)
    res = box_run(tmp_path / "rgmin", "cu-strained", "rgmin", extra)
    assert_converged(ref)
    assert_converged(res)
    assert_same_cell(res, ref, {"pzz": 5000.0})


def test_box_relax_steps_the_cell_at_most_vmax(tmp_path):
    extra = box_relax("iso") + "group one id 1\ndump cell one atom 1 cell.dump\n"
    res = box_run(tmp_path, "cu-strained", "rgmin", extra)
    assert_converged(res)
    lines = (tmp_path / "cell.dump").read_text().splitlines()
    lengths = [
        float(lines[i + 1].split()[1]) - float(lines[i + 1].split()[0])
        for i, line in enumerate(lines)
        if line.startswith("ITEM: BOX BOUNDS")
    ]
    steps = [abs(b - a) for a, b in itertools.pairwise(lengths)]
    # fix box/relax measures ds in units of the box of the run's start
    assert max(steps) <= BOX_VMAX * lengths[0] * (1 + 1e-9), max(steps)
    shrink = (lengths[0] - lengths[-1]) / lengths[0]
    assert shrink > 0.02, shrink
    assert len(steps) >= shrink / BOX_VMAX


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
        ("min_modify method fire", BOX_RELAX, "use min_modify method lbfgs, sd or bb"),
        ("min_modify method fire2", BOX_RELAX, "use min_modify method lbfgs, sd or bb"),
        ("min_modify trust 0.1", BOX_RELAX, "min_modify trust does not bound the box step"),
        ("min_modify dmax 0.0", BOX_RELAX, "needs min_modify dmax > 0 with fix box/relax"),
    ],
)
def test_errors(tmp_path, modify, extra, message):
    res = rgmin(tmp_path, "cu-vac", modify, extra=extra)
    assert res.returncode != 0
    assert message in res.log, res.error or res.log[-2000:]
