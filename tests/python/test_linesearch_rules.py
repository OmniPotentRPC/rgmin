# Copyright (c) 2026 Rohit Goswami
# SPDX-License-Identifier: MIT
"""A line-search kind the chosen step rule ignores raises ``ValueError``.

Euclidean sessions read the kind on steepest descent, the
conjugate-gradient family, BFGS, Adam and DIIS, unless accept is
``"step"``. L-BFGS reads only ``"wolfe"``, which selects its strong
Wolfe step rule. No other method reads it.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

import rgmin

KINDS = ("brent", "backtracking", "goldstein", "wolfe")
SEARCHED = (
    "sd",
    "steepest",
    "cg",
    "polak_ribiere",
    "fletcher_reeves",
    "hestenes_stiefel",
    "dai_yuan",
    "conjugate_descent",
    "hager_zhang",
    "liu_storey",
    "fr_pr",
    "prp",
    "eon_cg",
    "bfgs",
    "adam",
    "diis",
)
UNSEARCHED = (
    "fire",
    "fire2",
    "quickmin",
    "mdmin",
    "ode12r",
    "bb",
    "sr1",
    "sr2",
    "pso",
    "newton",
    "rfo",
    "dogleg",
    "newton_cg",
    "newton_pcg",
)
SEARCHED_ACCEPTS = (None, "energy", "nonmonotone")


def _rosen(x: np.ndarray) -> tuple[float, np.ndarray]:
    a, b = x[:-1], x[1:]
    g = np.zeros_like(x)
    g[:-1] = -400.0 * a * (b - a * a) - 2.0 * (1.0 - a)
    g[1:] += 200.0 * (b - a * a)
    return float(np.sum(100.0 * (b - a * a) ** 2 + (1.0 - a) ** 2)), g


def _digest(method: str, **options: object) -> str:
    solver = rgmin.Solver(6, rgmin.SolverConfig(method, maxiter=100, gtol=1e-12, **options))
    x = np.array([-1.2, 1.0, -0.5, 0.8, 0.3, -0.7])
    frames = [x.tobytes()]
    for _ in range(8):
        solver.step(_rosen, x)
        frames.append(x.tobytes())
    return hashlib.sha256(b"".join(frames)).hexdigest()


def _options(accept: str | None, **options: object) -> dict[str, object]:
    return options if accept is None else {"accept": accept, **options}


@pytest.mark.parametrize("accept", SEARCHED_ACCEPTS)
@pytest.mark.parametrize("kind", ("brent", "backtracking", "goldstein"))
def test_lbfgs_rejects_a_kind_its_eon_step_ignores(kind: str, accept: str | None) -> None:
    with pytest.raises(ValueError, match=rf"linesearch='{kind}' has no effect on lbfgs.*'wolfe'"):
        rgmin.SolverConfig("lbfgs", **_options(accept, linesearch=kind))


@pytest.mark.parametrize("method", UNSEARCHED)
@pytest.mark.parametrize("kind", KINDS)
def test_a_method_without_a_line_search_rejects_every_kind(method: str, kind: str) -> None:
    with pytest.raises(ValueError, match=rf"linesearch='{kind}' has no effect on {method}"):
        rgmin.SolverConfig(method, linesearch=kind)


@pytest.mark.parametrize("method", ("sd", "cg", "bfgs", "adam", "diis", "lbfgs"))
@pytest.mark.parametrize("kind", KINDS)
def test_accept_step_rejects_every_kind(method: str, kind: str) -> None:
    with pytest.raises(ValueError, match=rf"linesearch='{kind}' has no effect with accept='step'"):
        rgmin.SolverConfig(method, accept="step", linesearch=kind)


@pytest.mark.parametrize("accept", SEARCHED_ACCEPTS)
@pytest.mark.parametrize("method", SEARCHED)
def test_line_searched_methods_take_every_kind(method: str, accept: str | None) -> None:
    default = _digest(method, **_options(accept))
    taken = {kind: _digest(method, **_options(accept, linesearch=kind)) for kind in KINDS}
    # The session default is strong Wolfe with these constants.
    assert taken["wolfe"] == default
    assert taken["brent"] != default


@pytest.mark.parametrize("accept", SEARCHED_ACCEPTS)
def test_lbfgs_takes_wolfe(accept: str | None) -> None:
    assert _digest("lbfgs", **_options(accept, linesearch="wolfe")) != _digest("lbfgs", **_options(accept))


def test_minimize_rejects_an_ignored_kind() -> None:
    with pytest.raises(ValueError, match="has no effect on fire"):
        rgmin.minimize(_rosen, np.zeros(6), method="fire", linesearch="brent", maxiter=1)


def test_constants_follow_the_kind_they_tune() -> None:
    with pytest.raises(ValueError, match="has no effect on lbfgs"):
        rgmin.SolverConfig("lbfgs", linesearch="brent", linesearch_maxiter=5)
    rgmin.SolverConfig("sd", linesearch="brent", linesearch_maxiter=5)


def test_ase_lbfgs_rejects_an_ignored_kind() -> None:
    pytest.importorskip("ase")
    from ase.build import bulk
    from ase.calculators.emt import EMT

    from rgmin.ase import FIRE, LBFGS, SD

    atoms = bulk("Cu", cubic=True)
    atoms.rattle(0.05, seed=3)
    atoms.calc = EMT()
    with pytest.raises(ValueError, match="linesearch='brent' has no effect on lbfgs"):
        LBFGS(atoms, logfile=None, linesearch="brent")
    with pytest.raises(ValueError, match="linesearch='wolfe' has no effect on fire"):
        FIRE(atoms, logfile=None, linesearch="wolfe")
    LBFGS(atoms, logfile=None, linesearch="wolfe").run(fmax=0.05, steps=5)
    SD(atoms, logfile=None, linesearch="backtracking").run(fmax=0.05, steps=5)


def test_ase_names_its_own_step_accept() -> None:
    pytest.importorskip("ase")
    from ase.build import add_adsorbate, fcc100
    from ase.calculators.emt import EMT
    from ase.mep import NEB

    from rgmin.ase import LBFGS

    def image(offset: float) -> object:
        slab = fcc100("Al", size=(2, 2, 1), vacuum=5.0)
        add_adsorbate(slab, "Au", 1.7, "hollow")
        slab.positions[-1, 0] += offset
        slab.calc = EMT()
        return slab

    images = [image(0.0), image(0.0), image(1.5)]
    neb = NEB(images)
    neb.interpolate()
    with pytest.raises(ValueError, match=r"accept='step'.*non-conservative"):
        LBFGS(neb, logfile=None, linesearch="wolfe")
    LBFGS(neb, logfile=None, linesearch="wolfe", accept="energy")
