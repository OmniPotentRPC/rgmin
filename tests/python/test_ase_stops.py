# Copyright (c) 2026 Rohit Goswami
# SPDX-License-Identifier: MIT
"""``etol`` and ``max_evals`` through the ASE optimizers of ``rgmin.ase``."""

from __future__ import annotations

import io

import numpy as np
import pytest

pytest.importorskip("ase")
from ase import Atoms
from ase.calculators.emt import EMT

import rgmin
from rgmin.ase import FIRE, LBFGS


def _pt7() -> Atoms:
    a = 2.7
    base = np.array(
        [[0, 0, 0], [a, 0, 0], [-a, 0, 0], [0, a, 0], [0, -a, 0], [0, 0, a], [0, 0, -a]],
        dtype=float,
    )
    rng = np.random.default_rng(7)
    atoms = Atoms("Pt7", positions=base + rng.normal(scale=0.1, size=base.shape))
    atoms.calc = EMT()
    return atoms


def test_etol_ends_an_ase_run_before_the_force_test() -> None:
    plain = LBFGS(_pt7(), logfile=None)
    plain.run(fmax=1e-9, steps=200)
    assert plain.stop != rgmin.Stop.ENERGY_TOLERANCE
    cut = LBFGS(_pt7(), logfile=None, etol=1e-4)
    assert cut.run(fmax=1e-9, steps=200) is True
    assert cut.stop == rgmin.Stop.ENERGY_TOLERANCE
    assert cut.nsteps < plain.nsteps


def test_max_evals_ends_an_ase_run_and_reports_false() -> None:
    atoms = _pt7()
    opt = LBFGS(atoms, logfile=None, max_evals=6)
    assert opt.run(fmax=1e-9, steps=200) is False
    assert opt.stop == rgmin.Stop.EVAL_BUDGET


def test_stop_is_none_before_the_first_step_and_after_a_force_stop() -> None:
    opt = FIRE(_pt7(), logfile=None)
    assert opt.stop is None
    assert opt.run(fmax=0.05, steps=500) is True
    assert opt.stop != rgmin.Stop.ENERGY_TOLERANCE


def test_invalid_stops_are_rejected_when_the_optimizer_is_built() -> None:
    with pytest.raises(ValueError, match="etol"):
        LBFGS(_pt7(), logfile=io.StringIO(), etol=-1.0)
    with pytest.raises(ValueError, match="max_evals"):
        LBFGS(_pt7(), logfile=io.StringIO(), max_evals=0)
