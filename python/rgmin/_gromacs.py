# MIT License
# Copyright 2026 Rohit Goswami
"""GROMACS ``steep`` MDP parameter mapping to ASE units (eV, Å)."""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from typing import Any

GROMACS_KJMOL_NM_TO_EV_ANGSTROM = 1.0 / 964.8533212331001
GROMACS_NM_TO_ANGSTROM = 10.0


def _positive_finite_float(key: str, raw: object) -> float:
    if isinstance(raw, bool):
        raise ValueError(f"{key} must be positive and finite")
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{key} must be positive and finite") from None
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"{key} must be positive and finite")
    return value


def gromacs_mdp_options(
    mdp: Mapping[str, object] | None = None, /, **overrides: object
) -> dict[str, Any]:
    """Map GROMACS ``steep`` MDP keys to native rgmin options in ASE units.

    Defaults are ``emtol=10.0`` kJ mol⁻¹ nm⁻¹, ``emstep=0.01`` nm and
    ``nsteps=-1`` (no limit). The returned mapping can be passed to
    :func:`rgmin.minimize`; pop ``method`` for its positional method argument.
    """
    values: dict[str, object] = {
        "integrator": "steep",
        "emtol": 10.0,
        "emstep": 0.01,
        "nsteps": -1,
    }
    for source in ({} if mdp is None else mdp, overrides):
        for raw_key, value in source.items():
            key = str(raw_key).strip().lower().replace("-", "_")
            if key not in values:
                raise ValueError(f"unknown GROMACS steep MDP key {raw_key!r}")
            values[key] = value
    if str(values["integrator"]).strip().lower() != "steep":
        raise ValueError("integrator must be steep")
    emtol = _positive_finite_float("emtol", values["emtol"])
    emstep = _positive_finite_float("emstep", values["emstep"])
    raw_steps = values["nsteps"]
    if isinstance(raw_steps, bool):
        raise ValueError("nsteps must be -1 or a non-negative integer")
    try:
        nsteps = int(raw_steps)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        raise ValueError("nsteps must be -1 or a non-negative integer") from None
    if str(nsteps) != str(raw_steps).strip() or nsteps < -1:
        raise ValueError("nsteps must be -1 or a non-negative integer")
    return {
        "method": "gromacs-steep",
        "gtol": emtol * GROMACS_KJMOL_NM_TO_EV_ANGSTROM,
        "istep": emstep * GROMACS_NM_TO_ANGSTROM,
        "maxiter": sys.maxsize if nsteps == -1 else nsteps,
    }
