# MIT License
# Copyright 2026 Rohit Goswami
"""GROMACS ``steep``, ``cg``, and ``l-bfgs`` MDP mapping to ASE units (eV, Å)."""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from typing import Any, SupportsIndex, SupportsInt

GROMACS_KJMOL_NM_TO_EV_ANGSTROM = 1.0 / 964.8533212331001
GROMACS_NM_TO_ANGSTROM = 10.0
_NSTCGSTEEP_MAX = 2**32 - 1
_NBFGSCORR_MAX = 2**32 - 1
_LBFGS_CONSTRAINTS = (
    "The combination of constraints and L-BFGS minimization is not implemented. Either "
    "do not use constraints, or use another minimizer (e.g. steepest descent)."
)


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


def _parse_int(raw: object, *, min_value: int, error_message: str) -> int:
    if isinstance(raw, bool):
        raise ValueError(error_message)
    if not isinstance(raw, (str, bytes, bytearray, SupportsInt, SupportsIndex)):
        raise ValueError(error_message)
    try:
        parsed = int(raw)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(error_message) from None
    if str(parsed) != str(raw).strip() or parsed < min_value:
        raise ValueError(error_message)
    return parsed


def gromacs_mdp_options(
    mdp: Mapping[str, object] | None = None, /, **overrides: object
) -> dict[str, Any]:
    """Map GROMACS ``steep``, ``cg``, or ``l-bfgs`` MDP keys to native options.

    Defaults are ``integrator="steep"``, ``emtol=10.0`` kJ mol⁻¹ nm⁻¹,
    ``emstep=0.01`` nm, ``nsteps=-1`` (no limit), ``nstcgsteep=1000`` (for
    ``integrator="cg"``; ``0`` turns the periodic steepest-descent restart
    off), ``nbfgscorr=10`` (for ``integrator="l-bfgs"``), and
    ``constraints="none"``. ``integrator="cg"`` selects ``gromacs-cg``,
    which pairs the GROMACS-style Polak-Ribière direction update with
    rgmin's strong-Wolfe line search rather than GROMACS's own line search.
    ``integrator="l-bfgs"`` selects ``gromacs-lbfgs`` and follows
    ``do_lbfgs`` (``emtol``, ``emstep``, ``nbfgscorr``). The returned
    mapping can be passed to :func:`rgmin.minimize`; pop ``method`` for
    its positional method argument.
    """
    values: dict[str, object] = {
        "integrator": "steep",
        "emtol": 10.0,
        "emstep": 0.01,
        "nsteps": -1,
        "nstcgsteep": 1000,
        "nbfgscorr": 10,
        "constraints": "none",
    }
    seen_nstcgsteep = False
    seen_nbfgscorr = False
    for source in ({} if mdp is None else mdp, overrides):
        for raw_key, value in source.items():
            key = str(raw_key).strip().lower().replace("-", "_")
            if key not in values:
                raise ValueError(f"unknown GROMACS MDP key {raw_key!r}")
            if key == "nstcgsteep":
                seen_nstcgsteep = True
            if key == "nbfgscorr":
                seen_nbfgscorr = True
            values[key] = value
    integrator = str(values["integrator"]).strip().lower().replace("_", "-")
    if integrator not in ("steep", "cg", "l-bfgs"):
        raise ValueError("integrator must be steep, cg, or l-bfgs")
    constraints = (
        "none"
        if values["constraints"] is None
        else str(values["constraints"]).strip().lower().replace("_", "-")
    )
    if constraints != "none":
        if integrator == "cg":
            raise ValueError(
                "gromacs-cg does not implement constrained conjugate gradients; "
                "set constraints='none' or use integrator='steep'"
            )
        if integrator == "l-bfgs":
            raise ValueError(_LBFGS_CONSTRAINTS)
        raise ValueError(
            "gromacs_mdp_options only accepts constraints='none'; "
            "attach holonomic constraints on the session or ASE Atoms instead"
        )
    if seen_nstcgsteep and integrator != "cg":
        raise ValueError("nstcgsteep applies only to integrator = cg")
    if seen_nbfgscorr and integrator != "l-bfgs":
        raise ValueError("nbfgscorr applies only to integrator = l-bfgs")
    emtol = _positive_finite_float("emtol", values["emtol"])
    emstep = _positive_finite_float("emstep", values["emstep"])
    nsteps = _parse_int(
        values["nsteps"],
        min_value=-1,
        error_message="nsteps must be -1 or a non-negative integer",
    )
    if integrator == "cg":
        nstcgsteep = _parse_int(
            values["nstcgsteep"],
            min_value=0,
            error_message="nstcgsteep must be a non-negative integer",
        )
        if nstcgsteep > _NSTCGSTEEP_MAX:
            raise ValueError("nstcgsteep is out of range")
        return {
            "method": "gromacs-cg",
            "gtol": emtol * GROMACS_KJMOL_NM_TO_EV_ANGSTROM,
            "istep": emstep * GROMACS_NM_TO_ANGSTROM,
            "maxiter": sys.maxsize if nsteps == -1 else nsteps,
            "nstcgsteep": nstcgsteep,
        }
    if integrator == "l-bfgs":
        nbfgscorr = _parse_int(
            values["nbfgscorr"],
            min_value=1,
            error_message="nbfgscorr must be a positive integer",
        )
        if nbfgscorr > _NBFGSCORR_MAX:
            raise ValueError("nbfgscorr is out of range")
        return {
            "method": "gromacs-lbfgs",
            "gtol": emtol * GROMACS_KJMOL_NM_TO_EV_ANGSTROM,
            "istep": emstep * GROMACS_NM_TO_ANGSTROM,
            "maxiter": sys.maxsize if nsteps == -1 else nsteps,
            "nbfgscorr": nbfgscorr,
        }
    return {
        "method": "gromacs-steep",
        "gtol": emtol * GROMACS_KJMOL_NM_TO_EV_ANGSTROM,
        "istep": emstep * GROMACS_NM_TO_ANGSTROM,
        "maxiter": sys.maxsize if nsteps == -1 else nsteps,
    }
