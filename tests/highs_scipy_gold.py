#!/usr/bin/env python3
"""Gold dest HiGHS constrained QPs with SciPy.

stdin JSON:
  d: two-loop direction (used when H is identity)
  H: optional n-by-n Hessian (row-major nested lists)
  c: optional linear term; default -d (min 1/2 ||p-d||^2)
  lo, hi: optional bounds on p (not x+p)
  A, b: equality A p = b

stdout JSON: {p, fun, success, engine}
engine is scipy.optimize.minimize trust-constr. For an LP
(H all zeros) the script also prints linprog_highs_ipm.
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize


def solve_qp(prob: dict) -> dict:
    d = np.asarray(prob.get("d", []), dtype=float)
    n = int(prob.get("n", d.size))
    if n <= 0:
        raise SystemExit("n or d required")
    if d.size == 0:
        d = np.zeros(n)
    h_raw = prob.get("H")
    if h_raw is None:
        h = np.eye(n)
    else:
        h = np.asarray(h_raw, dtype=float).reshape(n, n)
    c = np.asarray(prob.get("c", (-d).tolist()), dtype=float)
    lo = prob.get("lo")
    hi = prob.get("hi")
    a = np.asarray(prob.get("A") or np.zeros((0, n)), dtype=float)
    if a.size:
        a = a.reshape(-1, n)
    b = np.asarray(prob.get("b") or [], dtype=float)

    def fun(p: np.ndarray) -> float:
        return float(0.5 * p @ h @ p + c @ p)

    def jac(p: np.ndarray) -> np.ndarray:
        return h @ p + c

    lower = np.full(n, -np.inf) if lo is None else np.asarray(lo, dtype=float)
    upper = np.full(n, np.inf) if hi is None else np.asarray(hi, dtype=float)
    cons = []
    if a.shape[0]:
        cons.append(LinearConstraint(a, b, b))
    res = minimize(
        fun,
        np.zeros(n),
        jac=jac,
        method="trust-constr",
        bounds=Bounds(lower, upper),
        constraints=cons,
        options={"xtol": 1e-14, "gtol": 1e-14, "maxiter": 400},
    )
    out = {
        "p": [float(v) for v in res.x],
        "fun": float(res.fun),
        "success": bool(res.success),
        "engine": "trust-constr",
    }
    if np.allclose(h, 0.0):
        lp = linprog(
            c,
            A_eq=a if a.shape[0] else None,
            b_eq=b if a.shape[0] else None,
            bounds=list(zip(lower.tolist(), upper.tolist(), strict=True)),
            method="highs-ipm",
        )
        out["linprog_highs_ipm"] = {
            "p": [float(v) for v in (lp.x if lp.x is not None else [])],
            "fun": float(lp.fun) if lp.success else None,
            "success": bool(lp.success),
        }
    return out


def main() -> None:
    prob = json.load(sys.stdin)
    json.dump(solve_qp(prob), sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
