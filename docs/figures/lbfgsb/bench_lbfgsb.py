#!/usr/bin/env python3
"""Run SciPy L-BFGS-B on the starts of ``examples/lbfgsb_bench.rs``.

    python3 docs/figures/lbfgsb/bench_lbfgsb.py starts.txt > scipy.csv

``lj75`` starts are relaxed under three option sets:

- ``scipy-default``: SciPy's defaults, ``maxcor = 10``, ``gtol = 1e-5``,
  ``ftol = 2.22e-9`` (stop when one iteration lowers ``f`` by less than
  ``ftol * max(|f|, 1)``).
- ``scipy-m8``: ``maxcor = 8``, ``gtol = 1e-6``, the rgmin defaults,
  with SciPy's ``ftol``.
- ``scipy-m8-noftol``: as ``scipy-m8`` with ``ftol = 0``, so only the
  gradient test, a step that does not lower ``f``, or a failed line
  search ends the run.

``driver1`` starts use the values of the L-BFGS-B driver,
``maxcor = 5`` and ``gtol = 1e-5``, with SciPy's ``ftol``
(``scipy-m5``) and with ``ftol = 0`` (``scipy-m5-noftol``).

The trace goes to stdout as CSV
(``problem,arm,start,eval,f,ginf,accepted``), the columns the Rust
example writes. ``ginf`` is the infinity norm of the projected
gradient. ``accepted`` marks the iterates SciPy reports to its
callback, the start included. The stop message of each run goes to
stderr. Needs numpy and scipy.
"""

import sys

import numpy as np
import scipy
from scipy.optimize import minimize

ARMS = {
    "lj75": {
        "scipy-default": {},
        "scipy-m8": {"maxcor": 8, "gtol": 1e-6},
        "scipy-m8-noftol": {"maxcor": 8, "gtol": 1e-6, "ftol": 0.0},
    },
    "driver1": {
        "scipy-m5": {"maxcor": 5, "gtol": 1e-5},
        "scipy-m5-noftol": {"maxcor": 5, "gtol": 1e-5, "ftol": 0.0},
    },
}


def lj(x):
    """Lennard-Jones value and gradient in reduced units, no cutoff."""
    p = x.reshape(-1, 3)
    i, j = np.triu_indices(len(p), 1)
    d = p[i] - p[j]
    inv2 = 1.0 / np.einsum("ij,ij->i", d, d)
    inv6 = inv2**3
    e = 4.0 * np.sum(inv6 * inv6 - inv6)
    coef = (24.0 * inv2 * (2.0 * inv6 * inv6 - inv6))[:, None] * d
    g = np.zeros_like(p)
    np.subtract.at(g, i, coef)
    np.add.at(g, j, coef)
    return e, g.ravel()


def driver1(x):
    """Extended Rosenbrock function of the L-BFGS-B 3.0 driver1."""
    t = x[1:] - x[:-1] ** 2
    f = 0.25 * (x[0] - 1.0) ** 2 + 4.0 * np.sum(t * t)
    g = np.zeros_like(x)
    g[0] = 0.5 * (x[0] - 1.0)
    g[1:] += 8.0 * t
    g[:-1] -= 16.0 * x[:-1] * t
    return f, g


def problem(name, n):
    if name == "lj75":
        return lj, np.full(n, -np.inf), np.full(n, np.inf)
    lo = np.where(np.arange(n) % 2 == 0, 1.0, -100.0)
    return driver1, lo, np.full(n, 100.0)


def relax(fun, lo, hi, x0, options):
    log = []

    def fg(x):
        f, g = fun(x)
        pg = np.minimum(np.maximum(g, x - hi), x - lo)
        log.append([float(f), float(np.abs(pg).max()), 0])
        return f, g

    def callback(intermediate_result):
        if log and log[-1][0] == intermediate_result.fun:
            log[-1][2] = 1

    bounds = None if np.all(np.isinf(lo)) else list(zip(lo, hi))
    res = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                   callback=callback, options={"maxiter": 3000, **options})
    log[0][2] = 1
    return res, log


def main():
    starts = {}
    for line in open(sys.argv[1]):
        name, *coords = line.split()
        starts.setdefault(name, []).append(np.array(coords, dtype=float))
    print(f"# scipy {scipy.__version__}, numpy {np.__version__}", file=sys.stderr)
    print("problem,arm,start,eval,f,ginf,accepted")
    for name, xs in starts.items():
        fun, lo, hi = problem(name, len(xs[0]))
        for arm, options in ARMS[name].items():
            for k, x0 in enumerate(xs):
                res, log = relax(fun, lo, hi, x0, options)
                for e, (f, gi, acc) in enumerate(log, start=1):
                    print(f"{name},{arm},{k},{e},{f!r},{gi!r},{acc}")
                print(f"{name} {arm} start {k}: {res.nfev} evals, f = {res.fun:.10f}, "
                      f"{res.message}", file=sys.stderr)


if __name__ == "__main__":
    main()
