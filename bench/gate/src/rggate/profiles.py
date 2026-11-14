"""Performance, data, and pairwise profiles.

A problem is one ``(system, seed)``. The cost is the call count of a
successful run. A censored run has infinite cost.

Dolan-More: ``rho(tau)`` is the fraction of problems solved within
``tau`` times the best finite cost on that problem. ``tau < 1`` solves
nothing.

More-Wild, adapted to force calls: ``d(alpha)`` is the fraction solved
with cost at most ``alpha * (n + 1)``. Without a ``dim`` column, ``n + 1``
is 1, so ``alpha`` is a force-call budget. With ``dim``, ``n`` is the
free coordinate count (frozen atoms excluded).

Gould-Scott pairwise profiles are emitted only when more than two
methods are present. For solvers A and B, the fraction of problems with
``cost_A <= tau * cost_B``, where an infinite cost is not within any
finite factor of a finite one, and a finite cost beats an infinite one
at every ``tau >= 1``.
"""

import numpy as np


def _costs(problems, methods, solved_cost):
    """solved_cost[(problem, method)] is a finite float or None."""
    table = []
    for method in methods:
        row = []
        for problem in problems:
            cost = solved_cost.get((problem, method))
            row.append(np.inf if cost is None else float(cost))
        table.append(np.asarray(row, dtype=float))
    return table


def performance_profile(problems, methods, solved_cost, taus):
    """Dolan-More ``rho`` at each ``tau``, one value per method."""
    if not methods:
        return []
    costs = _costs(problems, methods, solved_cost)
    n_prob = len(problems)
    out = []
    for tau in taus:
        rhos = []
        if n_prob == 0 or tau < 1.0:
            out.append([0.0] * len(methods))
            continue
        for j in range(n_prob):
            finite = [row[j] for row in costs if np.isfinite(row[j])]
            best = min(finite) if finite else None
            for i, row in enumerate(costs):
                if len(rhos) <= i:
                    rhos.append(0.0)
                if best is not None and np.isfinite(row[j]) and row[j] <= tau * best:
                    rhos[i] += 1.0
        out.append([value / n_prob for value in rhos])
    return out


def data_profile(problems, methods, solved_cost, dims, alphas):
    """More-Wild data profile. ``dims[problem]`` is ``n`` or None."""
    if not methods:
        return []
    costs = _costs(problems, methods, solved_cost)
    n_prob = len(problems)
    out = []
    for alpha in alphas:
        fractions = [0.0] * len(methods)
        if n_prob == 0 or alpha < 0.0:
            out.append(fractions)
            continue
        for j, problem in enumerate(problems):
            n = dims.get(problem)
            unit = 1.0 if n is None else float(n) + 1.0
            budget = alpha * unit
            for i, row in enumerate(costs):
                if np.isfinite(row[j]) and row[j] <= budget:
                    fractions[i] += 1.0
        out.append([value / n_prob for value in fractions])
    return out


def pairwise_profile(problems, methods, solved_cost, taus):
    """Gould-Scott pairwise fractions. Empty when fewer than three methods."""
    if len(methods) < 3:
        return []
    costs = {method: row for method, row in zip(methods, _costs(problems, methods, solved_cost))}
    n_prob = len(problems)
    rows = []
    if n_prob == 0:
        return rows
    for tau in taus:
        for left in methods:
            for right in methods:
                if left == right:
                    continue
                wins = 0
                for j in range(n_prob):
                    a = costs[left][j]
                    b = costs[right][j]
                    if not np.isfinite(a):
                        continue
                    if not np.isfinite(b):
                        if tau >= 1.0:
                            wins += 1
                    elif a <= tau * b:
                        wins += 1
                rows.append((left, right, float(tau), wins / n_prob))
    return rows
