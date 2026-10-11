"""Anytime-valid interval for the win probability over starts.

The target is the probability that the candidate wins a start drawn
from the suite's start distribution. Outcome comparisons sit beside
Goswami, AIP Adv. (2025), doi:10.1063/5.0283639.

The default boundary is the two-sided normal mixture of Howard, Ramdas,
McAuliffe, and Sekhon, Ann. Stat. 49, 1055 (2021),
doi:10.1214/20-AOS1991. Observations lie in ``[0, 1]``, so the Hoeffding
variance proxy is ``V_t = t/4``. The half-width is

    sqrt((V_t + rho) * log((V_t + rho) / (rho * alpha^2))) / t

``rho`` minimises that width at a planned number of starts ``plan_n``.
With ``V = plan_n / 4`` the minimiser solves ``V/rho + 1 = alpha^2
exp(V/rho)``, so ``V/rho + 1 = -W_{-1}(-alpha^2 / e)``. At ``rho = 1``
the interval needs 11 wins in a row to exclude 0.5 at alpha 0.05. The
rho optimised for that same horizon excludes 0.5 after 10.

``--boundary betting`` uses the hedged capital process of Waudby-Smith
and Ramdas, J. R. Stat. Soc. B (2024), doi:10.1093/jrsssb/qkad009.
For each null mean ``m`` on a grid, a predictable stake ``lambda`` in
``(-1/(1-m), 1/m)`` multiplies the capital by ``1 + lambda (X - m)``.
The interval is the set of ``m`` whose capital stays below ``1/alpha``.

The interval is clipped to ``[0, 1]``. The walk stops when it excludes
``0.5 - margin``: the lower end above that value is a pass, the upper
end below it is a fail. Starts are visited in a Philox order stratified
by system: systems are shuffled, starts are shuffled within a system,
then a round robin takes one start from each system in turn.
"""

import math

import numpy as np
from scipy.special import lambertw

from rggate.rng import STREAM_ORDER, philox


def optimal_rho(plan_n, alpha):
    """Mixing scale that minimises the normal-mixture width at ``plan_n``."""
    if plan_n < 1:
        raise ValueError("plan_n must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    variance = plan_n / 4.0
    branch = float(lambertw(-(alpha * alpha) / math.e, k=-1).real)
    ratio = -branch - 1.0
    return variance / ratio


def half_width(t, alpha, rho):
    """Normal-mixture half-width of the mean after ``t`` observations."""
    if t <= 0:
        raise ValueError("t must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if rho <= 0.0:
        raise ValueError("rho must be positive")
    variance = t / 4.0
    scale = variance + rho
    return math.sqrt(scale * math.log(scale / (rho * alpha * alpha))) / t


def walk(labels, alpha, margin, rho):
    """Walk an ordered sequence of 0/1 wins with the normal mixture."""
    if margin < 0.0:
        raise ValueError("margin must be non-negative")
    threshold = 0.5 - margin
    wins = 0
    low = math.nan
    high = math.nan
    estimate = math.nan
    if not labels:
        return _decision("undecided", 0, low, high, estimate)
    for t, label in enumerate(labels, start=1):
        wins += int(label)
        estimate = wins / t
        width = half_width(t, alpha, rho)
        low = max(0.0, estimate - width)
        high = min(1.0, estimate + width)
        if low > threshold:
            return _decision("pass", t, low, high, estimate)
        if high < threshold:
            return _decision("fail", t, low, high, estimate)
    return _decision("undecided", len(labels), low, high, estimate)


def walk_betting(labels, alpha, margin, grid=401):
    """Walk wins with the hedged betting capital process."""
    if margin < 0.0:
        raise ValueError("margin must be non-negative")
    if not labels:
        return _decision("undecided", 0, math.nan, math.nan, math.nan)
    threshold = 0.5 - margin
    means = np.linspace(0.002, 0.998, grid)
    capital = np.ones(grid, dtype=float)
    total = 0.0
    total_sq = 0.0
    low = 0.0
    high = 1.0
    estimate = math.nan
    for t, label in enumerate(labels, start=1):
        if t == 1:
            mu = 0.5
            var = 0.25
        else:
            mu = total / (t - 1)
            var = max(total_sq / (t - 1) - mu * mu, 1e-4)
        stake = (mu - means) / (var + (mu - means) ** 2)
        hi = (1.0 - 1e-8) / means
        lo = -(1.0 - 1e-8) / (1.0 - means)
        stake = np.clip(stake, lo, hi)
        capital *= 1.0 + stake * (float(label) - means)
        np.maximum(capital, 0.0, out=capital)
        total += float(label)
        total_sq += float(label) * float(label)
        estimate = total / t
        alive = capital < (1.0 / alpha)
        if not np.any(alive):
            low, high = 0.0, 1.0
        else:
            kept = means[alive]
            low = float(kept[0])
            high = float(kept[-1])
        if low > threshold:
            return _decision("pass", t, low, high, estimate)
        if high < threshold:
            return _decision("fail", t, low, high, estimate)
    return _decision("undecided", len(labels), low, high, estimate)


def stratified_labels(records, seed, pair_index):
    """Round-robin starts across systems.

    ``records`` is ``(system, label)`` with label 0 or 1. Systems are
    shuffled, starts are shuffled within a system, then one start is
    taken from each system in turn.
    """
    grouped = {}
    for system, label in records:
        grouped.setdefault(system, []).append(int(label))
    if not grouped:
        return []
    systems = sorted(grouped)
    rng = philox(seed, STREAM_ORDER, pair_index)
    system_order = rng.permutation(len(systems))
    systems = [systems[int(i)] for i in system_order]
    columns = []
    for system in systems:
        labels = grouped[system]
        perm = rng.permutation(len(labels))
        columns.append([labels[int(i)] for i in perm])
    ordered = []
    height = max(len(column) for column in columns)
    for step in range(height):
        for column in columns:
            if step < len(column):
                ordered.append(column[step])
    return ordered


def _decision(decision, used, low, high, estimate):
    return {
        "decision": decision,
        "starts_used": used,
        "low": low,
        "high": high,
        "estimate": estimate,
    }
