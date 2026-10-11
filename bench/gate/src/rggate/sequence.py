"""Anytime-valid interval for the paired win probability.

This is the two-sided normal-mixture boundary of Howard, Ramdas,
McAuliffe, and Sekhon, Ann. Stat. 49, 1055 (2021),
doi:10.1214/20-AOS1991. It is not the betting boundary. A win is a
system whose median seed-level log call ratio is negative (the
candidate used fewer calls). Ties are dropped before the walk.

Observations lie in ``[0, 1]``, so the Hoeffding variance proxy is
``1/4`` and ``V_t = t/4``. With mixing scale ``rho = 1`` the half-width
of the sample mean is

    sqrt((V_t + rho) * log((V_t + rho) / (rho * alpha^2))) / t

The interval is clipped to ``[0, 1]``. Clipping does not change whether
``0.5`` lies inside. The walk stops at the first ``t`` whose interval
excludes ``0.5 - margin``: the lower end above that value is a pass,
the upper end below it is a fail. Otherwise the result is undecided and
every informative system was used.

System order is a Philox permutation (stream ``STREAM_ORDER``, counter
index = pair ordinal), applied by the caller before ``walk``.
"""

import math

from rggate.rng import STREAM_ORDER, philox

RHO = 1.0


def half_width(t, alpha, rho=RHO):
    """Normal-mixture half-width of the mean after ``t`` observations."""
    if t <= 0:
        raise ValueError("t must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    variance = t / 4.0
    scale = variance + rho
    return math.sqrt(scale * math.log(scale / (rho * alpha * alpha))) / t


def walk(labels, alpha, margin, rho=RHO):
    """Walk an ordered sequence of 0/1 wins.

    Returns the decision (``pass``, ``fail``, or ``undecided``), the
    number of systems used, and the interval at that time.
    """
    if margin < 0.0:
        raise ValueError("margin must be non-negative")
    threshold = 0.5 - margin
    wins = 0
    low = math.nan
    high = math.nan
    estimate = math.nan
    if not labels:
        return {
            "decision": "undecided",
            "systems_used": 0,
            "low": low,
            "high": high,
            "estimate": estimate,
        }
    for t, label in enumerate(labels, start=1):
        wins += int(label)
        estimate = wins / t
        width = half_width(t, alpha, rho)
        low = max(0.0, estimate - width)
        high = min(1.0, estimate + width)
        if low > threshold:
            decision = "pass"
        elif high < threshold:
            decision = "fail"
        else:
            continue
        return {
            "decision": decision,
            "systems_used": t,
            "low": low,
            "high": high,
            "estimate": estimate,
        }
    return {
        "decision": "undecided",
        "systems_used": len(labels),
        "low": low,
        "high": high,
        "estimate": estimate,
    }


def system_labels(ratios):
    """Map per-seed log ratios to a stable list of ``(system, win)``.

    ``ratios`` is ``(system, seed, log_ratio)``. A system is a win when
    the median log ratio over its seeds is negative, a loss when that
    median is positive, and omitted on a tie or an empty seed list.
    """
    grouped = {}
    for system, _seed, log_ratio in ratios:
        grouped.setdefault(system, []).append(log_ratio)
    labels = []
    for system in sorted(grouped):
        values = sorted(grouped[system])
        mid = values[(len(values) - 1) // 2]
        if len(values) % 2 == 0:
            mid = 0.5 * (values[len(values) // 2 - 1] + values[len(values) // 2])
        if mid < 0.0:
            labels.append((system, 1))
        elif mid > 0.0:
            labels.append((system, 0))
    return labels


def ordered_labels(labels, seed, pair_index):
    """Permute system wins with Philox. ``labels`` is ``(system, win)``."""
    if not labels:
        return []
    rng = philox(seed, STREAM_ORDER, pair_index)
    order = rng.permutation(len(labels))
    return [labels[int(i)][1] for i in order]
