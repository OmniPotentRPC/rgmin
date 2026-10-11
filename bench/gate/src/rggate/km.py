"""Kaplan-Meier estimator of calls-to-success.

Unconverged runs and runs at a different stationary point are
right-censored at the call count they reached. Events at time ``t`` are
removed before censorings at ``t``, so a tie stays in the risk set.
Pointwise bands are the Greenwood variance with a normal quantile,
clipped to ``[0, 1]``. The median is the smallest ``t`` with
``S(t) <= 0.5``. Its interval is the set of times whose Greenwood band
still covers 0.5.
"""

import math

import numpy as np
from scipy.stats import norm


def kaplan_meier(times, events, alpha=0.05):
    """Return the step curve and the median.

    ``times`` and ``events`` are equal-length sequences. ``events[i]`` is
    true when ``times[i]`` is a success. The curve lists one point per
    distinct time, including censoring-only times (the survival does not
    jump there).
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=bool)
    if times.size == 0:
        return _empty()
    order = np.argsort(times, kind="mergesort")
    times = times[order]
    events = events[order]
    unique = np.unique(times)
    n = times.size
    survival = 1.0
    greenwood = 0.0
    z = float(norm.ppf(1.0 - alpha / 2.0))
    curve = []
    at_risk_end = n
    for t in unique:
        at = times == t
        d = int(np.count_nonzero(events[at]))
        n_risk = int(np.count_nonzero(times >= t))
        if d > 0 and n_risk > d:
            survival *= 1.0 - d / n_risk
            greenwood += d / (n_risk * (n_risk - d))
        elif d > 0 and n_risk == d:
            survival = 0.0
            greenwood = math.inf
        var = (survival ** 2) * greenwood if math.isfinite(greenwood) else math.inf
        se = math.sqrt(var) if math.isfinite(var) else math.inf
        low = 0.0 if not math.isfinite(se) else max(0.0, survival - z * se)
        high = 1.0 if not math.isfinite(se) else min(1.0, survival + z * se)
        if not math.isfinite(survival):
            low, high = 0.0, 1.0
        curve.append((float(t), float(survival), float(low), float(high), n_risk, d))
        at_risk_end = n_risk - int(np.count_nonzero(at))
    del at_risk_end
    median, median_low, median_high = _median(curve)
    return {
        "curve": curve,
        "median": median,
        "median_low": median_low,
        "median_high": median_high,
    }


def _median(curve):
    median = None
    for t, survival, _low, _high, _n, _d in curve:
        if survival <= 0.5:
            median = t
            break
    if median is None:
        return None, None, None
    cover = [t for t, _s, low, high, _n, _d in curve if low <= 0.5 <= high]
    if not cover:
        return median, None, None
    return median, cover[0], cover[-1]


def _empty():
    return {"curve": [], "median": None, "median_low": None, "median_high": None}
