"""Kaplan-Meier estimator of calls-to-success.

Unconverged runs and runs at a different stationary point are
right-censored at the call count they reached. Events at time ``t`` are
removed before censorings at ``t``, so a tie stays in the risk set.
Pointwise bands use the log-log transform of the Greenwood variance.
The median is the smallest ``t`` with ``S(t) <= 0.5``. Its interval is
the Brookmeyer-Crowley set: the times whose log-log band still covers
0.5.
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
        low, high = _loglog_band(survival, greenwood, z)
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


def _loglog_band(survival, greenwood, z):
    """Brookmeyer-Crowley band for one survival value."""
    if not math.isfinite(survival) or survival <= 0.0:
        return 0.0, 0.0
    if survival >= 1.0 or not math.isfinite(greenwood):
        return (1.0, 1.0) if survival >= 1.0 and math.isfinite(greenwood) else (0.0, 1.0)
    if greenwood == 0.0:
        return float(survival), float(survival)
    log_s = math.log(survival)
    theta = math.log(-log_s)
    se = math.sqrt(greenwood) / abs(log_s)
    low = math.exp(-math.exp(theta + z * se))
    high = math.exp(-math.exp(theta - z * se))
    return float(low), float(high)


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
