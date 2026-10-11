"""BCa intervals for the paired log call ratio.

Each observation is ``log(calls_candidate / calls_baseline)`` on one
``(system, seed)``. Censored runs stay in the sample: the call count is
observed even when the run is not a success. Pairing is done by forming
that ratio. ``scipy.stats.bootstrap`` then resamples the ratio vector
(``paired=False`` on a single sample, ``method='BCa'``).

Two intervals, on their own Philox streams:

- ``exp(median(log ratio))``, stream ``STREAM_MEDIAN``
- ``exp(mean(log ratio))``, the geometric-mean ratio, stream
  ``STREAM_GEOMEAN``

The counter's third word is the pair ordinal, so the draws do not
depend on which pair ran first or on ``--jobs``. Fewer than two finite
ratios leaves the bounds undefined.
"""

import math

import numpy as np
from scipy.stats import bootstrap

from rggate.rng import STREAM_GEOMEAN, STREAM_MEDIAN, philox


def log_call_ratios(rows, candidate, baseline):
    """Return ``(system, seed, log_ratio)`` for positive call counts."""
    left = {}
    right = {}
    for row in rows:
        key = (row.system, row.seed)
        if row.method == candidate and row.calls > 0.0 and math.isfinite(row.calls):
            left[key] = row.calls
        elif row.method == baseline and row.calls > 0.0 and math.isfinite(row.calls):
            right[key] = row.calls
    ratios = []
    for key in sorted(left):
        if key not in right:
            continue
        ratios.append((key[0], key[1], math.log(left[key] / right[key])))
    return ratios


def _median(sample, axis=-1):
    return np.median(sample, axis=axis)


def _mean(sample, axis=-1):
    return np.mean(sample, axis=axis)


def _one(logs, statistic, seed, stream, pair_index, n_resamples, confidence_level):
    rng = philox(seed, stream, pair_index)
    result = bootstrap(
        (logs,),
        statistic,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method="BCa",
        paired=False,
        vectorized=True,
        rng=rng,
    )
    low = float(result.confidence_interval.low)
    high = float(result.confidence_interval.high)
    return low, high


def ratio_intervals(logs, seed, n_resamples, confidence_level, pair_index=0):
    """BCa bounds on the median ratio and the geometric-mean ratio.

    ``logs`` is a one-dimensional array of log ratios. The returned
    bounds are on the ratio scale.
    """
    logs = np.asarray(logs, dtype=float)
    logs = logs[np.isfinite(logs)]
    point_median = math.nan
    point_geomean = math.nan
    median_low = median_high = math.nan
    geomean_low = geomean_high = math.nan
    if logs.size:
        point_median = float(np.exp(np.median(logs)))
        point_geomean = float(np.exp(np.mean(logs)))
    if logs.size >= 2:
        median_low, median_high = _one(
            logs, _median, seed, STREAM_MEDIAN, pair_index, n_resamples, confidence_level
        )
        geomean_low, geomean_high = _one(
            logs, _mean, seed, STREAM_GEOMEAN, pair_index, n_resamples, confidence_level
        )
        median_low, median_high = math.exp(median_low), math.exp(median_high)
        geomean_low, geomean_high = math.exp(geomean_low), math.exp(geomean_high)
    return {
        "n": int(logs.size),
        "median_ratio": point_median,
        "median_low": median_low,
        "median_high": median_high,
        "geomean": point_geomean,
        "geomean_low": geomean_low,
        "geomean_high": geomean_high,
    }
