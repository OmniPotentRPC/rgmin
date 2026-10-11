"""BCa intervals for the paired log call ratio.

The ratio is kept only for starts where both runs reached the same
basin. ``scipy.stats.bootstrap`` (``method='BCa'``) supplies two
intervals. The cluster interval resamples systems, then resamples
starts inside each drawn system. The i.i.d. interval resamples starts
directly and is a diagnostic. Both use a Philox generator. The printed
geometric mean is ``exp(math.fsum(logs) / n)``.
"""

import math
import warnings

import numpy as np
from scipy.stats import DegenerateDataWarning, bootstrap

from rggate.rng import (
    STREAM_CLUSTER_GEOMEAN,
    STREAM_CLUSTER_MEDIAN,
    STREAM_GEOMEAN,
    STREAM_MEDIAN,
    STREAM_WITHIN_GEOMEAN,
    STREAM_WITHIN_MEDIAN,
    philox,
)


def _mean(sample, axis=-1):
    sample = np.asarray(sample, dtype=float)
    moved = np.moveaxis(sample, axis, -1)
    flat = np.reshape(moved, (-1, moved.shape[-1]))
    width = flat.shape[1]
    out = np.empty(flat.shape[0], dtype=float)
    for index, row in enumerate(flat):
        out[index] = math.fsum(row.tolist()) / width
    return out.reshape(moved.shape[:-1])


def _median(sample, axis=-1):
    return np.median(sample, axis=axis)


def point_estimates(logs):
    """Median ratio and geometric-mean ratio on the original sample."""
    logs = np.asarray(logs, dtype=float)
    logs = logs[np.isfinite(logs)]
    if logs.size == 0:
        return math.nan, math.nan
    ordered = sorted(float(value) for value in logs)
    mid = ordered[(len(ordered) - 1) // 2]
    if len(ordered) % 2 == 0:
        mid = 0.5 * (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2])
    geomean = math.exp(math.fsum(ordered) / len(ordered))
    return math.exp(mid), geomean


def _one(logs, statistic, seed, stream, pair_index, n_resamples, confidence_level):
    rng = philox(seed, stream, pair_index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DegenerateDataWarning)
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
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


def ratio_intervals(logs, seed, n_resamples, confidence_level, pair_index=0):
    """i.i.d. BCa bounds. Bounds are on the ratio scale."""
    logs = np.asarray(logs, dtype=float)
    logs = logs[np.isfinite(logs)]
    median_ratio, geomean = point_estimates(logs)
    median_low = median_high = math.nan
    geomean_low = geomean_high = math.nan
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
        "median_ratio": median_ratio,
        "median_low": median_low,
        "median_high": median_high,
        "geomean": geomean,
        "geomean_low": geomean_low,
        "geomean_high": geomean_high,
    }


def _reduce(logs, kind):
    if logs.size == 0 or not np.any(np.isfinite(logs)):
        return math.nan
    finite = logs[np.isfinite(logs)]
    if kind == "median":
        return float(np.median(finite))
    return math.fsum(finite.tolist()) / finite.size


def _intervals_from_logs(groups, seed, n_resamples, confidence_level, pair_index):
    """Cluster BCa on the log scale, exponentiated at the end."""
    nan = {
        "median_low": math.nan,
        "median_high": math.nan,
        "geomean_low": math.nan,
        "geomean_high": math.nan,
    }
    groups = [np.asarray(group, dtype=float) for group in groups if len(group)]
    if len(groups) < 2:
        return nan
    n_systems = len(groups)
    lengths = np.asarray([group.size for group in groups], dtype=int)
    max_k = int(lengths.max())
    pools = np.zeros((n_systems, max_k), dtype=float)
    for index, group in enumerate(groups):
        pools[index, : group.size] = group

    def make(kind, index_stream, within_stream):
        def statistic(idx, axis=-1):
            del axis
            idx = np.asarray(idx, dtype=int)
            if idx.ndim == 1 and idx.size == n_systems and np.array_equal(idx, np.arange(n_systems)):
                return _reduce(np.concatenate(groups), kind)
            if idx.shape[-1] == n_systems - 1:
                return _leave_one_out(idx, groups, kind)
            return _two_stage(idx, pools, lengths, kind, seed, within_stream, pair_index)

        rng = philox(seed, index_stream, pair_index)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DegenerateDataWarning)
            result = bootstrap(
                (np.arange(n_systems),),
                statistic,
                n_resamples=n_resamples,
                confidence_level=confidence_level,
                method="BCa",
                vectorized=True,
                rng=rng,
            )
        low = float(result.confidence_interval.low)
        high = float(result.confidence_interval.high)
        if math.isfinite(low) and math.isfinite(high):
            return math.exp(low), math.exp(high)
        return math.nan, math.nan

    median_low, median_high = make("median", STREAM_CLUSTER_MEDIAN, STREAM_WITHIN_MEDIAN)
    geomean_low, geomean_high = make("geomean", STREAM_CLUSTER_GEOMEAN, STREAM_WITHIN_GEOMEAN)
    return {
        "median_low": median_low,
        "median_high": median_high,
        "geomean_low": geomean_low,
        "geomean_high": geomean_high,
    }


def _leave_one_out(idx, groups, kind):
    rows = idx if idx.ndim > 1 else idx.reshape(1, -1)
    out = np.empty(rows.shape[0], dtype=float)
    for index, row in enumerate(rows):
        out[index] = _reduce(np.concatenate([groups[int(item)] for item in row]), kind)
    if idx.ndim == 1:
        return float(out[0])
    return out


def _two_stage(idx, pools, lengths, kind, seed, stream, pair_index):
    squeeze = idx.ndim == 1
    if squeeze:
        idx = idx.reshape(1, -1)
    rows, cols = idx.shape
    max_k = pools.shape[1]
    rng = philox(seed, stream, pair_index)
    draws = rng.random((rows, cols, max_k))
    lens = lengths[idx]
    choice = (draws * lens[..., None]).astype(int)
    choice = np.minimum(choice, np.maximum(lens[..., None] - 1, 0))
    picked = np.take_along_axis(pools[idx], choice, axis=-1)
    mask = np.arange(max_k)[None, None, :] < lens[..., None]
    flat = np.where(mask, picked, np.nan).reshape(rows, -1)
    if kind == "median":
        out = np.nanmedian(flat, axis=1)
    else:
        out = np.empty(rows, dtype=float)
        for index in range(rows):
            finite = flat[index, np.isfinite(flat[index])]
            out[index] = math.fsum(finite.tolist()) / finite.size
    if squeeze:
        return float(out[0])
    return out


def paired_intervals(groups, seed, n_resamples, confidence_level, pair_index=0):
    """Cluster BCa and the i.i.d. diagnostic for one pair.

    ``groups`` is a list of log-ratio arrays, one per system.
    """
    logs = np.concatenate([np.asarray(group, dtype=float) for group in groups]) if groups else np.array([])
    iid = ratio_intervals(logs, seed, n_resamples, confidence_level, pair_index)
    cluster = _intervals_from_logs(groups, seed, n_resamples, confidence_level, pair_index)
    iid["cluster_median_low"] = cluster["median_low"]
    iid["cluster_median_high"] = cluster["median_high"]
    iid["cluster_geomean_low"] = cluster["geomean_low"]
    iid["cluster_geomean_high"] = cluster["geomean_high"]
    iid["n_systems"] = sum(1 for group in groups if len(group))
    return iid
