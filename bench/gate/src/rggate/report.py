"""Assemble the gate summary.

Kaplan-Meier curves are per method. ``--jobs`` may compute those curves
on a pool, then joins them in method-name order. The BCa resamples and
the system permutation stay on one thread: scipy's generator is consumed
in call order, and a pool would change that order.
"""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import math

import numpy as np

from rggate.bootstrap import log_call_ratios, ratio_intervals
from rggate.equivalence import attach_references
from rggate.km import kaplan_meier
from rggate.profiles import data_profile, pairwise_profile, performance_profile
from rggate.sequence import ordered_labels, system_labels, walk

PERF_TAUS = (1.0, 2.0)
DATA_ALPHAS = (100.0, 1000.0)
_EXIT = {"pass": 0, "fail": 1, "undecided": 2}


@dataclass
class MethodRow:
    method: str
    n: int
    success: int
    censored: int
    different: int
    call_median: float
    call_min: float
    call_max: float
    wall_median: float
    wall_min: float
    wall_max: float
    km_median: float | None
    km_low: float | None
    km_high: float | None


@dataclass
class SystemRow:
    system: str
    method: str
    n: int
    success: int
    censored: int
    different: int
    call_median: float
    call_min: float
    call_max: float
    wall_median: float
    wall_min: float
    wall_max: float


@dataclass
class PairRow:
    candidate: str
    baseline: str
    n: int
    geomean: float
    geomean_low: float
    geomean_high: float
    median_ratio: float
    median_low: float
    median_high: float
    estimate: float
    low: float
    high: float
    systems_used: int
    systems: int
    decision: str


@dataclass
class Report:
    e_tol: float
    rmsd_tol: float
    alpha: float
    margin: float
    resamples: int
    seed: int
    different_stationary: int
    rank_invariance: str
    methods: list
    systems: list
    pairs: list
    method_names: list
    perf_taus: tuple
    perf: list
    data_alphas: tuple
    data: list
    pairwise: list
    km: dict = field(repr=False, default_factory=dict)
    decision: str = "undecided"
    exit_code: int = 2


def _finite_stats(values):
    values = [float(value) for value in values if math.isfinite(value)]
    if not values:
        return math.nan, math.nan, math.nan
    ordered = sorted(values)
    mid = ordered[(len(ordered) - 1) // 2]
    if len(ordered) % 2 == 0:
        mid = 0.5 * (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2])
    return mid, ordered[0], ordered[-1]


def _collapse(rows, status):
    """One row per (system, method, seed). Compare thread copies when present."""
    groups = defaultdict(list)
    for row, label in zip(rows, status):
        groups[(row.system, row.method, row.seed)].append((row, label))
    saw_threads = False
    bad = []
    chosen = []
    for key in sorted(groups):
        items = groups[key]
        group_threads = any(row.threads is not None for row, _label in items)
        if group_threads:
            saw_threads = True
        if len(items) > 1 and group_threads:
            anchor = items[0][0]
            for row, _label in items[1:]:
                same = (
                    row.converged == anchor.converged
                    and row.calls == anchor.calls
                    and row.energy == anchor.energy
                    and row.fmax == anchor.fmax
                )
                if not same:
                    bad.append(key)
                    break
        items = sorted(items, key=lambda item: (item[0].threads is None, item[0].threads or 0, item[0].line))
        chosen.append(items[0])
    if not saw_threads:
        flag = "na"
    elif bad:
        flag = "fail"
    else:
        flag = "ok"
    return chosen, flag


def _km_map(by_method, alpha, jobs):
    names = sorted(by_method)
    payloads = []
    for name in names:
        times = [row.calls for row, _label in by_method[name]]
        events = [label == "success" for _row, label in by_method[name]]
        payloads.append((times, events))
    if jobs <= 1 or len(names) <= 1:
        curves = [kaplan_meier(times, events, alpha) for times, events in payloads]
    else:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = [
                pool.submit(kaplan_meier, times, events, alpha) for times, events in payloads
            ]
            curves = [future.result() for future in futures]
    return {name: curve for name, curve in zip(names, curves)}


def _counts(pairs):
    n = len(pairs)
    success = sum(label == "success" for _row, label in pairs)
    censored = sum(label == "censored" for _row, label in pairs)
    different = sum(label == "different" for _row, label in pairs)
    return n, success, censored, different


def analyze(
    rows,
    pairs,
    e_tol=1e-4,
    rmsd_tol=0.05,
    margin=0.0,
    alpha=0.05,
    resamples=9999,
    seed=1,
    jobs=1,
):
    """Run equivalence, censoring, profiles, BCa, and the stopping rule."""
    if jobs < 1:
        raise ValueError("jobs must be >= 1")
    status = attach_references(rows, e_tol, rmsd_tol)
    chosen, rank_flag = _collapse(rows, status)
    by_method = defaultdict(list)
    by_system = defaultdict(list)
    for row, label in chosen:
        by_method[row.method].append((row, label))
        by_system[(row.system, row.method)].append((row, label))

    km = _km_map(by_method, alpha, jobs)
    method_rows = []
    for name in sorted(by_method):
        items = by_method[name]
        n, success, censored, different = _counts(items)
        call_mid, call_lo, call_hi = _finite_stats([row.calls for row, _label in items])
        wall_mid, wall_lo, wall_hi = _finite_stats([row.wall_s for row, _label in items])
        curve = km[name]
        method_rows.append(
            MethodRow(
                name,
                n,
                success,
                censored,
                different,
                call_mid,
                call_lo,
                call_hi,
                wall_mid,
                wall_lo,
                wall_hi,
                curve["median"],
                curve["median_low"],
                curve["median_high"],
            )
        )
    system_rows = []
    for key in sorted(by_system):
        items = by_system[key]
        n, success, censored, different = _counts(items)
        call_mid, call_lo, call_hi = _finite_stats([row.calls for row, _label in items])
        wall_mid, wall_lo, wall_hi = _finite_stats([row.wall_s for row, _label in items])
        system_rows.append(
            SystemRow(
                key[0],
                key[1],
                n,
                success,
                censored,
                different,
                call_mid,
                call_lo,
                call_hi,
                wall_mid,
                wall_lo,
                wall_hi,
            )
        )

    problems = sorted({(row.system, row.seed) for row, _label in chosen})
    methods = sorted({row.method for row, _label in chosen})
    solved = {}
    dims = {}
    for row, label in chosen:
        problem = (row.system, row.seed)
        if row.dim is not None:
            dims.setdefault(problem, row.dim)
        if label == "success":
            previous = solved.get((problem, row.method))
            if previous is None or row.calls < previous:
                solved[(problem, row.method)] = row.calls
    perf = performance_profile(problems, methods, solved, PERF_TAUS)
    data = data_profile(problems, methods, solved, dims, DATA_ALPHAS)
    pairwise = pairwise_profile(problems, methods, solved, PERF_TAUS)

    flat_rows = [row for row, _label in chosen]
    pair_rows = []
    for index, (candidate, baseline) in enumerate(pairs):
        ratios = log_call_ratios(flat_rows, candidate, baseline)
        logs = np.array([item[2] for item in ratios], dtype=float)
        interval = ratio_intervals(logs, seed, resamples, 1.0 - alpha, index)
        labels = system_labels(ratios)
        ordered = ordered_labels(labels, seed, index)
        decision = walk(ordered, alpha, margin)
        pair_rows.append(
            PairRow(
                candidate,
                baseline,
                interval["n"],
                interval["geomean"],
                interval["geomean_low"],
                interval["geomean_high"],
                interval["median_ratio"],
                interval["median_low"],
                interval["median_high"],
                decision["estimate"],
                decision["low"],
                decision["high"],
                decision["systems_used"],
                len(labels),
                decision["decision"],
            )
        )

    if any(row.decision == "fail" for row in pair_rows) or rank_flag == "fail":
        overall = "fail"
    elif any(row.decision == "undecided" for row in pair_rows) or not pair_rows:
        overall = "undecided"
    else:
        overall = "pass"
    different = sum(row.different for row in system_rows)
    return Report(
        e_tol=e_tol,
        rmsd_tol=rmsd_tol,
        alpha=alpha,
        margin=margin,
        resamples=resamples,
        seed=seed,
        different_stationary=different,
        rank_invariance=rank_flag,
        methods=method_rows,
        systems=system_rows,
        pairs=pair_rows,
        method_names=methods,
        perf_taus=PERF_TAUS,
        perf=perf,
        data_alphas=DATA_ALPHAS,
        data=data,
        pairwise=pairwise,
        km=km,
        decision=overall,
        exit_code=_EXIT[overall],
    )


def _num(value):
    if value is None:
        return "na"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "na"
    if not math.isfinite(number):
        return "na"
    return f"{number:.10g}"


def _rate(success, n):
    if n == 0:
        return "na"
    return _num(success / n)


def render(report):
    """Plain-text summary. The text does not include a wall clock or ``--jobs``."""
    lines = [
        "rggate csv 1",
        " ".join(
            [
                f"e_tol={_num(report.e_tol)}",
                f"rmsd_tol={_num(report.rmsd_tol)}",
                f"alpha={_num(report.alpha)}",
                f"margin={_num(report.margin)}",
                f"resamples={report.resamples}",
                f"seed={report.seed}",
            ]
        ),
        f"different_stationary={report.different_stationary}",
        f"rank_invariance={report.rank_invariance}",
        "",
        " ".join(
            [
                "method",
                "n",
                "success",
                "censored",
                "different",
                "success_rate",
                "call_median",
                "call_min",
                "call_max",
                "wall_median",
                "wall_min",
                "wall_max",
                "km_median",
                "km_low",
                "km_high",
            ]
        ),
    ]
    for row in report.methods:
        lines.append(
            " ".join(
                [
                    row.method,
                    str(row.n),
                    str(row.success),
                    str(row.censored),
                    str(row.different),
                    _rate(row.success, row.n),
                    _num(row.call_median),
                    _num(row.call_min),
                    _num(row.call_max),
                    _num(row.wall_median),
                    _num(row.wall_min),
                    _num(row.wall_max),
                    _num(row.km_median),
                    _num(row.km_low),
                    _num(row.km_high),
                ]
            )
        )
    lines.append("")
    lines.append(
        " ".join(
            [
                "system",
                "method",
                "n",
                "success",
                "censored",
                "different",
                "call_median",
                "call_min",
                "call_max",
                "wall_median",
                "wall_min",
                "wall_max",
            ]
        )
    )
    for row in report.systems:
        lines.append(
            " ".join(
                [
                    row.system,
                    row.method,
                    str(row.n),
                    str(row.success),
                    str(row.censored),
                    str(row.different),
                    _num(row.call_median),
                    _num(row.call_min),
                    _num(row.call_max),
                    _num(row.wall_median),
                    _num(row.wall_min),
                    _num(row.wall_max),
                ]
            )
        )
    lines.append("")
    lines.append(
        " ".join(
            [
                "pair",
                "candidate",
                "baseline",
                "n",
                "geomean",
                "geomean_low",
                "geomean_high",
                "median_ratio",
                "median_low",
                "median_high",
                "p_hat",
                "cs_low",
                "cs_high",
                "systems_used",
                "systems",
                "decision",
            ]
        )
    )
    for row in report.pairs:
        lines.append(
            " ".join(
                [
                    "pair",
                    row.candidate,
                    row.baseline,
                    str(row.n),
                    _num(row.geomean),
                    _num(row.geomean_low),
                    _num(row.geomean_high),
                    _num(row.median_ratio),
                    _num(row.median_low),
                    _num(row.median_high),
                    _num(row.estimate),
                    _num(row.low),
                    _num(row.high),
                    str(row.systems_used),
                    str(row.systems),
                    row.decision,
                ]
            )
        )
    lines.append("")
    for tau, values in zip(report.perf_taus, report.perf):
        for name, value in zip(report.method_names, values):
            lines.append(f"performance tau={_num(tau)} {name} {_num(value)}")
    for alpha, values in zip(report.data_alphas, report.data):
        for name, value in zip(report.method_names, values):
            lines.append(f"data alpha={_num(alpha)} {name} {_num(value)}")
    for left, right, tau, fraction in report.pairwise:
        lines.append(
            f"pairwise tau={_num(tau)} {left} {right} {_num(fraction)}"
        )
    lines.append("")
    lines.append(f"decision={report.decision}")
    lines.append("")
    return "\n".join(lines)
