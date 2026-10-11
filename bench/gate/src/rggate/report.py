"""Assemble the gate summary.

Kaplan-Meier curves are per method. ``--jobs`` may compute those curves
on a pool, then joins them in method-name order. The BCa resamples and
the stratified walk stay on one thread.
"""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import math

import numpy as np
from scipy.stats import beta

from rggate.bootstrap import paired_intervals
from rggate.km import kaplan_meier
from rggate.pairing import classify_pair, row_succeeded
from rggate.profiles import data_profile, pairwise_profile, performance_profile
from rggate.sequence import decide_min, optimal_rho, pair_alpha, stratified_labels, walk, walk_agrapa

PERF_TAUS = (1.0, 1.5, 2.0, 4.0, 8.0)
DATA_ALPHAS = (10.0, 50.0, 100.0, 500.0, 1000.0)
PAIR_TAUS = (1.0, 2.0)
_EXIT = {"pass": 0, "fail": 1, "undecided": 2}


@dataclass
class MethodRow:
    method: str
    n: int
    success: int
    censored: int
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
    call_median: float
    call_min: float
    call_max: float
    wall_median: float
    wall_min: float
    wall_max: float


@dataclass
class SystemPair:
    system: str
    wins: int
    losses: int
    indeterminate: int
    splits: int


@dataclass
class PairRow:
    candidate: str
    baseline: str
    n: int
    geomean: float
    geomean_low: float
    geomean_high: float
    geomean_iid_low: float
    geomean_iid_high: float
    median_ratio: float
    median_low: float
    median_high: float
    median_iid_low: float
    median_iid_high: float
    estimate: float
    low: float
    high: float
    starts_used: int
    starts: int
    wins: int
    losses: int
    indeterminate: int
    splits: int
    split_n: int
    split_low: float
    split_high: float
    decision: str
    energy_only: bool
    clusters: int
    window_fallback: int
    forced: bool
    systems: list


@dataclass
class Report:
    e_tol: float
    rmsd_tol: float
    alpha: float
    margin: float
    resamples: int
    seed: int
    reference: str
    boundary: str
    plan_n: int
    rho: float
    alpha_pair: float
    decide_min: int
    min_clusters: int
    match: str
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


def _collapse(rows, events):
    """One row per (system, method, seed). Compare thread copies when present."""
    groups = defaultdict(list)
    for row, event in zip(rows, events):
        groups[(row.system, row.method, row.seed)].append((row, event))
    saw_threads = False
    bad = []
    chosen = []
    for key in sorted(groups):
        items = groups[key]
        group_threads = any(row.threads is not None for row, _event in items)
        if group_threads:
            saw_threads = True
        if len(items) > 1 and group_threads:
            anchor = items[0][0]
            for row, _event in items[1:]:
                same = (
                    row.converged == anchor.converged
                    and row.calls == anchor.calls
                    and row.energy == anchor.energy
                    and row.fmax == anchor.fmax
                )
                if not same:
                    bad.append(key)
                    break
        items = sorted(
            items,
            key=lambda item: (item[0].threads is None, item[0].threads or 0, item[0].line),
        )
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
        times = [row.calls for row, _event in by_method[name]]
        events = [bool(event) for _row, event in by_method[name]]
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


def _clopper(successes, trials, alpha):
    if trials <= 0:
        return math.nan, math.nan
    if successes <= 0:
        low = 0.0
    else:
        low = float(beta.ppf(alpha / 2.0, successes, trials - successes + 1))
    if successes >= trials:
        high = 1.0
    else:
        high = float(beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
    return low, high


def _lowest(rows):
    best = {}
    for row in rows:
        if row.converged and math.isfinite(row.energy):
            current = best.get(row.system)
            if current is None or row.energy < current:
                best[row.system] = row.energy
    return best


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
    reference="auto",
    plan_n=50,
    boundary="normal-mixture",
    min_clusters=10,
):
    """Run equivalence, censoring, profiles, BCa, and the stopping rule."""
    if jobs < 1:
        raise ValueError("jobs must be >= 1")
    if reference not in ("auto", "lowest"):
        raise ValueError("reference must be auto or lowest")
    if boundary not in ("normal-mixture", "agrapa"):
        raise ValueError("boundary must be normal-mixture or agrapa")
    if plan_n < 1:
        raise ValueError("plan_n must be positive")
    if min_clusters < 1:
        raise ValueError("min_clusters must be positive")
    level = pair_alpha(alpha, len(pairs))
    rho = optimal_rho(plan_n, level)
    earliest = decide_min(level, margin, rho)
    lowest = _lowest(rows)
    cache = {}
    events = [
        row_succeeded(row, reference, lowest.get(row.system), cache, e_tol, rmsd_tol) for row in rows
    ]
    chosen, rank_flag = _collapse(rows, events)
    by_method = defaultdict(list)
    by_system = defaultdict(list)
    for row, event in chosen:
        by_method[row.method].append((row, event))
        by_system[(row.system, row.method)].append((row, event))

    km = _km_map(by_method, alpha, jobs)
    method_rows = []
    for name in sorted(by_method):
        items = by_method[name]
        success = sum(bool(event) for _row, event in items)
        method_rows.append(
            MethodRow(
                name,
                len(items),
                success,
                len(items) - success,
                *_finite_stats([row.calls for row, _event in items]),
                *_finite_stats([row.wall_s for row, _event in items]),
                km[name]["median"],
                km[name]["median_low"],
                km[name]["median_high"],
            )
        )
    system_rows = []
    for key in sorted(by_system):
        items = by_system[key]
        success = sum(bool(event) for _row, event in items)
        system_rows.append(
            SystemRow(
                key[0],
                key[1],
                len(items),
                success,
                len(items) - success,
                *_finite_stats([row.calls for row, _event in items]),
                *_finite_stats([row.wall_s for row, _event in items]),
            )
        )

    flat_rows = [row for row, _event in chosen]
    problems = sorted({(row.system, row.seed) for row in flat_rows})
    methods = sorted({row.method for row in flat_rows})
    solved = {}
    dims = {}
    for row, event in chosen:
        problem = (row.system, row.seed)
        if row.dim is not None:
            dims.setdefault(problem, row.dim)
        if event:
            previous = solved.get((problem, row.method))
            if previous is None or row.calls < previous:
                solved[(problem, row.method)] = row.calls
    perf = performance_profile(problems, methods, solved, PERF_TAUS)
    data = data_profile(problems, methods, solved, dims, DATA_ALPHAS)
    pairwise = pairwise_profile(problems, methods, solved, PAIR_TAUS)

    pair_rows = []
    any_geometry = False
    for index, (candidate, baseline) in enumerate(pairs):
        records = classify_pair(flat_rows, candidate, baseline, e_tol, rmsd_tol, reference)
        pair_rows.append(
            _pair_row(
                candidate,
                baseline,
                records,
                seed,
                index,
                resamples,
                alpha,
                level,
                margin,
                rho,
                boundary,
                min_clusters,
                earliest,
            )
        )
        if records and not all(record.energy_only for record in records):
            any_geometry = True

    if any(row.decision == "fail" for row in pair_rows) or rank_flag == "fail":
        overall = "fail"
    elif any(row.decision == "undecided" for row in pair_rows) or not pair_rows:
        overall = "undecided"
    else:
        overall = "pass"
    return Report(
        e_tol=e_tol,
        rmsd_tol=rmsd_tol,
        alpha=alpha,
        margin=margin,
        resamples=resamples,
        seed=seed,
        reference=reference,
        boundary=boundary,
        plan_n=plan_n,
        rho=rho,
        alpha_pair=level,
        decide_min=earliest,
        min_clusters=min_clusters,
        match="aligned" if any_geometry else "energy-only",
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


def _pair_row(
    candidate,
    baseline,
    records,
    seed,
    index,
    resamples,
    alpha,
    alpha_pair,
    margin,
    rho,
    boundary,
    min_clusters,
    earliest,
):
    wins = [record for record in records if record.kind == "win"]
    losses = [record for record in records if record.kind == "loss"]
    indeterminate = [record for record in records if record.kind == "indeterminate"]
    splits = [record for record in records if record.kind == "split"]
    split_n = len(records)
    split_low, split_high = _clopper(len(splits), split_n, alpha)
    grouped = defaultdict(list)
    for record in records:
        if record.log_ratio is not None and math.isfinite(record.log_ratio):
            grouped[record.system].append(record.log_ratio)
    groups = [np.asarray(grouped[system], dtype=float) for system in sorted(grouped)]
    interval = paired_intervals(groups, seed, resamples, 1.0 - alpha, index)
    clusters = int(interval["n_systems"])
    if clusters < min_clusters:
        interval["cluster_geomean_low"] = math.nan
        interval["cluster_geomean_high"] = math.nan
        interval["cluster_median_low"] = math.nan
        interval["cluster_median_high"] = math.nan
    walk_records = [(record.system, 1 if record.kind == "win" else 0) for record in wins + losses]
    ordered = stratified_labels(walk_records, seed, index)
    if boundary == "agrapa":
        decision = walk_agrapa(ordered, alpha_pair, margin)
    else:
        decision = walk(ordered, alpha_pair, margin, rho)
    by_system = defaultdict(lambda: {"wins": 0, "losses": 0, "indeterminate": 0, "splits": 0})
    for record in records:
        bucket = by_system[record.system]
        if record.kind == "win":
            bucket["wins"] += 1
        elif record.kind == "loss":
            bucket["losses"] += 1
        elif record.kind == "split":
            bucket["splits"] += 1
        else:
            bucket["indeterminate"] += 1
    system_rows = [
        SystemPair(system, counts["wins"], counts["losses"], counts["indeterminate"], counts["splits"])
        for system, counts in sorted(by_system.items())
    ]
    energy_only = bool(records) and all(record.energy_only for record in records)
    fallback = sum(1 for record in records if record.window_fallback)
    forced = len(ordered) < earliest
    return PairRow(
        candidate,
        baseline,
        interval["n"],
        interval["geomean"],
        interval["cluster_geomean_low"],
        interval["cluster_geomean_high"],
        interval["geomean_low"],
        interval["geomean_high"],
        interval["median_ratio"],
        interval["cluster_median_low"],
        interval["cluster_median_high"],
        interval["median_low"],
        interval["median_high"],
        decision["estimate"],
        decision["low"],
        decision["high"],
        decision["starts_used"],
        len(ordered),
        len(wins),
        len(losses),
        len(indeterminate),
        len(splits),
        split_n,
        split_low,
        split_high,
        decision["decision"],
        energy_only,
        clusters,
        fallback,
        forced,
        system_rows,
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
                f"reference={report.reference}",
                f"boundary={report.boundary}",
                f"plan_n={report.plan_n}",
                f"rho={_num(report.rho)}",
                f"alpha_pair={_num(report.alpha_pair)}",
                f"decide_min={report.decide_min}",
                f"min_clusters={report.min_clusters}",
                f"match={report.match}",
            ]
        ),
        f"rank_invariance={report.rank_invariance}",
        "",
        " ".join(
            [
                "method",
                "n",
                "success",
                "censored",
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
    header = [
        "pair",
        "candidate",
        "baseline",
        "n",
        "clusters",
        "geomean",
        "geomean_low",
        "geomean_high",
        "geomean_iid_low",
        "geomean_iid_high",
        "median_ratio",
        "median_low",
        "median_high",
        "median_iid_low",
        "median_iid_high",
        "p_hat",
        "cs_low",
        "cs_high",
        "starts_used",
        "starts",
        "wins",
        "losses",
        "indeterminate",
        "splits",
        "split_n",
        "split_low",
        "split_high",
        "energy_only",
        "window_fallback",
        "forced",
        "decision",
    ]
    lines.append(" ".join(header))
    for row in report.pairs:
        lines.append(
            " ".join(
                [
                    "pair",
                    row.candidate,
                    row.baseline,
                    str(row.n),
                    str(row.clusters),
                    _num(row.geomean),
                    _num(row.geomean_low),
                    _num(row.geomean_high),
                    _num(row.geomean_iid_low),
                    _num(row.geomean_iid_high),
                    _num(row.median_ratio),
                    _num(row.median_low),
                    _num(row.median_high),
                    _num(row.median_iid_low),
                    _num(row.median_iid_high),
                    _num(row.estimate),
                    _num(row.low),
                    _num(row.high),
                    str(row.starts_used),
                    str(row.starts),
                    str(row.wins),
                    str(row.losses),
                    str(row.indeterminate),
                    str(row.splits),
                    str(row.split_n),
                    _num(row.split_low),
                    _num(row.split_high),
                    "yes" if row.energy_only else "no",
                    str(row.window_fallback),
                    "yes" if row.forced else "no",
                    row.decision,
                ]
            )
        )
        for system in row.systems:
            lines.append(
                " ".join(
                    [
                        "start_system",
                        system.system,
                        row.candidate,
                        row.baseline,
                        f"wins={system.wins}",
                        f"losses={system.losses}",
                        f"indeterminate={system.indeterminate}",
                        f"splits={system.splits}",
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
        lines.append(f"pairwise tau={_num(tau)} {left} {right} {_num(fraction)}")
    lines.append("")
    lines.append(f"decision={report.decision}")
    lines.append("")
    return "\n".join(lines)
