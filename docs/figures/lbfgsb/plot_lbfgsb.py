#!/usr/bin/env python3
"""Summarize and plot the traces of the L-BFGS-B benchmark.

    cargo run --release --example lbfgsb_bench -- starts.txt > rgmin.csv
    python3 docs/figures/lbfgsb/bench_lbfgsb.py starts.txt > scipy.csv
    python3 docs/figures/lbfgsb/plot_lbfgsb.py rgmin.csv scipy.csv

Prints one row per problem and arm: runs, mean and median evaluations
per run, mean final value, worst and median final projected gradient
(infinity norm, at the last accepted iterate), how many runs reached an
accepted gradient of at most 1e-5, the mean evaluations to get there,
the mean evaluations spent after it, and the mean evaluations spent
after the last accepted iterate. Writes ``lj75_tail.png`` next
to this script. Needs matplotlib.
"""

import collections
import csv
import pathlib
import statistics as st
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
TOL = 1e-5
STYLE = {"lbfgs-box": "--"}
LJ75_ARMS = {
    "lbfgs": "rgmin Lbfgs::minimize",
    "lbfgs-box": "rgmin Lbfgs::minimize_box",
    "scipy-m8-noftol": "SciPy, maxcor 8, gtol 1e-6, ftol 0",
    "scipy-default": "SciPy defaults",
}


def load(paths):
    runs = collections.defaultdict(lambda: collections.defaultdict(list))
    for path in paths:
        for r in csv.DictReader(open(path)):
            runs[(r["problem"], r["arm"])][int(r["start"])].append(
                (float(r["f"]), float(r["ginf"]), r["accepted"] == "1"))
    return runs


def reach(trace):
    """Evaluations up to the first accepted iterate with ``ginf <= TOL``."""
    return next((e for e, (_, g, a) in enumerate(trace, 1) if a and g <= TOL), None)


def summary(runs):
    print("problem,arm,runs,evals_mean,evals_median,f_mean,ginf_worst,ginf_median,"
          "reached,evals_to_reach,evals_after,evals_past_last")
    for (prob, arm), starts in runs.items():
        traces = list(starts.values())
        last = [[t for t in tr if t[2]][-1] for tr in traces]
        hits = [(len(tr), reach(tr)) for tr in traces if reach(tr) is not None]
        to = f"{st.mean(h for _, h in hits):.1f}" if hits else ""
        after = f"{st.mean(n - h for n, h in hits):.1f}" if hits else ""
        past = st.mean(len(tr) - max(e for e, t in enumerate(tr, 1) if t[2]) for tr in traces)
        print(f"{prob},{arm},{len(traces)},{st.mean(map(len, traces)):.1f},"
              f"{st.median(map(len, traces)):g},{st.mean(t[0] for t in last):.10g},"
              f"{max(t[1] for t in last):.2e},{st.median(t[1] for t in last):.2e},"
              f"{len(hits)},{to},{after},{past:.1f}")


def best_so_far(trace, horizon):
    out, best = [], float("inf")
    for k in range(horizon):
        if k < len(trace) and trace[k][2]:
            best = min(best, trace[k][1])
        out.append(best)
    return out


def plot(runs):
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.0, 3.6), dpi=150,
                                      gridspec_kw={"width_ratios": [1.6, 1]})
    horizon = 600
    names = [a for a in LJ75_ARMS if ("lj75", a) in runs]
    for arm in names:
        traces = list(runs[("lj75", arm)].values())
        curves = [best_so_far(tr, horizon) for tr in traces]
        median = [st.median(c[k] for c in curves) for k in range(horizon)]
        line, = left.semilogy(range(1, horizon + 1), median, STYLE.get(arm, "-"),
                              label=LJ75_ARMS[arm])
        end = st.median(map(len, traces))
        left.plot(end, median[int(end) - 1], "o", color=line.get_color(), markersize=4)
    left.axhline(TOL, color="grey", linestyle="--", linewidth=1)
    left.set_xlabel("evaluations")
    left.set_ylabel("median best accepted ‖g‖∞")
    left.set_title("LJ75, 400 starts; dot: median run length", fontsize=9)
    left.legend(frameon=False, fontsize=7)

    reached = [a for a in names if any(reach(tr) for tr in runs[("lj75", a)].values())]
    before, after = [], []
    for arm in reached:
        hits = [(len(tr), reach(tr)) for tr in runs[("lj75", arm)].values()]
        before.append(st.mean(h for _, h in hits))
        after.append(st.mean(n - h for n, h in hits))
    ypos = range(len(reached))
    right.barh(ypos, before, color="tab:blue", label="to ‖g‖∞ ≤ 1e-5")
    right.barh(ypos, after, left=before, color="tab:orange", label="after")
    right.set_yticks(list(ypos), [LJ75_ARMS[a].replace(", ", "\n", 1) for a in reached],
                     fontsize=7)
    right.invert_yaxis()
    right.set_xlabel("mean evaluations per relaxation")
    right.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(HERE / "lj75_tail.png", metadata={"Software": None})


def main():
    runs = load(sys.argv[1:])
    summary(runs)
    plot(runs)


if __name__ == "__main__":
    main()
