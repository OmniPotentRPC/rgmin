# rggate

Censored comparison of repeated `rgmin` runs. The CSV is the one
`bench/atomistic` writes. A pixi task `gate` runs `python -m rggate`.

Outcome comparisons sit beside Goswami, AIP Adv. (2025),
doi:10.1063/5.0283639. Calls-to-success use Kaplan and Meier, JASA 53,
457 (1958), doi:10.1080/01621459.1958.10501452. Profiles follow Dolan
and Moré, Math. Program. 91, 201 (2002), doi:10.1007/s101070100263,
Moré and Wild, SIAM J. Optim. 20, 172 (2009), doi:10.1137/080724083,
and, when more than two methods are present, Gould and Scott, ACM TOMS
43 (2016), doi:10.1145/2950048. Paired ratios use the BCa interval,
Efron, JASA 82, 171 (1987), doi:10.1080/01621459.1987.10478410. The
stopping rule is the normal-mixture confidence sequence of Howard,
Ramdas, McAuliffe, and Sekhon, Ann. Stat. 49, 1055 (2021),
doi:10.1214/20-AOS1991, the time-uniform form of Wald, Ann. Math. Stat.
16, 117 (1945), doi:10.1214/aoms/1177731118. `--boundary agrapa` uses
the approximate GRAPA stake of Waudby-Smith and Ramdas, J. R. Stat.
Soc. B (2024), doi:10.1093/jrsssb/qkad009, truncated at 3/4 of full
Kelly. It is one signed capital process.

## Contract version 1

A file is major version 1 when it has no marker, or a marker line

```text
# rggate-csv 1
```

A minor number is accepted (`# rggate-csv 1.2`). Any other major
version is refused. Other `#` lines are comments.

Required columns, in any order:

```text
system,method,seed,converged,iters,calls,energy,fmax,wall_s
```

Optional columns: `pot_s`, `overhead_us_per_iter`, `endpoint`,
`reference`, `dim`, `threads`. `converged` is `0` or `1`. `endpoint` is
a path. `reference` is either an energy or a path. `dim` is the
free coordinate count: three times the number of moving atoms, with
frozen atoms excluded, because `fmax` and the energy bound cover only
those atoms. The data profile uses that same count as `n`. `threads`
is a worker count. When the same `(system, method, seed)` is reported at more than
one `threads` value, those rows must agree on `converged`, `calls`,
`energy`, and `fmax`. The summary prints `rank_invariance=ok`, `fail`,
or `na` when the column is absent.

An endpoint file may start with `energy <float>` and
`cell a11 a12 a13 a21 a22 a23 a31 a32 a33` (row-major lattice vectors).
A zero cell is a free cluster. Remaining lines are `x y z`.

## Equivalence and censoring

The reference is chosen per `(system, seed)`. An explicit `reference`
comes first (an energy, or an endpoint path). Inside a `--pair`, the
baseline endpoint from that same start is next, when the baseline
converged. `--reference lowest` opts into the lowest converged energy
on the system, for a global-search benchmark. Otherwise the two runs
on that start are compared with each other.

Per-method Kaplan-Meier curves and profiles, when no explicit reference
and no `--reference lowest` are in force, treat a converged run as a
success. An unconverged run is right-censored at its call count.

Inside a pair, both runs converging in the same basin is an informative
start. Both converging in different basins is a basin split: the
summary prints the count and a Clopper-Pearson interval, and the start
is left out of the call comparison. It is not a loss. Without
endpoints the match is labelled `energy-only`.

The energy window is `--e-tol` (default `1e-4` eV) when the atom count
`N` is unknown. When `N` is known (`dim / 3`, or the number of sites in
an endpoint) it is `e_tol + N * max(fmax) * rmsd_tol`, from
`|dE| <= ||grad E||_2 ||dx||_2` with `||grad E||_2 <= sqrt(N) fmax` and
`||dx||_2 = sqrt(N) RMSD`. A pair line prints `window_fallback`, the
number of starts judged with the `e_tol` window. An energy-only split
with unknown `N` is an upper bound on a real basin split: the
force-based window is wider. A free cluster is Kabsch-rotated. A
periodic cell is not: after the minimum-image map only the mean
displacement is removed. The RMSD limit is `--rmsd-tol` (default
`0.05` Angstrom).

The median interval on a Kaplan-Meier curve is Brookmeyer-Crowley on
the log-log Greenwood band. The median is the smallest `t` with
`S(t) <= 0.5`.

## Profiles

A problem is one `(system, seed)`. Cost is the call count of a
successful run. Anything else has infinite cost and stays in the
denominator.

Dolan-Moré `rho(tau)` is the fraction of problems solved within `tau`
times the best finite cost. `tau < 1` solves nothing. The summary
prints `tau` of 1, 1.5, 2, 4 and 8.

Moré-Wild `d(alpha)` is the fraction solved with cost at most
`alpha * (n + 1)`. Without `dim`, `n + 1 = 1`, so `alpha` is a
force-call budget. The summary prints `alpha` of 10, 50, 100, 500 and
1000.

Gould-Scott pairwise fractions are printed only when more than two
methods are present. A finite cost beats an infinite one at every
`tau >= 1`. Two infinite costs do not count.

## Paired ratio and stopping

The target is the win probability over the suite's start distribution.
A start is one `(system, seed)`.

Gehan scoring: the candidate wins when it reached the basin in fewer
calls than the baseline's observed count (a later baseline success or a
later baseline censoring). A loss is the mirror image. Both unsolved,
or a censoring earlier than the other side's success, is indeterminate,
counted, and dropped from the walk. A fast unconverged candidate
therefore does not win.

The BCa median and geometric-mean ratio use only starts where both
runs reached the same basin. The summary prints that `n` and the
number of systems in the sample (`clusters`). The geometric mean is
`exp(math.fsum(log ratios) / n)`. The cluster interval resamples
systems, then starts within each drawn system, through
`scipy.stats.bootstrap` with a statistic on the cluster indices. It is
printed only when `clusters` is at least `--min-clusters` (default
10). Below that the cluster bounds are `na` and the i.i.d. interval
remains as a diagnostic (`*_iid_*`). A slow check, 300 replicates and
999 resamples, five starts per system, system effect sd 0.5 and
within-system sd 0.3, covered the ratio 1 at these rates. At 10
systems the median coverage was 0.9400 and the geometric-mean coverage
was 0.9167. At 20 systems the median coverage was 0.9633 and the
geometric-mean coverage was 0.9233.

The generator is NumPy Philox. The fixed key is
`(0xA341316C4C2F6E9A, 0xC1B54A32B6C7E8D9)` with `--seed` XOR-ed into
key word 1. Stream and pair index sit in counter words 2 and 3. Words
0 and 1 are left for draws, so seed `s + 1` is not seed `s` advanced
by one block.

Informative starts are walked in Philox order, stratified by system:
systems are shuffled, starts are shuffled within a system, then a
round robin takes one start from each system. With `k` pairs each walk
uses `alpha_pair = alpha / k` (Bonferroni). The header prints
`alpha_pair`. The default boundary is the two-sided normal mixture
with Hoeffding proxy `V_t = t/4`. `rho` minimises the width at
`--plan-n` (default 50) for `alpha_pair` and is printed.
`--boundary agrapa` uses one signed approximate-GRAPA stake, clipped
at 3/4 of full Kelly.

```text
half-width = sqrt((V_t + rho) * log((V_t + rho) / (rho * alpha_pair^2))) / t
```

`decide_min` is the smallest number of informative starts at which an
all-wins walk or an all-losses walk can cross. Before that count the
pair is undecided, and the pair line says `forced=yes`. At
`--plan-n 50` and one pair, `decide_min` is 11. The walk stops when
the interval excludes `0.5 - --margin` (default margin `0`): the lower
end above that value passes, the upper end below it fails. The summary
prints `starts_used` and the per-system win, loss, indeterminate and
split counts. Several pairs fail the run if any pair fails, and leave
it undecided if any pair is undecided. A rank-invariance mismatch also
fails the run. No pairs is undecided.

Exit codes: `0` pass, `1` fail, `2` undecided. A contract error or an
argument error prints to stderr and exits `3`.

## Jobs

`--jobs` computes the per-method Kaplan-Meier curves on a pool and
joins them in method-name order. The BCa resamples are not
parallelised, so the summary does not depend on `--jobs`. There is no
strong-scaling table. The limiter is that serial `scipy.stats.bootstrap`
call.

## Plots

`--plot out.svg` writes Kaplan-Meier curves and the performance
profile. Matplotlib is optional (`rggate[plot]`, pin `matplotlib==3.10.1`)
and is imported only when `--plot` is set.

## Example

```bash
python -m rggate results.csv \
  --pair lbfgs-exp=lbfgs-wolfe \
  --pair fire-exp=fire \
  --pair cg-exp=cg-wolfe
```
