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
16, 117 (1945), doi:10.1214/aoms/1177731118.

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
coordinate count `n` used by the data profile. `threads` is a worker
count. When the same `(system, method, seed)` is reported at more than
one `threads` value, those rows must agree on `converged`, `calls`,
`energy`, and `fmax`. The summary prints `rank_invariance=ok`, `fail`,
or `na` when the column is absent.

An endpoint file may start with `energy <float>` and
`cell a11 a12 a13 a21 a22 a23 a31 a32 a33` (row-major lattice vectors).
A zero cell is a free cluster. Remaining lines are `x y z`.

## Equivalence and censoring

The per-system reference energy is the lowest converged energy on that
system, unless a row sets `reference`. A run is a success only when it
converged, `|E - E_ref| <= --e-tol` (default `1e-4` eV), and, when both
geometries exist, the Kabsch RMSD after the minimum-image map is at
most `--rmsd-tol` (default `0.05` Angstrom). Atoms are paired by index.
A converged run that fails either test is a different stationary point.
An unconverged run, or a non-finite energy, is right-censored at its
call count and is not counted as a different stationary point.

Kaplan-Meier is computed per method on that censored sample. The curve
uses the Greenwood variance and a normal quantile, clipped to `[0, 1]`.
The median is the smallest `t` with `S(t) <= 0.5`.

## Profiles

A problem is one `(system, seed)`. Cost is the call count of a
successful run. Anything else has infinite cost and stays in the
denominator.

Dolan-Moré `rho(tau)` is the fraction of problems solved within `tau`
times the best finite cost. `tau < 1` solves nothing. The summary
prints `tau = 1` and `tau = 2`.

Moré-Wild `d(alpha)` is the fraction solved with cost at most
`alpha * (n + 1)`. Without `dim`, `n + 1 = 1`, so `alpha` is a
force-call budget. The summary prints `alpha = 100` and
`alpha = 1000`.

Gould-Scott pairwise fractions are printed only when more than two
methods are present. A finite cost beats an infinite one at every
`tau >= 1`. Two infinite costs do not count.

## Paired ratio and stopping

For each `--pair candidate=baseline` and each `(system, seed)` where
both call counts are positive, the observation is
`log(calls_candidate / calls_baseline)`, including censored runs.
`scipy.stats.bootstrap` (`method='BCa'`) returns an interval for the
median ratio and for the geometric-mean ratio. The generator is NumPy
`Philox` with key `(0xA341316C4C2F6E9A, 0xC1B54A32B6C7E8D9)`. The
counter is `(--seed, stream, pair index, 0)`. Stream 2 is the median,
stream 3 is the geometric mean, stream 1 shuffles systems.

A system is a win when the median log ratio over its seeds is negative.
The wins are walked in that Philox order. The boundary is the two-sided
normal mixture with Hoeffding proxy `V_t = t/4` and `rho = 1`:

```text
half-width = sqrt((V_t + rho) * log((V_t + rho) / (rho * alpha^2))) / t
```

This is not the betting boundary. The walk stops when the interval
excludes `0.5 - --margin` (default margin `0`): lower end above that
value passes, upper end below it fails. The summary prints
`systems_used`. Several pairs fail the run if any pair fails, and leave
it undecided if any pair is undecided. A rank-invariance mismatch also
fails the run. No pairs is undecided.

Exit codes: `0` pass, `1` fail, `2` undecided. A contract error prints
to stderr and exits `2`.

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
