`min_style rgmin` with `method fire` now applies LAMMPS's `delaystep`
(default 20, mapped to `n_min`) and `alpha0` (default 0.25, mapped to
`alpha`) alongside `dtgrow`, `dtshrink`, `alphashrink`, and `tmax`,
replacing the previous `FireConfig::default()` values (`n_min` 5 and
`alpha` 0.1). Iteration counts for FIRE 1.0 runs under `min_style rgmin`
change as a result.
