- The Brent line search now keeps its tolerance floor proportional to
  the abscissa, `f64::EPSILON * (|x| + 1 / 1024)`, instead of the fixed
  absolute `1e-12`. Searches whose minimum lies below `1e-12` in step
  length resolve it. Only Brent searches move: `LineSearch::search`,
  `search_from` and the one-shot `minimize` steps with
  `LineSearch::brent` or `LineSearch::default()`, the C one-shot
  `rgmin_minimize` family, which passes `LineSearch::default()`, a
  session after `set_linesearch` with Brent, including the constrained
  L-BFGS curve search, and the caller-driven `SuppliedSearch`,
  `rgmin_linesearch` and Python `LineSearch("brent", ...)`. The session
  default stays strong Wolfe and does not change. Results that were
  already resolved agree to rounding.
