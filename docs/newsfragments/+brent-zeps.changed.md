- The Brent line search now keeps its tolerance floor proportional to
  the abscissa, `f64::EPSILON / 1024 * (|x| + 1)`, instead of the fixed
  absolute `1e-12`. Searches whose minimum lies below `1e-12` in step
  length resolve it, and the default nonlinear conjugate gradient line
  search can take slightly different steps. Results that were already
  resolved agree to rounding.
