# ASE against rgmin

ASE 3.29.0 and rgmin, seven serial repeats, median wall time.
fmax 0.01, maxstep 0.2, step budget 100000.
rgmin L-BFGS memory is 8. ASE LBFGS memory is its default, 100,
and its initial inverse-Hessian scale is its default, 1/70.
ASE FIRE and rgmin FIRE 2.0 use timestep 0.1 and a Euclidean
step cap of 0.2. ASE LBFGS, rgmin capped L-BFGS and the geometry
step cap the longest atomic displacement at 0.2. rgmin Wolfe is
the line search without that cap. No parameter is refit per system.
Evaluations are oracle calls. They do not change across repeats.
Wall time is the whole run in that language, the optimizer and
the potential together. ASE is Python and rgmin is Rust, so the
milliseconds are not a comparison of the optimizer loops alone.
Equal energies are the same basin. A lower energy is a deeper one,
and the force counts of those rows are counts to different endpoints.

| problem | arm | evals | energy | fmax | median ms | converged |
|---|---|---:|---:|---:|---:|---|
| LJ7 | ase-lbfgs | 36 | -15.593210 | 4.494e-03 | 14.030 | yes |
| LJ7 | ase-fire | 78 | -15.593209 | 9.979e-03 | 24.422 | yes |
| LJ7 | ase-bfgslinesearch | 44 | -15.593210 | 7.470e-03 | 16.777 | yes |
| LJ7 | rgmin-capped | 37 | -15.593205 | 9.851e-03 | 0.055 | yes |
| LJ7 | rgmin-wolfe | 116 | -15.593210 | 4.145e-03 | 0.050 | yes |
| LJ7 | rgmin-geometry | 29 | -15.593210 | 4.251e-03 | 0.037 | yes |
| LJ7 | rgmin-fire2 | 67 | -15.593207 | 9.301e-03 | 0.051 | yes |
| LJ13 | ase-lbfgs | 81 | -41.471979 | 7.503e-03 | 36.842 | yes |
| LJ13 | ase-fire | 222 | -38.206466 | 8.075e-03 | 71.798 | yes |
| LJ13 | ase-bfgslinesearch | 116 | -39.469516 | 6.111e-03 | 46.002 | yes |
| LJ13 | rgmin-capped | 106 | -38.206503 | 7.848e-03 | 0.209 | yes |
| LJ13 | rgmin-wolfe | 339 | -38.206503 | 9.357e-03 | 0.259 | yes |
| LJ13 | rgmin-geometry | 103 | -38.206502 | 8.928e-03 | 0.198 | yes |
| LJ13 | rgmin-fire2 | 202 | -38.206487 | 9.950e-03 | 0.250 | yes |
| LJ38 | ase-lbfgs | 208 | -164.482333 | 5.430e-03 | 128.448 | yes |
| LJ38 | ase-fire | 575 | -159.773576 | 9.630e-03 | 225.296 | yes |
| LJ38 | ase-bfgslinesearch | 428 | -166.160678 | 7.423e-03 | 246.840 | yes |
| LJ38 | rgmin-capped | 267 | -165.103449 | 8.824e-03 | 1.510 | yes |
| LJ38 | rgmin-wolfe | 628 | -167.507766 | 9.301e-03 | 2.258 | yes |
| LJ38 | rgmin-geometry | 306 | -165.522101 | 6.672e-03 | 1.731 | yes |
| LJ38 | rgmin-fire2 | 589 | -159.773614 | 9.978e-03 | 2.721 | yes |
| MorsePt256 | ase-lbfgs | 20 | -1396.172971 | 9.044e-03 | 94.284 | yes |
| MorsePt256 | ase-fire | 44 | -1396.172348 | 8.774e-03 | 190.487 | yes |
| MorsePt256 | ase-bfgslinesearch | 11 | -1396.173012 | 6.224e-03 | 102.579 | yes |
| MorsePt256 | rgmin-capped | 11 | -1396.173090 | 3.933e-03 | 5.428 | yes |
| MorsePt256 | rgmin-wolfe | 37 | -1396.172211 | 9.802e-03 | 16.904 | yes |
| MorsePt256 | rgmin-geometry | 11 | -1396.173031 | 5.466e-03 | 5.391 | yes |
| MorsePt256 | rgmin-fire2 | 46 | -1396.172268 | 9.948e-03 | 21.007 | yes |

The chained Rosenbrock function
`sum_i (1 - x_i)^2 + 100 (x_{i+1} - x_i^2)^2`
has a local minimum for every n >= 4 in the basin of
`(-1, 1, ..., 1)`. SciPy BFGS started there converges
with a positive definite Hessian. For n = 4 the value is
3.701 and the first coordinate is -0.776. For n = 10 the
value is 3.987 and the first coordinate is -0.993, near
`(-1, 1, ..., 1)` with value about 3.98. For n = 3 that
start falls through to the valley floor at 0. A minimizer
that stops at the local minimum has converged.
SciPy BFGS from `(-1, 1, ..., 1)`:

```
n=3 f=0.000000 x0=1.000000 xmin=4.752e-01 ginf=8.882e-14
n=4 f=3.701429 x0=-0.775659 xmin=3.795e-01 ginf=9.282e-09
n=10 f=3.986579 x0=-0.993263 xmin=5.013e-01 ginf=2.561e-11
```
