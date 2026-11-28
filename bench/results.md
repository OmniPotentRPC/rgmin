# Optimizer comparison

Force calls and median wall time. Five serial repeats.
Clusters stop at a max-atom force of 0.01 and a step cap of 0.2.
Rosenbrock stops at an infinity-norm of 1e-5.
rgmin-eon is the default step with memory 8. rgmin-eon-m20 uses memory 20, the eOn default.
rgmin-scaled shortens the quasi-Newton step and keeps the pairs.
rgmin-fire uses the eOn FIRE time step of 1. rgmin-fire2 and ASE FIRE2 use 0.1.
rgmin-cg is the eOn line-search-free conjugate-gradient step.
rgmin-eon, rgmin-fire and rgmin-cg fill the gradient buffer in place.
The eOn columns are the client algorithms transcribed in bench/eon_ref.cpp
from TheochemUI/eOn commit 579738a9387e8d0ea55496f45dd057812cfab892.
SciPy L-BFGS-B has no per-atom cap. Wall times include the potential.
Raw rows are bench/raw/*.csv.

| problem | arm | evals | energy | grad | median ms | converged |
|---|---|---:|---:|---:|---:|---|
| LJ7 | ase-lbfgs | 36 | -1.559321e+01 | 4.494e-03 | 14.090 | yes |
| LJ7 | ase-fire | 78 | -1.559321e+01 | 9.979e-03 | 24.646 | yes |
| LJ7 | ase-fire2 | 47 | -1.559321e+01 | 9.489e-03 | 16.347 | yes |
| LJ13 | ase-lbfgs | 81 | -4.147198e+01 | 7.503e-03 | 37.285 | yes |
| LJ13 | ase-fire | 222 | -3.820647e+01 | 8.075e-03 | 72.760 | yes |
| LJ13 | ase-fire2 | 202 | -3.820646e+01 | 7.738e-03 | 74.336 | yes |
| LJ38 | ase-lbfgs | 208 | -1.644823e+02 | 5.430e-03 | 129.642 | yes |
| LJ38 | ase-fire | 575 | -1.597736e+02 | 9.630e-03 | 228.793 | yes |
| LJ38 | ase-fire2 | 672 | -1.597736e+02 | 9.508e-03 | 293.136 | yes |
| MorsePt256 | ase-lbfgs | 20 | -1.396173e+03 | 9.044e-03 | 110.717 | yes |
| MorsePt256 | ase-fire | 44 | -1.396172e+03 | 8.774e-03 | 231.311 | yes |
| MorsePt256 | ase-fire2 | 28 | -1.396173e+03 | 9.222e-03 | 150.331 | yes |
| Rosenbrock2 | eon-lbfgs | 66 | 7.068399e-17 | 3.224e-07 | 0.021 | yes |
| Rosenbrock10 | eon-lbfgs | 149 | 1.435751e-13 | 8.501e-06 | 0.074 | yes |
| Rosenbrock2 | eon-fire | 2714 | 2.024073e-10 | 9.753e-06 | 0.205 | yes |
| Rosenbrock10 | eon-fire | 4183 | 1.480342e-10 | 9.825e-06 | 0.477 | yes |
| Rosenbrock2 | eon-cg | 87 | 8.407728e-12 | 4.348e-06 | 0.008 | yes |
| Rosenbrock10 | eon-cg | 883 | 8.369137e-12 | 8.099e-06 | 0.119 | yes |
| LJ7 | eon-lbfgs | 33 | -1.559321e+01 | 6.387e-03 | 0.037 | yes |
| LJ7 | eon-fire | 82 | -1.559321e+01 | 8.331e-03 | 0.040 | yes |
| LJ7 | eon-cg | 37 | -1.559321e+01 | 7.274e-03 | 0.019 | yes |
| LJ13 | eon-lbfgs | 85 | -3.820650e+01 | 8.665e-03 | 0.238 | yes |
| LJ13 | eon-fire | 252 | -3.820650e+01 | 8.514e-03 | 0.377 | yes |
| LJ13 | eon-cg | 175 | -3.820650e+01 | 7.681e-03 | 0.269 | yes |
| LJ38 | eon-lbfgs | 191 | -1.597738e+02 | 6.264e-03 | 3.219 | yes |
| LJ38 | eon-fire | 485 | -1.652686e+02 | 3.956e-03 | 5.791 | yes |
| LJ38 | eon-cg | 397 | -1.597738e+02 | 6.802e-03 | 4.770 | yes |
| MorsePt256 | eon-lbfgs | 11 | -1.396173e+03 | 6.474e-03 | 7.011 | yes |
| MorsePt256 | eon-fire | 35 | -1.396173e+03 | 9.628e-03 | 20.454 | yes |
| MorsePt256 | eon-cg | 19 | -1.396173e+03 | 9.260e-03 | 11.481 | yes |
| Rosenbrock2 | rgmin-eon | 66 | 1.536941e-14 | 4.951e-06 | 0.015 | yes |
| Rosenbrock2 | rgmin-eon-m20 | 66 | 7.068399e-17 | 3.224e-07 | 0.018 | yes |
| Rosenbrock2 | rgmin-scaled | 45 | 1.068929e-15 | 1.272e-06 | 0.029 | yes |
| Rosenbrock2 | rgmin-wolfe | 176 | 2.036857e-14 | 3.052e-06 | 0.027 | yes |
| Rosenbrock2 | rgmin-geometry | 101 | 7.956400e-15 | 3.068e-06 | 0.048 | yes |
| Rosenbrock2 | rgmin-fire | 2714 | 2.024073e-10 | 9.753e-06 | 0.155 | yes |
| Rosenbrock2 | rgmin-fire2 | 2949 | 2.021755e-10 | 9.876e-06 | 0.236 | yes |
| Rosenbrock2 | rgmin-cg | 87 | 8.407728e-12 | 4.348e-06 | 0.004 | yes |
| Rosenbrock10 | rgmin-eon | 127 | 1.297030e-13 | 8.560e-06 | 0.050 | yes |
| Rosenbrock10 | rgmin-eon-m20 | 141 | 1.278271e-14 | 3.663e-06 | 0.058 | yes |
| Rosenbrock10 | rgmin-scaled | 80 | 3.986579e+00 | 8.616e-06 | 0.066 | yes |
| Rosenbrock10 | rgmin-wolfe | 310 | 1.154980e-14 | 2.091e-06 | 0.072 | yes |
| Rosenbrock10 | rgmin-geometry | 172 | 2.391928e-13 | 9.381e-06 | 0.125 | yes |
| Rosenbrock10 | rgmin-fire | 4183 | 1.480342e-10 | 9.825e-06 | 0.447 | yes |
| Rosenbrock10 | rgmin-fire2 | 4648 | 1.398306e-10 | 9.327e-06 | 0.634 | yes |
| Rosenbrock10 | rgmin-cg | 883 | 8.411096e-12 | 7.910e-06 | 0.069 | yes |
| LJ7 | rgmin-eon | 33 | -1.559321e+01 | 4.781e-03 | 0.027 | yes |
| LJ7 | rgmin-eon-m20 | 33 | -1.559321e+01 | 6.387e-03 | 0.028 | yes |
| LJ7 | rgmin-scaled | 37 | -1.559320e+01 | 9.851e-03 | 0.061 | yes |
| LJ7 | rgmin-wolfe | 116 | -1.559321e+01 | 4.145e-03 | 0.073 | yes |
| LJ7 | rgmin-geometry | 29 | -1.559321e+01 | 4.251e-03 | 0.043 | yes |
| LJ7 | rgmin-fire | 82 | -1.559321e+01 | 8.331e-03 | 0.043 | yes |
| LJ7 | rgmin-fire2 | 67 | -1.559321e+01 | 9.301e-03 | 0.039 | yes |
| LJ7 | rgmin-cg | 37 | -1.559321e+01 | 7.274e-03 | 0.018 | yes |
| LJ13 | rgmin-eon | 93 | -3.820650e+01 | 9.439e-03 | 0.179 | yes |
| LJ13 | rgmin-eon-m20 | 85 | -3.820650e+01 | 8.665e-03 | 0.187 | yes |
| LJ13 | rgmin-scaled | 106 | -3.820650e+01 | 7.848e-03 | 0.304 | yes |
| LJ13 | rgmin-wolfe | 339 | -3.820650e+01 | 9.357e-03 | 0.569 | yes |
| LJ13 | rgmin-geometry | 103 | -3.820650e+01 | 8.928e-03 | 0.286 | yes |
| LJ13 | rgmin-fire | 252 | -3.820650e+01 | 8.514e-03 | 0.372 | yes |
| LJ13 | rgmin-fire2 | 202 | -3.820649e+01 | 9.950e-03 | 0.311 | yes |
| LJ13 | rgmin-cg | 175 | -3.820650e+01 | 7.681e-03 | 0.243 | yes |
| LJ38 | rgmin-eon | 189 | -1.597738e+02 | 6.781e-03 | 2.400 | yes |
| LJ38 | rgmin-eon-m20 | 186 | -1.597738e+02 | 8.946e-03 | 2.501 | yes |
| LJ38 | rgmin-scaled | 269 | -1.651035e+02 | 6.366e-03 | 3.811 | yes |
| LJ38 | rgmin-wolfe | 642 | -1.675078e+02 | 3.886e-03 | 7.769 | yes |
| LJ38 | rgmin-geometry | 298 | -1.655221e+02 | 8.561e-03 | 4.226 | yes |
| LJ38 | rgmin-fire | 485 | -1.652686e+02 | 3.956e-03 | 5.759 | yes |
| LJ38 | rgmin-fire2 | 589 | -1.597736e+02 | 9.978e-03 | 7.144 | yes |
| LJ38 | rgmin-cg | 397 | -1.597738e+02 | 6.802e-03 | 4.617 | yes |
| MorsePt256 | rgmin-eon | 11 | -1.396173e+03 | 6.474e-03 | 4.914 | yes |
| MorsePt256 | rgmin-eon-m20 | 11 | -1.396173e+03 | 6.474e-03 | 4.909 | yes |
| MorsePt256 | rgmin-scaled | 11 | -1.396173e+03 | 3.933e-03 | 5.035 | yes |
| MorsePt256 | rgmin-wolfe | 37 | -1.396172e+03 | 9.802e-03 | 16.668 | yes |
| MorsePt256 | rgmin-geometry | 11 | -1.396173e+03 | 5.466e-03 | 4.990 | yes |
| MorsePt256 | rgmin-fire | 35 | -1.396173e+03 | 9.628e-03 | 15.613 | yes |
| MorsePt256 | rgmin-fire2 | 46 | -1.396172e+03 | 9.948e-03 | 20.655 | yes |
| MorsePt256 | rgmin-cg | 19 | -1.396173e+03 | 9.260e-03 | 8.419 | yes |
| Rosenbrock2 | scipy-lbfgsb | 45 | 1.705494e-17 | 3.246e-08 | 0.810 | yes |
| Rosenbrock2 | scipy-cg | 78 | 1.880625e-14 | 5.492e-06 | 2.851 | yes |
| Rosenbrock10 | scipy-lbfgsb | 92 | 5.632753e-15 | 1.469e-06 | 2.215 | yes |
| Rosenbrock10 | scipy-cg | 489 | 3.481379e-11 | 5.578e-06 | 20.535 | yes |
| LJ7 | scipy-lbfgsb | 39 | -1.559321e+01 | 6.985e-03 | 2.171 | yes |
| LJ13 | scipy-lbfgsb | 97 | -3.820649e+01 | 1.118e-02 | 6.113 | no |
