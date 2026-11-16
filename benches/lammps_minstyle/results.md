# min_style rgmin against LAMMPS minimizers

- rgmin 0.2.0 (C ABI 1.41), git 5a9f3960c35f4b1cadfbb0ec92bbc3908e5bd784-dirty
- LAMMPS (30 Sep 2026 - Update 1), MANYBODY and PLUGIN packages, lmp
- mpirun (Open MPI) 4.1.6
- Linux-6.12.113+-x86_64-with-glibc2.39, Intel(R) Xeon(R) Processor, 8 logical CPUs
- etol 0.0, ftol 1e-06 (two-norm of the global force), maxiter 100000, maxeval 1000000,
  5 repeats per row on 1 MPI rank(s), every style at its defaults.

Iterations and force calls are identical on every repeat.
Wall time is the LAMMPS loop time of the minimize command, in seconds.

| case | ranks | min_style | method | iterations | force calls | energy | fnorm | stop | median s | min s |
|---|---:|---|---|---:|---:|---:|---:|---|---:|---:|
| lj177 | 1 | cg |  | 81 | 162 | -5.4229838073 | 9.768e-07 | force tolerance | 0.00614 | 0.006114 |
| lj177 | 1 | sd |  | 559 | 1118 | -5.4229838073 | 9.648e-07 | force tolerance | 0.03951 | 0.03935 |
| lj177 | 1 | fire |  | 247 | 257 | -5.4229838073 | 8.274e-07 | force tolerance | 0.009137 | 0.009088 |
| lj177 | 1 | quickmin |  | 678 | 678 | -5.4229838073 | 9.807e-07 | force tolerance | 0.02402 | 0.02378 |
| lj177 | 1 | rgmin | lbfgs | 89 | 90 | -5.4229838073 | 7.476e-07 | force tolerance | 0.0046 | 0.004556 |
| lj177 | 1 | rgmin | fire | 237 | 237 | -5.4229838073 | 9.813e-07 | force tolerance | 0.01121 | 0.01109 |
| lj177 | 1 | rgmin | fire2 | 230 | 258 | -5.4229838073 | 8.917e-07 | force tolerance | 0.01189 | 0.01164 |
| cu-vac | 1 | cg |  | 32 | 64 | -901.4819694676 | 8.987e-07 | force tolerance | 0.01321 | 0.01315 |
| cu-vac | 1 | sd |  | 127 | 254 | -901.4819694676 | 9.567e-07 | force tolerance | 0.0509 | 0.05059 |
| cu-vac | 1 | fire |  | 158 | 163 | -901.4819694676 | 7.834e-07 | force tolerance | 0.03314 | 0.03301 |
| cu-vac | 1 | quickmin |  | 243 | 243 | -901.4819694676 | 9.953e-07 | force tolerance | 0.04921 | 0.04884 |
| cu-vac | 1 | rgmin | lbfgs | 35 | 36 | -901.4819694676 | 9.203e-07 | force tolerance | 0.008588 | 0.008528 |
| cu-vac | 1 | rgmin | fire | 154 | 154 | -901.4819694676 | 8.155e-07 | force tolerance | 0.03396 | 0.03379 |
| cu-vac | 1 | rgmin | fire2 | 115 | 120 | -901.4819694676 | 6.442e-07 | force tolerance | 0.02632 | 0.02625 |
| cu-sia-vac | 1 | cg |  | 66 | 129 | -1765.7522759100 | 8.215e-07 | force tolerance | 0.05803 | 0.05718 |
| cu-sia-vac | 1 | sd |  | 305 | 604 | -1765.7522759100 | 9.700e-07 | force tolerance | 0.269 | 0.2648 |
| cu-sia-vac | 1 | fire |  | 228 | 236 | -1765.7522759100 | 9.070e-07 | force tolerance | 0.1066 | 0.105 |
| cu-sia-vac | 1 | quickmin |  | 553 | 553 | -1765.7522759100 | 9.944e-07 | force tolerance | 0.2487 | 0.2448 |
| cu-sia-vac | 1 | rgmin | lbfgs | 64 | 65 | -1765.7522759100 | 8.679e-07 | force tolerance | 0.03302 | 0.03241 |
| cu-sia-vac | 1 | rgmin | fire | 175 | 175 | -1765.7522759100 | 8.499e-07 | force tolerance | 0.08474 | 0.08423 |
| cu-sia-vac | 1 | rgmin | fire2 | 204 | 215 | -1765.7522759100 | 9.447e-07 | force tolerance | 0.1033 | 0.1024 |
| cu32k-vac | 1 | cg |  | 121 | 243 | -113282.1732721033 | 3.881e-06 | linesearch alpha is zero | 5.688 | 5.585 |
| cu32k-vac | 1 | sd |  | 859 | 1723 | -113282.1732716549 | 2.332e-04 | linesearch alpha is zero | 39.03 | 38.8 |
| cu32k-vac | 1 | fire |  | 271 | 275 | -113282.1732721029 | 7.751e-07 | force tolerance | 6.264 | 6.244 |
| cu32k-vac | 1 | quickmin |  | 2680 | 2680 | -113282.1732721038 | 9.986e-07 | force tolerance | 60.65 | 60.32 |
| cu32k-vac | 1 | rgmin | lbfgs | 145 | 146 | -113282.1732720997 | 9.134e-07 | force tolerance | 3.909 | 3.891 |
| cu32k-vac | 1 | rgmin | fire | 270 | 270 | -113282.1732721039 | 9.904e-07 | force tolerance | 6.829 | 6.827 |
| cu32k-vac | 1 | rgmin | fire2 | 766 | 805 | -113282.1732721086 | 9.969e-07 | force tolerance | 20.09 | 20.02 |

## 4 MPI ranks

The same settings on 4 MPI ranks (`mpirun --oversubscribe -np 4`), the 32k cell only.

| case | ranks | min_style | method | iterations | force calls | energy | fnorm | stop | median s | min s |
|---|---:|---|---|---:|---:|---:|---:|---|---:|---:|
| cu32k-vac | 4 | cg |  | 130 | 260 | -113282.1732721032 | 9.595e-07 | force tolerance | 1.759 | 1.553 |
| cu32k-vac | 4 | sd |  | 2449 | 4898 | -113282.1732721022 | 9.957e-07 | force tolerance | 30.16 | 30.05 |
| cu32k-vac | 4 | fire |  | 271 | 275 | -113282.1732721048 | 7.753e-07 | force tolerance | 1.628 | 1.62 |
| cu32k-vac | 4 | quickmin |  | 2680 | 2680 | -113282.1732721020 | 9.990e-07 | force tolerance | 16.04 | 15.84 |
| cu32k-vac | 4 | rgmin | lbfgs | 145 | 146 | -113282.1732721039 | 9.504e-07 | force tolerance | 1.533 | 1.523 |
| cu32k-vac | 4 | rgmin | fire | 270 | 270 | -113282.1732721029 | 9.904e-07 | force tolerance | 2.401 | 2.38 |
| cu32k-vac | 4 | rgmin | fire2 | 766 | 805 | -113282.1732721024 | 9.969e-07 | force tolerance | 6.962 | 6.922 |
