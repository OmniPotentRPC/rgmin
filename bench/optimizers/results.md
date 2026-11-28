# Optimizer coverage benchmark

Release builds, 1 pinned thread, 12 order-flipped rounds, no warm-up and no rescue reruns. Convergence is 0.01 eV/A; cell rows also require 0.001 eV/A^3. Energy agreement tolerance is 0.001 eV.

Machine: Intel(R) Xeon(R) Processor (physical_cores=8, logical_cpus=8, threads_per_core=1); flags: fpu, vme, de, pse, tsc, msr, pae, mce, cx8, apic, sep, mtrr, ... (128 flags total); architecture: x86_64; available logical CPUs: 8/8; kernel: 6.12.113+; Python: 3.12.3; Rust: rustc 1.89.0 (29483883e 2025-08-04); C++: GNU-13.3.0; BLAS: blas=scipy-openblas 0.3.34.237.2 [OpenBLAS 0.3.34.237.2 USE64BITINT DYNAMIC_ARCH NO_AFFINITY SkylakeX MAX_THREADS=64]; lapack=scipy-openblas 0.3.34.237.2; runtime=openblas 0.3.34.237.2 (libscipy_openblas64_-4b78795a.so); thread env: MKL_NUM_THREADS=1, NUMEXPR_NUM_THREADS=1, OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.

P-values below are within-run timing-noise tests only (Mann-Whitney on wall and outside-potential timings).

| System | Method | Code | Success | Wall ms, median [IQR] | Outside potential ms | Force calls to converge | Final energy eV | Wall timing p | Outside timing p | Energy agreement |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cu108-cell | cell-lbfgs | rgmin | 12/12 | 257 [256, 259] | 219 [218, 220] | 22 [22, 22] | -292 [-292, -292] | — | — | — |
| Cu108-cell | cell-lbfgs | ase | 12/12 | 160 [158, 161] | 125 [123, 126] | 20 [20, 20] | -292 [-292, -292] | 3.66e-05 | 3.66e-05 | 12/12; max Δ=2.55e-05 |
| Cu108-cell | cell-lbfgs | eon | gap | — | — | — | — | — | — | The eOn client optimizer does not expose variable-cell relaxation. |
| Cu255-vacancy | fire | rgmin | 12/12 | 38.4 [38.2, 38.5] | 25 [24.9, 25.2] | 45 [45, 45] | -684 [-684, -684] | — | — | — |
| Cu255-vacancy | fire | ase | 12/12 | 32.1 [31.9, 32.2] | 19.3 [19.1, 19.3] | 44 [44, 44] | -684 [-684, -684] | 3.66e-05 | 3.66e-05 | 12/12; max Δ=0.000224 |
| Cu255-vacancy | fire | eon | 12/12 | 18.5 [18.2, 18.7] | 5.78 [5.64, 5.96] | 43 [43, 43] | -684 [-684, -684] | 3.66e-05 | 3.66e-05 | 12/12; max Δ=0.000396 |
| Cu255-vacancy | lbfgs | rgmin | 12/12 | 11.4 [11.3, 11.7] | 7 [6.9, 7.25] | 12 [12, 12] | -684 [-684, -684] | — | — | — |
| Cu255-vacancy | lbfgs | ase | 12/12 | 15.8 [15.5, 16.2] | 9.73 [9.52, 10.1] | 18 [18, 18] | -684 [-684, -684] | 3.66e-05 | 3.66e-05 | 12/12; max Δ=0.000423 |
| Cu255-vacancy | lbfgs | eon | 12/12 | 6.4 [6.35, 6.45] | 1.95 [1.91, 1.99] | 12 [12, 12] | -684 [-684, -684] | 3.66e-05 | 3.66e-05 | 12/12; max Δ=6.82e-13 |
| Cu255-vacancy | sd | rgmin | 12/12 | 14.9 [14.8, 15.1] | 8.89 [8.83, 9.08] | 18 [18, 18] | -684 [-684, -684] | — | — | — |
| Cu255-vacancy | sd | ase | gap | — | — | — | — | — | — | No steepest-descent comparator was requested for ASE. |
| Cu255-vacancy | sd | eon | 0/12 | 390 [389, 392] | 124 [123, 125] | — | — | 3.66e-05 | 3.66e-05 | n/a |
| LJ38 | fire | rgmin | 120/120 | 37.3 [34.7, 40.8] | 35.8 [33.3, 39.2] | 85 [78, 92] | -174 [-174, -174] | — | — | — |
| LJ38 | fire | ase | 120/120 | 25.3 [24.1, 26.4] | 24.1 [22.9, 25.1] | 69.5 [67, 73] | -174 [-174, -174] | 1.93e-39 | 1.93e-39 | 120/120; max Δ=3.95e-05 |
| LJ38 | fire | eon | 120/120 | 7.14 [6.82, 7.53] | 5.81 [5.55, 6.1] | 86.5 [83, 91] | -174 [-174, -174] | 7.14e-41 | 7.14e-41 | 120/120; max Δ=2.42e-05 |
| LJ38 | lbfgs | rgmin | 120/120 | 10.3 [9.77, 11.1] | 9.83 [9.35, 10.6] | 21.5 [21, 24] | -174 [-174, -174] | — | — | — |
| LJ38 | lbfgs | ase | 120/120 | 46.6 [41.4, 48.8] | 45 [39.9, 47.2] | 89.5 [82, 95] | -174 [-174, -174] | 7.14e-41 | 7.14e-41 | 120/120; max Δ=1.61e-05 |
| LJ38 | lbfgs | eon | 120/120 | 2.13 [1.98, 2.22] | 1.72 [1.6, 1.8] | 21.5 [21, 24] | -174 [-174, -174] | 2.55e-40 | 2.61e-40 | 120/120; max Δ=2.83e-06 |
| LJ38 | sd | rgmin | 120/120 | 39.1 [29.5, 57.1] | 37 [28, 54.1] | 114 [90, 185] | -174 [-174, -174] | — | — | — |
| LJ38 | sd | ase | gap | — | — | — | — | — | — | No steepest-descent comparator was requested for ASE. |
| LJ38 | sd | eon | 0/120 | 80.4 [79.7, 83.1] | 65.5 [65, 67.7] | — | — | 2.39e-38 | 3.44e-37 | n/a |
| LJ75 | fire | rgmin | 120/120 | 39 [38.3, 41] | 34.9 [34.2, 36.6] | 80 [79, 84] | -397 [-397, -397] | — | — | — |
| LJ75 | fire | ase | 120/120 | 29 [27.9, 30.4] | 25.4 [24.3, 26.5] | 72.5 [69, 75] | -397 [-397, -397] | 7.14e-41 | 7.14e-41 | 120/120; max Δ=2.65e-05 |
| LJ75 | fire | eon | 120/120 | 10.6 [9.8, 12] | 6.35 [5.88, 7.18] | 86 [80, 99] | -397 [-397, -397] | 7.14e-41 | 7.14e-41 | 120/120; max Δ=1.98e-05 |
| LJ75 | lbfgs | rgmin | 120/120 | 12.3 [11, 12.6] | 11 [9.82, 11.2] | 23.5 [21, 24] | -397 [-397, -397] | — | — | — |
| LJ75 | lbfgs | ase | 0/120 | 712 [706, 719] | 660 [655, 668] | — | — | 7.14e-41 | 7.14e-41 | n/a |
| LJ75 | lbfgs | eon | 120/120 | 3.16 [2.87, 3.25] | 1.96 [1.79, 2.01] | 23 [20, 24] | -397 [-397, -397] | 7.14e-41 | 7.14e-41 | 120/120; max Δ=3.5e-06 |
| LJ75 | sd | rgmin | 120/120 | 66.3 [63.4, 75.7] | 56.9 [54.4, 64.9] | 187 [177, 213] | -397 [-397, -397] | — | — | — |
| LJ75 | sd | ase | gap | — | — | — | — | — | — | No steepest-descent comparator was requested for ASE. |
| LJ75 | sd | eon | 0/120 | 118 [117, 118] | 70.2 [69.7, 70.5] | — | — | 7.14e-41 | 1.58e-38 | n/a |
| Si64 | fire | rgmin | 12/12 | 18.2 [18.1, 18.5] | 16.6 [16.5, 16.9] | 39 [39, 39] | -277 [-277, -277] | — | — | — |
| Si64 | fire | ase | 12/12 | 18 [17.9, 18.2] | 16.1 [16.1, 16.3] | 49 [49, 49] | -277 [-277, -277] | 0.175 | 0.00244 | 12/12; max Δ=7.84e-05 |
| Si64 | fire | eon | 12/12 | 4.44 [4.16, 4.52] | 2.47 [2.32, 2.55] | 45 [45, 45] | -277 [-277, -277] | 3.66e-05 | 3.66e-05 | 12/12; max Δ=1.5e-06 |
| Si64 | lbfgs | rgmin | 12/12 | 5.35 [5.27, 5.66] | 4.82 [4.73, 5.1] | 11 [11, 11] | -277 [-277, -277] | — | — | — |
| Si64 | lbfgs | ase | 12/12 | 8.03 [7.98, 8.1] | 7.22 [7.17, 7.28] | 18 [18, 18] | -277 [-277, -277] | 3.66e-05 | 3.66e-05 | 12/12; max Δ=1.32e-05 |
| Si64 | lbfgs | eon | 12/12 | 1.4 [1.36, 1.46] | 0.818 [0.798, 0.856] | 11 [11, 11] | -277 [-277, -277] | 3.66e-05 | 3.66e-05 | 12/12; max Δ=0 |
| Si64 | sd | rgmin | 12/12 | 8.92 [8.83, 9.09] | 7.97 [7.91, 8.16] | 21 [21, 21] | -277 [-277, -277] | — | — | — |
| Si64 | sd | ase | gap | — | — | — | — | — | — | No steepest-descent comparator was requested for ASE. |
| Si64 | sd | eon | 0/12 | 88 [87.3, 88.5] | 48.6 [47.9, 48.9] | — | — | 3.66e-05 | 3.66e-05 | n/a |
| Si64-cell | cell-lbfgs | rgmin | 12/12 | 98.4 [98, 99.3] | 95.1 [94.6, 95.9] | 11 [11, 11] | -277 [-277, -277] | — | — | — |
| Si64-cell | cell-lbfgs | ase | 12/12 | 54.1 [53.9, 54.6] | 51.6 [51.3, 52] | 9 [9, 9] | -277 [-277, -277] | 3.66e-05 | 3.66e-05 | 12/12; max Δ=5.37e-05 |
| Si64-cell | cell-lbfgs | eon | gap | — | — | — | — | — | — | The eOn client optimizer does not expose variable-cell relaxation. |

The result JSON contains the odd/even A/A wall-time noise band and its Mann-Whitney p-value for every system/method/code cell. The widest relative band is 3.75% for LJ38 lbfgs ase.

## Failures and gaps

Nonconverged or failed runs: 384.
- Cu255-vacancy sd eon: 12 nonconverged; force calls/attempt 1001 [1001, 1001].
- LJ38 sd eon: 120 nonconverged; force calls/attempt 1001 [1001, 1001].
- LJ75 lbfgs ase: 120 nonconverged; force calls/attempt 1001 [1001, 1001].
- LJ75 sd eon: 120 nonconverged; force calls/attempt 1001 [1001, 1001].
- Si64 sd eon: 12 nonconverged; force calls/attempt 1001 [1001, 1001].
The eOn client has no variable-cell optimizer. A generic rgpot EMT Python binding was absent and is delivered as a separate source revision; the selected Cu systems use the predeclared Morse potential.
eOn SD uses opt_convergence_metric=max_atom with opt_converged_force=0.01 eV/A, opt_step_size=None, opt_max_move=0.2, and opt_max_iterations=1000. 264/264 SD runs hit the iteration budget.

## Variable-cell nonconvergence

- Cu108-cell: rgmin converged 12/12; ASE converged 12/12.
- Si64-cell: rgmin converged 12/12; ASE converged 12/12.
- Minimal reproduction: `bench/optimizers/repro_cell.py` runs the same capped L-BFGS and filter without the benchmark scheduler.

Within-run caveat: endpoint and iteration-count comparisons are deterministic under this protocol; timing statistics quantify noise only within this run's machine and software stack.
