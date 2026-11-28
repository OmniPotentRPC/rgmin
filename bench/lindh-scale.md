# Scaled Lindh preconditioner

Five release repeats. The eOn cap is L-BFGS memory 20 with a
per-atom cap of 0.2. `lindh` uses that cap with the model Hessian
as `H0`. `lindh-atom` keeps the model direction when it is longer
than the cap and scales it so the longest atom sits on the cap.

Effective time is `calls × 1 s + optimizer overhead`. Overhead is
wall time minus time inside the potential. Raw rows are
`bench/raw/lindh-scale.csv`. The cold apply is
`bench/raw/lindh-overhead.csv`. The LJ150 step log is
`bench/raw/lindh-basin.csv`.

| problem | atoms | arm | calls | energy | wall ms | overhead ms | effective s |
|---|---:|---|---:|---:|---:|---:|---:|
| LJ75 | 75 | eon | 160 | -3.715040e2 | 8.42 | 0.67 | 160.001 |
| LJ75 | 75 | lindh | 130 | -3.706644e2 | 46.85 | 40.64 | 130.041 |
| LJ75 | 75 | lindh-atom | 83 | -3.721027e2 | 30.01 | 26.06 | 83.026 |
| LJ98 | 98 | eon | 108 | -5.156961e2 | 9.29 | 0.50 | 108.000 |
| LJ98 | 98 | lindh | 83 | -5.164192e2 | 43.93 | 37.18 | 83.037 |
| LJ98 | 98 | lindh-atom | 72 | -5.160755e2 | 38.33 | 32.48 | 72.032 |
| Morse Pt slab | 576 | eon | 22 | -2.932637e3 | 31.91 | 0.31 | 22.000 |
| Morse Pt slab | 576 | lindh | 18 | -2.932637e3 | 36.56 | 10.39 | 18.010 |
| Morse Pt slab | 576 | lindh-atom | 11 | -2.932636e3 | 22.02 | 6.02 | 11.006 |
| Gupta Pt slab | 1024 | eon | 11 | -5.862330e3 | 49.62 | 0.34 | 11.000 |
| Gupta Pt slab | 1024 | lindh | 9 | -5.862331e3 | 48.62 | 8.28 | 9.008 |
| Gupta Pt slab | 1024 | lindh-atom | 9 | -5.862331e3 | 48.71 | 8.29 | 9.008 |
| LJ150 | 150 | eon | 240 | -8.282532e2 | 47.62 | 1.84 | 240.002 |
| LJ150 | 150 | lindh | 242 | -8.309414e2 | 223.57 | 177.34 | 242.177 |
| LJ150 | 150 | lindh-atom | 122 | -8.197776e2 | 115.70 | 92.12 | 122.092 |
| Gupta Cu slab | 4704 | eon | 9 | -1.617472e4 | 703.15 | 1.20 | 9.001 |
| Gupta Cu slab | 4704 | lindh | 6 | -1.617472e4 | 489.90 | 20.89 | 6.021 |
| Gupta Cu slab | 4704 | lindh-atom | 6 | -1.617472e4 | 487.73 | 20.27 | 6.020 |
| Gupta Pt nano | 675 | eon | 17 | -3.797650e3 | 19.40 | 0.41 | 17.000 |
| Gupta Pt nano | 675 | lindh | 12 | -3.797649e3 | 21.21 | 7.90 | 12.008 |
| Gupta Pt nano | 675 | lindh-atom | 11 | -3.797649e3 | 19.40 | 7.16 | 11.007 |
| Gupta Cu nano | 2347 | eon | 15 | -7.871587e3 | 130.01 | 1.30 | 15.001 |
| Gupta Cu nano | 2347 | lindh | 9 | -7.871586e3 | 98.30 | 20.15 | 9.020 |
| Gupta Cu nano | 2347 | lindh-atom | 9 | -7.871586e3 | 97.22 | 20.26 | 9.020 |
| water 16 | 48 | eon | 931 | -1.560236e2 | 10.32 | 2.60 | 931.003 |
| water 16 | 48 | lindh | 212 | -1.596079e2 | 13.18 | 11.41 | 212.011 |
| water 16 | 48 | lindh-atom | 217 | -1.577544e2 | 13.41 | 11.59 | 217.012 |

Medians of five repeats. Every row converged.

## Cost of one apply

A cold build plus one solve, beside one force evaluation.

| system | atoms | force ms | apply ms | apply / force |
|---|---:|---:|---:|---:|
| Gupta Cu slab | 4704 | 77.720 | 4.768 | 6.1% |
| LJ cube | 103823 | 93407.672 | 253.818 | 0.27% |

The force is the O(N²) test potential. At 4704 the in-run
overhead on the copper slab is 20.89 ms over 6 calls, 3.5 ms
each, which is 4.5% of that force evaluation. At 10⁵ atoms the
apply is a fraction of a percent of one force call.

The solve is Jacobi-preconditioned CG, stopped at a relative
residual of 1e-2 or 32 iterations. An incomplete Cholesky factor
of the same sparsity took longer than the matvecs it removed, so
the apply stays matrix-free. Pairs come from a linked cell list
with the Lindh `ρ` cutoff.

## LJ150 basin

`lindh-atom` stops at −819.78. The eOn cap stops at −828.25.
Both meet the force tolerance. The step log has 120 accepted
`lindh-atom` steps and 236 eOn steps. Every step has a negative
`g·s` and a longest atom move of at most 0.2. The most negative
slope on the Lindh-atom path is −188.9, and the longest atom
move is the cap. This is not an uphill step and not a missed
cap. The preconditioned direction, shortened onto the per-atom
cap, leaves the basin the force-capped eOn step stays in.

Plain `lindh` on LJ150 takes 242 calls, two more than the cap,
and finishes at −830.94, a lower energy than −828.25. Water
drops from 931 calls to 212, at −159.61 instead of −156.02.
Those are different endpoints, not the same basin read to a
tighter tolerance.

The metals match the eOn energy to the last printed digit, or
within 0.001 on the platinum nanoparticle, with fewer calls.

## Default

Lindh does not win the call count on every held-out problem, and
where it uses fewer calls on LJ150 it is in another basin. It
stays opt-in. The eOn cap remains the default.
