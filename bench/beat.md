# Arms against the eOn cap

Five serial release repeats. The median is the optimizer and the
potential together. The baseline is L-BFGS memory 20 with the eOn
per-atom cap of 0.2, the same step as `bench/raw/algo.csv`. A row that
stops on the iteration budget is not a win. Train is LJ75, LJ98, the
Morse Pt slab and the Gupta Pt slab. Held out is LJ150, the Gupta Cu
slab (4704 atoms), both nanoparticles, and the water cluster.

Raw repeats are `bench/raw/beat.csv`. Directional curvatures are
`bench/raw/spread.csv`.

| problem | atoms | arm | evals | energy | median ms | converged |
|---|---:|---|---:|---:|---:|---|
| LJ75 | 75 | eon | 160 | -3.715040e2 | 8.33 | yes |
| LJ75 | 75 | cg-atom | 309 | -3.733238e2 | 14.84 | yes |
| LJ75 | 75 | model | 152 | -3.715040e2 | 7.91 | yes |
| LJ75 | 75 | gate | 112 | -3.675616e2 | 38.26 | yes |
| LJ75 | 75 | lindh | 94 | -3.730739e2 | 439.87 | yes |
| LJ75 | 75 | lindh-atom | 110 | -3.736166e2 | 522.17 | yes |
| LJ75 | 75 | compact | 142 | -3.733238e2 | 9.60 | yes |
| LJ75 | 75 | skip | 160 | -3.715040e2 | 8.26 | yes |
| LJ75 | 75 | tuned m8/0.4 | 146 | -3.715040e2 | 7.29 | yes |
| LJ98 | 98 | eon | 108 | -5.156961e2 | 9.38 | yes |
| LJ98 | 98 | cg-atom | 237 | -5.156961e2 | 19.42 | yes |
| LJ98 | 98 | model | 112 | -5.156961e2 | 9.61 | yes |
| LJ98 | 98 | gate | 108 | -5.158409e2 | 56.06 | yes |
| LJ98 | 98 | lindh | 80 | -5.158719e2 | 559.12 | yes |
| LJ98 | 98 | lindh-atom | 60 | -5.158718e2 | 416.67 | yes |
| LJ98 | 98 | compact | 100 | -5.154720e2 | 13.42 | yes |
| LJ98 | 98 | skip | 108 | -5.156961e2 | 9.26 | yes |
| LJ98 | 98 | tuned m8/0.4 | 89 | -5.158740e2 | 7.51 | yes |
| Morse Pt slab | 576 | eon | 22 | -2.932637e3 | 32.13 | yes |
| Morse Pt slab | 576 | cg-atom | 63 | -2.932637e3 | 91.29 | yes |
| Morse Pt slab | 576 | model | 18 | -2.932636e3 | 26.28 | yes |
| Morse Pt slab | 576 | gate | 30 | -2.932636e3 | 222.72 | yes |
| Morse Pt slab | 576 | lindh | 18 | -2.932637e3 | 1457.60 | yes |
| Morse Pt slab | 576 | lindh-atom | 11 | -2.932636e3 | 823.88 | yes |
| Morse Pt slab | 576 | compact | 16 | -2.932636e3 | 32.44 | yes |
| Morse Pt slab | 576 | skip | 22 | -2.932637e3 | 32.17 | yes |
| Morse Pt slab | 576 | tuned m8/0.4 | 21 | -2.932636e3 | 30.53 | yes |
| Gupta Pt slab | 1024 | eon | 11 | -5.862330e3 | 49.89 | yes |
| Gupta Pt slab | 1024 | cg-atom | 21 | -5.862331e3 | 94.56 | yes |
| Gupta Pt slab | 1024 | model | 11 | -5.862330e3 | 49.83 | yes |
| Gupta Pt slab | 1024 | gate | 20 | -5.862331e3 | 303.14 | yes |
| Gupta Pt slab | 1024 | compact | 11 | -5.862330e3 | 50.14 | yes |
| Gupta Pt slab | 1024 | skip | 11 | -5.862330e3 | 49.75 | yes |
| Gupta Pt slab | 1024 | tuned m8/0.4 | 11 | -5.862330e3 | 49.74 | yes |
| LJ150 | 150 | eon | 240 | -8.282532e2 | 47.66 | yes |
| LJ150 | 150 | cg-atom | 801 | -8.338469e2 | 153.63 | no |
| LJ150 | 150 | model | 402 | -8.344383e2 | 80.04 | no |
| LJ150 | 150 | gate | 309 | -8.476754e2 | 311.97 | yes |
| LJ150 | 150 | lindh | 380 | -8.354036e2 | 5224.09 | yes |
| LJ150 | 150 | lindh-atom | 174 | -8.194814e2 | 2363.58 | yes |
| LJ150 | 150 | compact | 317 | -8.426542e2 | 82.67 | yes |
| LJ150 | 150 | skip | 240 | -8.282532e2 | 48.20 | yes |
| LJ150 | 150 | tuned m8/0.4 | 352 | -8.204962e2 | 69.64 | yes |
| Gupta Cu slab | 4704 | eon | 9 | -1.617472e4 | 702.77 | yes |
| Gupta Cu slab | 4704 | cg-atom | 15 | -1.617472e4 | 1172.26 | yes |
| Gupta Cu slab | 4704 | model | 9 | -1.617472e4 | 703.13 | yes |
| Gupta Cu slab | 4704 | gate | 12 | -1.617472e4 | 1008.68 | yes |
| Gupta Cu slab | 4704 | compact | 9 | -1.617472e4 | 704.09 | yes |
| Gupta Cu slab | 4704 | skip | 9 | -1.617472e4 | 701.97 | yes |
| Gupta Cu slab | 4704 | tuned m8/0.4 | 9 | -1.617472e4 | 703.09 | yes |
| Gupta Pt nano | 675 | eon | 17 | -3.797650e3 | 19.60 | yes |
| Gupta Pt nano | 675 | cg-atom | 27 | -3.797650e3 | 30.30 | yes |
| Gupta Pt nano | 675 | model | 17 | -3.797650e3 | 19.47 | yes |
| Gupta Pt nano | 675 | gate | 19 | -3.797650e3 | 133.42 | yes |
| Gupta Pt nano | 675 | lindh | 12 | -3.797649e3 | 576.14 | yes |
| Gupta Pt nano | 675 | lindh-atom | 11 | -3.797649e3 | 519.66 | yes |
| Gupta Pt nano | 675 | compact | 17 | -3.797650e3 | 20.02 | yes |
| Gupta Pt nano | 675 | skip | 17 | -3.797650e3 | 19.45 | yes |
| Gupta Pt nano | 675 | tuned m8/0.4 | 17 | -3.797650e3 | 19.35 | yes |
| Gupta Cu nano | 2347 | eon | 15 | -7.871587e3 | 129.61 | yes |
| Gupta Cu nano | 2347 | cg-atom | 37 | -7.871591e3 | 319.42 | yes |
| Gupta Cu nano | 2347 | model | 15 | -7.871587e3 | 129.16 | yes |
| Gupta Cu nano | 2347 | gate | 18 | -7.871587e3 | 159.80 | yes |
| Gupta Cu nano | 2347 | compact | 15 | -7.871587e3 | 131.24 | yes |
| Gupta Cu nano | 2347 | skip | 15 | -7.871587e3 | 129.53 | yes |
| Gupta Cu nano | 2347 | tuned m8/0.4 | 15 | -7.871587e3 | 129.80 | yes |
| water 16 | 48 | eon | 931 | -1.560236e2 | 10.28 | yes |
| water 16 | 48 | cg-atom | 881 | -1.532385e2 | 7.58 | yes |
| water 16 | 48 | model | 1089 | -1.586816e2 | 11.97 | yes |
| water 16 | 48 | gate | 2005 | -1.559660e2 | 341.48 | no |
| water 16 | 48 | lindh | 285 | -1.596081e2 | 120.31 | yes |
| water 16 | 48 | lindh-atom | 160 | -1.559343e2 | 67.87 | yes |
| water 16 | 48 | compact | 1135 | -1.586804e2 | 65.82 | yes |
| water 16 | 48 | skip | 931 | -1.560236e2 | 10.33 | yes |
| water 16 | 48 | tuned m8/0.4 | 1157 | -1.561186e2 | 11.62 | yes |

Lindh is not run above 700 atoms. The banded factor refuses a pattern
whose elimination would exceed about 8e8 flops, which is the Gupta Pt
slab, the Cu slab, and the Cu nanoparticle.

## What each arm is

`cg-atom` skips the scalar clip on a positive model step and keeps
only the per-atom cap. That is the positive-step half of the reversed
`copysign`: Rust `step.copysign(max_move)` leaves a long positive
length, and C++ `std::copysign(max_move, step)` does not. The sign of
a negative model step is still clipped. Rosenbrock10 takes 297 force
calls with the option and 883 without it. The default stays at 883.

`model` is the same idea on L-BFGS. A long downhill two-loop step is
scaled so the longest atom sits on the cap, and the pairs are cleared.
The eOn cap replaces that step with `H0` times the force.

`gate` builds the Packwood exponential metric only when the ratio of
two Rayleigh quotients is at most 6. One quotient is along the force
and one is along the shortest bond. The three probes are included in
the count. Above 1600 atoms the metric is not installed even if the
gate is open.

`lindh` is the Lindh stretch and bend model in Cartesian coordinates,
factored by a banded Cholesky after reverse Cuthill–McKee, and used as
`H0`. `lindh-atom` keeps that direction when it is longer than the cap
instead of falling back to the force. Parameters are the Bakken and
Helgaker table of the 1995 constants. Torsions are omitted.

`compact` solves `(B + λ I) s = -g` in the Byrd–Nocedal–Schnabel
product and picks `λ` so the longest atom equals the cap. A short
two-loop step is left alone. The pairs are kept.

`skip` drops a pair with `s·y ≤ 0` and keeps the older pairs. The eOn
step stores the pair and clears the memory on the next iteration.

`tuned` is memory 8 and cap 0.4. On the training rows that pair has
the smallest total force-call count among the settings that converge
on every training problem (267, against 301 for memory 20 and cap 0.2).
The grid is in the same CSV.

## Held-out

No arm uses fewer force calls and less wall time than the eOn cap on
every held-out problem.

`cg-atom` beats the cap on water (881 calls and 7.58 ms against 931
and 10.28 ms) and misses the tolerance on LJ150. It uses more calls
on both nanoparticles and on the copper slab.

`model` misses LJ150 and uses more calls on water. Where the counts
match, the times match.

`gate` opens on every problem, including water. The two quotients
there are 1.940e3 and 1.886e3, ratio 1.028, because both probes sit on
stiff intramolecular modes. The soft intermolecular spectrum is not in
either direction, so the fallback never fires. Water stops at the
budget (2005 calls). LJ150 takes 309 calls and 312 ms against 240 and
48 ms.

`lindh-atom` uses fewer calls on LJ150 (174 against 240), the platinum
nanoparticle (11 against 17) and water (160 against 931). The wall
times are 2364 ms, 520 ms and 68 ms against 48 ms, 20 ms and 10 ms.
The LJ150 energy is -8.195e2 against the cap's -8.283e2. Plain Lindh
uses more calls than the cap on LJ150 (380) and is slower on every
problem it runs. The large held-out metals are not factored.

`compact` uses more calls on LJ150 (317) and on water (1135). The
metal counts match the cap and the times are not lower.

`skip` reproduces the cap's counts and energies. The median times
differ by about a millisecond.

`tuned` was selected on the training total only. It loses LJ150 (352
against 240) and water (1157 against 931). It is not the default.

The default remains the eOn cap. Memory 20 and cap 0.2 stay the
reference settings. `atom_only`, `model_atom`, `compact_tr`,
`skip_bad` and the Lindh factor stay off.
