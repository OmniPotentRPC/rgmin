# What changed, and what did not

Five serial repeats. Medians are the optimizer and the potential
together. The before column is the session loop at
`2e152a9447343190a763d84403d7e82913b49e6e`. The after column is
`Solver::run_fill` for the eOn cap, FIRE, and the new conjugate
gradient. eOn is `bench/eon_ref.cpp` from TheochemUI/eOn `579738a`.
Rows are `bench/raw/`.

| problem | arm | before ms | after ms | eOn ms | evals |
|---|---|---:|---:|---:|---:|
| Rosenbrock2 | L-BFGS m20 | 0.0584 | 0.0183 | 0.0211 | 66 |
| Rosenbrock10 | L-BFGS m20 | 0.1557 | 0.0580 | 0.0736 | 141 vs 149 |
| LJ7 | L-BFGS m20 | 0.0536 | 0.0284 | 0.0366 | 33 |
| Rosenbrock2 | FIRE | 0.9760 | 0.1548 | 0.2045 | 2714 |
| Rosenbrock10 | FIRE | 2.0701 | 0.4475 | 0.4765 | 4183 |
| LJ7 | FIRE | 0.0602 | 0.0434 | 0.0396 | 82 |
| Rosenbrock2 | CG |  | 0.0042 | 0.0084 | 87 |
| Rosenbrock10 | CG |  | 0.0693 | 0.1186 | 883 |
| LJ7 | CG |  | 0.0180 | 0.0188 | 37 |
| LJ13 | CG |  | 0.2426 | 0.2694 | 175 |
| LJ38 | CG |  | 4.6168 | 4.7702 | 397 |
| MorsePt256 | CG |  | 8.4192 | 11.4810 | 19 |

The L-BFGS and FIRE evaluation counts on these rows match the
starting commit, apart from LJ38 memory 20, which is 186 with
`powf` and was 191 with `powi`. CG matches eOn on every problem.
The diagnosis is below.

## Per-step cost

The eOn step cloned every stored pair at the start of the iteration
so a non-finite trial could be undone. With memory 20 that is forty
vectors before the two-loop allocates its own residual and
coefficient buffer. The undo log moves a pair aside and puts it
back. A successful step recycles the buffers. The two-loop writes
into a workspace and calls the same dot and axpy as `direction_h0`.

`Solver::run_fill` keeps the gradient in one buffer. Euclidean FIRE
1.0 and the eOn cap add the displacement in place. `Solver::step`
still returns a report that owns the coordinates and the gradient.

A counting allocator on `Solver::run` (report vectors suppressed)
measured about 2.3 heap allocations per L-BFGS step and 1.0 per FIRE
step on Rosenbrock 2, almost all of them the gradient returned by
the objective. `run_fill` removes that return.

## Scaling

For `H0 = gamma I` the secant `H0 y = s` is one equation in `n`
unknowns. The least-squares `gamma` is `(s · y) / (y · y)`:

```
d/d(gamma) ||gamma y - s||^2 = 2 (gamma y - s) · y = 0
gamma = (s · y) / (y · y)
```

SymPy reduces that quotient to `s·y/y·y`. A diagonal quadratic
`H = diag(2, 8)` gives a finite-difference curvature
`((g(x+δu) - g) · u) / δ` equal to the Rayleigh quotient `uᵀ H u`
(`5.840000000000134` against `5.840000000000001`). The probe
`H0 = 1/C` is that reciprocal. This is the scale already used
(Liu and Nocedal, Mathematical Programming 45, 1989,
<https://doi.org/10.1007/BF01589116>; Nocedal and Wright,
<https://doi.org/10.1007/978-0-387-40065-5>).

PySR 2.7.2, six iterations, features `(s·y, y·y, ||g||)` and target
`(s·y)/(y·y)`, returned `sy*sy/(sy*yy)`. That is the same scale.
Held-out mean absolute error was `3.7e-17` against an exact
Liu-Nocedal error of 0. It was not adopted as a new default.

## Not adopted

Goswami's dimer benchmarks compare CG and L-BFGS as rotation
optimizers and keep L-BFGS for the translational step
(<https://arxiv.org/abs/2505.13621>,
<https://doi.org/10.1063/5.0283639>;
<https://arxiv.org/html/2505.12519>). They do not change the
line-search-free cap used here.

A left-to-right slice product in the two-loop changed Rosenbrock 10
from 141 evaluations to 149 and moved the Wolfe LJ38 run. It was
dropped. The kept two-loop uses `vecops`.

CG_DESCENT and limited-memory CG (Hager and Zhang, SIAM J. Optim. 23,
2013, <https://doi.org/10.1137/120898097>; the 2005 method is the
reference listed in
<https://github.com/iskolbin/cg_descent/blob/master/cg_descent.h>)
need a Wolfe line search. Strong Wolfe stays opt-in, so neither
replaced the capped step.

FIRE 2.0 and ABC-FIRE (Guénolé et al., Computational Materials
Science 175, 2020, <https://doi.org/10.1016/j.commatsci.2020.109584>;
Echeverri Restrepo and Andric, Computational Materials Science 218,
2023, <https://doi.org/10.1016/j.commatsci.2022.111978>) are already
in the session. They are not the eOn FIRE default, which is the
Bitzek 2006 integrator
(<https://doi.org/10.1103/PhysRevLett.97.170201>) with time step 1.

The line-search-free cap remains the default.

## Conjugate gradients

The LJ38 gap was the argument order of `copysign`. C++
`std::copysign(max_move, step)` takes the magnitude from the cap and
the sign from the model step. `f64::copysign` is the other way round,
so `step.copysign(max_move)` left a positive model step unchanged.
The per-atom cap then scaled that long step up to 0.2 on the largest
atom, and the Euclidean length was longer than 0.2. Rosenbrock 2 hid
it: the vector is not a multiple of three, so the atom length is the
whole step and the two caps agree. LJ38 did not. It stopped in the
deeper basin (−161.03) after 543 evaluations. eOn stops in the
−159.77 basin after 397.

With the cap applied first, the evaluation counts match eOn on every
problem in `bench/raw/eon.csv`.

| problem | before | after | eOn |
|---|---:|---:|---:|
| Rosenbrock 2 | 87 | 87 | 87 |
| Rosenbrock 10 | 297 | 883 | 883 |
| LJ7 | 37 | 37 | 37 |
| LJ13 | 179 | 175 | 175 |
| LJ38 | 543 | 397 | 397 |
| Morse Pt256 | 17 | 19 | 19 |

The 297 and 17 were the long model step, not a better recurrence.
LJ38 ends at
−1.597738e2 with a max-atom gradient of 6.802130e-3, the same digits
as the eOn row. The Lennard-Jones force now uses `powf`, which is
what `std::pow` in `eon_ref.cpp` is. The memory-20 L-BFGS count on
LJ38 moved from 191 to 186 and stays in that same basin.

## Larger problems

`examples/bench_algo.rs` counts force calls to a max-atom tolerance.
The starts are rattled fcc slabs and balls, a Lennard-Jones grid, and
sixteen flexible TIP3P molecules. Rows are `bench/raw/algo.csv`. A
budget miss is not a win. Train is LJ75, LJ98, the Morse Pt slab and
the Gupta Pt slab. Held out is LJ150, the Gupta Cu slab (4704 atoms),
both nanoparticles, and the water cluster.

| problem | atoms | eOn cap | best other converged |
|---|---:|---:|---|
| LJ75 | 75 | 160 | exp 109, FIRE+exp 153 |
| LJ98 | 98 | 108 | scaled 99, exp 105 |
| LJ150 | 150 | 240 | scaled 292 |
| Morse Pt slab | 576 | 22 | scaled 17 |
| Gupta Pt slab | 1024 | 11 | scaled 11, trust 11 |
| Gupta Cu slab | 4704 | 9 | scaled 9 |
| Gupta Pt nano | 675 | 17 | scaled 15, exp 16 |
| Gupta Cu nano | 2347 | 15 | scaled 17 |
| water 16 | 48 | 931 | scaled 1054 |

Gupta parameters are Cleri and Rosato, Phys. Rev. B 48, 22 (1993),
<https://doi.org/10.1103/PhysRevB.48.22>. The pair and density terms
use a cosine from the fourth neighbour to the fifth so the force is
continuous at the cutoff the paper uses. Morse platinum is the well
already in `eon_ref.cpp`. Water is Jorgensen et al., J. Chem. Phys.
79, 926 (1983), <https://doi.org/10.1063/1.445869>, with the flexible
bond and angle constants from the LAMMPS `Howto_tip3p` page. Charges
use the real-unit Coulomb factor. Intramolecular pairs are the bond
and the angle only.

No arm beat the eOn cap on every held-out problem. The default is
unchanged.

The exponential preconditioner is Packwood et al., J. Chem. Phys.
144, 164109 (2016), <https://doi.org/10.1063/1.4947024>, which is the
model in ASE's PreconLBFGS. `μ` is estimated once, and that gradient
is included in the count. It wins LJ75 (109 against 160) and LJ98
(105 against 108) and loses the held-out LJ150 (306 against 240). On
the water cluster it does not reach the tolerance in 2000
evaluations. It is not run above 1600 atoms: the neighbour solve is
no longer the cheap part. A dense Lindh matrix (Lindh, Bernhardsson,
Karlström and Malmqvist, Chem. Phys. Lett. 241, 423, 1995,
<https://doi.org/10.1016/0009-2614(95)00646-L>) is the pair hook on
`direction_with_precon`. It was not added. At these sizes it is a
dense 3N system, and the exponential form is the one ASE uses for
materials.

FIRE 2.0 and ABC-FIRE stay opt-in. FIRE 2.0 converges on LJ75 where
FIRE 1.0 does not, in 316 evaluations against the cap's 160. ABC-FIRE
misses the tolerance on LJ75, LJ98, LJ150, the Morse slab, the copper
nanoparticle and the water cluster. A global Euclidean FIRE cap is
slower than the per-atom cap on the problems where both finish,
except the copper nanoparticle (31 against 44), and both lose to the
eOn cap (15). The Euclidean L-BFGS cap is the same comparison for the
quasi-Newton step. It converges, and it uses more evaluations than
the per-atom cap on every problem here.

`H0 = 1/70`, the fixed ASE scale, misses the tolerance on every
Lennard-Jones cluster and on the water cluster, and is worse than the
Nocedal-Wright scale on the metals. The scaled step, which shortens
the two-loop direction and keeps the pairs, wins LJ98 (99), the Morse
slab (17) and the platinum nanoparticle (15). It loses LJ150 (292
against 240) and the water cluster (1054 against 931). It stays
opt-in.

Barzilai-Borwein with the per-atom cap, and the same step with the
Grippo-Lampariello-Lucidi window (SIAM J. Numer. Anal. 23, 1986,
<https://doi.org/10.1137/0723046>), use more evaluations than the eOn
cap wherever they converge. On the water cluster both are non-finite
by evaluation 23. The window is not the default acceptance test.

The trust radius scales a long two-loop step and keeps the pairs,
then accepts or rejects with the Newton-model ratio `−½ g·s`
(Nocedal and Wright, algorithm 4.1). A ratio below 0.1 shrinks the
radius. This is the radius update, not the compact subproblem solve
of Burdakov, Gong, Yuan and Zikrin, Math. Program. Comput. 9, 2017,
<https://doi.org/10.1007/s12532-016-0109-7>, or of Brust, Erway and
Marcia, Comput. Optim. Appl. 66, 2017,
<https://doi.org/10.1007/s10589-016-9868-3>. It ties the cap on the
platinum slab (11) and loses everywhere else it converges. On LJ150
it misses the tolerance. It is opt-in, off by default.

## Step log

`Solver::start_eon_log` records, for each accepted eOn step, the
curvature `(g+ − g)·s / s·s`, `|F|` before and after, the cosine of
the angle between the force and the step, the cap ratio, the mean and
largest per-atom displacement, the Euclidean length, and whether the
pairs were cleared. `bench/raw/steps.csv` is that log for LJ75 and
LJ150.

PySR 2.7.2, eight iterations, trained on LJ75 and scored on LJ150.
The reset fit was `(cap_ratio − cos)¹⁶`, held-out accuracy 0.975.
The rule "the taken step is parallel to the force" scores 0.992 on
the same rows. The length fit was `9.49166 * atom_mean`, held-out
mean absolute error 0.043 against 0.092 for predicting the mean
length. That factor is the spread of a per-atom step, not a new cap.
Neither expression was installed. The earlier six-iteration fit on
synthetic `(s·y, y·y, ||g||)` is unchanged and still not a default.
