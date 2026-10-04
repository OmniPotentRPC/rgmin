# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `LineSearch::search_from`: a line search from a point whose value and
  gradient are known, returning the accepted point with its value and
  gradient (`LineOutcome`) and bounded by an `alpha_max`.
- `Solver::forget_evaluation` drops the cached evaluation and keeps the
  method memory, for a host whose oracle changes between steps (a
  min-mode effective gradient after the mode moves, a band arming its
  climbing image).
- C ABI `rgmin_solver_set_linesearch` with `rgmin_linesearch_t`
  (`abi_minor` 11).
- `FireVariant` and `Solver::set_fire_variant` (C:
  `rgmin_solver_set_fire_variant`): `FireVariant::Guenole2020` runs
  FIRE 2.0 as Guénolé et al. 2020 algorithm 2 publishes it
  (`fire::fire2_displacement`, `fire::guenole2020`, `Fire2Extras`):
  half-step back and zeroed velocity on an uphill step, `dt_min`, the
  initial delay, and the mix after the kick, with the paper's table 2
  parameters. On rgpot surfaces (5 starts each, per-atom cap 0.2) it
  takes 26.0 force calls on the Pt7 island against 57.8 for
  `FireKind::V2`, 33.8 on the EAM Al slab against 67.4, and 871 on LJ38
  from random packings against 437, so the default stays. `validation/fire2.py` checks the mix bounds and
  the step's closed forms.
- `Lbfgs::precon_fallbacks` counts Hessian preconditioners that were
  not symmetric positive definite.
- `validation/`: sympy checks of the line-search interpolants and of
  the L-BFGS two-loop map; `bench/atomistic`: rgmin on rgpot potentials
  (LJ38, Pt7 on Pt(111), EAM Al slab) counting force calls.
- `formal/`: Lean 4 and Mathlib contracts for the Armijo step, the BFGS
  and L-BFGS inverse updates, the per-atom max-move clamp and FIRE
  mixing, each mapped to its Rust function and precondition in
  `formal/README.md`. `formal/check.sh` builds them and refuses a
  `sorry`, an `axiom` or a non-foundational axiom dependency.
- `include/rgmin.h` is the C header under the `rgmin_*` names, with
  `include/rgmin/optimize.hpp` as its C++ wrapper; `include/xts.h` and
  `include/xts/optimize.hpp` stay as aliases. The repository carries the
  DLPack header it includes (`include/dlpack/dlpack.h`).
- C ABI minor 27 adds, beside `set_linesearch` and `set_fire_variant`:
  - session reuse: `rgmin_solver_rebase` (keep the method memory, drop
    the point and the acceptance window), `rgmin_solver_push_pair`,
    `rgmin_solver_pair_count` and `rgmin_solver_search_direction` for a
    host that owns its outer loop;
  - `rgmin_solver_set_objective_roundoff`: the Hager-Zhang approximate
    Wolfe test with an explicit relative value window for Euclidean
    `RGMIN_ACCEPT_NONE` sessions (Rust: `LineSearch::search_from_with_options`);
  - feasible steps under HiGHS: `rgmin_solver_set_box`,
    `rgmin_solver_set_trust` (alias `rgmin_solver_set_highs_trust`),
    `rgmin_solver_add_equality`, `rgmin_solver_clear_equalities`,
    `rgmin_solver_set_highs_solver`, `rgmin_solver_set_highs_crossover`
    and `rgmin_solver_set_highs_callback`; a malformed box fails before
    the first objective call, and an accepted step that breaks an equality
    returns `RGMIN_INVALID_PARAMETER`, restores the start and clears the
    method memory;
  - manifolds 7 to 23 of `rgmin_manifold_t` (SPD, Grassmann,
    hyperbolic, Poincare, oblique, multinomial, complex circle,
    symmetric, skew-symmetric, complex Euclidean, constant, doubly
    stochastic and symmetric multinomial, complex sphere, positive,
    centred matrix, unitary) with their shape setters
    (`rgmin_solver_set_stiefel`, `_oblique`, `_factor_shape`,
    `_unitary` and the rest);
  - Moller scaled conjugate gradients, `rgmin_minimize_scg` and
    `rgmin_minimize_scg_with_options` (`RGMIN_SCG_RELATIVE_EUCLIDEAN`
    tests the accepted step against `1 + ||x||_2`);
  - `rgmin_lowest_eigenpair` and `rgmin_lowest_eigenpair_with_options`,
    the matrix-free lowest Hessian eigenpair on Lanczos or a linked
    backend (PRIMME, SLEPc, ChASE, libkrylov and the dense ones of
    `rgmin_eigen_kind_t`); an unlinked kind returns `RGMIN_UNAVAILABLE`.
- C ABI minor 29: `rgmin_solver_set_masses` returns `rgmin_status_t`.
  A count other than one mass per atom is `RGMIN_INVALID_PARAMETER`
  and leaves the stored masses. Empty or null still restores unit
  mass. The C++ wrapper returns that status.
- C ABI minor 28: `rgmin_solver_forget_evaluation` (the C side of
  `Solver::forget_evaluation`; `rgmin_solver_rebase` also resets the
  step scale and the acceptance window) and
  `rgmin_solver_set_lbfgs_neb_guards` (the C side of
  `Solver::set_lbfgs_neb_guards`), with `forget_evaluation()` and
  `set_lbfgs_neb_guards()` on the C++ `Solver` and `xts_` aliases.
- `Solver::set_lbfgs_neb_guards`: opt-in distance, angle and curvature
  resets of the L-BFGS memory and an empty-memory scale of 0.01, for a
  band whose projected force is not a gradient.
- L-BFGS model steps honour explicit coordinate bounds (`lbfgs_qp`), and
  the sphere L-BFGS steps along geodesics and keeps its pairs in the
  tangent space.

### Changed

- A session's default line search is strong Wolfe (`c1 = 1e-4`,
  `c2 = 0.9`, 20 trials) instead of Brent. Brent spent about 25 force
  calls per L-BFGS iteration and, under a per-atom cap, stalled LJ38 at
  `fmax` above 100. `LineSearch::default()` stays Brent.
- Line-searched steps reuse the value and gradient at the start and
  return the gradient at the accepted point: steepest descent, NLCG,
  BFGS, SR1, SR2, Adam and line-searched L-BFGS make no evaluation
  twice. L-BFGS under Wolfe goes from 3.2 to 1.1 force calls per
  iteration (LJ38 476 to 168, Pt7 31 to 10, Al slab 42 to 14 calls).
- The displacement cap bounds the line search (`alpha_max`) instead of
  rescaling its result, so the accepted value belongs to the accepted
  point.
- The strong-Wolfe zoom interpolates with the safeguarded More-Thuente
  cubic, then the quadratic, then the midpoint, and extrapolates within
  1.1 to 4 increments.

### Fixed

- Powell dogleg retracts the tangent step and reports the horizontal
  gradient. An ambient `x + dir` left the sphere.
- `set_project_rigid` on a periodic cell drops translation and keeps
  rotation. An isolated cluster still drops both.
- `MwRigid::project` and `ManifoldKind::mw_rigid` use the stored
  per-atom masses. Unit mass still matches `RigidQuotient`.
  `set_masses` and `set_manifold` keep the session on those masses.
- A mass table whose length is not one mass per atom is
  `Error::MassCount`. The Eckart projection leaves the vector
  unchanged instead of using unit weight. `Solver::set_masses`
  refuses the table and keeps the masses already stored.
- The Barzilai-Borwein pair is the transported displacement and
  the horizontal gradient minus the transported previous gradient.
  An ambient `x - x_old` on the sphere is not tangent.
- Stored L-BFGS pairs are transported to the current point on every
  manifold. Sphere and Stiefel under an energy accept keep their
  geodesic update. The new pair is pushed after that transport.

- The line-search zoom bisects when the far end of the bracket is not
  finite or differs from the near end by more than
  `1e3 (1 + |f(lo)| + |phi'(lo)| (hi - lo))`. Cubic and quadratic
  interpolation of such a value drove the trial to the clamp of the
  bracket and accepted a different first step than bisection.

- A first-order L-BFGS session pushes the secant of a displacement the
  caller made between steps, transported like an accepted step, on
  translation manifolds. `rebase` and `forget_evaluation` still drop it.

- A `Control::maxmove` (or per-atom cap) that is not positive and finite
  is no cap; `Some(-c)` reversed every capped step.
- FIRE uses `alpha` clamped to `[0, 1]` in the mix, so an out-of-range
  public field cannot lengthen the velocity.
- The Hessian-preconditioned two-loop solves by Cholesky and falls back
  to `gamma I` on a matrix that is not SPD; an indefinite Hessian that
  passed the pivoted solve gave an ascent direction.
- An uphill direction (indefinite SR1/SR2) is searched backwards by
  every line search; backtracking took the Armijo test with a positive
  slope.

- A session on a translation manifold (Euclidean, rigid quotients) no
  longer re-evaluates the accepted point after a no-op retract round
  trip.
- The cached-point test is relative to the coordinate size, so a host's
  ulp-level round trip of the iterate does not cost a force call.
- `extra_updates` replays the newest pair, which leaves the BFGS map
  unchanged; the two-loop map ignores it and says so.
- The 0.3.0 entry below says `Accept::None` takes the two-loop direction
  with one call; that path belongs to `Accept::Step`, and `Accept::None`
  keeps the line search.
- RFO selects the lowest augmented mode and keeps its model constraints
  after a rejected trial. The dense Sella trust step returns inside its
  radius, and quasi-Newton derivatives keep the shifted mode's sign.
- Sessions apply the requested gradient tolerance to their stopping
  test, honour an explicit step policy in the first-order methods, and
  refuse a nonfinite gradient at the last step check.
- The strong-Wolfe zoom keeps the feasible endpoint when it runs out of
  trials, and resolves steps at the energy's precision limit.
- L-BFGS records the last accepted pair of a session step alone. It
  keeps the accepted point when a feasible solve fails, and retries a
  stalled curvature with a bounded gradient step.
- NLCG forms gradient differences before its scalar products. SCG keeps
  convergence across limits and objective offsets, and restarts a
  cancelled direction while the gradient is not stationary.
- Manifold steps apply an accepted retraction once, keep the SPD metric
  gradient, and complete dependent unitary columns. The sphere passes on
  a failed constrained direction.
- Krylov basis pruning follows the residual tolerance. The libkrylov
  backend serializes ownership of its solver state, and linked
  matrix-free eigen backends stay available.

## [0.3.0] - 2026-09-30

### Added

- `Control::ftol_rel`: a relative slack on every energy-decrease test,
  `ft - ref <= ftol_rel * (|ref| + 1)`. It covers the line-search
  accept in the first-order methods and the `Accept::Energy` /
  `Accept::Nonmonotone` backtracking. `None` keeps the strict test.

### Fixed

- An L-BFGS session honours `set_accept`. `Accept::None` and
  `Accept::Nonmonotone` take the two-loop direction through the same
  accept path as BB and the Hessian-preconditioned step: one oracle call
  and the clipped step under `Accept::None`, no line search. A driver
  feeding a non-conservative projected force (NEB) gets a step every
  iteration. `Accept::Energy` keeps the line search.
- FIRE rescales its velocity by the ratio the maxmove clamp (or the
  bounds clip) applied to the trial move, so the power it adapts on
  measures the motion taken. Under a clamp that bound every step the
  velocity integral grew unbounded and `dt` grew with it.
- `set_atom_maxmove` binds the line-searched session arms (L-BFGS,
  steepest descent, NLCG, BFGS, SR1, SR2, Adam); they read only the
  Euclidean `maxmove`.

## [0.2.1] - 2026-09-19

### Fixed

- Quasi-Newton methods (BFGS, L-BFGS, SR1, SR2, and their session
  variants) open every line search at `Control::istep` instead of half
  the last accepted step. The halving is the steepest-descent heuristic;
  under a backtracking search, which only shrinks, it made the step
  collapse geometrically and L-BFGS stalled far from `gtol` on a smooth
  convex energy. Steepest descent, NLCG and Adam keep the halving.
  `Control::istep` is documented as the opening step.

### Changed

- The crate is `rgmin`, at `OmniPotentRPC/rgmin`; it was
  `xtsci-optimize` under `HaoZeke`. The C ABI renames with it
  (`rgmin_*` symbols, `rgmin.h`) in one pre-publication sweep; the
  `xts::optimize` C++ namespace remains as source compatibility.
- The strong-Wolfe zoom proposes cubic-Hermite trials with both
  bracket slopes (Nocedal-Wright eq. 3.59) behind interior guards;
  measured on the LJ75 hopping battery this cut force calls per hop
  from 46 to 43.
- Every solver's length-n algebra flows through the `vecops` seam.
  The seam carries a DLPack-device-tagged `Vector` handle; a device
  tag without a kernel backend is refused at construction, never
  staged through the host.

### Added

- Matrix-free Newton: the `HessianVector` trait, a finite-difference
  action wrapper, Steihaug-Toint CG inside a Nocedal-Wright trust
  region (`minimize_newton_cg`), and preconditioned CG with the
  Conn-Gould-Toint metric recurrences (`steihaug_pcg`).
- `NystromPrecond`: the Frangella-Tropp-Udell randomized sketch as a
  CG preconditioner; randomness lives only in the sketch, and the
  test suite pins the preconditioned step to the plain step.
- `minimize_recognized`: per-iterate basin recognition with the
  caller's substitute carried out under a flag.
- Lean proofs (Mathlib) for the zoom guard and its geometric
  envelope, the Steihaug boundary root, preconditioner scale
  invariance, trust-radius honesty, and the sketch's positive
  semidefiniteness, indexed in `docs/orgmode/reference/proofs.org`.
- Diataxis explanation pages deriving the line search, the secant
  family, trust regions, SCG, and the randomized preconditioning.

### Fixed

- A non-finite gradient can no longer satisfy the convergence test
  under either gradient norm.
- The energy-accept fallback faces the same test as the step it
  replaces; a fallback that also fails reports the position unmoved.
- Failed C callbacks return NaN-filled gradients and Hessians rather
  than fabricated zeros or identity matrices.
- The FFI waist reuses standing DLPack shells per solve instead of
  allocating per evaluation: +26 percent evaluations per second at
  n=30, +19 percent at n=225, identical trajectories.
- HiGHS thread setup serializes through `std::sync::Once` instead of
  racing on the environment.
- The `par` feature keeps vectors under 65536 elements on the serial
  path, where the rayon reductions measured slower than serial.

## [0.2.0]

Rust rewrite of the C++ xtsci-optimize: solvers over eindir
`DifferentiableObjective`, session C ABI, manifolds, HiGHS-projected
steps. The C++ xtensor history is `0.0.1` on the previous `main`.
