# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking

- `Method` is `#[non_exhaustive]`: a match on it outside rgmin needs a
  wildcard arm. It ships with the next minor release; a method added
  after it is an additive change.

### Added

- `Method::Ode12r` (C `RGMIN_ODE12R`, ABI minor 34): forward Euler on
  `P^{-1} F` with the embedded second-order error estimate, the
  residual-controlled acceptance and the adaptive step of Makri, Ortner
  and Kermode (J. Chem. Phys. 150, 094109, 2019), Sec. II C as written:
  the residual is the infinity norm of the force, the local error is
  half the step times the infinity norm of the preconditioned force
  difference (the authors' ASE `ode12r`), the line-search candidate is
  the length at which the force interpolant is shortest in the `P`-norm,
  and the step grows at most fourfold (`rtol = 0.1`, `c1 = 1e-2`,
  `c2 = 2`). `src/ode12r.rs` holds
  the rule (`Ode12rState::judge`), the session owns the clamps: a capped
  move is judged at the length it took, a rejected trial keeps the
  iterate and the next call retries from the cache with a shorter step,
  one oracle call per trial either way. An installed geometry
  preconditioner is the metric, so `set_exp` gives the ODE12r-Exp
  variant. `tests/ode12r_session.rs` counts the calls, checks the
  rejection, the cap and the chain with the Exp metric.
- Optional `readcon` feature, not in `default`. `write_checkpoint` and
  `read_checkpoint` in `src/con_checkpoint.rs` depend on readcon-core
  `=0.17.1` from git `694e808` (main) until that version is on
  crates.io. The writer is `ConFrameWriter::new`: RoundTrip, canonical
  off, precision omitted. A finite `Report.value` calls `set_energy`.
  A non-finite value omits the energy key. `Report` gains no path
  field. `minimize` and `Solver::step` do not open a path. The
  `default`, `capi`, `highs`, and `all-features` CI rows do not link
  the crate. A separate `cargo test --features readcon` row compares
  a fully specified fixture with `cmp`, checks the restored f64 bits,
  and checks that a minimization started from `read_checkpoint`
  matches one started from the same in-memory vector.
- Quick-min options: `QuickMinOptions` and `Solver::set_quickmin_options`,
  C `rgmin_solver_set_quickmin_variant` with `rgmin_quickmin_variant_t`
  (`RGMIN_QUICKMIN_RGMIN`, `RGMIN_QUICKMIN_EON`,
  `RGMIN_QUICKMIN_PER_ATOM`), ABI minor 33. `QuickMinOptions::eon` is
  the eOn `Quickmin::step`: fixed time step and a capped step that
  leaves the velocity alone. `QuickMinOptions::per_atom` projects each
  atom's velocity on its own force. On rgpot surfaces (5 starts each,
  per-atom cap 0.2) it takes 645 force calls on LJ38 from random
  packings against 1344 for the default (which converges 4 of 5) and
  2001 for the eOn options (none converge), 32.0 against 19.6 on the
  Pt7 island, and 28.2 against 27.6 on the EAM Al slab.
  `curvature_cap` bounds the time step by `c / sqrt(|y| / |s|)` from the
  last force difference. `validation/quickmin.py` (sympy) derives the
  stability limit `dt^2 lam < 4`, the descent-from-rest condition
  `dt^2 lam < 2` and an energy rise of the eOn step below both;
  `validation/quickmin.sollya` bounds the binary64 lengthening of the
  projected velocity; `formal/RgminContracts/QuickMin.lean` proves the
  projection keeps `v` a nonnegative multiple of `F`, never lengthens
  it, and that the capped move is a positive multiple of `F`.

## [0.4.0] - 2026-10-11

### Added

- Python package `rgmin`: a free-threaded nanobind module `rgmin._core`
  (`Method`, `SolverConfig`, `Solver`, `Report`, `minimize`) and
  `rgmin.ase.Optimizer` with one rgmin iteration per `step`. ASE is an
  extra. The wheel is built with scikit-build-core, Corrosion, and
  `nanobind_add_module(_core FREE_THREADED ...)`.
- Vector kernels (`dot`, `sum`, `nrm2`, `axpy`, `scale`, `mul_assign`,
  `div_assign_floor`) dispatch at runtime. `dot` and `axpy` accumulate
  with `f64::mul_add`. `dot` keeps eight partials, adds those pairs
  from `p0+p4` upward, then fuses the tail. The portable clone is that
  same `mul_add` loop and is the bit reference for every ISA. The
  `simd` feature, now default, compiles the loop with multiversion 0.9
  for `x86_64+avx512f+fma` (rustc >= 1.89), `x86_64+avx2+fma`,
  `aarch64+sve`, and `aarch64+neon`. `RGMIN_SIMD` selects a clone.
  A CPU that cannot run the selected clone stays on the portable path.
  The `par` feature folds each contiguous chunk of 4096 with that tree,
  then adds the chunk totals from the first index upward.
  The wall ratios below were measured on the unfused kernels.
  On this host (KVM, Intel Xeon family 6 model 207, AVX2 and AVX-512F)
  the auto path versus `main`, five interleaved process runs, had
  axpy geometric-mean wall ratios 0.940, 0.870, and 1.000 at
  n = 300, 30000, and 3000000. The n = 30000 Holm upper bound was
  0.886. Dot ratios were 1.034, 0.993, and 1.025, and none of those
  six auto comparisons was slower beyond its interval. The atomistic
  wall interval on lj38, pt7, and al contained 1 (geometric mean
  1.016). lj38 alone was above 1. Force counts matched. Páll and Hess,
  Comput. Phys. Commun. 184, 2641 (2013),
  https://doi.org/10.1016/j.cpc.2013.06.003. Dröge et al., Softw.
  Pract. Exp. 53, 176 (2023), https://doi.org/10.1002/spe.3075.
  Holm, Scand. J. Stat. 6, 65 (1979), https://doi.org/10.2307/4615733.
- Geometry preconditioners install through `Solver::set_precon` and
  C `rgmin_solver_set_precon` (ABI minor 32): Exp, Lindh, the bonded
  metric, or `select_precon`.
- `rggate` (`bench/gate/`, pixi task `gate`) reads contract version 1
  of the atomistic CSV. The reference for a start is explicit, otherwise
  the baseline on that same start. `--reference lowest` opts into the
  lowest converged energy. Per-method curves count a converged run as a
  success. Inside a pair, Gehan scoring drops early censoring and basin
  splits from the call comparison. The BCa interval resamples systems
  and then starts. The walk is stratified by system. `dim` is the
  free coordinate count, and the energy window uses it. A split with
  unknown atom count is an upper bound. The cluster interval is printed
  from `--min-clusters` systems (default 10). Each pair is walked at
  `alpha` divided by the number of pairs. `decide_min` is the first
  time an all-wins or all-losses walk can cross. `--boundary agrapa`
  is one signed approximate-GRAPA stake, truncated at 3/4. Exit 0
  passes, 1 fails, 2 is undecided, and 3 is a contract or argument
  error. Goswami, AIP Adv. (2025),
  https://doi.org/10.1063/5.0283639. Dolan and Moré, Math. Program. 91,
  201 (2002), https://doi.org/10.1007/s101070100263. Moré and Wild,
  SIAM J. Optim. 20, 172 (2009), https://doi.org/10.1137/080724083.
  Gould and Scott, ACM TOMS 43 (2016),
  https://doi.org/10.1145/2950048. Kaplan and Meier, JASA 53, 457
  (1958), https://doi.org/10.1080/01621459.1958.10501452. Efron, JASA
  82, 171 (1987), https://doi.org/10.1080/01621459.1987.10478410.
  Howard, Ramdas, McAuliffe, and Sekhon, Ann. Stat. 49, 1055 (2021),
  https://doi.org/10.1214/20-AOS1991. Wald, Ann. Math. Stat. 16, 117
  (1945), https://doi.org/10.1214/aoms/1177731118. Waudby-Smith and
  Ramdas, J. R. Stat. Soc. B (2024),
  https://doi.org/10.1093/jrsssb/qkad009.
- ECnoise (`ecnoise`, `sample_noise`, `sample_gradient_noise`,
  `sample_gradient_noise_k`). Moré and Wild, SIAM J. Sci. Comput. 33,
  1292 (2011), https://doi.org/10.1137/100786125. The difference table
  uses `m + 1` samples at spacing `h` (default `m = 8`). `h` is the
  distance between samples, as in `ECnoise.m`, not the half-width of
  the stencil. The noise is the lowest order whose three estimates
  agree within a factor of 4 and whose column changes sign. A spacing
  that is too small (`inform` 2) is retried at `100 h`, and one that
  is too large (`inform` 3) at `h / 100`. `NoiseEstimate::declared`
  records a backend value and skips the samples. Gradient noise is
  the same table on `g(x + i h u) · w` with one counter-based unit
  vector `w`. For any covariance `Σ`, that projection has
  `E[eps_g²] = tr(Σ) / n` because `w` is uniform on the sphere, so
  `E[||e||²] = n * eps_g²`. `sample_gradient_noise_k` averages
  `eps_g²` over `k` directions. The normals use pure-Rust `libm`.
  `|f''|` and `|T|` are Moré and Wild, ACM Trans. Math. Softw. 38, 19
  (2012), Algorithm 5.1, https://doi.org/10.1145/2168773.2168777,
  not divided differences of the noise table. The forward step
  `8^{1/4} sqrt(eps_f / |f''|)` and the gradient probe
  `2 sqrt(eps_g / |T|)` use those estimates. A missing estimate uses
  the same clamp as a zero scale. The clamp is
  `[max(1e-16, 4 * ulp(max |x_i|)), 1]`. `NoiseFloor` stops at
  `c * sqrt(n_free) * eps_g` (`c = 3` by default) with
  `Status::NoiseFloor` after the `gtol` test. `NoiseFloor` also
  carries `eps_f`. A line search that finds no trial whose decrease
  clears `c * eps_f` is the same stop and keeps the pre-step point.
  `sample_value_gradient_noise` reads both from one stencil.
  `Status::StepTolerance`
  is SCG's step and objective stop while `||g||` is still above
  `gtol`. `Status` is `non_exhaustive`. The `par` feature evaluates
  the stencil by index.
- `Report` gains `status` and `Control` gains `noise_floor`
  (`NoiseFloor { eps_g, c, eps_f }`) in place of `noise_eps` and `noise_c`.
  The C ABI does not gain `eps_f`.
  Struct literals break. This ships with 0.4.0.
- C ABI (`abi_minor` 31, `layout_revision` 3): `rgmin_control_t`
  gains `noise_eps` and `noise_c` (`noise_eps <= 0` leaves the floor
  off, `noise_c <= 0` selects 3). `rgmin_report_t` gains `stop`
  (`RGMIN_STOP_TOLERANCE`, `RGMIN_STOP_NOISE_FLOOR`,
  `RGMIN_STOP_STEP_TOLERANCE`, `RGMIN_STOP_BUDGET`).
- Pulay residual subspace (`Method::Diis`, C `RGMIN_DIIS`, ABI minor 30).
  Coefficients on the stored gradients sum to one and minimize the
  combined residual. `memory` 0 keeps 6 pairs. One pair takes a
  steepest kick; two or more pairs move to the combination. The step
  does not line search on the oracle value. A kinetic
  preconditioner is not part of the step. Pulay, Chem. Phys. Lett.
  73, 393 (1980), https://doi.org/10.1016/0009-2614(80)80396-4.
  Wood and Zunger, J. Phys. A, 1343 (1985), has no DOI on this line.
- Quick-min (`Method::QuickMin`, C `RGMIN_QUICKMIN`, ABI minor 29).
  The reference step projects the velocity onto the force when that
  projection is non-negative and sets it to zero otherwise, then adds
  the time step times the force and moves by the time step times the
  updated velocity. The session also caps the displacement and rescales
  the velocity to the step that was taken, grows and shrinks the time
  step with FIRE's factors (a zero power does not shrink it), and can
  split atoms and cell into separate velocities
  (`Solver::set_quickmin_cell`, `rgmin_solver_set_quickmin_cell`).
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

- Nonlinear CG, with or without a metric, opens the line search with
  Nocedal-Wright equation 3.60 instead of resetting the trial to
  `control.istep` after every step. The descent fallback runs in both
  cases. `Conjugacy::PolakRibierePlus` is a new variant of the
  exhaustive `Conjugacy` enum: `max(β_PR, 0)` in the Euclidean product
  and in the metric product. A replaced metric restarts that recurrence
  at steepest descent.
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
