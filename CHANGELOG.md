# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2026-11-13

### Added

- `lowest_eigenpairs` is block LOBPCG for the k lowest eigenpairs of a symmetric Hessian product (Knyazev 2001). The seed is an n-by-k block. Each pair carries its residual and whether that residual met the tolerance. `k = 0`, `k` above the dimension, and a non-finite value or product return an error. C: `rgmin_lowest_eigenpairs`, `rgmin_eigenpair_t`. Python: `rgmin.lowest_eigenpairs`. ABI minor 39.
- `SuppliedSearch` and `rgmin_linesearch` take a caller-supplied energy and gradient and do not call an objective. ABI minor 37.
- `LbfgsPairs` and `lbfgs_direction` return the L-BFGS direction `-H g` from stored pairs and a caller-supplied initial inverse Hessian. The call does not evaluate an objective. ABI minor 38.
- The default L-BFGS step is the eOn line-search-free cap (`LbfgsStep::Capped` at `EON_MAX_MOVE`, 0.2). The first step probes curvature. A proposal at least as long as the cap clears the memory. `LbfgsStep::Wolfe` and `rgmin_solver_set_lbfgs_wolfe` opt into the strong Wolfe search. One-shot `minimize_method` and `rgmin_minimize` keep the line search the caller passes. `LbfgsStep::Scaled` is the ASE length control. `Lbfgs::builder`, `Lbfgs::minimize_box`, and `Lbfgs::ftol` are available.
- `Control::stop_norm` selects the norm a session compares to `gtol`. `Report` carries `stop_norm`, `grad`, and `status`.
- The `python/` package is an ASE `Optimizer` drop-in (`BFGS`, `LBFGS`, `FIRE`). `LBFGS` uses ASE's length control unless `eon_cap=True`.
- `Stepper::step` is the one-iteration seam a host calls on `Solver`.
- The combined C surface is ABI minor 40. Minor 37 is linesearch, 38 is `rgmin_lbfgs_direction`, and 39 is `rgmin_lowest_eigenpairs`. The symbols new at 40 are the 0.5.2 C functions develop did not have, including `rgmin_minimize_scg_with_options` and `rgmin_solver_set_objective_roundoff`. `rgmin_control_t` gains `noise_eps` and `noise_c`. `rgmin_report_t` gains `stop`.
- `rgmin_minimize_scg_with_options` takes an explicit SCG step rule. Null options and `RGMIN_SCG_ABSOLUTE_INFINITY` keep the absolute infinity-norm rule. `RGMIN_SCG_RELATIVE_EUCLIDEAN` stops on `||alpha d||_2 < tol_sol * (1 + ||x||_2)`. `Solver::set_objective_roundoff` sets the relative value window of a Euclidean Wolfe search. A positive window can accept a rise inside that window and stop at a stationary point that the strict test refuses.

### Changed

- A negative `maxmove` on `rgmin_control_t` leaves every step cap off, including the default per-atom cap. Zero keeps the defaults. A positive value is the Euclidean cap.
- SCG reductions stay in plain left-to-right order (`dot_ordered`), as released in 0.5.2. A positive finite `maxmove` also bounds every SCG trial. Other values leave that step unbounded.

### Fixed

- A backtracking or Goldstein `LineSearch` that ran out of trials returned its best point with the step length of its last trial.
  It now reports the step that produced that point.
- An L-BFGS session step whose own pair failed the curvature test also dropped the pair before it.
  It now keeps that pair.
- `minimize_scg`, `minimize_scg_exact` and `rgmin_minimize_scg` bound every trial step by a positive finite `Control::maxmove`, as that field documents.
  Before, scaled conjugate gradient (SCG) ignored it.
  The bound applies at the first trial and at each retry after a non-finite value.
  Zero, negative, infinite and NaN values leave SCG unbounded.
  Runs whose bound never binds give the same reports as before.
  The comparison ratio of a bounded step divides by the damped model's change at that step, as in Nocedal and Wright eq. 4.4.
  A short step therefore no longer halves the damping on an agreement as low as 0.375.
  The page `explanation/step-bounds` derives the ratio and sets out what can and cannot fix the size of a bound on log-hyperparameters.
- How-to samples that did not compile now do.
  The manifold sample defines its objective.
  The HiGHS sample fills the rest of `HighsStep` from `Default`.
  The intrinsic reaction coordinate (IRC) sample imports only what it uses.
  The C++ sample calls `rgmin::minimize_fn` from `rgmin/optimize.hpp`.
  The C samples include `rgmin.h`.
- `cargo cinstall` installs the hand-written `include/rgmin.h` and `dlpack/dlpack.h`.
  The header compiles as C and as C++.
  Version macros are `RGMIN_H_MAJOR`, `RGMIN_H_MINOR`, and `RGMIN_H_PATCH`.
  `HighsInt` stays out of the public header.
  Continuous integration (CI) installs into a prefix and compiles a C program and a C++ program against that header and the `rgmin` pkg-config file.
- The docs index and `reference/architecture` define the hourglass and its waist, the C ABI that C and C++ callers go through.
  A figure that `docs/figures/architecture/draw_hourglass.py` draws from the headers and `src/ffi.rs` replaces the old text diagram.
  The page gives the ABI version from `include/rgmin.h`, 1.29 with layout revision 4, instead of 1.9 with revision 2.
- The code of conduct, the security policy and the docs index name a contact, and the index links the issue tracker.
- A non-finite value or gradient is a refused answer in every line search and acceptance test.
  It never counts as a decrease.
  Every comparison against NaN is false, and `-inf` passes Armijo.
  Before this fix, `Lbfgs::minimize` doubled its step into an unanswered region and could accept a NaN or `-inf` value as the minimum.
  Now the strong Wolfe zoom bisects back.
  A non-finite start ends `Lbfgs::minimize`, `minimize_watched` and `minimize_recognized` at `x0` with value `+inf` after one evaluation.
  Before, they probed along a NaN direction.
  Brent, backtracking, Goldstein and Wolfe `LineSearch` searches score such a trial `+inf`.
  Given a non-finite start or direction, they return the start without probing.
  These methods return `Error::Oracle` when the start is not finite: steepest descent, nonlinear CG, Broyden-Fletcher-Goldfarb-Shanno (BFGS), L-BFGS, SR1, SR2 and Adam.
  So does a `Solver` step for every first-order method, BB and the fast inertial relaxation engine (FIRE) included.
  Energy and nonmonotone acceptance require a finite trial.
  A flat FIRE trial with no answer halves toward the start up to ten times, inside the clip box.
  FIRE, when no halving answers, and the geometry L-BFGS fallback return `Error::Oracle` instead of stepping into a non-finite answer.
  They leave `x`, the FIRE integrator and the L-BFGS curvature as they were, so a retry probes the same trial.
  Through the C ABI these errors arrive as `RGMIN_INTERNAL_ERROR` with no NaN or clipped-to-the-bound coordinate sent to the callback.
- An L-BFGS `Solver` step under `Accept::None` records the caller's move as a curvature pair before its one trial.
  A non-finite trial now takes that pair back, so the memory is as it was when the step returns `Error::Oracle`.
  Before, each retry from the moved point recorded the same pair again.
- The contributing page now gives the docs export command that the workflow runs, `cd docs && emacs --batch -l export.el`.
  Of the two commands it gave before, one failed and the other wrote nothing.
- On a curved manifold kind, a `Solver` step caps the tangent step and then retracts it, so every point it evaluates lies on the set.
  Before, the acceptance trial retracted, capped the chord and clipped, and then scored that point off the set.
  On the unit sphere with a 0.1 cap, steepest descent evaluated points 3.4e-2 off it, FIRE 6.2e-2 and FIRE 2.0 6.0e-3.
  A retraction that moves farther than the cap shrinks the step and retracts again, so a step on the positive orthant keeps the cap.
  A trial that the box clip would move is refused before the oracle sees it.
  FIRE and the FIRE 2.0 rollback use the same order.
  Dogleg on a curved kind steps in the tangent space, on the Hessian in an orthonormal basis of the tangent space, and its ratio compares the energy at the retracted point with that model.
  Before, it accepted `x + p` with no retraction, 0.113 off the sphere for `f = -x_3 + |x|^2 / 2`.
  On a curved kind the session no longer retracts and evaluates again after the step, except after the L-BFGS line search.
  Steepest descent on the sphere takes one oracle call per step instead of two.
  `Error::OffManifold` reports a FIRE step with no trial inside the box and the cap, and the C ABI returns it as `RGMIN_INVALID_PARAMETER`.
  Flat kinds keep the old order and the old numbers.
- On a curved manifold kind, Newton-CG and Newton-PCG retract their trust-region step, so every point the objective sees lies on the set.
  Before, they took `x + p`, and on the unit sphere with a 0.1 cap the worst evaluated point was 1.1e-1 off it.
  CG now runs on the model `P H P + (I - P)` for the tangent projection `P` of the kind, and flat kinds keep `x + p`.
  Where `P` is orthogonal, as on every curved kind but the three multinomial kinds, dogleg solves the same model.
- On a curved manifold kind, a `fire2` session projects its secant velocity onto the tangent space at the new point.
  Before, the normal part of that velocity reached 5.0e-2 of the speed on the unit sphere with a 0.1 cap.
- The cyclic Jacobi eigensolver no longer squares a rotation parameter `theta` above `1 / eps`, with eps the machine epsilon.
  Above that bound, the solver takes the tangent `t = 1 / (2 theta)`, the limit of the full formula.
  The Linear Algebra Package (LAPACK) takes the same branch in its one-sided Jacobi routines.
  The transition-state Broyden-Fletcher-Goldfarb-Shanno (TS-BFGS) update and the rational function optimization (RFO) step call this solver.
  An off-diagonal entry below about 3.7e-155 times its diagonal gap made `theta^2` overflow.
  With overflow trapped, the host stopped with the floating-point exception signal (SIGFPE); with the trap masked, the solver skipped the rotation.
  Every result that was finite before is unchanged.

### Added

- FIRE 2.0 stops when consecutive non-positive-power steps exceed
  `vdfmax`. The published schedule uses 2000, from Table 1 of Comput.
  Mater. Sci. 175 (2020) 109584. The stopping step does not move.

### Fixed

- The Exp pair weight and the μ sine probe call `libm` `exp` and
  `sin`. The host math library does not agree between x86_64 and
  aarch64. A hash of those `f64` bits is locked in the test.

## [0.5.2] - 2026-10-25

### Fixed

- SCG reduces its dot products and norms in plain left-to-right order
  again, so its trajectory reproduces the reference C++ implementation to
  the parity tolerance; the fused SIMD kernels of 0.5.0 had moved the
  iterates by roundoff that grew along the trajectory.

## [0.5.1] - 2026-10-25

### Changed

- A negative `maxmove` on `rgmin_control_t` leaves every step cap off,
  including the default per-atom cap, so a one-shot minimizer or a session
  over hyperparameters or weights can opt out without a session call.
  Zero keeps the defaults and a positive value is the Euclidean cap, as
  before.

## [0.5.0] - 2026-10-25

### Added

- Steihaug-Toint Newton-CG as `Method::NewtonCg` (`newton-cg`) and
  `Method::NewtonPcg` (`newton-pcg`, a rank-16 Nyström preconditioner
  rebuilt at the current point). `Solver::step` and `step_hess` run it
  with a forward-difference product; `Solver::step_hvp` and
  `minimize_method_hvp` take an analytic `H v`. The session counts
  `gradient_calls` and `hvp_calls` separately, so a harness can charge
  `gradient_calls + hvp_calls` force evaluations on this path. The C
  ABI adds `RGMIN_NEWTON_CG`, `RGMIN_NEWTON_PCG`,
  `rgmin_method_from_name`, `rgmin_minimize_hvp`,
  `rgmin_solver_step_hvp`, `rgmin_solver_gradient_calls` and
  `rgmin_solver_hvp_calls`; `Method::from_name` / `as_str` parse and
  print every method token (`newton-cg` and `newton_cg` are the same).
  `abi_minor` 36.
- The capped Cartesian L-BFGS geometry step, `Solver::set_geometry_step`
  (`rgmin_solver_set_geometry_step`): eOn's L-BFGS with a
  finite-difference inverse curvature on a severely clipped start and
  `0.1 I` otherwise, `s·y / y·y` after that, a reset when curvature is
  negative, the step exceeds the per-atom cap or opposes the force, the
  probe clipped to the objective bounds, and an energy test with at
  most ten halvings before a short steepest step. An installed
  exponential preconditioner is that step's initial inverse Hessian and
  the metric of its fallbacks. The step is an explicit opt-in because
  every Cartesian session already carries the default per-atom cap; off,
  the cap only clips the two-loop trial and the NEB guards apply.
- `cargo cinstall --features capi` installs the hand-written
  `include/rgmin.h`, the C++ wrappers and `dlpack/dlpack.h`, with the
  `rgmin` pkg-config file. The header carries `RGMIN_H_MAJOR`,
  `RGMIN_H_MINOR` and `RGMIN_H_PATCH`, which `build.rs` checks against
  the crate version, and stays plain C (no `HighsInt`, no namespace).
  `tests/cinstall/smoke.sh` installs into a prefix and compiles a C and
  a C++ program against it; CI runs it.

### Changed

- Newton and dogleg leave a first-order saddle. When the force is under
  the tolerance and the analytic Hessian has an eigenvalue below the
  stationarity floor, the session steps along that mode, scaled to the
  per-atom cap (or the initial step), at most four times per solve. The
  dogleg radius restarts from the initial step.

### Fixed

- Dogleg grows its trust radius past a per-atom cap. Nocedal and
  Wright, algorithm 4.1, grows Δ only when the trust step meets the
  boundary, and the clipped length made a boundary step look interior,
  so Δ froze at the cap and Morse bulk rebuilt the Hessian about 30
  times per start. Δ_max is now the Euclidean `maxmove` (or eight
  times the current radius) and the boundary test uses the dogleg
  length before the cap; the cap still limits the trial. Newton-CG
  follows the same rule.
- The default per-atom cap is chosen per method. A Cartesian
  session installs 0.2 for steepest descent, nonlinear CG, L-BFGS,
  SR1, SR2, Newton, RFO, FIRE 1.0, quick-min, DIIS,
  Barzilai-Borwein, Adam, and PSO. BFGS, dogleg, and FIRE 2.0 leave
  that default off. BFGS takes the length from a strong Wolfe line
  search (Nocedal and Wright, algorithms 3.5 and 3.6,
  https://doi.org/10.1007/978-0-387-40065-5). Dogleg keeps the step
  in a Euclidean trust region and updates the radius from the
  model-agreement ratio on that step (Nocedal and Wright, algorithm
  4.1). A per-atom clip after the dogleg point shortens a boundary
  step, the ratio update reads an interior length, and the radius
  stays put. The same clip also becomes the radius ceiling.
  FIRE 2.0 sizes the move with its own `dt` (Guénolé et al. 2020,
  https://doi.org/10.1016/j.commatsci.2020.109584, when that
  schedule is selected, and the FIRE 2.0 integrator otherwise).
  Newton keeps the cap: the session takes the shifted step under
  `Accept::None`, with no trust radius of its own. One-shot SR1
  and SR2 pass the cap into `sr_trust_limits` and then clip the
  trial with `scale_step_atom`. One-shot BFGS passes no atom cap
  into the line search. `scale_step_atom` limits the largest
  per-atom displacement. It does not limit the Euclidean norm of
  the whole step. `set_atom_maxmove` still applies to every method,
  including a value above 0.2. A non-positive value clears the cap,
  and that choice stays if `set_maxmove` or `set_manifold` runs later.
  A positive `Control.maxmove`, with no call to `set_atom_maxmove`, is
  the Euclidean step limit and leaves the per-atom default off. A
  non-Cartesian manifold has no default. Switching a FIRE session to
  the Guénolé schedule drops the implicit cap. Switching it back
  follows the kind then in force. The C setter is
  `rgmin_solver_set_atom_maxmove`. Python `maxstep` and `atom_maxmove`
  are that setter. `OptimizeControl.maxmove` is 0, so the C++
  `optimize` helper keeps the method's per-atom default. ASE
  `Optimizer.defaults` is `{"maxstep": 0.2}` in
  `ase/optimize/optimize.py` at tag 3.29.0,
  https://gitlab.com/ase/ase/-/blob/3.29.0/ase/optimize/optimize.py.
  eOn documents `max_move` as 0.2 Å,
  https://henkelmanlab.org/eon/optimizer.html. `OptimizerOptions.maxMove`
  is `0.2` in `schema/eon_params.capnp` at `4be157d4`,
  https://github.com/TheochemUI/eOn/blob/4be157d43265af6b1140aaac0e52e42add62693f/schema/eon_params.capnp.
  On the half-scaled seed-1 packing from `bench/atomistic` `lj38`,
  clearing the cap lets the first L-BFGS and BFGS atom move
  256.0348909921298. That cleared step is the BFGS default. The
  same start with Euclidean `maxmove` 1000 finishes in one L-BFGS
  step at energy -0.0000007684753483636236 with a pair of
  715.2615061441937. The L-BFGS default cap reaches tolerance in
  199 steps at -169.49430900932856 with maximum pair
  3.895343497108207. With the cap set to 0.2, BFGS on that start
  finishes in 540 steps at -166.8029675390938 with maximum pair
  4.185205749883185. The peak per-atom step was 0.2 for L-BFGS and
  0.20000000000000015 for that capped BFGS run.
- SR1 stores the direct Hessian and steps with dogleg when that model
  is positive definite, otherwise with the Cauchy point, inside a
  Nocedal-Wright trust region. A line search on the inverse SR1 Hessian
  flips the sign of an uphill direction and then skips the update.
- SR2 is Powell's symmetric Broyden update in that same trust region.
  The old rank-1 map was neither symmetric nor secant.
- DIIS searches along the GDIIS image `sum c_i (x_i - α g_i)` when that
  direction is downhill. A displacement past eight caps, an uphill
  image, or a step that does not lower the force drops the history and
  takes a steepest step. The line search chooses the length.
- Adam scales the Kingma-Ba direction to unit max-atom length and opens
  the next line search at the accepted step.

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
  initial delay, and the mix after the kick, with Table 1 of Comput.
  Mater. Sci. 175 (2020) 109584, DOI 10.1016/j.commatsci.2020.109584.
  On rgpot surfaces (5 starts each, per-atom cap 0.2) it
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
- Rigid-motion removal runs through one Eckart workspace per session
  (`rigid` module): the basis is ranked by column-pivoted QR, frozen
  coordinates keep the combinations that do not move them, a periodic
  axis drops the rotations, and the host can mark broken translations
  and rotations. `Solver::set_rigid_mode` selects `Off`, `Auto`,
  `Translation` or `TranslationRotation`; `set_project_rigid(true)` is
  `Auto`; `set_pbc_axes`, `set_frozen`, `set_broken_axes`,
  `set_rigid_mass_metric` and `set_rigid_mass_weighted` declare the
  molecule. The session projects the step, the L-BFGS pairs and each
  L-BFGS trial direction, the FIRE force and the FIRE velocity (the
  velocity in the momentum metric), and the dogleg direction; Lanczos
  can project inside the product (`lowest_mode_with`). A Cartesian
  force is not run through the mass metric.
- `align_rigid` aligns two geometries by the Horn / Coutsias
  quaternion; a periodic axis skips the rotation.
- `MwRigid` stores `q = sqrt(m) x`. The chart metric is Euclidean in
  `q`. One atom is a quotient with three translations.
- The C ABI exports the rigid workspace, `rgmin_rigid_project`,
  `rgmin_rigid_project_hessian`, `rgmin_rigid_align`, and the solver
  setters; `rgmin.h` declares them, `xts.h` aliases them, and the C++
  session wraps the setters. `abi_minor` 35.
- Tests cover frozen ranks, a slab, and quaternion alignment. The
  crate root exports `RigidWorkspace` and `align_rigid`.
- Tests check that the projector is idempotent and symmetric in
  the active metric, and that a rigid motion of the geometry moves
  the projected vector.
- `align_rigid` is Horn / Coutsias. Any periodic axis leaves the
  rotation at identity and shifts the mobile cloud onto the
  reference centroid.
- `project_out_rot_trans` borrows one shared workspace. It no
  longer builds a private Gram-Schmidt basis.
- `RigidWorkspace` projects a vector and replaces `H` with `P H P`.
  The caller owns the buffers. The workspace is reused after the
  first call at a given size.
- Frozen coordinates are zeroed before the Eckart rank. One pin
  keeps three rotations, two pins keep one axis, and three
  non-collinear pins give rank 0.
- The Eckart basis is ranked with column-pivoted QR. The tolerance
  is relative to the leading R diagonal. The six-by-six frozen
  overlap is diagonalized.
- Eckart helpers live in `rgmin::rigid`. `RigidMode` names Off, Auto,
  Translation, and TranslationRotation. Auto is the integer
  `set_project_rigid(1)` will mean. Eckart vectors are weighted in
  the active metric.
- Exponential preconditioner (Packwood, Kermode, Mones, Bernstein,
  Woolley, Gould, Ortner, and Csányi, J. Chem. Phys. 144, 164109
  (2016), <https://doi.org/10.1063/1.4947024>).
  `Solver::set_exp` builds `P` from a caller neighbour list.
  L-BFGS uses `H0 = P^{-1}`. FIRE and nonlinear CG use `P` as the
  metric. A dense pair or Lindh matrix passed into the two-loop
  still wins. C: `rgmin_exp_t`, `rgmin_solver_set_exp`,
  `rgmin_solver_clear_exp`, `rgmin_solver_exp_mu`. The C++ session
  wraps those three as `set_exp`, `clear_exp`, and `exp_mu`.
  `abi_minor` 28.
- Steihaug-Toint Newton-CG as `Method::NewtonCg` (`newton-cg`) and
  `Method::NewtonPcg` (`newton-pcg`). `Method::from_name` parses the
  token. The session uses Hessian-vector products only: analytic via
  `Solver::step_hvp` / `minimize_method_hvp`, otherwise one forward
  difference of the known gradient. `control.maxmove` is the Euclidean
  trust-radius ceiling; a per-atom cap clips the accepted trial, and
  the boundary test uses the pre-clip length. `Solver::gradient_calls`
  and `Solver::hvp_calls` count the two separately.
  C: `RGMIN_NEWTON_CG`, `RGMIN_NEWTON_PCG`, `rgmin_method_from_name`,
  `rgmin_solver_step_hvp`, `rgmin_minimize_hvp`,
  `rgmin_solver_gradient_calls`, `rgmin_solver_hvp_calls`.
  `abi_minor` 27.
  Steihaug, SIAM J. Numer. Anal. 20 (1983),
  <https://doi.org/10.1137/0720042>.
  Nocedal and Wright, *Numerical Optimization*, chapter 7.
- HiGHS user callback (`rgmin_solver_set_highs_callback`) for
  logging and IPM interrupt. Constrained dest QPs are checked
  against SciPy `trust-constr` (and `linprog` `highs-ipm` on LPs).
  `abi_minor` 25.
- Closed HiGHS engine waist: `rgmin_highs_solver_t` (choose / simplex /
  ipm / ipx / hipo / pdlp / hipdlp / qpasm) and
  `rgmin_highs_crossover_t`. Constrained dest defaults to IPM with
  crossover off. `rgmin_solver_set_highs_solver` /
  `rgmin_solver_set_highs_crossover`. Unknown tokens return 1.
  `abi_minor` 24.
- `rgmin_solver_push_pair` and `rgmin_solver_search_direction` for a
  one-oracle-per-outer L-BFGS waist. `abi_minor` 23.

### Changed

- The default L-BFGS step is the eOn client's line-search-free cap, including the pieces the previous cap left out. The first step probes curvature over `EON_FINITE_DIFFERENCE` (0.01) and sets `H0 = 1/C` when that curvature is positive. Later steps use `(s·y)/(y·y)` when the quotient is positive, and `EON_INVERSE_CURVATURE` (0.01) otherwise. A proposal whose longest atom move is at least the cap clears the memory and steps along `H0` times the force. A direction that is not descent does the same. Negative curvature clears the memory and steps along the force at the cap. The cap only shortens a step. `LbfgsStep::Scaled` and `rgmin_solver_set_lbfgs_scaled` keep the previous rule, which shortens the two-loop step and retains the pairs. `LbfgsStep::Wolfe` remains the explicit strong Wolfe search. SciPy's CG coefficient `max(0, β_PR)` is `Conjugacy::PolakRibierePlus` (`prp`). The eOn conjugate-gradient restart, `|g·g_old| >= 0.5 ||g_old||^2`, is `Restart::powell` and the method token `eon-cg`. Hager-Zhang now applies the CG_DESCENT floor `β = max(β_N, -1 / (||d|| min(0.01, ||g_old||)))`. FIRE 1.0 treats a zero power as downhill, as the eOn client does. `abi_minor` 35.
- L-BFGS no longer line-searches by default. The default step is the eOn cap: the longest atom moves at most `EON_MAX_MOVE` (0.2), the initial inverse Hessian is the Nocedal-Wright scale `(s·y)/(y·y)`, pairs are kept when the step is shortened, and a direction that is not descent clears the memory. `LbfgsStep::Wolfe` and `rgmin_solver_set_lbfgs_wolfe` opt into the strong Wolfe search. `rgmin_solver_set_lbfgs_capped` with a non-positive cap restores 0.2 rather than the line search. `abi_minor` 34.

### Added

- Opt-in `Lbfgs::minimize_box` and `Lbfgs::minimize_box_watched` relax inside a box `lower <= x <= upper` with the generalized Cauchy point and the subspace minimization of L-BFGS-B (Byrd, Lu, Nocedal and Zhu 1995).
  Each direction ends at the subspace minimizer from the Cauchy point, and the strong Wolfe search stops at the box.
  `Lbfgs::minimize` keeps the eOn cap. `LbfgsStep::Wolfe` remains opt-in.
- `benches/ase_h2h.py` compares ASE 3.29 LBFGS, FIRE and BFGSLineSearch with rgmin capped L-BFGS, Wolfe L-BFGS, the geometry step and FIRE 2.0 on the same LJ7, LJ13, LJ38 and Morse Pt256 starts. Every arm uses fmax 0.01 and maxstep 0.2. The table is `benches/ase_h2h/results.md`.
- Opt-in capped L-BFGS for atomistic steps. `Lbfgs::capped` and `LbfgsStep::Capped` drop the line search. Each two-loop step is shortened so the longest per-atom displacement is at most the caller's cap, and the initial inverse Hessian stays the Nocedal-Wright scale `(s·y)/(y·y)`. An empty memory steps along the force at that length. Pairs are kept when the step is shortened. The default `LbfgsStep::Wolfe` is unchanged, and `set_atom_maxmove` still selects the geometry step. On LJ7 and LJ13 at atomic force 0.01 the geometry step uses fewer evaluations and the same energy, and on the extended Rosenbrock function with four or more coordinates the value near 3.98 is a local minimum in the basin of `(-1, 1, ..., 1)`. `Solver::set_lbfgs_step` installs the rule on a Euclidean L-BFGS session. C: `rgmin_solver_set_lbfgs_capped`, the C++ method `set_lbfgs_capped`, and the xts alias. `abi_minor` 33. `cargo run --release --example lbfgs_capped` prints force calls and median wall time.
- Opt-in holonomic distance constraints on every `Solver` method except particle swarm, under every `Accept` rule.
  `DistanceConstraints::at` fixes atom pairs at their lengths in a geometry, and `DistanceConstraints::new` takes the lengths as given.
  `with_masses` sets one mass per atom, and masses are one until it is called.
  Every correction and projection uses the inverse-mass metric.
  Trials return to the set along the bonds of the point they left.
  The Gauss-Seidel bond solver (SHAKE) of Ryckaert, Ciccotti and Berendsen (1977) and the linear constraint solver (LINCS) of Hess et al. (1997) do that.
  A rigid triangle such as three-site water goes to an analytic SETTLE (Miyamoto and Kollman 1992) for any masses and any shape.
  `ConstraintSolver::Auto` picks LINCS only where its expansion converges at least as fast as SHAKE.
  An explicit `ConstraintSolver::Lincs` on a block where the expansion diverges is `ConstraintError::LincsUnsafe`.
  Tolerances default to the rounding floor of each constraint.
  The coupling of each block sets the LINCS order and every iteration budget.
  RATTLE (Andersen 1983) projects directions and gradients onto the tangent space, and the stop test reads the projected gradient.
  The session projects the stored L-BFGS pairs, the conjugate-gradient and Barzilai-Borwein memory and the velocity of the fast inertial relaxation engine (FIRE) onto each new tangent space.
  Tangent pairs feed the Broyden-Fletcher-Goldfarb-Shanno (BFGS), SR1 and SR2 updates.
  L-BFGS runs its line search along the constrained curve under `Accept::Energy` and `Accept::Nonmonotone`.
  The Hessian of the Lagrangian on the tangent space drives Newton, rational function optimization (RFO), dogleg and Newton-CG.
  A failure is `Error::Constraint` with a typed `ConstraintError`, and leaves `x` and the method memory as they were.
  These settings return `ConstraintError::Unsupported`: manifolds other than Euclidean, the exponential preconditioner, the HiGHS step, a periodic cell, particle swarm, and rigid-mode projection at an active coordinate bound.
  Coordinate bounds act only on atoms outside every pair.
  When a trial takes a constrained atom to its bound, the step returns `ConstraintError::Bound`.
  C: `rgmin_constraints_t` with `rgmin_constraints_create`, `rgmin_constraints_create_at`, five setters, `rgmin_constraints_excess` and `rgmin_constraints_constrain`.
  `rgmin_solver_set_distance_constraints` and `rgmin_solver_clear_distance_constraints` install and drop a set.
  Failures return the new status `RGMIN_CONSTRAINT_ERROR`, and `rgmin_last_constraint_error` names the kind.
  `abi_minor` 31.
  The C++ wrapper has `rgmin::DistanceConstraints` and `rgmin::ConstraintError`, and the xts aliases follow.
  How-to `docs/orgmode/howto/constraints.org` and reference page `docs/orgmode/reference/constraints.org` describe them.
- `benchmarks/minimize.py` is an airspeed velocity (asv) suite that times one `rgmin_minimize` call on a convex quadratic in 16 and 64 dimensions.
  `snakemake -c 1` calibrates an upper bound for each benchmark from repeated runs of `develop`.
  `snakemake -c 1 pr_gate` exits non-zero when a benchmark is slower than that bound, and `benchflow/README.md` describes the workflow.
- Opt-in FIRE 2.0 parameter set and accelerated bias-corrected (ABC) FIRE on `fire2` sessions.
  `Solver::set_fire_preset(FirePreset::Guenole2020)` takes Guenole et al. 2020 Table 1 and keeps `dt` when a shrink would pass `dt_min` (Algorithm 2).
  `Solver::set_fire_abc` divides the mixed velocity by `1 - (1 - alpha)^(N+1)` (Echeverri Restrepo and Andric, Comput. Mater. Sci. 218, 111978 (2023)), as ASE `FIRE2(use_abc=True)` does.
  Under ABC the cap bounds each displacement component.
  C: `rgmin_fire_preset_t`, `rgmin_solver_set_fire_preset` and `rgmin_solver_set_fire_abc`; the C++ session and xts aliases follow.
  `abi_minor` 30.
  How-to `docs/orgmode/howto/fire.org` shows both.
  `benches/stop_norm.rs` adds the two settings and a mean step count.
- How-to `docs/orgmode/howto/install.org` shows how to install the 0.5.2 release from crates.io or the `develop` branch.
  It gives the dependencies and features of each route, the C library through cargo-c, and the SciPy the HiGHS tests need.
  The README, the index and the quickstart point to it.
- How-to `docs/orgmode/howto/cite.org` gives a BibTeX entry from `CITATION.cff` through cffconvert.
  It takes the version from `cargo tree`, since the record names no version on `develop` and an older one on `main`.
  The README, the index and the Pt256 tutorial point to it.
- Tutorial `docs/orgmode/tutorials/relax-morse.org`: relax the Pt256 Morse fixture with L-BFGS, FIRE and FIRE 2.0 through a `Solver` session.
  `examples/relax_morse.rs` is the program it runs.
  `docs/figures/relax-morse/` has the force-trace plot, the session diagram and the scripts that draw them.
- `CONTRIBUTING.md` names the issue tracker, the `develop` branch and the checks that CI runs.
  It points to the contributing page for the rest.
- Opt-in session stop norm.
  `Solver::set_stop_norm` takes a `StopNorm`: `Two` (the default, `||g||_2`), `MaxAtom` (the largest norm over `(x, y, z)` triples: ASE `fmax`, eOn `max_atom`, LAMMPS `max`), or `Inf` (the largest absolute component: eOn `max_component`, LAMMPS `inf`).
  Every session stop test and `Report::grad_norm` use it.
  The default path computes the same numbers as before.
  C: `rgmin_stop_norm_t`, `rgmin_solver_set_stop_norm` and `rgmin_solver_report_gradient`.
  The setter returns 1 for an unknown integer.
  The report call copies the gradient behind the last step report.
  The C++ session wraps them as `set_stop_norm` and `report_gradient`.
  xts aliases follow.
  `abi_minor` 29.
- `benches/stop_norm.rs` compares stop points of geometry L-BFGS and FIRE on LJ38 and Pt256 Morse under the two-norm and the max-atom norm.
- Exponential preconditioner (Packwood, Kermode, Mones, Bernstein, Woolley, Gould, Ortner, and Csányi, J. Chem. Phys. 144, 164109 (2016), <https://doi.org/10.1063/1.4947024>).
  `Solver::set_exp` builds `P` from a caller neighbour list.
  L-BFGS uses `H0 = P^{-1}`.
  FIRE and nonlinear CG use `P` as the metric.
  A dense pair or Lindh matrix passed into the two-loop still wins.
  C: `rgmin_exp_t`, `rgmin_solver_set_exp`, `rgmin_solver_clear_exp`, `rgmin_solver_exp_mu`.
  The C++ session wraps those three as `set_exp`, `clear_exp`, and `exp_mu`.
  `abi_minor` 28.
- Steihaug-Toint Newton-CG as `Method::NewtonCg` (`newton-cg`) and `Method::NewtonPcg` (`newton-pcg`).
  `Method::from_name` parses the token.
  The session uses Hessian-vector products only: analytic via `Solver::step_hvp` / `minimize_method_hvp`, otherwise one forward difference of the known gradient.
  `control.maxmove` is the Euclidean trust-radius ceiling; a per-atom cap clips the accepted trial, and the boundary test uses the pre-clip length.
  `Solver::gradient_calls` and `Solver::hvp_calls` count the two separately.
  C: `RGMIN_NEWTON_CG`, `RGMIN_NEWTON_PCG`, `rgmin_method_from_name`, `rgmin_solver_step_hvp`, `rgmin_minimize_hvp`, `rgmin_solver_gradient_calls`, `rgmin_solver_hvp_calls`.
  `abi_minor` 27.
  Steihaug, Society for Industrial and Applied Mathematics (SIAM) Journal on Numerical Analysis 20 (1983), <https://doi.org/10.1137/0720042>.
  Nocedal and Wright, *Numerical Optimization*, chapter 7.
- HiGHS user callback (`rgmin_solver_set_highs_callback`) for logging and for interrupting the interior point method (IPM).
  Constrained dest QPs are checked against SciPy `trust-constr` (and `linprog` `highs-ipm` on LPs).
  `abi_minor` 25.
- Closed HiGHS engine waist: `rgmin_highs_solver_t` (choose / simplex / ipm / ipx / hipo / pdlp / hipdlp / qpasm) and `rgmin_highs_crossover_t`.
  Constrained dest defaults to IPM with crossover off.
  `rgmin_solver_set_highs_solver` / `rgmin_solver_set_highs_crossover`.
  Unknown tokens return 1.
  `abi_minor` 24.
- `rgmin_solver_push_pair` and `rgmin_solver_search_direction` for a one-oracle-per-outer L-BFGS waist.
  `abi_minor` 23.

### Changed

- A default `meson install` of rgmin as the top-level project installs `librgmin.so` into `libdir`.
  `-Dwith_asv_helper=true` also builds and installs `librgmin_asv`, the timing helper that the asv suite loads.
- The README opens with what rgmin does.
  It spells out each method name on first use, along with IRC and the C application binary interface (ABI) terms.
- `Report` gains `grad: Option<Array1<f64>>`, the gradient that `grad_norm` measures.
  Sessions fill it and one-shot drivers leave it `None`.
  Code that builds a `Report` literal must set the field.
- `Error` is `#[non_exhaustive]` and gains `AcceptStalled`.
  That breaks code outside the crate that matches `Error` without a wildcard arm.
  Add a `_` arm, and later variants need no change.
- Newton and dogleg leave a first-order saddle.
  When the force is already under `gtol`, the report reads the analytic Hessian.
  The floor is `max(|gtol|, 1e-6, 1e-8 ||H||_F)`.
  A lowest eigenvalue below minus that floor steps along the mode, scaled to the per-atom cap, at most four times.
  The dogleg radius restarts from the initial step.
  A converged report costs one extra Hessian.
- The docs export installs Org 9.7 or newer from the Emacs Lisp Package Archive (ELPA) of the GNU's Not Unix (GNU) project before ox-rst.
  Pull requests run the Org export and the Sphinx build.
  The Pages deploy stays on a push to `main` or `workflow_dispatch`.
  A pull request has its own concurrency group, so it does not cancel a deploy.
  `pages: write` and `id-token: write` belong to the deploy job; the workflow token stays read-only.
- CI runs on pushes and pull requests to `develop` as well as `main`, and installs numpy and scipy for the HiGHS reference tests.
- Capped Cartesian L-BFGS now reserves its finite-difference curvature probe for nominal steps at least 64 times larger than the cap.
  Ordinary starts use a `0.1 I` inverse scale and five-value nonmonotone acceptance.
  An installed exponential preconditioner replaces that scalar seed, including those steepest fallbacks.
  Uncapped L-BFGS keeps the two-loop step and the Wolfe search.
  A HiGHS feasible set and a non-Euclidean manifold stay on that path.
- FIRE 2.0 now follows Guénolé et al.'s semi-implicit update order, half-step-back uphill correction with a fresh corrected-point force, initial time-step delay, 2,000-step non-positive-power limit, and cap-consistent velocity.
  An installed exponential preconditioner uses `P` as the metric: power is `F · v`, and the kick and the mix use `P^{-1} F`.
- Dogleg updates its trust radius from the dogleg step.
  Its Euclidean ceiling is `maxmove` rather than a per-atom displacement cap.
  A cap still limits the accepted trial.
  Nocedal and Wright, algorithm 4.1.
  The 0.2 per-atom default added in 0.4.1 used that cap as the ceiling and fed the clipped length to the boundary test, so the radius stayed near its initial value on Morse bulk.
- The crate is `rgmin`, at `OmniPotentRPC/rgmin`; it was `xtsci-optimize` under `HaoZeke`.
  The C ABI renames with it in one pre-publication sweep, to `rgmin_*` symbols and `rgmin.h`.
  The `xts::optimize` C++ namespace remains for source compatibility.
- The strong-Wolfe zoom proposes cubic-Hermite trials with both bracket slopes (Nocedal-Wright eq. 3.59) behind interior guards; measured on the LJ75 hopping battery this cut force calls per hop from 46 to 43.
- Every solver's length-n algebra flows through the `vecops` seam.
  The seam carries a DLPack-device-tagged `Vector` handle.
  Construction refuses a device tag without a kernel backend and never stages it through the host.
- Under `Accept::None`, an L-BFGS trial on a curved kind that the box refuses returns `Error::OffManifold` instead of `Error::Oracle`.
  The message of `Error::OffManifold` now covers L-BFGS as well as `Method::Fire`.

### Added

- `rgmin_solver_set_box`: per-coordinate HiGHS box on `x + p`.
  A NULL side is unbounded.
  Same status as `set_highs` (0, or 1 without the `highs` feature).
  `rgmin_solver_set_trust` sets the L_inf radius; `rgmin_solver_add_equality` / `rgmin_solver_clear_equalities` carry `a · p = rhs`.
  `HighsStep.lo` / `hi` are per-coordinate vectors (length 1 is uniform).
  `abi_minor` 22.
- `ManifoldKind::CenteredMatrix`: manopt `centeredmatrixfactory`.
  Packed row-major `m n`.
  Projection is the centering operator (`X 1 = 0` by default, or `1^T X = 0`), retraction is `X + U` then center, transport is the identity.
  C token 22.
  Tokens 7-10 stay reserved.
  `abi_minor` 21.
- `ManifoldKind::Positive`: manopt `positivefactory`.
  Packed length `n` (positive orthant / positive reals).
  Projection is the identity, retraction is `x ⊙ exp(v ⊘ x)`, transport is the identity.
  C token 21.
  Tokens 7-10 stay reserved.
  `abi_minor` 20.
- `ManifoldKind::SphereComplex`: manopt `spherecomplexfactory`.
  Interleaved `(re, im)` pairs of length `2 n`.
  Projection is `v - Re(x^* v) x`, retraction is `(x+v)/||x+v||`, transport is projection at arrival.
  C token 20.
  Tokens 7-10 stay reserved.
  `abi_minor` 19.
- `ManifoldKind::MultinomialSymmetric`: manopt `multinomialsymmetricfactory`.
  Packed row-major `n^2`, `n >= 2`.
  Fisher-orthogonal projection solves `(I + X) alpha = V 1`, retraction is `X ⊙ exp(V ⊘ X)` then Sinkhorn then symmetrize, transport is projection at arrival.
  C token 19.
  Tokens 7-10 stay reserved.
  `abi_minor` 18.
- `ManifoldKind::MultinomialDoublyStochastic`: manopt `multinomialdoublystochasticfactory`.
  Packed row-major `n^2`, `n >= 2`.
  Fisher-orthogonal projection solves the `[I X; X^T I]` saddle, retraction is `X ⊙ exp(V ⊘ X)` then Sinkhorn, transport is projection at arrival.
  C token 18.
  Tokens 7-10 stay reserved.
  `abi_minor` 17.
- `ManifoldKind::Constant`: manopt `constantfactory`.
  Singleton of packed length `n`.
  Projection and transport are the zero tangent, retraction is the fixed point.
  C token 17.
  Tokens 7-10 stay reserved.
- `ManifoldKind::EuclideanComplex`: manopt `euclideancomplexfactory(n)`.
  Interleaved `(re, im)` pairs of length `2 n`.
  Projection is the identity, retraction is `x + v`, transport is the identity.
  C token 16.
  Tokens 7-10 stay reserved.
- `ManifoldKind::SkewSymmetric`: manopt `skewsymmetricfactory`.
  Packed row-major `n^2`, `n >= 2`.
  Projection is `multiskew` (`0.5 (A - A^T)`), retraction is `X + U`, transport is the identity.
  C token 15.
- `ManifoldKind::Symmetric`: manopt `symmetricfactory`.
  Packed row-major `n^2`.
  Projection is symmetrization, retraction is `X + U`, transport is the identity.
  C token 14.
- `ManifoldKind::ComplexCircle`: manopt `complexcirclefactory(n)`, interleaved `(re, im)` pairs of length `2 n`.
  Projection and retraction keep each pair on `S^1`.
  C token 13.
- Typed lowest-mode waist: closed `EigensolverKind` (Lanczos through EigenExa, `schema/eigen.capnp`) and `EigenParams` (`nev`, `krylov`, `max_iter`, `tol`).
  Lanczos, Rayleigh-Ritz, Jacobi-Davidson, and LOBPCG run matrix-free.
  Other named backends return `Error::EigenUnavailable` / `RGMIN_UNAVAILABLE`.
  No string keys.
- Matrix-free Newton: the `HessianVector` trait, a finite-difference action wrapper, Steihaug-Toint CG inside a Nocedal-Wright trust region (`minimize_newton_cg`), and preconditioned CG with the Conn-Gould-Toint metric recurrences (`steihaug_pcg`).
- `NystromPrecond`: the Frangella-Tropp-Udell randomized sketch as a CG preconditioner; randomness lives only in the sketch, and the test suite pins the preconditioned step to the plain step.
- `minimize_recognized`: per-iterate basin recognition with the caller's substitute carried out under a flag.
- Lean proofs (Mathlib) for the zoom guard and its geometric envelope, the Steihaug boundary root, preconditioner scale invariance, the bounds on the trust-radius update, and the sketch's positive semidefiniteness, indexed in `docs/orgmode/reference/proofs.org`.
- Diataxis explanation pages deriving the line search, the secant family, trust regions, the scaled conjugate gradient (SCG) method, and the randomized preconditioning.

### Fixed

- A non-finite gradient can no longer satisfy the convergence test under either gradient norm.
- The energy-accept fallback faces the same test as the step it replaces.
  If the fallback also fails, the report gives the position unmoved.
- Session nonlinear CG restarts from steepest descent after `Accept::Energy` or `Accept::Nonmonotone` refuses a step.
  The refused direction used to be retried from the same point.
  Polak-Ribiere on boxed Rosenbrock 2D from `(-1.2, 1.0)` stayed at `f = 3.742860`.
  `Accept::None` is unchanged.
- Session Barzilai-Borwein drops its stored `(s, y)` pair after `Accept::Energy` or `Accept::Nonmonotone` refuses a step, so the next call starts from the `istep` length.
  The refused spectral length used to be retried from the same point.
  On boxed Rosenbrock 2D from `(-1.2, 1.0)` the energy stayed at `f = 0.116646`.
  `Accept::None` is unchanged.
- Session BFGS and SR1 under `Accept::Energy` or `Accept::Nonmonotone` take `istep` along the starting identity and rescale it by `s·y / y·y` (Nocedal-Wright 6.20) before the first update, as L-BFGS does with an empty memory.
  The raw `-g` step rose on all ten halvings, so on boxed Rosenbrock 2D from `(-1.2, 1.0)` they stayed at `f = 24.2`.
  A refused step resets the matrix to that identity, so SR1 no longer retries an uphill direction from the same point.
  `Accept::None` is unchanged.
- SR2 is a symmetric secant update.
  The `sr2.hpp` map `B += δ (y + Bs)^T / (δ·s)`, with `δ = y - Bs`, gives `B+ s - y = 2 (s·Bs) δ / (δ·s)`.
  It broke the secant equation (Nocedal-Wright 6.6) and left `B` asymmetric.
  Once `s·Bs` exceeded `s·y`, it doubled `B` along `s` on every step, so the steps halved and SR2 stalled away from a stationary point.
  SR2 now applies Powell's symmetrization with `c = y + Bs`: `B += (δ c^T + c δ^T) / (c·s) - (δ·s) c c^T / (c·s)^2`.
  With `a = s·y` and `b = s·Bs` both nonzero, that is the Broyden class (6.32) with `φ = a (a - b) / (a + b)^2`.
  It is not Powell symmetric Broyden (`c = s`).
  A symmetric `B` stays exactly symmetric in floating point.
- Session SR2 under `Accept::Energy` or `Accept::Nonmonotone` starts as BFGS and SR1 do.
  It takes `istep` along `B = I`, sets `B = I y·y / s·y` before the first update, and resets `B` to `I` after a refused step.
  On boxed Rosenbrock 2D from `(-1.2, 1.0)` it used to stay at `f = 24.2`.
  `Accept::None` is unchanged.
- A C session created with `istep = 0` sets its step scale to `min(1, 1/||g||)` at the first energy-tested steepest, BFGS, SR1, SR2, BB, or unpreconditioned CG step.
  The scale outlasts that step and stays until `forget` or `rebase`.
  Steepest descent and CG multiply every later gradient term by it, so the cap shortens all their steps.
  BFGS, SR1, and SR2 use it while the matrix is the starting identity, and BB when it has no pair or no positive curvature.
  SciPy BFGS starts its search at `min(1, 1.01/||g||)`, and L-BFGS-B at `1/||d||`.
  The raw `-g` step rose on every energy halving, so from `(-1.2, 1.0)` on Rosenbrock 2D these sessions stayed at `f = 24.2`.
  A positive `istep` and `Accept::None` are unchanged.
- Session steepest descent, nonlinear CG, BB, BFGS, SR1, and SR2 return `Error::AcceptStalled` when `Accept::Energy` or `Accept::Nonmonotone` refuses the steepest restart that follows a refused step.
  Each later call would retry that restart from the same point, at eleven oracle calls per step.
  C sessions report `RGMIN_INTERNAL_ERROR`.
  `Accept::None` is unchanged.
- Failed C callbacks return NaN-filled gradients and Hessians rather than fabricated zeros or identity matrices.
- The foreign function interface (FFI) waist reuses standing DLPack shells per solve instead of allocating per evaluation: +26 percent evaluations per second at n=30, +19 percent at n=225, identical trajectories.
- HiGHS thread setup serializes through `std::sync::Once` instead of racing on the environment.
- The `par` feature keeps vectors under 65536 elements on the serial path.
  At those sizes the rayon reductions measured slower than serial.

<!-- towncrier release notes start -->

## [0.2.0]

Rust rewrite of the C++ xtsci-optimize: solvers over eindir `DifferentiableObjective`, session C ABI, manifolds, HiGHS-projected steps.
The C++ xtensor history is `0.0.1` on the previous `main`.
