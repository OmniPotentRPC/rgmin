# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

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
  FIRE and the geometry L-BFGS fallback return `Error::Oracle` instead of stepping into a non-finite answer.
  They leave `x`, the FIRE integrator and the L-BFGS curvature as they were, so a retry probes the same trial.
  Through the C ABI these errors arrive as `RGMIN_INTERNAL_ERROR` with no NaN or clipped-to-the-bound coordinate sent to the callback.
- An L-BFGS `Solver` step under `Accept::None` records the caller's move as a curvature pair before its one trial.
  A non-finite trial now takes that pair back, so the memory is as it was when the step returns `Error::Oracle`.
  Before, each retry from the moved point recorded the same pair again.
- The contributing page now gives the docs export command that the workflow runs, `cd docs && emacs --batch -l export.el`.
  Of the two commands it gave before, one failed and the other wrote nothing.

### Added

- How-to `docs/orgmode/howto/install.org` shows how to install the 0.5.2 release from crates.io or the `develop` branch.
  It gives the dependencies and features of each route, the C library through cargo-c, and the SciPy the HiGHS tests need.
  The README, the index and the quickstart point to it.
- Tutorial `docs/orgmode/tutorials/relax-morse.org`: relax the Pt256 Morse fixture with L-BFGS, FIRE and FIRE 2.0 through a `Solver` session.
  `examples/relax_morse.rs` is the program it runs.
  `docs/figures/relax-morse/` has the force-trace plot, the session diagram and the scripts that draw them.
- `CONTRIBUTING.md` names the issue tracker, the `develop` branch and the checks that CI runs.
  It points to the contributing page for the rest.
- Opt-in adaptive trust radius for session limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS).
  `Solver::set_lbfgs_trust` sets the starting radius and `Solver::lbfgs_trust_radius` reads the current one.
  The radius update is Algorithm 1 of Burdakov, Gong, Zikrin, and Yuan, Math. Program. Comput. 9, 101-134 (2017), <https://doi.org/10.1007/s12532-016-0109-7>.
  The step truncates the two-loop quasi-Newton direction at the radius, as in the third L-BFGS variant of their comparison, without solving their trust-region subproblem.
  The per-atom or Euclidean cap applies after the radius.
  With `set_project_rigid`, the rigid modes leave the direction first.
  Good agreement between the predicted and actual energy drop at the boundary doubles the radius.
  Poor agreement shrinks it.
  Their equation 64 counts a change within `1e-11 |f|` as full agreement, so the step accepts a rise that small.
  A larger rise is refused.
  A non-finite trial shrinks the radius and the step reports `x`, while other first-order `Solver` steps return `Error::Oracle` for such a trial.
  A radius below `1e-15` returns `Error::TrustCollapsed`, even after an accepted ratio.
  A finite slope `g^T d` that is not negative after the pairs are dropped returns the new `Error::NoDescent`, and a NaN or infinite slope returns the non-finite direction error.
  On Euclidean L-BFGS without HiGHS the radius replaces the acceptance test and the geometry step.
  It is off by default.
  In the C application binary interface (ABI), `rgmin_solver_set_lbfgs_trust` sets the radius.
  A non-positive, NaN or infinite radius turns it off, and a null session returns 1.
  `rgmin_solver_lbfgs_trust_radius` reads it, or 0 when off.
  A step whose direction is not downhill or not finite returns `RGMIN_INTERNAL_ERROR`.
  The C++ session wraps them as `set_lbfgs_trust` and `lbfgs_trust_radius`.
  xts aliases follow.
  `abi_minor` 37.
  `benches/stop_norm.rs` prints a second table of five L-BFGS step rules, with budget hits and errors per arm.
  With the cap, the radius-on arm runs the trust step in place of geometry L-BFGS, so the two capped arms compare two step rules.
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

## [0.2.0]

Rust rewrite of the C++ xtsci-optimize: solvers over eindir `DifferentiableObjective`, session C ABI, manifolds, HiGHS-projected steps.
The C++ xtensor history is `0.0.1` on the previous `main`.
