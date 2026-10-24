# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `cargo cinstall` installs the hand-written `include/rgmin.h` and
  `dlpack/dlpack.h`. The header compiles as C and as C++. Version
  macros are `RGMIN_H_MAJOR`, `RGMIN_H_MINOR`, and `RGMIN_H_PATCH`.
  `HighsInt` stays out of the public header. CI installs into a
  prefix and compiles a C program and a C++ program against that
  header and the `rgmin` pkg-config file.

### Added

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

- Newton and dogleg leave a first-order saddle. When the force is
  already under `gtol`, the report reads the analytic Hessian. The
  floor is `max(|gtol|, 1e-6, 1e-8 ||H||_F)`. A lowest eigenvalue
  below minus that floor steps along the mode, scaled to the
  per-atom cap, at most four times. The dogleg radius restarts from
  the initial step. A converged report costs one extra Hessian.
- The docs export installs Org 9.7 or newer from GNU ELPA before
  ox-rst. Pull requests run the Org export and the Sphinx build.
  The Pages deploy stays on a push to `main` or
  `workflow_dispatch`. A pull request has its own concurrency
  group, so it does not cancel a deploy. `pages: write` and
  `id-token: write` belong to the deploy job; the workflow token
  stays read-only.
- CI runs on pushes and pull requests to `develop` as well as `main`,
  and installs numpy and scipy for the HiGHS reference tests.
- Capped Cartesian L-BFGS now reserves its finite-difference curvature
  probe for nominal steps at least 64 times larger than the cap;
  ordinary starts use a `0.1 I` inverse scale and five-value
  nonmonotone acceptance. An installed exponential preconditioner
  replaces that scalar seed, including those steepest fallbacks.
  Uncapped L-BFGS keeps the two-loop step and the Wolfe search. A
  HiGHS feasible set and a non-Euclidean manifold stay on that path.
- FIRE 2.0 now follows Guénolé et al.'s semi-implicit update order,
  half-step-back uphill correction with a fresh corrected-point force,
  initial time-step delay, 2,000-step non-positive-power limit, and
  cap-consistent velocity. An installed exponential preconditioner
  uses `P` as the metric: power is `F · v`, and the kick and the mix
  use `P^{-1} F`.
- Dogleg updates its trust radius from the dogleg step, and the
  Euclidean ceiling is `maxmove` rather than a per-atom displacement
  cap. A cap still limits the accepted trial. Nocedal and Wright,
  algorithm 4.1. The 0.2 per-atom default added in 0.4.1 used that
  cap as the ceiling and fed the clipped length to the boundary
  test, so the radius stayed near its initial value on Morse bulk.
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

- `rgmin_solver_set_box`: per-coordinate HiGHS box on `x + p`.
  A NULL side is unbounded. Same status as `set_highs` (0, or 1
  without the `highs` feature). `rgmin_solver_set_trust` sets the
  L_inf radius; `rgmin_solver_add_equality` /
  `rgmin_solver_clear_equalities` carry `a · p = rhs`.
  `HighsStep.lo` / `hi` are per-coordinate vectors (length 1 is
  uniform). `abi_minor` 22.
- `ManifoldKind::CenteredMatrix`: manopt `centeredmatrixfactory`.
  Packed row-major `m n`. Projection is the centering operator
  (`X 1 = 0` by default, or `1^T X = 0`), retraction is `X + U`
  then center, transport is the identity. C token 22. Tokens
  7-10 stay reserved. `abi_minor` 21.
- `ManifoldKind::Positive`: manopt `positivefactory`. Packed
  length `n` (positive orthant / positive reals). Projection is
  the identity, retraction is `x ⊙ exp(v ⊘ x)`, transport is
  the identity. C token 21. Tokens 7-10 stay reserved.
  `abi_minor` 20.
- `ManifoldKind::SphereComplex`: manopt `spherecomplexfactory`.
  Interleaved `(re, im)` pairs of length `2 n`. Projection is
  `v - Re(x^* v) x`, retraction is `(x+v)/||x+v||`, transport is
  projection at arrival. C token 20. Tokens 7-10 stay reserved.
  `abi_minor` 19.
- `ManifoldKind::MultinomialSymmetric`: manopt
  `multinomialsymmetricfactory`. Packed row-major `n^2`,
  `n >= 2`. Fisher-orthogonal projection solves
  `(I + X) alpha = V 1`, retraction is `X ⊙ exp(V ⊘ X)` then
  Sinkhorn then symmetrize, transport is projection at arrival.
  C token 19. Tokens 7-10 stay reserved. `abi_minor` 18.
- `ManifoldKind::MultinomialDoublyStochastic`: manopt
  `multinomialdoublystochasticfactory`. Packed row-major `n^2`,
  `n >= 2`. Fisher-orthogonal projection solves the
  `[I X; X^T I]` saddle, retraction is `X ⊙ exp(V ⊘ X)` then
  Sinkhorn, transport is projection at arrival. C token 18.
  Tokens 7-10 stay reserved. `abi_minor` 17.
- `ManifoldKind::Constant`: manopt `constantfactory`. Singleton of
  packed length `n`. Projection and transport are the zero
  tangent, retraction is the fixed point. C token 17. Tokens 7-10
  stay reserved.
- `ManifoldKind::EuclideanComplex`: manopt
  `euclideancomplexfactory(n)`. Interleaved `(re, im)` pairs of
  length `2 n`. Projection is the identity, retraction is `x + v`,
  transport is the identity. C token 16. Tokens 7-10 stay reserved.
- `ManifoldKind::SkewSymmetric`: manopt `skewsymmetricfactory`.
  Packed row-major `n^2`, `n >= 2`. Projection is `multiskew`
  (`0.5 (A - A^T)`), retraction is `X + U`, transport is the
  identity. C token 15.
- `ManifoldKind::Symmetric`: manopt `symmetricfactory`. Packed
  row-major `n^2`. Projection is symmetrization, retraction is
  `X + U`, transport is the identity. C token 14.
- `ManifoldKind::ComplexCircle`: manopt `complexcirclefactory(n)`,
  interleaved `(re, im)` pairs of length `2 n`. Projection and
  retraction keep each pair on `S^1`. C token 13.
- Typed lowest-mode waist: closed `EigensolverKind` (Lanczos through
  EigenExa, `schema/eigen.capnp`) and `EigenParams` (`nev`, `krylov`,
  `max_iter`, `tol`). Lanczos, Rayleigh-Ritz, Jacobi-Davidson, and
  LOBPCG run matrix-free. Other named backends return
  `Error::EigenUnavailable` / `RGMIN_UNAVAILABLE`. No string keys.
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
