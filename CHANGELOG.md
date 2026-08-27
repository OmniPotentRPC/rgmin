# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Manifold kernels go through dest vecops (dlpk CPU `Vector`) and
  the `par` / rayon feature. Sphere / Euclidean / RigidQuotient /
  MwRigid proj/retr/transp use `vaxpy` / `dot` / `scale`. The 3N
  Eckart project (`src/rigid.rs`) reduces through `mul_assign` +
  `vdot` and updates through `axpy` / `scale`; no ndarray-only
  hot loop on a 3N cluster. Stay-on-set unit tests plus a
  `par`-feature rayon test (`par_dot_axpy_on_a_long_vector_uses_rayon`).
- `IrcTrust::project` / `cons` go through vecops (`axpy`,
  `mul_assign`, `scale`, `div_assign_floor`, `nrm2`). The inner
  IRC MW project does not call ELPA. Feature `par` has a
  sphere-bound test.
- Feature `chase`: assembled-H entry `lowest_mode_chase` calls
  `dchase_init_` / `dchase_` / `dchase_finalize_` through
  `src/chase_shim.c`. Seed is the only approximation (`mode=A`);
  degree optimization stays on (`opt=S`). `ApplyHessian`
  `lowest_mode` stays `Error::EigenUnavailable` and does not
  assemble `H`. Unbuilt (default, or the feature on without
  libchase) stays unavailable. `is_matrix_free` is false.
- `EigenParams.degree` / `extra` (schema `@5` / `@6`, C layout
  32 bytes). ChASE zeros map to degree 20 and
  `extra = max(8, ceil(0.2 * nev))`. `max_iter == 0` is 25 on
  the Chase arm, not `n`. No `chase_set` strings.
- ELPA / PDSYEVR window is range I: `il = 1`, `iu = nev`. A
  `nev = 1` kick gathers `Z` as `n x 1`, not `n x n`. Full
  `heev` only when `nev == n`.
- Closed ELPA stage tokens: kind `Elpa` is 1-stage (`solver = 1`),
  kind `Elpa2` is 2-stage (`solver = 2`). `ElpaParams.nblk` is a
  `u32` (0 selects 16). No `elpa_set` strings on the public ABI,
  no `ELPA_DEFAULT_*` env, no `Elpa2Cuda` / `Elpa2Hip` kinds.
- Assembled-H entry `lowest_mode_dense` is the only path that
  would call ELPA / ELPA2 / SLATE. `lowest_mode` on `ApplyHessian`
  stays `Error::EigenUnavailable` for those kinds and does not
  assemble `H`. Gate is `n >= 512`. C `rgmin_lowest_eigenpair`
  stays Hessian-vector and `RGMIN_UNAVAILABLE` for those kinds.
- DLA-Future assembled-H entry (`lowest_mode_dlaf`): `begin` must
  be 0 and `n >= 512`. ApplyHessian `lowest_mode` stays
  `Error::EigenUnavailable` and does not assemble `H`.
- EigenExa is full-spectrum only. `nev < n` is
  `Error::EigenFullSpectrum` and does not compute `n` pairs and
  drop them. `eigen_s` versus `eigen_sx` is `EigenExaAlgo` (0/1),
  not a string. The ApplyHessian entry never assembles `H`.
- Feature `primme`: PRIMME `dprimme` on the lowest-mode waist via a
  frozen `ApplyHessian` matvec and typed `primme_params` fields
  (`numEvals`, `primme_smallest`, `matrixMatvec`, optional
  `applyPreconditioner`). No PRIMME string keys
  (`primme_params_set` / `primme_set_member`). Unbuilt (default, or
  the feature on without libprimme) stays `Error::EigenUnavailable`.
- Feature `slepc`: SLEPc EPS on the lowest-mode waist via MatShell
  and typed `EPSSet*` / `STSet*` only. No `EPSSetFromOptions` and
  no PETSc options database. A PETSc host may pass a Pmat through
  `SlepcParams`. Unbuilt (default, or the feature on without
  PETSc/SLEPc) stays `Error::EigenUnavailable`. The waist does not
  call `SlepcInitializeNoArguments`.
- Typed `ApplyPreconditioner` on the lowest-mode waist:
  `PreconditionerKind` none / diagonal / block3 / user. LOBPCG,
  Jacobi-Davidson, and PRIMME apply `T` to the residual. Lanczos,
  dimer, and Rayleigh-Ritz ignore it.
- Lanczos two-pass full reorthogonalization of the built Q
  (SLEPc `EPS_ORTH_FULL`). Start-vector `||q0|| < eps` returns
  without a column (eOn). Residual `beta <= 1e-10 |alpha|` is
  linear dependence and does not emit another column (eOn /
  SLEPc breakdown-before-divide).
- L-BFGS `replace_newest` keeps the stored pair when the
  replacement fails the curvature test. Empty-memory
  `Accept::None` scales the first two-loop step by `istep`.
- Frozen golden-master fixtures for dest Sella steppers (RFO, QN,
  QN-IRC, P-RFO, TS-BFGS, RAS clip) minted from zadorlab/sella
  `optimize/stepper.py`, `optimize/restricted_step.py`, and
  `hessian_update.py`, plus manopt factory formulas
  (`tests/sella_manopt_gold.json`). Remint loads
  `RestrictedAtomicStep.cons` and records `sella_files`. Remint
  reads `MANOPT_ROOT` and records the factory files it used; dest
  tests load only the frozen JSON.
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
