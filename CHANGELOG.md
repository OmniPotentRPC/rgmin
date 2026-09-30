# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
