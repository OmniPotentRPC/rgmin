# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

<!-- towncrier release notes start -->

## [0.8.0] - 2026-11-22

### Added

- Added `rgmin::krylov`: restarted, right-preconditioned flexible GMRES
    (`fgmres`) with a caller-chosen product budget, an optional
    multi-cycle stagnation rule and a per-step observer, and the
    Eisenstat-Walker choice-2 forcing term (`EisenstatWalker`). Every
    cycle closes on a recomputed `b - A x`; only that residual converges
    a solve, is reported and restarts the next cycle.
- Added a generic level-set constraint `LevelSetConstraint` for `Solver` (`set_level_set` / `clear_level_set`) that projects trial directions onto the tangent space, retracts accepted points to `value(x)=target` by Newton along the oracle gradient, and reports explicit `Error::LevelSet*` failures when retraction data are invalid or Newton stalls.
  - Exposed level-set control through the C ABI (`rgmin_solver_set_level_set`, `rgmin_solver_clear_level_set`) and the Python session binding (`Session(..., level_set=...)`, `Session.set_level_set(...)`).
- Added an opt-in FIRE 2.0 stabilization that caps `dt_max` when an uphill
    reset follows one downhill move, preventing repeated semi-implicit-Euler
    overshoot cycles on stiff systems. The Guénolé 2020 schedule leaves it
    off by default.
  - Added per-step FIRE diagnostic tracing (`FireTrace`,
    `Solver::start_fire_trace`, `Solver::take_fire_trace`) recording `dt`,
    `dt_max`, `alpha`, `power`, `n_pos`, `half_step_backs`, and
    `clipped_moves`.
  - Added C ABI and Python toggles for `cap_two_step_overshoot`
    (`rgmin_solver_set_fire_cap_two_step_overshoot`, `xts` alias, C++
    wrapper, and `Session(..., cap_two_step_overshoot=...)`), and bumped
    `RGMIN_ABI_VERSION_MINOR` from 47 to 48.
- Added cautious limited-memory Riemannian BFGS, Algorithm 2 of Huang, Absil, and Gallivan (SIAM J. Optim. 28, 2018, doi:10.1137/17M1127582). `minimize_rlbfgs` takes a manifold, and `Rlbfgs` sets the memory, the cautious slope, the Armijo backtrack, and the relative gradient stop. The C entry is `rgmin_minimize_rlbfgs` with `rgmin_rlbfgs_default` at ABI minor 51. The default Euclidean L-BFGS step stays the capped eOn step.
- Added matrix-free stochastic Lanczos quadrature in `slq`: preconditioned CG, thick-restart extreme pairs, Gauss and Gauss-Radau quadratures, and deflated Hutch++ log-determinant estimates with Student-t intervals.
- Added opt-in `Control::etol` and `Control::max_evals`, the LAMMPS `minimize etol` and `maxeval` stops. `etol` stops a run when `|E_k - E_{k-1}| < etol * 0.5 * (|E_k| + |E_{k-1}| + sqrt(f64::EPSILON))` between accepted steps and reports `Status::EnergyTolerance`. `max_evals` counts objective evaluations, line-search trials included, and reports `Status::EvalBudget`. `Report::evals` and `Solver::evals()` hold the count. Both default to `None`, and every driver and session method honours them, with `Error::StopUnsupported` for `etol` on particle swarm and `Error::InvalidControl` for a negative or non-finite `etol` or a zero budget.
  - Exposed the stops through the C ABI (`rgmin_solver_set_etol`, `rgmin_solver_clear_etol`, `rgmin_solver_set_max_evals`, `rgmin_solver_clear_max_evals`, `rgmin_solver_evals`, `RGMIN_STOP_ENERGY_TOLERANCE`, `RGMIN_STOP_EVAL_BUDGET`, ABI minor 50), the C++ `Solver`, the Python `SolverConfig` (`etol=`, `max_evals=`), `Session`, and the `rgmin.ase` and `rgmin.optimize` optimizers, and the LAMMPS `min_style rgmin` plugin, which passes `minimize etol` to the core. The new how-to page on stopping rules maps them to LAMMPS and SciPy.
- The unmet stop now carries its report. C++ `minimize_fn`,
    `minimize_hess_fn` and `minimize_eindir` throw `rgmin::StopUnmetError`
    with `report()`, and Python `minimize` raises `rgmin.StopUnmetError`
    with `.report`. Both derive from the exception they threw before,
    `std::runtime_error` and `RuntimeError`, and keep the message.
- `Report` now carries a `StopReason` (`Converged`, `MaxIter`, `LineSearchFailed`, `NoiseFloor`, `StepTolerance`, `PolishUnmet`, `Iterating`) and `converged()`. A spent budget stays `Ok` with `Status::Budget`. `Control::with_error_on_unmet` returns `Error::StopUnmet` with the report, including when the iteration budget is zero. The gradient is tested after the last step as well as at the top of the loop. A budget in which no trial moves is `LineSearchFailed`.
  - `Control::with_backward_polish` takes a shifted Newton or RFO step on a supplied Hessian or Hessian-vector product. The point is accepted only when the Newton decrement, every coordinate step, and the gradient norm sit under `backward_error_floor`. The floor is `n * f64::EPSILON * (1 + |V| + sum |x_i|)`. A tiny step that misses the gradient test is `PolishUnmet`.
  - C ABI minor 49 adds `RGMIN_STOP_LINE_SEARCH`, `RGMIN_STOP_POLISH_UNMET`, `RGMIN_UNMET_STOP`, `rgmin_solver_set_error_on_unmet`, `rgmin_solver_set_backward_polish`, `rgmin_backward_error_floor`, and the `rgmin_minimize*_opts` entries. `xts.h` aliases the new status and stop constants. `rgmin_control_t` is unchanged. Python `SolverConfig` accepts the same two flags, `Report.converged` follows `Stop.TOLERANCE`, and `backward_error_floor` is the floor helper.
- Add a strict radii-polynomial comparison for a scalar majorant. Every bound in the comparison rounds upward and the margin rounds downward. A non-positive radius, a negative or non-finite bound, or an image that is not strictly inside the radius returns a typed refusal. Certificate fields are private and are read through getters.
- Add a-posteriori nonlinear certificates built from caller-supplied majorants: a Newton-Kantorovich existence radius, an adjoint functional bound with nonlinear and quadratic remainders, and a residual-robust Morse-gap test that returns a typed error when the gap cannot be certified.
- Added GPMin, the Gaussian-process-accelerated local geometry optimizer of Garijo del Río, Mortensen, and Jacobsen (Phys. Rev. B 100, 104103, 2019), as a general system-agnostic Rust stepper (`rgmin::GpMin`, `Method::GpMin`) with a matching `rgmin.optimize.GPMin` ASE drop-in and how-to documentation.
  `GpMinConfig` and `SquaredExponentialKernel` are non-exhaustive and are built from `Default` with `with_*` setters.
  The kernel default is ASE `GPMin` with `update_hyperparams` false: weight 1 and scale 0.4.
- Added GROMACS `integrator = steep` semantics across the Rust, C, C++, and Python APIs, including MDP-unit mapping and explicit tolerance, budget, and machine-precision stop reports.
- Added GROMACS-style `integrator = cg` energy minimization as `Method::GromacsCg` with `GromacsCgConfig` and `GromacsCg`, the C selector `RGMIN_GROMACS_CG` with `rgmin_solver_set_gromacs_cg`, C++ `rgmin::Method::GromacsCg` with `Solver::set_gromacs_cg`, and Python `gromacs_mdp_options(integrator="cg")` with `rgmin.optimize.GromacsCg`. It pairs the `do_cg` Polak-Ribière direction rules (`nstcgsteep` periodic restart, `|beta| <= 5` cap, steepest-descent fallback, `sqrt(eps) * |E|` downhill energy-rise tolerance, `MaxAtom` force norm) with `rgmin`'s strong-Wolfe line search (rather than GROMACS's own line search) and refuses holonomic constraints.
- Added `HighsOptions::time_limit` and `HighsOptions::with_time_limit`. A value
  replaces the time limit of both the projection and the feasible step HiGHS
  solves, and `None`, the default, keeps the 0.05 s and 1.0 s limits they had.
  The feasible step solve also reports a refused `parallel` or `threads` option
  as `Error::Highs` instead of ignoring it.
- Added a sixth-order Magnus propagator with an embedded fourth-order estimate and Padé matrix exponential.
- Added the `inverse_laplace` module: fixed-Talbot and Weeks (Möbius–Laguerre) complex-contour inverse Laplace kernels, orthonormal Laguerre series evaluation, and complex dense Gaussian elimination with an inverse infinity norm. Contour results are typed estimates, never certified bars. `TalbotConfig` and `WeeksConfig` are `non_exhaustive`, start from `Default`, and take chainable `with_*` setters. `TalbotConfig::sigma` shifts the Talbot contour, and the optional `singularity_re_max` and `singularity_im_max` bounds make `talbot_invert` refuse with `ContourMissesSingularity` when the contour cannot enclose them. `weeks_invert` samples off both DFT grids and refuses a Laguerre mode that aliases onto them with `AliasedLaguerreModes`, and `WeeksConfig::laguerre_degree` must be a multiple of 4. A disk majorant whose singularity lies inside the disk is refused, an observed dyadic Laguerre block that fails the finite-resolution `n^{-3/2}` model is refused, and collapsed or non-finite Talbot nodes, underflowed Laguerre seeds, and non-finite inputs are refused with typed errors.
- CI checks that every C function in `include/rgmin.h` has a doc comment (`scripts/check_header_docs.py`), that every DOI and URL in the tracked sources resolves (`scripts/check_links.py`), and that rustdoc builds with warnings as errors.
  The docs workflow builds with `-W` and fails when the tracked rst under `docs/source` is stale.
- New docs: how-tos and tutorials for the level-set constraint, the supplied line search and the caller-supplied L-BFGS `H0`, and a Python reference with every `SolverConfig` option and default and the maturin classes.
  The tutorials include the example programs that CI runs, and the Python quickstart blocks run as a test. The 18 C functions that had no doc comment are documented.
- The GROMACS cg symbols `RGMIN_GROMACS_CG` and `rgmin_solver_set_gromacs_cg` are ABI minor 54.
- `Solver::try_step_hvp` with the `TryHessianVector` trait, and
  `try_lowest_eigenpairs` / `try_lowest_eigenpairs_precond` with the
  `TryApplyHessian` trait, take host callbacks that can fail. On callbacks
  that succeed they take the steps of `step_hvp` and `lowest_eigenpairs`
  bit for bit. The first callback error ends the call as `Error::Host`,
  nothing is evaluated after it, and `x` and the session stay as they
  were before the step. A Newton-CG or Newton-PCG session with a chart or
  distance constraints returns `Error::FallibleUnsupported`. Any other
  method returns `Error::NeedHvp`. The C ABI and the Python binding are
  unchanged.
- `python -m pip install rgmin` installs the nanobind package from PyPI.
  A `vX.Y.Z` tag builds the sdist and the wheels for Linux (`manylinux_2_28`) on x86_64 and aarch64, macOS 14 or newer on arm64 and x86_64, and Windows on x86_64, tests every wheel, and uploads them to PyPI through trusted publishing.
  CPython 3.10 and 3.11 each get a wheel, one abi3 wheel serves 3.12 and newer, and the free-threaded 3.14t and 3.15t each get a wheel.
  The package version comes from `Cargo.toml`, and the upload stops unless the tag names that version.
- `rgmin::secant::update_symmetric` applies a Powell symmetric Broyden, SR1 or Bofill update to a dense row-major symmetric model, invariant under scaling of the step and gradient change.
- `rgmin_control_default`, `rgmin_exp_default`, `rgmin_scg_params_default`, `rgmin_scg_options_default`, `rgmin_eigen_params_default` and `rgmin_eigen_options_default` return the Rust defaults of the C records a caller fills in, bit for bit, at ABI minor 46. `xts.h` spells them `xts_*_default`.
  A C caller must start each record from its initializer, and `rgmin_precon_t` from `rgmin_precon_init`, then assign only the fields it changes. A field added later then holds its default rather than stack bytes or zero.
  The C++ wrappers, the LAMMPS plugin, the asv helper, the C tests and the C ABI docs start from them.
  In `rgmin_minimize_scg` and `rgmin_minimize_scg_with_options`, a null params or options pointer stands for the default record, which holds the values a null pointer selected before.
  `RGMIN_C_TEST_RUNNER` and `RGMIN_LAMMPS_RUNNER` put a runner such as valgrind in front of the C test consumers and the plugin tests' `lmp`, and CI runs both under valgrind.

### Changed

- The Brent line search now keeps its tolerance floor proportional to
    the abscissa, `f64::EPSILON * (|x| + 1 / 1024)`, instead of the fixed
    absolute `1e-12`. Searches whose minimum lies below `1e-12` in step
    length resolve it. Only Brent searches move: `LineSearch::search`,
    `search_from` and the one-shot `minimize` steps with
    `LineSearch::brent` or `LineSearch::default()`, the C one-shot
    `rgmin_minimize` family, which passes `LineSearch::default()`, a
    session after `set_linesearch` with Brent, including the constrained
    L-BFGS curve search, and the caller-driven `SuppliedSearch`,
    `rgmin_linesearch` and Python `LineSearch("brent", ...)`. The session
    default stays strong Wolfe and does not change. Results that were
    already resolved agree to rounding.
- GROMACS steepest descent multiplies a rejected displacement by 0.5, matching `do_steep` in GROMACS 2025.4. `GromacsSteepConfig::with_shrink` selects another factor in `(0, 1)`, including the reference manual's 0.2.
- Public output structs are non-exhaustive, so a later field stays compatible for callers that read the fields and match with `..`. This covers the matrix-free Lanczos outputs (`TridiagEigh`, `QuadratureEstimate`, `PcgResult`, `DeflatedPair`, `ExtremePairsReport`, `SlqLogDetReport`), the Gauss–Legendre and Lobatto results, the Magnus step and matrix exponential, and the eOn step record `EonFeat`. A caller that supplies its own deflation pair uses `DeflatedPair::new`. `LineSearchOptions` is non-exhaustive as well and still starts from `Default` or `with_objective_roundoff`. The C ABI is unchanged.

### Fixed

- An L-BFGS run that spends its budget without moving `x` is now
    `StopReason::LineSearchFailed` from `Solver::run` and
    `Solver::run_gauged`, as `minimize_method` and `rgmin_minimize`
    already reported it. Python `minimize` reports `Stop.LINE_SEARCH`
    for the same run, where it reported `Stop.BUDGET`.
- Non-finite starts and trials now fail before any clip bookkeeping is
    recorded in `step_first_order` (`src/session.rs:3710`) and `assess`
    (`src/session.rs:5237`), and `clip_free` (`src/session/constrained.rs:97`)
    keeps its finite-only coordinate clipping.
  - ABC-FIRE component clipping now leaves the velocity unchanged when
    `vmax <= 0` in `fire_clip_components` (`src/fire.rs:609`) instead of
    forcing every component to zero.
- Synced C ABI bookkeeping at minor 48: `tests/c_abi.rs`, the exported-name
    baseline (`scripts/abi_baseline.txt`), and the architecture docs now agree
    with `include/rgmin.h` / `src/ffi.rs`.
- The C ABI no longer folds `StopReason::Iterating` into
    `RGMIN_STOP_BUDGET` for a finished run. `Iterating` is a step in
    progress, so it has no `rgmin_stop_t` of its own and no code is
    renumbered. A session step that has not stopped still reports
    `RGMIN_STOP_BUDGET`, as documented. The one-shots now return
    `RGMIN_INTERNAL_ERROR` and write nothing if a finished run ever
    carried `Iterating`, which no driver does today.
- `irun` on every `rgmin.optimize` optimizer now yields `False` for a run that stopped with `stop == "EvalBudget"`, as `run` returns it. It yielded `True` there before, because the ASE loop stops by asking `converged`.
- `rgmin.optimize.GromacsSteep.run` and `irun` now return `False` when the session stops with `stop == "EvalBudget"`, matching the other ASE-style optimizers in `rgmin.optimize`.
  - `Solver::try_step_hvp` now validates `etol` and `max_evals`, counts objective evaluations into `Solver::evals()` and `Report::evals`, and applies `etol` and `max_evals` at the end of a successful step while rolling back the evaluation count and stop state on `Error::Host`.
  - `polish_newton`, `polish_hvp`, and the `backward_polish` passes of `minimize_method_hess`, `minimize_method_hvp`, `minimize_newton`, and `minimize_newton_cg` now count the objective evaluations made during the polish into `Report::evals`.
  - `minimize_rlbfgs` now evaluates `etol` and `max_evals` at the end of each outer iteration rather than after each rejected Armijo trial.
- `rgmin::minimize_eindir` ignored `Control::error_on_unmet`, so a spent
    budget came back as a report. It now throws, as `minimize_fn` and
    `minimize_hess_fn` do. The C ABI gains `rgmin_minimize_eindir_opts`,
    the eindir one-shot with the `error_on_unmet` and `backward_polish`
    flags, at ABI minor 53. `rgmin_minimize_eindir` is unchanged.
- A GROMACS steep trial that misses its distance constraints is rejected and the displacement shrinks, matching `do_steep`. Crossing `GROMACS_STEEP_MIN_USTEP` stops with `Status::MachinePrecision`. An explicit `GromacsSteepConfig::max_constraint_shrinks` that runs out while the displacement is still at least that floor returns `Error::Constraint`.
- A nonnegative certificate majorant of `-0.0` is returned as `+0.0`. The acceptance test compared the value with zero and used to keep the sign bit.
- A zero-budget `gromacs-steep` report now fails when a per-atom gradient norm is not finite. The previous reduction kept a finite atom through `std::max` and dropped the NaN.
- Certificate radii, remainders, and the Morse-gap `certified` field are documented as IEEE 754 binary64 estimates in the default round-to-nearest, ties-to-even mode. `certified` was previously described as a strict lower bound on the exact gap.
- Constrained FIRE 2.0 adapts `dt`, `n_pos` and `alpha` once per step. The preparation step already applies that update, and the forward step no longer applies it again, so a rollback shrinks `dt` by `f_dec` once and a downhill step counts `n_pos` once.
- Riemannian L-BFGS now rejects zero-curvature pairs, applies gradient
  tolerances in the manifold metric, and covers retraction fallback behavior.
- The C header, `src/ffi.rs`, the manifold modules and the manifold how-tos no longer call `rgmin_manifold_t` tokens 7-10 reserved.
  They are SPD, Grassmann, hyperboloid and Poincare ball, and token 23 is U(n). `rgmin_solver_set_manifold` has always mapped them.
- The DIIS reference DOI in `Method::Diis` and `src/diis.rs` is `10.1016/S0022-2860(84)87198-7`, which resolves.
  The Manopt links point at the JMLR page and the Holm reference at JSTOR, because the identifiers they used are not registered.
  The relax-morse tutorial quotes what `examples/relax_morse.rs` prints.
- The GROMACS steep symbols are ABI minor 52, and `xts.h` aliases `RGMIN_STOP_MACHINE_PRECISION`. The `1e-12` displacement floor and the default of 34 constraint shrinks are rgmin safeguards.
- The IRC trust projection treats a mass weight equal to its floor as present
  for both the degenerate axis and the radial divisor, so the two no longer
  disagree about that coordinate.
- The Python tests pass on macOS x86_64, where Apple's libm `exp` rounds one value on the Pt7 EMT trajectory down and glibc rounds it up.
  The Pt7 EMT digest, like the LJ38 one, is pinned on glibc only, and every other platform compares the Pt7 frames and energies with `tests/python/data/pt7_golden.json` within 1e-12, absolute in positions and relative in energies.
  Every platform still requires the Pt7 runs at 1, 2 and 8 Python threads and at 1, 2, 3 and 8 Rayon threads to match a reference run on the same machine bit for bit.
  Windows takes the same comparison, so `PT7_EMT_WINDOWS` and the dispatch-only `windows-pt7-digest` CI job that recorded it are gone.
- The bordered cyclic determinant in `rgmin::periodic` now scales `I - T_0` as one block for a single period. A transfer matrix with a unit diagonal entry and tiny couplings, such as `[[1, 1e-160], [1e-160, 2]]`, used to overflow two separately scaled terms into NaN. A non-finite triangular diagonal now returns a non-finite error instead of a NaN log-determinant.
- The bordered cyclic solver now rejects invalid border dimensions, non-finite inputs, and overflowing or non-finite norm estimates with specific errors instead of returning misleading determinant or singularity results.
- The public `StopUnmetError` stub keeps its `report` attribute typed without
  diverging from the generated extension stub.
- The shrink budget for a GROMACS steep session is now a closed-form logarithm, capped at `GROMACS_STEEP_SHRINK_BUDGET_CAP` (one million), so a shrink factor near one no longer walks a loop toward `u32::MAX` and hangs construction.
- Use an energy safeguarded L BFGS step for ASE cell filters so scaled
  cell deformation coordinates converge without accepting uphill steps.
- `GromacsSteepConfig::default().with_shrink(...)` derived `max_constraint_shrinks` from the canonical `emstep = 0.01`, not the session's actual `istep`. `Solver::set_gromacs_steep_config` and `GromacsSteep::set_config` now rederive that budget from the session's own step size unless `with_max_constraint_shrinks` fixed it, so a noncanonical Rust session gets a budget sized for its own displacement instead of exhausting constraints too early or too late.
- `Method::GromacsCg` now keeps `-z + beta * d` unscaled in its Polak-Ribiere direction update under `Accept::Step` instead of multiplying `-z` by `istep`, and `Solver::set_objective_roundoff` updates only the objective-roundoff window while preserving the `sqrt(eps)` downhill-rise tolerance installed for `Method::GromacsCg`.
- `Method::GromacsCg` now matches GROMACS `do_cg`'s `sqrt(GMX_REAL_EPS) * |E|` downhill energy-increase tolerance near machine precision, `rgmin.optimize.GromacsCg.run` and `irun` return `False` on an `EvalBudget` stop, `nsteps` step counting is pinned by analytic tests, and documentation and docstrings accurately state that `gromacs-cg` pairs the GROMACS-style direction update with `rgmin`'s strong-Wolfe line search.
- `Session("gromacs-steep")` no longer installs the generic per-atom displacement cap. An explicit `max_atom` still applies, and other methods keep that cap when the argument is omitted.
- `lanczos_quadrature_fn` no longer reports `bracketed: true` when the Lanczos recurrence terminates early on an invariant subspace. The function takes no lower bound, so it never forms a Gauss-Radau bracket, and every other return path already reported `false`. The flag is `true` only when a caller-supplied `lower_bound` produced a bracket, as `lanczos_quadrature_log` reports it.
- `minimize_scg`, `minimize_scg_exact`, and `rgmin_minimize_scg` now apply a positive finite `Control::maxmove` to every trial step, including retries after non-finite values.
  Bounded steps compare their measured change with the damped model at the shortened step; zero, negative, infinite, and NaN caps remain unbounded.
- `rgmin_minimize_hess_opts`, `rgmin_minimize_hvp_opts`,
  `rgmin_minimize_eindir` and `rgmin_minimize_scg_with_options` return
  `RGMIN_INVALID_PARAMETER` when the initial `x` is not finite, as
  `rgmin_minimize` does. A block tensor import such as the seed of
  `rgmin_lowest_eigenpairs` rejects a DLPack version other than 1.0 with the
  same status.
- `slq_logdet_hutchpp` no longer decides whether a Hutch++ sketch vector joins the trace subspace with an absolute `1e-14` cutoff. The residual `(A - M_cv) s` is now compared with `1e-12` times the larger of `||A s||` and `||M_cv s||`, so rescaling the operator `A -> c * A` keeps the same sketch vectors. At `c <= 1e-16` the old cutoff emptied the subspace and inflated the standard error, and at `c = 1e16` it promoted roundoff left by an exactly cancelling control variate into the subspace. A sketch whose operator scale is zero is dropped, and one whose scale is not finite is refused with an error.
- `slq_logdet_hutchpp` now drops a Hutch++ sketch vector that lies in the deflated subspace. When projecting out the kernel and deflated pairs left at most `1e-12` of the vector's norm, the remainder was kept and the power iteration normalised that roundoff into a basis vector of the trace subspace. The vector is now discarded, as the stochastic probes already do.

## [0.6.0] - 2026-11-13

### Added

- `Report::grad` is the gradient that `Report::grad_norm` measures. Sessions fill it, and one-shot drivers leave it `None`.
- The combined C surface is ABI minor 40. Minor 37 is linesearch, 38 is `rgmin_lbfgs_direction`, and 39 is `rgmin_lowest_eigenpairs`. The symbols new at 40 are the 0.5.2 C functions develop did not have, including `rgmin_minimize_scg_with_options` and `rgmin_solver_set_objective_roundoff`. `rgmin_control_t` gains `noise_eps` and `noise_c`. `rgmin_report_t` gains `stop`.
- `rgmin_minimize_scg_with_options` takes an explicit SCG step rule. Null options and `RGMIN_SCG_ABSOLUTE_INFINITY` keep the absolute infinity-norm rule. `RGMIN_SCG_RELATIVE_EUCLIDEAN` stops on `||alpha d||_2 < tol_sol * (1 + ||x||_2)`. `Solver::set_objective_roundoff` sets the relative value window of a Euclidean Wolfe search. A positive window can accept a rise inside that window and stop at a stationary point that the strict test refuses.
- `Lbfgs::ftol` turns on the relative-decrease stop of L-BFGS-B.
  A line-search step that passes the test ends `minimize`,
  `minimize_watched` and `minimize_recognized`.
  A step from `f_k` to `f_{k+1}` passes when `f_k - f_{k+1}` is at most
  `ftol` times the largest of `|f_k|`, `|f_{k+1}|` and 1.
  The default of zero turns the test off, so a default `Lbfgs` takes the
  same steps as before.
  `LBFGSB_FTOL` is the SciPy default, `LBFGSB_FACTR` (1e7) times the
  machine epsilon.
  On a 38-atom Lennard-Jones cluster from a jittered grid, `LBFGSB_FTOL`
  cuts the evaluations of a default `Lbfgs` from 415 to 316 and leaves the
  largest gradient component at 3.4e-3.
  `cargo run --release --example lbfgs_ftol` prints these runs. ([#8](https://github.com/OmniPotentRPC/rgmin/issues/8))
- `Lbfgs::minimize_on` and `Lbfgs::minimize_box_on` offer each accepted iterate to a callback as the iteration, the value, and the coordinates.
  Returning false ends the run at that point, as `minimize_watched` does.
  A caller can project those coordinates, a bond length or a height above a slab, without the solver storing that map.
  `minimize_watched` and `minimize_box_watched` keep their two-argument callbacks and call these methods.
  A start outside the box is clipped before the first `minimize_box_on` call. ([#15](https://github.com/OmniPotentRPC/rgmin/issues/15))
- Opt-in steps beside the eOn cap: per-atom-only CG, a quasi-Newton step that keeps its direction under the per-atom cap, a compact trust-region solve, a curvature gate on the exponential preconditioner, and a sparse Lindh model Hessian.
  None of them replaces the default. The cap stays the eOn client step.
- The Lindh model Hessian is built from a linked cell list and applied with Jacobi-preconditioned CG, so the factor is not refused above 700 atoms.
  It stays opt-in. On the held-out set it does not win every problem in the same basin as the eOn cap.
- The `strong_scale` example, behind the `par` feature, measures CPU
  strong scaling on one fixed FCC Morse crystal.
  It runs L-BFGS (m = 10) under the strong Wolfe search, and FIRE 2.0
  with the Guenole 2020 preset and a 0.2 Angstrom per-atom cap, on
  1, 2, 4, ... threads up to the physical cores.
  The CSV lists the hardware, threads, N, median wall time, force calls,
  iterations, final gradient norm, the largest coordinate deviation from
  1 thread, and the parallel efficiency.
  An untimed solve per pool checks that every L-BFGS step ran the line
  search and met the strong Wolfe conditions.
  CI runs a short smoke of it.
- The maturin package under `python/` builds `BFGS`, `LBFGS` and `FIRE` as ASE `Optimizer` subclasses.
  `LBFGS` uses ASE's length control unless `eon_cap=True`.
  The nanobind `rgmin.ase` classes released in 0.4.0 are unchanged.
- `Control::stop_norm` selects the norm that a `Solver` session compares to `gtol`.
  It defaults to `StopNorm::Two`, so a default `Control` stops where it did.
  `Solver::set_stop_norm` still replaces it on a live session.
  `StopNorm::Rms` is `||g||_2 / sqrt(n)`, the `rms` convergence metric of eOn.
  `Report::stop_norm` names the norm of `Report::grad_norm`, and drivers that run no session report `StopNorm::Two`.
  C: `RGMIN_STOP_NORM_RMS` is 3 and `XTS_STOP_NORM_RMS` aliases it, at `abi_minor` 32.
- `Lbfgs::approx_wolfe` is an opt-in approximate Wolfe test for the rounding floor.
  A positive value accepts a trial within that fraction of `abs(f)` of the start of a strong Wolfe search when the strong curvature test holds.
  Zero, the default, keeps the strict search.
  The eOn cap does not consult it. `LbfgsStep::Wolfe` and `Lbfgs::minimize_box` do.
- `Lbfgs::builder` sets the memory, the tolerance, and the step rule.
  `Lbfgs::capped` and `Lbfgs::scaled` select the strong Wolfe search when the cap is not a positive finite length.
- `LbfgsPairs::push_rho` stores a pair with the caller's `rho` in place of `1 / (s · y)`, for an L-BFGS direction with a caller-supplied `H0`. The pair passes the curvature gate of `LbfgsPairs::push` and `Lbfgs::record` first, and a `rho` that is not finite and positive is declined, so a pair with non-positive curvature never enters the two-loop. `LbfgsPairs`, `Lbfgs::two_loop` and the eOn capped step share one two-loop, so the direction from a capped session's pairs is that session's step bit for bit. C: `rgmin_lbfgs_pairs_push_rho`, ABI minor 43. `rgmin_lbfgs_direction` keeps its signature and minor 38. C++: `rgmin::LbfgsPairs::push_rho`. Python: `LbfgsPairs.push(s, y, rho=None)`.
- `LbfgsPairs`, `lbfgs_direction`, and `Lbfgs::two_loop_h0` return the L-BFGS direction `-H g` from stored `(s, y)` pairs and a caller-supplied initial inverse Hessian. `H0` is a scalar, a diagonal, or a callback that applies `H0` to one vector. The call does not evaluate an objective. A scalar `H0` equal to the factor `Lbfgs::two_loop` multiplies by matches that direction bit for bit. An empty memory returns `-H0 g`. C: `rgmin_lbfgs_pairs_create`, `rgmin_lbfgs_direction`, and the xts aliases. C++: `rgmin::LbfgsPairs`. Python: `LbfgsPairs`, with `H0` a float, a NumPy array, or a callable. ABI minor 38.
- `LbfgsStep::Capped` drops the L-BFGS line search and shortens each two-loop step to a per-atom cap.
  The initial inverse Hessian stays the Nocedal-Wright scale `(s·y)/(y·y)`.
  `Solver::set_lbfgs_step` installs the rule, and `rgmin_solver_set_lbfgs_capped` is the C entry.
  `abi_minor` 33.
- `ManifoldKind::Product` is the weighted Riemannian product of other kinds, with the metric `g = sum_i w_i g_i` for weights `w_i > 0`.
  `ManifoldKind::product` builds it from `ProductFactor` values, each a kind, a packed length and a weight, and `ManifoldKind::power` is manopt's `powermanifold`; `Solver::set_product` and `Solver::set_power` select them.
  Projection, retraction, transport, exponential, logarithm and parallel transport act per factor, a factor's gradient and Hessian are divided by its weight, and the distance is `sqrt(sum_i w_i d_i^2)`.
  A session steps a weighted product in the chart `z_i = sqrt(w_i) x_i`, so steepest descent, L-BFGS and the Newton-CG trust region run on it unchanged, and the iterates do not depend on the unit of a factor whose weight carries that unit.
  The exponential preconditioner, the Lindh model, `set_project_rigid` and the HiGHS box read raw coordinates and return `Error::ProductChart` on such a product.
  `MwRigid` and `RigidQuotient` factors project with the session's masses and periodic flag.
  `explanation/product-manifold` derives the geometry and compares it with manopt, Manopt.jl and Manifolds.jl, pymanopt and geomstats.
- `ManifoldKind::sphere_within(basis)` is the unit sphere of the span of the rows of `basis`, and `ManifoldKind::sphere_excluding(basis)` the unit sphere orthogonal to them, such as a unit normal that leaves a centre of mass in place.
  Both are `SubspaceSphere`, with the projection `P_U - x x^T`, the metric-projection retraction, and the sphere's exponential, logarithm, distance, parallel transport and Hessian.
  `Error::SubspaceBasis` reports an empty, non-finite or dependent basis, or excluded rows that span the whole space.
- `Manifold` gains manopt's `inner`, `norm`, `ehess2rhess`, `exp`, `log`, `dist`, `parallel_transport`, `parallel_transport_along`, `transport_is_isometric` and `typical_dist`.
  Their defaults keep every kind as it was: the Euclidean metric, and `None` where no closed form is supplied.
  Every kind but the multinomial and rigid kinds supplies its Riemannian Hessian, the sphere family, SO(3), SE(3), `Positive` and the linear kinds also their geodesics, and `Positive`, `Spd` and the doubly stochastic kinds expose the metrics their `egrad2rgrad` already used, as does the trait implementation of `RiemannianManifold`.
  `reference/product-manifold` lists what each kind supplies.
- `ProductFactor::with_metric` gives a factor a caller's `TangentMetric` `G_x`, self-adjoint and positive definite on each tangent space, so that its part of the product metric is `w g(u, G_x v)`.
  `egrad2rgrad` returns the factor's gradient times `(w G_x)^{-1}`, and `minimize_rtr` then has the iterates of manopt's truncated CG preconditioned by `(w G_x)^{-1}`.
  Such a product has no exponential, logarithm, distance or parallel transport, and sessions refuse it with `Error::ProductTangentMetric`.
- `RiemannianManifold` is a Stiefel manifold `St(n, p)` with a QR or polar retraction, an SPD cone `SPD(p)` under the affine-invariant or log-Euclidean metric, or a product of them.
  Each has projection, Riemannian gradient, retraction, exponential, SPD logarithm, transport, inner product and one gradient step on caller-owned column-major buffers.
  C: `rgmin_manifold_handle_t` and the `rgmin_manifold_*` functions, ABI minor 41.
  The C++ wrapper is `rgmin::Manifold`, and `xts_manifold_*` aliases the C entries.
  Python: `rgmin.Stiefel`, `rgmin.SPD` and `rgmin.Product` on NumPy arrays.
  `examples/tutorial_manifold_fit.py` is run in CI.
- `Solver::set_lbfgs_trust_radius` truncates the L-BFGS ray and updates the radius by Burdakov's Algorithm 1.
  It is off by default. `set_lbfgs_trust` stays the eOn Newton-model radius.
  The capped step stays the default until the radius is on, and the eOn fast path does not skip it.
  C: `rgmin_solver_set_lbfgs_trust` and `rgmin_solver_lbfgs_trust_radius`, ABI minor 45.
- `Stepper::step` is the one-iteration seam a host such as rgsaddle calls on `Solver`.
  `FireConfig::ase` is the ASE FIRE parameter set, and `Solver::set_fire_config` installs it.
- `SuppliedSearch` and `rgmin_linesearch` take a caller-supplied energy and gradient.
  The search does not call an objective. Its trials match `LineSearch::search`.
  ABI minor 37.
- `benches/ase_h2h.py` compares ASE 3.29 LBFGS, FIRE and BFGSLineSearch with rgmin capped L-BFGS, Wolfe L-BFGS, the geometry step and FIRE 2.0.
  Every arm uses fmax 0.01 and maxstep 0.2 on the same LJ7, LJ13, LJ38 and Morse Pt256 starts.
  The table is `benches/ase_h2h/results.md`.
- `lammps/` builds `min_style rgmin`, a LAMMPS plugin (`plugin load`, LAMMPS 22Jul2025 or later) that runs the rgmin `lbfgs`, `fire`, `fire2`, `sd` and `bb` sessions through the C ABI.
  `minimize etol ftol maxiter maxeval` keeps its meaning: `ftol` is the force norm of `min_modify norm` over the full force after fixes, and `etol` is the relative energy test of `min_style cg`.
  `min_modify method`, `memory`, `trust`, `linesearch`, `accept` and `istep` select and tune the session, and `dmax` is the per-atom step cap.
  `fix box/relax`, per-atom extra degrees of freedom and unknown keywords stop with an error.
  Under MPI every rank runs the same session on the full coordinate vector, and 1, 2 and 4 ranks take the same steps.
  `howto/lammps-migration` and `reference/lammps-min-style` document the plugin, and `benches/lammps_minstyle.py` compares it with `cg`, `sd`, `fire` and `quickmin`.
- `lowest_eigenpairs` is block LOBPCG for the k lowest eigenpairs of a symmetric Hessian product (Knyazev 2001).
  The seed is an n-by-k block. Each pair carries its residual and whether that residual met the tolerance.
  `k = 0`, `k` above the dimension, and a non-finite value or product return an error.
  C: `rgmin_lowest_eigenpairs`, `rgmin_eigenpair_t`, and `rgmin_eigen_options_t`.
  The C++ wrapper is `rgmin::lowest_eigenpairs`, and `xts_lowest_eigenpairs` aliases the C entry.
  `rgmin_lowest_eigenpairs` is ABI minor 39. `rgmin_lbfgs_direction` stays 38.
- `rgmin.lowest_eigenpairs` is the Python binding of block LOBPCG.
  It takes a Hessian-vector callable and a seed block and returns NumPy arrays.
  A pair that misses the tolerance raises `RuntimeError`.
  `examples/tutorial_lowest_eigenpairs.py` is run in CI.
- `rgmin.optimize` is a drop-in for `ase.optimize` on the maturin session.
  `BFGS`, `LBFGS`, `LBFGSLineSearch`, `FIRE` and `FIRE2` take ASE's constructor arguments and drive `self.optimizable`, so cell filters and `FixAtoms`, `FixedLine`, `FixCartesian` and `FixedPlane` work.
  The other gradient methods of `Session` have ASE-style classes. Importing `rgmin` does not import ASE.
  `Session` takes `h0`, `maxmove`, `preset`, `abc`, `accept` and `istep`, and can `push_pair`, `forget` and report `pair_count` and `steps`.
  `Adam` takes `learning_rate` (default 0.02) as its step scale.
  A restart file replays the recorded evaluations into a fresh session.
- `rgmin_last_error_message` is the borrowed string `rgmin_last_error` already returns.
  `rgmin_solver_dim` and `rgmin_solver_steps` read a const session, including a null one as zero.
  The C++ `Method` names `EonCg` and `PolakRibierePlus`. ABI minor 36.
- `rgmin_lowest_eigenpairs_ex` reports each pair's residual `||Hv − θv||` and a converged flag, and takes an optional preconditioner callback. A null preconditioner is the identity and matches `rgmin_lowest_eigenpairs` bit for bit. `rgmin_eigenpair_t` keeps the minor-39 layout. Python `lowest_eigenpairs` returns residuals and converged flags. `strict=False` returns those flags instead of raising `RuntimeError`. `precond` is a callable on an `(n, m)` block. ABI minor 42. Minor 40 is the union with the 0.5.2 C surface, and 41 is the manifold API.
- `rgmin_solver_set_power(solver, factor, block, count)` is `Solver::set_power` for the manifold tokens without shape parameters.
  `xts.h` aliases it and `rgmin::Solver::set_power` wraps it. `abi_minor` 44.
  Weighted products, subspace spheres, tangent metrics and gauges stay Rust only.
- `rtr::minimize_rtr_gauged` calls a gauge after every accepted step: a map to another representation of the same point, such as the pivot of a plane moved along the plane.
  The step is accepted on the value before the move, and the moved point is evaluated afresh.
  `Solver::run_gauged` calls the gauge before every session step but the first and rebases the session when it moves the point.
- `rtr::minimize_rtr` is the Riemannian trust region of Absil, Baker and Gallivan with Steihaug-Toint truncated CG, measured in `Manifold::inner`.
  Its Hessian action is `ehess2rhess`, Weingarten term included, where the kind supplies one, and the projected Euclidean action otherwise.
  `RtrOptions` takes the defaults of manopt's `trustregions`, and `RtrMin` returns the point, the value, the gradient norm and the outer, accepted and inner iteration counts.
  `rtr::truncated_cg_riemannian` is one inner solve in the manifold metric, for a caller that accepts or rejects steps itself with `RtrRadius`.
  `truncated_cg_projected` keeps the Euclidean inner product.
- `scripts/check_changelog.py` fails when `docs/newsfragments/` holds a fragment and the Unreleased section of `CHANGELOG.md` or `docs/orgmode/changelog.org` also holds a line.
  A file there whose type `towncrier.toml` does not name fails the same way.
  `towncrier build --draft` renders the fragments and changes no file.
- `towncrier.toml` reads the fragments in `docs/newsfragments/` again.
  `towncrier build` writes them into `CHANGELOG.md` under the Unreleased
  heading, above the newest release, with the six section types of Keep a
  Changelog.
- FIRE 2.0 stops when consecutive non-positive-power steps exceed
  `vdfmax`. The published schedule uses 2000, from Table 1 of Comput.
  Mater. Sci. 175 (2020) 109584. The stopping step does not move.

### Changed

- `rgmin_solver_create` opens a session with `RGMIN_ACCEPT_ENERGY`. In 0.5 it opened with `RGMIN_ACCEPT_NONE`. The Rust `Solver::new` default stays `Accept::None`. The default L-BFGS step is the eOn cap and runs no energy test under either policy, and line searches take the same value window under both. Under the energy policy a step that is not line-searched (BB, the Hessian steps) backtracks on a rise above 1e-8, a non-positive `istep` opens steepest descent, unpreconditioned nonlinear CG, BFGS, SR1, SR2 and BB with a unit-length step, and BFGS, SR1 and SR2 rescale their first step. To keep the 0.5 behaviour, call `rgmin_solver_set_accept(s, RGMIN_ACCEPT_NONE)` after `rgmin_solver_create`. The C++ `rgmin::Solver` follows the C default unless its configuration sets `accept`.
- A negative `maxmove` on `rgmin_control_t` leaves every step cap off, including the default per-atom cap. Zero keeps the defaults. A positive value is the Euclidean cap.
- SCG reductions stay in plain left-to-right order (`dot_ordered`), as released in 0.5.2. A positive finite `maxmove` also bounds every SCG trial. Other values leave that step unbounded.
- One-shot `minimize_method` keeps the line search the caller passes for L-BFGS, and `rgmin_minimize` keeps its Brent search.
- An interior point skips the bounds clip, and a Euclidean session keeps the gradient it already owns.
  Polak-Ribiere forms `g · (g - g_old)` in the dot product.
  A backtracking line search keeps its lowest trial without copying that vector.
  FIRE 2.0 does not build the post-step force, which only FIRE 1.0 mixes.
  L-BFGS reuses the curvature dots stored with each pair.
  The Morse-chain iterates are unchanged.
- Every public configuration type is `#[non_exhaustive]` with chainable `with_*` setters, one per field, so a knob added later is not a breaking change.
  `Control`, `NoiseFloor`, `EigenParams`, `ScgParams`, `ScgOptions`, `HighsOptions`, `HighsStep`, `ExpParams`, `LindhParams`, `FfParams`, `SlepcParams`, `EigenExaParams`, `DlaFutureParams`, `ElpaParams`, `EonCg`, `QuickMinOptions`, `FireConfig`, `TrustRegion`, `IrcTrust`, `RtrRadius`, `LindhSpec` and `Report` are built from `default()` or a named constructor (`NoiseFloor::new`, `TrustRegion::new`, `IrcTrust::new`, `RtrRadius::new`, `LindhSpec::new`, `FireConfig::ase`, `Report::new`); a struct literal outside the crate no longer compiles.
  A setter for an `Option` field takes `impl Into<Option<T>>`: `Control::default().with_maxmove(0.2)` is `Some(0.2)`.
  The result types (`LowestMode`, `Eigenpair`, `Eigenpairs`, `RestrictedStep`, `TcgResult`, `EonCgReport`, `LineOutcome`, `NoiseEstimate`, `ValueGradientNoise`, `Ode12rVerdict`, `ConstraintReport`, `ElpaSpectrum`) are non-exhaustive as well.
  The parameterised variants of `Method`, `LineSearch`, `LbfgsStep`, `Restart` and `Conjugacy::Hybrid` are non-exhaustive; build them with `Method::lbfgs_with_memory`, `Method::diis_with_memory`, `Method::fire`, `Method::newton`, `Method::nlcg_with_restart`, `Method::adam_with`, `Method::pso_with`, `LineSearch::brent` / `backtracking` / `goldstein` / `wolfe`, `LbfgsStep::capped` / `scaled`, `Restart::njws_at` / `powell_at` and `Conjugacy::hybrid`, and match them with `..`.
  `LbfgsStep::default()` is still the eOn cap at `EON_MAX_MOVE`.
  Fallible entry points return `rgmin::Result`: `fire2_displacement` and `fire2_displacement_metric` stall with `Error::FireStalled` (the `Fire2Stalled` type is gone), `StiefelNp::new`, `Grassmann::new` and `Unitary::new` refuse a shape with `Error::ManifoldShape`, and `Vector::try_on` refuses a device with `Error::UnsupportedDevice` (the `UnsupportedDevice` type is gone).
  The how-to "Configure a solver" documents the pattern.
- L-BFGS no longer line-searches by default.
  The default step is the eOn cap: the longest atom moves at most `EON_MAX_MOVE` (0.2), the initial inverse Hessian is the Nocedal-Wright scale `(s·y)/(y·y)`, pairs are kept when the step is shortened, and a direction that is not descent clears the memory.
  `LbfgsStep::Wolfe` and `rgmin_solver_set_lbfgs_wolfe` opt into the strong Wolfe search.
  A non-positive cap passed to `rgmin_solver_set_lbfgs_capped` restores 0.2.
  `abi_minor` 34.
- The contributing guide asks for a towncrier fragment instead of an edit
  to `CHANGELOG.md`, and names the fragment types.
  It lists the clippy commands that CI runs with warnings denied on Rust
  1.88.0 and stable.

  The Unreleased sections stay empty, and `scripts/check_changelog.py` fails when a fragment is paired with an Unreleased line.
- The default L-BFGS step is the eOn line-search-free cap.
  The first step probes curvature over 0.01.
  A proposal at least as long as the cap clears the memory and steps along `H0` times the force.
  `LbfgsStep::Scaled` keeps the shorten-and-retain rule.
  FIRE 1.0 treats a zero power as downhill.
- The optional `readcon` feature takes readcon-core `=0.17.1` from crates.io instead of git `694e808`. The published crate's manifest matches that commit's, and the library sources the feature compiles are the same; the files that differ are behind readcon-core's `parallel` and `python` features, its tests, and its docs. A crate that depends on rgmin and on rgsaddle, which takes readcon-core from crates.io, now resolves a single readcon-core.
- The session sample of the quickstart runs the `Solver` until the
  gradient norm drops below 1e-4, with `Accept::Energy`.
  It used to take one step and check only that the energy was finite.
  The list of manifolds moved out of the tutorial into a link to the
  manifolds how-to.
- `Control` and `Report` each gain a public `stop_norm` field, and `StopNorm` gains the `Rms` variant.
  Set it with `Control::with_stop_norm`; an exhaustive `match` on `StopNorm` needs an `Rms` arm.
- `ManifoldKind` is no longer `Copy`, since a product and a subspace sphere share their geometry through an `Arc`; clone it where a copy was implied.
  `stiefel_p` and `as_str` take `&self`, and the new `Product` and `SubspaceSphere` variants extend exhaustive matches on the kind.

### Fixed

- A Wolfe L-BFGS session with an explicit per-atom cap took the geometry step even after `set_geometry_step(false)`.
  Its energy window could accept a rise, so the step met neither strong Wolfe condition.
  Off now keeps the strong Wolfe search, with the cap scaling each trial.
  Unset, the combination is still the geometry step.
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
- The Exp pair weight and the μ sine probe call `libm` `exp` and
  `sin`. The host math library does not agree between x86_64 and
  aarch64. A hash of those `f64` bits is locked in the test.
- A `Control::maxmove` or `Solver::set_maxmove` value that is not a positive finite length is no Euclidean cap on every path that reads it: the dogleg, Newton-CG and Newton-PCG trust ceilings in a session and in `minimize_method_hess`, `minimize_method_hvp` and `minimize_newton_cg`, the constrained trust step and clips, the FIRE fill of `Solver::run_fill`, and every step clip. Before, such a value pinned the session trust ceiling at the current radius, so the radius never grew, and pinned the `minimize_newton_cg` ceiling at its radius floor. `-inf` broke the constrained runs of steepest descent, nonlinear CG and QuickMin. The FIRE fill scaled a step by `cap / |step|`, so a negative cap reversed it and the run climbed, from 30.1 to 125 in six steps under `-1`. A negative or infinite value still turns the default per-atom cap off, and zero and NaN still keep it. The C `rgmin_solver_set_maxmove` reaches the same session paths and is fixed with them. The C signatures are unchanged.
- A zero iteration budget (`maxiter = 0`) now judges the start it was given. `Solver::run`, the `minimize_*` one-shots, `rgmin_minimize`, `rgmin_minimize_hess`, `rgmin_minimize_hvp`, `rgmin_minimize_eindir` and Python `minimize` evaluate `x` once, take no step, and report that value and gradient norm with `Tolerance` below `gtol`, `NoiseFloor` under the noise floor, else `Budget`. Before, `Solver::run` and Python `minimize` reported value 0 and gradient norm 0 (`Solver::run` with `Tolerance`) without an evaluation, and the FIRE, BB, QuickMin, ODE12r, DIIS, Newton-CG and dogleg one-shots reported `Budget` with an infinite gradient norm even at a stationary start. A positive budget also changed: `Solver::run` reports `Budget` instead of `Iterating` when it runs out, and `Solver::run` and those one-shots end on a step's `NoiseFloor` instead of stepping past it. The eOn CG report reads `converged` from the gradient it returns. `rgmin_minimize_scg` keeps `maxiter = 0` as no limit. The C signatures are unchanged.
- An L-BFGS session driven by the caller's own `Solver::step` loop (C `rgmin_solver_step`, `rgmin_solver_step_fg`, Python `Solver.step`) with no iteration budget (`maxiter = 0`) stops on a plateau the objective cannot resolve. After `max(n, 2)` consecutive accepted steps that move `x`, keep the value inside the objective roundoff window (`set_objective_roundoff`, else four epsilon relative) and keep the gradient norm within `sqrt(eps)` relative of the first of them while it is still at or above `gtol`, the step reports `Status::StepTolerance` (C `RGMIN_STOP_STEP_TOLERANCE`). Before, such a fit reported `Iterating` forever: on the default capped eOn step and on the strong Wolfe opt-in, whose approximate Wolfe test accepts steps of about `1e-10` with no measurable decrease. A single flat step, such as an overshoot, does not stop the fit. A positive `maxiter` reports what it did before, and `Solver::run`, `minimize_lbfgs` and the C and Python one-shots still read `maxiter = 0` as no steps. The C signature is unchanged.
- An ODE12r session step that no cap or bound shortened is judged at its own length `h`, and the next trial starts from the length the controller chose.
  Before, the session recomputed the taken length as `h |x' - x| / |dx|` for every move, and that ratio rounds to a few ulps below `h` about as often as not.
  An accepted move that rounded short then took the capped-move floor, so the next trial reused `h` in place of the shorter line-search or error-controlled length.
  Whether a step was floored depended on rounding alone: a change in the last bits of the forces, such as the libm Exp weight, moved the step count of a run.
  A capped or clipped move is judged and floored as before.
- An unconstrained Newton-CG or Newton-PCG `Solver::run` ends with the Riemannian gradient in its report.
  It reported the Euclidean gradient that its difference products keep, so on a curved manifold a run that stopped on a vanishing Riemannian gradient reported the normal part of the Euclidean one.
- Links between docs pages and links to the Lean proofs open on the built
  site.
  They were `file:` links, which the export turns into links to `.rst`
  files that the site does not serve.
  Links between pages are now `:doc:` roles, and the Lean links go to the
  files on GitHub.
- SCG with no iteration budget (`maxiter = 0`) stops on a flat plateau. After `max(n, 2)` consecutive accepted steps whose objective change passes `tol_func` and whose gradient norm never falls below its value at the first of them, it returns `Status::StepTolerance` (C `RGMIN_STOP_STEP_TOLERANCE`), or `Tolerance` or `NoiseFloor` when the gradient qualifies for those. Before, such a fit ran until `lambda_limit` and reported a gradient error, and it never returned when `lambda_limit` was at or above the internal damping cap of `1e100`. A single flat step, such as a Newton overshoot to a point of equal value, does not stop the fit. Fits with a positive `maxiter` follow the same trajectory as before. The C signature is unchanged.
- The FIRE fill step cap folds atom distances with iterators instead of an indexed `f64::max` loop.
  rustc 1.93 and 1.94 (LLVM 21) segfault in LoopVectorizePass on that indexed loop, so `cargo build --release` failed there.
  A non-positive or non-finite cap still leaves the trial alone.
  The rescaled coordinates match the indexed loops bit for bit.
- The table of manifold tokens in the manifolds how-to renders again.
  One row was a character short, so Sphinx dropped the whole table.
  The table now lists every token that the Org page lists, `Positive`
  included.
- The user docs describe the 0.6.0 code.
  L-BFGS takes the eOn step by default, with a per-atom cap and no line search, and `LbfgsStep::Wolfe` opts into the strong Wolfe search.
  A session's line search defaults to strong Wolfe for the methods that use one.
  The install guide covers crates.io, git, cargo-c, Meson, both Python packages and the LAMMPS plugin, and its commands run as written.
  The README, the reference pages and the contributing guide give the current ABI minor, methods, manifold kinds and CI rows.
  The LAMMPS benchmark table names the commit and crate version it was measured at.
  The Sphinx and rustdoc builds pass with warnings as errors.
- `Lbfgs::minimize_box` honors `ftol`, the same relative-decrease stop as `minimize`.
  A short gradient, a NaN bound, an inverted bound, or a bound of the wrong length ends the call at `+inf` instead of panicking.
- `Solver::set_atom_maxmove` and the C `rgmin_solver_set_atom_maxmove` treat a non-finite value as no per-atom cap, the same as a non-positive one. Before, infinity was stored as the cap, so an L-BFGS session took it as its eOn per-atom cap and its next step failed with "capped L-BFGS: the cap is not positive" (C `RGMIN_INVALID_PARAMETER`); the session now falls back to 0.2, as it does for zero. The C++ `Solver::set_atom_maxmove` and the `atom_maxmove` and `maxstep` options of `SolverConfig` take the same call.
- `Solver::set_objective_roundoff` and `rgmin_solver_set_objective_roundoff` apply under every acceptance policy. The window used to hold only under `Accept::None`, so a C session, which opens with `RGMIN_ACCEPT_ENERGY`, dropped back to the four-epsilon window in its strong Wolfe search. A rise inside the configured window, such as `2^-24` at a value of 100 with a window of `1e-6`, was refused, and the step stayed at its start after the whole line-search budget. It is accepted again with two evaluations, as in 0.5. Manifold searches keep their own acceptance, and the `Accept::Energy` backtracking of a step that is not line-searched keeps its absolute `1e-8` rise. The C signature is unchanged.
- `cargo clippy --all-targets` with warnings denied passes on Rust 1.88.0 and 1.99.1.
  It passes for each feature set that CI tests: the defaults, `capi`, `highs`, and all features.
  Every fix keeps the solvers' behaviour, and no public signature changes.
- `lowest_eigenpair` returns the Lanczos `Result`.
  An empty seed is `Error::Dim` instead of a panic that claimed Lanczos was not linked.
- `minimize_eon_cg` treats an `EonCg::max_move` that is not a positive finite length as no cap: neither the scalar clip nor the per-atom cap applies, and a step along non-positive curvature has length `EON_MAX_MOVE`. Before, a zero `max_move` left the run at its start, `-inf` turned the first step into NaN, and a negative finite `max_move` flipped the sign of the step at each clip, so a model step shorter than `-max_move` went uphill and a longer one was capped at that length. A positive finite `max_move` takes the same steps as before.
- `rgmin_solver_set_highs_callback` compiles when `capi` and `highs` are both on. It wraps the callback and its `user` pointer in `HighsCallback::new`, and a NULL callback clears it. The C signature is unchanged. `examples/compare_highs.rs` and the `highs` C tests build again.

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
  Holm, Scand. J. Stat. 6, 65 (1979), https://www.jstor.org/stable/4615733.
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

## [0.2.0]

Rust rewrite of the C++ xtsci-optimize: solvers over eindir `DifferentiableObjective`, session C ABI, manifolds, HiGHS-projected steps.
The C++ xtensor history is `0.0.1` on the previous `main`.
