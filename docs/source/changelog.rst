

Changelog
---------

Unreleased

~~~~~~~~~~

A backtracking or Goldstein line search that runs out of trials now reports the step that produced the point it returns.
An L-BFGS session step that drops its own curvature pair now keeps the pair before it.

A default ``meson install`` of rgmin as the top-level project installs ``librgmin.so`` into ``libdir``.
``-Dwith_asv_helper=true`` also builds and installs ``librgmin_asv``, the timing helper that the asv suite loads.

``benchmarks/minimize.py`` is an airspeed velocity (asv) suite that times one ``rgmin_minimize`` call on a convex quadratic in 16 and 64 dimensions.
``snakemake -c 1`` calibrates an upper bound for each benchmark from repeated runs of ``develop``.
``snakemake -c 1 pr_gate`` exits non-zero when a benchmark is slower than that bound, and ``benchflow/README.md`` describes the workflow.

Under ``Accept::None``, an L-BFGS trial on a curved kind that the box refuses returns ``Error::OffManifold`` instead of ``Error::Oracle``.

On a curved manifold kind, a ``fire2`` session projects its secant velocity onto the tangent space at the new point.

On a curved manifold kind, Newton-CG and Newton-PCG retract their trust-region step, so every point the objective sees lies on the set.
CG runs on the model ``P H P + (I - P)`` for the tangent projection ``P`` of the kind, and flat kinds keep ``x + p``.

On a curved manifold kind, a ``Solver`` step caps the tangent step and then retracts it, so every point it evaluates lies on the set.
On the unit sphere with a 0.1 cap, steepest descent, FIRE, FIRE 2.0 and dogleg evaluated points up to 1.1e-1 off it before.
Dogleg steps in the tangent space, and steepest descent and FIRE take one oracle call per step instead of two.
A trial that the box clip would move is refused before the oracle sees it, and a FIRE step with no trial left returns ``Error::OffManifold``.
Flat kinds keep the old order and the old numbers.

``Solver::set_fire_preset`` selects the FIRE 2.0 parameters of Guenole et al. 2020 Table 1 on a ``fire2`` session.
``Solver::set_fire_abc`` turns on the accelerated bias-corrected (ABC) FIRE update of Echeverri Restrepo and Andric 2023, which ASE runs as ``FIRE2(use_abc=True)``.
Both are opt-in, and :doc:`howto/fire` shows them.
The C header ``include/rgmin.h`` declares ``rgmin_solver_set_fire_preset`` and ``rgmin_solver_set_fire_abc``.
``abi_minor`` 30.

``cargo cinstall`` installs the hand-written ``include/rgmin.h`` and ``dlpack/dlpack.h``.
The header compiles as C and as C++.
Version macros are ``RGMIN_H_MAJOR``, ``RGMIN_H_MINOR``, and ``RGMIN_H_PATCH``.
``HighsInt`` stays out of the public header.

How-to samples that did not compile now do.
The manifold sample defines its objective.
The HiGHS sample fills the rest of ``HighsStep`` from ``Default``.
The intrinsic reaction coordinate (IRC) sample imports only what it uses.
The C++ sample calls ``rgmin::minimize_fn`` from ``rgmin/optimize.hpp``.
The C samples include ``rgmin.h``.

:doc:`Install rgmin <howto/install>` shows how to install the 0.5.2 release from crates.io or the ``develop`` branch.
It gives the dependencies and features of each route, the C library through cargo-c, and the SciPy the HiGHS tests need.
The README, the index and the quickstart point to it.

The README opens with what rgmin does.
It spells out each method name on first use, along with IRC and the C application binary interface (ABI) terms.

The index and :doc:`reference/architecture` define the hourglass and its waist, the C ABI that C and C++ callers go through.
``docs/figures/architecture/draw_hourglass.py`` draws the layer figure from the headers and ``src/ffi.rs``.
The code of conduct, the security policy and the index name a contact, and the index links the issue tracker.

The tutorial :doc:`Relax a platinum crystal <tutorials/relax-morse>` relaxes the Pt256 Morse fixture through a ``Solver`` session with limited-memory Broyden-Fletcher-Goldfarb-Shanno (BFGS), the fast inertial relaxation engine (FIRE) and FIRE 2.0.
``examples/relax_morse.rs`` is the program it runs.
``docs/figures/relax-morse/`` has the force-trace plot, the session diagram and the scripts that draw them.

``CONTRIBUTING.md`` names the issue tracker, the ``develop`` branch and the checks that continuous integration (CI) runs.
It points to the contributing page for the rest.

:doc:`Contributing <contributing/index>` now gives the docs export command that the workflow runs, ``cd docs && emacs --batch -l export.el``.
Of the two commands it gave before, one failed and the other wrote nothing.

L-BFGS session ``Accept::None`` takes the clipped two-loop direction.
Wolfe line search stays on Energy / Nonmonotone.

The exponential preconditioner of Packwood et al., J. Chem. Phys. 144, 164109 (2016) (`10.1063/1.4947024 <https://doi.org/10.1063/1.4947024>`_) is ``Solver::set_exp``.
The caller supplies the neighbour list.
L-BFGS uses ``H0 = P^{-1}``.
FIRE and nonlinear CG use ``P`` as the metric.
A dense pair or Lindh matrix passed into the two-loop still wins.
C: ``rgmin_exp_t``, ``rgmin_solver_set_exp``, ``rgmin_solver_clear_exp``, ``rgmin_solver_exp_mu``.
The C++ session wraps those three as ``set_exp``, ``clear_exp``, and ``exp_mu``.
``abi_minor`` 28.

With ``rgmin_solver_set_box``, the caller supplies a per-coordinate HiGHS box on ``x + p``.
A NULL side is unbounded.
Same status as ``set_highs`` (0, or 1 without the ``highs`` feature).
``rgmin_solver_set_trust`` is the L\ :sub:`inf`\ radius; ``rgmin_solver_add_equality`` / ``rgmin_solver_clear_equalities`` carry ``a · p = rhs``.
``HighsStep.lo`` / ``hi`` are per-coordinate vectors (length 1 is uniform).
``abi_minor`` 22.

``xts.h`` declares the exported ``rgmin_minimize`` / ``rgmin_solver_*`` symbols; ``xts_minimize`` and ``xts_solver_*`` are macro aliases.

Centered matrices (``centeredmatrix`` / ``RGMIN_MANIFOLD_CENTERED_MATRIX`` = 22): manopt ``centeredmatrixfactory``.
Packed row-major ``m n``.
Projection is the centering operator (``X 1 = 0`` by default, or ``1^T X = 0``), retraction is ``X + U`` then center, transport is the identity.
Not the sphere and not a 3N cluster.
Tokens 7-10 stay reserved.
``abi_minor`` 21.

Positive orthant (``positive`` / ``RGMIN_MANIFOLD_POSITIVE`` = 21): manopt ``positivefactory``.
Packed length ``n``.
Projection is the identity, retraction is ``x ⊙ exp(v ⊘ x)``, transport is the identity.
Not the sphere and not a 3N cluster.
Tokens 7-10 stay reserved.
``abi_minor`` 20.

Complex unit sphere (``spherecomplex`` / ``RGMIN_MANIFOLD_SPHERE_COMPLEX`` = 20): manopt ``spherecomplexfactory``.
Interleaved ``(re, im)``, length ``2 n``.
Projection is ``v - Re(x^* v) x``, retraction is ``(x+v)/||x+v||``, transport is projection at arrival.
Not the real sphere and not ``(S^1)^n``.
Tokens 7-10 stay reserved.
``abi_minor`` 19.

Symmetric doubly-stochastic matrices (``multinomialsymmetric`` / ``RGMIN_MANIFOLD_MULTINOMIAL_SYM`` = 19) and doubly-stochastic matrices (``multinomialdoublystochastic`` / ``RGMIN_MANIFOLD_MULTINOMIAL_DS`` = 18): manopt ``multinomialsymmetricfactory`` and ``multinomialdoublystochasticfactory``.
Both are packed row-major ``n^2``, ``n >= 2``, with Fisher projection and transport at arrival.
The retraction is Sinkhorn, and on the symmetric manifold it then symmetrizes.
A 3N cluster is rejected.
Tokens 7-10 stay reserved.
``abi_minor`` 18 and 17.

Symmetric matrices (``symmetric`` / ``RGMIN_MANIFOLD_SYMMETRIC`` = 14): manopt ``symmetricfactory``.
Packed row-major ``n^2``.
Projection is symmetrization, retraction is ``X + U``, transport is the identity.
A 3N cluster is rejected.
The ABI layout revision is 4.

Scaled conjugate gradient (SCG) C waist: closed ``xts_conjugacy_t`` on ``xts_scg_params_t`` (FR=0 .. FrPr=7, dest leaf order).
``params==NULL`` is Netlab Polak-Ribiere.
Unknown codes are ``RGMIN_INVALID_PARAMETER``.
The ABI version is 1.13.3, with layout revision 3.
Hybrid and Restart stay Rust-only.

Typed lowest-mode waist: closed ``EigensolverKind`` (Lanczos=0 through EigenExa=13, ``schema/eigen.capnp``) and ``EigenParams`` (``nev``, ``krylov``, ``max_iter``, ``tol``).
No Text.
C ``rgmin_lowest_eigenpair`` plus ``rgmin_eigen_params_t``.
Unlinked backends return ``RGMIN_UNAVAILABLE``.

The gpr_optim consumer that writes ``RGMIN_CONJUGACY_LIU_STOREY`` is ``RgminScg.inl``, not in-tree ``SCG.inl``.

Molecular manifolds: ``rigid_quotient`` is Sella Cartesian ``R^{3N}/SE(3)`` (``fix_translation`` + ``fix_rotation``); ``mw_rigid`` is the Page-McIver / Sella IRC Eckart metric on the same quotient (``rgmin_solver_set_masses``).
SO(3) is length 9 and SE(3) is length 12; a 3N cluster is rejected rather than packed or prefix-interpreted.
Periodic bulk sets ``rgmin_solver_set_periodic``: Sella ``proj_rot = false``, quotient ``R^{3N}/T(3)``.
Isolated stays ``R^{3N}/SE(3)``.
``abi_minor`` 10.

Beside the molecular manifolds, the embedded manopt tokens are Euclidean (default), sphere, SO(3), Stiefel ``St(n,1)`` and ``St(n,p)`` (``rgmin_solver_set_stiefel``), SE(3), Oblique token 11, Multinomial token 12, complex circle ``(S^1)^n``.
Reserved: 7 symmetric positive definite (SPD), 8 Grassmann, 9 Hyperbolic, 10 Poincare.
Hourglass mark and the Diataxis book under ``docs/orgmode/``.

0.2.0
~~~~~

``rgmin_solver_set_highs``: opt-in HiGHS feasible-set step.
With a host Hessian this is the convex Newton QP ``min 1/2 p^T P p + g^T p`` and per-coordinate boxes.
Without the ``highs`` feature the setter returns 1.
eindir / rgpot / eOn share the same session.

0.1.0
~~~~~

Rust rewrite of the C++ xtensor tree (``0.0.1`` on the previous ``main``).
Solvers live in Rust.
C and C++ reach them through the ``rgmin_solver_t`` session API and ``rgmin_minimize`` over dlpk tensors.

- Session API: ``create`` / ``step`` / ``step_hess`` / ``step_fg`` / fused ``evalgrad`` / ``forget`` / ``free``.

- Methods: L-BFGS, BFGS, SR1, SR2, shifted Newton, rational function optimization (RFO), eight nonlinear conjugate gradient (NLCG) conjugacies, steepest, Adam, particle swarm optimization (PSO), FIRE, FIRE 2.0, Barzilai-Borwein, Powell dogleg.

- Host Hessian policy: ``qn_step`` two-loop / Newton / RFO; pair ``H0 = P^{-1}``.

- Accept: none / energy / nonmonotone.

- Per-atom maxmove and optional rigid-body projection.

- Optional ``highs`` feature: two-loop direction, HiGHS only for a feasible set (box, trust, equalities).

The previous C++ ``main`` is unrelated history.
This tag is the first Rust release.
