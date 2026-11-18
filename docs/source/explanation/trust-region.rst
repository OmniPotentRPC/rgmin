=======================================
Trust regions and the Steihaug boundary
=======================================


.. contents::

Line searches commit to a direction and haggle over the length. Trust
regions invert the deal: fix a budget on the length, then ask for the
best step the local model affords inside it. This page derives the
machinery behind ``src/trust.rs`` (dense dogleg) and ``src/hvp.rs``
(matrix-free Steihaug-Toint), and names the exact rational facts the
Lean proofs pin.

1 The model and the ratio
-------------------------

Around the current point the quadratic model is

m(p) = f + g . p + (1/2) p . H p,

trusted only within norm(p) <= Delta. After solving the subproblem
approximately for a step ``p``, the honesty check compares the actual
reduction to the predicted one:

rho = (f - f(x + p)) / (m(0) - m(p)).

The radius update (Nocedal-Wright algorithm 4.1, ``update_radius``)
shrinks by four when rho < 1/4, doubles (capped) when rho > 3/4 and
the step pressed the boundary, and otherwise holds. Three properties
make this loop honest, each proved in
`proofs/lean/Rgmin/Trust.lean <https://github.com/OmniPotentRPC/rgmin/blob/develop/proofs/lean/Rgmin/Trust.lean>`_:

- ``shrink_ge_floor`` / ``shrink_pos``: the radius never leaves the
  positive floor, so no division or comparison downstream degenerates.

- ``shrink_strict`` and the ``collapse`` recursion: consecutive failures
  decay the radius geometrically (one factor of four per rejection),
  so the loop must either accept a step or hit the floor in finitely
  many rejections -- at which point ``minimize_newton_cg`` returns the
  ``TrustCollapsed`` error instead of reporting convergence it does not
  have.

- ``rejection_propagates``: with a positive predicted reduction, a step
  that failed to lower the objective has rho <= 0 < eta and cannot be
  accepted. The acceptance test cannot be talked into an uphill step
  by a large denominator.

2 Steihaug-Toint: conjugate gradients under a leash
---------------------------------------------------

For large ``n`` the dense subproblem solve is the whole bill, so the
matrix-free path never forms ``H``: it runs conjugate gradients on
H p = -g, spending one Hessian action per iteration, and stops early
on any of three events.

1. Small residual: the inexact-Newton forcing sequence
   rtol = min(sqrt(norm(g)), 1/2) keeps early outer iterations cheap and
   the tail superlinear (Nocedal-Wright eq. 7.3).

2. Negative curvature: if d . H d <= 0 the model is unbounded along
   ``d``, and the right amount of greed is to walk the ray to the trust
   boundary. ``Rgmin.ray_further_is_lower`` in
   `Steihaug.lean <https://github.com/OmniPotentRPC/rgmin/blob/develop/proofs/lean/Rgmin/Steihaug.lean>`_
   proves the model is nonincreasing along such a ray, so the boundary
   point minimizes the model on it.

3. Boundary crossing: if the next iterate would leave the region, the
   code solves for the exact crossing.

The crossing length ``tau`` is the positive root of

dd tau\ :sup:`2`\ + 2 zd tau + (zz - r\ :sup:`2`\) = 0,

where zz, zd, dd are the tracked inner products of the current iterate
and direction. With the discriminant named as a square
(s\ :sup:`2`\ = zd\ :sup:`2`\ + dd (r\ :sup:`2`\ - zz)), two exact rational facts close the
construction, both in ``Steihaug.lean``:

- ``boundary_on_sphere``: the root lands exactly on the sphere,
  norm(z + tau d)\ :sup:`2`\ = r\ :sup:`2`\, in whichever inner product the caller
  tracked -- the same algebra serves the Euclidean and preconditioned
  runs.

- ``boundary_tau_nonneg``: inside the region the root is nonnegative;
  a boundary exit extends the step, never retracts it.

The Rust test ``negative_curvature_walks_to_the_trust_boundary`` pins
the implemented step to the sphere at 1e-10 on an indefinite model.

3 The metric bookkeeping under a preconditioner
-----------------------------------------------

``steihaug_pcg`` runs the same loop under a preconditioner ``M``, and the
trust boundary lives in the M-norm. The Conn-Gould-Toint recurrences
track p.Mp, p.Md, d.Md from quantities already on hand (with M z = r,
every needed product reduces to Euclidean dots), so norms cost no
extra applications of ``M``. ``Rgmin.metric_update_scales`` in
`Precond.lean <https://github.com/OmniPotentRPC/rgmin/blob/develop/proofs/lean/Rgmin/Precond.lean>`_ pins the
linearity that makes the tracked comparison scale-consistent, and the
Rust test ``the_preconditioned_boundary_lives_in_the_sketch_metric``
holds the identity-preconditioner run bit-equal to the plain one.

4 Dogleg, for when the Hessian is cheap
---------------------------------------

When a dense Hessian is affordable (``HessianObjective``), the dogleg
path (``dogleg_direction``) interpolates between the Cauchy point (the
model minimizer along -g) and the Newton point, taking whichever the
radius allows. It is the classical picture the Steihaug path
generalizes: CG's first iterate is the Cauchy step, its limit is the
Newton step, and truncation by the boundary lands in between.

5 Newton and dogleg leave a saddle
----------------------------------

A small force is not a positive Hessian, so dense Newton and dogleg
can stop on a first-order saddle. When the force is already under
``gtol``, the report reads the analytic Hessian before it returns.
The floor is max(abs(gtol), 1e-6, 1e-8 times the Frobenius norm of
H). A lowest eigenvalue below minus that floor is a kick: the step
is that mode, scaled so the largest per-atom displacement equals
the per-atom cap (``set_atom_maxmove``, or ``istep`` when the cap is
unset). At most four kicks in one solve. Each kick restarts the
dogleg radius at max(istep, 1e-8), the radius a new dogleg session
starts from. A lowest eigenvalue that is not below that floor still
evaluated the Hessian, so a converged Newton or dogleg report costs
one extra Hessian. ``newton-cg`` and ``newton-pcg`` do not take this
path, and neither does a HiGHS feasible set.

6 Selecting it: ``newton-cg`` and ``newton-pcg``
------------------------------------------------

``Method::NewtonCg`` (token ``newton-cg``) and ``Method::NewtonPcg``
(token ``newton-pcg``) are session methods beside dense Newton and
dogleg. The string parser is ``Method::from_name``; the C waist is
``rgmin_method_from_name``, with tokens ``RGMIN_NEWTON_CG`` and
``RGMIN_NEWTON_PCG``. ``Solver::step`` and ``Solver::step_hess`` on these
methods never call ``hessian``. The product is analytic when the
caller uses ``Solver::step_hvp`` (or ``minimize_method_hvp``), and a
forward difference of the known gradient otherwise: one extra
gradient per product, ``h = 1e-8 / ||v||``. ``Solver::try_step_hvp``
is that step when the host's callbacks return ``Result``. The same
values, gradients and products take the same step. The first error is
``Error::Host``; nothing is evaluated after it, and ``x`` and the
session stay as they were before the step. A chart or distance
constraints on a Newton-CG or Newton-PCG session return
``Error::FallibleUnsupported``. ``try_lowest_eigenpairs`` and
``try_lowest_eigenpairs_precond`` are the same split for one product.
The central-difference
wrapper ``FdHvp`` is unchanged and still costs two gradients; it is
not what the session uses.

``newton-pcg`` rebuilds a rank-16 Nyström preconditioner
(Frangella, Tropp, Udell,
`10.1137/21M1466244 <https://doi.org/10.1137/21M1466244>`_) at the
current point and runs ``steihaug_pcg``. The subproblem boundary is
the preconditioner norm. If that step is longer than the Euclidean
radius, it is pulled back onto the Euclidean sphere before the
displacement cap. The trust ceiling stays Euclidean either way.

``control.maxmove`` is that ceiling, the same rule as dogleg: when
it is unset the ceiling is eight times the current radius, and it
is never below the current radius. A per-atom cap
(``set_atom_maxmove``) clips only the accepted trial. The boundary
test that decides whether to double the radius uses the pre-clip
length, so a cap that shortens the trial does not freeze the
radius. Acceptance is rho > 1/10 (Nocedal-Wright), not dogleg's
rho > 0.

``Solver::gradient_calls`` and ``Solver::hvp_calls`` accumulate for
the life of the session. ``forget`` does not clear them. A
forward-difference product is one ``hvp_calls`` entry, not a
gradient, so a harness charges ``gradient_calls + hvp_calls`` force
evaluations here and ``n`` force evaluations per dense Hessian on
Newton and dogleg.

Steihaug, *The Conjugate Gradient Method and Trust Regions in Large Scale Optimization*, SIAM J. Numer. Anal. 20 (1983),
`10.1137/0720042 <https://doi.org/10.1137/0720042>`_.
Nocedal and Wright, *Numerical Optimization*, chapter 7,
`10.1007/978-0-387-40065-5 <https://doi.org/10.1007/978-0-387-40065-5>`_.
