=============================================================
Why the line search terminates: brackets, cubics, and a guard
=============================================================


.. contents::

Every solver in this crate that follows a descent direction ``d`` from a
point ``x`` reduces the problem to one dimension. Write the restriction
of the objective along the ray as phi(a) = f(x + a d), so
phi'(0) = g . d is negative for any descent direction. The line
search's job is to pick a step ``a`` that provably makes progress
without spending the caller's evaluation budget, and every claim below
is about where those two guarantees come from.

1 The two Wolfe conditions
--------------------------

Sufficient decrease (Armijo) demands phi(a) <= phi(0) + c1 a phi'(0)
with 0 < c1 < 1: the step must capture at least a ``c1`` fraction of the
decrease the initial slope promised. Because phi'(0) < 0 and a > 0,
the right-hand side sits strictly below phi(0), so an accepted point is
a real improvement, not a rounding story. That one-line consequence is
the Lean theorem ``Rgmin.armijo_strict`` in
`proofs/lean/Rgmin/Zoom.lean <https://github.com/OmniPotentRPC/rgmin/blob/develop/proofs/lean/Rgmin/Zoom.lean>`_.

The curvature condition demands abs(phi'(a)) <= c2 abs(phi'(0)) with
c1 < c2 < 1: the slope must have flattened enough that the quasi-Newton
update built from this step carries curvature information. Nocedal and
Wright's algorithms 3.5 and 3.6 (the bracket and the zoom) find a point
satisfying both, and ``src/linesearch/zoom.rs`` is a direct
implementation.

2 The bracket and the zoom
--------------------------

The outer loop (``wolfe_search``) expands a trial step until it either
satisfies both conditions, overshoots (value above the Armijo line, or
above the previous trial), or the slope turns nonnegative. Overshoot
and slope-reversal both certify that a Wolfe point lies between the
last two trials: that pair becomes the bracket ``[lo, hi]``, ordered so
``lo`` carries the lower value.

The zoom then shrinks the bracket. Each iteration proposes a trial
inside ``[lo, hi]``, evaluates it, and replaces whichever end the Wolfe
tests disqualify. The proposal is where the evaluation budget is won
or lost, and the guard is where termination is proved.

3 The cubic Hermite proposal
----------------------------

At the two ends the zoom holds four numbers: phi(lo), phi'(lo),
phi(hi), phi'(hi). There is exactly one cubic through both points with
both slopes, and its interior minimizer has the closed form (Nocedal
and Wright eq. 3.59)

d1 = phi'(lo) + phi'(hi) - 3 (phi(lo) - phi(hi)) / (lo - hi)
d2 = sign(hi - lo) sqrt(d1\ :sup:`2`\ - phi'(lo) phi'(hi))
t  = hi - (hi - lo) (phi'(hi) + d2 - d1) / (phi'(hi) - phi'(lo) + 2 d2)

On a well-behaved function the cubic model is third-order accurate, so
``t`` lands close to the true minimizer and the zoom converges in a
handful of evaluations; the measured effect in anneal was 46 to 43
force calls per hop across the whole hopping battery, purely from this
proposal replacing bisection. When the discriminant goes negative or a
slope is non-finite the code falls back to the quadratic through
phi(lo), phi'(lo), phi(hi), and below that to bisection: the ladder
degrades in accuracy, never in safety.

4 The guard is the termination proof
------------------------------------

A cubic through awkward data can propose a point arbitrarily close to
either end, and a zoom that accepts such proposals can stall. The code
clamps every proposal into the middle eighty percent of the bracket:
no closer to an end than a tenth of the width. Two exact rational
facts follow, proved in
`Zoom.lean <https://github.com/OmniPotentRPC/rgmin/blob/develop/proofs/lean/Rgmin/Zoom.lean>`_:

- ``Rgmin.guarded_above`` / ``Rgmin.guarded_below``: the clamp really does
  land in the interior band.

- ``Rgmin.zoom_shrinks``: whichever end the trial replaces, the surviving
  bracket is at most ``9/10`` of the old width.

The width therefore decays geometrically (``Rgmin.width_envelope`` is
the bound, ``width_envelope_tendsto`` takes it to zero in the limit),
and the evaluation budget of the zoom is logarithmic in the
requested tolerance regardless of what the cubic does. The Rust test
``the_cubic_zoom_keeps_the_evaluation_budget`` in
``tests/lbfgs_state.rs`` holds the crate to that budget on a quadratic
bowl.

5 What the accept layer adds
----------------------------

Above the line search, ``src/accept.rs`` offers three policies: take the
clipped step (one oracle call), refuse any energy rise (``Accept::Energy``,
up to ten halvings), or the Grippo-Lampariello-Lucidi nonmonotone
window. The fallback steepest-descent probe inside the energy policies
faces the same energy test as the step it replaces: a fallback that
also fails reports the position unmoved, and the caller's stall
machinery owns what happens next. The test
``accept_energy_uphill_does_not_take_ten_steepest_retries`` pins the
oracle count of that path.

6 At the rounding floor
-----------------------

The decrease test compares two values of f.
Near a minimum they differ by less than the rounding error of f, and the test then passes or fails on rounding alone.
On the 75-atom Lennard-Jones cluster of :doc:`../reference/sota`, with the energy near -380, the last ten values that the strict strong Wolfe search tries after its last accepted iterate differ from the value there by up to 1.6e-12, or 4.2e-15 relative.
The slope phi'(a) is still accurate there.
Nocedal and Wright end the search after about ten trials that fail to lower f.
This caps the cost, and x stays at the last point the search accepted (*Numerical Optimization*, 2nd edition, `Springer 2006 <https://doi.org/10.1007/978-0-387-40065-5>`_, section 3.5; :cite:`nocedalNumericalOptimization2006`).
Hager and Zhang instead accept a trial with phi(a) <= phi(0) + eps abs(f) that meets the approximate Wolfe conditions (2 c1 - 1) phi'(0) >= phi'(a) >= c2 phi'(0).
These conditions use only the slope.
They report that a Wolfe search reaches an accuracy near the square root of the machine epsilon, and one with the approximate conditions an accuracy near the machine epsilon itself.
Their paper is *A new conjugate gradient method with guaranteed descent and an efficient line search*, Society for Industrial and Applied Mathematics (SIAM) Journal on Optimization 16, 170 (2005), section 4 (`10.1137/030601880 <https://doi.org/10.1137/030601880>`_; :cite:`hagerNewConjugateGradient2005`).

With a positive ``approx_wolfe``, ``Lbfgs::minimize`` and ``Lbfgs::minimize_box`` use that stop in strong form.
The value error is ``approx_wolfe`` times abs(f), and a trial at most that far above the start is within it.
Such a trial is accepted when abs(phi'(a)) <= c2 abs(phi'(0)).
Since c2 = 0.9 is below 1 - 2 c1, both approximate Wolfe inequalities follow, and the new pair keeps y\ :sup:`T`\ s >= (1 - c2) a abs(phi'(0)) > 0.
A trial within the error that fails the curvature test is placed by the sign of its slope, as in their interval update: the bracket grows while phi'(a) < 0, and the zoom moves its low end to a trial whose slope still points down toward the high end.
On that cluster ``approx_wolfe = 1e-12`` brings rgmin level with SciPy.

The default, ``approx_wolfe = 0``, keeps the strict search, because of what a relaxation leaves for the next call.
On that cluster 394 of the 400 strict runs end with a failed search from steepest descent and no pairs left.
So the next call starts cold, with a first trial step of length at most 1.
With the test on, all 400 end on the ``gtol`` test and keep their pairs.
On the next call the first trial is a full step along the direction they give.
After a large perturbation of a cluster that step can throw the atoms apart.
The search accepts it there, since the energy falls and the slope is near zero.
Call ``Lbfgs::forget`` before reusing a solver after a large change.
``LineSearch::Wolfe``, the search of the ``Solver`` sessions, does not change.
