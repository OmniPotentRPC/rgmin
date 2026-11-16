On a curved manifold kind, a `Solver` step caps the tangent step and then retracts it, so every point it evaluates lies on the set.
Before, the acceptance trial retracted, capped the chord and clipped, and then scored that point off the set.
On the unit sphere with a 0.1 cap, steepest descent evaluated points 3.4e-2 off it, FIRE 6.2e-2 and FIRE 2.0 6.0e-3.
A retraction that moves farther than the cap shrinks the step and retracts again, so a step on the positive orthant keeps the cap.
A trial that the box clip would move is refused before the oracle sees it.
FIRE and the FIRE 2.0 rollback use the same order.
Dogleg on a curved kind steps in the tangent space, on the Hessian in an orthonormal basis of the tangent space, and its ratio compares the energy at the retracted point with that model.
Before, it accepted `x + p` with no retraction, 0.113 off the sphere for `f = -x_3 + |x|^2 / 2`.
On a curved kind the session no longer retracts and evaluates again after the step, except after the L-BFGS line search.
Steepest descent and FIRE on the sphere take one oracle call per step instead of two.
`Error::OffManifold` reports a FIRE step with no trial inside the box and the cap, and the C ABI returns it as `RGMIN_INVALID_PARAMETER`.
Flat kinds keep the old order and the old numbers.
