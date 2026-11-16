On a curved manifold kind, Newton-CG and Newton-PCG retract their trust-region step, so every point the objective sees lies on the set.
Before, they took `x + p`, and on the unit sphere with a 0.1 cap the worst evaluated point was 1.1e-1 off it.
CG now runs on the model `P H P + (I - P)` for the tangent projection `P` of the kind, and flat kinds keep `x + p`.
Where `P` is orthogonal, as on every curved kind but the three multinomial kinds, dogleg solves the same model.
