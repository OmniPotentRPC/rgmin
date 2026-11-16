A non-finite value or gradient is a refused answer in every line search and acceptance test.
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
