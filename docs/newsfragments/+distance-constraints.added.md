Opt-in holonomic distance constraints: `Solver::set_distance_constraints` keeps atom pairs at fixed distances on every `Solver` method except particle swarm, under every `Accept` rule.
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
