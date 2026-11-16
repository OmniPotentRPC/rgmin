Opt-in `Solver::set_lbfgs_trust_radius` truncates the session L-BFGS ray at a Burdakov radius (Algorithm 1 of Burdakov, Gong, Zikrin and Yuan 2017).
It is off by default. `Solver::set_lbfgs_trust` remains the eOn Newton-model radius, also off by default.
The capped step stays the default. C: `rgmin_solver_set_lbfgs_trust` and `rgmin_solver_lbfgs_trust_radius`. `abi_minor` 40.
