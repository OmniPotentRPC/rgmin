`rgmin_solver_set_box`: per-coordinate HiGHS box on `x + p`.
A NULL side is unbounded.
Same status as `set_highs` (0, or 1 without the `highs` feature).
`rgmin_solver_set_trust` sets the L_inf radius; `rgmin_solver_add_equality` / `rgmin_solver_clear_equalities` carry `a · p = rhs`.
`HighsStep.lo` / `hi` are per-coordinate vectors (length 1 is uniform).
`abi_minor` 22.
