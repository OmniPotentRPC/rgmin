`Solver::set_objective_roundoff` stores a relative value window
for an explicit Wolfe search. Accept::None does not read it, so
the one-oracle step stays the step it was.
