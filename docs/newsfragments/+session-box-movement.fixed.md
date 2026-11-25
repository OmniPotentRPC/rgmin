A session coordinate box set with `Solver::set_box` (`rgmin_solver_set_box`
in C) now bounds movement accounting too. Projecting a start onto that box
is no longer counted as a step, so an L-BFGS run that never moves past the
projection reports `LineSearchFailed` instead of `MaxIter`, and the
energy-tolerance stop and the L-BFGS plateau check no longer read the
projection as an accepted step.
