`LbfgsStep::Capped` is the atomistic L-BFGS step, and it is the default.
It drops the line search.
The two-loop step is shortened so the longest per-atom displacement is at most the caller's cap, and the initial inverse Hessian stays the Nocedal-Wright scale `(s·y)/(y·y)`.
An empty memory steps along the force at that length.
Pairs are kept when the step is shortened.
`Lbfgs::capped` builds it, and `Solver::set_lbfgs_step` installs the rule on a Euclidean L-BFGS session.
`set_atom_maxmove` selects the geometry step only after `LbfgsStep::Wolfe`.
The C header declares `rgmin_solver_set_lbfgs_capped`, the C++ wrapper has `set_lbfgs_capped`, and `xts.h` aliases it.
`abi_minor` 33.
`cargo run --release --example lbfgs_capped` prints the force calls and the median wall time.
`reference/sota` records that run.
On LJ7 and LJ13 at atomic force 0.01 the geometry step uses fewer evaluations at the same energy, and on the extended Rosenbrock function with four or more coordinates the value near 3.98 is a local minimum in the basin of `(-1, 1, ..., 1)`.
