L-BFGS no longer line-searches by default.
The default step is the eOn cap: the longest atom moves at most `EON_MAX_MOVE` (0.2), the initial inverse Hessian is the Nocedal-Wright scale `(s·y)/(y·y)`, pairs are kept when the step is shortened, and a direction that is not descent clears the memory.
`LbfgsStep::Wolfe` and `rgmin_solver_set_lbfgs_wolfe` opt into the strong Wolfe search.
A non-positive cap passed to `rgmin_solver_set_lbfgs_capped` restores 0.2 rather than the line search.
`abi_minor` 34.
