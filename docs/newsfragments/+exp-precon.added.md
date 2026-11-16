C entry points for the exponential preconditioner: `rgmin_exp_t`, `rgmin_solver_set_exp`, `rgmin_solver_clear_exp`, `rgmin_solver_exp_mu`.
The C++ session wraps those three as `set_exp`, `clear_exp`, and `exp_mu`.
A dense pair or Lindh matrix passed into the two-loop still wins over the exponential preconditioner.
`abi_minor` 28.
