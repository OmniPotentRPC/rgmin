Exponential preconditioner (Packwood, Kermode, Mones, Bernstein, Woolley, Gould, Ortner, and Csányi, J. Chem. Phys. 144, 164109 (2016), <https://doi.org/10.1063/1.4947024>).
`Solver::set_exp` builds `P` from a caller neighbour list.
L-BFGS uses `H0 = P^{-1}`.
FIRE and nonlinear CG use `P` as the metric.
A dense pair or Lindh matrix passed into the two-loop still wins.
C: `rgmin_exp_t`, `rgmin_solver_set_exp`, `rgmin_solver_clear_exp`, `rgmin_solver_exp_mu`.
The C++ session wraps those three as `set_exp`, `clear_exp`, and `exp_mu`.
`abi_minor` 28.
