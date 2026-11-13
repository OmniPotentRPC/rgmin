`Lbfgs::minimize_box` honors `ftol`.
A gradient or a box bound of the wrong length returns `+inf` instead of panicking.
`lowest_eigenpair` returns `Result`.
`rgmin_last_error_message`, `rgmin_solver_dim`, and `rgmin_solver_steps` are on the C ABI at `abi_minor` 36.
The Python package exposes `Lbfgs` and an ASE `LBFGS`.
