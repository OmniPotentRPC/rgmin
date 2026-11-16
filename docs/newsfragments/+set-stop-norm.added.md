Opt-in session stop norm.
`Solver::set_stop_norm` takes a `StopNorm`: `Two` (the default, `||g||_2`), `MaxAtom` (the largest norm over `(x, y, z)` triples: ASE `fmax`, eOn `max_atom`, LAMMPS `max`), or `Inf` (the largest absolute component: eOn `max_component`, LAMMPS `inf`).
Every session stop test and `Report::grad_norm` use it.
The default path computes the same numbers as before.
C: `rgmin_stop_norm_t`, `rgmin_solver_set_stop_norm` and `rgmin_solver_report_gradient`.
The setter returns 1 for an unknown integer.
The report call copies the gradient behind the last step report.
The C++ session wraps them as `set_stop_norm` and `report_gradient`.
xts aliases follow.
`abi_minor` 29.
