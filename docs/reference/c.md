# C

The header is `include/rgmin.h`. Handles are opaque.
`rgmin_last_error_message` borrows the thread-local detail.
`rgmin_solver_dim` and `rgmin_solver_steps` are const accessors.

`docs/examples/quadratic.c` checks that `x^2 + y^2` from `(1, -2)`
ends below `1e-8`.
