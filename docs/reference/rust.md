# Rust

`Lbfgs`, `Solver`, and `minimize_eon_cg` are the entry points.
Errors are `Result` with `thiserror`. `lowest_eigenpair` returns
`Result` and reports an empty seed as `Error::Dim`.

The quadratic `x^2 + y^2` from `(1, -2)` ends below `1e-8`.
`examples/quadratic.rs` asserts that.
