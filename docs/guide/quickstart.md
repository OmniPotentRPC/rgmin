# Rust

`Lbfgs::minimize` takes the start, an iteration cap, and a closure
that returns the value and the gradient. The quadratic `x^2 + y^2`
from `(1, -2)` ends below `1e-8`.

```{literalinclude} ../../examples/quadratic.rs
:language: rust
```

`cargo run --example quadratic` runs that file.
A short gradient, or a box bound of the wrong length, returns `+inf`
and does not panic.

`Lbfgs::ftol` is the relative decrease
`f_k - f_{k+1} <= ftol * max(|f_k|, |f_{k+1}|, 1)`.
It applies to `minimize` and to `minimize_box`. Zero, the default,
leaves it off.

Rosenbrock from `(-1.2, 1.0)` with `LbfgsStep::Wolfe` ends at `(1, 1)`.
