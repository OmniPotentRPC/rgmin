

# Keep a session on a level set

A level-set constraint keeps every accepted point on the surface `value(x) = target`, where `value` is the energy the objective reports.
The session uses the objective's gradient as the normal of that surface.
It projects each trial direction, the capped eOn L-BFGS displacement included, and the FIRE velocity onto the tangent plane.
Then it retracts each accepted point onto the surface by Newton steps along the gradient.

The start does not have to lie on the surface. The first accepted point is the first one that does.
Retraction stops when `|value - target|` is below a tolerance in the units of the objective, built from the rounding of the value and of one ulp of `x`.
It returns an error when the residual stops contracting before it reaches that tolerance, and when the gradient is zero or not finite.
With no constraint set, the session takes the same steps, bit for bit, as one that never had it.


## Rust

The listing is `examples/howto_level_set.rs`, whole. Continuous integration runs it.

```rust
//! Keep a session on the level set `value(x) = target`.
//!
//! `cargo run --example howto_level_set`

use eindir_core::DifferentiableObjective;
use ndarray::{Array1, ArrayView1, array};
use rgmin::{Control, LevelSetConstraint, Method, Oracle, Solver};

fn radius_squared(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    (x.dot(&x), x.mapv(|v| 2.0 * v))
}

fn main() {
    let obj = Oracle::unbounded(2, radius_squared);
    let target = 1.0;
    let control = Control::default().with_maxiter(3).with_gtol(0.0);
    let mut solver = Solver::new(Method::lbfgs(), control, 2);
    solver
        .set_level_set(LevelSetConstraint::new(target))
        .unwrap();

    // The start is off the set. Each accepted point is back on it.
    let mut x = array![2.0, 0.0];
    for _ in 0..3 {
        solver.step(&obj, &mut x).unwrap();
        let (value, _) = obj.value_and_gradient(x.view());
        assert!((value - target).abs() < 1e-6, "value {value} at {x}");
    }

    solver.clear_level_set();
    assert!(solver.level_set().is_none());
}
```

`Solver::set_level_set` refuses a target that is not finite with `Error::LevelSet`.
A failed retraction at a step is `Error::LevelSetStalled` or `Error::LevelSet`.
`Solver::clear_level_set` drops the constraint, and `Solver::level_set` reads it.


## C

`rgmin_solver_set_level_set(solver, target)` and `rgmin_solver_clear_level_set(solver)` are ABI minor 47.
The setter returns `RGMIN_SUCCESS`, or `RGMIN_INVALID_PARAMETER` for a null session or a target that is not finite.
A retraction that fails during a step also returns `RGMIN_INVALID_PARAMETER`, and `rgmin_last_error()` holds the message.


## Python

The class `rgmin.Session` of the maturin package under `python/` takes `level_set=target` in its constructor, and `Session.set_level_set(target)` sets it later.
`Session.set_level_set(None)` clears it.
The nanobind wheel on PyPI has no level-set option.

```python
import numpy as np
import rgmin

session = rgmin.Session("lbfgs", 2, level_set=1.0)
x = [2.0, 0.0]
x, value, grad_norm = session.step(x, lambda p: (float(np.dot(p, p)), [2.0 * v for v in p]))
assert abs(value - 1.0) < 1e-6
```
