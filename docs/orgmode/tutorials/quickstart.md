

# Quickstart

This tutorial minimizes the 2-D Rosenbrock banana from the classic start `(-1.2, 1.0)` with the default L-BFGS step, the line-search-free cap of 0.2.


## What you will need

Rust 1.88 or newer, and a new Cargo project with the release from crates.io:

```bash
cargo new rosenbrock
cd rosenbrock
cargo add rgmin@0.8 ndarray@0.17 eindir-core@0.6
```

Each sample below is a whole program. Put it in `src/main.rs` and run it with `cargo run`.
The same programs are `examples/tutorial_quickstart.rs` and `examples/tutorial_session.rs` in the repository, and continuous integration runs both.

To follow the `develop` branch instead, take rgmin from git as shows.
`eindir-core` 0.6 still comes from crates.io, so the samples below compile on either route.


## Rust

A closure is enough.
The stepper does not care who owns the potential.

```rust
//! Minimize the 2-D Rosenbrock banana with the default L-BFGS step.
//!
//! `cargo run --example tutorial_quickstart`

use ndarray::array;
use rgmin::Lbfgs;

fn main() {
    let mut opt = Lbfgs::default();
    let x0 = array![-1.2, 1.0];
    let (_f, x, _evals) = opt.minimize(x0.view(), 200, |x| {
        let a = 1.0 - x[0];
        let b = x[1] - x[0] * x[0];
        Some((
            a * a + 100.0 * b * b,
            array![-2.0 * a - 400.0 * x[0] * b, 200.0 * b],
        ))
    });
    assert!(x[0] > 0.9);
}
```


## Session

Hosts that already own an outer loop hold a `Solver` and call `step` once per iteration.
The host also decides when to stop, here when the gradient norm drops below 1e-4:

```rust
//! Drive a `Solver` one step at a time, and stop when the host says so.
//!
//! `cargo run --example tutorial_session`

use eindir_core::objectives::Rosenbrock;
use ndarray::array;
use rgmin::{Accept, Control, Method, Solver};

fn main() {
    let obj = Rosenbrock::<2>::new();
    let mut x = array![-1.2, 1.0];
    let mut solver = Solver::new(Method::lbfgs(), Control::default(), 2);
    solver.set_accept(Accept::Energy);
    let mut steps = 0;
    loop {
        let rep = solver.step(&obj, &mut x).unwrap();
        steps += 1;
        if rep.grad_norm < 1e-4 || steps == 200 {
            break;
        }
    }
    assert!((x[0] - 1.0).abs() < 1e-3 && (x[1] - 1.0).abs() < 1e-3);
}
```

`Accept::Energy` makes each step refuse a rise in the energy.
The default, `Accept::None`, takes every step without that check, for hosts that have forces but no energy.
The session uses the same line-search-free cap unless the caller sets `LbfgsStep::Wolfe`.

runs that loop to convergence on a 256-atom system.

shows how `set_manifold` keeps each step on a sphere, a rigid-body quotient or another set.


## What just happened

1.  `Lbfgs::minimize` ran the two-loop recursion on the Rosenbrock banana from `(-1.2, 1.0)`.
2.  The closure returned `(f, g)`; the crate never needed a named potential type.
3.  The `Solver` session took one step per call, refused any step that raised the energy, and left the stop test to the caller.
