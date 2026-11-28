

# One L-BFGS direction with your own H0

`LbfgsPairs` is the two-loop recursion with the pair memory handed to the caller. It returns the direction `-H g` for a gradient you supply and does not evaluate an objective, so a host that has its own step rule, line search or preconditioner can use the L-BFGS metric alone.

The starting matrix `H0` is the argument of `LbfgsPairs::direction`, and it takes three forms of `LbfgsH0`: `Scale(gamma)` is `gamma I`, `Diagonal(d)` is one entry per coordinate, and `Apply(f)` is a callback `v -> H0 v` that runs once, on the vector in the middle of the two loops.

The example first checks the empty memory, where the direction is `-gamma g`. It then pushes one pair, compares the direction with the one `Lbfgs::two_loop` gives for the same pair, and finishes by checking that a diagonal and the callback that applies it agree.

The listing is `examples/tutorial_lbfgs_direction.rs`, whole. Continuous integration runs it, so it is the program as it builds.

```rust
//! Caller-supplied `H0` for one L-BFGS direction. The example is run in CI.
//!
//! The direction is `-H g`. Nothing here evaluates an objective.

use ndarray::array;
use rgmin::{Lbfgs, LbfgsH0, LbfgsPairs};

fn main() {
    let g = array![1.0, -2.0, 0.5];
    let mut pairs = LbfgsPairs::new(g.len(), 4).unwrap();
    let empty = pairs.direction(g.view(), LbfgsH0::Scale(0.25)).unwrap();
    assert_eq!(empty, array![-0.25, 0.5, -0.125]);

    let s = array![0.2, -0.1, 0.05];
    let y = array![0.4, 0.3, 0.2];
    assert!(pairs.push(s.view(), y.view()).unwrap());
    let mut opt = Lbfgs::with_capacity(4);
    opt.record(s.clone(), y.clone());
    let gamma = s.dot(&y) / y.dot(&y);
    let from_handle = pairs.direction(g.view(), LbfgsH0::Scale(gamma)).unwrap();
    assert_eq!(from_handle, opt.two_loop(g.view()));

    let diag = array![0.5, 0.25, 2.0];
    let from_diag = pairs
        .direction(g.view(), LbfgsH0::Diagonal(diag.view()))
        .unwrap();
    let from_callback = pairs
        .direction(g.view(), LbfgsH0::Apply(&|v| Ok(&diag * &v)))
        .unwrap();
    assert_eq!(from_callback, from_diag);
    assert_eq!(from_diag.len(), g.len());
}
```

`LbfgsPairs::push` returns `Ok(true)` when it stored the pair. A pair that fails the relative-curvature test is dropped and the call returns `Ok(false)`, which is not an error.
