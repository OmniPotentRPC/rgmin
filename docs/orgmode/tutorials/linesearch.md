

# A line search on values you already have

A host that owns the potential cannot hand it to a library: it counts every force call itself, and a second call inside the optimizer would spoil the count.
`SuppliedSearch` turns the search around. The host passes the energy and gradient at the start, then at each trial the search names, and the search says either "evaluate here next" or "accepted". It never calls an objective.

The search constants are the strong Wolfe ones, `LineSearch::wolfe(c1, c2, maxiter)` with `c1 < c2`. The example runs `LineSearch::search` on a closure first, then drives `SuppliedSearch` over the same function, and asserts that the two accept the same step length and the same point. explains the rules.

The listing is `examples/tutorial_linesearch.rs`, whole. Continuous integration runs it, so it is the program as it builds.

```rust
//! Caller-supplied line search. The example is run in CI.
//!
//! `SuppliedSearch` takes the energy and gradient the caller already
//! has. It asks for the next trial and never calls an objective. The
//! accepted point is the one `LineSearch::search` returns.

use ndarray::{Array1, array};
use rgmin::{LineSearch, SearchStep, SuppliedSearch};

fn quad(x: f64) -> (f64, f64) {
    let z = x - 1.0;
    (z * z, 2.0 * z)
}

fn main() {
    let kind = LineSearch::wolfe(1e-4, 0.9, 20);
    let start = array![0.0];
    let dir = array![1.0];
    let (from_library, _, alpha_library) = kind.search(
        |x| {
            let (f, g) = quad(x[0]);
            (f, array![g])
        },
        start.view(),
        dir.view(),
        0.25,
    );
    let mut search = SuppliedSearch::new(kind, start.view(), dir.view(), 0.25);
    let (f0, g0) = quad(0.0);
    let mut step = search.offer(f0, array![g0].view());
    let accepted = loop {
        match step {
            SearchStep::Evaluate { x, .. } => {
                let (f, g) = quad(x[0]);
                step = search.offer(f, array![g].view());
            }
            SearchStep::Accepted { x, alpha, .. } => {
                assert!((alpha - alpha_library).abs() < 1e-12);
                break x;
            }
        }
    };
    assert!((accepted[0] - from_library[0]).abs() < 1e-12);
    let _: Array1<f64> = accepted;
}
```

The same loop is the C entry `rgmin_linesearch` and the Python class `rgmin.LineSearch` of the maturin package.
