

# Conjugate gradients

Two steps share the Polak-Ribière coefficient. `Method::eon_cg` is that coefficient with the Powell restart at 0.5, and the session still line-searches. `minimize_eon_cg` is the client step: one curvature probe, no line search, then the per-atom cap.

The listing is `examples/tutorial_cg.rs`, whole. Continuous integration runs it, so it is the program as it builds.

```rust
//! Line-search-free conjugate gradients, the eOn client step.
//!
//! `cargo run --example tutorial_cg`

use rgmin::{EonCg, minimize_eon_cg};

fn main() {
    let mut x = vec![1.0, -2.0, 0.5];
    let opts = EonCg::default().with_gtol(1e-8).with_max_iter(200);
    let report = minimize_eon_cg(
        &mut x,
        |xs, gs| {
            let mut f = 0.0;
            for (x, g) in xs.iter().zip(gs.iter_mut()) {
                f += x * x;
                *g = 2.0 * x;
            }
            f
        },
        &opts,
    );
    assert!(report.converged, "grad {}", report.grad);
    assert!(report.energy < 1e-12, "energy {}", report.energy);
    assert!(x.iter().all(|v| v.abs() < 1e-5), "{x:?}");
}
```

`Method::polak_ribiere_plus` is SciPy's `method="CG"`: `max(0, β)`. The Python session named `cg` uses that coefficient with a strong Wolfe search, because SciPy does. The eOn client step does not.
