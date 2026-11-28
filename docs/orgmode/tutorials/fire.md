

# FIRE

`Method::fire(FireKind::V2)` is FIRE 2.0. `Solver::set_fire_preset(FirePreset::Guenole2020)` installs the published parameter set. `FireConfig::ase` is the ASE constructor (`dtmax = 1`) rather than the session default `dt_max = 2.5 dt`.

The listing is `examples/tutorial_fire.rs`, whole. Continuous integration runs it, so it is the program as it builds.

```rust
//! FIRE 2.0 with the Guénolé parameters on a quadratic bowl.
//!
//! `cargo run --example tutorial_fire`

use ndarray::{Array1, ArrayView1, array};
use rgmin::{Control, FireKind, FirePreset, Method, Oracle, Solver};

fn main() {
    let control = Control::default()
        .with_maxiter(500)
        .with_gtol(1e-6)
        .with_istep(0.1)
        .with_maxmove(0.2);
    let mut solver = Solver::new(Method::fire(FireKind::V2), control, 2);
    solver.set_fire_preset(FirePreset::Guenole2020).unwrap();
    let obj = Oracle::unbounded(2, |x: ArrayView1<f64>| {
        (x.dot(&x), Array1::from_iter(x.iter().map(|v| 2.0 * v)))
    });
    let mut x = array![0.8, -0.3];
    let report = solver.run(&obj, &mut x).unwrap();
    assert!(report.value < 1e-8, "value {}", report.value);
    assert!(x.iter().all(|v| v.abs() < 1e-4), "{x}");
}
```

The Python class `rgmin.FIRE` passes the ASE arguments (`dt`, `dtmax`, `Nmin`, `finc`, `fdec`, `astart`, `fa`) into `FireConfig`.
