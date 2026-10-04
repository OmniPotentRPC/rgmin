//! Session gaps in retraction, rigid projection, masses, and curvature pairs.

use ndarray::{Array2, ArrayView1, array};
use rgmin::{Control, HessianOracle, ManifoldKind, Method, Solver};

fn control(istep: f64) -> Control {
    Control {
        maxiter: 8,
        gtol: 1e-14,
        istep,
        maxmove: None,
        ftol_rel: None,
    }
}

/// Powell dogleg on the sphere lands on the retraction, not on `x + dir`.
#[test]
fn dogleg_on_the_sphere_retracts() {
    let obj = HessianOracle::unbounded(
        3,
        |x: ArrayView1<f64>| {
            let g = array![x[0] - 1.0, x[1], x[2]];
            let f = 0.5 * g.dot(&g);
            (f, g)
        },
        |_x: ArrayView1<f64>| Array2::<f64>::eye(3),
    );
    let mut x = array![0.0, 1.0, 0.0];
    let mut solver = Solver::new(Method::Dogleg, control(4.0), 3);
    solver.set_manifold(ManifoldKind::Sphere);
    let rep = solver.step_hess(&obj, &mut x).expect("dogleg step");
    let norm = x.dot(&x).sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-10,
        "dogleg left the sphere at {x:?} norm {norm} value {}",
        rep.value
    );
    assert!(
        (x[1] - 1.0).abs() > 1e-8,
        "dogleg accepted a zero step {x:?}"
    );
}
