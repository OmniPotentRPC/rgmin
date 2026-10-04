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

/// A periodic cell drops translation and keeps rotation. An isolated
/// cluster still drops both.
#[test]
fn periodic_project_rigid_keeps_rotation() {
    let pos = array![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0];
    let rot = array![0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0];

    let displace = |periodic: bool| {
        let g = rot.clone();
        let obj = HessianOracle::unbounded(
            9,
            move |x: ArrayView1<f64>| (g.dot(&x), g.clone()),
            |_x: ArrayView1<f64>| Array2::<f64>::eye(9),
        );
        let mut x = pos.clone();
        let mut solver = Solver::new(Method::Dogleg, control(4.0), 9);
        solver.set_project_rigid(true);
        solver.set_periodic(periodic);
        let rep = solver.step_hess(&obj, &mut x).expect("dogleg step");
        let disp = (&x - &pos).mapv(|v| v * v).sum().sqrt();
        (disp, rep.grad_norm)
    };

    let (periodic, pnorm) = displace(true);
    assert!(
        periodic > 0.5,
        "periodic project_rigid removed the rotation; displacement {periodic} grad_norm {pnorm}"
    );
    let (isolated, _) = displace(false);
    assert!(
        isolated < 1e-8,
        "isolated project_rigid kept a rigid rotation; displacement {isolated}"
    );

    let trans = array![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let obj_t = HessianOracle::unbounded(
        9,
        move |x: ArrayView1<f64>| (trans.dot(&x), trans.clone()),
        |_x: ArrayView1<f64>| Array2::<f64>::eye(9),
    );
    let mut y = pos.clone();
    let mut solver = Solver::new(Method::Dogleg, control(4.0), 9);
    solver.set_project_rigid(true);
    solver.set_periodic(true);
    let rep = solver.step_hess(&obj_t, &mut y).expect("translation step");
    let drift = (&y - &pos).mapv(|v| v * v).sum().sqrt();
    assert!(
        rep.grad_norm < 1e-8 && drift < 1e-8,
        "translation survived periodic project_rigid, grad {} drift {drift}",
        rep.grad_norm
    );
}
