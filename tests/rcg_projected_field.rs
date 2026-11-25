//! Riemannian CG reads the field the session reports in.
//!
//! The session stops and reports on the horizontal gradient: the rigid
//! blocks of a product projected with the session masses and periodic
//! flag, and the declared rigid motion removed at the point. The Wolfe
//! slopes, the conjugacy coefficient and the transport of the step must
//! be formed in that same field. Otherwise a run descends in one field,
//! stops on another, and stalls with a reported gradient that no step
//! reduces.

use ndarray::{Array1, ArrayView1, array, s};
use rgmin::manifold::{Manifold, ProductFactor};
use rgmin::{Control, ManifoldKind, Method, Oracle, RcgOptions, Solver, Status, minimize_rcg};

/// Translation removed per axis and rotation kept, one molecule per block
/// of the given length: `R^{3N}/T(3)` for a single block.
struct Translations(usize);

impl Manifold for Translations {
    fn required_dim(&self, n: usize) -> Result<(), usize> {
        if n >= 3 && n.is_multiple_of(3) {
            Ok(())
        } else {
            Err(n)
        }
    }

    fn project(&self, _x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        let atoms = self.0 / 3;
        let mut w = v.clone();
        for block in 0..v.len() / self.0 {
            let at = block * self.0;
            for axis in 0..3 {
                let mean = (0..atoms).map(|a| v[at + 3 * a + axis]).sum::<f64>() / atoms as f64;
                for a in 0..atoms {
                    w[at + 3 * a + axis] -= mean;
                }
            }
        }
        w
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        x + v
    }

    fn transport(&self, _from: &Array1<f64>, to: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        self.project(to, v)
    }
}

fn triangle() -> Array1<f64> {
    array![0.0, 0.0, 0.0, 1.3, 0.0, 0.1, 0.2, 0.9, -0.1]
}

/// The triangle turned by 0.7 rad about z and shifted. A restraint to it
/// is not rotation invariant, so its horizontal gradient keeps a rotation
/// part wherever the periodic session projects translations only.
fn turned_target() -> Array1<f64> {
    let (sin, cos) = 0.7f64.sin_cos();
    let start = triangle();
    let mut t = Array1::zeros(9);
    for a in 0..3 {
        let (x, y, z) = (start[3 * a], start[3 * a + 1], start[3 * a + 2]);
        t[3 * a] = cos * x - sin * y + 0.4;
        t[3 * a + 1] = sin * x + cos * y - 0.2;
        t[3 * a + 2] = z + 0.1;
    }
    t
}

fn restraint(target: Array1<f64>) -> impl Fn(ArrayView1<f64>) -> (f64, Array1<f64>) {
    move |x| {
        let g = &x - &target;
        (0.5 * g.dot(&g), g)
    }
}

#[test]
fn a_periodic_quotient_session_steps_in_its_reported_field() {
    let target = turned_target();
    let obj = Oracle::unbounded(9, restraint(target.clone()));
    let field = Translations(9);
    let mut solver = Solver::new(Method::rcg(), Control::default().with_gtol(1e-14), 9);
    solver.set_manifold(ManifoldKind::RigidQuotient);
    solver.set_periodic(true);
    let mut x = triangle();
    for k in 1..=6 {
        solver.step(&obj, &mut x).unwrap();
        let reference = minimize_rcg(
            &field,
            triangle(),
            restraint(target.clone()),
            &RcgOptions::default().with_max_iter(k).with_grad_tol(0.0),
        )
        .unwrap();
        let gap = (&x - &reference.x)
            .iter()
            .fold(0.0f64, |m, d| m.max(d.abs()));
        assert!(
            gap < 1e-11,
            "iteration {k}: the session left the field by {gap}"
        );
    }
}

#[test]
fn a_periodic_quotient_session_converges_on_its_reported_gradient() {
    let obj = Oracle::unbounded(9, restraint(turned_target()));
    let mut solver = Solver::new(
        Method::rcg(),
        Control::default().with_maxiter(200).with_gtol(1e-6),
        9,
    );
    solver.set_manifold(ManifoldKind::RigidQuotient);
    solver.set_periodic(true);
    let mut x = triangle();
    let report = solver.run(&obj, &mut x).unwrap();
    assert_eq!(report.status, Status::Tolerance, "{report:?}");
    assert!(report.grad_norm <= 1e-6, "grad {}", report.grad_norm);
    // Only a translation is left between the point and the target.
    let gap = &x - &turned_target();
    for axis in 0..3 {
        let first = gap[axis];
        for a in 1..3 {
            assert!(
                (gap[3 * a + axis] - first).abs() < 1e-5,
                "axis {axis}: {gap}"
            );
        }
    }
}

#[test]
fn a_periodic_product_of_rigid_blocks_converges_on_its_reported_gradient() {
    // Two molecules, the second a shifted copy of the first, each
    // restrained to a turned copy of its own geometry.
    let mut target = Array1::zeros(18);
    target.slice_mut(s![..9]).assign(&turned_target());
    target.slice_mut(s![9..]).assign(&(turned_target() + 3.0));
    let mut start = Array1::zeros(18);
    start.slice_mut(s![..9]).assign(&triangle());
    start.slice_mut(s![9..]).assign(&(triangle() + 3.0));
    let obj = Oracle::unbounded(18, restraint(target.clone()));
    let mut solver = Solver::new(
        Method::rcg(),
        Control::default().with_maxiter(300).with_gtol(1e-6),
        18,
    );
    solver
        .set_product([
            ProductFactor::new(ManifoldKind::RigidQuotient, 9),
            ProductFactor::new(ManifoldKind::RigidQuotient, 9),
        ])
        .unwrap();
    solver.set_periodic(true);
    let mut x = start.clone();
    let report = solver.run(&obj, &mut x).unwrap();
    assert_eq!(report.status, Status::Tolerance, "{report:?}");
    assert!(report.grad_norm <= 1e-6, "grad {}", report.grad_norm);

    // The slopes and the conjugacy coefficient of every step are those of
    // the same run on the translation-reduced field, block by block.
    let mut stepper = Solver::new(Method::rcg(), Control::default().with_gtol(1e-14), 18);
    stepper
        .set_product([
            ProductFactor::new(ManifoldKind::RigidQuotient, 9),
            ProductFactor::new(ManifoldKind::RigidQuotient, 9),
        ])
        .unwrap();
    stepper.set_periodic(true);
    let mut y = start.clone();
    for k in 1..=6 {
        stepper.step(&obj, &mut y).unwrap();
        let reference = minimize_rcg(
            &Translations(9),
            start.clone(),
            restraint(target.clone()),
            &RcgOptions::default().with_max_iter(k).with_grad_tol(0.0),
        )
        .unwrap();
        let gap = (&y - &reference.x)
            .iter()
            .fold(0.0f64, |m, d| m.max(d.abs()));
        assert!(
            gap < 1e-11,
            "iteration {k}: the product session left the field by {gap}"
        );
    }
}
