//! FIRE schedule selected by `Solver::set_fire_variant`.
//!
//! A zero gradient stops when the norm is under `Control::gtol`.
//! The call does not install a displacement cap.

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};
use rgmin::{Control, FireKind, FireVariant, Method, Solver};

struct Sphere;

impl Objective<f64> for Sphere {
    fn dim(&self) -> usize {
        2
    }
    fn bounds(&self) -> &Bounds<f64> {
        use std::sync::OnceLock;
        static B: OnceLock<Bounds<f64>> = OnceLock::new();
        B.get_or_init(|| Bounds::new(array![-1e6, -1e6], array![1e6, 1e6], 0.0))
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        x[0] * x[0] + x[1] * x[1]
    }
}

impl Gradient<f64> for Sphere {
    fn dim(&self) -> usize {
        2
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        array![2.0 * x[0], 2.0 * x[1]]
    }
}

impl DifferentiableObjective<f64> for Sphere {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.grad(x))
    }
}

fn control() -> Control {
    Control {
        maxiter: 200,
        gtol: 1e-8,
        istep: 0.2,
        maxmove: None,
        ..Control::default()
    }
}

#[test]
fn guenole_from_either_integrator_reaches_the_origin() {
    for kind in [FireKind::V1, FireKind::V2] {
        let mut x = array![1.5, -2.0];
        let mut solver = Solver::new(Method::Fire { kind }, control(), 2);
        solver.set_fire_variant(FireVariant::Guenole2020);
        let mut last = solver.step(&Sphere, &mut x).unwrap();
        for _ in 1..200 {
            if last.grad_norm < 1e-7 {
                break;
            }
            last = solver.step(&Sphere, &mut x).unwrap();
        }
        assert!(
            last.grad_norm < 1e-5 && last.value < 1e-10,
            "{kind:?} gnorm {} value {}",
            last.grad_norm,
            last.value
        );
    }
}

#[test]
fn rgmin_variant_keeps_the_built_integrator() {
    let mut x = array![1.5, -2.0];
    let mut solver = Solver::new(Method::Fire { kind: FireKind::V1 }, control(), 2);
    solver.set_fire_variant(FireVariant::Rgmin);
    let rep = solver.step(&Sphere, &mut x).unwrap();
    assert!(rep.grad_norm.is_finite());
    assert!(x[0] < 1.5);
}

#[test]
fn a_non_fire_session_ignores_the_variant() {
    let mut plain = Solver::new(Method::lbfgs(), control(), 2);
    let mut tagged = Solver::new(Method::lbfgs(), control(), 2);
    tagged.set_fire_variant(FireVariant::Guenole2020);
    let mut a = array![1.5, -2.0];
    let mut b = a.clone();
    let ra = plain.step(&Sphere, &mut a).unwrap();
    let rb = tagged.step(&Sphere, &mut b).unwrap();
    assert_eq!(a, b);
    assert_eq!(ra.grad_norm.to_bits(), rb.grad_norm.to_bits());
}
