//! MDMin through the session: the ASE half-kick, the per-atom scale, and
//! a positive Euclidean cap that always bounds the step, and a negative
//! Euclidean cap that leaves the displacement unscaled.

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};
use rgmin::{Control, Method, Solver};

struct Push;

impl Objective<f64> for Push {
    fn dim(&self) -> usize {
        3
    }
    fn bounds(&self) -> &Bounds<f64> {
        static B: std::sync::OnceLock<Bounds<f64>> = std::sync::OnceLock::new();
        B.get_or_init(|| Bounds::new(array![-20.0, -20.0, -20.0], array![20.0, 20.0, 20.0], 0.0))
    }
    fn eval(&self, _x: ArrayView1<f64>) -> f64 {
        1.0
    }
}

impl Gradient<f64> for Push {
    fn dim(&self) -> usize {
        3
    }
    fn grad(&self, _x: ArrayView1<f64>) -> Array1<f64> {
        array![-100.0, 0.0, 0.0]
    }
}

impl DifferentiableObjective<f64> for Push {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.grad(x))
    }
}

#[test]
fn the_default_cap_uses_the_ase_offset_scale() {
    let control = Control::default()
        .with_maxiter(2)
        .with_gtol(0.0)
        .with_istep(0.2);
    let mut solver = Solver::new(Method::MdMin, control, 3);
    let mut x = array![0.0, 0.0, 0.0];
    solver.step(&Push, &mut x).unwrap();
    let raw = 0.2 * (0.5 * 0.2 * 100.0);
    let scaling = 0.2 / (rgmin::mdmin::SCALE_OFFSET + raw);
    assert!(scaling < 1.0);
    assert!((x[0] - raw * scaling).abs() < 1e-12, "{x:?}");
    assert_eq!(x[1], 0.0);
}

#[test]
fn an_explicit_euclidean_cap_bounds_the_step() {
    let control = Control::default()
        .with_maxiter(2)
        .with_gtol(0.0)
        .with_istep(0.2)
        .with_maxmove(0.05);
    let mut solver = Solver::new(Method::MdMin, control, 3);
    let mut x = array![0.0, 0.0, 0.0];
    solver.step(&Push, &mut x).unwrap();
    let moved = x.dot(&x).sqrt();
    assert!((moved - 0.05).abs() < 1e-12, "{x:?}");
}

#[test]
fn an_explicit_euclidean_cap_binds_below_an_explicit_atom_cap() {
    let control = Control::default()
        .with_maxiter(2)
        .with_gtol(0.0)
        .with_istep(0.2)
        .with_maxmove(0.05);
    let mut solver = Solver::new(Method::MdMin, control, 3);
    solver.set_atom_maxmove(1.0);
    let mut x = array![0.0, 0.0, 0.0];
    solver.step(&Push, &mut x).unwrap();
    let moved = x.dot(&x).sqrt();
    assert!((moved - 0.05).abs() < 1e-12, "{x:?}");
}

#[test]
fn a_negative_euclidean_cap_leaves_the_displacement_unscaled() {
    let control = Control::default()
        .with_maxiter(2)
        .with_gtol(0.0)
        .with_istep(0.2)
        .with_maxmove(-1.0);
    let mut solver = Solver::new(Method::MdMin, control, 3);
    let mut x = array![0.0, 0.0, 0.0];
    solver.step(&Push, &mut x).unwrap();
    let raw = 0.2 * (0.5 * 0.2 * 100.0);
    assert!((x[0] - raw).abs() < 1e-12, "{x:?}");
}
