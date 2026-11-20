//! Opt-in inverse-Hessian scale and line-search multiplier bound.
//! Unset, both leave the session step that shipped before they existed.

use std::sync::OnceLock;

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};
use rgmin::{Accept, Control, Method, Solver};

struct Quadratic;

impl Objective<f64> for Quadratic {
    fn dim(&self) -> usize {
        1
    }
    fn bounds(&self) -> &Bounds<f64> {
        static BOUNDS: OnceLock<Bounds<f64>> = OnceLock::new();
        BOUNDS.get_or_init(|| Bounds::new(array![-1e6], array![1e6], 0.0))
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        0.5 * x.dot(&x)
    }
}

impl Gradient<f64> for Quadratic {
    fn dim(&self) -> usize {
        1
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        x.to_owned()
    }
}

impl DifferentiableObjective<f64> for Quadratic {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.grad(x))
    }
}

fn open_control() -> Control {
    Control::default().with_istep(1.0).with_maxmove(None)
}

fn bfgs() -> Solver {
    let mut solver = Solver::new(Method::Bfgs, open_control(), 1);
    solver.set_atom_maxmove(0.0);
    solver.set_accept(Accept::Energy);
    solver
}

#[test]
fn an_unset_multiplier_bound_takes_the_full_quadratic_step() {
    let mut solver = bfgs();
    let mut x = Array1::from_elem(1, 1.0);
    solver.step(&Quadratic, &mut x).unwrap();
    assert_eq!(x[0], 0.0);
}

#[test]
fn a_tighter_multiplier_bound_shortens_that_step() {
    let mut solver = bfgs();
    solver.set_line_alpha_max(0.25).unwrap();
    let mut x = Array1::from_elem(1, 1.0);
    solver.step(&Quadratic, &mut x).unwrap();
    assert_eq!(x[0], 0.75);
}

#[test]
fn a_looser_multiplier_bound_matches_the_unset_step() {
    let mut loose = bfgs();
    loose.set_line_alpha_max(10.0).unwrap();
    let mut bounded = Array1::from_elem(1, 1.0);
    loose.step(&Quadratic, &mut bounded).unwrap();
    let mut open = bfgs();
    let mut plain = Array1::from_elem(1, 1.0);
    open.step(&Quadratic, &mut plain).unwrap();
    assert_eq!(bounded[0], plain[0]);
}

#[test]
fn clearing_the_multiplier_bound_matches_the_unset_step() {
    let mut cleared = bfgs();
    cleared.set_line_alpha_max(0.25).unwrap();
    cleared.set_line_alpha_max(0.0).unwrap();
    let mut x = Array1::from_elem(1, 1.0);
    cleared.step(&Quadratic, &mut x).unwrap();
    assert_eq!(x[0], 0.0);
}

#[test]
fn a_non_finite_multiplier_bound_is_refused() {
    let mut solver = bfgs();
    assert!(solver.set_line_alpha_max(f64::NAN).is_err());
    let mut x = Array1::from_elem(1, 1.0);
    solver.step(&Quadratic, &mut x).unwrap();
    assert_eq!(x[0], 0.0);
}

#[test]
fn fixed_h0_scales_a_fresh_bfgs_step_without_a_search() {
    let mut scaled = Solver::new(Method::Bfgs, open_control(), 1);
    scaled.set_atom_maxmove(0.0);
    scaled.set_accept(Accept::Step);
    scaled.set_fixed_h0(0.5).unwrap();
    let mut x = Array1::from_elem(1, 1.0);
    scaled.step(&Quadratic, &mut x).unwrap();
    assert_eq!(x[0], 0.5);

    let mut identity = Solver::new(Method::Bfgs, open_control(), 1);
    identity.set_atom_maxmove(0.0);
    identity.set_accept(Accept::Step);
    let mut y = Array1::from_elem(1, 1.0);
    identity.step(&Quadratic, &mut y).unwrap();
    assert_eq!(y[0], 0.0);
}

#[test]
fn forget_restores_the_bfgs_scale() {
    let mut solver = Solver::new(Method::Bfgs, open_control(), 1);
    solver.set_atom_maxmove(0.0);
    solver.set_accept(Accept::Step);
    solver.set_fixed_h0(0.5).unwrap();
    let mut x = Array1::from_elem(1, 1.0);
    solver.step(&Quadratic, &mut x).unwrap();
    solver.forget();
    x[0] = 1.0;
    solver.step(&Quadratic, &mut x).unwrap();
    assert_eq!(x[0], 0.5);
}

#[test]
fn fixed_h0_scales_an_lbfgs_step_without_a_search() {
    let mut scaled = Solver::new(Method::lbfgs(), open_control(), 1);
    scaled.set_atom_maxmove(0.0);
    scaled.set_accept(Accept::Step);
    scaled.set_fixed_h0(0.5).unwrap();
    let mut x = Array1::from_elem(1, 1.0);
    scaled.step(&Quadratic, &mut x).unwrap();

    let mut plain = Solver::new(Method::lbfgs(), open_control(), 1);
    plain.set_atom_maxmove(0.0);
    plain.set_accept(Accept::Step);
    let mut y = Array1::from_elem(1, 1.0);
    plain.step(&Quadratic, &mut y).unwrap();
    assert_ne!(x[0], y[0]);
    assert_eq!(x[0], 0.5);
}

#[test]
fn fixed_h0_is_refused_on_steepest_descent() {
    let mut solver = Solver::new(Method::Steepest, open_control(), 1);
    assert!(solver.set_fixed_h0(0.5).is_err());
    assert!(solver.set_fixed_h0(0.0).is_err());
    assert!(solver.set_fixed_h0(f64::NAN).is_err());
}
