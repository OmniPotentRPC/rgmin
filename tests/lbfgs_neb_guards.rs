//! Opt-in L-BFGS guards. Off, the two-loop step is the default step.

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};
use rgmin::{Control, Method, Solver};

struct Steep;

impl Objective<f64> for Steep {
    fn dim(&self) -> usize {
        3
    }
    fn bounds(&self) -> &Bounds<f64> {
        use std::sync::OnceLock;
        static B: OnceLock<Bounds<f64>> = OnceLock::new();
        B.get_or_init(|| Bounds::new(array![-20.0, -20.0, -20.0], array![20.0, 20.0, 20.0], 0.0))
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        0.5 * x.dot(&x)
    }
}

impl Gradient<f64> for Steep {
    fn dim(&self) -> usize {
        3
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        x.to_owned()
    }
}

impl DifferentiableObjective<f64> for Steep {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.grad(x))
    }
}

fn session() -> Solver {
    Solver::new(
        Method::lbfgs(),
        Control {
            gtol: 1e-14,
            istep: 1.0,
            ..Control::default()
        },
        3,
    )
}

#[test]
fn guards_left_off_match_a_session_that_never_sets_them() {
    let mut plain = session();
    let mut named = session();
    named.set_lbfgs_neb_guards(false);
    let mut a = array![1.0, -0.5, 0.25];
    let mut b = a.clone();
    let ra = plain.step(&Steep, &mut a).unwrap();
    let rb = named.step(&Steep, &mut b).unwrap();
    assert_eq!(a, b);
    assert_eq!(ra.grad_norm.to_bits(), rb.grad_norm.to_bits());
    assert_eq!(plain.pair_count(), named.pair_count());
}

#[test]
fn guards_on_replace_a_step_longer_than_the_cap() {
    let mut plain = session();
    let mut guarded = session();
    plain.set_geometry_step(false);
    guarded.set_geometry_step(false);
    plain.set_atom_maxmove(0.2);
    guarded.set_atom_maxmove(0.2);
    guarded.set_lbfgs_neb_guards(true);
    let mut a = array![4.0, 0.0, 0.0];
    let mut b = a.clone();
    plain.step(&Steep, &mut a).unwrap();
    guarded.step(&Steep, &mut b).unwrap();
    plain.step(&Steep, &mut a).unwrap();
    guarded.step(&Steep, &mut b).unwrap();
    assert!(a != b, "guarded step matched the unguarded two-loop {a:?}");
}
