//! The published FIRE 2.0 schedule stops when `vdfmax` is exceeded.
use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};
use rgmin::{Control, Error, FireKind, FireVariant, Method, Solver};

struct Bowl;

impl Objective<f64> for Bowl {
    fn dim(&self) -> usize {
        1
    }
    fn bounds(&self) -> &Bounds<f64> {
        use std::sync::OnceLock;
        static B: OnceLock<Bounds<f64>> = OnceLock::new();
        B.get_or_init(|| Bounds::new(array![-10.0], array![10.0], 0.0))
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        x[0] * x[0]
    }
}

impl Gradient<f64> for Bowl {
    fn dim(&self) -> usize {
        1
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        array![2.0 * x[0]]
    }
}

impl DifferentiableObjective<f64> for Bowl {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.grad(x))
    }
}

fn published() -> Solver {
    let mut solver = Solver::new(
        Method::Fire { kind: FireKind::V2 },
        Control {
            maxiter: 5,
            gtol: 1e-12,
            istep: 0.1,
            ..Control::default()
        },
        1,
    );
    solver.set_fire_variant(FireVariant::Guenole2020);
    solver
}

#[test]
fn a_zero_vdfmax_stops_the_first_step_without_moving() {
    let mut solver = published();
    solver.set_fire_vdfmax(0);
    let mut x = array![1.0];
    let err = solver.step(&Bowl, &mut x).unwrap_err();
    assert!(matches!(err, Error::FireStalled { steps: 0 }), "{err}");
    assert_eq!(x[0], 1.0);
}

#[test]
fn forget_clears_the_negative_power_count() {
    let mut solver = published();
    solver.set_fire_vdfmax(1);
    let mut x = array![1.0];
    solver
        .step(&Bowl, &mut x)
        .expect("the first non-positive step is allowed");
    solver.forget();
    let again = solver.step(&Bowl, &mut x);
    assert!(again.is_ok(), "{again:?}");
}
