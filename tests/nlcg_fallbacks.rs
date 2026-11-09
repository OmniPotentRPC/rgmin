//! The NLCG descent-fallback counter.
//!
//! A Euclidean session does not replace its direction, so the counter
//! stays at zero. A zero gradient still stops under the tolerance.

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};
use rgmin::{Control, Method, Solver};

struct Bowl;

impl Objective<f64> for Bowl {
    fn dim(&self) -> usize {
        2
    }
    fn bounds(&self) -> &Bounds<f64> {
        use std::sync::OnceLock;
        static B: OnceLock<Bounds<f64>> = OnceLock::new();
        B.get_or_init(|| Bounds::new(array![-8.0, -8.0], array![8.0, 8.0], 0.0))
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        0.5 * (x[0] * x[0] + 4.0 * x[1] * x[1])
    }
}

impl Gradient<f64> for Bowl {
    fn dim(&self) -> usize {
        2
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        array![x[0], 4.0 * x[1]]
    }
}

impl DifferentiableObjective<f64> for Bowl {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.grad(x))
    }
}

#[test]
fn a_euclidean_quadratic_reports_no_fallbacks() {
    let mut solver = Solver::new(Method::polak_ribiere(), Control::default(), 2);
    assert_eq!(solver.nlcg_descent_fallbacks(), 0);
    let mut x = array![1.2, -0.7];
    for _ in 0..8 {
        solver.step(&Bowl, &mut x).unwrap();
    }
    assert!(x.iter().all(|v| v.is_finite()));
    assert_eq!(solver.nlcg_descent_fallbacks(), 0);
}
