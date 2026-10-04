//! Accept::Step on session L-BFGS takes the clipped step even when the
//! value rises. A band force is not the gradient of the summed energy.

use std::sync::atomic::{AtomicUsize, Ordering};

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};
use rgmin::{Accept, Control, Method, Solver};

struct Uphill {
    evals: AtomicUsize,
}

impl Objective<f64> for Uphill {
    fn dim(&self) -> usize {
        3
    }
    fn bounds(&self) -> &Bounds<f64> {
        use std::sync::OnceLock;
        static B: OnceLock<Bounds<f64>> = OnceLock::new();
        B.get_or_init(|| Bounds::new(array![-10.0, -10.0, -10.0], array![10.0, 10.0, 10.0], 0.0))
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.evals.fetch_add(1, Ordering::Relaxed);
        x[0]
    }
}

impl Gradient<f64> for Uphill {
    fn dim(&self) -> usize {
        3
    }
    fn grad(&self, _x: ArrayView1<f64>) -> Array1<f64> {
        array![-1.0, 0.0, 0.0]
    }
}

impl DifferentiableObjective<f64> for Uphill {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.grad(x))
    }
}

#[test]
fn accept_step_clips_an_uphill_step() {
    let obj = Uphill {
        evals: AtomicUsize::new(0),
    };
    let mut x = array![0.0, 0.0, 0.0];
    let mut solver = Solver::new(
        Method::lbfgs(),
        Control {
            maxiter: 1,
            gtol: 0.0,
            istep: 1.0,
            maxmove: None,
            ftol_rel: None,
        },
        3,
    );
    solver.set_accept(Accept::Step);
    solver.set_atom_maxmove(0.2);
    let rep = solver.step(&obj, &mut x).unwrap();
    assert!((x[0] - 0.2).abs() < 1e-12, "x0 {}", x[0]);
    assert_eq!(x[1], 0.0);
    assert_eq!(x[2], 0.0);
    assert!((rep.value - 0.2).abs() < 1e-12, "value {}", rep.value);
    // Opening gradient plus one trial. A line search would evaluate more.
    assert_eq!(obj.evals.load(Ordering::Relaxed), 2);
}
