//! Accept::Step on session L-BFGS takes the clipped step even when the
//! value rises. A band force is not the gradient of the summed energy.

use std::sync::atomic::{AtomicUsize, Ordering};

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};
use rgmin::{Accept, Control, Method, Oracle, Solver};

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
            noise_floor: None,
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

#[test]
fn predicted_cautious_step_is_one_oracle_of_length_h0() {
    let calls = AtomicUsize::new(0);
    let objective = Oracle::unbounded(3, |x: ArrayView1<f64>| {
        calls.fetch_add(1, Ordering::Relaxed);
        (0.5 * x.dot(&x), x.to_owned())
    });
    let mut x = array![2.0, 0.0, 0.0];
    let mut solver = Solver::new(Method::Lbfgs { memory: 150 }, Control::default(), 3);
    solver.set_accept(Accept::Predicted);
    solver.set_maxmove(0.25);
    assert!(solver.set_lbfgs_scale(0.08, true));
    let rep = solver.step(&objective, &mut x).unwrap();
    // ||g|| = 2, so the cautious step has length h0 = 0.08 along -g.
    assert!((x[0] - 1.92).abs() < 1e-12, "x {}", x[0]);
    assert!(rep.grad_norm > 0.01);
    assert_eq!(calls.load(Ordering::Relaxed), 2);
}
