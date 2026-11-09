//! Default L-BFGS trace with no preconditioner installed.
//!
//! The locked bits are the session at develop 39e08b85. A later
//! opt-in preconditioner has to leave this trace unchanged.

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1};
use rgmin::{Control, Method, Solver};

struct Anisotropic {
    bounds: Bounds<f64>,
    calls: std::sync::atomic::AtomicUsize,
}

impl Objective<f64> for Anisotropic {
    fn dim(&self) -> usize {
        6
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.value_and_gradient(x).0
    }
}

impl Gradient<f64> for Anisotropic {
    fn dim(&self) -> usize {
        6
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.value_and_gradient(x).1
    }
}

impl DifferentiableObjective<f64> for Anisotropic {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let w = [1.0, 100.0, 0.25, 4.0, 9.0, 0.5];
        let mut g = Array1::zeros(6);
        let mut f = 0.0;
        for i in 0..6 {
            g[i] = w[i] * x[i];
            f += 0.5 * w[i] * x[i] * x[i];
        }
        (f, g)
    }
}

fn run() -> (usize, usize, f64, Array1<f64>, f64) {
    let obj = Anisotropic {
        bounds: Bounds::new(Array1::from_elem(6, -1e6), Array1::from_elem(6, 1e6), 0.0),
        calls: std::sync::atomic::AtomicUsize::new(0),
    };
    let control = Control {
        maxiter: 40,
        gtol: 1e-10,
        istep: 1.0,
        maxmove: None,
        ..Control::default()
    };
    let mut solver = Solver::new(Method::Lbfgs { memory: 7 }, control, 6);
    let mut x = Array1::from_vec(vec![0.4, -0.2, 0.7, -0.3, 0.15, -0.55]);
    let mut steps = 0;
    let mut value = 0.0;
    let mut gnorm = f64::INFINITY;
    for _ in 0..40 {
        let report = solver.step(&obj, &mut x).unwrap();
        steps = report.steps;
        value = report.value;
        gnorm = report.grad_norm;
        if gnorm < 1e-10 {
            break;
        }
    }
    let calls = obj.calls.load(std::sync::atomic::Ordering::Relaxed);
    (calls, steps, value, x, gnorm)
}

#[test]
fn default_lbfgs_trace_keeps_the_develop_bits() {
    let (calls, steps, _value, x, _gnorm) = run();
    assert_eq!(calls, 41);
    assert_eq!(steps, 40);
    let mut bits = String::new();
    for v in x.iter() {
        bits.push_str(&format!("{:016x}", v.to_bits()));
    }
    assert_eq!(
        bits,
        "3ddcac8384388f42bd9ddd60b56688303df3c40f6eb00b02bdc96b00e1c17172bd839a809a638a503ddbd8b2d1b683ba"
    );
}
