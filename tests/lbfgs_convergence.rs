//! L-BFGS convergence when an energy offset hides small decreases.

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};
use rgmin::{Accept, Control, Method, Solver};
use std::sync::OnceLock;

struct OffsetBowl;

impl Objective<f64> for OffsetBowl {
    fn dim(&self) -> usize {
        2
    }

    fn bounds(&self) -> &Bounds<f64> {
        static B: OnceLock<Bounds<f64>> = OnceLock::new();
        B.get_or_init(|| Bounds::new(array![-1e6, -1e6], array![1e6, 1e6], 0.0))
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        24.5 + 0.5 * (x[0] * x[0] + (x[1] - 1.0).powi(2))
    }
}

impl Gradient<f64> for OffsetBowl {
    fn dim(&self) -> usize {
        2
    }

    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        array![x[0], x[1] - 1.0]
    }
}

impl DifferentiableObjective<f64> for OffsetBowl {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.grad(x))
    }
}

#[test]
fn lbfgs_energy_acceptance_converges_below_energy_resolution() {
    let obj = OffsetBowl;
    for mut x in [array![3.0, -4.0], array![8e-9, 1.0 - 6e-9]] {
        let ctrl = Control {
            maxiter: 200,
            gtol: 1e-12,
            istep: 1.0,
            maxmove: None,
            ftol_rel: None,
        };
        let mut solver = Solver::new(Method::lbfgs(), ctrl, 2);
        solver.set_accept(Accept::Energy);
        let mut report = solver.step(&obj, &mut x).unwrap();
        for _ in 1..200 {
            if report.grad_norm <= 1e-12 {
                break;
            }
            report = solver.step(&obj, &mut x).unwrap();
        }
        assert!(
            report.grad_norm <= 1e-12,
            "gradient {}, x={x}",
            report.grad_norm
        );
        assert!(x[0].abs() <= 1e-12 && (x[1] - 1.0).abs() <= 1e-12);
    }
}

#[test]
fn stateful_step_discards_curvature_when_its_line_search_cannot_move() {
    use rgmin::{Lbfgs, LineSearch};

    let obj = OffsetBowl;
    let mut opt = Lbfgs::default();
    opt.record(array![1.0, 0.0], array![1e-12, 0.0]);
    assert_eq!(opt.len(), 1);
    let mut x = array![3.0, -4.0];
    let (mut value, mut gradient) = obj.value_and_gradient(x.view());
    let mut step = 1.0;
    opt.step_objective(
        &obj,
        &mut x,
        &mut value,
        &mut gradient,
        &mut step,
        LineSearch::Backtracking {
            c: 1e-4,
            beta: 0.5,
            maxiter: 20,
        },
        &Control::default(),
        None,
    );
    assert_eq!(x, array![0.0, 1.0]);
    assert_eq!(value, 24.5);
    assert_eq!(gradient, array![0.0, 0.0]);
}
