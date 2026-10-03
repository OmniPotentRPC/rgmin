use ndarray::{ArrayView1, array};
use rgmin::{Control, Method, Oracle, Solver};

#[test]
fn tightening_the_tolerance_reaches_the_requested_gradient() {
    let objective = Oracle::unbounded(1, |x: ArrayView1<f64>| {
        (x[0] * x[0], array![2.0 * x[0]])
    });
    for method in [Method::lbfgs(), Method::Bfgs, Method::Steepest] {
        let mut solver = Solver::new(
            method,
            Control {
                maxiter: 10,
                gtol: 1e-3,
                istep: 0.5,
                maxmove: None,
                ftol_rel: None,
            },
            1,
        )
        .with_gtol(1e-10);
        let mut x = array![1e-6];
        let report = solver.step(&objective, &mut x).unwrap();
        assert!(report.grad_norm < 1e-10, "gradient {}", report.grad_norm);
        assert!(x[0].abs() < 5e-11);
        assert_eq!(report.steps, 1);
    }
}

#[test]
fn loosening_the_tolerance_leaves_a_stationary_point_in_place() {
    let objective = Oracle::unbounded(1, |x: ArrayView1<f64>| {
        (x[0] * x[0], array![2.0 * x[0]])
    });
    for method in [Method::lbfgs(), Method::Bfgs, Method::Steepest] {
        let mut solver = Solver::new(
            method,
            Control {
                maxiter: 10,
                gtol: 1e-12,
                istep: 0.5,
                maxmove: None,
                ftol_rel: None,
            },
            1,
        )
        .with_gtol(1e-3);
        let mut x = array![1e-4];
        let report = solver.step(&objective, &mut x).unwrap();
        assert_eq!(x[0], 1e-4);
        assert_eq!(report.grad_norm, 2e-4);
        assert_eq!(report.steps, 0);
    }
}
