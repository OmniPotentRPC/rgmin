//! A small conjugate direction cannot certify a nonstationary point.

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};
use rgmin::nlcg::{Conjugacy, Restart};
use rgmin::{Control, DirectionalCurvature, ScgParams, minimize_scg, minimize_scg_exact};

struct Bowl {
    bounds: Bounds<f64>,
}
impl Objective<f64> for Bowl {
    fn dim(&self) -> usize {
        2
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        0.5 * x.dot(&x)
    }
}
impl Gradient<f64> for Bowl {
    fn dim(&self) -> usize {
        2
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        x.to_owned()
    }
}
impl DifferentiableObjective<f64> for Bowl {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.grad(x))
    }
}
impl DirectionalCurvature for Bowl {
    fn directional_curvature(&self, _x: ArrayView1<f64>, d: ArrayView1<f64>) -> Option<f64> {
        Some(d.dot(&d))
    }
}

#[test]
fn tiny_liu_storey_direction_restarts_until_the_gradient_is_small() {
    let objective = Bowl {
        bounds: Bounds::new(array![-10.0, -10.0], array![10.0, 10.0], 0.0),
    };
    let control = Control {
        maxiter: 20,
        gtol: 1e-8,
        ..Control::default()
    };
    let params = ScgParams {
        lambda: 1e-5,
        tol_sol: 0.0,
        tol_func: 0.0,
        ..ScgParams::default()
    };
    for exact in [false, true] {
        let report = if exact {
            minimize_scg_exact(
                &objective,
                array![3.0, -4.0],
                &control,
                &params,
                Conjugacy::LiuStorey,
                Restart::Never,
            )
        } else {
            minimize_scg(
                &objective,
                array![3.0, -4.0],
                &control,
                &params,
                Conjugacy::LiuStorey,
                Restart::Never,
            )
        }
        .unwrap();
        let physical_gradient_norm = report.coords[0].hypot(report.coords[1]);
        assert!(physical_gradient_norm < 1e-8, "exact={exact}: {report:?}");
        assert!(report.value < 5e-17, "exact={exact}: {report:?}");
        assert!(
            report.steps >= 2 && report.steps < control.maxiter,
            "{report:?}"
        );
    }
}
