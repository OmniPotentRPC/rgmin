#![allow(clippy::type_complexity)]

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};
use rgmin::{
    Control, Error, HessianOracle, HvpOracle, LineSearch, Method, NewtonKind, Restart, Status,
    minimize, minimize_bfgs, minimize_lbfgs, minimize_method, minimize_method_hess,
    minimize_method_hvp, minimize_newton, minimize_sd, minimize_sr1, minimize_sr2,
};

fn strict_control(maxiter: usize) -> Control {
    Control::default()
        .with_gtol(1e-8)
        .with_maxiter(maxiter)
        .with_require_convergence(true)
}

fn lax_control(maxiter: usize) -> Control {
    strict_control(maxiter).with_require_convergence(false)
}

fn quad() -> rgmin::Oracle<impl Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync> {
    rgmin::Oracle::unbounded(1, |x: ArrayView1<f64>| (0.5 * x[0] * x[0], array![x[0]]))
}

fn quad_hvp() -> HvpOracle<
    impl Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
    impl Fn(ArrayView1<f64>, ArrayView1<f64>) -> Array1<f64> + Send + Sync,
> {
    HvpOracle::unbounded(
        1,
        |x: ArrayView1<f64>| (0.5 * x[0] * x[0], array![x[0]]),
        |_x, v| v.to_owned(),
    )
}

fn uphill_hess() -> HessianOracle<
    impl Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
    impl Fn(ArrayView1<f64>) -> ndarray::Array2<f64> + Send + Sync,
> {
    HessianOracle::unbounded(
        1,
        |x: ArrayView1<f64>| {
            let value = 0.5 * x[0] * x[0];
            let grad = array![-x[0]];
            (value, grad)
        },
        |_| ndarray::array![[1.0]],
    )
}

fn uphill_hvp() -> HvpOracle<
    impl Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
    impl Fn(ArrayView1<f64>, ArrayView1<f64>) -> Array1<f64> + Send + Sync,
> {
    HvpOracle::unbounded(
        1,
        |x: ArrayView1<f64>| {
            let value = 0.5 * x[0] * x[0];
            let grad = array![-x[0]];
            (value, grad)
        },
        |_x, v| v.to_owned(),
    )
}

fn wall_box() -> Bounds<f64> {
    Bounds::new(array![0.0], array![0.0], 0.0)
}

/// A bounded objective with a non-zero gradient at `x = 0`.
///
/// Every descent direction points outside the lower bound, so clipped
/// trial points stay at `x = 0` and the stepper reports no movement.
struct WallObjective {
    bounds: Bounds<f64>,
}

impl WallObjective {
    fn new() -> Self {
        Self { bounds: wall_box() }
    }
}

impl Objective<f64> for WallObjective {
    fn dim(&self) -> usize {
        1
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        x[0]
    }
}

impl Gradient<f64> for WallObjective {
    fn dim(&self) -> usize {
        1
    }
    fn grad(&self, _x: ArrayView1<f64>) -> Array1<f64> {
        array![1.0]
    }
}

impl DifferentiableObjective<f64> for WallObjective {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.grad(x))
    }
}

fn assert_not_converged(err: Error, status: Status) {
    match err {
        Error::NotConverged(report) => {
            assert!(!report.converged);
            assert_eq!(report.status, status);
        }
        other => panic!("expected NotConverged, got {other:?}"),
    }
}

#[test]
fn qn_drivers_strict_and_lax() {
    let ls = LineSearch::brent(32, 1e-10);
    let strict_budget = strict_control(0);
    let lax_budget = lax_control(0);
    let strict_stall = strict_control(4);
    let lax_stall = lax_control(4);

    assert_not_converged(
        minimize_bfgs(&quad(), array![4.0], &strict_budget, ls).expect_err("strict budget"),
        Status::Budget,
    );
    assert_not_converged(
        minimize_lbfgs(&quad(), array![4.0], &strict_budget, ls, 6).expect_err("strict budget"),
        Status::Budget,
    );
    assert_not_converged(
        minimize_sr1(&quad(), array![4.0], &strict_budget, ls).expect_err("strict budget"),
        Status::Budget,
    );
    assert_not_converged(
        minimize_sr2(&quad(), array![4.0], &strict_budget, ls).expect_err("strict budget"),
        Status::Budget,
    );
    assert_not_converged(
        minimize_sd(&quad(), array![4.0], &strict_budget, ls).expect_err("strict budget"),
        Status::Budget,
    );

    for report in [
        minimize_bfgs(&quad(), array![4.0], &lax_budget, ls).expect("lax budget"),
        minimize_lbfgs(&quad(), array![4.0], &lax_budget, ls, 6).expect("lax budget"),
        minimize_sr1(&quad(), array![4.0], &lax_budget, ls).expect("lax budget"),
        minimize_sr2(&quad(), array![4.0], &lax_budget, ls).expect("lax budget"),
        minimize_sd(&quad(), array![4.0], &lax_budget, ls).expect("lax budget"),
    ] {
        assert_eq!(report.status, Status::Budget);
        assert!(!report.converged);
    }

    let wall = WallObjective::new();
    assert_not_converged(
        minimize_bfgs(&wall, array![0.0], &strict_stall, ls).expect_err("strict stall"),
        Status::StepTolerance,
    );
    let lax = minimize_bfgs(&wall, array![0.0], &lax_stall, ls).expect("lax stall");
    assert_eq!(lax.status, Status::Budget);
    assert!(!lax.converged);
}

#[test]
fn session_and_newton_drivers_strict_and_lax() {
    let strict_budget = strict_control(0);
    let lax_budget = lax_control(0);
    let strict_stall = strict_control(4);
    let lax_stall = lax_control(4);

    assert_not_converged(
        minimize(
            &quad(),
            array![4.0],
            &strict_budget,
            rgmin::Conjugacy::PolakRibiere,
            Restart::Never,
            LineSearch::brent(32, 1e-10),
        )
        .expect_err("nlcg strict budget"),
        Status::Budget,
    );
    let nlcg_lax = minimize(
        &quad(),
        array![4.0],
        &lax_budget,
        rgmin::Conjugacy::PolakRibiere,
        Restart::Never,
        LineSearch::brent(32, 1e-10),
    )
    .expect("nlcg lax budget");
    assert_eq!(nlcg_lax.status, Status::Budget);
    assert!(!nlcg_lax.converged);

    assert_not_converged(
        minimize_newton(
            &uphill_hess(),
            array![1.0],
            &strict_stall,
            NewtonKind::Shifted,
        )
        .expect_err("newton strict stall"),
        Status::StepTolerance,
    );
    let newton_lax = minimize_newton(&uphill_hess(), array![1.0], &lax_stall, NewtonKind::Shifted)
        .expect("newton lax stall");
    assert_eq!(newton_lax.status, Status::Budget);
    assert!(!newton_lax.converged);

    assert_not_converged(
        minimize_method(
            &WallObjective::new(),
            array![0.0],
            &strict_stall,
            Method::Steepest,
            LineSearch::brent(32, 1e-10),
        )
        .expect_err("session strict stall"),
        Status::StepTolerance,
    );
    let run_lax = minimize_method(
        &WallObjective::new(),
        array![0.0],
        &lax_stall,
        Method::Steepest,
        LineSearch::brent(32, 1e-10),
    )
    .expect("session lax stall");
    assert_eq!(run_lax.status, Status::Budget);
    assert!(!run_lax.converged);

    assert_not_converged(
        minimize_method_hess(
            &uphill_hess(),
            array![1.0],
            &strict_stall,
            Method::Dogleg,
            LineSearch::default(),
        )
        .expect_err("session hess strict stall"),
        Status::StepTolerance,
    );
    let hess_lax = minimize_method_hess(
        &uphill_hess(),
        array![1.0],
        &lax_stall,
        Method::Dogleg,
        LineSearch::default(),
    )
    .expect("session hess lax stall");
    assert_eq!(hess_lax.status, Status::Budget);
    assert!(!hess_lax.converged);

    assert_not_converged(
        minimize_method_hvp(
            &quad_hvp(),
            array![4.0],
            &strict_budget,
            Method::NewtonCg,
            LineSearch::default(),
        )
        .expect_err("session hvp strict budget"),
        Status::Budget,
    );
    let hvp_lax = minimize_method_hvp(
        &quad_hvp(),
        array![4.0],
        &lax_budget,
        Method::NewtonCg,
        LineSearch::default(),
    )
    .expect("session hvp lax budget");
    assert_eq!(hvp_lax.status, Status::Budget);
    assert!(!hvp_lax.converged);

    assert_not_converged(
        minimize_method_hvp(
            &uphill_hvp(),
            array![1.0],
            &strict_stall,
            Method::NewtonCg,
            LineSearch::default(),
        )
        .expect_err("session hvp strict stall"),
        Status::StepTolerance,
    );
}

#[test]
fn converged_flag_tracks_gtol() {
    let control = strict_control(8).with_require_convergence(false);
    let ls = LineSearch::brent(32, 1e-10);

    let report = minimize_sd(&quad(), array![0.0], &control, ls).expect("converged");
    assert!(report.converged);
    assert_eq!(report.status, Status::Tolerance);

    let report = minimize_bfgs(&quad(), array![4.0], &lax_control(0), ls).expect("budget");
    assert!(!report.converged);
}
