//! Strict convergence is opt in. The default still returns `Ok`.

use eindir_core::{DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};
use rgmin::{
    Control, Error, HessianObjective, HessianOracle, HessianVector, Lbfgs, LineSearch, Method,
    NewtonKind, NoiseFloor, Oracle, Report, ScgParams, Solver, Status, StopReason, minimize,
    minimize_adam, minimize_bfgs, minimize_method_hess, minimize_method_hvp, minimize_newton,
    minimize_pso, minimize_scg, minimize_sd, polish_newton,
};
use std::sync::OnceLock;

fn quad() -> Oracle<impl Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync> {
    Oracle::unbounded(1, |x: ArrayView1<f64>| (0.5 * x[0] * x[0], array![x[0]]))
}

fn rose() -> Oracle<impl Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync> {
    Oracle::unbounded(2, |x: ArrayView1<f64>| {
        let (a, b) = (x[0], x[1]);
        (
            (1.0 - a).powi(2) + 100.0 * (b - a * a).powi(2),
            array![
                -2.0 * (1.0 - a) - 400.0 * a * (b - a * a),
                200.0 * (b - a * a)
            ],
        )
    })
}

fn uphill() -> Oracle<impl Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync> {
    Oracle::unbounded(1, |x: ArrayView1<f64>| (0.5 * x[0] * x[0], array![-x[0]]))
}

fn uphill_hess() -> impl HessianObjective {
    HessianOracle::unbounded(
        1,
        |x: ArrayView1<f64>| (0.5 * x[0] * x[0], array![-x[0]]),
        |_| array![[1.0]],
    )
}

struct Wall;

impl Objective<f64> for Wall {
    fn dim(&self) -> usize {
        1
    }
    fn bounds(&self) -> &eindir_core::Bounds<f64> {
        static BOUNDS: OnceLock<eindir_core::Bounds<f64>> = OnceLock::new();
        BOUNDS.get_or_init(|| eindir_core::Bounds::new(array![0.0], array![0.0], 0.0))
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        x[0]
    }
}

impl Gradient<f64> for Wall {
    fn dim(&self) -> usize {
        1
    }
    fn grad(&self, _x: ArrayView1<f64>) -> Array1<f64> {
        array![1.0]
    }
}

impl DifferentiableObjective<f64> for Wall {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.grad(x))
    }
}

struct WallHess;

impl Objective<f64> for WallHess {
    fn dim(&self) -> usize {
        1
    }
    fn bounds(&self) -> &eindir_core::Bounds<f64> {
        static BOUNDS: OnceLock<eindir_core::Bounds<f64>> = OnceLock::new();
        BOUNDS.get_or_init(|| eindir_core::Bounds::new(array![0.0], array![0.0], 0.0))
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        x[0]
    }
}

impl Gradient<f64> for WallHess {
    fn dim(&self) -> usize {
        1
    }
    fn grad(&self, _x: ArrayView1<f64>) -> Array1<f64> {
        array![1.0]
    }
}

impl DifferentiableObjective<f64> for WallHess {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.grad(x))
    }
}

impl HessianObjective for WallHess {
    fn hessian(&self, _x: ArrayView1<f64>) -> ndarray::Array2<f64> {
        array![[1.0]]
    }
}

impl HessianVector for WallHess {
    fn hessian_vector(&self, _x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
        v.to_owned()
    }
}

fn strict(control: Control) -> Control {
    control.with_require_convergence(true)
}

fn refused(result: rgmin::Result<Report>) -> Report {
    match result {
        Err(Error::NotConverged(report)) => *report,
        Ok(report) => panic!("strict mode returned Ok({report:?})"),
        Err(err) => panic!("strict mode returned {err}"),
    }
}

fn same_stop(lax: &Report, tight: &Report) {
    assert_eq!(lax.stop_reason(), tight.stop_reason());
    assert_eq!(lax.status, tight.status);
    assert!(!lax.converged);
    assert!(!tight.converged);
    assert!(!lax.converged());
}

/// A point already at the minimum is `Ok` with the flag on or off.
#[test]
fn a_met_tolerance_stays_ok() {
    let control = Control::default().with_maxiter(0).with_gtol(1e-8);
    for on in [false, true] {
        let report = minimize_sd(
            &quad(),
            array![0.0],
            &control.clone().with_require_convergence(on),
            LineSearch::default(),
        )
        .unwrap();
        assert_eq!(report.status, Status::Tolerance);
        assert!(report.converged());
        assert!(report.converged);
    }
}

/// A zero budget above `gtol` is `MaxIter` either way, and an error only
/// when strict convergence is on.
#[test]
fn a_zero_budget_errors_only_when_strict() {
    let lax_control = Control::default().with_maxiter(0).with_gtol(1e-8);
    let calls: Vec<(&str, rgmin::Result<Report>, rgmin::Result<Report>)> = vec![
        (
            "bfgs",
            minimize_bfgs(&quad(), array![4.0], &lax_control, LineSearch::default()),
            minimize_bfgs(
                &quad(),
                array![4.0],
                &strict(lax_control.clone()),
                LineSearch::default(),
            ),
        ),
        (
            "sd",
            minimize_sd(&quad(), array![4.0], &lax_control, LineSearch::default()),
            minimize_sd(
                &quad(),
                array![4.0],
                &strict(lax_control.clone()),
                LineSearch::default(),
            ),
        ),
        (
            "adam",
            minimize_adam(
                &quad(),
                array![4.0],
                &lax_control,
                LineSearch::default(),
                0.9,
                0.999,
                1e-8,
            ),
            minimize_adam(
                &quad(),
                array![4.0],
                &strict(lax_control.clone()),
                LineSearch::default(),
                0.9,
                0.999,
                1e-8,
            ),
        ),
        (
            "cg",
            minimize(
                &quad(),
                array![4.0],
                &lax_control,
                rgmin::Conjugacy::PolakRibiere,
                rgmin::Restart::Never,
                LineSearch::default(),
            ),
            minimize(
                &quad(),
                array![4.0],
                &strict(lax_control.clone()),
                rgmin::Conjugacy::PolakRibiere,
                rgmin::Restart::Never,
                LineSearch::default(),
            ),
        ),
        (
            "pso",
            minimize_pso(&quad(), array![4.0], &lax_control, 4, 0.5, 1.5, 1.5),
            minimize_pso(
                &quad(),
                array![4.0],
                &strict(lax_control.clone()),
                4,
                0.5,
                1.5,
                1.5,
            ),
        ),
    ];
    for (name, lax, tight) in calls {
        let lax = lax.unwrap_or_else(|err| panic!("{name} default: {err}"));
        let tight = refused(tight);
        assert_eq!(lax.stop_reason(), StopReason::MaxIter, "{name}");
        same_stop(&lax, &tight);
    }
}

/// Energy noise on BFGS and steepest descent uses the same strict check
/// as Adam and CG.
#[test]
fn energy_noise_stalls_share_the_strict_check() {
    let floor = NoiseFloor::new(0.0).with_eps_f(1.0);
    let lax_control = Control::default()
        .with_maxiter(20)
        .with_gtol(1e-8)
        .with_noise_floor(floor);
    let pairs = [
        (
            "bfgs",
            minimize_bfgs(
                &quad(),
                array![0.5],
                &lax_control,
                LineSearch::wolfe(1e-4, 0.9, 20),
            ),
            minimize_bfgs(
                &quad(),
                array![0.5],
                &strict(lax_control.clone()),
                LineSearch::wolfe(1e-4, 0.9, 20),
            ),
        ),
        (
            "sd",
            minimize_sd(
                &quad(),
                array![0.5],
                &lax_control,
                LineSearch::wolfe(1e-4, 0.9, 20),
            ),
            minimize_sd(
                &quad(),
                array![0.5],
                &strict(lax_control.clone()),
                LineSearch::wolfe(1e-4, 0.9, 20),
            ),
        ),
        (
            "adam",
            minimize_adam(
                &quad(),
                array![0.5],
                &lax_control,
                LineSearch::wolfe(1e-4, 0.9, 20),
                0.9,
                0.999,
                1e-8,
            ),
            minimize_adam(
                &quad(),
                array![0.5],
                &strict(lax_control.clone()),
                LineSearch::wolfe(1e-4, 0.9, 20),
                0.9,
                0.999,
                1e-8,
            ),
        ),
        (
            "cg",
            minimize(
                &quad(),
                array![0.5],
                &lax_control,
                rgmin::Conjugacy::PolakRibiere,
                rgmin::Restart::Never,
                LineSearch::wolfe(1e-4, 0.9, 20),
            ),
            minimize(
                &quad(),
                array![0.5],
                &strict(lax_control),
                rgmin::Conjugacy::PolakRibiere,
                rgmin::Restart::Never,
                LineSearch::wolfe(1e-4, 0.9, 20),
            ),
        ),
    ];
    for (name, lax, tight) in pairs {
        let lax = lax.unwrap_or_else(|err| panic!("{name} default: {err}"));
        let tight = refused(tight);
        assert_eq!(lax.status, Status::NoiseFloor, "{name}: {lax:?}");
        same_stop(&lax, &tight);
        assert_eq!(lax.coords[0].to_bits(), 0.5f64.to_bits(), "{name}");
    }
}

/// A pinned point is a step stall under strict mode and a budget when
/// the flag is off.
#[test]
fn an_unmoved_bfgs_step_is_step_tolerance_only_when_strict() {
    let lax_control = Control::default().with_maxiter(4).with_gtol(1e-8);
    let lax = minimize_bfgs(&Wall, array![0.0], &lax_control, LineSearch::default()).unwrap();
    let tight = refused(minimize_bfgs(
        &Wall,
        array![0.0],
        &strict(lax_control),
        LineSearch::default(),
    ));
    assert_eq!(lax.status, Status::Budget);
    assert_eq!(lax.stop_reason(), StopReason::MaxIter);
    assert_eq!(tight.status, Status::StepTolerance);
    assert_eq!(tight.stop_reason(), StopReason::StepTolerance);
    assert!(!tight.converged);
}

#[test]
fn a_failed_line_search_errors_only_when_strict() {
    let lax_control = Control::default().with_maxiter(3).with_gtol(1e-8);
    let mut plain = Lbfgs::with_capacity(4);
    plain.norm = rgmin::GradNorm::Euclidean;
    let mut tight_solver = Lbfgs::with_capacity(4);
    tight_solver.norm = rgmin::GradNorm::Euclidean;
    let lax = plain
        .minimize_objective(&uphill(), array![1.0], &lax_control, LineSearch::default())
        .unwrap();
    let tight = refused(tight_solver.minimize_objective(
        &uphill(),
        array![1.0],
        &strict(lax_control),
        LineSearch::default(),
    ));
    assert_eq!(lax.stop_reason(), StopReason::LineSearchFailed);
    same_stop(&lax, &tight);
    assert_eq!(lax.coords[0].to_bits(), 1.0f64.to_bits());
}

#[test]
fn an_unmet_polish_errors_only_when_strict() {
    let saddle = HessianOracle::unbounded(
        1,
        |x: ArrayView1<f64>| (-0.5 * x[0] * x[0], array![-x[0]]),
        |_| array![[-1.0]],
    );
    let lax_control = Control::default()
        .with_maxiter(3)
        .with_backward_polish(true);
    let lax = polish_newton(&saddle, array![1.0], &lax_control, NewtonKind::Shifted).unwrap();
    let tight = refused(polish_newton(
        &saddle,
        array![1.0],
        &strict(lax_control),
        NewtonKind::Shifted,
    ));
    assert!(matches!(
        lax.stop_reason(),
        StopReason::PolishUnmet | StopReason::LineSearchFailed
    ));
    same_stop(&lax, &tight);
}

#[test]
fn scg_step_tolerance_errors_only_when_strict() {
    const VALUE: f64 = 1_152_921_504_606_846_976.0;
    let flat = Oracle::unbounded(2, move |x: ArrayView1<f64>| {
        (VALUE + x[0], array![1.0, 0.0])
    });
    let lax_control = Control::default().with_maxiter(0).with_gtol(0.0);
    let params = ScgParams::default()
        .with_tol_sol(0.0)
        .with_lambda_limit(f64::INFINITY);
    let lax = minimize_scg(
        &flat,
        array![0.0, 0.0],
        &lax_control,
        &params,
        rgmin::Conjugacy::PolakRibiere,
        rgmin::Restart::Never,
    )
    .unwrap();
    let tight = refused(minimize_scg(
        &flat,
        array![0.0, 0.0],
        &strict(lax_control),
        &params,
        rgmin::Conjugacy::PolakRibiere,
        rgmin::Restart::Never,
    ));
    assert_eq!(lax.status, Status::StepTolerance);
    same_stop(&lax, &tight);
}

#[test]
fn a_shifted_newton_stall_is_step_tolerance_only_when_strict() {
    let lax_control = Control::default().with_maxiter(4).with_gtol(1e-8);
    let lax = minimize_newton(
        &uphill_hess(),
        array![1.0],
        &lax_control,
        NewtonKind::Shifted,
    )
    .unwrap();
    let tight = refused(minimize_newton(
        &uphill_hess(),
        array![1.0],
        &strict(lax_control),
        NewtonKind::Shifted,
    ));
    assert_eq!(lax.status, Status::Budget);
    assert_eq!(tight.status, Status::StepTolerance);
    assert!(!tight.converged);
}

#[test]
fn session_stops_honour_strict_convergence() {
    let quad = quad();
    let rose = rose();

    let zero = Control::default().with_maxiter(0).with_gtol(1e-8);
    let mut solver = Solver::new(Method::Steepest, zero.clone(), 1);
    let mut x = array![1.0];
    let lax = solver.run(&quad, &mut x).unwrap();
    let mut solver = Solver::new(Method::Steepest, strict(zero), 1);
    let mut x = array![1.0];
    let tight = refused(solver.run(&quad, &mut x));
    assert_eq!(lax.status, Status::Budget);
    same_stop(&lax, &tight);

    let spent = Control::default().with_maxiter(3).with_gtol(1e-10);
    let mut solver = Solver::new(Method::Bfgs, spent.clone(), 2);
    let mut x = array![-1.2, 1.0];
    let lax = solver.run(&rose, &mut x).unwrap();
    let mut solver = Solver::new(Method::Bfgs, strict(spent), 2);
    let mut x = array![-1.2, 1.0];
    let tight = refused(solver.run(&rose, &mut x));
    assert_eq!(lax.status, Status::Budget);
    same_stop(&lax, &tight);

    let gauged = Control::default().with_maxiter(0).with_gtol(1e-8);
    let mut solver = Solver::new(Method::Steepest, gauged.clone(), 1);
    let mut x = array![1.0];
    let lax = solver.run_gauged(&quad, &mut x, |_| {}).unwrap();
    let mut solver = Solver::new(Method::Steepest, strict(gauged), 1);
    let mut x = array![1.0];
    let tight = refused(solver.run_gauged(&quad, &mut x, |_| {}));
    assert_eq!(lax.status, Status::Budget);
    same_stop(&lax, &tight);

    let floor = Control::default()
        .with_maxiter(20)
        .with_gtol(1e-8)
        .with_noise_floor(NoiseFloor::new(0.0).with_eps_f(1.0));
    let mut solver = Solver::new(Method::Bfgs, floor.clone(), 1);
    let mut x = array![0.5];
    let lax = solver.run(&quad, &mut x).unwrap();
    let mut solver = Solver::new(Method::Bfgs, strict(floor), 1);
    let mut x = array![0.5];
    let tight = refused(solver.run(&quad, &mut x));
    assert_eq!(lax.status, Status::NoiseFloor);
    same_stop(&lax, &tight);

    let wall = Control::default().with_maxiter(4).with_gtol(1e-8);
    let mut solver = Solver::new(Method::Steepest, wall.clone(), 1);
    let mut x = array![0.0];
    let lax = solver.run(&Wall, &mut x).unwrap();
    let mut solver = Solver::new(Method::Steepest, strict(wall.clone()), 1);
    let mut x = array![0.0];
    let tight = refused(solver.run(&Wall, &mut x));
    assert_eq!(lax.status, Status::Budget);
    assert_eq!(tight.status, Status::StepTolerance);

    let lax = minimize_method_hess(
        &WallHess,
        array![0.0],
        &wall,
        Method::Dogleg,
        LineSearch::default(),
    )
    .unwrap();
    let tight = refused(minimize_method_hess(
        &WallHess,
        array![0.0],
        &strict(wall.clone()),
        Method::Dogleg,
        LineSearch::default(),
    ));
    assert_eq!(lax.status, Status::Budget);
    assert_eq!(tight.status, Status::StepTolerance);

    let lax = minimize_method_hvp(
        &WallHess,
        array![0.0],
        &wall,
        Method::NewtonCg,
        LineSearch::default(),
    )
    .unwrap();
    let tight = refused(minimize_method_hvp(
        &WallHess,
        array![0.0],
        &strict(wall),
        Method::NewtonCg,
        LineSearch::default(),
    ));
    assert_eq!(lax.status, Status::Budget);
    assert_eq!(tight.status, Status::StepTolerance);
}

/// Fill runs are finished runs, so they honour the flag too.
#[test]
fn fill_runs_honour_strict_convergence() {
    let control = Control::default().with_maxiter(0).with_gtol(1e-8);
    let fill = |_: ArrayView1<f64>, g: &mut Array1<f64>| {
        g.fill(1.0);
        0.5
    };
    for method in [Method::lbfgs(), Method::fire(rgmin::FireKind::V1)] {
        let mut solver = Solver::new(method.clone(), control.clone(), 3);
        let mut x = array![1.0, 0.0, 0.0];
        let lax = solver.run_fill(&mut x, fill).unwrap();
        assert!(!lax.converged, "{method:?}");
        let mut solver = Solver::new(method.clone(), strict(control.clone()), 3);
        let mut x = array![1.0, 0.0, 0.0];
        let tight = refused(solver.run_fill(&mut x, fill));
        assert_eq!(lax.status, tight.status, "{method:?}");
        assert!(!tight.converged, "{method:?}");
    }
}

/// Both flags: strict convergence wins, and the report still names the stop.
#[test]
fn strict_convergence_wins_over_the_unmet_stop_error() {
    let control = Control::default()
        .with_maxiter(0)
        .with_gtol(1e-8)
        .with_require_convergence(true)
        .with_error_on_unmet(true);
    let err = minimize_sd(&quad(), array![4.0], &control, LineSearch::default()).unwrap_err();
    let Error::NotConverged(report) = err else {
        panic!("expected NotConverged, got {err}");
    };
    assert_eq!(report.stop_reason(), StopReason::MaxIter);
}
