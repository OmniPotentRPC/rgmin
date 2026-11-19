//! Strict convergence is opt in. The default still returns `Ok`, and a
//! strict run that the default run would converge converges identically.

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};
use rgmin::{
    Conjugacy, Control, Error, FireKind, HessianObjective, HessianOracle, Lbfgs, LineSearch,
    Method, NewtonKind, NoiseFloor, Oracle, Report, Restart, ScgParams, Solver, Status, StopReason,
    minimize, minimize_adam, minimize_bfgs, minimize_method, minimize_method_hess, minimize_newton,
    minimize_pso, minimize_scg, minimize_sd, polish_newton,
};

const ROSE_START: [f64; 2] = [0.5, 0.5];
const ROSE_GTOL: f64 = 1e-4;
const ROSE_BUDGET: usize = 3000;

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

/// `f(x) = x` on `[lo, hi]`: the gradient is one and every downhill step
/// runs into the lower wall.
struct Ramp {
    bounds: Bounds<f64>,
}

impl Ramp {
    fn new(lo: f64, hi: f64) -> Self {
        Self {
            bounds: Bounds::new(array![lo], array![hi], 0.0),
        }
    }
}

impl Objective<f64> for Ramp {
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

impl Gradient<f64> for Ramp {
    fn dim(&self) -> usize {
        1
    }
    fn grad(&self, _x: ArrayView1<f64>) -> Array1<f64> {
        array![1.0]
    }
}

impl DifferentiableObjective<f64> for Ramp {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.grad(x))
    }
}

impl HessianObjective for Ramp {
    fn hessian(&self, _x: ArrayView1<f64>) -> ndarray::Array2<f64> {
        array![[1.0]]
    }
}

/// A box of zero width: no step can move.
fn wall() -> Ramp {
    Ramp::new(0.0, 0.0)
}

fn strict(control: &Control) -> Control {
    control.clone().with_require_convergence(true)
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

fn same_run(lax: &Report, tight: &Report) {
    assert_eq!(lax.steps, tight.steps);
    assert_eq!(lax.status, tight.status);
    assert_eq!(lax.stop_reason(), tight.stop_reason());
    assert_eq!(lax.converged, tight.converged);
    assert_eq!(lax.value.to_bits(), tight.value.to_bits());
    assert_eq!(lax.grad_norm.to_bits(), tight.grad_norm.to_bits());
    let lax_bits: Vec<u64> = lax.coords.iter().map(|v| v.to_bits()).collect();
    let tight_bits: Vec<u64> = tight.coords.iter().map(|v| v.to_bits()).collect();
    assert_eq!(lax_bits, tight_bits);
}

fn rose_control() -> Control {
    Control::default()
        .with_maxiter(ROSE_BUDGET)
        .with_gtol(ROSE_GTOL)
}

fn rose_run(method: Method, control: &Control) -> rgmin::Result<Report> {
    minimize_method(
        &rose(),
        array![ROSE_START[0], ROSE_START[1]],
        control,
        method,
        LineSearch::default(),
    )
}

/// A run that converges in default mode converges in strict mode with
/// the same trajectory. A run that does not is refused with the same
/// report. SR1 and SR2 reject trust region trials and ODE12R rejects
/// adaptive steps on the way; none of that is a stall.
#[test]
fn many_step_runs_converge_identically_in_strict_mode() {
    let cases: Vec<(&str, Method, Option<usize>)> = vec![
        ("sr1", Method::Sr1, Some(76)),
        ("sr2", Method::Sr2, Some(34)),
        ("ode12r", Method::Ode12r, Some(558)),
        ("bfgs", Method::Bfgs, None),
        ("lbfgs", Method::lbfgs(), None),
        ("steepest", Method::Steepest, None),
        ("fire", Method::fire(FireKind::V1), None),
        ("bb", Method::Bb, None),
        ("adam", Method::adam(), None),
        ("nlcg", Method::nlcg(Conjugacy::PolakRibiere), None),
    ];
    let control = rose_control();
    for (name, method, steps) in cases {
        let lax = rose_run(method.clone(), &control);
        let tight = rose_run(method, &strict(&control));
        match lax {
            Ok(lax) if lax.converged => {
                let tight = tight.unwrap_or_else(|err| panic!("{name} strict: {err}"));
                assert_eq!(lax.status, Status::Tolerance, "{name}");
                assert!(tight.converged && tight.converged(), "{name}");
                same_run(&lax, &tight);
                if let Some(steps) = steps {
                    assert_eq!(lax.steps, steps, "{name}");
                }
            }
            Ok(lax) => {
                assert!(steps.is_none(), "{name} must converge: {lax:?}");
                let tight = refused(tight);
                assert_eq!(lax.steps, tight.steps, "{name}");
                same_stop(&lax, &tight);
            }
            Err(err) => panic!("{name} default: {err}"),
        }
    }
}

/// SR1 and SR2 also reject trials from the classic far start.
#[test]
fn trust_region_rejections_are_not_stalls() {
    let control = Control::default()
        .with_maxiter(ROSE_BUDGET)
        .with_gtol(ROSE_GTOL);
    for method in [Method::Sr1, Method::Sr2] {
        let lax = minimize_method(
            &rose(),
            array![-1.2, 1.0],
            &control,
            method.clone(),
            LineSearch::default(),
        )
        .unwrap();
        let tight = minimize_method(
            &rose(),
            array![-1.2, 1.0],
            &strict(&control),
            method,
            LineSearch::default(),
        );
        if lax.converged {
            same_run(&lax, &tight.unwrap());
        } else {
            same_stop(&lax, &refused(tight));
        }
    }
}

/// The session runs the same methods and agrees with the one-shot.
#[test]
fn session_runs_converge_identically_in_strict_mode() {
    for (method, steps) in [(Method::Sr1, 76), (Method::Sr2, 34), (Method::Ode12r, 558)] {
        let mut lax_solver = Solver::new(method.clone(), rose_control(), 2);
        let mut lax_x = array![ROSE_START[0], ROSE_START[1]];
        let lax = lax_solver.run(&rose(), &mut lax_x).unwrap();
        let mut tight_solver = Solver::new(method.clone(), strict(&rose_control()), 2);
        let mut tight_x = array![ROSE_START[0], ROSE_START[1]];
        let tight = tight_solver.run(&rose(), &mut tight_x).unwrap();
        assert_eq!(lax.steps, steps, "{method:?}");
        assert!(lax.converged && tight.converged, "{method:?}");
        same_run(&lax, &tight);
    }
}

/// `Control::converged` is the one comparison: equality is not cleared.
#[test]
fn the_gtol_comparison_is_strict_below() {
    let control = Control::default().with_gtol(1.0);
    assert!(!control.converged(1.0));
    assert!(control.converged(1.0 - f64::EPSILON));
    assert!(!control.converged(1.0 + f64::EPSILON));
    assert!(!control.converged(f64::NAN));
    assert!(!control.converged(f64::INFINITY));
    assert_eq!(control.classify(1.0, 1), Status::Iterating);
    assert_eq!(control.classify(1.0 - f64::EPSILON, 1), Status::Tolerance);
    assert!(control.halt(1.0, 1).is_none());
    assert!(control.halt(1.0 - f64::EPSILON, 1).is_some());
}

/// At `grad_norm == gtol` with no budget the run is a spent budget, the
/// flag is false, and strict mode refuses it. The report, the status and
/// the flag agree.
#[test]
fn a_gradient_exactly_at_gtol_is_not_converged() {
    let control = Control::default().with_maxiter(0).with_gtol(1.0);
    let at = |x: f64| array![x];
    let lax = minimize_sd(&quad(), at(1.0), &control, LineSearch::default()).unwrap();
    assert_eq!(lax.grad_norm, 1.0);
    assert_eq!(lax.status, Status::Budget);
    assert_eq!(lax.stop_reason(), StopReason::MaxIter);
    assert!(!lax.converged);
    let tight = refused(minimize_sd(
        &quad(),
        at(1.0),
        &strict(&control),
        LineSearch::default(),
    ));
    same_stop(&lax, &tight);
    assert_eq!(tight.grad_norm, 1.0);

    let below = minimize_sd(
        &quad(),
        at(1.0 - f64::EPSILON),
        &strict(&control),
        LineSearch::default(),
    )
    .unwrap();
    assert_eq!(below.status, Status::Tolerance);
    assert!(below.converged && below.converged());

    for method in [Method::Bfgs, Method::Sr1, Method::Ode12r, Method::lbfgs()] {
        let lax = minimize_method(
            &quad(),
            at(1.0),
            &control,
            method.clone(),
            LineSearch::default(),
        )
        .unwrap();
        assert!(!lax.converged, "{method:?}");
        assert_ne!(lax.status, Status::Tolerance, "{method:?}");
        let tight = refused(minimize_method(
            &quad(),
            at(1.0),
            &strict(&control),
            method.clone(),
            LineSearch::default(),
        ));
        assert_eq!(tight.status, lax.status, "{method:?}");
        assert!(!tight.converged, "{method:?}");
    }

    let mut solver = Solver::new(Method::Steepest, control.clone(), 1);
    let mut x = at(1.0);
    let session = solver.run(&quad(), &mut x).unwrap();
    assert_eq!(session.status, Status::Budget);
    assert!(!session.converged);
    let mut solver = Solver::new(Method::Steepest, strict(&control), 1);
    let mut x = at(1.0);
    let session = refused(solver.run(&quad(), &mut x));
    assert_eq!(session.status, Status::Budget);
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

type Run = Box<dyn Fn(&Control) -> rgmin::Result<Report>>;

type Pair = (&'static str, rgmin::Result<Report>, rgmin::Result<Report>);

/// A zero budget above `gtol` is `MaxIter` either way, and an error only
/// when strict convergence is on.
#[test]
fn a_zero_budget_errors_only_when_strict() {
    let lax_control = Control::default().with_maxiter(0).with_gtol(1e-8);
    let tight_control = strict(&lax_control);
    let start = || array![4.0];
    let calls: Vec<Pair> = vec![
        (
            "bfgs",
            minimize_bfgs(&quad(), start(), &lax_control, LineSearch::default()),
            minimize_bfgs(&quad(), start(), &tight_control, LineSearch::default()),
        ),
        (
            "sd",
            minimize_sd(&quad(), start(), &lax_control, LineSearch::default()),
            minimize_sd(&quad(), start(), &tight_control, LineSearch::default()),
        ),
        (
            "adam",
            minimize_adam(
                &quad(),
                start(),
                &lax_control,
                LineSearch::default(),
                0.9,
                0.999,
                1e-8,
            ),
            minimize_adam(
                &quad(),
                start(),
                &tight_control,
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
                start(),
                &lax_control,
                Conjugacy::PolakRibiere,
                Restart::Never,
                LineSearch::default(),
            ),
            minimize(
                &quad(),
                start(),
                &tight_control,
                Conjugacy::PolakRibiere,
                Restart::Never,
                LineSearch::default(),
            ),
        ),
        (
            "pso",
            minimize_pso(&quad(), start(), &lax_control, 4, 0.5, 1.5, 1.5),
            minimize_pso(&quad(), start(), &tight_control, 4, 0.5, 1.5, 1.5),
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
    let tight_control = strict(&lax_control);
    let wolfe = || LineSearch::wolfe(1e-4, 0.9, 20);
    let start = || array![0.5];
    let pairs: Vec<Pair> = vec![
        (
            "bfgs",
            minimize_bfgs(&quad(), start(), &lax_control, wolfe()),
            minimize_bfgs(&quad(), start(), &tight_control, wolfe()),
        ),
        (
            "sd",
            minimize_sd(&quad(), start(), &lax_control, wolfe()),
            minimize_sd(&quad(), start(), &tight_control, wolfe()),
        ),
        (
            "adam",
            minimize_adam(&quad(), start(), &lax_control, wolfe(), 0.9, 0.999, 1e-8),
            minimize_adam(&quad(), start(), &tight_control, wolfe(), 0.9, 0.999, 1e-8),
        ),
        (
            "cg",
            minimize(
                &quad(),
                start(),
                &lax_control,
                Conjugacy::PolakRibiere,
                Restart::Never,
                wolfe(),
            ),
            minimize(
                &quad(),
                start(),
                &tight_control,
                Conjugacy::PolakRibiere,
                Restart::Never,
                wolfe(),
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

const PINNED_BUDGET: usize = 40;

/// The trust region opens at radius one and shrinks by a quarter on
/// each refused trial until it reaches the floor of `1e-16` after 27
/// trials. The 28th step finds it there and cannot shrink it further.
const SR_FLOOR_STEPS: usize = 28;

fn pinned_control() -> Control {
    Control::default()
        .with_maxiter(PINNED_BUDGET)
        .with_gtol(1e-8)
}

/// A step that cannot move, with every later step the same, ends a
/// strict run on the first step. Without the flag it spends the budget.
fn assert_pinned_stall(name: &str, run: impl Fn(&Control) -> rgmin::Result<Report>) {
    let lax = run(&pinned_control()).unwrap_or_else(|err| panic!("{name} default: {err}"));
    assert_eq!(lax.status, Status::Budget, "{name}");
    assert_eq!(lax.stop_reason(), StopReason::MaxIter, "{name}");
    assert_eq!(lax.steps, PINNED_BUDGET, "{name}");
    let tight = refused(run(&strict(&pinned_control())));
    assert_eq!(tight.status, Status::StepTolerance, "{name}");
    assert_eq!(tight.stop_reason(), StopReason::StepTolerance, "{name}");
    assert_eq!(tight.steps, 1, "{name}");
    assert!(!tight.converged, "{name}");
    assert_eq!(tight.coords[0].to_bits(), lax.coords[0].to_bits(), "{name}");
    assert_eq!(tight.value.to_bits(), lax.value.to_bits(), "{name}");
}

#[test]
fn a_pinned_bfgs_step_is_step_tolerance_only_when_strict() {
    assert_pinned_stall("bfgs", |control| {
        minimize_bfgs(&wall(), array![0.0], control, LineSearch::default())
    });
}

#[test]
fn a_pinned_steepest_step_is_step_tolerance_only_when_strict() {
    assert_pinned_stall("sd", |control| {
        minimize_sd(&wall(), array![0.0], control, LineSearch::default())
    });
}

#[test]
fn a_pinned_adam_step_is_step_tolerance_only_when_strict() {
    assert_pinned_stall("adam", |control| {
        minimize_adam(
            &wall(),
            array![0.0],
            control,
            LineSearch::default(),
            0.9,
            0.999,
            1e-8,
        )
    });
}

#[test]
fn a_pinned_nlcg_step_is_step_tolerance_only_when_strict() {
    assert_pinned_stall("nlcg", |control| {
        minimize(
            &wall(),
            array![0.0],
            control,
            Conjugacy::PolakRibiere,
            Restart::Never,
            LineSearch::default(),
        )
    });
}

/// Newton falls back to a steepest backtrack. When that cannot lower
/// the energy either, nothing changes and the next iteration repeats.
#[test]
fn a_refused_shifted_newton_step_is_step_tolerance_only_when_strict() {
    let lax = minimize_newton(
        &uphill_hess(),
        array![1.0],
        &pinned_control(),
        NewtonKind::Shifted,
    )
    .unwrap();
    let tight = refused(minimize_newton(
        &uphill_hess(),
        array![1.0],
        &strict(&pinned_control()),
        NewtonKind::Shifted,
    ));
    assert_eq!(lax.status, Status::Budget);
    assert_eq!(lax.steps, PINNED_BUDGET);
    assert_eq!(tight.status, Status::StepTolerance);
    assert_eq!(tight.steps, 1);
    assert_eq!(tight.coords[0].to_bits(), 1.0f64.to_bits());
}

/// An SR trust region that can no longer shrink and still cannot move is
/// a stall. Each rejected trial before that shrinks the radius and is
/// not one, so the stop comes only after the radius has reached its
/// floor.
#[test]
fn a_collapsed_sr_trust_region_is_step_tolerance_only_when_strict() {
    for (name, method) in [("sr1", Method::Sr1), ("sr2", Method::Sr2)] {
        let run = |control: &Control| {
            minimize_method(
                &wall(),
                array![0.0],
                control,
                method.clone(),
                LineSearch::default(),
            )
        };
        let lax = run(&pinned_control()).unwrap();
        assert_eq!(lax.status, Status::Budget, "{name}");
        assert_eq!(lax.steps, PINNED_BUDGET, "{name}");
        let tight = refused(run(&strict(&pinned_control())));
        assert_eq!(tight.status, Status::StepTolerance, "{name}");
        assert_eq!(tight.steps, SR_FLOOR_STEPS, "{name}");
    }
}

/// An uphill gradient makes every SR trial a rejection until the radius
/// floor, which is the same stall.
#[test]
fn rejected_sr_trials_end_in_a_stall_at_the_radius_floor() {
    for (name, method) in [("sr1", Method::Sr1), ("sr2", Method::Sr2)] {
        let control = Control::default().with_maxiter(200).with_gtol(1e-8);
        let run = |control: &Control| {
            minimize_method(
                &uphill(),
                array![1.0],
                control,
                method.clone(),
                LineSearch::default(),
            )
        };
        let lax = run(&control).unwrap();
        assert_eq!(lax.status, Status::Budget, "{name}");
        assert_eq!(lax.steps, 200, "{name}");
        let tight = refused(run(&strict(&control)));
        assert_eq!(tight.status, Status::StepTolerance, "{name}");
        assert!(
            tight.steps > 1 && tight.steps < 200,
            "{name}: {}",
            tight.steps
        );
        assert_eq!(tight.coords[0].to_bits(), 1.0f64.to_bits(), "{name}");
    }
}

/// The capped steepest steps walk into the wall in three moves. The
/// first unmoved step opens at half the last accepted length, not at
/// `istep`, so it is not a repeat of the one after it. Only the second
/// unmoved step opens at `istep` and is a stall.
#[test]
fn steepest_descent_waits_for_the_opening_step_to_repeat() {
    let ramp = Ramp::new(0.0, 1.0);
    let control = Control::default()
        .with_maxiter(PINNED_BUDGET)
        .with_gtol(1e-8);
    let lax = minimize_sd(&ramp, array![0.5], &control, LineSearch::default()).unwrap();
    assert_eq!(lax.status, Status::Budget);
    assert_eq!(lax.coords[0], 0.0);
    let tight = refused(minimize_sd(
        &ramp,
        array![0.5],
        &strict(&control),
        LineSearch::default(),
    ));
    assert_eq!(tight.status, Status::StepTolerance);
    assert_eq!(tight.steps, 5);
    assert_eq!(tight.coords[0], 0.0);
}

/// CG keeps its direction between steps, so an unmoved step is a repeat
/// only when the next direction equals this one. Fletcher-Reeves grows
/// it by the previous direction every time, so it never repeats.
#[test]
fn a_growing_cg_direction_is_not_a_stall() {
    let run = |control: &Control| {
        minimize(
            &wall(),
            array![0.0],
            control,
            Conjugacy::FletcherReeves,
            Restart::Never,
            LineSearch::default(),
        )
    };
    let lax = run(&pinned_control()).unwrap();
    assert_eq!(lax.status, Status::Budget);
    assert_eq!(lax.steps, PINNED_BUDGET);
    let tight = refused(run(&strict(&pinned_control())));
    assert_eq!(tight.status, Status::Budget);
    assert_eq!(tight.steps, PINNED_BUDGET);
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
        &strict(&lax_control),
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
        &strict(&lax_control),
        NewtonKind::Shifted,
    ));
    assert!(matches!(
        lax.stop_reason(),
        StopReason::PolishUnmet | StopReason::LineSearchFailed
    ));
    same_stop(&lax, &tight);
}

/// The inner run of a polished solve is quiet: the polish decides the
/// result. A strict flag that reached the inner run would refuse the
/// unpolished point before the polish improved it.
#[test]
fn a_polish_runs_after_a_strict_inner_run_that_stops_short() {
    let bowl = HessianOracle::unbounded(
        2,
        |x: ArrayView1<f64>| {
            (
                0.5 * (x[0] * x[0] + 1.0e4 * x[1] * x[1]),
                array![x[0], 1.0e4 * x[1]],
            )
        },
        |_| array![[1.0, 0.0], [0.0, 1.0e4]],
    );
    let base = Control::default()
        .with_maxiter(4)
        .with_gtol(1e-8)
        .with_backward_polish(true);
    let lax = minimize_method_hess(
        &bowl,
        array![0.05, 0.05],
        &base,
        Method::Steepest,
        LineSearch::default(),
    )
    .unwrap();
    assert!(lax.converged, "the polish should reach gtol: {lax:?}");
    let tight = minimize_method_hess(
        &bowl,
        array![0.05, 0.05],
        &strict(&base),
        Method::Steepest,
        LineSearch::default(),
    )
    .unwrap();
    same_run(&lax, &tight);
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
        Conjugacy::PolakRibiere,
        Restart::Never,
    )
    .unwrap();
    let tight = refused(minimize_scg(
        &flat,
        array![0.0, 0.0],
        &strict(&lax_control),
        &params,
        Conjugacy::PolakRibiere,
        Restart::Never,
    ));
    assert_eq!(lax.status, Status::StepTolerance);
    same_stop(&lax, &tight);
}

/// Session runs: a pinned first-order step ends a strict run at once.
#[test]
fn session_stalls_stop_a_strict_run_only_for_first_order_methods() {
    for (name, method) in [
        ("steepest", Method::Steepest),
        ("bfgs", Method::Bfgs),
        ("adam", Method::adam()),
    ] {
        let mut solver = Solver::new(method.clone(), pinned_control(), 1);
        let mut x = array![0.0];
        let lax = solver.run(&wall(), &mut x).unwrap();
        assert_eq!(lax.status, Status::Budget, "{name}");
        assert_eq!(lax.steps, PINNED_BUDGET, "{name}");
        let mut solver = Solver::new(method, strict(&pinned_control()), 1);
        let mut x = array![0.0];
        let tight = refused(solver.run(&wall(), &mut x));
        assert_eq!(tight.status, Status::StepTolerance, "{name}");
        assert_eq!(tight.steps, 1, "{name}");
        assert!(!tight.converged, "{name}");
    }
}

/// Trust region and adaptive step methods change their radius or step
/// on a refused trial, so a pinned step is not proof of a repeat. A
/// strict run spends the budget as the default does.
#[test]
fn trust_region_methods_are_never_stalls() {
    let control = pinned_control();
    let runs: Vec<(&str, Run)> = vec![
        (
            "dogleg",
            Box::new(|c| {
                minimize_method_hess(
                    &wall(),
                    array![0.0],
                    c,
                    Method::Dogleg,
                    LineSearch::default(),
                )
            }),
        ),
        (
            "ode12r",
            Box::new(|c| {
                minimize_method(
                    &wall(),
                    array![0.0],
                    c,
                    Method::Ode12r,
                    LineSearch::default(),
                )
            }),
        ),
    ];
    for (name, run) in runs {
        let lax = run(&control).unwrap_or_else(|err| panic!("{name} default: {err}"));
        assert_eq!(lax.status, Status::Budget, "{name}");
        let tight = refused(run(&strict(&control)));
        assert_eq!(tight.status, Status::Budget, "{name}");
        assert_eq!(tight.steps, lax.steps, "{name}");
    }
}

#[test]
fn session_stops_honour_strict_convergence() {
    let quad = quad();
    let rose = rose();

    let zero = Control::default().with_maxiter(0).with_gtol(1e-8);
    let mut solver = Solver::new(Method::Steepest, zero.clone(), 1);
    let mut x = array![1.0];
    let lax = solver.run(&quad, &mut x).unwrap();
    let mut solver = Solver::new(Method::Steepest, strict(&zero), 1);
    let mut x = array![1.0];
    let tight = refused(solver.run(&quad, &mut x));
    assert_eq!(lax.status, Status::Budget);
    same_stop(&lax, &tight);

    let spent = Control::default().with_maxiter(3).with_gtol(1e-10);
    let mut solver = Solver::new(Method::Bfgs, spent.clone(), 2);
    let mut x = array![-1.2, 1.0];
    let lax = solver.run(&rose, &mut x).unwrap();
    let mut solver = Solver::new(Method::Bfgs, strict(&spent), 2);
    let mut x = array![-1.2, 1.0];
    let tight = refused(solver.run(&rose, &mut x));
    assert_eq!(lax.status, Status::Budget);
    same_stop(&lax, &tight);

    let gauged = Control::default().with_maxiter(0).with_gtol(1e-8);
    let mut solver = Solver::new(Method::Steepest, gauged.clone(), 1);
    let mut x = array![1.0];
    let lax = solver.run_gauged(&quad, &mut x, |_| {}).unwrap();
    let mut solver = Solver::new(Method::Steepest, strict(&gauged), 1);
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
    let mut solver = Solver::new(Method::Bfgs, strict(&floor), 1);
    let mut x = array![0.5];
    let tight = refused(solver.run(&quad, &mut x));
    assert_eq!(lax.status, Status::NoiseFloor);
    same_stop(&lax, &tight);
}

/// The session report carries the `gtol` flag on every path that builds
/// one, and the flag follows the stop.
#[test]
fn session_reports_carry_the_gtol_flag() {
    let control = Control::default().with_maxiter(200).with_gtol(1e-6);
    let mut solver = Solver::new(Method::Bfgs, control.clone(), 2);
    let mut x = array![-1.2, 1.0];
    let first = solver.step(&rose(), &mut x).unwrap();
    assert_eq!(first.status, Status::Iterating);
    assert!(!first.converged);
    let mut last = first;
    for _ in 0..control.maxiter {
        last = solver.step(&rose(), &mut x).unwrap();
        if last.status != Status::Iterating {
            break;
        }
        assert!(!last.converged);
    }
    assert_eq!(last.status, Status::Tolerance);
    assert!(last.converged);
    assert!(last.grad_norm < control.gtol);

    let mut solver = Solver::new(Method::Steepest, Control::default().with_gtol(1.0), 1);
    let mut x = array![1.0];
    let at = solver.step(&quad(), &mut x).unwrap();
    assert!(at.grad_norm < 1.0 && at.converged || at.grad_norm >= 1.0 && !at.converged);
}

/// `Solver::set_require_convergence` is the session side of the flag.
#[test]
fn the_session_setter_turns_strict_mode_on() {
    let control = Control::default().with_maxiter(0).with_gtol(1e-8);
    let mut solver = Solver::new(Method::Steepest, control, 1);
    let mut x = array![4.0];
    assert!(solver.run(&quad(), &mut x).is_ok());
    solver.set_require_convergence(true);
    let mut x = array![4.0];
    assert!(matches!(
        solver.run(&quad(), &mut x),
        Err(Error::NotConverged(_))
    ));
    solver.set_require_convergence(false);
    let mut x = array![4.0];
    assert!(solver.run(&quad(), &mut x).is_ok());
}

/// Fill runs are finished runs, so they honour the flag too.
#[test]
fn fill_runs_honour_strict_convergence() {
    let control = Control::default().with_maxiter(0).with_gtol(1e-8);
    let fill = |_: ArrayView1<f64>, g: &mut Array1<f64>| {
        g.fill(1.0);
        0.5
    };
    for method in [Method::lbfgs(), Method::fire(FireKind::V1)] {
        let mut solver = Solver::new(method.clone(), control.clone(), 3);
        let mut x = array![1.0, 0.0, 0.0];
        let lax = solver.run_fill(&mut x, fill).unwrap();
        assert!(!lax.converged, "{method:?}");
        let mut solver = Solver::new(method.clone(), strict(&control), 3);
        let mut x = array![1.0, 0.0, 0.0];
        let tight = refused(solver.run_fill(&mut x, fill));
        assert_eq!(lax.status, tight.status, "{method:?}");
        assert!(!tight.converged, "{method:?}");
    }
}

/// Fill runs stop on the same comparison as every other driver: a
/// gradient exactly at `gtol` is not below it.
#[test]
fn fill_runs_use_the_shared_gtol_comparison() {
    let gtol = 1.0;
    let fill = |_: ArrayView1<f64>, g: &mut Array1<f64>| {
        g.fill(0.0);
        g[0] = gtol;
        0.5
    };
    for method in [Method::lbfgs(), Method::fire(FireKind::V1)] {
        let control = Control::default().with_maxiter(2).with_gtol(gtol);
        let mut solver = Solver::new(method.clone(), control, 3);
        let mut x = array![1.0, 0.0, 0.0];
        let report = solver.run_fill(&mut x, fill).unwrap();
        assert_eq!(report.steps, 2, "{method:?}: stopped on equality");
        assert!(!report.converged, "{method:?}");
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

/// A run that stops on the budget with the strict flag off is `Ok` and
/// still carries the flag, false.
#[test]
fn the_default_leaves_every_budget_ok() {
    let control = Control::default().with_maxiter(3).with_gtol(1e-12);
    let report = rose_run(Method::Bfgs, &control).unwrap();
    assert_eq!(report.status, Status::Budget);
    assert!(!report.converged);
    assert!(!Control::default().require_convergence);
    assert!(
        Control::default()
            .with_require_convergence(true)
            .require_convergence
    );
}

/// A session that walks into a wall and then stands still stops once
/// the standing step repeats, not on the first one. Steepest descent
/// takes three capped steps to the wall, BFGS one uncapped step.
#[test]
fn session_stalls_wait_for_a_repeat() {
    let ramp = Ramp::new(0.0, 1.0);
    let control = Control::default()
        .with_maxiter(PINNED_BUDGET)
        .with_gtol(1e-8);
    for (name, method, moves) in [("steepest", Method::Steepest, 3), ("bfgs", Method::Bfgs, 1)] {
        let mut solver = Solver::new(method, strict(&control), 1);
        let mut x = array![0.5];
        let session = refused(solver.run(&ramp, &mut x));
        assert_eq!(session.status, Status::StepTolerance, "{name}");
        assert!(session.steps > moves, "{name}: {}", session.steps);
        assert!(session.steps < PINNED_BUDGET, "{name}: {}", session.steps);
        assert_eq!(x[0], 0.0, "{name}");
    }
}

/// `run_gauged` stops on the first converged step, not at the budget.
#[test]
fn run_gauged_stops_where_it_converges() {
    let control = Control::default().with_maxiter(400).with_gtol(1e-6);
    for strict_on in [false, true] {
        let control = control.clone().with_require_convergence(strict_on);
        let mut solver = Solver::new(Method::Bfgs, control.clone(), 2);
        let mut x = array![-1.2, 1.0];
        let report = solver.run_gauged(&rose(), &mut x, |_| {}).unwrap();
        assert_eq!(report.status, Status::Tolerance, "strict={strict_on}");
        assert!(report.converged, "strict={strict_on}");
        assert!(report.grad_norm < control.gtol);
        assert!(report.steps > 1 && report.steps < control.maxiter);
    }
    let spent = Control::default().with_maxiter(3).with_gtol(1e-6);
    let mut solver = Solver::new(Method::Bfgs, spent.clone(), 2);
    let mut x = array![-1.2, 1.0];
    let lax = solver.run_gauged(&rose(), &mut x, |_| {}).unwrap();
    assert_eq!(lax.status, Status::Budget);
    assert!(!lax.converged);
    let mut solver = Solver::new(Method::Bfgs, strict(&spent), 2);
    let mut x = array![-1.2, 1.0];
    same_stop(&lax, &refused(solver.run_gauged(&rose(), &mut x, |_| {})));
}
