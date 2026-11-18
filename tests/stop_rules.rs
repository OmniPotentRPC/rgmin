//! `Control::etol` and `Control::max_evals`: the LAMMPS `minimize etol` and
//! `maxeval` stops, on a quadratic, Rosenbrock and a Morse trimer.

mod common;

use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use eindir_core::objectives::Rosenbrock;
use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, Array2, ArrayView1, array};
use rgmin::{
    Conjugacy, Control, Error, FireKind, HessianObjective, Method, NewtonKind, Report, Restart,
    ScgParams, Solver, Status, minimize_method, minimize_method_hess, minimize_method_hvp,
    minimize_newton, minimize_newton_cg, minimize_pso, minimize_scg,
};
use rgmin::{HessianVector, LineSearch};

use common::{MORSE_R0, morse_value_grad};

/// Counts every evaluation the solver asks of the wrapped objective.
struct Tally<O> {
    inner: O,
    calls: AtomicUsize,
}

impl<O> Tally<O> {
    fn new(inner: O) -> Self {
        Self {
            inner,
            calls: AtomicUsize::new(0),
        }
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    fn bump(&self) {
        self.calls.fetch_add(1, Ordering::SeqCst);
    }
}

impl<O: Objective<f64>> Objective<f64> for Tally<O> {
    fn dim(&self) -> usize {
        Objective::dim(&self.inner)
    }

    fn bounds(&self) -> &Bounds<f64> {
        self.inner.bounds()
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.bump();
        self.inner.eval(x)
    }
}

impl<O: DifferentiableObjective<f64>> Gradient<f64> for Tally<O> {
    fn dim(&self) -> usize {
        Gradient::dim(&self.inner)
    }

    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.bump();
        self.inner.grad(x)
    }
}

impl<O: DifferentiableObjective<f64>> DifferentiableObjective<f64> for Tally<O> {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.bump();
        self.inner.value_and_gradient(x)
    }
}

impl<O: HessianObjective> HessianObjective for Tally<O> {
    fn hessian(&self, x: ArrayView1<f64>) -> Array2<f64> {
        self.inner.hessian(x)
    }
}

impl<O: HessianVector> HessianVector for Tally<O> {
    fn hessian_vector(&self, x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
        self.inner.hessian_vector(x, v)
    }
}

const STIFFNESS: [f64; 4] = [1.0, 4.0, 16.0, 40.0];

/// `sum_i k_i x_i^2 / 2`.
struct Quadratic;

impl Objective<f64> for Quadratic {
    fn dim(&self) -> usize {
        STIFFNESS.len()
    }

    fn bounds(&self) -> &Bounds<f64> {
        static BOX: OnceLock<Bounds<f64>> = OnceLock::new();
        BOX.get_or_init(|| {
            Bounds::new(
                Array1::from_elem(STIFFNESS.len(), -1e6),
                Array1::from_elem(STIFFNESS.len(), 1e6),
                0.0,
            )
        })
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.value_and_gradient(x).0
    }
}

impl Gradient<f64> for Quadratic {
    fn dim(&self) -> usize {
        STIFFNESS.len()
    }

    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.value_and_gradient(x).1
    }
}

impl DifferentiableObjective<f64> for Quadratic {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let value = x.iter().zip(STIFFNESS).map(|(v, k)| 0.5 * k * v * v).sum();
        let grad = x.iter().zip(STIFFNESS).map(|(v, k)| k * v).collect();
        (value, grad)
    }
}

impl HessianObjective for Quadratic {
    fn hessian(&self, _x: ArrayView1<f64>) -> Array2<f64> {
        Array2::from_diag(&Array1::from(STIFFNESS.to_vec()))
    }
}

impl HessianVector for Quadratic {
    fn hessian_vector(&self, _x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
        v.iter().zip(STIFFNESS).map(|(v, k)| k * v).collect()
    }
}

fn quad_start() -> Array1<f64> {
    array![1.0, 1.0, 1.0, 1.0]
}

/// Three Morse atoms, started off the equilateral minimum.
struct MorseTrimer;

impl Objective<f64> for MorseTrimer {
    fn dim(&self) -> usize {
        9
    }

    fn bounds(&self) -> &Bounds<f64> {
        static BOX: OnceLock<Bounds<f64>> = OnceLock::new();
        BOX.get_or_init(|| Bounds::new(Array1::from_elem(9, -1e3), Array1::from_elem(9, 1e3), 0.0))
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.value_and_gradient(x).0
    }
}

impl Gradient<f64> for MorseTrimer {
    fn dim(&self) -> usize {
        9
    }

    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.value_and_gradient(x).1
    }
}

impl DifferentiableObjective<f64> for MorseTrimer {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        morse_value_grad(x, [100.0, 100.0, 100.0])
    }
}

fn morse_start() -> Array1<f64> {
    let r = MORSE_R0 * 1.12;
    array![
        0.0,
        0.0,
        0.0,
        r,
        0.0,
        0.0,
        0.5 * r + 0.2,
        0.866 * r - 0.1,
        0.05
    ]
}

fn rosenbrock_start() -> Array1<f64> {
    array![-1.2, 1.0]
}

fn strict() -> Control {
    Control::default()
        .with_maxiter(5000)
        .with_gtol(1e-13)
        .with_istep(0.05)
}

/// Step a fresh session until a step reports anything but `Iterating`.
fn trace<O>(method: Method, control: &Control, obj: &O, start: Array1<f64>) -> (Solver, Vec<Report>)
where
    O: DifferentiableObjective<f64>,
{
    let mut solver = Solver::new(method, control.clone(), start.len());
    let mut x = start;
    let mut reports = Vec::new();
    for _ in 0..control.maxiter {
        let report = solver.step(obj, &mut x).unwrap();
        let done = report.status != Status::Iterating;
        reports.push(report);
        if done {
            break;
        }
    }
    (solver, reports)
}

fn session_methods() -> Vec<Method> {
    vec![
        Method::lbfgs(),
        Method::polak_ribiere(),
        Method::Steepest,
        Method::Bfgs,
        Method::Sr1,
        Method::Sr2,
        Method::adam(),
        Method::fire(FireKind::V1),
        Method::Bb,
        Method::QuickMin,
        Method::Ode12r,
        Method::diis(),
        Method::NewtonCg,
    ]
}

#[test]
fn etol_ends_a_session_at_the_first_step_the_lammps_test_passes() {
    let etol = 1e-6;
    for method in [Method::lbfgs(), Method::polak_ribiere(), Method::Steepest] {
        let (_, plain) = trace(method.clone(), &strict(), &Quadratic, quad_start());
        let control = strict().with_etol(etol);
        let (_, cut) = trace(method.clone(), &control, &Quadratic, quad_start());
        let last = cut.last().unwrap();
        assert_eq!(last.status, Status::EnergyTolerance, "{method:?}");
        assert!(
            cut.len() < plain.len(),
            "{method:?}: {} vs {}",
            cut.len(),
            plain.len()
        );
        let prior = &cut[cut.len() - 2];
        assert!(
            control.energy_converged(prior.value, last.value),
            "{method:?}"
        );
        for pair in cut[..cut.len() - 1].windows(2) {
            assert!(
                !control.energy_converged(pair[0].value, pair[1].value),
                "{method:?}: the test passed one step earlier"
            );
        }
        assert_eq!(last.steps, cut.len(), "{method:?}");
    }
}

#[test]
fn etol_stops_every_session_method_earlier_than_gtol() {
    for method in session_methods() {
        let (_, plain) = trace(method.clone(), &strict(), &MorseTrimer, morse_start());
        let control = strict().with_etol(1e-4);
        let (_, cut) = trace(method.clone(), &control, &MorseTrimer, morse_start());
        let last = cut.last().unwrap();
        assert_eq!(last.status, Status::EnergyTolerance, "{method:?}: {last:?}");
        assert!(
            cut.len() < plain.len(),
            "{method:?}: {} vs {}",
            cut.len(),
            plain.len()
        );
    }
}

#[test]
fn etol_stops_one_shot_drivers_with_the_status() {
    let etol = 1e-4;
    for method in session_methods() {
        let loose = strict();
        let plain = minimize_method(
            &MorseTrimer,
            morse_start(),
            &loose,
            method.clone(),
            LineSearch::default(),
        )
        .unwrap();
        let cut = minimize_method(
            &MorseTrimer,
            morse_start(),
            &loose.with_etol(etol),
            method.clone(),
            LineSearch::default(),
        )
        .unwrap();
        assert_eq!(cut.status, Status::EnergyTolerance, "{method:?}: {cut:?}");
        assert!(cut.steps < plain.steps, "{method:?}");
        assert!(cut.value >= plain.value - 1e-12, "{method:?}");
    }
}

#[test]
fn etol_stops_rosenbrock_before_gtol() {
    let control = strict().with_maxiter(100_000);
    let obj = Rosenbrock::<2>::new();
    let plain = minimize_method(
        &obj,
        rosenbrock_start(),
        &control,
        Method::lbfgs(),
        LineSearch::default(),
    )
    .unwrap();
    let cut = minimize_method(
        &obj,
        rosenbrock_start(),
        &control.clone().with_etol(1e-6),
        Method::lbfgs(),
        LineSearch::default(),
    )
    .unwrap();
    assert_eq!(cut.status, Status::EnergyTolerance, "{cut:?}");
    assert!(cut.steps < plain.steps, "{} vs {}", cut.steps, plain.steps);
}

#[test]
fn newton_scg_and_newton_cg_drivers_honour_etol() {
    let control = strict().with_etol(1e-3);
    let free = strict();
    let scg = |control: &Control| {
        minimize_scg(
            &MorseTrimer,
            morse_start(),
            control,
            &ScgParams::default(),
            Conjugacy::PolakRibiere,
            Restart::Never,
        )
        .unwrap()
    };
    let cut = scg(&control);
    assert_eq!(cut.status, Status::EnergyTolerance, "scg: {cut:?}");
    assert!(cut.steps < scg(&free).steps);

    let cut = minimize_newton(
        &RosenHess,
        rosenbrock_start(),
        &control,
        NewtonKind::Shifted,
    )
    .unwrap();
    let full = minimize_newton(&RosenHess, rosenbrock_start(), &free, NewtonKind::Shifted).unwrap();
    assert_eq!(cut.status, Status::EnergyTolerance, "newton: {cut:?}");
    assert!(cut.steps < full.steps);

    let coarse = strict().with_etol(0.2);
    let cut = minimize_newton_cg(&RosenHess, rosenbrock_start(), &coarse).unwrap();
    let full = minimize_newton_cg(&RosenHess, rosenbrock_start(), &free).unwrap();
    assert_eq!(cut.status, Status::EnergyTolerance, "newton-cg: {cut:?}");
    assert!(cut.steps < full.steps);
}

/// Rosenbrock with its analytic Hessian.
struct RosenHess;

impl Objective<f64> for RosenHess {
    fn dim(&self) -> usize {
        2
    }

    fn bounds(&self) -> &Bounds<f64> {
        static BOX: OnceLock<Bounds<f64>> = OnceLock::new();
        BOX.get_or_init(|| Bounds::new(Array1::from_elem(2, -1e6), Array1::from_elem(2, 1e6), 0.0))
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.value_and_gradient(x).0
    }
}

impl Gradient<f64> for RosenHess {
    fn dim(&self) -> usize {
        2
    }

    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.value_and_gradient(x).1
    }
}

impl DifferentiableObjective<f64> for RosenHess {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let (a, b) = (x[0], x[1]);
        (
            (1.0 - a).powi(2) + 100.0 * (b - a * a).powi(2),
            array![
                -2.0 * (1.0 - a) - 400.0 * a * (b - a * a),
                200.0 * (b - a * a)
            ],
        )
    }
}

impl HessianObjective for RosenHess {
    fn hessian(&self, x: ArrayView1<f64>) -> Array2<f64> {
        let (a, b) = (x[0], x[1]);
        array![
            [2.0 + 1200.0 * a * a - 400.0 * b, -400.0 * a],
            [-400.0 * a, 200.0]
        ]
    }
}

impl HessianVector for RosenHess {
    fn hessian_vector(&self, x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
        self.hessian(x).dot(&v)
    }
}

/// The cumulative evaluation count after every step of an unlimited run.
fn evals_per_step(method: &Method) -> Vec<(usize, usize)> {
    let control = strict();
    let obj = Tally::new(MorseTrimer);
    let mut solver = Solver::new(method.clone(), control.clone(), 9);
    let mut x = morse_start();
    let mut spent = Vec::new();
    for _ in 0..control.maxiter {
        let report = solver.step(&obj, &mut x).unwrap();
        assert_eq!(solver.evals(), obj.calls(), "{method:?}");
        spent.push((solver.steps(), solver.evals()));
        if report.status != Status::Iterating {
            break;
        }
    }
    spent
}

#[test]
fn max_evals_stops_at_the_first_step_boundary_with_the_exact_count() {
    for method in [
        Method::lbfgs(),
        Method::polak_ribiere(),
        Method::Steepest,
        Method::Bfgs,
        Method::fire(FireKind::V1),
        Method::QuickMin,
        Method::Bb,
    ] {
        let spent = evals_per_step(&method);
        let (last_step, total) = *spent.last().unwrap();
        assert!(total > 12, "{method:?}: {total}");
        let mut budgets: Vec<usize> = spent
            .iter()
            .take(25)
            .flat_map(|(_, e)| [e.saturating_sub(1).max(1), *e, e + 1])
            .collect();
        budgets.extend([total - 1, total]);
        budgets.retain(|b| *b <= total);
        budgets.dedup();
        for budget in budgets {
            let (steps, evals) = *spent.iter().find(|(_, e)| *e >= budget).unwrap();
            let obj = Tally::new(MorseTrimer);
            let control = strict().with_max_evals(budget);
            let mut solver = Solver::new(method.clone(), control, 9);
            let mut x = morse_start();
            let report = solver.run(&obj, &mut x).unwrap();
            assert_eq!(report.steps, steps, "{method:?} {budget}");
            assert_eq!(report.evals, evals, "{method:?} {budget}");
            assert_eq!(obj.calls(), evals, "{method:?} {budget}");
            if steps < last_step {
                assert_eq!(report.status, Status::EvalBudget, "{method:?} {budget}");
            } else {
                assert_ne!(report.status, Status::EvalBudget, "{method:?} {budget}");
            }
        }
    }
}

#[test]
fn a_report_counts_the_calls_the_oracle_saw() {
    for method in session_methods() {
        for control in [
            strict().with_maxiter(30),
            strict().with_maxiter(30).with_max_evals(20),
            strict().with_maxiter(30).with_etol(1e-3),
        ] {
            let obj = Tally::new(MorseTrimer);
            let report = minimize_method(
                &obj,
                morse_start(),
                &control,
                method.clone(),
                LineSearch::default(),
            )
            .unwrap();
            assert_eq!(report.evals, obj.calls(), "{method:?}: {report:?}");
            assert!(report.evals > 0, "{method:?}");
        }
    }
    let obj = Tally::new(Rosenbrock::<2>::new());
    let report = minimize_pso(
        &obj,
        rosenbrock_start(),
        &strict().with_maxiter(20),
        8,
        0.7,
        1.5,
        1.5,
    )
    .unwrap();
    assert_eq!(report.evals, obj.calls());
    let obj = Tally::new(RosenHess);
    let control = strict().with_maxiter(15);
    for method in [Method::newton(NewtonKind::Shifted), Method::Dogleg] {
        obj.calls.store(0, Ordering::SeqCst);
        let report = minimize_method_hess(
            &obj,
            rosenbrock_start(),
            &control,
            method.clone(),
            LineSearch::default(),
        )
        .unwrap();
        assert_eq!(report.evals, obj.calls(), "{method:?}");
    }
    obj.calls.store(0, Ordering::SeqCst);
    let report = minimize_method_hvp(
        &obj,
        rosenbrock_start(),
        &control,
        Method::NewtonCg,
        LineSearch::default(),
    )
    .unwrap();
    assert_eq!(report.evals, obj.calls());
    obj.calls.store(0, Ordering::SeqCst);
    let report = minimize_newton_cg(&obj, rosenbrock_start(), &control).unwrap();
    assert_eq!(report.evals, obj.calls());
}

#[test]
fn max_evals_ends_every_one_shot_driver_with_eval_budget() {
    let budget = 9;
    let control = strict().with_max_evals(budget);
    for method in session_methods() {
        let obj = Tally::new(MorseTrimer);
        let report = minimize_method(
            &obj,
            morse_start(),
            &control,
            method.clone(),
            LineSearch::default(),
        )
        .unwrap();
        assert_eq!(report.status, Status::EvalBudget, "{method:?}: {report:?}");
        assert!(report.evals >= budget, "{method:?}");
        assert_eq!(report.evals, obj.calls(), "{method:?}");
    }
    let obj = Tally::new(Rosenbrock::<2>::new());
    let report = minimize_pso(&obj, rosenbrock_start(), &control, 8, 0.7, 1.5, 1.5).unwrap();
    assert_eq!(report.status, Status::EvalBudget);
    assert_eq!(report.evals, obj.calls());
    assert!(report.evals >= budget);
    let report = minimize_scg(
        &Tally::new(MorseTrimer),
        morse_start(),
        &control,
        &ScgParams::default(),
        Conjugacy::PolakRibiere,
        Restart::Never,
    )
    .unwrap();
    assert_eq!(report.status, Status::EvalBudget, "scg: {report:?}");
    let report = minimize_newton(
        &Tally::new(RosenHess),
        rosenbrock_start(),
        &control,
        NewtonKind::Shifted,
    )
    .unwrap();
    assert_eq!(report.status, Status::EvalBudget, "newton: {report:?}");
    let report = minimize_newton_cg(&Tally::new(RosenHess), rosenbrock_start(), &control).unwrap();
    assert_eq!(report.status, Status::EvalBudget, "newton-cg: {report:?}");
}

#[test]
fn max_evals_stops_an_unbounded_scg_run() {
    let control = strict().with_maxiter(0).with_max_evals(60);
    let obj = Tally::new(RosenHess);
    let report = minimize_scg(
        &obj,
        rosenbrock_start(),
        &control,
        &ScgParams::default(),
        Conjugacy::PolakRibiere,
        Restart::Never,
    )
    .unwrap();
    assert_eq!(report.status, Status::EvalBudget, "{report:?}");
    assert_eq!(report.evals, obj.calls());
    assert!(report.evals >= 60);
}

#[test]
fn invalid_stops_are_refused_before_any_evaluation() {
    for etol in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let control = strict().with_etol(etol);
        for method in session_methods().into_iter().chain([Method::pso()]) {
            let obj = Tally::new(Rosenbrock::<2>::new());
            let err = minimize_method(
                &obj,
                rosenbrock_start(),
                &control,
                method.clone(),
                LineSearch::default(),
            )
            .unwrap_err();
            assert!(
                matches!(err, Error::InvalidControl { .. }),
                "{method:?}: {err:?}"
            );
            assert_eq!(obj.calls(), 0, "{method:?}");
            let mut solver = Solver::new(method.clone(), control.clone(), 2);
            let err = solver.step(&obj, &mut rosenbrock_start()).unwrap_err();
            assert!(
                matches!(err, Error::InvalidControl { .. }),
                "{method:?}: {err:?}"
            );
            assert_eq!(obj.calls(), 0, "{method:?}");
        }
        let err = minimize_newton(
            &RosenHess,
            rosenbrock_start(),
            &control,
            NewtonKind::Shifted,
        )
        .unwrap_err();
        assert!(matches!(err, Error::InvalidControl { .. }), "{err:?}");
        let mut solver = Solver::new(Method::lbfgs(), strict(), 2);
        assert!(matches!(
            solver.set_etol(Some(etol)),
            Err(Error::InvalidControl { .. })
        ));
    }
    let control = strict().with_max_evals(0);
    let obj = Tally::new(Rosenbrock::<2>::new());
    let err = minimize_method(
        &obj,
        rosenbrock_start(),
        &control,
        Method::lbfgs(),
        LineSearch::default(),
    )
    .unwrap_err();
    assert!(matches!(err, Error::InvalidControl { .. }), "{err:?}");
    assert_eq!(obj.calls(), 0);
    let mut solver = Solver::new(Method::lbfgs(), strict(), 2);
    assert!(matches!(
        solver.set_max_evals(Some(0)),
        Err(Error::InvalidControl { .. })
    ));
}

#[test]
fn a_valid_etol_of_zero_is_accepted_and_never_fires_on_a_moving_run() {
    let control = strict().with_maxiter(40).with_etol(0.0);
    let report = minimize_method(
        &Tally::new(MorseTrimer),
        morse_start(),
        &control,
        Method::lbfgs(),
        LineSearch::default(),
    )
    .unwrap();
    assert_ne!(report.status, Status::EnergyTolerance, "{report:?}");
}

/// Step `solver` `n` times on the Morse trimer from `morse_start` and
/// return the iterate and the value of the last step.
fn warm_up(solver: &mut Solver, n: usize) -> (Array1<f64>, f64) {
    let mut x = morse_start();
    let mut value = f64::NAN;
    for _ in 0..n {
        let report = solver.step(&MorseTrimer, &mut x).unwrap();
        assert_eq!(report.status, Status::Iterating, "{report:?}");
        value = report.value;
    }
    (x, value)
}

/// The least warm-up whose next step is accepted and still iterating,
/// with the energies on either side of that step.
fn first_moving_step(method: &Method) -> (usize, f64, f64) {
    for warm in 1..12 {
        let mut probe = Solver::new(method.clone(), strict(), 9);
        let (mut x, before) = warm_up(&mut probe, warm);
        let report = probe.step(&MorseTrimer, &mut x).unwrap();
        if report.status == Status::Iterating && report.value != before {
            return (warm, before, report.value);
        }
    }
    panic!("{method:?} never took an accepted step");
}

#[test]
fn set_etol_on_a_cached_point_compares_on_the_next_step() {
    // FIRE reads no energy inside its delay after a velocity reset.
    for method in session_methods()
        .into_iter()
        .filter(|method| !matches!(method, Method::Fire { .. }))
    {
        let (warm, before, after) = first_moving_step(&method);
        let mean = 0.5 * (before.abs() + after.abs());
        let etol = 4.0 * (after - before).abs() / mean;
        assert!(
            strict().with_etol(etol).energy_converged(before, after),
            "{method:?}: {before} {after}"
        );

        let mut solver = Solver::new(method.clone(), strict(), 9);
        let (mut x, value) = warm_up(&mut solver, warm);
        assert_eq!(value, before, "{method:?}");
        solver.set_etol(Some(etol)).unwrap();
        let report = solver.step(&MorseTrimer, &mut x).unwrap();
        assert_eq!(report.value, after, "{method:?}");
        assert_eq!(
            report.status,
            Status::EnergyTolerance,
            "{method:?}: the first comparison was deferred a step"
        );
    }
}

#[test]
fn particle_swarm_refuses_etol_with_a_typed_error() {
    let control = strict().with_etol(1e-3);
    let obj = Tally::new(Rosenbrock::<2>::new());
    let err = minimize_pso(&obj, rosenbrock_start(), &control, 8, 0.7, 1.5, 1.5).unwrap_err();
    assert!(matches!(err, Error::StopUnsupported { .. }), "{err:?}");
    let mut solver = Solver::new(Method::pso(), control, 2);
    let err = solver.step(&obj, &mut rosenbrock_start()).unwrap_err();
    assert!(matches!(err, Error::StopUnsupported { .. }), "{err:?}");
    let mut solver = Solver::new(Method::pso(), strict(), 2);
    assert!(matches!(
        solver.set_etol(Some(1e-3)),
        Err(Error::StopUnsupported { .. })
    ));
    assert_eq!(obj.calls(), 0);
}

#[test]
fn the_fill_paths_honour_both_stops() {
    for method in [Method::lbfgs(), Method::fire(FireKind::V1)] {
        let calls = AtomicUsize::new(0);
        let fill = |at: ArrayView1<f64>, grad: &mut Array1<f64>| {
            calls.fetch_add(1, Ordering::SeqCst);
            let (value, g) = MorseTrimer.value_and_gradient(at);
            grad.assign(&g);
            value
        };
        let mut solver = Solver::new(method.clone(), strict().with_max_evals(7), 9);
        let report = solver.run_fill(&mut morse_start(), fill).unwrap();
        assert_eq!(report.status, Status::EvalBudget, "{method:?}: {report:?}");
        assert_eq!(report.evals, calls.load(Ordering::SeqCst), "{method:?}");
        assert!(report.evals >= 7, "{method:?}");

        calls.store(0, Ordering::SeqCst);
        let mut solver = Solver::new(method.clone(), strict().with_etol(1e-4), 9);
        let report = solver.run_fill(&mut morse_start(), fill).unwrap();
        assert_eq!(
            report.status,
            Status::EnergyTolerance,
            "{method:?}: {report:?}"
        );
        assert_eq!(report.evals, calls.load(Ordering::SeqCst), "{method:?}");

        calls.store(0, Ordering::SeqCst);
        let mut solver = Solver::new(method.clone(), strict(), 9);
        let free = solver.run_fill(&mut morse_start(), fill).unwrap();
        assert!(report.steps < free.steps, "{method:?}");
    }
}

#[test]
fn fire_does_not_test_energy_until_the_power_has_stayed_positive_past_n_min() {
    let control = strict().with_etol(1e3);
    let (_, steps) = trace(
        Method::fire(FireKind::V1),
        &control,
        &MorseTrimer,
        morse_start(),
    );
    let last = steps.last().unwrap();
    assert_eq!(last.status, Status::EnergyTolerance, "{last:?}");
    assert!(last.steps > 5, "{last:?}");
    let (_, steps) = trace(Method::lbfgs(), &control, &MorseTrimer, morse_start());
    assert_eq!(steps.last().unwrap().steps, 1);
}

#[test]
fn a_converged_gradient_is_tolerance_not_energy_tolerance() {
    let control = strict().with_etol(1e3);
    let mut solver = Solver::new(Method::lbfgs(), control, 4);
    let mut x = Array1::zeros(4);
    let report = solver.run(&Quadratic, &mut x).unwrap();
    assert_eq!(report.status, Status::Tolerance, "{report:?}");
}
