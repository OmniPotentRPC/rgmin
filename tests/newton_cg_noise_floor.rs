//! Trust-region Newton-CG where the objective value has run out of digits.
//!
//! `f(x) = OFFSET + STIFF / 4 * sum x_i^4` is convex with its minimum at
//! the origin. The Newton step is `-x / 3`, so every iteration scales `x`
//! by 2/3 and the quartic by 16/81. Once the quartic is under half a unit
//! in the last place of `OFFSET`, the value is `OFFSET` at the point and at
//! every trial, the actual reduction is exactly zero, and each trial is
//! refused. The gradient there is still of order one.

use ndarray::{Array1, ArrayView1, array};
use rgmin::{
    Control, Error, HessianVector, HvpOracle, IdentityPrecond, Method, NoiseFloor, Solver, Status,
    minimize_newton_cg, minimize_newton_cg_precond,
};

const OFFSET: f64 = 1e8;
const STIFF: f64 = 1e24;
/// One unit in the last place of `OFFSET`, `2^-26`.
const ULP: f64 = 1.0 / 67_108_864.0;

fn value_and_gradient(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let quartic: f64 = x.iter().map(|v| 0.25 * STIFF * v.powi(4)).sum();
    (OFFSET + quartic, x.mapv(|v| STIFF * v.powi(3)))
}

fn offset_quartic() -> impl HessianVector {
    HvpOracle::unbounded(
        3,
        value_and_gradient,
        |x: ArrayView1<f64>, v: ArrayView1<f64>| {
            Array1::from_iter(
                x.iter()
                    .zip(v.iter())
                    .map(|(xi, vi)| 3.0 * STIFF * xi * xi * vi),
            )
        },
    )
}

fn start() -> Array1<f64> {
    array![1e-3, -1e-3, 1e-3]
}

fn start_grad_norm() -> f64 {
    start()
        .iter()
        .map(|v| (STIFF * v.powi(3)).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn control() -> Control {
    Control::default()
        .with_maxiter(200)
        .with_gtol(1e-6)
        .with_istep(1.0)
}

/// A few units in the last place of the value, as the energy noise.
fn value_noise() -> NoiseFloor {
    NoiseFloor::new(0.0).with_eps_f(4.0 * ULP)
}

fn assert_stopped_at_the_floor(report: &rgmin::Report) {
    assert_eq!(report.status, Status::NoiseFloor);
    assert_eq!(report.stop_reason(), rgmin::StopReason::NoiseFloor);
    assert!(!report.converged, "the gradient is not below gtol");
    assert!(
        (report.value - OFFSET).abs() <= 2.0 * ULP,
        "value {:e}",
        report.value
    );
    // The gradient fell by more than twelve orders and is still far
    // above gtol: the value, not the gradient, ended the run.
    assert!(
        report.grad_norm < 1e-12 * start_grad_norm(),
        "grad_norm {:e} from {:e}",
        report.grad_norm,
        start_grad_norm()
    );
    assert!(report.grad_norm > 1e-3, "grad_norm {:e}", report.grad_norm);
    assert_eq!(report.coords.len(), 3);
    for v in report.coords.iter() {
        assert!(v.abs() < 1e-7 && v.abs() > 0.0, "coordinate {v:e}");
    }
}

#[test]
fn without_an_energy_noise_the_radius_collapses() {
    let err = minimize_newton_cg(&offset_quartic(), start(), &control()).unwrap_err();
    assert!(matches!(err, Error::TrustCollapsed { .. }), "{err}");

    // A gradient noise alone does not change that.
    let gradient_only = control().with_noise_floor(NoiseFloor::new(1e-30));
    let err = minimize_newton_cg(&offset_quartic(), start(), &gradient_only).unwrap_err();
    assert!(matches!(err, Error::TrustCollapsed { .. }), "{err}");

    // Neither does an energy noise that is switched off.
    for off in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let floor = NoiseFloor::new(0.0).with_eps_f(off);
        let err = minimize_newton_cg(
            &offset_quartic(),
            start(),
            &control().with_noise_floor(floor),
        )
        .unwrap_err();
        assert!(
            matches!(err, Error::TrustCollapsed { .. }),
            "eps_f {off}: {err}"
        );
    }
}

#[test]
fn an_energy_noise_stops_at_the_last_accepted_point() {
    let collapsed = match minimize_newton_cg(&offset_quartic(), start(), &control()) {
        Err(Error::TrustCollapsed { steps }) => steps,
        other => panic!("expected a collapse, got {other:?}"),
    };
    let with_noise = control().with_noise_floor(value_noise());
    let report = minimize_newton_cg(&offset_quartic(), start(), &with_noise).unwrap();
    assert_stopped_at_the_floor(&report);
    // The stop is the first refused trial, not the last of the shrinks.
    assert!(
        report.steps < collapsed,
        "{} steps against a collapse at {collapsed}",
        report.steps
    );
    // The reported point is one the objective was evaluated at.
    let (value, grad) = value_and_gradient(report.coords.view());
    assert_eq!(value.to_bits(), report.value.to_bits());
    let norm = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
    assert_eq!(norm.to_bits(), report.grad_norm.to_bits());
}

#[test]
fn the_caller_preconditioned_solve_stops_at_the_same_point() {
    let with_noise = control().with_noise_floor(value_noise());
    let plain = minimize_newton_cg(&offset_quartic(), start(), &with_noise).unwrap();
    let pre = minimize_newton_cg_precond(&offset_quartic(), start(), &with_noise, &IdentityPrecond)
        .unwrap();
    assert_stopped_at_the_floor(&pre);
    assert_eq!(plain.steps, pre.steps);
    assert_eq!(plain.grad_norm.to_bits(), pre.grad_norm.to_bits());
    for (u, v) in plain.coords.iter().zip(pre.coords.iter()) {
        assert_eq!(u.to_bits(), v.to_bits());
    }
}

#[test]
fn strict_convergence_still_refuses_the_floor() {
    let strict = control()
        .with_noise_floor(value_noise())
        .with_require_convergence(true);
    match minimize_newton_cg(&offset_quartic(), start(), &strict) {
        Err(Error::NotConverged(report)) => assert_stopped_at_the_floor(&report),
        other => panic!("expected NotConverged, got {other:?}"),
    }
}

#[test]
fn an_accepted_step_under_the_noise_does_not_stop_the_run() {
    // No offset: the value resolves every decrease, each trial is
    // accepted, and the run reaches gtol with predicted reductions far
    // below `c * eps_f` on the way.
    let bowl = HvpOracle::unbounded(
        4,
        |x: ArrayView1<f64>| {
            let f = x
                .iter()
                .enumerate()
                .map(|(i, v)| (i + 1) as f64 * v * v / 2.0)
                .sum();
            let g = Array1::from_iter(x.iter().enumerate().map(|(i, v)| (i + 1) as f64 * v));
            (f, g)
        },
        |_, v: ArrayView1<f64>| {
            Array1::from_iter(v.iter().enumerate().map(|(i, w)| (i + 1) as f64 * w))
        },
    );
    let init = Array1::from_elem(4, 1.0);
    let quiet = control().with_gtol(1e-9);
    let noisy = quiet
        .clone()
        .with_noise_floor(NoiseFloor::new(0.0).with_eps_f(1.0));
    let want = minimize_newton_cg(&bowl, init.clone(), &quiet).unwrap();
    let got = minimize_newton_cg(&bowl, init, &noisy).unwrap();
    assert_eq!(want.status, Status::Tolerance);
    assert_eq!(got.status, Status::Tolerance);
    assert_eq!(want.steps, got.steps);
    assert_eq!(want.grad_norm.to_bits(), got.grad_norm.to_bits());
}

#[test]
fn a_newton_cg_session_reports_the_floor_from_the_refused_step() {
    let obj = offset_quartic();
    let with_noise = control().with_noise_floor(value_noise());
    let mut solver = Solver::new(Method::NewtonCg, with_noise, 3);
    let mut x = start();
    let mut last = None;
    for _ in 0..200 {
        let before = x.clone();
        let report = solver.step_hvp(&obj, &mut x).unwrap();
        if report.status == Status::NoiseFloor {
            assert_eq!(x, before, "the refused trial moved the point");
            last = Some(report);
            break;
        }
        assert_eq!(report.status, Status::Iterating);
    }
    let report = last.expect("the session never reached the floor");
    assert!(
        report.grad_norm < 1e-12 * start_grad_norm(),
        "grad_norm {:e}",
        report.grad_norm
    );
    assert!((report.value - OFFSET).abs() <= 2.0 * ULP);

    // The same session without an energy noise collapses instead.
    let mut solver = Solver::new(Method::NewtonCg, control(), 3);
    let mut x = start();
    let mut collapsed = false;
    for _ in 0..400 {
        match solver.step_hvp(&obj, &mut x) {
            Ok(report) => assert_eq!(report.status, Status::Iterating),
            Err(Error::TrustCollapsed { .. }) => {
                collapsed = true;
                break;
            }
            Err(other) => panic!("{other}"),
        }
    }
    assert!(collapsed, "the session never collapsed");
}
