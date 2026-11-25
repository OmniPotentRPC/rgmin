//! One L-BFGS run, one stop, whichever front end drives it.
//!
//! The objective reports a gradient that points uphill, so no trial
//! accepts a step and the iterate never moves. The one-shot driver, the
//! session driver, the gauged session driver and the C one-shots must all
//! name that run `LineSearchFailed`, not a generic spent budget.

#![cfg(feature = "capi")]

use std::os::raw::c_void;

use dlpk::sys::DLManagedTensorVersioned;
use eindir_core::DifferentiableObjective;
use ndarray::{Array1, ArrayView1, array};
use rgmin::ffi::{
    rgmin_control_t, rgmin_method_t, rgmin_minimize, rgmin_minimize_opts, rgmin_report_t,
    rgmin_status_t, rgmin_stop_t, rgmin_tensor_borrow_cpu_f64, rgmin_tensor_free,
};
use rgmin::{
    Accept, Control, Error, LbfgsStep, LineSearch, Method, Solver, StopReason, minimize_method,
};

const BUDGET: usize = 3;
const START: f64 = 1.0;
/// Outside the +-1e12 box of `Oracle::unbounded`, which clips it.
const OUTSIDE: f64 = 1e13;
const EDGE: f64 = 1e12;

fn uphill() -> impl DifferentiableObjective<f64> {
    rgmin::Oracle::unbounded(1, |x: ArrayView1<f64>| (0.5 * x[0] * x[0], array![-x[0]]))
}

fn control() -> Control {
    Control::default().with_maxiter(BUDGET).with_gtol(1e-8)
}

/// The session the C waist creates, searching the strong Wolfe line like
/// the one-shot driver does. The default capped step takes no search.
fn session() -> Solver {
    let mut solver = Solver::new(Method::lbfgs(), control(), 1);
    solver.set_accept(Accept::Energy);
    solver.set_lbfgs_step(LbfgsStep::Wolfe).unwrap();
    solver.set_geometry_step(false);
    solver
}

unsafe fn point(t: *const DLManagedTensorVersioned) -> *mut f64 {
    let dl = unsafe { &(*t).dl_tensor };
    unsafe { (dl.data as *mut u8).add(dl.byte_offset as usize) as *mut f64 }
}

unsafe extern "C" fn uphill_eval(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
) -> rgmin_status_t {
    let v = unsafe { *point(x) };
    unsafe { *value_out = 0.5 * v * v };
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn uphill_grad(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    g: *mut DLManagedTensorVersioned,
) -> rgmin_status_t {
    let v = unsafe { *point(x) };
    unsafe { *point(g) = -v };
    rgmin_status_t::RGMIN_SUCCESS
}

fn c_control() -> rgmin_control_t {
    rgmin_control_t {
        maxiter: BUDGET,
        gtol: 1e-8,
        istep: 1.0,
        memory: 10,
        maxmove: 0.0,
        noise_eps: 0.0,
        noise_c: 0.0,
        require_convergence: 0,
    }
}

fn c_run(start: f64, error_on_unmet: i32) -> (rgmin_status_t, rgmin_report_t, f64) {
    let mut x = [start];
    let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 1) };
    assert!(!xt.is_null());
    let ctrl = c_control();
    let mut out = rgmin_report_t {
        value: 0.0,
        steps: 0,
        grad_norm: 0.0,
        stop: rgmin_stop_t::RGMIN_STOP_TOLERANCE,
        converged: 0,
    };
    let status = unsafe {
        if error_on_unmet == 0 {
            rgmin_minimize(
                Some(uphill_eval),
                Some(uphill_grad),
                std::ptr::null_mut(),
                xt,
                &ctrl,
                rgmin_method_t::RGMIN_LBFGS,
                &mut out,
            )
        } else {
            rgmin_minimize_opts(
                Some(uphill_eval),
                Some(uphill_grad),
                std::ptr::null_mut(),
                xt,
                &ctrl,
                rgmin_method_t::RGMIN_LBFGS,
                error_on_unmet,
                0,
                &mut out,
            )
        }
    };
    unsafe { rgmin_tensor_free(xt) };
    (status, out, x[0])
}

#[test]
fn every_front_end_names_a_no_move_lbfgs_run_the_same_way() {
    let one_shot = minimize_method(
        &uphill(),
        array![START],
        &control(),
        Method::lbfgs(),
        LineSearch::default(),
    )
    .unwrap();

    let mut x: Array1<f64> = array![START];
    let stepped = session().run(&uphill(), &mut x).unwrap();
    assert_eq!(x[0], START);

    let mut xg: Array1<f64> = array![START];
    let gauged = session().run_gauged(&uphill(), &mut xg, |_| {}).unwrap();
    assert_eq!(xg[0], START);

    for (front_end, report) in [
        ("minimize_method", &one_shot),
        ("Solver::run", &stepped),
        ("Solver::run_gauged", &gauged),
    ] {
        assert_eq!(
            report.stop_reason(),
            StopReason::LineSearchFailed,
            "{front_end}"
        );
        assert!(!report.converged(), "{front_end}");
    }

    let (status, out, moved) = c_run(START, 0);
    assert_eq!(status, rgmin_status_t::RGMIN_SUCCESS);
    assert_eq!(out.stop, rgmin_stop_t::RGMIN_STOP_LINE_SEARCH);
    assert_eq!(moved, START);

    let (status, out, moved) = c_run(START, 1);
    assert_eq!(status, rgmin_status_t::RGMIN_UNMET_STOP);
    assert_eq!(out.stop, rgmin_stop_t::RGMIN_STOP_LINE_SEARCH);
    assert_eq!(moved, START);
}

/// A run that moves is a spent budget on every front end, so the rule
/// names only the runs that never moved.
#[test]
fn a_moving_run_that_spends_its_budget_stays_maxiter_everywhere() {
    let rose = rgmin::Oracle::unbounded(2, |x: ArrayView1<f64>| {
        let (a, b) = (x[0], x[1]);
        (
            (1.0 - a).powi(2) + 100.0 * (b - a * a).powi(2),
            array![
                -2.0 * (1.0 - a) - 400.0 * a * (b - a * a),
                200.0 * (b - a * a)
            ],
        )
    });
    let control = Control::default().with_maxiter(4).with_gtol(1e-12);
    let one_shot = minimize_method(
        &rose,
        array![-1.2, 1.0],
        &control,
        Method::lbfgs(),
        LineSearch::default(),
    )
    .unwrap();
    let mut x: Array1<f64> = array![-1.2, 1.0];
    let session = Solver::new(Method::lbfgs(), control, 2)
        .run(&rose, &mut x)
        .unwrap();
    assert_eq!(one_shot.stop_reason(), StopReason::MaxIter);
    assert_eq!(session.stop_reason(), StopReason::MaxIter);
}

/// A start outside the box is projected onto it before the first step.
/// Every front end measures movement from that projected point, so a run
/// that never leaves the edge is `LineSearchFailed` everywhere, as it is
/// for a start inside the box.
#[test]
fn a_start_outside_the_box_is_measured_from_its_clipped_point_everywhere() {
    let one_shot = minimize_method(
        &uphill(),
        array![OUTSIDE],
        &control(),
        Method::lbfgs(),
        LineSearch::default(),
    )
    .unwrap();

    let mut x: Array1<f64> = array![OUTSIDE];
    let stepped = session().run(&uphill(), &mut x).unwrap();
    assert_eq!(x[0], EDGE);

    let mut xg: Array1<f64> = array![OUTSIDE];
    let gauged = session().run_gauged(&uphill(), &mut xg, |_| {}).unwrap();
    assert_eq!(xg[0], EDGE);

    for (front_end, report) in [
        ("minimize_method", &one_shot),
        ("Solver::run", &stepped),
        ("Solver::run_gauged", &gauged),
    ] {
        assert_eq!(
            report.stop_reason(),
            StopReason::LineSearchFailed,
            "{front_end}"
        );
        assert!(!report.converged(), "{front_end}");
    }

    let (status, out, at) = c_run(OUTSIDE, 0);
    assert_eq!(status, rgmin_status_t::RGMIN_SUCCESS);
    assert_eq!(out.stop, rgmin_stop_t::RGMIN_STOP_LINE_SEARCH);
    assert_eq!(at, EDGE);

    let (status, out, at) = c_run(OUTSIDE, 1);
    assert_eq!(status, rgmin_status_t::RGMIN_UNMET_STOP);
    assert_eq!(out.stop, rgmin_stop_t::RGMIN_STOP_LINE_SEARCH);
    assert_eq!(at, EDGE);
}

/// A gauge that moves the clipped start is a move even when the steps
/// do not.
#[test]
fn a_gauge_move_from_a_clipped_start_is_still_a_move() {
    let mut x: Array1<f64> = array![OUTSIDE];
    let report = session()
        .run_gauged(&uphill(), &mut x, |x| x[0] = EDGE / 2.0)
        .unwrap();
    assert_ne!(report.stop_reason(), StopReason::LineSearchFailed);
}

#[test]
fn session_runs_reject_mismatched_dimensions_before_clipping() {
    let wide = rgmin::Oracle::unbounded(2, |x: ArrayView1<f64>| {
        (x.dot(&x), array![2.0 * x[0], 2.0 * x[1]])
    });
    for gauged in [false, true] {
        let mut point = array![1.0];
        let err = if gauged {
            session().run_gauged(&wide, &mut point, |_| {}).unwrap_err()
        } else {
            session().run(&wide, &mut point).unwrap_err()
        };
        assert!(matches!(err, Error::Dim { got: 2, dim: 1 }));

        let mut point = array![1.0, 2.0];
        let err = if gauged {
            session()
                .run_gauged(&uphill(), &mut point, |_| {})
                .unwrap_err()
        } else {
            session().run(&uphill(), &mut point).unwrap_err()
        };
        assert!(matches!(err, Error::Dim { got: 2, dim: 1 }));
    }
}

#[cfg(feature = "highs")]
#[test]
fn session_box_is_part_of_the_no_move_baseline() {
    fn boxed_session() -> Solver {
        let mut solver = session();
        solver.set_highs(true);
        assert!(solver.set_box(Some(vec![-1.0]), Some(vec![1.0])));
        solver
    }
    let boxed_uphill =
        rgmin::Oracle::unbounded(1, |x: ArrayView1<f64>| (-0.5 * x[0] * x[0], array![x[0]]));

    let mut x = array![2.0];
    let report = boxed_session().run(&boxed_uphill, &mut x).unwrap();
    assert_eq!(x[0], 1.0);
    assert_eq!(report.stop_reason(), StopReason::LineSearchFailed);

    let mut x = array![2.0];
    let report = boxed_session()
        .run_gauged(&boxed_uphill, &mut x, |_| {})
        .unwrap();
    assert_eq!(x[0], 1.0);
    assert_eq!(report.stop_reason(), StopReason::LineSearchFailed);
}
