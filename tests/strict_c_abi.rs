//! Strict convergence through the C ABI. The default still returns success.

#![cfg(feature = "capi")]

use std::os::raw::c_void;

use dlpk::sys::DLManagedTensorVersioned;
use rgmin::ffi::{
    rgmin_control_t, rgmin_method_t, rgmin_minimize_hess_opts, rgmin_minimize_opts,
    rgmin_minimize_scg, rgmin_report_t, rgmin_scg_params_default, rgmin_status_t, rgmin_stop_t,
    rgmin_tensor_borrow_cpu_f64, rgmin_tensor_free,
};

unsafe fn cpu_f64(t: *const DLManagedTensorVersioned) -> (*const f64, usize) {
    let dl = unsafe { &(*t).dl_tensor };
    let n = unsafe { *dl.shape as usize };
    let p = unsafe { (dl.data as *const u8).add(dl.byte_offset as usize) as *const f64 };
    (p, n)
}

unsafe extern "C" fn quad_eval(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
) -> rgmin_status_t {
    let (p, _) = unsafe { cpu_f64(x) };
    let x0 = unsafe { *p };
    unsafe { *value_out = 0.5 * x0 * x0 };
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn quad_grad(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    g: *mut DLManagedTensorVersioned,
) -> rgmin_status_t {
    let (p, _) = unsafe { cpu_f64(x) };
    let (gp, _) = unsafe { cpu_f64(g as *const _) };
    unsafe { *(gp as *mut f64) = *p };
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn uphill_grad(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    g: *mut DLManagedTensorVersioned,
) -> rgmin_status_t {
    let (p, _) = unsafe { cpu_f64(x) };
    let (gp, _) = unsafe { cpu_f64(g as *const _) };
    unsafe { *(gp as *mut f64) = -*p };
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn saddle_eval(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
) -> rgmin_status_t {
    let (p, _) = unsafe { cpu_f64(x) };
    let x0 = unsafe { *p };
    unsafe { *value_out = -0.5 * x0 * x0 };
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn saddle_hess(
    _user: *mut c_void,
    _x: *const DLManagedTensorVersioned,
    hess: *mut DLManagedTensorVersioned,
) -> rgmin_status_t {
    let (hp, _) = unsafe { cpu_f64(hess as *const _) };
    unsafe { *(hp as *mut f64) = -1.0 };
    rgmin_status_t::RGMIN_SUCCESS
}

const VALUE: f64 = 1_152_921_504_606_846_976.0;

unsafe extern "C" fn flat_eval(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
) -> rgmin_status_t {
    let (p, _) = unsafe { cpu_f64(x) };
    unsafe { *value_out = VALUE + *p };
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn flat_grad(
    _user: *mut c_void,
    _x: *const DLManagedTensorVersioned,
    g: *mut DLManagedTensorVersioned,
) -> rgmin_status_t {
    let (gp, n) = unsafe { cpu_f64(g as *const _) };
    unsafe {
        *(gp as *mut f64) = 1.0;
        if n > 1 {
            *(gp as *mut f64).add(1) = 0.0;
        }
    }
    rgmin_status_t::RGMIN_SUCCESS
}

fn control(maxiter: usize, gtol: f64, noise_eps: f64, strict: bool) -> rgmin_control_t {
    rgmin_control_t {
        maxiter,
        gtol,
        istep: 1.0,
        memory: 0,
        maxmove: 0.0,
        noise_eps,
        noise_c: 0.0,
        require_convergence: i32::from(strict),
    }
}

fn blank() -> rgmin_report_t {
    rgmin_report_t {
        value: 0.0,
        steps: 0,
        grad_norm: 0.0,
        stop: rgmin_stop_t::RGMIN_STOP_TOLERANCE,
        converged: 0,
    }
}

fn run(
    eval: rgmin::ffi::rgmin_eval_fn,
    grad: rgmin::ffi::rgmin_grad_fn,
    x: &mut [f64],
    ctrl: &rgmin_control_t,
    method: rgmin_method_t,
) -> (rgmin_status_t, rgmin_report_t) {
    let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), x.len()) };
    let mut out = blank();
    let status = unsafe {
        rgmin_minimize_opts(
            Some(eval),
            Some(grad),
            std::ptr::null_mut(),
            xt,
            ctrl,
            method,
            0,
            0,
            &mut out,
        )
    };
    unsafe { rgmin_tensor_free(xt) };
    (status, out)
}

fn expect_ok(status: rgmin_status_t, out: &rgmin_report_t, stop: rgmin_stop_t, name: &str) {
    assert_eq!(status, rgmin_status_t::RGMIN_SUCCESS, "{name}");
    assert_eq!(out.stop, stop, "{name}");
}

fn expect_strict(status: rgmin_status_t, out: &rgmin_report_t, stop: rgmin_stop_t, name: &str) {
    assert_eq!(status, rgmin_status_t::RGMIN_NOT_CONVERGED, "{name}");
    assert_eq!(out.stop, stop, "{name}");
    assert_eq!(out.converged, 0, "{name}");
}

#[test]
fn c_tolerance_stays_success() {
    for strict in [false, true] {
        let mut x = [0.0];
        let (status, out) = run(
            quad_eval,
            quad_grad,
            &mut x,
            &control(0, 1e-8, 0.0, strict),
            rgmin_method_t::RGMIN_STEEPEST,
        );
        assert_eq!(status, rgmin_status_t::RGMIN_SUCCESS);
        assert_eq!(out.stop, rgmin_stop_t::RGMIN_STOP_TOLERANCE);
        assert_ne!(out.converged, 0);
    }
}

#[test]
fn c_budget_noise_and_strict_share_the_stop() {
    let mut x = [4.0];
    let (lax, lax_out) = run(
        quad_eval,
        quad_grad,
        &mut x,
        &control(0, 1e-8, 0.0, false),
        rgmin_method_t::RGMIN_BFGS,
    );
    let mut x = [4.0];
    let (tight, tight_out) = run(
        quad_eval,
        quad_grad,
        &mut x,
        &control(0, 1e-8, 0.0, true),
        rgmin_method_t::RGMIN_BFGS,
    );
    expect_ok(lax, &lax_out, rgmin_stop_t::RGMIN_STOP_BUDGET, "budget");
    expect_strict(tight, &tight_out, rgmin_stop_t::RGMIN_STOP_BUDGET, "budget");

    let mut x = [0.5];
    let (lax, lax_out) = run(
        quad_eval,
        quad_grad,
        &mut x,
        &control(0, 1e-8, 1.0, false),
        rgmin_method_t::RGMIN_STEEPEST,
    );
    let mut x = [0.5];
    let (tight, tight_out) = run(
        quad_eval,
        quad_grad,
        &mut x,
        &control(0, 1e-8, 1.0, true),
        rgmin_method_t::RGMIN_STEEPEST,
    );
    expect_ok(lax, &lax_out, rgmin_stop_t::RGMIN_STOP_NOISE_FLOOR, "noise");
    expect_strict(
        tight,
        &tight_out,
        rgmin_stop_t::RGMIN_STOP_NOISE_FLOOR,
        "noise",
    );
}

#[test]
fn c_scg_step_tolerance_errors_only_when_strict() {
    let mut params = rgmin_scg_params_default();
    params.tol_sol = 0.0;
    params.lambda_limit = f64::INFINITY;
    let run_scg = |strict: bool| {
        let mut x = [0.0, 0.0];
        let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 2) };
        let ctrl = control(0, 0.0, 0.0, strict);
        let mut out = blank();
        let status = unsafe {
            rgmin_minimize_scg(
                Some(flat_eval),
                Some(flat_grad),
                None,
                std::ptr::null_mut(),
                xt,
                &ctrl,
                &params,
                &mut out,
            )
        };
        unsafe { rgmin_tensor_free(xt) };
        (status, out)
    };
    let (lax, lax_out) = run_scg(false);
    let (tight, tight_out) = run_scg(true);
    expect_ok(
        lax,
        &lax_out,
        rgmin_stop_t::RGMIN_STOP_STEP_TOLERANCE,
        "scg",
    );
    expect_strict(
        tight,
        &tight_out,
        rgmin_stop_t::RGMIN_STOP_STEP_TOLERANCE,
        "scg",
    );
}

#[test]
fn c_polish_unmet_errors_only_when_strict() {
    let once = |strict: bool| {
        let mut x = [1.0];
        let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 1) };
        let ctrl = control(3, 1e-8, 0.0, strict);
        let mut out = blank();
        let status = unsafe {
            rgmin_minimize_hess_opts(
                Some(saddle_eval),
                Some(uphill_grad),
                Some(saddle_hess),
                std::ptr::null_mut(),
                xt,
                &ctrl,
                rgmin_method_t::RGMIN_NEWTON,
                0,
                1,
                &mut out,
            )
        };
        unsafe { rgmin_tensor_free(xt) };
        (status, out)
    };
    let (lax, lax_out) = once(false);
    let (tight, tight_out) = once(true);
    assert!(
        matches!(
            lax_out.stop,
            rgmin_stop_t::RGMIN_STOP_POLISH_UNMET | rgmin_stop_t::RGMIN_STOP_LINE_SEARCH
        ),
        "polish stop {:?}",
        lax_out.stop
    );
    expect_ok(lax, &lax_out, lax_out.stop, "polish");
    expect_strict(tight, &tight_out, lax_out.stop, "polish");
}

#[test]
fn c_uphill_steepest_stall_is_step_tolerance_only_when_strict() {
    let mut x = [1.0];
    let (lax, lax_out) = run(
        quad_eval,
        uphill_grad,
        &mut x,
        &control(4, 1e-8, 0.0, false),
        rgmin_method_t::RGMIN_STEEPEST,
    );
    let mut x = [1.0];
    let (tight, tight_out) = run(
        quad_eval,
        uphill_grad,
        &mut x,
        &control(4, 1e-8, 0.0, true),
        rgmin_method_t::RGMIN_STEEPEST,
    );
    expect_ok(lax, &lax_out, rgmin_stop_t::RGMIN_STOP_BUDGET, "stall");
    expect_strict(
        tight,
        &tight_out,
        rgmin_stop_t::RGMIN_STOP_STEP_TOLERANCE,
        "stall",
    );
}
