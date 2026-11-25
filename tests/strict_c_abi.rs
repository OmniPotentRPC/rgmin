//! Strict convergence through the C ABI. The default still returns success.

#![cfg(feature = "capi")]

use std::os::raw::c_void;

use dlpk::sys::DLManagedTensorVersioned;
use rgmin::ffi::{
    rgmin_control_t, rgmin_method_t, rgmin_minimize_hess_opts_sized, rgmin_minimize_opts_sized,
    rgmin_minimize_scg_with_options_sized, rgmin_report_t, rgmin_scg_params_default,
    rgmin_solver_create_sized, rgmin_solver_free, rgmin_solver_set_lbfgs_wolfe,
    rgmin_solver_set_level_set, rgmin_solver_set_require_convergence, rgmin_solver_step,
    rgmin_status_t, rgmin_stop_t, rgmin_tensor_borrow_cpu_f64, rgmin_tensor_free,
};

const CTRL_SIZE: usize = std::mem::size_of::<rgmin_control_t>();

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

unsafe extern "C" fn rose_eval(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
) -> rgmin_status_t {
    let (p, _) = unsafe { cpu_f64(x) };
    let (a, b) = unsafe { (*p, *p.add(1)) };
    unsafe { *value_out = (1.0 - a).powi(2) + 100.0 * (b - a * a).powi(2) };
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn rose_grad(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    g: *mut DLManagedTensorVersioned,
) -> rgmin_status_t {
    let (p, _) = unsafe { cpu_f64(x) };
    let (a, b) = unsafe { (*p, *p.add(1)) };
    let (gp, _) = unsafe { cpu_f64(g as *const _) };
    unsafe {
        *(gp as *mut f64) = -2.0 * (1.0 - a) - 400.0 * a * (b - a * a);
        *(gp as *mut f64).add(1) = 200.0 * (b - a * a);
    }
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn ramp_eval(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
) -> rgmin_status_t {
    let (p, _) = unsafe { cpu_f64(x) };
    unsafe { *value_out = *p };
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn ramp_grad(
    _user: *mut c_void,
    _x: *const DLManagedTensorVersioned,
    g: *mut DLManagedTensorVersioned,
) -> rgmin_status_t {
    let (gp, _) = unsafe { cpu_f64(g as *const _) };
    unsafe { *(gp as *mut f64) = 1.0 };
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
        rgmin_minimize_opts_sized(
            Some(eval),
            Some(grad),
            std::ptr::null_mut(),
            xt,
            ctrl,
            CTRL_SIZE,
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
            rgmin_minimize_scg_with_options_sized(
                Some(flat_eval),
                Some(flat_grad),
                None,
                std::ptr::null_mut(),
                xt,
                &ctrl,
                CTRL_SIZE,
                &params,
                std::ptr::null(),
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
            rgmin_minimize_hess_opts_sized(
                Some(saddle_eval),
                Some(uphill_grad),
                Some(saddle_hess),
                std::ptr::null_mut(),
                xt,
                &ctrl,
                CTRL_SIZE,
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

/// SR1, SR2 and ODE12R reject trials on Rosenbrock and still converge.
/// Strict mode must not refuse what the default run converges.
#[test]
fn c_many_step_runs_converge_identically_in_strict_mode() {
    for (name, method) in [
        ("sr1", rgmin_method_t::RGMIN_SR1),
        ("sr2", rgmin_method_t::RGMIN_SR2),
        ("ode12r", rgmin_method_t::RGMIN_ODE12R),
    ] {
        let mut lax_x = [0.5, 0.5];
        let (lax, lax_out) = run(
            rose_eval,
            rose_grad,
            &mut lax_x,
            &control(3000, 1e-4, 0.0, false),
            method,
        );
        let mut tight_x = [0.5, 0.5];
        let (tight, tight_out) = run(
            rose_eval,
            rose_grad,
            &mut tight_x,
            &control(3000, 1e-4, 0.0, true),
            method,
        );
        expect_ok(lax, &lax_out, rgmin_stop_t::RGMIN_STOP_TOLERANCE, name);
        expect_ok(tight, &tight_out, rgmin_stop_t::RGMIN_STOP_TOLERANCE, name);
        assert!(lax_out.steps > 1, "{name}");
        assert_eq!(lax_out.steps, tight_out.steps, "{name}");
        assert_ne!(lax_out.converged, 0, "{name}");
        assert_ne!(tight_out.converged, 0, "{name}");
        assert_eq!(lax_x.map(f64::to_bits), tight_x.map(f64::to_bits), "{name}");
    }
}

/// A gradient exactly at `gtol` is not below it: the run is a spent
/// budget, the flag is zero, and strict mode refuses it.
#[test]
fn c_a_gradient_exactly_at_gtol_is_not_converged() {
    let mut x = [1.0];
    let (lax, lax_out) = run(
        quad_eval,
        quad_grad,
        &mut x,
        &control(0, 1.0, 0.0, false),
        rgmin_method_t::RGMIN_STEEPEST,
    );
    assert_eq!(lax_out.grad_norm, 1.0);
    expect_ok(lax, &lax_out, rgmin_stop_t::RGMIN_STOP_BUDGET, "at gtol");
    assert_eq!(lax_out.converged, 0);
    let mut x = [1.0];
    let (tight, tight_out) = run(
        quad_eval,
        quad_grad,
        &mut x,
        &control(0, 1.0, 0.0, true),
        rgmin_method_t::RGMIN_STEEPEST,
    );
    expect_strict(
        tight,
        &tight_out,
        rgmin_stop_t::RGMIN_STOP_BUDGET,
        "at gtol",
    );

    let mut x = [f64::from_bits(1.0_f64.to_bits() - 1)];
    let (below, below_out) = run(
        quad_eval,
        quad_grad,
        &mut x,
        &control(0, 1.0, 0.0, true),
        rgmin_method_t::RGMIN_STEEPEST,
    );
    expect_ok(
        below,
        &below_out,
        rgmin_stop_t::RGMIN_STOP_TOLERANCE,
        "below gtol",
    );
    assert_ne!(below_out.converged, 0);
}

fn session_step(
    solver: *mut rgmin::ffi::rgmin_solver_t,
    eval: rgmin::ffi::rgmin_eval_fn,
    grad: rgmin::ffi::rgmin_grad_fn,
    x: &mut [f64],
) -> (rgmin_status_t, rgmin_report_t) {
    let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), x.len()) };
    let mut out = blank();
    let status = unsafe {
        rgmin_solver_step(
            solver,
            Some(eval),
            Some(grad),
            std::ptr::null_mut(),
            xt,
            &mut out,
        )
    };
    unsafe { rgmin_tensor_free(xt) };
    (status, out)
}

/// A step report carries `converged` from the same comparison. A
/// gradient at `gtol` on every step is never converged.
#[test]
fn c_step_report_converged_follows_the_shared_comparison() {
    let ctrl = control(100, 1.0, 0.0, false);
    let solver =
        unsafe { rgmin_solver_create_sized(rgmin_method_t::RGMIN_STEEPEST, &ctrl, CTRL_SIZE, 1) };
    assert!(!solver.is_null());
    let mut x = [0.0];
    for _ in 0..3 {
        let (status, out) = session_step(solver, ramp_eval, ramp_grad, &mut x);
        assert_eq!(status, rgmin_status_t::RGMIN_SUCCESS);
        assert_eq!(out.grad_norm, 1.0);
        assert_eq!(out.converged, 0);
        assert_ne!(out.stop, rgmin_stop_t::RGMIN_STOP_TOLERANCE);
    }
    unsafe { rgmin_solver_free(solver) };

    let ctrl = control(100, 1e-3, 0.0, false);
    let solver =
        unsafe { rgmin_solver_create_sized(rgmin_method_t::RGMIN_STEEPEST, &ctrl, CTRL_SIZE, 1) };
    let mut x = [0.5];
    let mut flags = Vec::new();
    for _ in 0..200 {
        let (status, out) = session_step(solver, quad_eval, quad_grad, &mut x);
        assert_eq!(status, rgmin_status_t::RGMIN_SUCCESS);
        flags.push((
            out.converged != 0,
            out.stop == rgmin_stop_t::RGMIN_STOP_TOLERANCE,
        ));
        if out.stop == rgmin_stop_t::RGMIN_STOP_TOLERANCE {
            assert!(out.grad_norm < 1e-3);
            break;
        }
        assert!(out.grad_norm >= 1e-3);
    }
    unsafe { rgmin_solver_free(solver) };
    let (flag, tolerance) = *flags.last().unwrap();
    assert!(flag && tolerance, "never converged: {}", flags.len());
    assert!(flags[..flags.len() - 1].iter().all(|(flag, _)| !flag));
}

/// `rgmin_solver_set_require_convergence` turns the session stall on:
/// an uphill steepest step that cannot move reports a step tolerance
/// stop, and a session without it does not.
#[test]
fn c_solver_setter_turns_the_session_stall_on() {
    let ctrl = control(100, 1e-8, 0.0, false);
    let solver =
        unsafe { rgmin_solver_create_sized(rgmin_method_t::RGMIN_STEEPEST, &ctrl, CTRL_SIZE, 1) };
    let mut x = [1.0];
    let (status, out) = session_step(solver, quad_eval, uphill_grad, &mut x);
    assert_eq!(status, rgmin_status_t::RGMIN_SUCCESS);
    assert_eq!(out.stop, rgmin_stop_t::RGMIN_STOP_BUDGET);
    assert_eq!(out.converged, 0);

    unsafe { rgmin_solver_set_require_convergence(solver, 1) };
    let (status, out) = session_step(solver, quad_eval, uphill_grad, &mut x);
    assert_eq!(status, rgmin_status_t::RGMIN_SUCCESS);
    assert_eq!(out.stop, rgmin_stop_t::RGMIN_STOP_STEP_TOLERANCE);
    assert_eq!(out.converged, 0);
    assert_eq!(x[0], 1.0);

    unsafe { rgmin_solver_set_require_convergence(solver, 0) };
    let (_, out) = session_step(solver, quad_eval, uphill_grad, &mut x);
    assert_eq!(out.stop, rgmin_stop_t::RGMIN_STOP_BUDGET);
    unsafe { rgmin_solver_free(solver) };
    unsafe { rgmin_solver_set_require_convergence(std::ptr::null_mut(), 1) };

    let ctrl = control(100, 1e-8, 0.0, true);
    for method in [
        rgmin_method_t::RGMIN_STEEPEST,
        rgmin_method_t::RGMIN_POLAK_RIBIERE,
    ] {
        let solver = unsafe { rgmin_solver_create_sized(method, &ctrl, CTRL_SIZE, 1) };
        let mut x = [1.0];
        let (_, out) = session_step(solver, quad_eval, uphill_grad, &mut x);
        assert_eq!(out.stop, rgmin_stop_t::RGMIN_STOP_STEP_TOLERANCE);
        unsafe { rgmin_solver_free(solver) };
    }
}

/// Trust region sessions never stall: a rejected trial changes the
/// radius, so strict mode does not end the step early.
#[test]
fn c_trust_region_session_steps_are_not_stalls() {
    for (name, method) in [
        ("sr1", rgmin_method_t::RGMIN_SR1),
        ("ode12r", rgmin_method_t::RGMIN_ODE12R),
    ] {
        let ctrl = control(100, 1e-8, 0.0, true);
        let solver = unsafe { rgmin_solver_create_sized(method, &ctrl, CTRL_SIZE, 1) };
        let mut x = [1.0];
        for _ in 0..3 {
            let (status, out) = session_step(solver, quad_eval, uphill_grad, &mut x);
            assert_eq!(status, rgmin_status_t::RGMIN_SUCCESS, "{name}");
            assert_ne!(out.stop, rgmin_stop_t::RGMIN_STOP_STEP_TOLERANCE, "{name}");
        }
        unsafe { rgmin_solver_free(solver) };
    }
}

struct LevelSetRetractState {
    drift: bool,
    saw_trial: bool,
    origin_after_trial: usize,
}

unsafe extern "C" fn level_set_retract_eval(
    user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
) -> rgmin_status_t {
    let state = unsafe { &mut *(user as *mut LevelSetRetractState) };
    let (p, _) = unsafe { cpu_f64(x) };
    let x0 = unsafe { *p };
    if state.drift && state.origin_after_trial >= 1 {
        unsafe { *value_out = x0 - 0.25 };
        return rgmin_status_t::RGMIN_SUCCESS;
    }
    if x0 != 0.0 {
        state.saw_trial = true;
    } else if state.saw_trial {
        state.origin_after_trial += 1;
    }
    unsafe { *value_out = 0.0 };
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn level_set_retract_grad(
    user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    g: *mut DLManagedTensorVersioned,
) -> rgmin_status_t {
    let state = unsafe { &*(user as *const LevelSetRetractState) };
    let (p, _) = unsafe { cpu_f64(x) };
    let (gp, _) = unsafe { cpu_f64(g as *const _) };
    let x0 = unsafe { *p };
    let slope = if state.drift && state.origin_after_trial >= 1 && (x0 - 0.25).abs() <= f64::EPSILON
    {
        0.25
    } else {
        1.0
    };
    unsafe { *(gp as *mut f64) = slope };
    rgmin_status_t::RGMIN_SUCCESS
}

#[test]
fn c_level_set_retraction_refreshes_report_converged_flag() {
    let step_with = |drift: bool, gtol: f64| {
        let ctrl = control(10, gtol, 0.0, true);
        let solver =
            unsafe { rgmin_solver_create_sized(rgmin_method_t::RGMIN_LBFGS, &ctrl, CTRL_SIZE, 1) };
        assert!(!solver.is_null());
        assert_eq!(unsafe { rgmin_solver_set_lbfgs_wolfe(solver) }, 0);
        assert_eq!(
            unsafe { rgmin_solver_set_level_set(solver, 0.0) },
            rgmin_status_t::RGMIN_SUCCESS
        );
        unsafe { rgmin_solver_set_require_convergence(solver, 1) };
        let mut state = LevelSetRetractState {
            drift,
            saw_trial: false,
            origin_after_trial: 0,
        };
        let mut x = [0.0];
        let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 1) };
        let mut out = blank();
        let status = unsafe {
            rgmin_solver_step(
                solver,
                Some(level_set_retract_eval),
                Some(level_set_retract_grad),
                (&mut state as *mut LevelSetRetractState).cast(),
                xt,
                &mut out,
            )
        };
        unsafe { rgmin_tensor_free(xt) };
        unsafe { rgmin_solver_free(solver) };
        (status, out, x)
    };

    let (status, converged, x_conv) = step_with(true, 0.5);
    assert_eq!(status, rgmin_status_t::RGMIN_SUCCESS);
    assert_eq!(x_conv, [0.25]);
    assert_eq!(converged.stop, rgmin_stop_t::RGMIN_STOP_TOLERANCE);
    assert!(converged.grad_norm < 0.5);
    assert_eq!(converged.converged, 1);

    let (status, iterating, x_iter) = step_with(true, 0.1);
    assert_eq!(status, rgmin_status_t::RGMIN_SUCCESS);
    assert_eq!(x_iter, [0.25]);
    assert_eq!(iterating.stop, rgmin_stop_t::RGMIN_STOP_BUDGET);
    assert_eq!(iterating.converged, 0);

    let (status, kept, x_kept) = step_with(false, 0.5);
    assert_eq!(status, rgmin_status_t::RGMIN_SUCCESS);
    assert_eq!(x_kept, [0.0]);
    assert_eq!(kept.stop, rgmin_stop_t::RGMIN_STOP_LINE_SEARCH_ROUNDING);
    assert_eq!(kept.converged, 0);
}
