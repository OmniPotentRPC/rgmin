//! Matrix-free Newton-CG through the C ABI, driven with C callbacks.

#![cfg(feature = "capi")]

use std::ffi::CStr;
use std::os::raw::c_void;

use dlpk::sys::DLManagedTensorVersioned;
use rgmin::ffi::{
    rgmin_control_t, rgmin_evalgrad_fn, rgmin_hvp_fn, rgmin_last_error,
    rgmin_minimize_hvp_precond_sized, rgmin_report_t, rgmin_status_t, rgmin_stop_t,
    rgmin_tensor_borrow_cpu_f64, rgmin_tensor_free, rgmin_vec_precond_fn,
};

/// The one-shot entry at the layout this test file compiles.
#[allow(clippy::too_many_arguments)]
unsafe fn newton(
    evalgrad: Option<rgmin_evalgrad_fn>,
    hvp: Option<rgmin_hvp_fn>,
    precond: Option<rgmin_vec_precond_fn>,
    user: *mut c_void,
    x: *mut DLManagedTensorVersioned,
    ctrl: *const rgmin_control_t,
    out: *mut rgmin_report_t,
) -> rgmin_status_t {
    unsafe {
        rgmin_minimize_hvp_precond_sized(
            evalgrad,
            hvp,
            precond,
            user,
            x,
            ctrl,
            std::mem::size_of::<rgmin_control_t>(),
            0,
            0,
            out,
        )
    }
}

const MAGIC: u64 = 0x5247_4d49_4e5f_4e43;

unsafe fn view<'a>(t: *const DLManagedTensorVersioned) -> &'a [f64] {
    let dl = unsafe { &(*t).dl_tensor };
    let n = unsafe { *dl.shape as usize };
    let p = unsafe { (dl.data as *const u8).add(dl.byte_offset as usize) as *const f64 };
    unsafe { std::slice::from_raw_parts(p, n) }
}

unsafe fn view_mut<'a>(t: *mut DLManagedTensorVersioned) -> &'a mut [f64] {
    let s = unsafe { view(t as *const _) };
    unsafe { std::slice::from_raw_parts_mut(s.as_ptr() as *mut f64, s.len()) }
}

fn last_error() -> String {
    unsafe { CStr::from_ptr(rgmin_last_error()) }
        .to_string_lossy()
        .into_owned()
}

fn control(maxiter: usize, gtol: f64) -> rgmin_control_t {
    rgmin_control_t {
        maxiter,
        gtol,
        istep: 1.0,
        ..rgmin_control_t::default()
    }
}

fn blank_report() -> rgmin_report_t {
    rgmin_report_t {
        value: 0.0,
        steps: 0,
        grad_norm: 0.0,
        stop: rgmin_stop_t::RGMIN_STOP_BUDGET,
        converged: 0,
    }
}

/// Per-solve state behind the `user` pointer: a tag every callback checks
/// and one counter per callback.
struct Probe {
    magic: u64,
    evalgrad_calls: usize,
    hvp_calls: usize,
    precond_calls: usize,
    nonpositive_curvature: usize,
    fail_hvp: bool,
    fail_evalgrad: bool,
    fail_precond: bool,
}

impl Probe {
    fn new() -> Self {
        Probe {
            magic: MAGIC,
            evalgrad_calls: 0,
            hvp_calls: 0,
            precond_calls: 0,
            nonpositive_curvature: 0,
            fail_hvp: false,
            fail_evalgrad: false,
            fail_precond: false,
        }
    }
}

unsafe fn probe<'a>(user: *mut c_void) -> &'a mut Probe {
    let p = unsafe { &mut *(user as *mut Probe) };
    assert_eq!(p.magic, MAGIC, "user pointer did not reach the callback");
    p
}

// Extended Rosenbrock: f = sum_i 100 (x_{i+1} - x_i^2)^2 + (1 - x_i)^2.
unsafe extern "C" fn rosen_evalgrad(
    user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
    g: *mut DLManagedTensorVersioned,
) -> rgmin_status_t {
    let p = unsafe { probe(user) };
    p.evalgrad_calls += 1;
    if p.fail_evalgrad && p.evalgrad_calls >= 2 {
        return rgmin_status_t::RGMIN_INTERNAL_ERROR;
    }
    let x = unsafe { view(x) };
    let g = unsafe { view_mut(g) };
    let n = x.len();
    g.fill(0.0);
    let mut f = 0.0;
    for i in 0..n - 1 {
        let t = x[i + 1] - x[i] * x[i];
        f += 100.0 * t * t + (1.0 - x[i]).powi(2);
        g[i] += -400.0 * x[i] * t - 2.0 * (1.0 - x[i]);
        g[i + 1] += 200.0 * t;
    }
    unsafe { *value_out = f };
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn rosen_hvp(
    user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    v: *const DLManagedTensorVersioned,
    hv: *mut DLManagedTensorVersioned,
) -> rgmin_status_t {
    let p = unsafe { probe(user) };
    p.hvp_calls += 1;
    if p.fail_hvp {
        return rgmin_status_t::RGMIN_INTERNAL_ERROR;
    }
    let x = unsafe { view(x) };
    let v = unsafe { view(v) };
    let hv = unsafe { view_mut(hv) };
    let n = x.len();
    assert_eq!(v.len(), n);
    assert_eq!(hv.len(), n);
    for i in 0..n {
        let mut diag = 0.0;
        if i + 1 < n {
            diag += 1200.0 * x[i] * x[i] - 400.0 * x[i + 1] + 2.0;
        }
        if i > 0 {
            diag += 200.0;
        }
        let mut s = diag * v[i];
        if i + 1 < n {
            s += -400.0 * x[i] * v[i + 1];
        }
        if i > 0 {
            s += -400.0 * x[i - 1] * v[i - 1];
        }
        hv[i] = s;
    }
    let vhv: f64 = v.iter().zip(hv.iter()).map(|(a, b)| a * b).sum();
    if vhv <= 0.0 {
        p.nonpositive_curvature += 1;
    }
    rgmin_status_t::RGMIN_SUCCESS
}

fn rosenbrock_from_start(
    n: usize,
    maxiter: usize,
) -> (Vec<f64>, rgmin_status_t, Probe, rgmin_report_t) {
    let mut x: Vec<f64> = (0..n)
        .map(|i| if i % 2 == 0 { -1.2 } else { 1.0 })
        .collect();
    let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), n) };
    assert!(!xt.is_null());
    let ctrl = control(maxiter, 1e-8);
    let mut out = blank_report();
    let mut pr = Probe::new();
    let st = unsafe {
        newton(
            Some(rosen_evalgrad),
            Some(rosen_hvp),
            None,
            &mut pr as *mut Probe as *mut c_void,
            xt,
            &ctrl,
            &mut out,
        )
    };
    unsafe { rgmin_tensor_free(xt) };
    (x, st, pr, out)
}

fn assert_rosenbrock_solved(n: usize) {
    let (x, st, pr, out) = rosenbrock_from_start(n, 1000);
    assert_eq!(st, rgmin_status_t::RGMIN_SUCCESS, "{}", last_error());
    assert!(out.grad_norm < 1e-8, "grad_norm {}", out.grad_norm);
    assert!(out.value < 1e-14, "value {}", out.value);
    for (i, xi) in x.iter().enumerate() {
        assert!((xi - 1.0).abs() < 1e-6, "x[{i}] = {xi}");
    }
    // The start point sits in a region of negative curvature; the solve
    // follows it to the trust boundary rather than failing.
    assert!(
        pr.nonpositive_curvature > 0,
        "no negative-curvature direction reached the action"
    );
    assert!(
        pr.evalgrad_calls >= out.steps,
        "{} evals",
        pr.evalgrad_calls
    );
}

#[test]
fn rosenbrock_10_converges_through_negative_curvature() {
    assert_rosenbrock_solved(10);
}

#[test]
fn rosenbrock_100_converges_through_negative_curvature() {
    assert_rosenbrock_solved(100);
}

// Diagonal quadratic with d_i spanning 1 .. 1e6.
fn diag(i: usize, n: usize) -> f64 {
    10f64.powf(6.0 * i as f64 / (n - 1) as f64)
}

unsafe extern "C" fn diag_evalgrad(
    user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
    g: *mut DLManagedTensorVersioned,
) -> rgmin_status_t {
    unsafe { probe(user) }.evalgrad_calls += 1;
    let x = unsafe { view(x) };
    let g = unsafe { view_mut(g) };
    let n = x.len();
    let mut f = 0.0;
    for i in 0..n {
        f += 0.5 * diag(i, n) * x[i] * x[i];
        g[i] = diag(i, n) * x[i];
    }
    unsafe { *value_out = f };
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn diag_hvp(
    user: *mut c_void,
    _x: *const DLManagedTensorVersioned,
    v: *const DLManagedTensorVersioned,
    hv: *mut DLManagedTensorVersioned,
) -> rgmin_status_t {
    unsafe { probe(user) }.hvp_calls += 1;
    let v = unsafe { view(v) };
    let hv = unsafe { view_mut(hv) };
    let n = v.len();
    for i in 0..n {
        hv[i] = diag(i, n) * v[i];
    }
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn diag_precond(
    user: *mut c_void,
    r: *const DLManagedTensorVersioned,
    z: *mut DLManagedTensorVersioned,
) -> rgmin_status_t {
    let p = unsafe { probe(user) };
    p.precond_calls += 1;
    if p.fail_precond {
        return rgmin_status_t::RGMIN_UNSUPPORTED_DEVICE;
    }
    let r = unsafe { view(r) };
    let z = unsafe { view_mut(z) };
    let n = r.len();
    for i in 0..n {
        z[i] = r[i] / diag(i, n);
    }
    rgmin_status_t::RGMIN_SUCCESS
}

fn diag_solve(use_precond: bool) -> (Vec<f64>, Probe, rgmin_report_t) {
    let n = 50;
    let mut x = vec![1.0; n];
    let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), n) };
    let ctrl = rgmin_control_t {
        maxiter: 50,
        gtol: 1e-8,
        istep: 1.0e8,
        ..rgmin_control_t::default()
    };
    let mut out = blank_report();
    let mut pr = Probe::new();
    let st = unsafe {
        newton(
            Some(diag_evalgrad),
            Some(diag_hvp),
            if use_precond {
                Some(diag_precond)
            } else {
                None
            },
            &mut pr as *mut Probe as *mut c_void,
            xt,
            &ctrl,
            &mut out,
        )
    };
    unsafe { rgmin_tensor_free(xt) };
    assert_eq!(st, rgmin_status_t::RGMIN_SUCCESS, "{}", last_error());
    (x, pr, out)
}

#[test]
fn the_preconditioner_callback_cuts_the_hessian_actions() {
    let (xi, plain, rep_i) = diag_solve(false);
    let (xp, pre, rep_p) = diag_solve(true);
    for (x, rep) in [(&xi, &rep_i), (&xp, &rep_p)] {
        assert!(rep.grad_norm < 1e-8, "grad_norm {}", rep.grad_norm);
        assert!(x.iter().all(|v| v.abs() < 1e-8));
    }
    assert_eq!(plain.precond_calls, 0);
    assert!(pre.precond_calls >= 1);
    assert!(
        plain.hvp_calls >= 10 * pre.hvp_calls,
        "identity {} actions vs preconditioned {}",
        plain.hvp_calls,
        pre.hvp_calls
    );
}

#[test]
fn the_user_pointer_reaches_all_three_callbacks() {
    // `probe` asserts the tag on every call; the counters show each of
    // the three callbacks ran.
    let (_, pr, _) = diag_solve(true);
    assert!(pr.evalgrad_calls >= 1);
    assert!(pr.hvp_calls >= 1);
    assert!(pr.precond_calls >= 1);
    assert_eq!(pr.magic, MAGIC);
}

#[test]
fn null_callbacks_return_the_error_status_and_set_the_message() {
    let mut x = [0.5, 0.5];
    let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 2) };
    let ctrl = control(10, 1e-8);
    let mut out = blank_report();
    let mut pr = Probe::new();
    let user = &mut pr as *mut Probe as *mut c_void;

    let st = unsafe { newton(None, Some(diag_hvp), None, user, xt, &ctrl, &mut out) };
    assert_eq!(st, rgmin_status_t::RGMIN_INVALID_PARAMETER);
    assert!(
        last_error().contains("evalgrad is NULL"),
        "{}",
        last_error()
    );

    let st = unsafe { newton(Some(diag_evalgrad), None, None, user, xt, &ctrl, &mut out) };
    assert_eq!(st, rgmin_status_t::RGMIN_INVALID_PARAMETER);
    assert!(last_error().contains("hvp is NULL"), "{}", last_error());

    let st = unsafe {
        newton(
            Some(diag_evalgrad),
            Some(diag_hvp),
            None,
            user,
            xt,
            std::ptr::null(),
            &mut out,
        )
    };
    assert_eq!(st, rgmin_status_t::RGMIN_INVALID_PARAMETER);
    assert!(last_error().contains("ctrl/out null"), "{}", last_error());

    assert_eq!((pr.evalgrad_calls, pr.hvp_calls), (0, 0));
    assert_eq!(x, [0.5, 0.5]);
    unsafe { rgmin_tensor_free(xt) };
}

#[test]
fn a_failing_hvp_callback_aborts_with_its_status() {
    let n = 10;
    let start: Vec<f64> = (0..n)
        .map(|i| if i % 2 == 0 { -1.2 } else { 1.0 })
        .collect();
    let mut x = start.clone();
    let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), n) };
    let ctrl = control(100, 1e-8);
    let mut out = blank_report();
    let mut pr = Probe::new();
    pr.fail_hvp = true;
    let st = unsafe {
        newton(
            Some(rosen_evalgrad),
            Some(rosen_hvp),
            None,
            &mut pr as *mut Probe as *mut c_void,
            xt,
            &ctrl,
            &mut out,
        )
    };
    unsafe { rgmin_tensor_free(xt) };
    assert_eq!(st, rgmin_status_t::RGMIN_INTERNAL_ERROR);
    assert!(last_error().contains("hvp callback"), "{}", last_error());
    assert_eq!(
        pr.hvp_calls, 1,
        "callbacks after the failure must be skipped"
    );
    assert_eq!(x, start, "x is untouched on failure");
}

#[test]
fn a_failing_evalgrad_callback_aborts_with_its_status() {
    let n = 10;
    let start: Vec<f64> = (0..n)
        .map(|i| if i % 2 == 0 { -1.2 } else { 1.0 })
        .collect();
    let mut x = start.clone();
    let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), n) };
    let ctrl = control(100, 1e-8);
    let mut out = blank_report();
    let mut pr = Probe::new();
    pr.fail_evalgrad = true;
    let st = unsafe {
        newton(
            Some(rosen_evalgrad),
            Some(rosen_hvp),
            None,
            &mut pr as *mut Probe as *mut c_void,
            xt,
            &ctrl,
            &mut out,
        )
    };
    unsafe { rgmin_tensor_free(xt) };
    assert_eq!(st, rgmin_status_t::RGMIN_INTERNAL_ERROR);
    assert!(
        last_error().contains("evalgrad callback"),
        "{}",
        last_error()
    );
    assert_eq!(
        pr.evalgrad_calls, 2,
        "callbacks after the failure must be skipped"
    );
    assert_eq!(x, start, "x is untouched on failure");
}

#[test]
fn a_failing_precond_callback_aborts_with_its_status() {
    let n = 50;
    let mut x = vec![1.0; n];
    let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), n) };
    let ctrl = control(50, 1e-8);
    let mut out = blank_report();
    let mut pr = Probe::new();
    pr.fail_precond = true;
    let st = unsafe {
        newton(
            Some(diag_evalgrad),
            Some(diag_hvp),
            Some(diag_precond),
            &mut pr as *mut Probe as *mut c_void,
            xt,
            &ctrl,
            &mut out,
        )
    };
    unsafe { rgmin_tensor_free(xt) };
    assert_eq!(st, rgmin_status_t::RGMIN_UNSUPPORTED_DEVICE);
    assert!(
        last_error().contains("precond callback"),
        "{}",
        last_error()
    );
    assert_eq!(
        pr.precond_calls, 1,
        "callbacks after the failure must be skipped"
    );
    assert!(x.iter().all(|v| *v == 1.0), "x is untouched on failure");
}
