//! The HiGHS host callback through the C session waist.

#![cfg(all(feature = "capi", feature = "highs"))]

use std::ffi::CStr;
use std::os::raw::{c_char, c_void};
use std::sync::atomic::{AtomicUsize, Ordering};

use dlpk::sys::DLManagedTensorVersioned;
use rgmin::ffi::{
    rgmin_accept_t, rgmin_control_t, rgmin_highs_crossover_t, rgmin_highs_solver_t,
    rgmin_last_error, rgmin_method_t, rgmin_qn_step_t, rgmin_report_t, rgmin_solver_add_equality,
    rgmin_solver_create, rgmin_solver_free, rgmin_solver_pair_count, rgmin_solver_set_accept,
    rgmin_solver_set_highs, rgmin_solver_set_highs_callback, rgmin_solver_set_highs_crossover,
    rgmin_solver_set_highs_solver, rgmin_solver_set_qn_step, rgmin_solver_set_trust,
    rgmin_solver_step_fg, rgmin_solver_step_hess_fg, rgmin_solver_t, rgmin_status_t, rgmin_stop_t,
    rgmin_tensor_borrow_cpu_f64, rgmin_tensor_free,
};

unsafe fn cpu_f64(t: *const DLManagedTensorVersioned) -> (*mut f64, usize) {
    let dl = unsafe { &(*t).dl_tensor };
    let n = unsafe { *dl.shape as usize };
    let p = unsafe { (dl.data as *mut u8).add(dl.byte_offset as usize) as *mut f64 };
    (p, n)
}

/// `f = x . x / 2`.
unsafe extern "C" fn half_square(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
    g: *mut DLManagedTensorVersioned,
) -> rgmin_status_t {
    let (p, n) = unsafe { cpu_f64(x) };
    let (gp, gn) = unsafe { cpu_f64(g) };
    assert_eq!(n, gn);
    let mut f = 0.0;
    for i in 0..n {
        let xi = unsafe { *p.add(i) };
        f += 0.5 * xi * xi;
        unsafe { *gp.add(i) = xi };
    }
    unsafe { *value_out = f };
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn identity_hess(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    h: *mut DLManagedTensorVersioned,
) -> rgmin_status_t {
    let (_, n) = unsafe { cpu_f64(x) };
    let (hp, hn) = unsafe { cpu_f64(h) };
    assert_eq!(hn, n * n);
    for k in 0..hn {
        unsafe { *hp.add(k) = if k % (n + 1) == 0 { 1.0 } else { 0.0 } };
    }
    rgmin_status_t::RGMIN_SUCCESS
}

/// `f = c . x` with `c` in `user`, and the number of evaluations.
struct Linear {
    gradient: Vec<f64>,
    calls: AtomicUsize,
}

unsafe extern "C" fn linear(
    user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
    g: *mut DLManagedTensorVersioned,
) -> rgmin_status_t {
    let model = unsafe { &*(user as *const Linear) };
    model.calls.fetch_add(1, Ordering::Relaxed);
    let (p, n) = unsafe { cpu_f64(x) };
    let (gp, gn) = unsafe { cpu_f64(g) };
    assert_eq!((n, gn), (model.gradient.len(), model.gradient.len()));
    let mut f = 0.0;
    for (i, ci) in model.gradient.iter().enumerate() {
        f += ci * unsafe { *p.add(i) };
        unsafe { *gp.add(i) = *ci };
    }
    unsafe { *value_out = f };
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn zero_hess(
    _user: *mut c_void,
    _x: *const DLManagedTensorVersioned,
    h: *mut DLManagedTensorVersioned,
) -> rgmin_status_t {
    let (hp, hn) = unsafe { cpu_f64(h) };
    for k in 0..hn {
        unsafe { *hp.add(k) = 0.0 };
    }
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn count_calls(
    _kind: i32,
    _message: *const c_char,
    _interrupt: *mut i32,
    user: *mut c_void,
) {
    unsafe { (*(user as *const AtomicUsize)).fetch_add(1, Ordering::Relaxed) };
}

unsafe extern "C" fn interrupt_polls(
    kind: i32,
    _message: *const c_char,
    interrupt: *mut i32,
    user: *mut c_void,
) {
    if kind == 1 || kind == 2 {
        unsafe {
            (*(user as *const AtomicUsize)).fetch_add(1, Ordering::Relaxed);
            *interrupt = 1;
        }
    }
}

fn control() -> rgmin_control_t {
    rgmin_control_t {
        maxiter: 100,
        gtol: 1e-8,
        istep: 1.0,
        maxmove: 0.0,
        memory: 10,
        noise_eps: 0.0,
        noise_c: 0.0,
    }
}

fn report() -> rgmin_report_t {
    rgmin_report_t {
        value: 0.0,
        steps: 0,
        grad_norm: 0.0,
        stop: rgmin_stop_t::RGMIN_STOP_TOLERANCE,
    }
}

fn ipm_without_crossover(session: *mut rgmin_solver_t) {
    assert_eq!(
        unsafe {
            rgmin_solver_set_highs_solver(session, rgmin_highs_solver_t::RGMIN_HIGHS_IPM as i32)
        },
        0
    );
    assert_eq!(
        unsafe {
            rgmin_solver_set_highs_crossover(
                session,
                rgmin_highs_crossover_t::RGMIN_HIGHS_CROSSOVER_OFF as i32,
            )
        },
        0
    );
}

fn last_error() -> String {
    unsafe { CStr::from_ptr(rgmin_last_error()) }
        .to_string_lossy()
        .into_owned()
}

#[test]
fn the_c_callback_reaches_first_and_second_order_steps_and_clears() {
    for second_order in [false, true] {
        let hits = AtomicUsize::new(0);
        let session = unsafe { rgmin_solver_create(rgmin_method_t::RGMIN_LBFGS, &control(), 2) };
        assert!(!session.is_null());
        unsafe { rgmin_solver_set_accept(session, rgmin_accept_t::RGMIN_ACCEPT_STEP) };
        ipm_without_crossover(session);
        let user = (&hits as *const AtomicUsize).cast_mut().cast();
        assert_eq!(
            unsafe { rgmin_solver_set_highs_callback(session, Some(count_calls), user) },
            0
        );
        assert_eq!(unsafe { rgmin_solver_set_highs(session, 1) }, 0);
        if second_order {
            unsafe { rgmin_solver_set_qn_step(session, rgmin_qn_step_t::RGMIN_QN_NEWTON) };
        }
        let idx = [0usize, 1];
        let coef = [1.0, 1.0];
        assert_eq!(
            unsafe { rgmin_solver_add_equality(session, idx.as_ptr(), coef.as_ptr(), 2, 0.0) },
            0
        );

        let start = [2.0, -1.0];
        let mut x = start;
        let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 2) };
        let step = |out: &mut rgmin_report_t| unsafe {
            if second_order {
                rgmin_solver_step_hess_fg(
                    session,
                    Some(half_square),
                    Some(identity_hess),
                    std::ptr::null_mut(),
                    xt,
                    out,
                )
            } else {
                rgmin_solver_step_fg(session, Some(half_square), std::ptr::null_mut(), xt, out)
            }
        };
        let mut out = report();
        assert_eq!(
            step(&mut out),
            rgmin_status_t::RGMIN_SUCCESS,
            "{}",
            last_error()
        );
        let called = hits.load(Ordering::Relaxed);
        assert!(called > 0, "the C callback did not reach the session QP");
        assert_eq!(out.steps, 1);
        assert!(
            ((x[0] - start[0]) + (x[1] - start[1])).abs() < 1e-7,
            "{x:?}"
        );
        assert!(out.value < 2.5, "{}", out.value);

        assert_eq!(
            unsafe { rgmin_solver_set_highs_callback(session, None, user) },
            0
        );
        x = start;
        let mut out = report();
        assert_eq!(
            step(&mut out),
            rgmin_status_t::RGMIN_SUCCESS,
            "{}",
            last_error()
        );
        assert_eq!(out.steps, 2);
        assert_eq!(
            hits.load(Ordering::Relaxed),
            called,
            "a cleared callback ran"
        );
        assert!((x[0] + x[1] - 1.0).abs() < 1e-7, "{x:?}");
        unsafe {
            rgmin_tensor_free(xt);
            rgmin_solver_free(session);
        }
    }
}

#[test]
fn a_c_interrupt_fails_the_step_and_keeps_the_point() {
    let n = 48;
    let model = Linear {
        gradient: (0..n).map(|j| ((j * 13 + 1) % 19) as f64 - 9.0).collect(),
        calls: AtomicUsize::new(0),
    };
    let polls = AtomicUsize::new(0);
    let session = unsafe { rgmin_solver_create(rgmin_method_t::RGMIN_LBFGS, &control(), n) };
    assert!(!session.is_null());
    unsafe { rgmin_solver_set_qn_step(session, rgmin_qn_step_t::RGMIN_QN_NEWTON) };
    assert_eq!(unsafe { rgmin_solver_set_highs(session, 1) }, 0);
    ipm_without_crossover(session);
    assert_eq!(unsafe { rgmin_solver_set_trust(session, 0.1) }, 0);
    for row in 0..12 {
        let idx: Vec<usize> = (0..n).collect();
        let coef: Vec<f64> = (0..n)
            .map(|j| (((row + 1) * (j + 3)) % 37) as f64 - 18.0)
            .collect();
        assert_eq!(
            unsafe { rgmin_solver_add_equality(session, idx.as_ptr(), coef.as_ptr(), n, 0.0) },
            0
        );
    }
    let user = (&polls as *const AtomicUsize).cast_mut().cast();
    assert_eq!(
        unsafe { rgmin_solver_set_highs_callback(session, Some(interrupt_polls), user) },
        0
    );

    let mut x = vec![0.0; n];
    let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), n) };
    let model_user = (&model as *const Linear).cast_mut().cast();
    let step = |out: &mut rgmin_report_t| unsafe {
        rgmin_solver_step_hess_fg(session, Some(linear), Some(zero_hess), model_user, xt, out)
    };
    let mut out = report();
    assert_eq!(step(&mut out), rgmin_status_t::RGMIN_INVALID_PARAMETER);
    assert!(last_error().starts_with("HiGHS"), "{}", last_error());
    assert!(
        polls.load(Ordering::Relaxed) > 0,
        "the interrupt was not polled"
    );
    assert!(x.iter().all(|&v| v == 0.0), "{x:?}");
    assert_eq!(unsafe { rgmin_solver_pair_count(session) }, 0);
    assert_eq!(model.calls.load(Ordering::Relaxed), 1);

    assert_eq!(
        unsafe { rgmin_solver_set_highs_callback(session, None, std::ptr::null_mut()) },
        0
    );
    let mut out = report();
    assert_eq!(
        step(&mut out),
        rgmin_status_t::RGMIN_SUCCESS,
        "{}",
        last_error()
    );
    assert!(out.value < 0.0, "{}", out.value);
    assert!(x.iter().all(|v| v.abs() <= 0.1 + 1e-7), "{x:?}");
    unsafe {
        rgmin_tensor_free(xt);
        rgmin_solver_free(session);
    }
}
