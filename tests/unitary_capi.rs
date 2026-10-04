#![cfg(feature = "capi")]

use dlpk::sys::DLManagedTensorVersioned;
use rgmin::ffi::*;
use std::os::raw::c_void;

unsafe fn cpu_f64(t: *const DLManagedTensorVersioned) -> (*const f64, usize) {
    let dl = unsafe { &(*t).dl_tensor };
    let n = unsafe { *dl.shape as usize };
    let p = unsafe { (dl.data as *const u8).add(dl.byte_offset as usize) as *const f64 };
    (p, n)
}

fn unitary_gram(x: &[f64], n: usize, j: usize, k: usize) -> (f64, f64) {
    let mut re = 0.0;
    let mut im = 0.0;
    for i in 0..n {
        let uj = 2 * (i * n + j);
        let uk = 2 * (i * n + k);
        let a = x[uj];
        let b = x[uj + 1];
        let c = x[uk];
        let d = x[uk + 1];
        re += a * c + b * d;
        im += a * d - b * c;
    }
    (re, im)
}

fn assert_u_star_u_is_identity(x: &[f64], n: usize) {
    assert_eq!(x.len(), 2 * n * n);
    for j in 0..n {
        for k in 0..n {
            let (re, im) = unitary_gram(x, n, j, k);
            let want = if j == k { 1.0 } else { 0.0 };
            assert!(
                (re - want).abs() < 1e-10 && im.abs() < 1e-10,
                "U^* U[{j},{k}] = {re}+{im}i, want {want}; x={x:?}"
            );
        }
    }
}

unsafe extern "C" fn minus_re_tr_eval(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
) -> rgmin_status_t {
    let (p, n) = unsafe { cpu_f64(x) };
    assert_eq!(n, 8);
    unsafe { *value_out = -(*p + *p.add(6)) };
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn minus_re_tr_grad(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    g: *mut DLManagedTensorVersioned,
) -> rgmin_status_t {
    let (_p, n) = unsafe { cpu_f64(x) };
    assert_eq!(n, 8);
    let (gp, gn) = unsafe { cpu_f64(g as *const _) };
    assert_eq!(gn, 8);
    unsafe {
        for i in 0..8 {
            *(gp as *mut f64).add(i) = 0.0;
        }
        *(gp as *mut f64) = -1.0;
        *(gp as *mut f64).add(6) = -1.0;
    }
    rgmin_status_t::RGMIN_SUCCESS
}

#[test]
fn c_abi_unitary_step_stays_on_the_set() {
    use rgmin::ffi::{rgmin_manifold_t, rgmin_solver_set_manifold, rgmin_solver_set_unitary};

    assert_eq!(rgmin_manifold_t::RGMIN_MANIFOLD_UNITARY as i32, 23);

    let ctrl = rgmin_control_t {
        maxiter: 20,
        gtol: 1e-8,
        istep: 0.15,
        memory: 8,
        maxmove: 0.0,
    };
    let session = unsafe { rgmin_solver_create(rgmin_method_t::RGMIN_STEEPEST, &ctrl, 8) };
    assert!(!session.is_null());
    unsafe {
        // Token 23 defaults to U(1); set_unitary selects U(2).
        rgmin_solver_set_manifold(session, rgmin_manifold_t::RGMIN_MANIFOLD_UNITARY);
        rgmin_solver_set_unitary(session, 2);
        rgmin_solver_set_accept(session, rgmin_accept_t::RGMIN_ACCEPT_STEP);
    }

    // Hadamard-like unitary: 1/sqrt(2) * [1, 1; i, -i].
    let s = 0.5_f64.sqrt();
    let mut x = [s, 0.0, s, 0.0, 0.0, s, 0.0, -s];
    assert_u_star_u_is_identity(&x, 2);

    let mut out = rgmin_report_t {
        value: 0.0,
        steps: 0,
        grad_norm: 0.0,
    };
    for _ in 0..20 {
        let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 8) };
        let st = unsafe {
            rgmin_solver_step(
                session,
                Some(minus_re_tr_eval),
                Some(minus_re_tr_grad),
                std::ptr::null_mut(),
                xt,
                &mut out,
            )
        };
        unsafe { rgmin_tensor_free(xt) };
        assert_eq!(st, rgmin_status_t::RGMIN_SUCCESS);
        assert_u_star_u_is_identity(&x, 2);
    }
    unsafe { rgmin_solver_free(session) };
}
