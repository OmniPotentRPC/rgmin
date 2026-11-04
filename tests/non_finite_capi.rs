//! A refused callback gradient must not produce a coordinate probe.

#![cfg(feature = "capi")]

use std::os::raw::c_void;

use dlpk::sys::DLManagedTensorVersioned;
use rgmin::ffi::{
    rgmin_control_t, rgmin_method_t, rgmin_minimize, rgmin_report_t, rgmin_solver_create,
    rgmin_solver_free, rgmin_solver_step, rgmin_status_t, rgmin_stop_t,
    rgmin_tensor_borrow_cpu_f64, rgmin_tensor_free,
};

unsafe extern "C" fn sent_eval(
    user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
) -> rgmin_status_t {
    let sent = unsafe { &mut *(user as *mut (usize, bool)) };
    let dl = unsafe { &(*x).dl_tensor };
    let n = unsafe { *dl.shape as usize };
    let p = unsafe { (dl.data as *const u8).add(dl.byte_offset as usize) as *const f64 };
    sent.0 += 1;
    sent.1 |= (0..n).any(|i| !unsafe { *p.add(i) }.is_finite());
    unsafe { *value_out = 1.0 };
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn refused_grad(
    _user: *mut c_void,
    _x: *const DLManagedTensorVersioned,
    _g: *mut DLManagedTensorVersioned,
) -> rgmin_status_t {
    rgmin_status_t::RGMIN_INTERNAL_ERROR
}

#[test]
fn a_refused_start_gradient_sends_no_nan_coordinate() {
    use rgmin_method_t::*;
    let ctrl = rgmin_control_t {
        maxiter: 50,
        gtol: 1e-10,
        istep: 1.0,
        memory: 0,
        maxmove: 0.0,
        noise_eps: 0.0,
        noise_c: 0.0,
    };
    for method in [
        RGMIN_STEEPEST, RGMIN_POLAK_RIBIERE, RGMIN_BB, RGMIN_BFGS, RGMIN_LBFGS,
        RGMIN_SR1, RGMIN_SR2, RGMIN_FIRE, RGMIN_FIRE2,
    ] {
        let mut x = [1.0, 1.0];
        let mut sent = (0usize, false);
        let user = &mut sent as *mut (usize, bool) as *mut c_void;
        let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 2) };
        let mut out = rgmin_report_t {
            value: 0.0,
            steps: 0,
            grad_norm: 0.0,
            stop: rgmin_stop_t::RGMIN_STOP_BUDGET,
        };
        let st = unsafe {
            let session = rgmin_solver_create(method, &ctrl, 2);
            let st = [
                rgmin_minimize(Some(sent_eval), Some(refused_grad), user, xt, &ctrl, method, &mut out),
                rgmin_solver_step(session, Some(sent_eval), Some(refused_grad), user, xt, &mut out),
            ];
            rgmin_solver_free(session);
            rgmin_tensor_free(xt);
            st
        };
        assert_eq!(st, [rgmin_status_t::RGMIN_INTERNAL_ERROR; 2], "{method:?}");
        assert_eq!((sent, x), ((2, false), [1.0, 1.0]), "{method:?}");
    }
}
