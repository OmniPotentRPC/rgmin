//! Borrowed error string and const session queries.

#![cfg(feature = "capi")]

use std::ffi::CStr;
use std::ptr;

use rgmin::ffi::{
    rgmin_control_t, rgmin_last_error, rgmin_last_error_message, rgmin_method_from_name,
    rgmin_method_t, rgmin_solver_create, rgmin_solver_dim, rgmin_solver_free, rgmin_solver_steps,
};

#[test]
fn the_error_string_is_borrowed_and_named_twice() {
    let mut out = rgmin_method_t::RGMIN_LBFGS;
    let st = unsafe { rgmin_method_from_name(c"not-a-method".as_ptr(), &mut out) };
    assert_ne!(st, rgmin::ffi::rgmin_status_t::RGMIN_SUCCESS);
    let a = rgmin_last_error();
    let b = rgmin_last_error_message();
    assert!(!a.is_null() && a == b);
    let text = unsafe { CStr::from_ptr(a) }.to_string_lossy();
    assert!(
        text.contains("not-a-method") || text.contains("unknown"),
        "{text}"
    );
}

#[test]
fn dim_and_steps_read_a_const_session() {
    let ctrl = rgmin_control_t {
        maxiter: 4,
        gtol: 1e-8,
        istep: 0.1,
        memory: 5,
        maxmove: 0.2,
        noise_eps: 0.0,
        noise_c: 0.0,
    };
    let solver = unsafe { rgmin_solver_create(rgmin_method_t::RGMIN_LBFGS, &ctrl, 3) };
    assert!(!solver.is_null());
    assert_eq!(unsafe { rgmin_solver_dim(solver) }, 3);
    assert_eq!(unsafe { rgmin_solver_steps(solver) }, 0);
    assert_eq!(unsafe { rgmin_solver_dim(ptr::null()) }, 0);
    assert_eq!(unsafe { rgmin_solver_steps(ptr::null()) }, 0);
    unsafe { rgmin_solver_free(solver) };
}
