#![cfg(feature = "capi")]

use rgmin::ffi::*;

#[test]
fn optional_highs_policy_setters_report_build_availability() {
    let control = rgmin_control_t {
        maxiter: 100,
        gtol: 1e-8,
        istep: 1.0,
        maxmove: 0.0,
        memory: 10,
    };
    let session = unsafe { rgmin_solver_create(rgmin_method_t::RGMIN_LBFGS, &control, 2) };
    assert!(!session.is_null());
    let expected = if cfg!(feature = "highs") { 0 } else { 1 };
    assert_eq!(
        unsafe {
            rgmin_solver_set_highs_solver(session, rgmin_highs_solver_t::RGMIN_HIGHS_IPM as i32)
        },
        expected
    );
    assert_eq!(
        unsafe {
            rgmin_solver_set_highs_crossover(
                session,
                rgmin_highs_crossover_t::RGMIN_HIGHS_CROSSOVER_OFF as i32,
            )
        },
        expected
    );
    assert_eq!(
        unsafe { rgmin_solver_set_highs_callback(session, None, std::ptr::null_mut()) },
        expected
    );
    assert_eq!(unsafe { rgmin_solver_set_highs_solver(session, -1) }, 1);
    assert_eq!(unsafe { rgmin_solver_set_highs_crossover(session, -1) }, 1);
    assert_eq!(
        unsafe {
            rgmin_solver_set_highs_callback(std::ptr::null_mut(), None, std::ptr::null_mut())
        },
        1
    );
    unsafe { rgmin_solver_free(session) };
}
