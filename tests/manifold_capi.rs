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

unsafe extern "C" fn cplx_bowl_eval(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
) -> rgmin_status_t {
    let (p, n) = unsafe { cpu_f64(x) };
    assert_eq!(n, 4);
    let mut s = 0.0;
    for i in 0..4 {
        let v = unsafe { *p.add(i) };
        s += v * v;
    }
    unsafe { *value_out = 0.5 * s };
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn cplx_bowl_grad(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    g: *mut DLManagedTensorVersioned,
) -> rgmin_status_t {
    let (p, n) = unsafe { cpu_f64(x) };
    assert_eq!(n, 4);
    let (gp, gn) = unsafe { cpu_f64(g as *const _) };
    assert_eq!(gn, 4);
    for i in 0..4 {
        unsafe { *(gp as *mut f64).add(i) = *p.add(i) };
    }
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn const_bowl_eval(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
) -> rgmin_status_t {
    let (p, n) = unsafe { cpu_f64(x) };
    assert_eq!(n, 3);
    let mut s = 0.0;
    for i in 0..3 {
        let v = unsafe { *p.add(i) };
        s += v * v;
    }
    unsafe { *value_out = 0.5 * s };
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn const_bowl_grad(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    g: *mut DLManagedTensorVersioned,
) -> rgmin_status_t {
    let (p, n) = unsafe { cpu_f64(x) };
    assert_eq!(n, 3);
    let (gp, gn) = unsafe { cpu_f64(g as *const _) };
    assert_eq!(gn, 3);
    for i in 0..3 {
        unsafe { *(gp as *mut f64).add(i) = *p.add(i) };
    }
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn ds_linear_eval(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
) -> rgmin_status_t {
    let (p, n) = unsafe { cpu_f64(x) };
    assert_eq!(n, 4);
    unsafe { *value_out = *p };
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn ds_linear_grad(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    g: *mut DLManagedTensorVersioned,
) -> rgmin_status_t {
    let (_p, n) = unsafe { cpu_f64(x) };
    assert_eq!(n, 4);
    let (gp, gn) = unsafe { cpu_f64(g as *const _) };
    assert_eq!(gn, 4);
    unsafe {
        *(gp as *mut f64) = 1.0;
        *(gp as *mut f64).add(1) = 0.0;
        *(gp as *mut f64).add(2) = 0.0;
        *(gp as *mut f64).add(3) = 0.0;
    }
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn scplx_linear_eval(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
) -> rgmin_status_t {
    let (p, n) = unsafe { cpu_f64(x) };
    assert_eq!(n, 4);
    unsafe { *value_out = *p.add(2) };
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn scplx_linear_grad(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    g: *mut DLManagedTensorVersioned,
) -> rgmin_status_t {
    let (_p, n) = unsafe { cpu_f64(x) };
    assert_eq!(n, 4);
    let (gp, gn) = unsafe { cpu_f64(g as *const _) };
    assert_eq!(gn, 4);
    unsafe {
        *(gp as *mut f64) = 0.0;
        *(gp as *mut f64).add(1) = 0.0;
        *(gp as *mut f64).add(2) = 1.0;
        *(gp as *mut f64).add(3) = 0.0;
    }
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn pos_linear_eval(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
) -> rgmin_status_t {
    let (p, n) = unsafe { cpu_f64(x) };
    assert_eq!(n, 3);
    let mut s = 0.0;
    for i in 0..3 {
        s += unsafe { *p.add(i) };
    }
    unsafe { *value_out = s };
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn pos_linear_grad(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    g: *mut DLManagedTensorVersioned,
) -> rgmin_status_t {
    let (_p, n) = unsafe { cpu_f64(x) };
    assert_eq!(n, 3);
    let (gp, gn) = unsafe { cpu_f64(g as *const _) };
    assert_eq!(gn, 3);
    unsafe {
        *(gp as *mut f64) = 1.0;
        *(gp as *mut f64).add(1) = 1.0;
        *(gp as *mut f64).add(2) = 1.0;
    }
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn centered_linear_eval(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
) -> rgmin_status_t {
    let (p, n) = unsafe { cpu_f64(x) };
    assert_eq!(n, 4);
    let mut s = 0.0;
    for i in 0..4 {
        s += unsafe { *p.add(i) } * (i as f64 + 1.0);
    }
    unsafe { *value_out = s };
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn centered_linear_grad(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    g: *mut DLManagedTensorVersioned,
) -> rgmin_status_t {
    let (_p, n) = unsafe { cpu_f64(x) };
    assert_eq!(n, 4);
    let (gp, gn) = unsafe { cpu_f64(g as *const _) };
    assert_eq!(gn, 4);
    unsafe {
        *(gp as *mut f64) = 1.0;
        *(gp as *mut f64).add(1) = 2.0;
        *(gp as *mut f64).add(2) = 3.0;
        *(gp as *mut f64).add(3) = 4.0;
    }
    rgmin_status_t::RGMIN_SUCCESS
}

#[test]
fn c_abi_euclidean_complex_stays_on_the_set() {
    use rgmin::ffi::{rgmin_manifold_t, rgmin_solver_set_euclidean_complex};
    let ctrl = rgmin_control_t {
        maxiter: 20,
        gtol: 1e-8,
        istep: 0.1,
        memory: 0,
        maxmove: 0.0,
    };
    let session = unsafe { rgmin_solver_create(rgmin_method_t::RGMIN_STEEPEST, &ctrl, 4) };
    assert!(!session.is_null());
    unsafe { rgmin_solver_set_euclidean_complex(session, 2) };
    let mut x = [2.0_f64, 1.0, -1.0, 3.0];
    let mut out = rgmin_report_t {
        value: 0.0,
        steps: 0,
        grad_norm: 0.0,
    };
    let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 4) };
    let st = unsafe {
        rgmin_solver_step(
            session,
            Some(cplx_bowl_eval),
            Some(cplx_bowl_grad),
            std::ptr::null_mut(),
            xt,
            &mut out,
        )
    };
    unsafe { rgmin_tensor_free(xt) };
    assert_eq!(st, rgmin_status_t::RGMIN_SUCCESS);
    assert_eq!(x.len(), 4);
    assert!(x.iter().all(|a| a.is_finite()));
    let fro = x.iter().map(|a| a * a).sum::<f64>().sqrt();
    assert!((fro - 1.0).abs() > 0.5, "must not be the sphere {x:?}");
    let n0 = (x[0] * x[0] + x[1] * x[1]).sqrt();
    let n1 = (x[2] * x[2] + x[3] * x[3]).sqrt();
    assert!((n0 - 1.0).abs() > 0.05, "must not force S^1 {x:?}");
    assert!((n1 - 1.0).abs() > 0.05, "must not force S^1 {x:?}");
    assert_eq!(
        rgmin_manifold_t::RGMIN_MANIFOLD_EUCLIDEAN_COMPLEX as i32,
        16
    );
    assert_eq!(rgmin_manifold_t::RGMIN_MANIFOLD_MW_RIGID as i32, 6);
    assert_eq!(rgmin_manifold_t::RGMIN_MANIFOLD_OBLIQUE as i32, 11);
    unsafe { rgmin_solver_free(session) };
}

#[test]
fn c_abi_constant_stays_on_the_set() {
    use rgmin::ffi::{rgmin_manifold_t, rgmin_solver_set_constant};
    let ctrl = rgmin_control_t {
        maxiter: 20,
        gtol: 1e-8,
        istep: 0.1,
        memory: 0,
        maxmove: 0.0,
    };
    let session = unsafe { rgmin_solver_create(rgmin_method_t::RGMIN_STEEPEST, &ctrl, 3) };
    assert!(!session.is_null());
    unsafe { rgmin_solver_set_constant(session, 3) };
    let start = [1.25_f64, -0.5, 2.0];
    let mut x = start;
    let mut out = rgmin_report_t {
        value: 0.0,
        steps: 0,
        grad_norm: 0.0,
    };
    let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 3) };
    let st = unsafe {
        rgmin_solver_step(
            session,
            Some(const_bowl_eval),
            Some(const_bowl_grad),
            std::ptr::null_mut(),
            xt,
            &mut out,
        )
    };
    unsafe { rgmin_tensor_free(xt) };
    assert_eq!(st, rgmin_status_t::RGMIN_SUCCESS);
    assert_eq!(x.len(), 3);
    for i in 0..3 {
        assert!((x[i] - start[i]).abs() < 1e-15, "left the singleton {x:?}");
    }
    let fro = x.iter().map(|a| a * a).sum::<f64>().sqrt();
    assert!((fro - 1.0).abs() > 0.5, "must not be the sphere {x:?}");
    assert_eq!(rgmin_manifold_t::RGMIN_MANIFOLD_CONSTANT as i32, 17);
    assert_eq!(rgmin_manifold_t::RGMIN_MANIFOLD_MULTINOMIAL_DS as i32, 18);
    assert_eq!(rgmin_manifold_t::RGMIN_MANIFOLD_MULTINOMIAL_SYM as i32, 19);
    assert_eq!(rgmin_manifold_t::RGMIN_MANIFOLD_SPHERE_COMPLEX as i32, 20);
    assert_eq!(
        rgmin_manifold_t::RGMIN_MANIFOLD_EUCLIDEAN_COMPLEX as i32,
        16
    );
    assert_eq!(rgmin_manifold_t::RGMIN_MANIFOLD_MW_RIGID as i32, 6);
    assert_eq!(rgmin_manifold_t::RGMIN_MANIFOLD_OBLIQUE as i32, 11);
    unsafe { rgmin_solver_free(session) };
}

#[test]
fn c_abi_multinomial_ds_stays_on_the_set() {
    use rgmin::ffi::{rgmin_manifold_t, rgmin_solver_set_multinomial_ds};
    let ctrl = rgmin_control_t {
        maxiter: 20,
        gtol: 1e-8,
        istep: 0.1,
        memory: 0,
        maxmove: 0.0,
    };
    let session = unsafe { rgmin_solver_create(rgmin_method_t::RGMIN_STEEPEST, &ctrl, 4) };
    assert!(!session.is_null());
    unsafe { rgmin_solver_set_multinomial_ds(session, 2) };
    let mut x = [0.5_f64, 0.5, 0.5, 0.5];
    let mut out = rgmin_report_t {
        value: 0.0,
        steps: 0,
        grad_norm: 0.0,
    };
    let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 4) };
    let st = unsafe {
        rgmin_solver_step(
            session,
            Some(ds_linear_eval),
            Some(ds_linear_grad),
            std::ptr::null_mut(),
            xt,
            &mut out,
        )
    };
    unsafe { rgmin_tensor_free(xt) };
    assert_eq!(st, rgmin_status_t::RGMIN_SUCCESS);
    assert!(x.iter().all(|&xi| xi > 0.0), "left the interior {x:?}");
    let r0 = x[0] + x[1];
    let r1 = x[2] + x[3];
    let c0 = x[0] + x[2];
    let c1 = x[1] + x[3];
    assert!((r0 - 1.0).abs() < 1e-10, "row0 {r0}");
    assert!((r1 - 1.0).abs() < 1e-10, "row1 {r1}");
    assert!((c0 - 1.0).abs() < 1e-10, "col0 {c0}");
    assert!((c1 - 1.0).abs() < 1e-10, "col1 {c1}");
    assert!(
        (x[0] - 0.5).abs() > 1e-8,
        "expected a retraction step {x:?}"
    );
    let fro = x.iter().map(|a| a * a).sum::<f64>().sqrt();
    assert!((fro - 1.0).abs() > 1e-8, "must not be the sphere {x:?}");
    assert_eq!(rgmin_manifold_t::RGMIN_MANIFOLD_MULTINOMIAL_DS as i32, 18);
    assert_eq!(rgmin_manifold_t::RGMIN_MANIFOLD_MW_RIGID as i32, 6);
    assert_eq!(rgmin_manifold_t::RGMIN_MANIFOLD_OBLIQUE as i32, 11);
    unsafe { rgmin_solver_free(session) };
}

#[test]
fn c_abi_multinomial_sym_stays_on_the_set() {
    use rgmin::ffi::{rgmin_manifold_t, rgmin_solver_set_multinomial_sym};
    let ctrl = rgmin_control_t {
        maxiter: 20,
        gtol: 1e-8,
        istep: 0.1,
        memory: 0,
        maxmove: 0.0,
    };
    let session = unsafe { rgmin_solver_create(rgmin_method_t::RGMIN_STEEPEST, &ctrl, 4) };
    assert!(!session.is_null());
    unsafe { rgmin_solver_set_multinomial_sym(session, 2) };
    let mut x = [0.5_f64, 0.5, 0.5, 0.5];
    let mut out = rgmin_report_t {
        value: 0.0,
        steps: 0,
        grad_norm: 0.0,
    };
    let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 4) };
    let st = unsafe {
        rgmin_solver_step(
            session,
            Some(ds_linear_eval),
            Some(ds_linear_grad),
            std::ptr::null_mut(),
            xt,
            &mut out,
        )
    };
    unsafe { rgmin_tensor_free(xt) };
    assert_eq!(st, rgmin_status_t::RGMIN_SUCCESS);
    assert!(x.iter().all(|&xi| xi > 0.0), "left the interior {x:?}");
    assert!((x[1] - x[2]).abs() < 1e-10, "not symmetric {x:?}");
    let r0 = x[0] + x[1];
    let r1 = x[2] + x[3];
    let c0 = x[0] + x[2];
    let c1 = x[1] + x[3];
    assert!((r0 - 1.0).abs() < 1e-10, "row0 {r0}");
    assert!((r1 - 1.0).abs() < 1e-10, "row1 {r1}");
    assert!((c0 - 1.0).abs() < 1e-10, "col0 {c0}");
    assert!((c1 - 1.0).abs() < 1e-10, "col1 {c1}");
    assert!(
        (x[0] - 0.5).abs() > 1e-8,
        "expected a retraction step {x:?}"
    );
    let fro = x.iter().map(|a| a * a).sum::<f64>().sqrt();
    assert!((fro - 1.0).abs() > 1e-8, "must not be the sphere {x:?}");
    assert_eq!(rgmin_manifold_t::RGMIN_MANIFOLD_MULTINOMIAL_SYM as i32, 19);
    assert_eq!(rgmin_manifold_t::RGMIN_MANIFOLD_MULTINOMIAL_DS as i32, 18);
    assert_eq!(rgmin_manifold_t::RGMIN_MANIFOLD_MW_RIGID as i32, 6);
    assert_eq!(rgmin_manifold_t::RGMIN_MANIFOLD_OBLIQUE as i32, 11);
    unsafe { rgmin_solver_free(session) };
}

#[test]
fn c_abi_sphere_complex_stays_on_the_set() {
    use rgmin::ffi::{rgmin_manifold_t, rgmin_solver_set_sphere_complex};
    let ctrl = rgmin_control_t {
        maxiter: 20,
        gtol: 1e-8,
        istep: 0.1,
        memory: 0,
        maxmove: 0.0,
    };
    let session = unsafe { rgmin_solver_create(rgmin_method_t::RGMIN_STEEPEST, &ctrl, 4) };
    assert!(!session.is_null());
    unsafe { rgmin_solver_set_sphere_complex(session, 2) };
    let mut x = [1.0_f64, 0.0, 0.0, 0.0];
    let mut out = rgmin_report_t {
        value: 0.0,
        steps: 0,
        grad_norm: 0.0,
    };
    let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 4) };
    let st = unsafe {
        rgmin_solver_step(
            session,
            Some(scplx_linear_eval),
            Some(scplx_linear_grad),
            std::ptr::null_mut(),
            xt,
            &mut out,
        )
    };
    unsafe { rgmin_tensor_free(xt) };
    assert_eq!(st, rgmin_status_t::RGMIN_SUCCESS);
    let fro = x.iter().map(|a| a * a).sum::<f64>().sqrt();
    assert!((fro - 1.0).abs() < 1e-12, "left the complex sphere {x:?}");
    assert!(
        (x[0] - 1.0).abs() > 1e-8,
        "expected a retraction step {x:?}"
    );
    assert_eq!(rgmin_manifold_t::RGMIN_MANIFOLD_SPHERE_COMPLEX as i32, 20);
    assert_eq!(rgmin_manifold_t::RGMIN_MANIFOLD_SPHERE as i32, 1);
    unsafe { rgmin_solver_free(session) };
}

#[test]
fn c_abi_positive_stays_on_the_set() {
    use rgmin::ffi::{rgmin_manifold_t, rgmin_solver_set_positive};
    let ctrl = rgmin_control_t {
        maxiter: 20,
        gtol: 1e-8,
        istep: 0.1,
        memory: 0,
        maxmove: 0.0,
    };
    let session = unsafe { rgmin_solver_create(rgmin_method_t::RGMIN_STEEPEST, &ctrl, 3) };
    assert!(!session.is_null());
    unsafe { rgmin_solver_set_positive(session, 3) };
    let start = [2.0_f64, 0.5, 4.0];
    let mut x = start;
    let mut out = rgmin_report_t {
        value: 0.0,
        steps: 0,
        grad_norm: 0.0,
    };
    let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 3) };
    let st = unsafe {
        rgmin_solver_step(
            session,
            Some(pos_linear_eval),
            Some(pos_linear_grad),
            std::ptr::null_mut(),
            xt,
            &mut out,
        )
    };
    unsafe { rgmin_tensor_free(xt) };
    assert_eq!(st, rgmin_status_t::RGMIN_SUCCESS);
    assert!(x.iter().all(|xi| *xi > 0.0), "left the positive set {x:?}");
    assert!(
        (x[0] - start[0]).abs() > 1e-12,
        "expected a retraction step {x:?}"
    );
    let fro = x.iter().map(|a| a * a).sum::<f64>().sqrt();
    assert!((fro - 1.0).abs() > 0.5, "must not be the sphere {x:?}");
    assert_eq!(rgmin_manifold_t::RGMIN_MANIFOLD_POSITIVE as i32, 21);
    assert_eq!(rgmin_manifold_t::RGMIN_MANIFOLD_SPHERE as i32, 1);
    assert_eq!(rgmin_manifold_t::RGMIN_MANIFOLD_MW_RIGID as i32, 6);
    assert_eq!(rgmin_manifold_t::RGMIN_MANIFOLD_OBLIQUE as i32, 11);
    unsafe { rgmin_solver_free(session) };
}

#[test]
fn c_abi_centered_matrix_stays_on_the_set() {
    use rgmin::ffi::{rgmin_manifold_t, rgmin_solver_set_centered_matrix};
    let ctrl = rgmin_control_t {
        maxiter: 20,
        gtol: 1e-8,
        istep: 0.1,
        memory: 0,
        maxmove: 0.0,
    };
    let session = unsafe { rgmin_solver_create(rgmin_method_t::RGMIN_STEEPEST, &ctrl, 4) };
    assert!(!session.is_null());
    unsafe { rgmin_solver_set_centered_matrix(session, 2, 2, 0) };
    let start = [1.0_f64, -1.0, 2.0, -2.0];
    let mut x = start;
    let mut out = rgmin_report_t {
        value: 0.0,
        steps: 0,
        grad_norm: 0.0,
    };
    let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 4) };
    let st = unsafe {
        rgmin_solver_step(
            session,
            Some(centered_linear_eval),
            Some(centered_linear_grad),
            std::ptr::null_mut(),
            xt,
            &mut out,
        )
    };
    unsafe { rgmin_tensor_free(xt) };
    assert_eq!(st, rgmin_status_t::RGMIN_SUCCESS);
    assert!((x[0] + x[1]).abs() < 1e-12, "left the centered set {x:?}");
    assert!((x[2] + x[3]).abs() < 1e-12, "left the centered set {x:?}");
    assert!(
        (x[0] - start[0]).abs() > 1e-12,
        "expected a retraction step {x:?}"
    );
    let fro = x.iter().map(|a| a * a).sum::<f64>().sqrt();
    assert!((fro - 1.0).abs() > 0.5, "must not be the sphere {x:?}");
    assert_eq!(rgmin_manifold_t::RGMIN_MANIFOLD_CENTERED_MATRIX as i32, 22);
    assert_eq!(rgmin_manifold_t::RGMIN_MANIFOLD_POSITIVE as i32, 21);
    assert_eq!(rgmin_manifold_t::RGMIN_MANIFOLD_SPHERE as i32, 1);
    assert_eq!(rgmin_manifold_t::RGMIN_MANIFOLD_MW_RIGID as i32, 6);
    assert_eq!(rgmin_manifold_t::RGMIN_MANIFOLD_OBLIQUE as i32, 11);
    unsafe { rgmin_solver_free(session) };
}

#[test]
fn canonical_manifold_tokens_preserve_main_and_resolve_the_unitary_collision() {
    let tokens = [
        rgmin_manifold_t::RGMIN_MANIFOLD_EUCLIDEAN,
        rgmin_manifold_t::RGMIN_MANIFOLD_SPHERE,
        rgmin_manifold_t::RGMIN_MANIFOLD_SO3,
        rgmin_manifold_t::RGMIN_MANIFOLD_STIEFEL,
        rgmin_manifold_t::RGMIN_MANIFOLD_SE3,
        rgmin_manifold_t::RGMIN_MANIFOLD_RIGID_QUOTIENT,
        rgmin_manifold_t::RGMIN_MANIFOLD_MW_RIGID,
        rgmin_manifold_t::RGMIN_MANIFOLD_SPD,
        rgmin_manifold_t::RGMIN_MANIFOLD_GRASSMANN,
        rgmin_manifold_t::RGMIN_MANIFOLD_HYPERBOLIC,
        rgmin_manifold_t::RGMIN_MANIFOLD_POINCARE,
        rgmin_manifold_t::RGMIN_MANIFOLD_OBLIQUE,
        rgmin_manifold_t::RGMIN_MANIFOLD_MULTINOMIAL,
        rgmin_manifold_t::RGMIN_MANIFOLD_COMPLEX_CIRCLE,
        rgmin_manifold_t::RGMIN_MANIFOLD_SYMMETRIC,
        rgmin_manifold_t::RGMIN_MANIFOLD_SKEWSYMMETRIC,
        rgmin_manifold_t::RGMIN_MANIFOLD_EUCLIDEAN_COMPLEX,
        rgmin_manifold_t::RGMIN_MANIFOLD_CONSTANT,
        rgmin_manifold_t::RGMIN_MANIFOLD_MULTINOMIAL_DS,
        rgmin_manifold_t::RGMIN_MANIFOLD_MULTINOMIAL_SYM,
        rgmin_manifold_t::RGMIN_MANIFOLD_SPHERE_COMPLEX,
        rgmin_manifold_t::RGMIN_MANIFOLD_POSITIVE,
        rgmin_manifold_t::RGMIN_MANIFOLD_CENTERED_MATRIX,
        rgmin_manifold_t::RGMIN_MANIFOLD_UNITARY,
    ];
    for (value, token) in tokens.iter().enumerate() {
        assert_eq!(*token as usize, value);
    }
    assert_eq!(
        rgmin_manifold_t::RGMIN_MANIFOLD_EUCLIDEAN_COMPLEX as i32,
        16
    );
    assert_eq!(rgmin_manifold_t::RGMIN_MANIFOLD_UNITARY as i32, 23);
}

#[test]
fn manifold_shape_setters_accept_null_handles() {
    unsafe {
        rgmin_solver_set_stiefel(std::ptr::null_mut(), 2, 2);
        rgmin_solver_set_oblique(std::ptr::null_mut(), 2, 2);
        rgmin_solver_set_complex_circle(std::ptr::null_mut(), 2);
        rgmin_solver_set_euclidean_complex(std::ptr::null_mut(), 2);
        rgmin_solver_set_constant(std::ptr::null_mut(), 2);
        rgmin_solver_set_multinomial_ds(std::ptr::null_mut(), 2);
        rgmin_solver_set_multinomial_sym(std::ptr::null_mut(), 2);
        rgmin_solver_set_sphere_complex(std::ptr::null_mut(), 2);
        rgmin_solver_set_positive(std::ptr::null_mut(), 2);
        rgmin_solver_set_centered_matrix(std::ptr::null_mut(), 2, 2, 0);
        rgmin_solver_set_factor_shape(std::ptr::null_mut(), 2, 2);
        rgmin_solver_set_unitary(std::ptr::null_mut(), 2);
    }
}
