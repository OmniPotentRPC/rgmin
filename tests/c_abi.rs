//! Drive the DLPack C waist from Rust (Rosenbrock 2D).

#![cfg(feature = "capi")]

use std::os::raw::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};

use dlpk::sys::{DLDeviceType, DLManagedTensorVersioned};
use eindir_core::ffi::eindir_core_abi_stamp;
use eindir_core::ffi::{eindir_objective_t, eindir_status_t};
use rgmin::ffi::{
    rgmin_abi_compatible, rgmin_abi_stamp, rgmin_accept_t, rgmin_conjugacy_t, rgmin_control_t,
    rgmin_curv_fn, rgmin_eigen_kind_t, rgmin_eigen_params_t, rgmin_exp_t, rgmin_last_error,
    rgmin_lowest_eigenpair, rgmin_lowest_mode_t, rgmin_method_from_name, rgmin_method_t,
    rgmin_minimize, rgmin_minimize_eindir, rgmin_minimize_scg, rgmin_report_t, rgmin_scg_params_t,
    rgmin_solver_clear_exp, rgmin_solver_create, rgmin_solver_exp_mu, rgmin_solver_free,
    rgmin_solver_gradient_calls, rgmin_solver_hvp_calls, rgmin_solver_set_accept,
    rgmin_solver_set_exp, rgmin_solver_step, rgmin_solver_step_fg, rgmin_solver_step_hvp,
    rgmin_status_t, rgmin_tensor_borrow_cpu_f64, rgmin_tensor_free,
};
use rgpot_core::eindir::{rgpot_potential_free_eindir, rgpot_potential_new_eindir};
use rgpot_core::status::rgpot_status_t;
use rgpot_core::types::{rgpot_force_input_t, rgpot_force_out_t};

unsafe extern "C" fn quadratic_eval(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
) -> eindir_status_t {
    let (p, n) = unsafe { cpu_f64(x) };
    assert_eq!(n, 2);
    unsafe { *value_out = *p * *p + *p.add(1) * *p.add(1) };
    eindir_status_t::EINDIR_SUCCESS
}

unsafe extern "C" fn quadratic_grad(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    g: *mut DLManagedTensorVersioned,
) -> eindir_status_t {
    let (p, n) = unsafe { cpu_f64(x) };
    assert_eq!(n, 2);
    let (gp, gn) = unsafe { cpu_f64(g) };
    assert_eq!(gn, 2);
    unsafe {
        *(gp as *mut f64) = 2.0 * *p;
        *(gp as *mut f64).add(1) = 2.0 * *p.add(1);
    }
    eindir_status_t::EINDIR_SUCCESS
}

fn quadratic_objective() -> Box<eindir_objective_t> {
    Box::new(eindir_objective_t {
        dim: 2,
        low: std::ptr::null_mut(),
        high: std::ptr::null_mut(),
        eval_fn: quadratic_eval,
        grad_fn: Some(quadratic_grad),
        user_data: std::ptr::null_mut(),
        free_fn: None,
    })
}

unsafe fn cpu_f64(t: *const DLManagedTensorVersioned) -> (*const f64, usize) {
    let dl = unsafe { &(*t).dl_tensor };
    let n = unsafe { *dl.shape as usize };
    let p = unsafe { (dl.data as *const u8).add(dl.byte_offset as usize) as *const f64 };
    (p, n)
}

unsafe extern "C" fn rosen_eval(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
) -> rgmin_status_t {
    let (p, n) = unsafe { cpu_f64(x) };
    assert_eq!(n, 2);
    let x0 = unsafe { *p };
    let x1 = unsafe { *p.add(1) };
    unsafe {
        *value_out = 100.0 * (x1 - x0 * x0).powi(2) + (1.0 - x0).powi(2);
    }
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn rosen_evalgrad(
    user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
    g: *mut DLManagedTensorVersioned,
) -> rgmin_status_t {
    let n = unsafe { &mut *(user as *mut usize) };
    *n += 1;
    let ev = rosen_eval(std::ptr::null_mut(), x, value_out);
    if ev != rgmin_status_t::RGMIN_SUCCESS {
        return ev;
    }
    rosen_grad(std::ptr::null_mut(), x, g)
}

unsafe extern "C" fn rosen_grad(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    g: *mut DLManagedTensorVersioned,
) -> rgmin_status_t {
    let (p, n) = unsafe { cpu_f64(x) };
    assert_eq!(n, 2);
    let x0 = unsafe { *p };
    let x1 = unsafe { *p.add(1) };
    let t = x1 - x0 * x0;
    let (gp, gn) = unsafe { cpu_f64(g as *const _) };
    assert_eq!(gn, 2);
    unsafe {
        *(gp as *mut f64) = -400.0 * x0 * t + 2.0 * (x0 - 1.0);
        *(gp as *mut f64).add(1) = 200.0 * t;
    }
    rgmin_status_t::RGMIN_SUCCESS
}

#[test]
fn c_abi_lbfgs_rosenbrock() {
    let mut x = [-1.2, 1.0];
    let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 2) };
    assert!(!xt.is_null());
    let ctrl = rgmin_control_t {
        maxiter: 200,
        gtol: 1e-10,
        istep: 0.1,
        memory: 10,
        maxmove: 0.0,
    };
    let mut out = rgmin_report_t {
        value: 0.0,
        steps: 0,
        grad_norm: 0.0,
    };
    let st = unsafe {
        rgmin_minimize(
            Some(rosen_eval),
            Some(rosen_grad),
            std::ptr::null_mut(),
            xt,
            &ctrl,
            rgmin_method_t::RGMIN_LBFGS,
            &mut out,
        )
    };
    unsafe { rgmin_tensor_free(xt) };
    assert_eq!(st, rgmin_status_t::RGMIN_SUCCESS);
    assert!(out.value < 1e-6, "C ABI L-BFGS value {}", out.value);
    assert!((x[0] - 1.0).abs() < 1e-3);
    assert!((x[1] - 1.0).abs() < 1e-3);
}

unsafe extern "C" fn quad_eval(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
) -> rgmin_status_t {
    let (p, n) = unsafe { cpu_f64(x) };
    let mut acc = 0.0;
    for i in 0..n {
        let xi = unsafe { *p.add(i) };
        acc += xi * xi;
    }
    unsafe { *value_out = acc };
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn quad_grad(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    g: *mut DLManagedTensorVersioned,
) -> rgmin_status_t {
    let (p, n) = unsafe { cpu_f64(x) };
    let (gp, gn) = unsafe { cpu_f64(g as *const _) };
    assert_eq!(n, gn);
    for i in 0..n {
        unsafe { *(gp as *mut f64).add(i) = 2.0 * *p.add(i) };
    }
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn quadratic_curv(
    _user: *mut c_void,
    _x: *const DLManagedTensorVersioned,
    d: *const DLManagedTensorVersioned,
    curv_out: *mut f64,
) -> rgmin_status_t {
    let (p, n) = unsafe { cpu_f64(d) };
    let mut acc = 0.0;
    for i in 0..n {
        let di = unsafe { *p.add(i) };
        acc += 2.0 * di * di;
    }
    unsafe { *curv_out = acc };
    rgmin_status_t::RGMIN_SUCCESS
}

#[test]
fn c_abi_scg_quadratic_bowl_with_exact_curvature() {
    let mut x = [3.0, -4.0];
    let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 2) };
    assert!(!xt.is_null());
    let ctrl = rgmin_control_t {
        maxiter: 50,
        gtol: 1e-10,
        istep: 1.0,
        memory: 0,
        maxmove: 0.0,
    };
    let params = rgmin_scg_params_t {
        sigma0: 1e-4,
        lambda: 1.0,
        lambda_limit: 1e60,
        tol_sol: 1e-10,
        tol_func: 1e-12,
        conjugacy: rgmin_conjugacy_t::RGMIN_CONJUGACY_LIU_STOREY as i32,
    };
    let mut out = rgmin_report_t {
        value: 0.0,
        steps: 0,
        grad_norm: 0.0,
    };
    let st = unsafe {
        rgmin_minimize_scg(
            Some(quad_eval),
            Some(quad_grad),
            Some(quadratic_curv),
            std::ptr::null_mut(),
            xt,
            &ctrl,
            &params,
            &mut out,
        )
    };
    unsafe { rgmin_tensor_free(xt) };
    assert_eq!(st, rgmin_status_t::RGMIN_SUCCESS);
    assert!(out.value < 1e-8, "C ABI SCG value {}", out.value);
    assert!(x[0].abs() < 1e-4);
    assert!(x[1].abs() < 1e-4);
}

fn scg_bowl_ctrl() -> rgmin_control_t {
    rgmin_control_t {
        maxiter: 50,
        gtol: 1e-10,
        istep: 1.0,
        memory: 0,
        maxmove: 0.0,
    }
}

fn scg_bowl_params(conjugacy: i32) -> rgmin_scg_params_t {
    rgmin_scg_params_t {
        sigma0: 1e-4,
        lambda: 1.0,
        lambda_limit: 1e60,
        tol_sol: 1e-10,
        tol_func: 1e-12,
        conjugacy,
    }
}

fn last_error_text() -> String {
    unsafe { std::ffi::CStr::from_ptr(rgmin_last_error()) }
        .to_string_lossy()
        .into_owned()
}

#[test]
fn c_abi_scg_null_params_uses_netlab_polak_ribiere() {
    let mut x = [3.0, -4.0];
    let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 2) };
    let ctrl = scg_bowl_ctrl();
    let mut out = rgmin_report_t {
        value: 0.0,
        steps: 0,
        grad_norm: 0.0,
    };
    let st = unsafe {
        rgmin_minimize_scg(
            Some(quad_eval),
            Some(quad_grad),
            Some(quadratic_curv),
            std::ptr::null_mut(),
            xt,
            &ctrl,
            std::ptr::null(),
            &mut out,
        )
    };
    unsafe { rgmin_tensor_free(xt) };
    assert_eq!(st, rgmin_status_t::RGMIN_SUCCESS);
    assert!(out.value < 1e-8, "null-params SCG value {}", out.value);
    assert!(x[0].abs() < 1e-4);
    assert!(x[1].abs() < 1e-4);
}

#[test]
fn c_abi_scg_zero_conjugacy_is_fletcher_reeves() {
    let mut x = [3.0, -4.0];
    let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 2) };
    let ctrl = scg_bowl_ctrl();
    let params = scg_bowl_params(rgmin_conjugacy_t::RGMIN_CONJUGACY_FLETCHER_REEVES as i32);
    let mut out = rgmin_report_t {
        value: 0.0,
        steps: 0,
        grad_norm: 0.0,
    };
    let st = unsafe {
        rgmin_minimize_scg(
            Some(quad_eval),
            Some(quad_grad),
            Some(quadratic_curv),
            std::ptr::null_mut(),
            xt,
            &ctrl,
            &params,
            &mut out,
        )
    };
    unsafe { rgmin_tensor_free(xt) };
    assert_eq!(st, rgmin_status_t::RGMIN_SUCCESS);
    assert!(out.value < 1e-8, "FR SCG value {}", out.value);
    assert!(x[0].abs() < 1e-4);
    assert!(x[1].abs() < 1e-4);
}

#[test]
fn c_abi_scg_unknown_conjugacy_is_invalid() {
    let mut x = [3.0, -4.0];
    let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 2) };
    let ctrl = scg_bowl_ctrl();
    let params = scg_bowl_params(99);
    let mut out = rgmin_report_t {
        value: 0.0,
        steps: 0,
        grad_norm: 0.0,
    };
    let st = unsafe {
        rgmin_minimize_scg(
            Some(quad_eval),
            Some(quad_grad),
            Some(quadratic_curv),
            std::ptr::null_mut(),
            xt,
            &ctrl,
            &params,
            &mut out,
        )
    };
    unsafe { rgmin_tensor_free(xt) };
    assert_eq!(st, rgmin_status_t::RGMIN_INVALID_PARAMETER);
    assert!(
        last_error_text().contains("conjugacy"),
        "last_error={}",
        last_error_text()
    );
}

#[test]
fn c_abi_scg_method_t_codes_are_not_conjugacy() {
    let ctrl = scg_bowl_ctrl();
    for raw in [
        rgmin_method_t::RGMIN_FIRE as i32,
        rgmin_method_t::RGMIN_LIU_STOREY as i32,
    ] {
        let mut x = [3.0, -4.0];
        let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 2) };
        let params = scg_bowl_params(raw);
        let mut out = rgmin_report_t {
            value: 0.0,
            steps: 0,
            grad_norm: 0.0,
        };
        let st = unsafe {
            rgmin_minimize_scg(
                Some(quad_eval),
                Some(quad_grad),
                Some(quadratic_curv),
                std::ptr::null_mut(),
                xt,
                &ctrl,
                &params,
                &mut out,
            )
        };
        unsafe { rgmin_tensor_free(xt) };
        assert_eq!(
            st,
            rgmin_status_t::RGMIN_INVALID_PARAMETER,
            "method_t {raw} accepted as conjugacy"
        );
        assert!(
            last_error_text().contains("conjugacy"),
            "last_error for {raw}: {}",
            last_error_text()
        );
    }
}

#[test]
fn c_abi_eindir_objective_minimizes_without_taking_ownership() {
    let mut x = [3.0, -4.0];
    let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 2) };
    assert!(!xt.is_null());
    let objective = quadratic_objective();
    let ctrl = rgmin_control_t {
        maxiter: 100,
        gtol: 1e-10,
        istep: 0.1,
        memory: 10,
        maxmove: 0.0,
    };
    let mut out = rgmin_report_t {
        value: 0.0,
        steps: 0,
        grad_norm: 0.0,
    };
    let st = unsafe {
        rgmin_minimize_eindir(
            &*objective,
            &eindir_core_abi_stamp(),
            xt,
            &ctrl,
            rgmin_method_t::RGMIN_LBFGS,
            &mut out,
        )
    };
    assert_eq!(st, rgmin_status_t::RGMIN_SUCCESS);
    assert!(out.value < 1e-12, "eindir objective value {}", out.value);
    assert!(x.iter().all(|value| value.abs() < 1e-5));
    assert_eq!(objective.dim, 2);
    unsafe { rgmin_tensor_free(xt) };
}

#[test]
fn c_abi_eindir_rejects_incompatible_stamp_before_evaluation() {
    let mut x = [3.0, -4.0];
    let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 2) };
    let objective = quadratic_objective();
    let mut stamp = eindir_core_abi_stamp();
    stamp.objective_layout += 1;
    let ctrl = rgmin_control_t {
        maxiter: 1,
        gtol: 1e-10,
        istep: 0.1,
        memory: 10,
        maxmove: 0.0,
    };
    let mut out = rgmin_report_t {
        value: 0.0,
        steps: 0,
        grad_norm: 0.0,
    };
    let st = unsafe {
        rgmin_minimize_eindir(
            &*objective,
            &stamp,
            xt,
            &ctrl,
            rgmin_method_t::RGMIN_LBFGS,
            &mut out,
        )
    };
    assert_eq!(st, rgmin_status_t::RGMIN_INVALID_PARAMETER);
    assert_eq!(x, [3.0, -4.0]);
    unsafe { rgmin_tensor_free(xt) };
}

struct QuadraticContext {
    forces: [f64; 6],
}

unsafe extern "C" fn rgpot_quadratic_callback(
    user: *mut c_void,
    input: *const rgpot_force_input_t,
    output: *mut rgpot_force_out_t,
) -> rgpot_status_t {
    let context = unsafe { &mut *(user as *mut QuadraticContext) };
    let input = unsafe { &*input };
    let output = unsafe { &mut *output };
    let positions = unsafe { &(*input.positions).dl_tensor };
    let n = unsafe { (positions.shape as *const i64).read() as usize * 3 };
    let values = unsafe { std::slice::from_raw_parts(positions.data as *const f64, n) };
    output.energy = values.iter().map(|value| value * value).sum();
    context
        .forces
        .iter_mut()
        .zip(values)
        .for_each(|(force, value)| *force = -2.0 * value);
    output.forces = unsafe {
        rgpot_core::tensor::rgpot_tensor_cpu_f64_2d(context.forces.as_mut_ptr(), n as i64 / 3, 3)
    };
    rgpot_status_t::RGPOT_SUCCESS
}

#[test]
fn c_abi_rgpot_potential_reaches_eindir_optimizer() {
    let atomic_numbers = [1i32, 1];
    let cell = [10.0f64, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];
    let mut context = QuadraticContext { forces: [0.0; 6] };
    let potential = unsafe {
        rgpot_potential_new_eindir(
            rgpot_quadratic_callback,
            (&mut context as *mut QuadraticContext).cast(),
            None,
            2,
            atomic_numbers.as_ptr(),
            cell.as_ptr(),
            std::ptr::null(),
            std::ptr::null(),
        )
    };
    assert!(!potential.is_null());

    let mut x = [2.0, -1.5, 0.5, -2.5, 1.0, -0.25];
    let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), x.len()) };
    let ctrl = rgmin_control_t {
        maxiter: 100,
        gtol: 1e-8,
        istep: 0.1,
        memory: 10,
        maxmove: 0.0,
    };
    let mut out = rgmin_report_t {
        value: 0.0,
        steps: 0,
        grad_norm: 0.0,
    };
    let stamp = rgpot_core::eindir::rgpot_eindir_abi_stamp();
    let status = unsafe {
        rgmin_minimize_eindir(
            potential.cast(),
            &stamp,
            xt,
            &ctrl,
            rgmin_method_t::RGMIN_LBFGS,
            &mut out,
        )
    };

    assert_eq!(status, rgmin_status_t::RGMIN_SUCCESS);
    assert!(out.value < 1e-12, "rgpot objective value {}", out.value);
    assert!(x.iter().all(|value| value.abs() < 1e-5));
    unsafe {
        rgmin_tensor_free(xt);
        rgpot_potential_free_eindir(potential);
    }
}

#[test]
fn cuda_tagged_tensor_is_unsupported() {
    let mut x = [-1.2, 1.0];
    let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 2) };
    assert!(!xt.is_null());
    unsafe {
        (*xt).dl_tensor.device.device_type = DLDeviceType::kDLCUDA;
    }
    let ctrl = rgmin_control_t {
        maxiter: 1,
        gtol: 1e-10,
        istep: 0.1,
        memory: 10,
        maxmove: 0.0,
    };
    let mut out = rgmin_report_t {
        value: 0.0,
        steps: 0,
        grad_norm: 0.0,
    };
    let st = unsafe {
        rgmin_minimize(
            Some(rosen_eval),
            Some(rosen_grad),
            std::ptr::null_mut(),
            xt,
            &ctrl,
            rgmin_method_t::RGMIN_LBFGS,
            &mut out,
        )
    };
    unsafe { rgmin_tensor_free(xt) };
    assert_eq!(st, rgmin_status_t::RGMIN_UNSUPPORTED_DEVICE);
}

#[test]
fn c_abi_solver_step_keeps_lbfgs_history() {
    let mut warm = [-1.2, 1.0];
    let mut cold = [-1.2, 1.0];
    let ctrl = rgmin_control_t {
        maxiter: 80,
        gtol: 1e-10,
        istep: 0.1,
        memory: 10,
        maxmove: 0.0,
    };
    let session = unsafe { rgmin_solver_create(rgmin_method_t::RGMIN_LBFGS, &ctrl, 2) };
    assert!(!session.is_null());
    let mut out = rgmin_report_t {
        value: 0.0,
        steps: 0,
        grad_norm: 0.0,
    };
    for _ in 0..80 {
        let xt = unsafe { rgmin_tensor_borrow_cpu_f64(warm.as_mut_ptr(), 2) };
        let st = unsafe {
            rgmin_solver_step(
                session,
                Some(rosen_eval),
                Some(rosen_grad),
                std::ptr::null_mut(),
                xt,
                &mut out,
            )
        };
        unsafe { rgmin_tensor_free(xt) };
        assert_eq!(st, rgmin_status_t::RGMIN_SUCCESS);
        if out.grad_norm < 1e-8 {
            break;
        }
    }
    unsafe { rgmin_solver_free(session) };
    assert!(out.value < 1e-6, "session value {}", out.value);

    for _ in 0..80 {
        let one = unsafe { rgmin_solver_create(rgmin_method_t::RGMIN_LBFGS, &ctrl, 2) };
        let xt = unsafe { rgmin_tensor_borrow_cpu_f64(cold.as_mut_ptr(), 2) };
        let st = unsafe {
            rgmin_solver_step(
                one,
                Some(rosen_eval),
                Some(rosen_grad),
                std::ptr::null_mut(),
                xt,
                &mut out,
            )
        };
        unsafe {
            rgmin_tensor_free(xt);
            rgmin_solver_free(one);
        }
        assert_eq!(st, rgmin_status_t::RGMIN_SUCCESS);
        if out.grad_norm < 1e-8 {
            break;
        }
    }
    // A live session is L-BFGS. Recreating every step is steepest-plus-Wolfe.
    assert!(
        (warm[0] - 1.0).abs() < 1e-3,
        "warm end {} {}",
        warm[0],
        warm[1]
    );
}

/// `istep = 0` used to mean a raw `-g` first step. From `(-1.2, 1.0)`
/// every energy halving of it rises, so these sessions stayed at
/// `f = 24.2`. A unit first step accepts on its third trial, the
/// second halving, at `f = 6.321495`.
#[test]
fn c_abi_default_istep_leaves_the_rosenbrock_start() {
    let ctrl = rgmin_control_t {
        maxiter: 80,
        gtol: 1e-10,
        istep: 0.0,
        memory: 0,
        maxmove: 0.0,
    };
    for method in [
        rgmin_method_t::RGMIN_BFGS,
        rgmin_method_t::RGMIN_SR1,
        rgmin_method_t::RGMIN_SR2,
        rgmin_method_t::RGMIN_BB,
    ] {
        let mut x = [-1.2, 1.0];
        let session = unsafe { rgmin_solver_create(method, &ctrl, 2) };
        assert!(!session.is_null());
        let mut out = rgmin_report_t {
            value: 0.0,
            steps: 0,
            grad_norm: 0.0,
        };
        let mut first = f64::NAN;
        for k in 0..80 {
            let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 2) };
            let st = unsafe {
                rgmin_solver_step(
                    session,
                    Some(rosen_eval),
                    Some(rosen_grad),
                    std::ptr::null_mut(),
                    xt,
                    &mut out,
                )
            };
            unsafe { rgmin_tensor_free(xt) };
            assert_eq!(st, rgmin_status_t::RGMIN_SUCCESS);
            if k == 0 {
                first = out.value;
            }
            if out.grad_norm < 1e-8 {
                break;
            }
        }
        unsafe { rgmin_solver_free(session) };
        assert!(
            (first - 6.321495).abs() < 1e-6,
            "{method:?} first step f = {first}"
        );
        assert!(out.value < 1e-6, "{method:?} ended at f = {}", out.value);
        if method != rgmin_method_t::RGMIN_BB {
            assert!(out.grad_norm < 1e-6, "{method:?} |g| = {}", out.grad_norm);
            assert!(
                (x[0] - 1.0).abs() < 1e-6 && (x[1] - 1.0).abs() < 1e-6,
                "{method:?} ended at {x:?}"
            );
        }
    }
}

#[test]
fn version_is_nul_terminated() {
    let p = rgmin::ffi::rgmin_version();
    assert!(!p.is_null());
}

#[test]
fn fused_evalgrad_is_one_callback_per_oracle() {
    let mut x = [-1.2, 1.0];
    let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 2) };
    let ctrl = rgmin_control_t {
        maxiter: 1,
        gtol: 1e-12,
        istep: 0.1,
        memory: 10,
        maxmove: 0.2,
    };
    let session = unsafe { rgmin_solver_create(rgmin_method_t::RGMIN_LBFGS, &ctrl, 2) };
    assert!(!session.is_null());
    unsafe { rgmin_solver_set_accept(session, rgmin_accept_t::RGMIN_ACCEPT_NONE) };
    let mut calls = 0usize;
    let mut out = rgmin_report_t {
        value: 0.0,
        steps: 0,
        grad_norm: 0.0,
    };
    let st = unsafe {
        rgmin_solver_step_fg(
            session,
            Some(rosen_evalgrad),
            (&mut calls as *mut usize).cast(),
            xt,
            &mut out,
        )
    };
    unsafe {
        rgmin_solver_free(session);
        rgmin_tensor_free(xt);
    }
    assert_eq!(st, rgmin_status_t::RGMIN_SUCCESS);
    assert!(calls >= 1, "fused oracle never ran");
}

unsafe extern "C" fn gapped_hvp(
    _user: *mut c_void,
    _x: *const DLManagedTensorVersioned,
    v: *const DLManagedTensorVersioned,
    hv_out: *mut DLManagedTensorVersioned,
) -> rgmin_status_t {
    let (vp, n) = unsafe { cpu_f64(v) };
    let (hp, hn) = unsafe { cpu_f64(hv_out as *const DLManagedTensorVersioned) };
    assert_eq!(n, hn);
    let hp = hp as *mut f64;
    unsafe {
        *hp = -8.0 * *vp;
        for i in 1..n {
            *hp.add(i) = 4.0 * *vp.add(i);
        }
    }
    rgmin_status_t::RGMIN_SUCCESS
}

#[test]
fn lowest_eigenpair_lanczos_recovers_gapped_mode() {
    let mut x = [0.0_f64; 6];
    let mut seed = [0.2, 0.7, 0.1, 0.0, 0.0, 0.0];
    let mut mode = [0.0_f64; 6];
    let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 6) };
    let st = unsafe { rgmin_tensor_borrow_cpu_f64(seed.as_mut_ptr(), 6) };
    let mt = unsafe { rgmin_tensor_borrow_cpu_f64(mode.as_mut_ptr(), 6) };
    let params = rgmin_eigen_params_t {
        kind: rgmin_eigen_kind_t::RGMIN_EIGEN_LANCZOS as i32,
        nev: 1,
        krylov: 6,
        max_iter: 0,
        tol: 0.0,
    };
    let mut out = rgmin_lowest_mode_t {
        value: 0.0,
        actions: 0,
    };
    let status = unsafe {
        rgmin_lowest_eigenpair(
            Some(gapped_hvp),
            std::ptr::null_mut(),
            xt,
            st,
            mt,
            &params,
            &mut out,
        )
    };
    unsafe {
        rgmin_tensor_free(xt);
        rgmin_tensor_free(st);
        rgmin_tensor_free(mt);
    }
    assert_eq!(status, rgmin_status_t::RGMIN_SUCCESS);
    assert!(out.value < 0.0, "curvature {}", out.value);
    assert!(mode[0].abs() > 0.9, "mode {mode:?}");
    assert!(out.actions <= 6);
}

#[test]
fn lowest_eigenpair_elpa_is_unavailable() {
    let mut x = [0.0_f64; 4];
    let mut seed = [1.0, 0.0, 0.0, 0.0];
    let mut mode = [0.0_f64; 4];
    let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 4) };
    let st = unsafe { rgmin_tensor_borrow_cpu_f64(seed.as_mut_ptr(), 4) };
    let mt = unsafe { rgmin_tensor_borrow_cpu_f64(mode.as_mut_ptr(), 4) };
    let params = rgmin_eigen_params_t {
        kind: rgmin_eigen_kind_t::RGMIN_EIGEN_ELPA as i32,
        nev: 1,
        krylov: 0,
        max_iter: 0,
        tol: 0.0,
    };
    let mut out = rgmin_lowest_mode_t {
        value: 0.0,
        actions: 0,
    };
    let status = unsafe {
        rgmin_lowest_eigenpair(
            Some(gapped_hvp),
            std::ptr::null_mut(),
            xt,
            st,
            mt,
            &params,
            &mut out,
        )
    };
    unsafe {
        rgmin_tensor_free(xt);
        rgmin_tensor_free(st);
        rgmin_tensor_free(mt);
    }
    assert_eq!(status, rgmin_status_t::RGMIN_UNAVAILABLE);
}

#[test]
fn abi_stamp_identifies_this_optimizer_layout() {
    let stamp = rgmin_abi_stamp();
    assert_eq!(stamp.abi_major, 1);
    assert_eq!(stamp.abi_minor, 37);
    assert_eq!(stamp.layout_revision, 4);
    assert_eq!(unsafe { rgmin_abi_compatible(&stamp) }, 1);
}

/// The C session reaches the Rosenbrock minimum with the radius on and
/// reports the radius it ends with.
#[test]
fn c_abi_lbfgs_trust_radius() {
    use rgmin::ffi::{rgmin_solver_lbfgs_trust_radius, rgmin_solver_set_lbfgs_trust};
    assert_eq!(
        unsafe { rgmin_solver_set_lbfgs_trust(std::ptr::null_mut(), 0.2) },
        1
    );
    assert_eq!(
        unsafe { rgmin_solver_lbfgs_trust_radius(std::ptr::null()) },
        0.0
    );
    let ctrl = rgmin_control_t {
        maxiter: 200,
        gtol: 1e-6,
        istep: 0.2,
        memory: 20,
        maxmove: 0.0,
    };
    let session = unsafe { rgmin_solver_create(rgmin_method_t::RGMIN_LBFGS, &ctrl, 2) };
    assert!(!session.is_null());
    assert_eq!(unsafe { rgmin_solver_lbfgs_trust_radius(session) }, 0.0);
    assert_eq!(unsafe { rgmin_solver_set_lbfgs_trust(session, 0.2) }, 0);
    assert_eq!(unsafe { rgmin_solver_lbfgs_trust_radius(session) }, 0.2);
    let mut x = [-1.2, 1.0];
    let mut calls = 0usize;
    let mut out = rgmin_report_t {
        value: 0.0,
        steps: 0,
        grad_norm: f64::INFINITY,
    };
    for _ in 0..200 {
        let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 2) };
        let st = unsafe {
            rgmin_solver_step_fg(
                session,
                Some(rosen_evalgrad),
                (&mut calls as *mut usize).cast(),
                xt,
                &mut out,
            )
        };
        unsafe { rgmin_tensor_free(xt) };
        assert_eq!(st, rgmin_status_t::RGMIN_SUCCESS);
        if out.grad_norm < 1e-6 {
            break;
        }
    }
    assert!(out.grad_norm < 1e-6, "{} {x:?}", out.grad_norm);
    assert!((x[0] - 1.0).abs() < 1e-5 && (x[1] - 1.0).abs() < 1e-5);
    assert!(unsafe { rgmin_solver_lbfgs_trust_radius(session) } > 0.0);
    unsafe { rgmin_solver_free(session) };
}

unsafe extern "C" fn flat_evalgrad(
    _user: *mut c_void,
    _x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
    g: *mut DLManagedTensorVersioned,
) -> rgmin_status_t {
    let (gp, gn) = unsafe { cpu_f64(g as *const _) };
    unsafe {
        *value_out = 1.0;
        for i in 0..gn {
            *(gp as *mut f64).add(i) = 0.0;
        }
    }
    rgmin_status_t::RGMIN_SUCCESS
}

/// NaN and infinite radii turn the radius off and return 0. A zero
/// gradient at `gtol = 0` has slope 0 and is no descent.
#[test]
fn c_abi_lbfgs_trust_non_finite_radius_and_no_descent() {
    use rgmin::ffi::{rgmin_solver_lbfgs_trust_radius, rgmin_solver_set_lbfgs_trust};
    let ctrl = rgmin_control_t {
        maxiter: 10,
        gtol: 0.0,
        istep: 0.2,
        memory: 20,
        maxmove: 0.0,
    };
    let session = unsafe { rgmin_solver_create(rgmin_method_t::RGMIN_LBFGS, &ctrl, 2) };
    assert!(!session.is_null());
    for radius in [f64::NAN, f64::INFINITY, -1.0] {
        assert_eq!(unsafe { rgmin_solver_set_lbfgs_trust(session, 0.5) }, 0);
        assert_eq!(unsafe { rgmin_solver_set_lbfgs_trust(session, radius) }, 0);
        assert_eq!(unsafe { rgmin_solver_lbfgs_trust_radius(session) }, 0.0);
    }
    assert_eq!(unsafe { rgmin_solver_set_lbfgs_trust(session, 0.5) }, 0);
    let mut x = [3.0, -1.0];
    let mut out = rgmin_report_t {
        value: 0.0,
        steps: 0,
        grad_norm: f64::INFINITY,
    };
    let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 2) };
    let st = unsafe {
        rgmin_solver_step_fg(
            session,
            Some(flat_evalgrad),
            std::ptr::null_mut(),
            xt,
            &mut out,
        )
    };
    unsafe { rgmin_tensor_free(xt) };
    assert_eq!(st, rgmin_status_t::RGMIN_INTERNAL_ERROR);
    assert!(last_error_text().contains("no descent direction"));
    assert_eq!(x, [3.0, -1.0]);
    unsafe { rgmin_solver_free(session) };
}

unsafe extern "C" fn steep_evalgrad(
    _user: *mut c_void,
    _x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
    g: *mut DLManagedTensorVersioned,
) -> rgmin_status_t {
    let (gp, gn) = unsafe { cpu_f64(g as *const _) };
    unsafe {
        *value_out = 1.0;
        for i in 0..gn {
            *(gp as *mut f64).add(i) = 1e200;
        }
    }
    rgmin_status_t::RGMIN_SUCCESS
}

/// A gradient of `1e200` overflows the slope to `-inf`. The step reports a
/// non-finite direction and leaves `x` where it was.
#[test]
fn c_abi_lbfgs_trust_infinite_slope_is_an_error() {
    use rgmin::ffi::rgmin_solver_set_lbfgs_trust;
    let ctrl = rgmin_control_t {
        maxiter: 10,
        gtol: 1e-6,
        istep: 0.2,
        memory: 20,
        maxmove: 0.0,
    };
    let session = unsafe { rgmin_solver_create(rgmin_method_t::RGMIN_LBFGS, &ctrl, 2) };
    assert!(!session.is_null());
    assert_eq!(unsafe { rgmin_solver_set_lbfgs_trust(session, 0.5) }, 0);
    let mut x = [3.0, -1.0];
    let mut out = rgmin_report_t {
        value: 0.0,
        steps: 0,
        grad_norm: f64::INFINITY,
    };
    let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 2) };
    let st = unsafe {
        rgmin_solver_step_fg(
            session,
            Some(steep_evalgrad),
            std::ptr::null_mut(),
            xt,
            &mut out,
        )
    };
    unsafe { rgmin_tensor_free(xt) };
    assert_eq!(st, rgmin_status_t::RGMIN_INTERNAL_ERROR);
    assert!(last_error_text().contains("non-finite search direction"));
    assert_eq!(x, [3.0, -1.0]);
    unsafe { rgmin_solver_free(session) };
}

/// Two atoms at force 0.9 each: max-atom 0.9, two-norm 1.27. At gtol 1
/// the max-atom session stops at the start and the default session
/// moves. Both hand back the gradient their report measured.
#[test]
fn c_abi_stop_norm_and_report_gradient() {
    use rgmin::ffi::{rgmin_solver_report_gradient, rgmin_solver_set_stop_norm, rgmin_stop_norm_t};
    let ctrl = rgmin_control_t {
        maxiter: 10,
        gtol: 1.0,
        istep: 0.1,
        memory: 0,
        maxmove: 0.0,
    };
    let max_atom = rgmin_stop_norm_t::RGMIN_STOP_NORM_MAX_ATOM as i32;
    assert_eq!(
        unsafe { rgmin_solver_set_stop_norm(std::ptr::null_mut(), max_atom) },
        1
    );
    let mut probe = [0.0; 6];
    let st = unsafe { rgmin_solver_report_gradient(std::ptr::null(), probe.as_mut_ptr(), 6) };
    assert_eq!(st, rgmin_status_t::RGMIN_INVALID_PARAMETER);
    for norm in [Some(max_atom), None] {
        let session = unsafe { rgmin_solver_create(rgmin_method_t::RGMIN_LBFGS, &ctrl, 6) };
        assert!(!session.is_null());
        let mut g = [f64::NAN; 6];
        let st = unsafe { rgmin_solver_report_gradient(session, g.as_mut_ptr(), 6) };
        assert_eq!(st, rgmin_status_t::RGMIN_INVALID_PARAMETER);
        assert_eq!(unsafe { rgmin_solver_set_stop_norm(session, 3) }, 1);
        if let Some(norm) = norm {
            assert_eq!(unsafe { rgmin_solver_set_stop_norm(session, norm) }, 0);
        }
        let start = [0.45, 0.0, 0.0, 0.45, 0.0, 0.0];
        let mut x = start;
        let mut out = rgmin_report_t {
            value: 0.0,
            steps: 0,
            grad_norm: 0.0,
        };
        let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 6) };
        let st = unsafe {
            rgmin_solver_step(
                session,
                Some(quad_eval),
                Some(quad_grad),
                std::ptr::null_mut(),
                xt,
                &mut out,
            )
        };
        unsafe { rgmin_tensor_free(xt) };
        assert_eq!(st, rgmin_status_t::RGMIN_SUCCESS);
        let st = unsafe { rgmin_solver_report_gradient(session, g.as_mut_ptr(), 5) };
        assert_eq!(st, rgmin_status_t::RGMIN_INVALID_PARAMETER);
        let st = unsafe { rgmin_solver_report_gradient(session, g.as_mut_ptr(), 6) };
        assert_eq!(st, rgmin_status_t::RGMIN_SUCCESS);
        for i in 0..6 {
            assert_eq!(g[i], 2.0 * x[i]);
        }
        if norm.is_some() {
            assert_eq!(x, start);
            assert_eq!(out.steps, 0);
            assert_eq!(out.grad_norm, 0.9);
        } else {
            assert_ne!(x, start);
            assert_eq!(out.steps, 1);
            assert_eq!(out.grad_norm, g.iter().map(|v| v * v).sum::<f64>().sqrt());
        }
        unsafe { rgmin_solver_free(session) };
    }
}

#[test]
fn abi_stamp_rejects_an_incompatible_layout() {
    let mut stamp = rgmin_abi_stamp();
    stamp.layout_revision += 1;
    assert_eq!(unsafe { rgmin_abi_compatible(&stamp) }, 0);
}

#[test]
fn abi_stamp_ignores_minor() {
    let mut stamp = rgmin_abi_stamp();
    stamp.abi_minor = stamp.abi_minor.saturating_add(1);
    assert_eq!(unsafe { rgmin_abi_compatible(&stamp) }, 1);
}

#[test]
fn c_abi_respects_maxmove_when_initial_step_is_larger() {
    let mut x = [1.0, 0.0];
    let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 2) };
    let ctrl = rgmin_control_t {
        maxiter: 1,
        gtol: 1e-12,
        istep: 10.0,
        memory: 10,
        maxmove: 0.1,
    };
    let mut out = rgmin_report_t {
        value: 0.0,
        steps: 0,
        grad_norm: 0.0,
    };
    let st = unsafe {
        rgmin_minimize(
            Some(rosen_eval),
            Some(rosen_grad),
            std::ptr::null_mut(),
            xt,
            &ctrl,
            rgmin_method_t::RGMIN_LBFGS,
            &mut out,
        )
    };
    unsafe { rgmin_tensor_free(xt) };
    assert_eq!(st, rgmin_status_t::RGMIN_SUCCESS);
    assert!((1.0 - x[0]).abs() <= 0.1 + 1e-12);
}

/// Every C setter the header exports has to survive being called: on a
/// live session, and on a null one, where the contract is a silent
/// no-op rather than a crash. These were the thirteen exported symbols
/// with no test at the C surface -- the ABI cannot be called stable
/// while most of it has never been dialled from C.
#[test]
fn c_abi_every_setter_survives_live_and_null_sessions() {
    use rgmin::ffi::{
        rgmin_manifold_t, rgmin_qn_step_t, rgmin_solver_add_equality,
        rgmin_solver_clear_equalities, rgmin_solver_forget, rgmin_solver_set_atom_maxmove,
        rgmin_solver_set_box, rgmin_solver_set_cautious, rgmin_solver_set_centered_matrix,
        rgmin_solver_set_constant, rgmin_solver_set_euclidean_complex,
        rgmin_solver_set_extra_updates, rgmin_solver_set_highs, rgmin_solver_set_manifold,
        rgmin_solver_set_masses, rgmin_solver_set_maxmove, rgmin_solver_set_multinomial_ds,
        rgmin_solver_set_multinomial_sym, rgmin_solver_set_oblique, rgmin_solver_set_periodic,
        rgmin_solver_set_positive, rgmin_solver_set_project_rigid, rgmin_solver_set_qn_step,
        rgmin_solver_set_sphere_complex, rgmin_solver_set_stiefel, rgmin_solver_set_trust,
    };
    let ctrl = rgmin_control_t {
        maxiter: 20,
        gtol: 1e-8,
        istep: 0.1,
        memory: 8,
        maxmove: 0.0,
    };
    let session = unsafe { rgmin_solver_create(rgmin_method_t::RGMIN_LBFGS, &ctrl, 2) };
    assert!(!session.is_null());
    let masses = [1.0_f64, 12.0, 16.0];
    unsafe {
        rgmin_solver_set_maxmove(session, 0.5);
        rgmin_solver_set_atom_maxmove(session, 0.2);
        rgmin_solver_set_qn_step(session, rgmin_qn_step_t::RGMIN_QN_LBFGS);
        rgmin_solver_set_extra_updates(session, 2);
        rgmin_solver_set_cautious(session, 1e-6, 0.01);
        rgmin_solver_set_project_rigid(session, 1);
        rgmin_solver_set_periodic(session, 1);
        rgmin_solver_set_manifold(session, rgmin_manifold_t::RGMIN_MANIFOLD_MW_RIGID);
        rgmin_solver_set_masses(session, masses.as_ptr(), masses.len());
        rgmin_solver_set_masses(session, std::ptr::null(), 0);
        rgmin_solver_set_manifold(session, rgmin_manifold_t::RGMIN_MANIFOLD_SYMMETRIC);
        rgmin_solver_set_manifold(session, rgmin_manifold_t::RGMIN_MANIFOLD_SKEWSYMMETRIC);
        rgmin_solver_set_manifold(session, rgmin_manifold_t::RGMIN_MANIFOLD_EUCLIDEAN_COMPLEX);
        rgmin_solver_set_euclidean_complex(session, 2);
        rgmin_solver_set_manifold(session, rgmin_manifold_t::RGMIN_MANIFOLD_CONSTANT);
        rgmin_solver_set_constant(session, 2);
        rgmin_solver_set_manifold(session, rgmin_manifold_t::RGMIN_MANIFOLD_MULTINOMIAL_DS);
        rgmin_solver_set_multinomial_ds(session, 2);
        rgmin_solver_set_manifold(session, rgmin_manifold_t::RGMIN_MANIFOLD_MULTINOMIAL_SYM);
        rgmin_solver_set_multinomial_sym(session, 2);
        rgmin_solver_set_manifold(session, rgmin_manifold_t::RGMIN_MANIFOLD_SPHERE_COMPLEX);
        rgmin_solver_set_sphere_complex(session, 1);
        rgmin_solver_set_manifold(session, rgmin_manifold_t::RGMIN_MANIFOLD_POSITIVE);
        rgmin_solver_set_positive(session, 2);
        rgmin_solver_set_manifold(session, rgmin_manifold_t::RGMIN_MANIFOLD_CENTERED_MATRIX);
        rgmin_solver_set_centered_matrix(session, 2, 2, 0);
        rgmin_solver_set_manifold(session, rgmin_manifold_t::RGMIN_MANIFOLD_MULTINOMIAL);
        rgmin_solver_set_oblique(session, 3, 2);
        rgmin_solver_set_stiefel(session, 4, 2);
        rgmin_solver_set_manifold(session, rgmin_manifold_t::RGMIN_MANIFOLD_SYMMETRIC);
        rgmin_solver_set_manifold(session, rgmin_manifold_t::RGMIN_MANIFOLD_EUCLIDEAN);
        let lo = [-1.0_f64, -2.0];
        let hi = [1.0_f64, 2.0];
        let idx = [0usize, 1];
        let coef = [1.0_f64, -1.0];
        let _ = rgmin_solver_set_highs(session, 1);
        let _ = rgmin_solver_set_box(session, lo.as_ptr(), hi.as_ptr(), 2);
        let _ = rgmin_solver_set_box(session, std::ptr::null(), hi.as_ptr(), 2);
        let _ = rgmin_solver_set_trust(session, 0.25);
        let _ = rgmin_solver_add_equality(session, idx.as_ptr(), coef.as_ptr(), 2, 0.0);
        let _ = rgmin_solver_clear_equalities(session);
        rgmin_solver_forget(session);
    }
    // The configured session still relaxes: setters must leave a
    // working solver behind, not only avoid crashing.
    let mut x = [-1.2_f64, 1.0];
    let mut out = rgmin_report_t {
        value: 0.0,
        steps: 0,
        grad_norm: 0.0,
    };
    let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 2) };
    let st = unsafe {
        rgmin_solver_step(
            session,
            Some(rosen_eval),
            Some(rosen_grad),
            std::ptr::null_mut(),
            xt,
            &mut out,
        )
    };
    unsafe { rgmin_tensor_free(xt) };
    assert_eq!(st, rgmin_status_t::RGMIN_SUCCESS);
    assert!(out.value.is_finite());
    unsafe { rgmin_solver_free(session) };

    // The null contract: every setter is a no-op, never a crash.
    let null = std::ptr::null_mut();
    unsafe {
        rgmin_solver_set_maxmove(null, 0.5);
        rgmin_solver_set_atom_maxmove(null, 0.2);
        rgmin_solver_set_qn_step(null, rgmin_qn_step_t::RGMIN_QN_NEWTON);
        rgmin_solver_set_extra_updates(null, 2);
        rgmin_solver_set_cautious(null, 1e-6, 0.01);
        rgmin_solver_set_project_rigid(null, 1);
        rgmin_solver_set_periodic(null, 0);
        rgmin_solver_set_manifold(null, rgmin_manifold_t::RGMIN_MANIFOLD_SPHERE);
        rgmin_solver_set_manifold(null, rgmin_manifold_t::RGMIN_MANIFOLD_MULTINOMIAL);
        rgmin_solver_set_manifold(null, rgmin_manifold_t::RGMIN_MANIFOLD_EUCLIDEAN_COMPLEX);
        rgmin_solver_set_euclidean_complex(null, 2);
        rgmin_solver_set_manifold(null, rgmin_manifold_t::RGMIN_MANIFOLD_CONSTANT);
        rgmin_solver_set_constant(null, 2);
        rgmin_solver_set_manifold(null, rgmin_manifold_t::RGMIN_MANIFOLD_MULTINOMIAL_DS);
        rgmin_solver_set_multinomial_ds(null, 2);
        rgmin_solver_set_manifold(null, rgmin_manifold_t::RGMIN_MANIFOLD_MULTINOMIAL_SYM);
        rgmin_solver_set_multinomial_sym(null, 2);
        rgmin_solver_set_manifold(null, rgmin_manifold_t::RGMIN_MANIFOLD_SPHERE_COMPLEX);
        rgmin_solver_set_sphere_complex(null, 2);
        rgmin_solver_set_manifold(null, rgmin_manifold_t::RGMIN_MANIFOLD_POSITIVE);
        rgmin_solver_set_positive(null, 2);
        rgmin_solver_set_manifold(null, rgmin_manifold_t::RGMIN_MANIFOLD_CENTERED_MATRIX);
        rgmin_solver_set_centered_matrix(null, 2, 2, 0);
        rgmin_solver_set_oblique(null, 3, 2);
        rgmin_solver_set_stiefel(null, 4, 2);
        rgmin_solver_set_masses(null, masses.as_ptr(), masses.len());
        let lo = [-1.0_f64, -2.0];
        let hi = [1.0_f64, 2.0];
        let idx = [0usize, 1];
        let coef = [1.0_f64, -1.0];
        let _ = rgmin_solver_set_highs(null, 1);
        let _ = rgmin_solver_set_box(null, lo.as_ptr(), hi.as_ptr(), 2);
        let _ = rgmin_solver_set_trust(null, 0.25);
        let _ = rgmin_solver_add_equality(null, idx.as_ptr(), coef.as_ptr(), 2, 0.0);
        let _ = rgmin_solver_clear_equalities(null);
        rgmin_solver_forget(null);
    }
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

/// Token 16 / set_euclidean_complex keeps the interleaved C^n packing.
/// Retraction is translation: the point stays in C^n and is not
/// sphere-normalized.
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

/// Token 17 / set_constant keeps the singleton packing. Retraction
/// is the fixed point: the iterate does not move and is not
/// sphere-normalized. Tokens 7-10 stay unused.
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

/// Token 18 / set_multinomial_ds stays on the Birkhoff polytope.
/// Retraction is Sinkhorn, not sphere-normalization. Tokens 7-10 unused.
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

/// Token 19 / set_multinomial_sym stays on the symmetric Birkhoff set.
/// Retraction is Sinkhorn+symm, not sphere-normalization. Tokens 7-10 unused.
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

/// Token 20 / set_sphere_complex stays on the complex unit sphere.
/// Reserved tokens 7-10 unused.
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

/// Token 21 / set_positive stays in the open positive orthant.
/// Reserved tokens 7-10 unused.
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

/// Token 22 / set_centered_matrix stays on the centered-column set.
/// Reserved tokens 7-10 unused.
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

/// HiGHS box / trust / equality setters share the set_highs status
/// convention: 0 with the feature, 1 without, 1 on a null solver.
#[test]
fn c_abi_set_box_status_matches_highs_feature() {
    use rgmin::ffi::{
        rgmin_solver_add_equality, rgmin_solver_clear_equalities, rgmin_solver_set_box,
        rgmin_solver_set_highs, rgmin_solver_set_trust,
    };
    let ctrl = rgmin_control_t {
        maxiter: 4,
        gtol: 1e-8,
        istep: 0.1,
        memory: 4,
        maxmove: 0.0,
    };
    let session = unsafe { rgmin_solver_create(rgmin_method_t::RGMIN_LBFGS, &ctrl, 2) };
    assert!(!session.is_null());
    let lo = [-1.0_f64, -2.0];
    let hi = [1.0_f64, 2.0];
    let idx = [0usize, 1];
    let coef = [1.0_f64, -1.0];
    let st_highs = unsafe { rgmin_solver_set_highs(session, 1) };
    let st_box = unsafe { rgmin_solver_set_box(session, lo.as_ptr(), hi.as_ptr(), 2) };
    let st_null_side = unsafe { rgmin_solver_set_box(session, std::ptr::null(), hi.as_ptr(), 2) };
    let st_clear = unsafe { rgmin_solver_set_box(session, std::ptr::null(), std::ptr::null(), 0) };
    let st_bad_n = unsafe { rgmin_solver_set_box(session, lo.as_ptr(), hi.as_ptr(), 3) };
    let st_trust = unsafe { rgmin_solver_set_trust(session, 0.25) };
    let st_eq = unsafe { rgmin_solver_add_equality(session, idx.as_ptr(), coef.as_ptr(), 2, 0.0) };
    let st_eq_clear = unsafe { rgmin_solver_clear_equalities(session) };
    unsafe { rgmin_solver_free(session) };
    let null = std::ptr::null_mut();
    let st_null = unsafe { rgmin_solver_set_box(null, lo.as_ptr(), hi.as_ptr(), 2) };
    assert_eq!(st_null, 1);
    #[cfg(feature = "highs")]
    {
        assert_eq!(st_highs, 0);
        assert_eq!(st_box, 0);
        assert_eq!(st_null_side, 0);
        assert_eq!(st_clear, 0);
        assert_eq!(st_bad_n, 1);
        assert_eq!(st_trust, 0);
        assert_eq!(st_eq, 0);
        assert_eq!(st_eq_clear, 0);
    }
    #[cfg(not(feature = "highs"))]
    {
        assert_eq!(st_highs, 1);
        assert_eq!(st_box, 1);
        assert_eq!(st_null_side, 1);
        assert_eq!(st_clear, 1);
        assert_eq!(st_bad_n, 1);
        assert_eq!(st_trust, 1);
        assert_eq!(st_eq, 1);
        assert_eq!(st_eq_clear, 1);
    }
}

#[test]
fn c_abi_one_eval_search_direction_is_steepest_when_empty() {
    use rgmin::ffi::{
        rgmin_solver_create, rgmin_solver_free, rgmin_solver_push_pair,
        rgmin_solver_search_direction, rgmin_status_t,
    };
    let ctrl = rgmin_control_t {
        maxiter: 1,
        gtol: 0.0,
        istep: 1.0,
        memory: 4,
        maxmove: 0.0,
    };
    let session = unsafe { rgmin_solver_create(rgmin_method_t::RGMIN_LBFGS, &ctrl, 2) };
    assert!(!session.is_null());
    let g = [2.0_f64, 0.0];
    let mut dir = [0.0_f64, 0.0];
    let st = unsafe { rgmin_solver_search_direction(session, g.as_ptr(), dir.as_mut_ptr(), 2) };
    assert_eq!(st, rgmin_status_t::RGMIN_SUCCESS);
    assert!(
        (dir[0] + 2.0).abs() < 1e-12,
        "empty two-loop is -g, got {dir:?}"
    );
    assert!(dir[1].abs() < 1e-12);
    let s = [-0.02_f64, 0.0];
    let y = [-0.04_f64, 0.0];
    assert_eq!(
        unsafe { rgmin_solver_push_pair(session, s.as_ptr(), y.as_ptr(), 2) },
        0
    );
    unsafe { rgmin_solver_free(session) };
}

#[test]
fn c_abi_pair_count_tracks_accepted_curvature_and_forget() {
    use rgmin::ffi::{
        rgmin_solver_create, rgmin_solver_forget, rgmin_solver_free, rgmin_solver_pair_count,
        rgmin_solver_push_pair,
    };
    let ctrl = rgmin_control_t {
        maxiter: 1,
        gtol: 0.0,
        istep: 1.0,
        memory: 2,
        maxmove: 0.0,
    };
    let session = unsafe { rgmin_solver_create(rgmin_method_t::RGMIN_LBFGS, &ctrl, 2) };
    assert!(!session.is_null());
    assert_eq!(unsafe { rgmin_solver_pair_count(session) }, 0);
    let s = [1.0_f64, 0.0];
    for y in [[0.0_f64, 0.0], [-1.0, 0.0]] {
        assert_eq!(
            unsafe { rgmin_solver_push_pair(session, s.as_ptr(), y.as_ptr(), 2) },
            0
        );
        assert_eq!(unsafe { rgmin_solver_pair_count(session) }, 0);
    }
    let positive = [2.0_f64, 0.0];
    for expected in [1, 2, 2] {
        assert_eq!(
            unsafe { rgmin_solver_push_pair(session, s.as_ptr(), positive.as_ptr(), 2) },
            0
        );
        assert_eq!(unsafe { rgmin_solver_pair_count(session) }, expected);
    }
    unsafe { rgmin_solver_forget(session) };
    assert_eq!(unsafe { rgmin_solver_pair_count(session) }, 0);
    unsafe { rgmin_solver_free(session) };
    assert_eq!(unsafe { rgmin_solver_pair_count(std::ptr::null()) }, 0);
}

#[test]
fn c_abi_rebase_keeps_pairs_and_forget_drops_them() {
    use rgmin::ffi::{
        rgmin_solver_create, rgmin_solver_forget, rgmin_solver_free, rgmin_solver_pair_count,
        rgmin_solver_push_pair, rgmin_solver_rebase,
    };
    let ctrl = rgmin_control_t {
        maxiter: 1,
        gtol: 0.0,
        istep: 1.0,
        memory: 2,
        maxmove: 0.0,
    };
    let session = unsafe { rgmin_solver_create(rgmin_method_t::RGMIN_LBFGS, &ctrl, 2) };
    assert!(!session.is_null());
    let s = [1.0_f64, 0.0];
    let y = [2.0_f64, 0.0];
    for _ in 0..2 {
        assert_eq!(
            unsafe { rgmin_solver_push_pair(session, s.as_ptr(), y.as_ptr(), 2) },
            0
        );
    }
    assert_eq!(unsafe { rgmin_solver_pair_count(session) }, 2);
    unsafe { rgmin_solver_rebase(session) };
    assert_eq!(unsafe { rgmin_solver_pair_count(session) }, 2);
    unsafe { rgmin_solver_forget(session) };
    assert_eq!(unsafe { rgmin_solver_pair_count(session) }, 0);
    unsafe { rgmin_solver_free(session) };
    unsafe { rgmin_solver_rebase(std::ptr::null_mut()) };
}

struct BowlCount {
    grads: AtomicUsize,
    hvps: AtomicUsize,
}

unsafe extern "C" fn bowl_eval(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
) -> rgmin_status_t {
    let (p, n) = unsafe { cpu_f64(x) };
    let mut f = 0.0;
    for i in 0..n {
        let v = unsafe { *p.add(i) };
        f += 0.5 * v * v;
    }
    unsafe { *value_out = f };
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn bowl_grad(
    user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    g: *mut DLManagedTensorVersioned,
) -> rgmin_status_t {
    let count = unsafe { &*(user as *const BowlCount) };
    count.grads.fetch_add(1, Ordering::Relaxed);
    let (p, n) = unsafe { cpu_f64(x) };
    let (gp, gn) = unsafe { cpu_f64(g) };
    assert_eq!(n, gn);
    for i in 0..n {
        unsafe { *(gp as *mut f64).add(i) = *p.add(i) };
    }
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn bowl_hvp(
    user: *mut c_void,
    _x: *const DLManagedTensorVersioned,
    v: *const DLManagedTensorVersioned,
    hv: *mut DLManagedTensorVersioned,
) -> rgmin_status_t {
    let count = unsafe { &*(user as *const BowlCount) };
    count.hvps.fetch_add(1, Ordering::Relaxed);
    let (vp, n) = unsafe { cpu_f64(v) };
    let (hp, hn) = unsafe { cpu_f64(hv) };
    assert_eq!(n, hn);
    for i in 0..n {
        unsafe { *(hp as *mut f64).add(i) = *vp.add(i) };
    }
    rgmin_status_t::RGMIN_SUCCESS
}

#[test]
fn newton_cg_method_string_and_counters_cross_the_c_abi() {
    let mut tag = rgmin_method_t::RGMIN_STEEPEST;
    let cg = std::ffi::CString::new("newton-cg").unwrap();
    assert_eq!(
        unsafe { rgmin_method_from_name(cg.as_ptr(), &mut tag) },
        rgmin_status_t::RGMIN_SUCCESS
    );
    assert_eq!(tag, rgmin_method_t::RGMIN_NEWTON_CG);
    let pcg = std::ffi::CString::new("Newton-PCG").unwrap();
    assert_eq!(
        unsafe { rgmin_method_from_name(pcg.as_ptr(), &mut tag) },
        rgmin_status_t::RGMIN_SUCCESS
    );
    assert_eq!(tag, rgmin_method_t::RGMIN_NEWTON_PCG);
    let bad = std::ffi::CString::new("not-a-solver").unwrap();
    assert_eq!(
        unsafe { rgmin_method_from_name(bad.as_ptr(), &mut tag) },
        rgmin_status_t::RGMIN_INVALID_PARAMETER
    );
    assert_eq!(
        unsafe { rgmin_method_from_name(std::ptr::null(), &mut tag) },
        rgmin_status_t::RGMIN_INVALID_PARAMETER
    );
    assert_eq!(unsafe { rgmin_solver_gradient_calls(std::ptr::null()) }, 0);
    assert_eq!(unsafe { rgmin_solver_hvp_calls(std::ptr::null()) }, 0);

    let count = BowlCount {
        grads: AtomicUsize::new(0),
        hvps: AtomicUsize::new(0),
    };
    let ctrl = rgmin_control_t {
        maxiter: 20,
        gtol: 1e-10,
        istep: 0.1,
        memory: 0,
        maxmove: 0.0,
    };
    let user = &count as *const BowlCount as *mut c_void;
    let session = unsafe { rgmin_solver_create(rgmin_method_t::RGMIN_NEWTON_CG, &ctrl, 4) };
    assert!(!session.is_null());
    let mut x = [1.0_f64, -0.5, 0.25, 0.1];
    let mut out = rgmin_report_t {
        value: 0.0,
        steps: 0,
        grad_norm: 0.0,
    };
    for _ in 0..20 {
        let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 4) };
        let st = unsafe {
            rgmin_solver_step(
                session,
                Some(bowl_eval),
                Some(bowl_grad),
                user,
                xt,
                &mut out,
            )
        };
        unsafe { rgmin_tensor_free(xt) };
        assert_eq!(st, rgmin_status_t::RGMIN_SUCCESS);
        if out.grad_norm < 1e-8 {
            break;
        }
    }
    assert!(out.grad_norm < 1e-8, "|g| {}", out.grad_norm);
    let grads = unsafe { rgmin_solver_gradient_calls(session) };
    let hvps = unsafe { rgmin_solver_hvp_calls(session) };
    assert!(grads > 0 && hvps > 0, "grads {grads} hvps {hvps}");
    assert_eq!(count.grads.load(Ordering::Relaxed), grads + hvps);
    unsafe { rgmin_solver_free(session) };

    count.grads.store(0, Ordering::Relaxed);
    count.hvps.store(0, Ordering::Relaxed);
    let session = unsafe { rgmin_solver_create(rgmin_method_t::RGMIN_NEWTON_CG, &ctrl, 4) };
    let mut x = [1.0_f64, -0.5, 0.25, 0.1];
    for _ in 0..20 {
        let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 4) };
        let st = unsafe {
            rgmin_solver_step_hvp(
                session,
                Some(bowl_eval),
                Some(bowl_grad),
                Some(bowl_hvp),
                user,
                xt,
                &mut out,
            )
        };
        unsafe { rgmin_tensor_free(xt) };
        assert_eq!(st, rgmin_status_t::RGMIN_SUCCESS);
        if out.grad_norm < 1e-8 {
            break;
        }
    }
    assert!(out.grad_norm < 1e-8, "|g| {}", out.grad_norm);
    let grads = unsafe { rgmin_solver_gradient_calls(session) };
    let hvps = unsafe { rgmin_solver_hvp_calls(session) };
    assert_eq!(count.grads.load(Ordering::Relaxed), grads);
    assert_eq!(count.hvps.load(Ordering::Relaxed), hvps);
    assert!(hvps > 0, "analytic products were not counted");
    unsafe { rgmin_solver_free(session) };
}

unsafe extern "C" fn exp_chain_eval(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
) -> rgmin_status_t {
    let (p, n) = unsafe { cpu_f64(x) };
    let mut v = 0.0;
    for i in 0..n {
        let c = unsafe { *p.add(i) };
        v += c * c;
    }
    unsafe { *value_out = 0.5 * v };
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn exp_chain_grad(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    g: *mut DLManagedTensorVersioned,
) -> rgmin_status_t {
    let (p, n) = unsafe { cpu_f64(x) };
    let (gp, gn) = unsafe { cpu_f64(g as *const _) };
    assert_eq!(gn, n);
    for i in 0..n {
        unsafe { *(gp as *mut f64).add(i) = *p.add(i) };
    }
    rgmin_status_t::RGMIN_SUCCESS
}

#[test]
fn c_abi_exp_installs_and_steps() {
    let ctrl = rgmin_control_t {
        maxiter: 1,
        gtol: 1e-12,
        istep: 1.0,
        memory: 4,
        maxmove: 0.0,
    };
    let bad = unsafe { rgmin_solver_create(rgmin_method_t::RGMIN_LBFGS, &ctrl, 2) };
    let params = rgmin_exp_t {
        a: f64::NAN,
        c_stab: f64::NAN,
        mu: 2.0,
        r_nn: 1.0,
        r_cut: 0.0,
        length_x: 0.0,
        length_y: 0.0,
        length_z: 0.0,
        recalc_mu: 0,
    };
    let pair_i = [0u32];
    let pair_j = [1u32];
    let dist = [1.0f64];
    let st = unsafe {
        rgmin_solver_set_exp(
            bad,
            &params,
            pair_i.as_ptr(),
            pair_j.as_ptr(),
            dist.as_ptr(),
            1,
            std::ptr::null(),
            std::ptr::null(),
            0,
        )
    };
    assert_eq!(st, rgmin_status_t::RGMIN_INVALID_PARAMETER);
    unsafe { rgmin_solver_free(bad) };

    let session = unsafe { rgmin_solver_create(rgmin_method_t::RGMIN_LBFGS, &ctrl, 6) };
    assert!(!session.is_null());
    let st = unsafe {
        rgmin_solver_set_exp(
            session,
            &params,
            pair_i.as_ptr(),
            pair_j.as_ptr(),
            dist.as_ptr(),
            1,
            std::ptr::null(),
            std::ptr::null(),
            0,
        )
    };
    assert_eq!(st, rgmin_status_t::RGMIN_SUCCESS);
    let mu = unsafe { rgmin_solver_exp_mu(session) };
    assert!((mu - 2.0).abs() < 1e-12, "mu {mu}");

    let mut x = [0.4, 0.1, -0.2, -0.3, 0.2, 0.05];
    let mut out = rgmin_report_t {
        value: 0.0,
        steps: 0,
        grad_norm: 0.0,
    };
    let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 6) };
    let st = unsafe {
        rgmin_solver_step(
            session,
            Some(exp_chain_eval),
            Some(exp_chain_grad),
            std::ptr::null_mut(),
            xt,
            &mut out,
        )
    };
    unsafe { rgmin_tensor_free(xt) };
    assert_eq!(st, rgmin_status_t::RGMIN_SUCCESS);
    assert!(out.grad_norm < 1.0, "grad {}", out.grad_norm);
    unsafe {
        rgmin_solver_clear_exp(session);
        assert!(rgmin_solver_exp_mu(session).is_nan());
        rgmin_solver_free(session);
    }
}

/// Counts calls in `user`, a `(calls, saw a non-finite coordinate)` pair.
unsafe extern "C" fn sent_eval(
    user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
) -> rgmin_status_t {
    let sent = unsafe { &mut *(user as *mut (usize, bool)) };
    let (p, n) = unsafe { cpu_f64(x) };
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

/// A refused gradient reaches the solver as `(+inf, NaN)`. At the start no
/// direction exists, so every method refuses after one evaluation instead
/// of sending the callback a NaN or clipped-to-the-bound coordinate.
#[test]
fn c_abi_a_refused_start_gradient_sends_no_nan_coordinate() {
    use rgmin_method_t::*;
    let ctrl = scg_bowl_ctrl();
    for method in [
        RGMIN_STEEPEST,
        RGMIN_POLAK_RIBIERE,
        RGMIN_BB,
        RGMIN_BFGS,
        RGMIN_LBFGS,
        RGMIN_SR1,
        RGMIN_SR2,
        RGMIN_FIRE,
        RGMIN_FIRE2,
    ] {
        let mut x = [1.0, 1.0];
        let mut sent = (0usize, false);
        let user = &mut sent as *mut (usize, bool) as *mut c_void;
        let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 2) };
        let mut out = rgmin_report_t {
            value: 0.0,
            steps: 0,
            grad_norm: 0.0,
        };
        let st = unsafe {
            let session = rgmin_solver_create(method, &ctrl, 2);
            let st = [
                rgmin_minimize(
                    Some(sent_eval),
                    Some(refused_grad),
                    user,
                    xt,
                    &ctrl,
                    method,
                    &mut out,
                ),
                rgmin_solver_step(
                    session,
                    Some(sent_eval),
                    Some(refused_grad),
                    user,
                    xt,
                    &mut out,
                ),
            ];
            rgmin_solver_free(session);
            rgmin_tensor_free(xt);
            st
        };
        assert_eq!(st, [rgmin_status_t::RGMIN_INTERNAL_ERROR; 2], "{method:?}");
        assert_eq!((sent, x), ((2, false), [1.0, 1.0]), "{method:?}");
    }
}
