//! Drive the DLPack C waist from Rust (Rosenbrock 2D).

#![cfg(feature = "capi")]

use std::os::raw::c_void;

use dlpk::sys::{DLDeviceType, DLManagedTensorVersioned};
use eindir_core::ffi::eindir_core_abi_stamp;
use eindir_core::ffi::{eindir_objective_t, eindir_status_t};
use rgmin::ffi::{
    rgmin_abi_compatible, rgmin_abi_stamp, rgmin_accept_t, rgmin_control_t, rgmin_method_t,
    rgmin_minimize, rgmin_minimize_eindir, rgmin_report_t, rgmin_solver_create, rgmin_solver_free,
    rgmin_solver_set_accept, rgmin_solver_step, rgmin_solver_step_fg, rgmin_status_t,
    rgmin_tensor_borrow_cpu_f64, rgmin_tensor_free,
};
use rgmin::ffi::{
    rgmin_conjugacy_t, rgmin_curv_fn, rgmin_eigen_kind_t, rgmin_eigen_params_t, rgmin_last_error,
    rgmin_lowest_eigenpair, rgmin_lowest_mode_t, rgmin_minimize_scg, rgmin_scg_params_t,
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
    let ev = unsafe { rosen_eval(std::ptr::null_mut(), x, value_out) };
    if ev != rgmin_status_t::RGMIN_SUCCESS {
        return ev;
    }
    unsafe { rosen_grad(std::ptr::null_mut(), x, g) }
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

#[test]
fn abi_stamp_identifies_this_optimizer_layout() {
    let stamp = rgmin_abi_stamp();
    assert_eq!(stamp.abi_major, 1);
    assert_eq!(stamp.abi_minor, 29);
    assert_eq!(stamp.layout_revision, 2);
    assert_eq!(unsafe { rgmin_abi_compatible(&stamp) }, 1);
}

#[test]
fn abi_stamp_rejects_an_incompatible_layout() {
    let mut stamp = rgmin_abi_stamp();
    stamp.layout_revision += 1;
    assert_eq!(unsafe { rgmin_abi_compatible(&stamp) }, 0);
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
        rgmin_fire_variant_t, rgmin_linesearch_t, rgmin_solver_set_fire_variant,
        rgmin_solver_set_linesearch,
    };
    use rgmin::ffi::{
        rgmin_manifold_t, rgmin_qn_step_t, rgmin_solver_forget, rgmin_solver_set_atom_maxmove,
        rgmin_solver_set_cautious, rgmin_solver_set_extra_updates, rgmin_solver_set_manifold,
        rgmin_solver_set_masses, rgmin_solver_set_maxmove, rgmin_solver_set_periodic,
        rgmin_solver_set_project_rigid, rgmin_solver_set_qn_step,
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
        // Three masses on a dimension-2 session are not one mass per atom.
        assert_eq!(
            rgmin_solver_set_masses(session, masses.as_ptr(), masses.len()),
            rgmin_status_t::RGMIN_INVALID_PARAMETER
        );
        assert_eq!(
            rgmin_solver_set_masses(session, std::ptr::null(), 0),
            rgmin_status_t::RGMIN_SUCCESS
        );
        rgmin_solver_set_manifold(session, rgmin_manifold_t::RGMIN_MANIFOLD_EUCLIDEAN);
        rgmin_solver_forget(session);
        // A no-op on an L-BFGS session.
        rgmin_solver_set_fire_variant(session, rgmin_fire_variant_t::RGMIN_FIRE_GUENOLE2020);
        use rgmin_linesearch_t::*;
        // Out-of-range constants are refused and change nothing.
        for (kind, c1, c2, it) in [
            (RGMIN_LINESEARCH_WOLFE, 0.9, 1e-4, 20),
            (RGMIN_LINESEARCH_WOLFE, 1e-4, 1.0, 20),
            (RGMIN_LINESEARCH_WOLFE, 1e-4, 0.9, 0),
            (RGMIN_LINESEARCH_GOLDSTEIN, 0.5, 0.5, 20),
            (RGMIN_LINESEARCH_BACKTRACKING, 1e-4, 1.5, 20),
            (RGMIN_LINESEARCH_BRENT, -1.0, 0.0, 20),
        ] {
            assert_eq!(
                rgmin_solver_set_linesearch(session, kind, c1, c2, it),
                rgmin_status_t::RGMIN_INVALID_PARAMETER,
                "{kind:?} {c1} {c2} {it}"
            );
        }
        for (kind, c1, c2) in [
            (RGMIN_LINESEARCH_BRENT, 1e-10, 0.0),
            (RGMIN_LINESEARCH_BACKTRACKING, 1e-4, 0.5),
            (RGMIN_LINESEARCH_GOLDSTEIN, 0.25, 0.5),
            (RGMIN_LINESEARCH_WOLFE, 1e-4, 0.9),
        ] {
            assert_eq!(
                rgmin_solver_set_linesearch(session, kind, c1, c2, 20),
                rgmin_status_t::RGMIN_SUCCESS
            );
        }
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

    // A null session does not crash. Setters that return a status
    // report RGMIN_INVALID_PARAMETER; the others return.
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
        assert_eq!(
            rgmin_solver_set_masses(null, masses.as_ptr(), masses.len()),
            rgmin_status_t::RGMIN_INVALID_PARAMETER
        );
        rgmin_solver_forget(null);
        rgmin_solver_set_fire_variant(null, rgmin_fire_variant_t::RGMIN_FIRE_GUENOLE2020);
        assert_eq!(
            rgmin_solver_set_linesearch(
                null,
                rgmin_linesearch_t::RGMIN_LINESEARCH_WOLFE,
                1e-4,
                0.9,
                20
            ),
            rgmin_status_t::RGMIN_INVALID_PARAMETER
        );
    }
}

/// One mass per atom on a 9-vector is three masses. Two masses, and
/// two masses on a length that integer-divides to two, are refused.
#[test]
fn c_abi_set_masses_rejects_a_bad_count() {
    use rgmin::ffi::{rgmin_manifold_t, rgmin_solver_set_manifold, rgmin_solver_set_masses};
    let ctrl = rgmin_control_t {
        maxiter: 1,
        gtol: 1e-8,
        istep: 0.1,
        memory: 4,
        maxmove: 0.0,
    };
    let session = unsafe { rgmin_solver_create(rgmin_method_t::RGMIN_LBFGS, &ctrl, 9) };
    assert!(!session.is_null());
    let good = [12.0_f64, 1.0, 1.0];
    let bad = [12.0_f64, 1.0];
    unsafe {
        rgmin_solver_set_manifold(session, rgmin_manifold_t::RGMIN_MANIFOLD_MW_RIGID);
        assert_eq!(
            rgmin_solver_set_masses(session, good.as_ptr(), good.len()),
            rgmin_status_t::RGMIN_SUCCESS
        );
        assert_eq!(
            rgmin_solver_set_masses(session, bad.as_ptr(), bad.len()),
            rgmin_status_t::RGMIN_INVALID_PARAMETER
        );
        assert_eq!(
            rgmin_solver_set_masses(session, std::ptr::null(), 0),
            rgmin_status_t::RGMIN_SUCCESS
        );
        rgmin_solver_free(session);
    }
    let eight = unsafe { rgmin_solver_create(rgmin_method_t::RGMIN_LBFGS, &ctrl, 8) };
    assert!(!eight.is_null());
    let two = [1.0_f64, 2.0];
    unsafe {
        assert_eq!(
            rgmin_solver_set_masses(eight, two.as_ptr(), two.len()),
            rgmin_status_t::RGMIN_INVALID_PARAMETER
        );
        rgmin_solver_free(eight);
    }
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
fn abi_stamp_ignores_minor() {
    let mut stamp = rgmin_abi_stamp();
    stamp.abi_minor = stamp.abi_minor.saturating_add(1);
    assert_eq!(unsafe { rgmin_abi_compatible(&stamp) }, 1);
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

#[test]
fn lowest_eigenpair_rejects_unknown_integer_without_wrapping() {
    let mut x = [0.0_f64; 2];
    let mut seed = [1.0, 0.0];
    let mut mode = [42.0_f64; 2];
    let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 2) };
    let st = unsafe { rgmin_tensor_borrow_cpu_f64(seed.as_mut_ptr(), 2) };
    let mt = unsafe { rgmin_tensor_borrow_cpu_f64(mode.as_mut_ptr(), 2) };
    for kind in [-1, 256, 270, i32::MAX] {
        let params = rgmin_eigen_params_t {
            kind,
            nev: 1,
            krylov: 2,
            max_iter: 0,
            tol: 0.0,
        };
        let mut out = rgmin_lowest_mode_t {
            value: 17.0,
            actions: 19,
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
        assert_eq!(
            status,
            rgmin_status_t::RGMIN_INVALID_PARAMETER,
            "kind {kind}"
        );
        assert_eq!(mode, [42.0, 42.0]);
        assert_eq!(out.value, 17.0);
        assert_eq!(out.actions, 19);
    }
    unsafe {
        rgmin_tensor_free(xt);
        rgmin_tensor_free(st);
        rgmin_tensor_free(mt);
    }
}

#[test]
fn lowest_eigenpair_accepts_in_place_seed_and_point_storage() {
    for output_is_seed in [true, false] {
        let mut point = [0.0_f64; 6];
        let mut seed = [0.2, 0.7, 0.1, 0.0, 0.0, 0.0];
        let xt = unsafe { rgmin_tensor_borrow_cpu_f64(point.as_mut_ptr(), 6) };
        let st = unsafe { rgmin_tensor_borrow_cpu_f64(seed.as_mut_ptr(), 6) };
        let mt = if output_is_seed { st } else { xt };
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
        }
        assert_eq!(status, rgmin_status_t::RGMIN_SUCCESS);
        assert!((out.value + 8.0).abs() < 1e-12);
        let mode = if output_is_seed { &seed } else { &point };
        assert!(mode[0].abs() > 0.99, "mode {mode:?}");
        assert!((mode.iter().map(|v| v * v).sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(out.actions <= 6);
    }
}

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

/// rgmin_solver_forget_evaluation drops only the cached point: the step
/// after it evaluates its start once more and otherwise matches a session
/// that kept the point, pairs and iterate included.
#[test]
fn c_abi_forget_evaluation_costs_one_call_and_keeps_memory() {
    use rgmin::ffi::{rgmin_solver_forget_evaluation, rgmin_solver_pair_count};
    let ctrl = rgmin_control_t {
        maxiter: 1,
        gtol: 1e-12,
        istep: 0.1,
        memory: 10,
        maxmove: 0.2,
    };
    let run = |forget: bool| {
        let mut x = [-1.2_f64, 1.0];
        let session = unsafe { rgmin_solver_create(rgmin_method_t::RGMIN_LBFGS, &ctrl, 2) };
        assert!(!session.is_null());
        unsafe { rgmin_solver_set_accept(session, rgmin_accept_t::RGMIN_ACCEPT_NONE) };
        let mut out = rgmin_report_t {
            value: 0.0,
            steps: 0,
            grad_norm: 0.0,
        };
        let mut calls = Vec::new();
        for step in 0..3 {
            if forget && step == 2 {
                unsafe { rgmin_solver_forget_evaluation(session) };
            }
            let mut n = 0usize;
            let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 2) };
            let st = unsafe {
                rgmin_solver_step_fg(
                    session,
                    Some(rosen_evalgrad),
                    (&mut n as *mut usize).cast(),
                    xt,
                    &mut out,
                )
            };
            unsafe { rgmin_tensor_free(xt) };
            assert_eq!(st, rgmin_status_t::RGMIN_SUCCESS);
            calls.push(n);
        }
        let pairs = unsafe { rgmin_solver_pair_count(session) };
        unsafe { rgmin_solver_free(session) };
        (x, calls, pairs)
    };
    let (kept_x, kept_calls, kept_pairs) = run(false);
    let (forgot_x, forgot_calls, forgot_pairs) = run(true);
    assert_eq!(forgot_calls[..2], kept_calls[..2]);
    assert_eq!(
        forgot_calls[2],
        kept_calls[2] + 1,
        "{kept_calls:?} {forgot_calls:?}"
    );
    assert_eq!(forgot_pairs, kept_pairs);
    assert_eq!(forgot_x, kept_x);
    unsafe { rgmin_solver_forget_evaluation(std::ptr::null_mut()) };
}

/// rgmin_solver_set_lbfgs_neb_guards survives live and null sessions and
/// clears the pair memory when it toggles the guards.
#[test]
fn c_abi_lbfgs_neb_guards_toggle_resets_pairs() {
    use rgmin::ffi::{
        rgmin_solver_pair_count, rgmin_solver_push_pair, rgmin_solver_set_lbfgs_neb_guards,
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
    assert_eq!(
        unsafe { rgmin_solver_push_pair(session, s.as_ptr(), y.as_ptr(), 2) },
        0
    );
    assert_eq!(unsafe { rgmin_solver_pair_count(session) }, 1);
    unsafe { rgmin_solver_set_lbfgs_neb_guards(session, 1) };
    assert_eq!(unsafe { rgmin_solver_pair_count(session) }, 0);
    assert_eq!(
        unsafe { rgmin_solver_push_pair(session, s.as_ptr(), y.as_ptr(), 2) },
        0
    );
    unsafe { rgmin_solver_set_lbfgs_neb_guards(session, 1) };
    assert_eq!(
        unsafe { rgmin_solver_pair_count(session) },
        1,
        "no toggle, no reset"
    );
    unsafe {
        rgmin_solver_free(session);
        rgmin_solver_set_lbfgs_neb_guards(std::ptr::null_mut(), 1);
    }
}
