#![cfg(feature = "capi")]

use std::ffi::CStr;
use std::os::raw::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};

use dlpk::sys::DLManagedTensorVersioned;
use eindir_core::ffi::{eindir_core_abi_stamp, eindir_objective_t, eindir_status_t};
use rgmin::ffi::{
    rgmin_abi_compatible, rgmin_abi_stamp, rgmin_control_t, rgmin_eigen_kind_t,
    rgmin_eigen_params_t, rgmin_eigenpair_t, rgmin_last_error, rgmin_lowest_eigenpairs,
    rgmin_method_from_name, rgmin_method_t, rgmin_minimize, rgmin_minimize_eindir,
    rgmin_minimize_hess, rgmin_minimize_hess_opts, rgmin_minimize_hvp, rgmin_minimize_hvp_opts,
    rgmin_minimize_scg, rgmin_minimize_scg_with_options, rgmin_report_t, rgmin_scg_params_t,
    rgmin_status_t, rgmin_stop_t, rgmin_tensor_borrow_cpu_f64, rgmin_tensor_free,
};

unsafe extern "C" fn square_eval(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    value: *mut f64,
) -> rgmin_status_t {
    let data = unsafe { (*x).dl_tensor.data.cast::<f64>() };
    unsafe { *value = *data * *data };
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn square_grad(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    grad: *mut DLManagedTensorVersioned,
) -> rgmin_status_t {
    let data = unsafe { (*x).dl_tensor.data.cast::<f64>() };
    let output = unsafe { (*grad).dl_tensor.data.cast::<f64>() };
    unsafe { *output = 2.0 * *data };
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn square_hess(
    _user: *mut c_void,
    _x: *const DLManagedTensorVersioned,
    hess: *mut DLManagedTensorVersioned,
) -> rgmin_status_t {
    let n = unsafe { *(*hess).dl_tensor.shape as usize };
    let output = unsafe { (*hess).dl_tensor.data.cast::<f64>() };
    for i in 0..n {
        unsafe { *output.add(i) = 2.0 };
    }
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn square_hvp(
    _user: *mut c_void,
    _x: *const DLManagedTensorVersioned,
    v: *const DLManagedTensorVersioned,
    hv: *mut DLManagedTensorVersioned,
) -> rgmin_status_t {
    let n = unsafe { *(*v).dl_tensor.shape as usize };
    let input = unsafe { (*v).dl_tensor.data.cast::<f64>() };
    let output = unsafe { (*hv).dl_tensor.data.cast::<f64>() };
    for i in 0..n {
        unsafe { *output.add(i) = 2.0 * *input.add(i) };
    }
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn eindir_eval(
    _user_data: *mut c_void,
    x: *const DLManagedTensorVersioned,
    value: *mut f64,
) -> eindir_status_t {
    let data = unsafe { (*x).dl_tensor.data.cast::<f64>() };
    unsafe { *value = *data * *data };
    eindir_status_t::EINDIR_SUCCESS
}

unsafe extern "C" fn eindir_grad(
    _user_data: *mut c_void,
    x: *const DLManagedTensorVersioned,
    grad: *mut DLManagedTensorVersioned,
) -> eindir_status_t {
    let n = unsafe { *(*grad).dl_tensor.shape as usize };
    let data = unsafe { (*x).dl_tensor.data.cast::<f64>() };
    let output = unsafe { (*grad).dl_tensor.data.cast::<f64>() };
    for i in 0..n {
        unsafe { *output.add(i) = 2.0 * *data.add(i) };
    }
    eindir_status_t::EINDIR_SUCCESS
}

fn control() -> rgmin_control_t {
    rgmin_control_t {
        maxiter: 0,
        gtol: f64::EPSILON,
        istep: 1.0,
        memory: 1,
        maxmove: 0.0,
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

#[test]
fn null_entry_arguments_return_typed_failures() {
    let ctrl = control();
    let mut out = report();
    let mut value = [1.0];
    let tensor = unsafe { rgmin_tensor_borrow_cpu_f64(value.as_mut_ptr(), value.len()) };
    assert!(!tensor.is_null());

    assert_eq!(
        unsafe {
            rgmin_minimize(
                None,
                Some(square_grad),
                std::ptr::null_mut(),
                tensor,
                &ctrl,
                rgmin_method_t::RGMIN_STEEPEST,
                &mut out,
            )
        },
        rgmin_status_t::RGMIN_INVALID_PARAMETER
    );
    assert_eq!(
        unsafe {
            rgmin_minimize(
                Some(square_eval),
                None,
                std::ptr::null_mut(),
                tensor,
                &ctrl,
                rgmin_method_t::RGMIN_STEEPEST,
                &mut out,
            )
        },
        rgmin_status_t::RGMIN_INVALID_PARAMETER
    );
    assert_eq!(
        unsafe {
            rgmin_minimize(
                Some(square_eval),
                Some(square_grad),
                std::ptr::null_mut(),
                tensor,
                std::ptr::null(),
                rgmin_method_t::RGMIN_STEEPEST,
                &mut out,
            )
        },
        rgmin_status_t::RGMIN_INVALID_PARAMETER
    );
    assert_eq!(
        unsafe {
            rgmin_minimize(
                Some(square_eval),
                Some(square_grad),
                std::ptr::null_mut(),
                tensor,
                &ctrl,
                rgmin_method_t::RGMIN_STEEPEST,
                std::ptr::null_mut(),
            )
        },
        rgmin_status_t::RGMIN_INVALID_PARAMETER
    );
    assert_eq!(
        unsafe {
            rgmin_minimize(
                Some(square_eval),
                Some(square_grad),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &ctrl,
                rgmin_method_t::RGMIN_STEEPEST,
                &mut out,
            )
        },
        rgmin_status_t::RGMIN_INVALID_PARAMETER
    );

    unsafe {
        rgmin_tensor_free(tensor);
        rgmin_tensor_free(std::ptr::null_mut());
    }
}

#[test]
fn tensor_helpers_and_method_parser_reject_nulls() {
    assert!(unsafe { rgmin_tensor_borrow_cpu_f64(std::ptr::null_mut(), 1) }.is_null());
    let mut value = 0.0;
    assert!(unsafe { rgmin_tensor_borrow_cpu_f64(&mut value, 0) }.is_null());

    let mut method = rgmin_method_t::RGMIN_STEEPEST;
    assert_eq!(
        unsafe { rgmin_method_from_name(std::ptr::null(), &mut method) },
        rgmin_status_t::RGMIN_INVALID_PARAMETER
    );
    assert_eq!(
        unsafe { rgmin_method_from_name(c"lbfgs".as_ptr(), std::ptr::null_mut()) },
        rgmin_status_t::RGMIN_INVALID_PARAMETER
    );
}

#[test]
fn malformed_tensor_sizes_and_layouts_are_rejected() {
    let ctrl = control();
    let mut out = report();
    let mut values = [1.0, 2.0];
    let tensor = unsafe { rgmin_tensor_borrow_cpu_f64(values.as_mut_ptr(), values.len()) };
    assert!(!tensor.is_null());

    let shape = unsafe { (*tensor).dl_tensor.shape };
    let saved_shape = unsafe { *shape };
    unsafe { *shape = 0 };
    assert_eq!(
        unsafe {
            rgmin_minimize(
                Some(square_eval),
                Some(square_grad),
                std::ptr::null_mut(),
                tensor,
                &ctrl,
                rgmin_method_t::RGMIN_STEEPEST,
                &mut out,
            )
        },
        rgmin_status_t::RGMIN_INVALID_PARAMETER
    );
    unsafe { *shape = saved_shape };

    unsafe { (*tensor).dl_tensor.ndim = 2 };
    assert_eq!(
        unsafe {
            rgmin_minimize(
                Some(square_eval),
                Some(square_grad),
                std::ptr::null_mut(),
                tensor,
                &ctrl,
                rgmin_method_t::RGMIN_STEEPEST,
                &mut out,
            )
        },
        rgmin_status_t::RGMIN_INVALID_PARAMETER
    );
    unsafe { (*tensor).dl_tensor.ndim = 1 };

    let stride = unsafe { (*tensor).dl_tensor.strides };
    let saved_stride = unsafe { *stride };
    unsafe { *stride = 2 };
    assert_eq!(
        unsafe {
            rgmin_minimize(
                Some(square_eval),
                Some(square_grad),
                std::ptr::null_mut(),
                tensor,
                &ctrl,
                rgmin_method_t::RGMIN_STEEPEST,
                &mut out,
            )
        },
        rgmin_status_t::RGMIN_INVALID_PARAMETER
    );
    unsafe {
        *stride = saved_stride;
        rgmin_tensor_free(tensor);
    }
}

#[test]
fn nonfinite_initial_values_are_reported() {
    for initial in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut ctrl = control();
        ctrl.maxiter = 1;
        let objective = eindir_objective_t {
            dim: 2,
            low: std::ptr::null_mut(),
            high: std::ptr::null_mut(),
            eval_fn: eindir_eval,
            grad_fn: Some(eindir_grad),
            user_data: std::ptr::null_mut(),
            free_fn: None,
        };
        let stamp = eindir_core_abi_stamp();
        for case in 0..6 {
            let mut values = [1.0, initial];
            let tensor = unsafe { rgmin_tensor_borrow_cpu_f64(values.as_mut_ptr(), values.len()) };
            let mut out = report();
            let status = unsafe {
                match case {
                    0 => rgmin_minimize(
                        Some(square_eval),
                        Some(square_grad),
                        std::ptr::null_mut(),
                        tensor,
                        &ctrl,
                        rgmin_method_t::RGMIN_STEEPEST,
                        &mut out,
                    ),
                    1 => rgmin_minimize(
                        Some(square_eval),
                        Some(square_grad),
                        std::ptr::null_mut(),
                        tensor,
                        &ctrl,
                        rgmin_method_t::RGMIN_PSO,
                        &mut out,
                    ),
                    2 => rgmin_minimize_hess(
                        Some(square_eval),
                        Some(square_grad),
                        Some(square_hess),
                        std::ptr::null_mut(),
                        tensor,
                        &ctrl,
                        rgmin_method_t::RGMIN_NEWTON,
                        &mut out,
                    ),
                    3 => rgmin_minimize_hvp(
                        Some(square_eval),
                        Some(square_grad),
                        Some(square_hvp),
                        std::ptr::null_mut(),
                        tensor,
                        &ctrl,
                        rgmin_method_t::RGMIN_NEWTON_CG,
                        &mut out,
                    ),
                    4 => rgmin_minimize_eindir(
                        &objective,
                        &stamp,
                        tensor,
                        &ctrl,
                        rgmin_method_t::RGMIN_STEEPEST,
                        &mut out,
                    ),
                    _ => rgmin_minimize_scg(
                        Some(square_eval),
                        Some(square_grad),
                        None,
                        std::ptr::null_mut(),
                        tensor,
                        &ctrl,
                        std::ptr::null(),
                        &mut out,
                    ),
                }
            };
            unsafe { rgmin_tensor_free(tensor) };
            assert_eq!(
                status,
                rgmin_status_t::RGMIN_INVALID_PARAMETER,
                "case {case}"
            );
            assert_eq!(values[0].to_bits(), 1.0_f64.to_bits(), "case {case}");
            assert_eq!(values[1].to_bits(), initial.to_bits(), "case {case}");
        }
    }
}

#[test]
fn incompatible_abi_and_dlpack_versions_are_rejected() {
    assert_eq!(unsafe { rgmin_abi_compatible(std::ptr::null()) }, 0);
    let current = rgmin_abi_stamp();
    let wrong_major = rgmin::ffi::rgmin_abi_stamp_t {
        abi_major: current.abi_major + 1,
        ..current
    };
    assert_eq!(unsafe { rgmin_abi_compatible(&wrong_major) }, 0);
    let wrong_layout = rgmin::ffi::rgmin_abi_stamp_t {
        layout_revision: current.layout_revision + 1,
        ..current
    };
    assert_eq!(unsafe { rgmin_abi_compatible(&wrong_layout) }, 0);

    for (major, minor) in [(2, 0), (1, 1)] {
        let mut value = [1.0];
        let tensor = unsafe { rgmin_tensor_borrow_cpu_f64(value.as_mut_ptr(), value.len()) };
        unsafe {
            (*tensor).version.major = major;
            (*tensor).version.minor = minor;
        }
        let mut out = report();
        let status = unsafe {
            rgmin_minimize(
                Some(square_eval),
                Some(square_grad),
                std::ptr::null_mut(),
                tensor,
                &control(),
                rgmin_method_t::RGMIN_STEEPEST,
                &mut out,
            )
        };
        unsafe { rgmin_tensor_free(tensor) };
        assert_eq!(status, rgmin_status_t::RGMIN_INVALID_PARAMETER);

        let mut x = [1.0];
        let mut seed = [1.0];
        let mut modes = [0.0];
        let mut values = [0.0];
        let mut pairs = [rgmin_eigenpair_t {
            value: 0.0,
            actions: 0,
        }];
        let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), x.len()) };
        let st = unsafe { rgmin_tensor_borrow_cpu_f64(seed.as_mut_ptr(), seed.len()) };
        let mt = unsafe { rgmin_tensor_borrow_cpu_f64(modes.as_mut_ptr(), modes.len()) };
        let vt = unsafe { rgmin_tensor_borrow_cpu_f64(values.as_mut_ptr(), values.len()) };
        unsafe {
            (*st).version.major = major;
            (*st).version.minor = minor;
        }
        let status = unsafe {
            rgmin_lowest_eigenpairs(
                Some(square_hvp),
                std::ptr::null_mut(),
                xt,
                st,
                mt,
                vt,
                std::ptr::null(),
                std::ptr::null(),
                pairs.as_mut_ptr(),
                1,
            )
        };
        unsafe {
            rgmin_tensor_free(xt);
            rgmin_tensor_free(st);
            rgmin_tensor_free(mt);
            rgmin_tensor_free(vt);
        }
        assert_eq!(status, rgmin_status_t::RGMIN_INVALID_PARAMETER);
    }
}

/// Counts host evaluations for one call. Each caller owns its counter and
/// hands it to the entry through the `user` pointer.
fn count_call(user: *mut c_void) {
    unsafe { &*user.cast::<AtomicUsize>() }.fetch_add(1, Ordering::Relaxed);
}

unsafe extern "C" fn counted_eval(
    user: *mut c_void,
    _x: *const DLManagedTensorVersioned,
    value: *mut f64,
) -> rgmin_status_t {
    count_call(user);
    unsafe { *value = 1.0 };
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn counted_grad(
    _user: *mut c_void,
    _x: *const DLManagedTensorVersioned,
    grad: *mut DLManagedTensorVersioned,
) -> rgmin_status_t {
    let output = unsafe { (*grad).dl_tensor.data.cast::<f64>() };
    unsafe { *output = 0.0 };
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn counted_hess(
    _user: *mut c_void,
    _x: *const DLManagedTensorVersioned,
    hess: *mut DLManagedTensorVersioned,
) -> rgmin_status_t {
    let output = unsafe { (*hess).dl_tensor.data.cast::<f64>() };
    unsafe { *output = 1.0 };
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn counted_hvp(
    _user: *mut c_void,
    _x: *const DLManagedTensorVersioned,
    _v: *const DLManagedTensorVersioned,
    hv: *mut DLManagedTensorVersioned,
) -> rgmin_status_t {
    let output = unsafe { (*hv).dl_tensor.data.cast::<f64>() };
    unsafe { *output = 0.0 };
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn counted_eindir_eval(
    user: *mut c_void,
    _x: *const DLManagedTensorVersioned,
    value: *mut f64,
) -> eindir_status_t {
    count_call(user);
    unsafe { *value = 1.0 };
    eindir_status_t::EINDIR_SUCCESS
}

unsafe extern "C" fn counted_eindir_grad(
    _user: *mut c_void,
    _x: *const DLManagedTensorVersioned,
    grad: *mut DLManagedTensorVersioned,
) -> eindir_status_t {
    let output = unsafe { (*grad).dl_tensor.data.cast::<f64>() };
    unsafe { *output = 0.0 };
    eindir_status_t::EINDIR_SUCCESS
}

fn last_error() -> String {
    unsafe { CStr::from_ptr(rgmin_last_error()) }
        .to_string_lossy()
        .into_owned()
}

/// A refused start leaves `x` alone, calls the host never, and names the
/// entry point that refused it.
fn assert_refused(
    entry: &str,
    initial: f64,
    status: rgmin_status_t,
    calls: &AtomicUsize,
    value: f64,
) {
    assert_eq!(
        status,
        rgmin_status_t::RGMIN_INVALID_PARAMETER,
        "{entry} {initial}"
    );
    assert_eq!(calls.load(Ordering::Relaxed), 0, "{entry} {initial}");
    assert_eq!(
        last_error(),
        format!("{entry}: x contains a non-finite value"),
        "{initial}"
    );
    assert_eq!(value.to_bits(), initial.to_bits(), "{entry}");
}

/// The four entries that used to skip the check must refuse a nonfinite
/// start the way `rgmin_minimize` does: `RGMIN_INVALID_PARAMETER`, an
/// unchanged `x`, and no host evaluation. The message names the entry
/// point the host called, as the other messages of that entry do.
#[test]
fn remaining_entries_reject_a_nonfinite_start() {
    for initial in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        refuse_minimize(initial);
        refuse_hess(initial);
        refuse_hvp(initial);
        refuse_eindir(initial);
        refuse_scg(initial);
    }
}

fn refuse_minimize(initial: f64) {
    let calls = AtomicUsize::new(0);
    let mut value = [initial];
    let tensor = unsafe { rgmin_tensor_borrow_cpu_f64(value.as_mut_ptr(), value.len()) };
    let mut out = report();
    let mut ctrl = control();
    ctrl.maxiter = 1;
    let status = unsafe {
        rgmin_minimize(
            Some(counted_eval),
            Some(counted_grad),
            std::ptr::from_ref(&calls).cast_mut().cast(),
            tensor,
            &ctrl,
            rgmin_method_t::RGMIN_STEEPEST,
            &mut out,
        )
    };
    unsafe { rgmin_tensor_free(tensor) };
    assert_refused("rgmin_minimize", initial, status, &calls, value[0]);
}

fn refuse_hess(initial: f64) {
    let calls = AtomicUsize::new(0);
    let mut value = [initial];
    let tensor = unsafe { rgmin_tensor_borrow_cpu_f64(value.as_mut_ptr(), value.len()) };
    let mut out = report();
    let mut ctrl = control();
    ctrl.maxiter = 1;
    let status = unsafe {
        rgmin_minimize_hess_opts(
            Some(counted_eval),
            Some(counted_grad),
            Some(counted_hess),
            std::ptr::from_ref(&calls).cast_mut().cast(),
            tensor,
            &ctrl,
            rgmin_method_t::RGMIN_NEWTON,
            0,
            0,
            &mut out,
        )
    };
    unsafe { rgmin_tensor_free(tensor) };
    assert_refused("rgmin_minimize_hess", initial, status, &calls, value[0]);
}

fn refuse_hvp(initial: f64) {
    let calls = AtomicUsize::new(0);
    let mut value = [initial];
    let tensor = unsafe { rgmin_tensor_borrow_cpu_f64(value.as_mut_ptr(), value.len()) };
    let mut out = report();
    let mut ctrl = control();
    ctrl.maxiter = 1;
    let status = unsafe {
        rgmin_minimize_hvp_opts(
            Some(counted_eval),
            Some(counted_grad),
            Some(counted_hvp),
            std::ptr::from_ref(&calls).cast_mut().cast(),
            tensor,
            &ctrl,
            rgmin_method_t::RGMIN_NEWTON_CG,
            0,
            0,
            &mut out,
        )
    };
    unsafe { rgmin_tensor_free(tensor) };
    assert_refused("rgmin_minimize_hvp", initial, status, &calls, value[0]);
}

fn refuse_eindir(initial: f64) {
    let calls = AtomicUsize::new(0);
    let mut value = [initial];
    let tensor = unsafe { rgmin_tensor_borrow_cpu_f64(value.as_mut_ptr(), value.len()) };
    let objective = Box::new(eindir_objective_t {
        dim: 1,
        low: std::ptr::null_mut(),
        high: std::ptr::null_mut(),
        eval_fn: counted_eindir_eval,
        grad_fn: Some(counted_eindir_grad),
        user_data: std::ptr::from_ref(&calls).cast_mut().cast(),
        free_fn: None,
    });
    let mut out = report();
    let mut ctrl = control();
    ctrl.maxiter = 1;
    let status = unsafe {
        rgmin_minimize_eindir(
            &*objective,
            &eindir_core_abi_stamp(),
            tensor,
            &ctrl,
            rgmin_method_t::RGMIN_STEEPEST,
            &mut out,
        )
    };
    unsafe { rgmin_tensor_free(tensor) };
    assert_refused("rgmin_minimize_eindir", initial, status, &calls, value[0]);
}

fn refuse_scg(initial: f64) {
    let calls = AtomicUsize::new(0);
    let mut value = [initial];
    let tensor = unsafe { rgmin_tensor_borrow_cpu_f64(value.as_mut_ptr(), value.len()) };
    let mut out = report();
    let mut ctrl = control();
    ctrl.maxiter = 1;
    let params = rgmin_scg_params_t::default();
    let status = unsafe {
        rgmin_minimize_scg_with_options(
            Some(counted_eval),
            Some(counted_grad),
            None,
            std::ptr::from_ref(&calls).cast_mut().cast(),
            tensor,
            &ctrl,
            &params,
            std::ptr::null(),
            &mut out,
        )
    };
    unsafe { rgmin_tensor_free(tensor) };
    assert_refused("rgmin_minimize_scg", initial, status, &calls, value[0]);
}

/// The diagonal operator of one block product test, and the number of
/// products it has applied. It travels through the `user` pointer so
/// each test counts its own calls.
struct BlockHvp {
    lam: [f64; 2],
    calls: AtomicUsize,
}

unsafe extern "C" fn counted_block_hvp(
    user: *mut c_void,
    _x: *const DLManagedTensorVersioned,
    v: *const DLManagedTensorVersioned,
    hv: *mut DLManagedTensorVersioned,
) -> rgmin_status_t {
    let operator = unsafe { &*user.cast::<BlockHvp>() };
    operator.calls.fetch_add(1, Ordering::Relaxed);
    let lam = operator.lam.as_ptr();
    let source = unsafe { (*v).dl_tensor.data.cast::<f64>() };
    let dest = unsafe { (*hv).dl_tensor.data.cast::<f64>() };
    unsafe {
        *dest = *lam * *source;
        *dest.add(1) = *lam.add(1) * *source.add(1);
    }
    rgmin_status_t::RGMIN_SUCCESS
}

/// `cpu_dl` imports the seed block. A version other than 1.0 must be
/// `RGMIN_INVALID_PARAMETER` with the same message as `cpu_f64_slice`,
/// before any product.
#[test]
fn block_import_rejects_a_foreign_dlpack_version() {
    let operator = BlockHvp {
        lam: [1.0, 4.0],
        calls: AtomicUsize::new(0),
    };
    let mut x = [0.0_f64; 2];
    let mut seed = [1.0_f64, 0.2];
    let mut modes = [0.0_f64; 2];
    let mut values = [0.0_f64; 1];
    let mut pairs = [rgmin_eigenpair_t {
        value: 0.0,
        actions: 0,
    }];
    let params = rgmin_eigen_params_t {
        kind: rgmin_eigen_kind_t::RGMIN_EIGEN_LOBPCG as i32,
        nev: 1,
        krylov: 0,
        max_iter: 20,
        tol: 1e-8,
    };
    for (major, minor) in [(2_u32, 0_u32), (1, 1)] {
        operator.calls.store(0, Ordering::Relaxed);
        let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 2) };
        let st = unsafe { rgmin_tensor_borrow_cpu_f64(seed.as_mut_ptr(), 2) };
        let mt = unsafe { rgmin_tensor_borrow_cpu_f64(modes.as_mut_ptr(), 2) };
        let vt = unsafe { rgmin_tensor_borrow_cpu_f64(values.as_mut_ptr(), 1) };
        unsafe {
            (*st).version.major = major;
            (*st).version.minor = minor;
        }
        let status = unsafe {
            rgmin_lowest_eigenpairs(
                Some(counted_block_hvp),
                std::ptr::from_ref(&operator).cast_mut().cast(),
                xt,
                st,
                mt,
                vt,
                &params,
                std::ptr::null(),
                pairs.as_mut_ptr(),
                1,
            )
        };
        unsafe {
            rgmin_tensor_free(xt);
            rgmin_tensor_free(st);
            rgmin_tensor_free(mt);
            rgmin_tensor_free(vt);
        }
        assert_eq!(
            status,
            rgmin_status_t::RGMIN_INVALID_PARAMETER,
            "version {major}.{minor}"
        );
        assert_eq!(operator.calls.load(Ordering::Relaxed), 0, "{major}.{minor}");
        assert!(
            last_error().contains("unsupported DLPack version"),
            "{}",
            last_error()
        );
    }
}
