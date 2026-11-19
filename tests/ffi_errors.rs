#![cfg(feature = "capi")]

use std::os::raw::c_void;

use dlpk::sys::DLManagedTensorVersioned;
use rgmin::ffi::{
    rgmin_abi_compatible, rgmin_abi_stamp, rgmin_control_t, rgmin_method_from_name, rgmin_method_t,
    rgmin_minimize, rgmin_report_t, rgmin_status_t, rgmin_stop_t, rgmin_tensor_borrow_cpu_f64,
    rgmin_tensor_free,
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
        let mut value = [initial];
        let tensor = unsafe { rgmin_tensor_borrow_cpu_f64(value.as_mut_ptr(), value.len()) };
        let mut out = report();
        let mut ctrl = control();
        ctrl.maxiter = 1;
        let status = unsafe {
            rgmin_minimize(
                Some(square_eval),
                Some(square_grad),
                std::ptr::null_mut(),
                tensor,
                &ctrl,
                rgmin_method_t::RGMIN_STEEPEST,
                &mut out,
            )
        };
        unsafe { rgmin_tensor_free(tensor) };
        assert_eq!(status, rgmin_status_t::RGMIN_INVALID_PARAMETER);
        assert_eq!(value[0].to_bits(), initial.to_bits());
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

    let mut value = [1.0];
    let tensor = unsafe { rgmin_tensor_borrow_cpu_f64(value.as_mut_ptr(), value.len()) };
    unsafe { (*tensor).version.major += 1 };
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
}
