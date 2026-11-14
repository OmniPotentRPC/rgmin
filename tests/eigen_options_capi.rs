#![cfg(feature = "capi")]

use dlpk::sys::DLManagedTensorVersioned;
use rgmin::ffi::*;
use std::ffi::c_void;
use std::mem::{offset_of, size_of};

unsafe extern "C" fn diagonal(
    user: *mut c_void,
    _x: *const DLManagedTensorVersioned,
    v: *const DLManagedTensorVersioned,
    hv: *mut DLManagedTensorVersioned,
) -> rgmin_status_t {
    unsafe {
        *(user as *mut usize) += 1;
        let a = (*v).dl_tensor.data.cast::<f64>();
        let b = (*hv).dl_tensor.data.cast::<f64>();
        for (i, value) in [-4.0, 2.0, 7.0].into_iter().enumerate() {
            *b.add(i) = value * *a.add(i);
        }
    }
    rgmin_status_t::RGMIN_SUCCESS
}

#[test]
fn base_record_layout_and_extended_selector_are_stable() {
    assert_eq!(size_of::<rgmin_eigen_params_t>(), 24);
    assert_eq!(offset_of!(rgmin_eigen_params_t, tol), 16);
    assert_eq!(size_of::<rgmin_eigen_options_t>(), 8);
    assert_eq!(offset_of!(rgmin_eigen_options_t, extra), 4);
    assert_eq!(rgmin_eigen_kind_t::RGMIN_EIGEN_LIBKRYLOV as i32, 15);
}

#[test]
fn options_entry_preserves_matrix_free_result_actions_and_aliasing() {
    let mut reference = None;
    for options in [
        None,
        Some(rgmin_eigen_options_t {
            degree: 31,
            extra: 5,
        }),
    ] {
        let mut calls = 0usize;
        let mut x = [0.0; 3];
        let mut seed = [1.0, 1.0, 1.0];
        let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 3) };
        let st = unsafe { rgmin_tensor_borrow_cpu_f64(seed.as_mut_ptr(), 3) };
        let params = rgmin_eigen_params_t {
            kind: 0,
            nev: 1,
            krylov: 3,
            max_iter: 0,
            tol: 1e-12,
        };
        let mut result = rgmin_lowest_mode_t {
            value: 42.0,
            actions: 43,
        };
        let status = unsafe {
            rgmin_lowest_eigenpair_with_options(
                Some(diagonal),
                (&mut calls as *mut usize).cast(),
                xt,
                st,
                st,
                &params,
                options.as_ref().map_or(std::ptr::null(), |o| o),
                &mut result,
            )
        };
        unsafe {
            rgmin_tensor_free(xt);
            rgmin_tensor_free(st);
        }
        assert_eq!(status, rgmin_status_t::RGMIN_SUCCESS);
        assert!((result.value + 4.0).abs() < 1e-12);
        assert!((seed[0].abs() - 1.0).abs() < 1e-12);
        assert!(seed[1].abs() < 1e-12 && seed[2].abs() < 1e-12);
        assert_eq!(result.actions, calls);
        assert_eq!(calls, 3);
        let measured = (seed, result.value, result.actions);
        if let Some(expected) = reference {
            assert_eq!(measured, expected);
        }
        reference = Some(measured);
    }
}

#[test]
fn options_entry_rejects_unknown_kind_without_callbacks_or_output_changes() {
    let mut calls = 0usize;
    let mut x = [0.0; 3];
    let mut seed = [1.0; 3];
    let mut mode = [42.0; 3];
    let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), 3) };
    let st = unsafe { rgmin_tensor_borrow_cpu_f64(seed.as_mut_ptr(), 3) };
    let mt = unsafe { rgmin_tensor_borrow_cpu_f64(mode.as_mut_ptr(), 3) };
    let params = rgmin_eigen_params_t {
        kind: 271,
        nev: 1,
        krylov: 3,
        max_iter: 0,
        tol: 0.0,
    };
    let options = rgmin_eigen_options_t {
        degree: 20,
        extra: 8,
    };
    let mut result = rgmin_lowest_mode_t {
        value: 17.0,
        actions: 19,
    };
    let status = unsafe {
        rgmin_lowest_eigenpair_with_options(
            Some(diagonal),
            (&mut calls as *mut usize).cast(),
            xt,
            st,
            mt,
            &params,
            &options,
            &mut result,
        )
    };
    unsafe {
        rgmin_tensor_free(xt);
        rgmin_tensor_free(st);
        rgmin_tensor_free(mt);
    }
    assert_eq!(status, rgmin_status_t::RGMIN_INVALID_PARAMETER);
    assert_eq!(calls, 0);
    assert_eq!(mode, [42.0; 3]);
    assert_eq!(result.value, 17.0);
    assert_eq!(result.actions, 19);
}
