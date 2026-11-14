//! The C direction takes stored pairs and a caller-supplied H0.
//! A null handle is an error and does not panic.

#![cfg(feature = "capi")]

use ndarray::{ArrayView1, array};
use rgmin::ffi::{
    rgmin_last_error, rgmin_lbfgs_direction, rgmin_lbfgs_pairs, rgmin_lbfgs_pairs_clear,
    rgmin_lbfgs_pairs_count, rgmin_lbfgs_pairs_create, rgmin_lbfgs_pairs_free,
    rgmin_lbfgs_pairs_push, rgmin_lbfgs_pairs_push_rho, rgmin_status_t,
};
use rgmin::{LbfgsH0, LbfgsPairs};

unsafe extern "C" fn diag_h0(
    user: *mut std::ffi::c_void,
    v: *const f64,
    out: *mut f64,
    n: usize,
) -> i32 {
    let diag = unsafe { std::slice::from_raw_parts(user.cast::<f64>(), n) };
    let src = unsafe { std::slice::from_raw_parts(v, n) };
    let dst = unsafe { std::slice::from_raw_parts_mut(out, n) };
    for i in 0..n {
        dst[i] = diag[i] * src[i];
    }
    0
}

unsafe extern "C" fn fail_h0(
    _user: *mut std::ffi::c_void,
    _v: *const f64,
    _out: *mut f64,
    _n: usize,
) -> i32 {
    1
}

#[test]
fn null_handles_and_bad_lengths_are_errors() {
    let mut dir = [0.0; 2];
    let grad = [1.0, -1.0];
    let diag = [0.5, 0.25];
    let status = unsafe {
        rgmin_lbfgs_direction(
            std::ptr::null(),
            grad.as_ptr(),
            diag.as_ptr(),
            None,
            std::ptr::null_mut(),
            dir.as_mut_ptr(),
            2,
        )
    };
    assert_eq!(status, rgmin_status_t::RGMIN_INVALID_PARAMETER);
    unsafe {
        rgmin_lbfgs_pairs_free(std::ptr::null_mut());
        rgmin_lbfgs_pairs_clear(std::ptr::null_mut());
    }
    assert_eq!(unsafe { rgmin_lbfgs_pairs_count(std::ptr::null()) }, 0);
    assert!(unsafe { rgmin_lbfgs_pairs_create(0, 4) }.is_null());

    let pairs = unsafe { rgmin_lbfgs_pairs_create(2, 4) };
    assert!(!pairs.is_null());
    let both = unsafe {
        rgmin_lbfgs_direction(
            pairs,
            grad.as_ptr(),
            diag.as_ptr(),
            Some(diag_h0),
            std::ptr::null_mut(),
            dir.as_mut_ptr(),
            2,
        )
    };
    assert_eq!(both, rgmin_status_t::RGMIN_INVALID_PARAMETER);
    let neither = unsafe {
        rgmin_lbfgs_direction(
            pairs,
            grad.as_ptr(),
            std::ptr::null(),
            None,
            std::ptr::null_mut(),
            dir.as_mut_ptr(),
            2,
        )
    };
    assert_eq!(neither, rgmin_status_t::RGMIN_INVALID_PARAMETER);
    let short = [1.0];
    let mut stored = 1;
    let bad =
        unsafe { rgmin_lbfgs_pairs_push(pairs, short.as_ptr(), grad.as_ptr(), 1, &mut stored) };
    assert_eq!(bad, rgmin_status_t::RGMIN_INVALID_PARAMETER);
    assert_eq!(unsafe { rgmin_lbfgs_pairs_count(pairs) }, 0);
    unsafe { rgmin_lbfgs_pairs_free(pairs) };
}

#[test]
fn empty_memory_and_a_callback_match_the_diagonal() {
    let pairs = unsafe { rgmin_lbfgs_pairs_create(2, 0) };
    assert!(!pairs.is_null());
    let grad = [1.0, -2.0];
    let diag = [0.5, 0.25];
    let mut from_diag = [0.0; 2];
    let status = unsafe {
        rgmin_lbfgs_direction(
            pairs,
            grad.as_ptr(),
            diag.as_ptr(),
            None,
            std::ptr::null_mut(),
            from_diag.as_mut_ptr(),
            2,
        )
    };
    assert_eq!(status, rgmin_status_t::RGMIN_SUCCESS);
    assert_eq!(from_diag, [-0.5, 0.5]);

    let mut from_callback = [0.0; 2];
    let status = unsafe {
        rgmin_lbfgs_direction(
            pairs,
            grad.as_ptr(),
            std::ptr::null(),
            Some(diag_h0),
            diag.as_ptr() as *mut std::ffi::c_void,
            from_callback.as_mut_ptr(),
            2,
        )
    };
    assert_eq!(status, rgmin_status_t::RGMIN_SUCCESS);
    assert_eq!(from_callback, from_diag);

    let failed = unsafe {
        rgmin_lbfgs_direction(
            pairs,
            grad.as_ptr(),
            std::ptr::null(),
            Some(fail_h0),
            std::ptr::null_mut(),
            from_callback.as_mut_ptr(),
            2,
        )
    };
    assert_eq!(failed, rgmin_status_t::RGMIN_INVALID_PARAMETER);

    let s = [1.0, 0.0];
    let y = [2.0, 1.0];
    let mut stored = 0;
    let pushed = unsafe { rgmin_lbfgs_pairs_push(pairs, s.as_ptr(), y.as_ptr(), 2, &mut stored) };
    assert_eq!(pushed, rgmin_status_t::RGMIN_SUCCESS);
    assert_eq!(stored, 1);
    assert_eq!(unsafe { rgmin_lbfgs_pairs_count(pairs) }, 1);
    unsafe { rgmin_lbfgs_pairs_free(pairs) };
}

fn direction_c(
    pairs: *const rgmin_lbfgs_pairs,
    grad: &[f64],
    h0_diag: Option<&[f64]>,
    n: usize,
) -> (rgmin_status_t, Vec<f64>) {
    let mut dir = vec![f64::NAN; grad.len()];
    let status = unsafe {
        rgmin_lbfgs_direction(
            pairs,
            grad.as_ptr(),
            h0_diag.map_or(std::ptr::null(), <[f64]>::as_ptr),
            None,
            std::ptr::null_mut(),
            dir.as_mut_ptr(),
            n,
        )
    };
    (status, dir)
}

fn last_error() -> String {
    unsafe { std::ffi::CStr::from_ptr(rgmin_last_error()) }
        .to_string_lossy()
        .into_owned()
}

/// Pairs pushed with a caller `rho` through C give the Rust handle's
/// direction bit for bit, for a diagonal, a scalar written as a
/// constant diagonal, and a callback `H0`.
#[test]
fn push_rho_and_direction_round_trip_through_c() {
    let history = [
        (array![1.0, 0.5, -0.25], array![2.0, 1.0, 0.5], 0.4),
        (array![0.5, -0.25, 1.0], array![1.0, 0.5, 2.0], 0.3),
    ];
    let grad = [1.0, -2.0, 0.5];
    let diag = [0.5, 0.25, 2.0];
    let gamma = 0.375;

    let mut reference = LbfgsPairs::new(3, 4).unwrap();
    let handle = unsafe { rgmin_lbfgs_pairs_create(3, 4) };
    assert!(!handle.is_null());
    for (s, y, rho) in &history {
        assert!(reference.push_rho(s.view(), y.view(), *rho).unwrap());
        let mut stored = -1;
        let status = unsafe {
            rgmin_lbfgs_pairs_push_rho(handle, s.as_ptr(), y.as_ptr(), *rho, 3, &mut stored)
        };
        assert_eq!(status, rgmin_status_t::RGMIN_SUCCESS);
        assert_eq!(stored, 1);
    }
    assert_eq!(unsafe { rgmin_lbfgs_pairs_count(handle) }, 2);

    let g = ArrayView1::from(&grad);
    let want_diag = reference
        .direction(g, LbfgsH0::Diagonal(ArrayView1::from(&diag)))
        .unwrap();
    let (status, got) = direction_c(handle, &grad, Some(&diag), 3);
    assert_eq!(status, rgmin_status_t::RGMIN_SUCCESS);
    assert_eq!(got, want_diag.to_vec());

    let want_scale = reference.direction(g, LbfgsH0::Scale(gamma)).unwrap();
    let (status, got) = direction_c(handle, &grad, Some(&[gamma; 3]), 3);
    assert_eq!(status, rgmin_status_t::RGMIN_SUCCESS);
    assert_eq!(got, want_scale.to_vec());

    let mut via_callback = [f64::NAN; 3];
    let status = unsafe {
        rgmin_lbfgs_direction(
            handle,
            grad.as_ptr(),
            std::ptr::null(),
            Some(diag_h0),
            diag.as_ptr() as *mut std::ffi::c_void,
            via_callback.as_mut_ptr(),
            3,
        )
    };
    assert_eq!(status, rgmin_status_t::RGMIN_SUCCESS);
    assert_eq!(via_callback.to_vec(), want_diag.to_vec());

    // push and push_rho with 1 / (s . y) store the same pair.
    let by_gate = unsafe { rgmin_lbfgs_pairs_create(3, 4) };
    let by_rho = unsafe { rgmin_lbfgs_pairs_create(3, 4) };
    for (s, y, _) in &history {
        let mut stored = 0;
        let status =
            unsafe { rgmin_lbfgs_pairs_push(by_gate, s.as_ptr(), y.as_ptr(), 3, &mut stored) };
        assert_eq!((status, stored), (rgmin_status_t::RGMIN_SUCCESS, 1));
        let rho = 1.0 / s.dot(y);
        let status = unsafe {
            rgmin_lbfgs_pairs_push_rho(by_rho, s.as_ptr(), y.as_ptr(), rho, 3, &mut stored)
        };
        assert_eq!((status, stored), (rgmin_status_t::RGMIN_SUCCESS, 1));
    }
    assert_eq!(
        direction_c(by_gate, &grad, Some(&diag), 3),
        direction_c(by_rho, &grad, Some(&diag), 3)
    );
    unsafe {
        rgmin_lbfgs_pairs_free(handle);
        rgmin_lbfgs_pairs_free(by_gate);
        rgmin_lbfgs_pairs_free(by_rho);
    }
}

/// A non-positive curvature pair or a bad `rho` is skipped with success
/// and `stored` 0, as `rgmin_lbfgs_pairs_push` skips the pair. A length
/// other than the handle dimension and a null argument are refused.
#[test]
fn push_rho_skips_bad_pairs_and_refuses_bad_sizes() {
    let handle = unsafe { rgmin_lbfgs_pairs_create(2, 4) };
    assert!(!handle.is_null());
    let s = [1.0, 0.0];
    let y_down = [-1.0, 0.0];
    let y_flat = [0.0, 1.0];
    for y in [&y_down, &y_flat] {
        let mut by_gate = -1;
        let status =
            unsafe { rgmin_lbfgs_pairs_push(handle, s.as_ptr(), y.as_ptr(), 2, &mut by_gate) };
        assert_eq!((status, by_gate), (rgmin_status_t::RGMIN_SUCCESS, 0));
        let mut by_rho = -1;
        let status = unsafe {
            rgmin_lbfgs_pairs_push_rho(handle, s.as_ptr(), y.as_ptr(), 1.0, 2, &mut by_rho)
        };
        assert_eq!((status, by_rho), (rgmin_status_t::RGMIN_SUCCESS, 0));
    }
    let y = [2.0, 1.0];
    for rho in [0.0, -0.5, f64::NAN, f64::INFINITY] {
        let mut stored = -1;
        let status = unsafe {
            rgmin_lbfgs_pairs_push_rho(handle, s.as_ptr(), y.as_ptr(), rho, 2, &mut stored)
        };
        assert_eq!(
            (status, stored),
            (rgmin_status_t::RGMIN_SUCCESS, 0),
            "rho {rho}"
        );
    }
    assert_eq!(unsafe { rgmin_lbfgs_pairs_count(handle) }, 0);

    let mut stored = -1;
    let status =
        unsafe { rgmin_lbfgs_pairs_push_rho(handle, s.as_ptr(), y.as_ptr(), 0.5, 1, &mut stored) };
    assert_eq!(status, rgmin_status_t::RGMIN_INVALID_PARAMETER);
    assert!(last_error().contains("length"), "{}", last_error());
    assert_eq!(stored, -1, "a refused push leaves stored untouched");
    let status = unsafe {
        rgmin_lbfgs_pairs_push_rho(handle, std::ptr::null(), y.as_ptr(), 0.5, 2, &mut stored)
    };
    assert_eq!(status, rgmin_status_t::RGMIN_INVALID_PARAMETER);
    assert!(
        last_error().contains("rgmin_lbfgs_pairs_push_rho"),
        "{}",
        last_error()
    );
    let status = unsafe {
        rgmin_lbfgs_pairs_push_rho(
            std::ptr::null_mut(),
            s.as_ptr(),
            y.as_ptr(),
            0.5,
            2,
            &mut stored,
        )
    };
    assert_eq!(status, rgmin_status_t::RGMIN_INVALID_PARAMETER);
    assert_eq!(unsafe { rgmin_lbfgs_pairs_count(handle) }, 0);

    let status = unsafe {
        rgmin_lbfgs_pairs_push_rho(handle, s.as_ptr(), y.as_ptr(), 0.5, 2, std::ptr::null_mut())
    };
    assert_eq!(status, rgmin_status_t::RGMIN_SUCCESS);
    assert_eq!(unsafe { rgmin_lbfgs_pairs_count(handle) }, 1);

    let grad = [1.0, -1.0, 0.5];
    let (status, _) = direction_c(handle, &grad, Some(&[0.5; 3]), 3);
    assert_eq!(status, rgmin_status_t::RGMIN_INVALID_PARAMETER);
    let (status, dir) = direction_c(handle, &grad[..2], Some(&[0.5; 2]), 2);
    assert_eq!(status, rgmin_status_t::RGMIN_SUCCESS);
    assert!(dir.iter().all(|v| v.is_finite()), "{dir:?}");
    unsafe { rgmin_lbfgs_pairs_free(handle) };
}
