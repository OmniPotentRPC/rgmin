//! The C line search takes the caller's energy and gradient.

#![cfg(feature = "capi")]

use rgmin::ffi::{rgmin_linesearch, rgmin_linesearch_create, rgmin_linesearch_free};

fn quad(x: f64) -> (f64, f64) {
    let z = x - 1.0;
    (z * z, 2.0 * z)
}

fn drive(kind: i32, istep: f64) -> (f64, f64, f64) {
    drive_with(kind, istep, 20, 1e-8, quad)
}

fn drive_with(
    kind: i32,
    istep: f64,
    maxiter: usize,
    tol: f64,
    fg: impl Fn(f64) -> (f64, f64),
) -> (f64, f64, f64) {
    let x = [0.0];
    let dir = [1.0];
    let search = unsafe {
        rgmin_linesearch_create(
            kind,
            x.as_ptr(),
            dir.as_ptr(),
            1,
            istep,
            1e-4,
            0.9,
            0.5,
            maxiter,
            tol,
        )
    };
    assert!(!search.is_null());
    let mut trial = [0.0];
    let mut alpha = 0.0;
    let mut value = 0.0;
    let mut need = 1;
    let (mut f, g0) = fg(0.0);
    let mut g = [g0];
    for _ in 0..10 * maxiter {
        let st = unsafe {
            rgmin_linesearch(
                search,
                f,
                g.as_ptr(),
                trial.as_mut_ptr(),
                &mut alpha,
                &mut value,
                &mut need,
            )
        };
        assert_eq!(st, 0);
        if need == 0 {
            break;
        }
        let (next_f, next_g) = fg(trial[0]);
        f = next_f;
        g = [next_g];
    }
    assert_eq!(need, 0);
    unsafe { rgmin_linesearch_free(search) };
    (trial[0], alpha, value)
}

#[test]
fn wolfe_accepts_the_first_unit_fraction() {
    let (x, alpha, value) = drive(3, 0.25);
    assert!((x - 0.25).abs() < 1e-12, "{x}");
    assert!((alpha - 0.25).abs() < 1e-12, "{alpha}");
    assert!((value - quad(0.25).0).abs() < 1e-12);
}

#[test]
fn backtracking_reaches_the_quadratic_minimum() {
    let (x, _alpha, value) = drive(1, 8.0);
    assert!((x - 1.0).abs() < 1e-8, "{x}");
    assert!(value < 1e-14, "{value}");
}

#[test]
fn brent_with_zero_tol_resolves_a_kink() {
    let kink = |x: f64| {
        let z = x - 1.0e-5;
        (z.abs(), z.signum())
    };
    let (x, alpha, _value) = drive_with(0, 1.0e-6, 128, 0.0, kink);
    assert_eq!(x, alpha);
    assert!((alpha - 1.0e-5).abs() < 1e-15, "alpha={alpha:e}");
}

#[test]
fn a_null_search_is_an_error() {
    let mut x = [0.0];
    let mut alpha = 0.0;
    let mut value = 0.0;
    let mut need = 0;
    let g = [0.0];
    let st = unsafe {
        rgmin_linesearch(
            std::ptr::null_mut(),
            0.0,
            g.as_ptr(),
            x.as_mut_ptr(),
            &mut alpha,
            &mut value,
            &mut need,
        )
    };
    assert_eq!(st, 1);
    unsafe { rgmin_linesearch_free(std::ptr::null_mut()) };
}
