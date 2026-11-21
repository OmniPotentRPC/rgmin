//! Public Brent line searches resolve steps below the old `1e-12` floor.

use ndarray::{Array1, ArrayView1, array};
use rgmin::LineSearch;

fn tiny_quad(x: ArrayView1<'_, f64>) -> (f64, Array1<f64>) {
    let z = x[0] - 5.0e-14;
    (z * z, array![2.0 * z])
}

fn kink(x: ArrayView1<'_, f64>) -> (f64, Array1<f64>) {
    let z = x[0] - 1.0e-5;
    (z.abs(), array![z.signum()])
}

fn brent_from(istep: f64, fg: impl Fn(ArrayView1<'_, f64>) -> (f64, Array1<f64>)) -> f64 {
    let x0 = array![0.0];
    let dir = array![1.0];
    let (f0, g0) = fg(x0.view());
    let out = LineSearch::brent(128, 0.0)
        .search_from(&fg, x0.view(), f0, g0.view(), dir.view(), istep, 0.0)
        .expect("Brent finds a decrease");
    assert_eq!(out.x[0], out.alpha);
    out.alpha
}

#[test]
fn search_from_resolves_a_minimum_below_the_old_floor() {
    let alpha = brent_from(1.0, tiny_quad);
    let err = (alpha - 5.0e-14).abs();
    assert!(err < 1.0e-15, "alpha={alpha:e} err={err:e}");
}

#[test]
fn search_from_with_zero_tol_resolves_a_kink() {
    let alpha = brent_from(1.0e-6, kink);
    let err = (alpha - 1.0e-5).abs();
    assert!(err < 1.0e-15, "alpha={alpha:e} err={err:e}");
}

#[test]
fn search_resolves_a_minimum_below_the_old_floor() {
    let (x, _f, alpha) =
        LineSearch::brent(128, 0.0).search(tiny_quad, array![0.0].view(), array![1.0].view(), 1.0);
    let err = (alpha - 5.0e-14).abs();
    assert_eq!(x[0], alpha);
    assert!(err < 1.0e-15, "alpha={alpha:e} err={err:e}");
}
