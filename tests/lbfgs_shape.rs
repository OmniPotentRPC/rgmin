//! A bad callback or a bad box is a refused start, not a panic.

use ndarray::{Array1, array};
use rgmin::Lbfgs;

fn quad(x: ndarray::ArrayView1<f64>) -> Option<(f64, Array1<f64>)> {
    Some((x[0] * x[0], array![2.0 * x[0]]))
}

#[test]
fn a_short_gradient_ends_the_relaxation() {
    let mut opt = Lbfgs::default();
    let (f, x, evals) = opt.minimize(array![1.0, -2.0].view(), 10, |_| Some((1.0, array![1.0])));
    assert!(!f.is_finite());
    assert_eq!(evals, 1);
    assert_eq!(x, array![1.0, -2.0]);
}

#[test]
fn a_short_gradient_on_a_later_probe_does_not_panic() {
    let mut calls = 0usize;
    let mut opt = Lbfgs::default();
    let (f, _, evals) = opt.minimize(array![1.0, -2.0].view(), 5, |x| {
        calls += 1;
        if calls == 1 {
            Some((x[0] * x[0] + x[1] * x[1], array![2.0 * x[0], 2.0 * x[1]]))
        } else {
            Some((0.0, array![0.0]))
        }
    });
    assert!(evals >= 1);
    assert!(f.is_finite());
}

#[test]
fn a_bound_of_the_wrong_length_is_refused() {
    let mut opt = Lbfgs::default();
    let (f, x, evals) = opt.minimize_box(
        array![0.0, 0.0].view(),
        array![0.0].view(),
        array![1.0, 1.0].view(),
        10,
        quad,
    );
    assert!(!f.is_finite());
    assert_eq!(evals, 0);
    assert_eq!(x, array![0.0, 0.0]);
}

#[test]
fn an_inverted_bound_is_refused() {
    let mut opt = Lbfgs::default();
    let (f, _, evals) = opt.minimize_box(
        array![0.0].view(),
        array![1.0].view(),
        array![0.0].view(),
        10,
        quad,
    );
    assert!(!f.is_finite());
    assert_eq!(evals, 0);
}

#[test]
fn a_nan_bound_is_refused() {
    let mut opt = Lbfgs::default();
    let (f, _, evals) = opt.minimize_box(
        array![0.0].view(),
        array![f64::NAN].view(),
        array![1.0].view(),
        10,
        quad,
    );
    assert!(!f.is_finite());
    assert_eq!(evals, 0);
}
