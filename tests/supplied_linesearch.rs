//! The caller-supplied line search takes the same steps as `LineSearch::search`.

use ndarray::{Array1, ArrayView1, array};
use rgmin::{LineSearch, SearchStep, SuppliedSearch};

fn quad(x: ArrayView1<'_, f64>) -> (f64, Array1<f64>) {
    let z = x[0] - 1.0;
    (z * z, array![2.0 * z])
}

fn rosen(x: ArrayView1<'_, f64>) -> (f64, Array1<f64>) {
    let a = 1.0 - x[0];
    let b = x[1] - x[0] * x[0];
    (
        a * a + 100.0 * b * b,
        array![-2.0 * a - 400.0 * x[0] * b, 200.0 * b],
    )
}

fn alpha_of(x0: ArrayView1<'_, f64>, dir: ArrayView1<'_, f64>, x: ArrayView1<'_, f64>) -> f64 {
    for i in 0..dir.len() {
        if dir[i].abs() > 1e-12 {
            return (x[i] - x0[i]) / dir[i];
        }
    }
    0.0
}

fn supplied(
    kind: LineSearch,
    x0: &Array1<f64>,
    dir: &Array1<f64>,
    istep: f64,
    fg: impl Fn(ArrayView1<'_, f64>) -> (f64, Array1<f64>),
) -> (Array1<f64>, f64, f64, Vec<f64>) {
    let mut search = SuppliedSearch::new(kind, x0.view(), dir.view(), istep);
    let (f, g) = fg(x0.view());
    let mut step = search.offer(f, g.view());
    let mut alphas = Vec::new();
    for _ in 0..10_000 {
        match step {
            SearchStep::Evaluate { alpha, x } => {
                alphas.push(alpha);
                let (f, g) = fg(x.view());
                step = search.offer(f, g.view());
            }
            SearchStep::Accepted { x, value, alpha } => return (x, value, alpha, alphas),
        }
    }
    panic!("search did not finish");
}

fn library(
    kind: LineSearch,
    x0: &Array1<f64>,
    dir: &Array1<f64>,
    istep: f64,
    fg: impl Fn(ArrayView1<'_, f64>) -> (f64, Array1<f64>),
) -> (Array1<f64>, f64, f64, Vec<f64>) {
    let mut first = true;
    let mut alphas = Vec::new();
    let (x, value, alpha) = kind.search(
        |p| {
            if first {
                first = false;
            } else {
                alphas.push(alpha_of(x0.view(), dir.view(), p));
            }
            fg(p)
        },
        x0.view(),
        dir.view(),
        istep,
    );
    (x, value, alpha, alphas)
}

fn same(
    kind: LineSearch,
    x0: Array1<f64>,
    dir: Array1<f64>,
    istep: f64,
    fg: impl Fn(ArrayView1<'_, f64>) -> (f64, Array1<f64>),
) {
    let lib = library(kind, &x0, &dir, istep, &fg);
    let got = supplied(kind, &x0, &dir, istep, &fg);
    assert_eq!(got.3.len(), lib.3.len(), "trial count {got:?} vs {lib:?}");
    for (a, b) in got.3.iter().zip(lib.3.iter()) {
        assert!((a - b).abs() <= 1e-12 * (1.0 + a.abs()), "trial {a} != {b}");
    }
    assert!(
        (got.2 - lib.2).abs() <= 1e-12,
        "alpha {} != {}",
        got.2,
        lib.2
    );
    assert!(
        (got.1 - lib.1).abs() <= 1e-9 * (1.0 + lib.1.abs()),
        "value {} != {}",
        got.1,
        lib.1
    );
    for (a, b) in got.0.iter().zip(lib.0.iter()) {
        assert!((a - b).abs() <= 1e-9, "{a} != {b}");
    }
}

#[test]
fn wolfe_matches_on_the_quadratic() {
    same(
        LineSearch::wolfe(1e-4, 0.9, 20),
        array![0.0],
        array![1.0],
        0.25,
        quad,
    );
}

#[test]
fn backtracking_matches_when_the_first_trial_is_too_long() {
    same(
        LineSearch::backtracking(1e-4, 0.5, 20),
        array![0.0],
        array![1.0],
        8.0,
        quad,
    );
}

#[test]
fn goldstein_matches_on_the_quadratic() {
    same(
        LineSearch::goldstein(0.1, 0.5, 20),
        array![0.0],
        array![1.0],
        4.0,
        quad,
    );
}

#[test]
fn brent_matches_on_the_quadratic() {
    same(
        LineSearch::brent(30, 1e-8),
        array![0.0],
        array![1.0],
        0.2,
        quad,
    );
}

#[test]
fn brent_resolves_a_minimum_below_the_old_floor() {
    let tiny = |x: ArrayView1<'_, f64>| {
        let z = x[0] - 5.0e-14;
        (z * z, array![2.0 * z])
    };
    let kind = LineSearch::brent(128, 0.0);
    let (x0, dir) = (array![0.0], array![1.0]);
    same(kind, x0.clone(), dir.clone(), 1.0, tiny);
    let (x, _value, alpha, _) = supplied(kind, &x0, &dir, 1.0, tiny);
    assert_eq!(x[0], alpha);
    assert!((alpha - 5.0e-14).abs() < 1.0e-15, "alpha={alpha:e}");
}

#[test]
fn wolfe_matches_rosenbrock_along_the_gradient() {
    let x0 = array![-1.2, 1.0];
    let (_, g) = rosen(x0.view());
    let dir = -g;
    same(LineSearch::wolfe(1e-4, 0.9, 16), x0, dir, 1.0, rosen);
}

#[test]
fn a_short_gradient_does_not_panic() {
    let mut search = SuppliedSearch::new(
        LineSearch::default(),
        array![1.0, -2.0].view(),
        array![1.0, 0.0].view(),
        1.0,
    );
    let SearchStep::Accepted { alpha, x, .. } = search.offer(1.0, array![1.0].view()) else {
        panic!("a short gradient is a finished search");
    };
    assert_eq!(alpha, 0.0);
    assert_eq!(x, array![1.0, -2.0]);
}

#[test]
fn an_uphill_direction_stays_put() {
    let kind = LineSearch::wolfe(1e-4, 0.9, 8);
    same(kind, array![0.0], array![-1.0], 1.0, quad);
}
