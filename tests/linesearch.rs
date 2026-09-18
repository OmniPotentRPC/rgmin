//! Wolfe line search on Rosenbrock; zoom stays inside its bracket.

use eindir_core::DifferentiableObjective;
use eindir_core::objectives::Rosenbrock;
use ndarray::{Array1, ArrayView1, array};
use rgmin::linesearch::zoom;
use rgmin::{Conjugacy, Control, LineSearch, Restart, minimize};

fn control() -> Control {
    Control {
        maxiter: 200,
        gtol: 1e-8,
        istep: 0.1,
        maxmove: None,
    }
}

fn wolfe() -> LineSearch {
    LineSearch::Wolfe {
        c1: 1e-4,
        c2: 0.9,
        maxiter: 40,
    }
}

/// `f(x) = (x[0] - 1)^2`. Line `x = α` along `d = 1` has minimizer `α = 1`.
fn quadratic(x: ArrayView1<'_, f64>) -> (f64, Array1<f64>) {
    let z = x[0] - 1.0;
    (z * z, array![2.0 * z])
}

fn bounded_linear(x: ArrayView1<'_, f64>) -> (f64, Array1<f64>) {
    if x[0] < -0.1 {
        (f64::INFINITY, array![f64::NAN])
    } else {
        (x[0], array![1.0])
    }
}

#[test]
fn zoom_exhaustion_retains_the_feasible_armijo_endpoint() {
    let alpha = zoom(
        &mut bounded_linear,
        array![0.0].view(),
        array![-1.0].view(),
        0.0,
        1.0,
        1e-4,
        0.9,
        6,
    );
    assert_eq!(alpha, 3.0 / 32.0);
    let (value, gradient) = bounded_linear(array![-alpha].view());
    assert!(value.is_finite() && gradient[0].is_finite());
    assert!(value <= -1e-4 * alpha);
}

#[test]
fn wolfe_domain_boundary_keeps_a_measured_descent() {
    let search = LineSearch::Wolfe {
        c1: 1e-4,
        c2: 0.9,
        maxiter: 6,
    };
    let (point, value, alpha) = search.search(
        bounded_linear,
        array![0.0].view(),
        array![-1.0].view(),
        1.0,
    );
    assert_eq!(point[0], -3.0 / 32.0);
    assert_eq!(value, point[0]);
    assert_eq!(alpha, 3.0 / 32.0);
}

#[test]
fn wolfe_on_rosenbrock_descends() {
    let obj = Rosenbrock::<2>::new();
    let start = array![-1.2, 1.0];
    let f0 = obj.value_and_gradient(start.view()).0;
    let report = minimize(
        &obj,
        start,
        &control(),
        Conjugacy::PolakRibiere,
        Restart::Never,
        wolfe(),
    )
    .unwrap();
    assert!(report.value < f0, "Wolfe {} -> {}", f0, report.value);
}

#[test]
fn zoom_stays_inside_the_bracket() {
    let pos = array![0.0];
    let dir = array![1.0];
    let lo = 0.0;
    let hi = 2.0;
    let alpha = zoom(
        &mut quadratic,
        pos.view(),
        dir.view(),
        lo,
        hi,
        1e-4,
        0.9,
        40,
    );
    assert!(
        alpha >= lo && alpha <= hi,
        "zoom left the bracket: {alpha} not in [{lo}, {hi}]"
    );
    let flipped = zoom(
        &mut quadratic,
        pos.view(),
        dir.view(),
        hi,
        lo,
        1e-4,
        0.9,
        40,
    );
    let a = lo.min(hi);
    let b = lo.max(hi);
    assert!(
        flipped >= a && flipped <= b,
        "zoom left the flipped bracket: {flipped} not in [{a}, {b}]"
    );
}

#[test]
fn wolfe_zoom_finds_strong_wolfe_on_quadratic() {
    // φ(α) = (α - 1)^2, φ'(α) = 2(α - 1). Strong curvature with c2 = 0.1
    // is |α - 1| <= 0.1. istep = 3 fails Armijo, so algorithm 3.5 zooms.
    let pos = array![0.0];
    let dir = array![1.0];
    let ls = LineSearch::Wolfe {
        c1: 1e-4,
        c2: 0.1,
        maxiter: 40,
    };
    let (x, f, a) = ls.search(quadratic, pos.view(), dir.view(), 3.0);
    assert!(a > 0.0, "Wolfe returned a zero step");
    assert!(
        x[0] >= 0.0 && x[0] <= 3.0,
        "Wolfe left [0, 3]: x={}, a={a}",
        x[0]
    );
    assert!(
        (x[0] - 1.0).abs() <= 0.1 + 1e-12,
        "not a strong-Wolfe step: x={}, a={a}",
        x[0]
    );
    assert!(f < 1.0, "Wolfe did not descend: {f}");
}

#[test]
fn goldstein_expands_a_too_short_step() {
    // On φ(α) = (α - 1)^2 the (1-c) lower bound needs α >= 2c. A 1e-8
    // start only shrinks under Armijo backtracking; Goldstein must expand.
    let pos = array![0.0];
    let dir = array![1.0];
    let c = 0.1;
    let ls = LineSearch::Goldstein {
        c,
        beta: 0.5,
        maxiter: 40,
    };
    let (_x, f, a) = ls.search(quadratic, pos.view(), dir.view(), 1e-8);
    assert!(
        a + 1e-12 >= 2.0 * c,
        "Goldstein did not expand: alpha={a}, need >= {}",
        2.0 * c
    );
    assert!(f < 1.0, "Goldstein did not descend: {f}");
}

#[test]
fn zoom_takes_a_single_fnmut_oracle() {
    let pos = array![0.0];
    let dir = array![1.0];
    let mut n = 0usize;
    let mut oracle = |x: ArrayView1<'_, f64>| {
        n += 1;
        quadratic(x)
    };
    let alpha = zoom(&mut oracle, pos.view(), dir.view(), 0.0, 2.0, 1e-4, 0.9, 40);
    assert!(
        alpha >= 0.0 && alpha <= 2.0,
        "zoom left the bracket: {alpha}"
    );
    assert!(n > 0, "oracle was never called");
}

#[test]
fn zoom_without_iterations_returns_lo() {
    let pos = array![0.0];
    let dir = array![1.0];
    let lo = 0.25;
    let alpha = zoom(
        &mut quadratic,
        pos.view(),
        dir.view(),
        lo,
        1.75,
        1e-4,
        0.9,
        0,
    );
    assert!(
        (alpha - lo).abs() < 1e-15,
        "empty zoom should keep lo={lo}, got {alpha}"
    );
}

#[test]
fn goldstein_backtracking_descends() {
    let obj = Rosenbrock::<2>::new();
    let start = array![-1.2, 1.0];
    let f0 = obj.value_and_gradient(start.view()).0;
    let report = minimize(
        &obj,
        start,
        &control(),
        Conjugacy::PolakRibiere,
        Restart::Never,
        LineSearch::Goldstein {
            c: 1e-4,
            beta: 0.5,
            maxiter: 40,
        },
    )
    .unwrap();
    assert!(report.value < f0, "Goldstein {} -> {}", f0, report.value);
}
