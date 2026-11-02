//! `Lbfgs::minimize_box`: generalized Cauchy point and subspace step.

use ndarray::{Array1, ArrayView1};
use rgmin::Lbfgs;

/// `driver1` of the L-BFGS-B 3.0 distribution (Zhu, Byrd, Lu, Nocedal
/// 1997): an extended Rosenbrock function on 25 variables, odd entries
/// (1-based) in `[1, 100]`, even entries in `[-100, 100]`, from `x = 3`.
fn driver1(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let n = x.len();
    let mut f = 0.25 * (x[0] - 1.0).powi(2);
    let mut g = Array1::zeros(n);
    g[0] = 0.5 * (x[0] - 1.0);
    for i in 1..n {
        let t = x[i] - x[i - 1] * x[i - 1];
        f += 4.0 * t * t;
        g[i] += 8.0 * t;
        g[i - 1] -= 16.0 * x[i - 1] * t;
    }
    (f, g)
}

fn driver1_box() -> (Array1<f64>, Array1<f64>) {
    let lo = Array1::from_shape_fn(25, |i| if i % 2 == 0 { 1.0 } else { -100.0 });
    (lo, Array1::from_elem(25, 100.0))
}

fn projected_inf(x: &Array1<f64>, g: &Array1<f64>, lo: &Array1<f64>, hi: &Array1<f64>) -> f64 {
    (0..x.len()).fold(0.0_f64, |m, i| {
        m.max(g[i].max(x[i] - hi[i]).min(x[i] - lo[i]).abs())
    })
}

#[test]
fn driver1_reaches_the_bound_constrained_minimum() {
    let (lo, hi) = driver1_box();
    let mut opt = Lbfgs::with_capacity(5);
    opt.gtol = 1e-8;
    let mut outside = 0;
    let (f, x, evals) = opt.minimize_box(
        Array1::from_elem(25, 3.0).view(),
        lo.view(),
        hi.view(),
        500,
        |v| {
            outside += v
                .iter()
                .zip(&lo)
                .zip(&hi)
                .filter(|((x, l), h)| *x < *l || *x > *h)
                .count();
            Some(driver1(v))
        },
    );
    let (_, g) = driver1(x.view());
    assert_eq!(outside, 0, "fg saw a point outside the box");
    assert!(f < 1e-12, "f = {f} after {evals} evaluations");
    assert!(projected_inf(&x, &g, &lo, &hi) < 1e-8);
    assert!(evals < 200, "{evals} evaluations");
}

#[test]
fn a_quadratic_whose_minimum_is_outside_the_box_stops_on_the_face() {
    // Separable, so the box minimum is the projection of the free minimum.
    let scales = [1.0, 10.0, 100.0, 1000.0];
    let quad = |x: ArrayView1<f64>| {
        let g = Array1::from_shape_fn(x.len(), |i| scales[i % 4] * (x[i] - 1.0));
        (
            0.5 * g
                .iter()
                .zip(&x)
                .map(|(gi, xi)| gi * (xi - 1.0))
                .sum::<f64>(),
            g,
        )
    };
    let lo = Array1::from_shape_fn(8, |i| if i < 4 { 1.5 } else { -5.0 });
    let hi = Array1::from_elem(8, 5.0);
    let mut opt = Lbfgs::default();
    let (_, x, _) = opt.minimize_box(
        Array1::from_elem(8, 4.0).view(),
        lo.view(),
        hi.view(),
        200,
        |v| Some(quad(v)),
    );
    for (i, xi) in x.iter().enumerate() {
        let want = if i < 4 { 1.5 } else { 1.0 };
        assert!((xi - want).abs() < 1e-6, "x[{i}] = {xi}");
    }
}

#[test]
fn infinite_bounds_minimize_rosenbrock() {
    let rosen = |x: ArrayView1<f64>| {
        let (a, b) = (x[0], x[1]);
        let f = (1.0 - a).powi(2) + 100.0 * (b - a * a).powi(2);
        let g = Array1::from(vec![
            -2.0 * (1.0 - a) - 400.0 * a * (b - a * a),
            200.0 * (b - a * a),
        ]);
        (f, g)
    };
    let inf = Array1::from_elem(2, f64::INFINITY);
    let mut opt = Lbfgs::default();
    let (f, x, _) = opt.minimize_box(
        Array1::from(vec![-1.2, 1.0]).view(),
        (-&inf).view(),
        inf.view(),
        500,
        |v| Some(rosen(v)),
    );
    assert!(f < 1e-12, "f = {f}");
    assert!((x[0] - 1.0).abs() < 1e-6 && (x[1] - 1.0).abs() < 1e-6);
}
