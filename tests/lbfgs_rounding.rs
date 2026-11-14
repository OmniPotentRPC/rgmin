//! The approximate Wolfe test of `Lbfgs` on values that carry rounding noise.
//!
//! The searches opt into [`LbfgsStep::Wolfe`]. The default step is the
//! eOn cap, which does not consult `approx_wolfe`.

use ndarray::{Array1, ArrayView1, array};
use rgmin::{Lbfgs, LbfgsStep};

const SCALE: [f64; 4] = [1.0, 10.0, 100.0, 1000.0];

fn wolfe(approx_wolfe: f64) -> Lbfgs {
    let mut opt = Lbfgs::default();
    opt.step = LbfgsStep::Wolfe;
    opt.gtol = 1e-9;
    opt.approx_wolfe = approx_wolfe;
    opt
}

/// Returns a bowl above an offset of 1, plus a value error of up to 1e-14.
/// The error depends on the bits of `x`, like the rounding of a long sum.
/// The gradient is exact.
fn noisy(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let g = Array1::from_shape_fn(x.len(), |i| SCALE[i % 4] * x[i]);
    let h = x.iter().fold(0x9e37_79b9_7f4a_7c15_u64, |h, v| {
        (h ^ v.to_bits()).wrapping_mul(0x0100_0000_01b3)
    });
    let noise = 1e-14 * ((h >> 11) as f64 / (1u64 << 53) as f64 - 0.5);
    (1.0 + 0.5 * g.dot(&x) + noise, g)
}

fn rosen(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let n = x.len();
    let mut f = 0.0;
    let mut g = Array1::zeros(n);
    for i in 0..n - 1 {
        let t = x[i + 1] - x[i] * x[i];
        f += 100.0 * t * t + (1.0 - x[i]).powi(2);
        g[i] += -400.0 * x[i] * t - 2.0 * (1.0 - x[i]);
        g[i + 1] += 200.0 * t;
    }
    (f, g)
}

fn ginf(fg: fn(ArrayView1<f64>) -> (f64, Array1<f64>), x: &Array1<f64>) -> f64 {
    fg(x.view()).1.iter().fold(0.0, |m: f64, v| m.max(v.abs()))
}

#[test]
fn rounding_noise_stops_the_strict_search_but_not_the_approximate_one() {
    let x0 = Array1::from_elem(8, 1.0);
    let run = |approx_wolfe: f64| {
        let mut opt = wolfe(approx_wolfe);
        let (_, x, _) = opt.minimize(x0.view(), 500, |v| Some(noisy(v)));
        ginf(noisy, &x)
    };
    assert_eq!(Lbfgs::default().approx_wolfe, 0.0);
    assert!(run(0.0) > 1e-9, "strict: {}", run(0.0));
    assert!(run(1e-12) < 1e-9, "approximate: {}", run(1e-12));
}

#[test]
fn the_box_mode_also_needs_the_approximate_test_under_noise() {
    let x0 = Array1::from_elem(8, 1.0);
    let lower = Array1::from_elem(8, -2.0);
    let upper = Array1::from_elem(8, 2.0);
    let run = |approx_wolfe: f64| {
        let mut opt = wolfe(approx_wolfe);
        let (_, x, _) = opt.minimize_box(x0.view(), lower.view(), upper.view(), 500, |v| {
            Some(noisy(v))
        });
        ginf(noisy, &x)
    };
    assert!(run(0.0) > 1e-9, "strict: {}", run(0.0));
    assert!(run(1e-12) < 1e-9, "approximate: {}", run(1e-12));
}

#[test]
fn a_clean_descent_never_needs_the_approximate_test() {
    let x0 = array![-1.2, 1.0, -1.2, 1.0, -1.2, 1.0, -1.2, 1.0, -1.2, 1.0];
    let mut strict = wolfe(0.0);
    strict.gtol = 1e-6;
    let mut loose = wolfe(1e-12);
    loose.gtol = 1e-6;
    let (fs, xs, es) = strict.minimize(x0.view(), 2000, |v| Some(rosen(v)));
    let (f, x, e) = loose.minimize(x0.view(), 2000, |v| Some(rosen(v)));
    assert_eq!(f.to_bits(), fs.to_bits());
    assert_eq!(x, xs);
    assert_eq!(e, es);
}
