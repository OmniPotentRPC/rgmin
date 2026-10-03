//! Safeguarded one-dimensional interpolation for the line searches.
//!
//! Both minimisers are derived and checked in
//! `validation/linesearch_interpolation.py` (sympy). With `w = b - a`:
//!
//! * [`cubic_min`] is the stationary point of the cubic Hermite
//!   interpolant through `(a, fa, da)` and `(b, fb, db)` at which the
//!   cubic's second derivative is positive, in the More-Thuente scaled
//!   form
//!   `theta = 3 (fa - fb) / w + da + db`,
//!   `gamma = sign(w) sqrt(theta^2 - da db)`,
//!   `t = a + w ((gamma - da) + theta) / ((gamma - da) + gamma + db)`.
//!   It equals Nocedal-Wright eq. 3.59. When the data come from a
//!   quadratic the cubic coefficient vanishes and the formula reduces
//!   exactly to the secant step `a - da w / (db - da)`, so the quadratic
//!   is not a special case. With `da < 0 < db` both the numerator and the
//!   denominator are sums of positive terms and `0 < r < 1`: the
//!   minimiser lies strictly inside the bracket with no cancellation
//!   (in floating point it can round onto an end it is within an ulp
//!   of, never past it).
//!   Scaling by `s = max(|theta|, |da|, |db|)` before squaring keeps
//!   `theta^2` and `da db` from overflowing.
//! * [`quad_min`] is the minimiser of the quadratic through
//!   `(a, fa, da)` and `(b, fb)`,
//!   `t = a - da w^2 / (2 (fb - fa - da w))`, defined when the curvature
//!   `fb - fa - da w` is positive.
//!
//! Neither function decides whether its answer is acceptable; the
//! callers clamp it into a safeguard interval.
//!
//! More and Thuente, *Line search algorithms with guaranteed sufficient
//! decrease*, <https://doi.org/10.1145/192115.192132>.
//! Nocedal and Wright, *Numerical Optimization*,
//! <https://doi.org/10.1007/978-0-387-40065-5>.

/// Local minimiser of the cubic Hermite interpolant, or `None` when the
/// data are not finite, `a == b`, or the cubic has no real local
/// minimiser (negative discriminant).
#[must_use]
pub(crate) fn cubic_min(a: f64, fa: f64, da: f64, b: f64, fb: f64, db: f64) -> Option<f64> {
    let w = b - a;
    if w == 0.0 || ![a, fa, da, b, fb, db, w].iter().all(|v| v.is_finite()) {
        return None;
    }
    let theta = 3.0 * (fa - fb) / w + da + db;
    let s = theta.abs().max(da.abs()).max(db.abs());
    if s.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) || !s.is_finite() {
        return None;
    }
    let disc = (theta / s) * (theta / s) - (da / s) * (db / s);
    if disc.is_nan() || disc < 0.0 {
        return None;
    }
    let mut gamma = s * disc.sqrt();
    if w < 0.0 {
        gamma = -gamma;
    }
    let p = (gamma - da) + theta;
    let q = ((gamma - da) + gamma) + db;
    if q == 0.0 {
        return None;
    }
    let t = a + (p / q) * w;
    t.is_finite().then_some(t)
}

/// Minimiser of the quadratic through `(a, fa, da)` and `(b, fb)`, or
/// `None` when the data are not finite, `a == b`, or the quadratic is
/// not convex.
#[must_use]
pub(crate) fn quad_min(a: f64, fa: f64, da: f64, b: f64, fb: f64) -> Option<f64> {
    let w = b - a;
    if w == 0.0 || ![a, fa, da, b, fb].iter().all(|v| v.is_finite()) {
        return None;
    }
    let curv = fb - fa - da * w;
    if curv.is_nan() || curv <= 0.0 {
        return None;
    }
    let t = a - da * w * w / (2.0 * curv);
    t.is_finite().then_some(t)
}

/// `t` clamped into the closed interval between `lo` and `hi`, in either
/// order.
#[inline]
pub(crate) fn clamp_between(t: f64, lo: f64, hi: f64) -> f64 {
    let (l, h) = if lo <= hi { (lo, hi) } else { (hi, lo) };
    t.clamp(l, h)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// phi(t) = (t - 0.3)^2 on [0, 1]: slopes -0.6 and 1.4.
    #[test]
    fn cubic_recovers_a_quadratic_minimiser_exactly() {
        let f = |t: f64| (t - 0.3) * (t - 0.3);
        let d = |t: f64| 2.0 * (t - 0.3);
        let t = cubic_min(0.0, f(0.0), d(0.0), 1.0, f(1.0), d(1.0)).unwrap();
        assert!((t - 0.3).abs() < 1e-15, "{t}");
        // Reversed bracket gives the same point.
        let r = cubic_min(1.0, f(1.0), d(1.0), 0.0, f(0.0), d(0.0)).unwrap();
        assert!((r - 0.3).abs() < 1e-15, "{r}");
        let q = quad_min(0.0, f(0.0), d(0.0), 1.0, f(1.0)).unwrap();
        assert!((q - 0.3).abs() < 1e-15, "{q}");
    }

    /// phi(t) = t^3 - 3 t has its local minimum at t = 1.
    #[test]
    fn cubic_recovers_a_cubic_minimiser_exactly() {
        let f = |t: f64| t * t * t - 3.0 * t;
        let d = |t: f64| 3.0 * t * t - 3.0;
        let t = cubic_min(0.0, f(0.0), d(0.0), 2.0, f(2.0), d(2.0)).unwrap();
        assert!((t - 1.0).abs() < 1e-14, "{t}");
        // Extrapolation: both points left of the minimum.
        let e = cubic_min(0.0, f(0.0), d(0.0), 0.5, f(0.5), d(0.5)).unwrap();
        assert!((e - 1.0).abs() < 1e-14, "{e}");
    }

    #[test]
    fn concave_data_have_no_minimiser() {
        // phi = -t^2: no local minimum anywhere.
        assert!(quad_min(0.0, 0.0, 0.0, 1.0, -1.0).is_none());
        // phi = -t^3 + ... a cubic with a negative discriminant:
        // phi(t) = t^3 + t has phi' = 3 t^2 + 1 > 0, no stationary point.
        let f = |t: f64| t * t * t + t;
        let d = |t: f64| 3.0 * t * t + 1.0;
        assert!(cubic_min(0.0, f(0.0), d(0.0), 1.0, f(1.0), d(1.0)).is_none());
        assert!(cubic_min(0.0, 0.0, f64::NAN, 1.0, 1.0, 1.0).is_none());
        assert!(cubic_min(1.0, 0.0, -1.0, 1.0, 1.0, 1.0).is_none());
    }

    #[test]
    fn sign_change_bracket_keeps_the_minimiser_inside() {
        // da < 0 < db: the minimiser lies strictly between a and b in
        // exact arithmetic; in floating point it can round onto an end
        // when the true point is within an ulp of it (fa - fb = 2e8 with
        // db = 1e-12 puts it 1e-20 from b), never outside.
        for &(fa, fb) in &[(0.0, 0.0), (0.0, 10.0), (10.0, 0.0), (1e8, -1e8)] {
            for &(da, db) in &[(-1.0, 1.0), (-1e-12, 1.0), (-1.0, 1e-12), (-3e5, 2e-4)] {
                let t = cubic_min(2.0, fa, da, 5.0, fb, db).unwrap();
                assert!(
                    (2.0..=5.0).contains(&t),
                    "fa {fa} fb {fb} da {da} db {db} t {t}"
                );
            }
        }
    }
}
