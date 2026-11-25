//! Nocedal-Wright zoom (algorithm 3.6) as a standalone bracket refiner.
//!
//! Trial interpolation here is bisection (`BisectionStepSize`): the next
//! α is the midpoint of the current bracket, so every trial stays inside
//! it. The line searches themselves zoom with safeguarded cubic
//! interpolation from a known start (`known.rs`).
//!
//! Wolfe, *Convergence Conditions for Ascent Methods*,
//! <https://doi.org/10.1137/1011036>.
//! Nocedal and Wright, *Numerical Optimization*,
//! <https://doi.org/10.1007/978-0-387-40065-5>.

use ndarray::{Array1, ArrayView1};

use super::axpy;
use super::conditions::{armijo, roundoff_strong_wolfe, strong_curvature, strong_wolfe};
use super::{Path, scored, unusable};

/// Evaluate `φ(α) = f(x + α d)` and `φ'(α) = ∇f(x + α d) · d`.
///
/// One oracle call yields both values so `φ` and `φ'` do not each hold a
/// mutable borrow of the same closure.
fn phi_pair<F>(
    oracle: &mut F,
    pos: ArrayView1<'_, f64>,
    dir: ArrayView1<'_, f64>,
    alpha: f64,
) -> (f64, f64)
where
    F: FnMut(ArrayView1<'_, f64>) -> (f64, Array1<f64>),
{
    let x = axpy(pos, alpha, dir);
    let (f, g) = oracle(x.view());
    let dphi = g.iter().zip(dir.iter()).map(|(gi, di)| gi * di).sum();
    (f, dphi)
}

/// Midpoint of `[lo, hi]`. Independent of endpoint order.
#[inline]
fn bisect(lo: f64, hi: f64) -> f64 {
    0.5 * (lo + hi)
}

/// Nocedal-Wright algorithm 3.6.
///
/// Interpolates inside the closed interval between `lo` and `hi` until the
/// strong Wolfe conditions hold or `maxiter` is exhausted. The returned α
/// always lies between `lo` and `hi` (inclusive).
///
/// Nocedal and Wright, *Numerical Optimization*,
/// <https://doi.org/10.1007/978-0-387-40065-5>.
pub fn zoom<F>(
    oracle: &mut F,
    pos: ArrayView1<'_, f64>,
    dir: ArrayView1<'_, f64>,
    lo: f64,
    hi: f64,
    c1: f64,
    c2: f64,
    maxiter: usize,
) -> f64
where
    F: FnMut(ArrayView1<'_, f64>) -> (f64, Array1<f64>),
{
    let (phi0, dphi0) = phi_pair(oracle, pos, dir, 0.0);
    zoom_into(oracle, pos, dir, lo, hi, c1, c2, maxiter, phi0, dphi0).0
}

/// Zoom with a precomputed `(φ(0), φ'(0))`.
fn zoom_into<F>(
    oracle: &mut F,
    pos: ArrayView1<'_, f64>,
    dir: ArrayView1<'_, f64>,
    mut lo: f64,
    mut hi: f64,
    c1: f64,
    c2: f64,
    maxiter: usize,
    phi0: f64,
    dphi0: f64,
) -> (f64, f64)
where
    F: FnMut(ArrayView1<'_, f64>) -> (f64, Array1<f64>),
{
    // The low endpoint retains a measured Armijo decrease. An infeasible
    // trial only contracts the other endpoint and cannot erase that evidence.
    let (mut phi_lo, mut dphi_lo) = if lo == 0.0 {
        (phi0, dphi0)
    } else {
        phi_pair(oracle, pos, dir, lo)
    };
    let mut unresolved_failures = 0;
    for _ in 0..maxiter {
        if !lo.is_finite() || !hi.is_finite() || super::bracket_rounding_blocked(lo, hi) {
            break;
        }
        let alpha = bisect(lo, hi);
        let pair = phi_pair(oracle, pos, dir, alpha);
        let phi_a = pair.0;
        let dphi_a = pair.1;
        if !phi_a.is_finite() || !dphi_a.is_finite() {
            hi = alpha;
            continue;
        }
        if (armijo(phi_a, phi0, alpha, dphi0, c1) && strong_curvature(dphi_a, dphi0, c2))
            || roundoff_strong_wolfe(phi_a, phi0, dphi_a, dphi0, c1, c2)
        {
            return (alpha, phi_a);
        }
        let armijo_holds = armijo(phi_a, phi0, alpha, dphi0, c1);
        if !armijo_holds && super::value_rounding_blocked(phi_a, phi_lo) {
            unresolved_failures += 1;
        } else {
            unresolved_failures = 0;
        }
        if super::rounding_stalled(unresolved_failures) {
            if super::resolvable_slope_crossing(dphi_a, dphi_lo, dphi0) {
                unresolved_failures = 0;
                hi = alpha;
                continue;
            }
            break;
        }
        if !armijo_holds || phi_a >= phi_lo {
            hi = alpha;
        } else {
            if dphi_a * (hi - lo) >= 0.0 {
                hi = lo;
            }
            lo = alpha;
            phi_lo = phi_a;
            dphi_lo = dphi_a;
        }
    }
    (lo, phi_lo)
}

fn phi_on_path<F, P>(oracle: &mut F, path: &mut P, alpha: f64) -> (f64, f64)
where
    F: FnMut(ArrayView1<'_, f64>) -> (f64, Array1<f64>),
    P: Path + ?Sized,
{
    let Some(x) = path.point(alpha) else {
        return (f64::INFINITY, f64::NAN);
    };
    let (f, g) = oracle(x.view());
    let dphi = path.slope(alpha, x.view(), g.view());
    (scored(f, g.view()), dphi)
}

/// Strong Wolfe constants with the measured `(phi(0), phi'(0))`.
#[derive(Clone, Copy)]
struct Wolfe {
    c1: f64,
    c2: f64,
    phi0: f64,
    dphi0: f64,
}

/// Zoom with a precomputed `(φ(0), φ'(0))` along a [`Path`].
fn zoom_on_path<F, P>(
    oracle: &mut F,
    path: &mut P,
    mut lo: f64,
    mut hi: f64,
    maxiter: usize,
    wolfe: Wolfe,
) -> (f64, f64)
where
    F: FnMut(ArrayView1<'_, f64>) -> (f64, Array1<f64>),
    P: Path + ?Sized,
{
    let Wolfe {
        c1,
        c2,
        phi0,
        dphi0,
    } = wolfe;
    // The low endpoint retains a measured Armijo decrease. An infeasible
    // trial only contracts the other endpoint and cannot erase that evidence.
    let (mut phi_lo, mut dphi_lo) = if lo == 0.0 {
        (phi0, dphi0)
    } else {
        phi_on_path(oracle, path, lo)
    };
    let mut unresolved_failures = 0;
    for _ in 0..maxiter {
        if !lo.is_finite() || !hi.is_finite() || super::bracket_rounding_blocked(lo, hi) {
            break;
        }
        let alpha = bisect(lo, hi);
        let pair = phi_on_path(oracle, path, alpha);
        let phi_a = pair.0;
        let dphi_a = pair.1;
        if !phi_a.is_finite() || !dphi_a.is_finite() {
            hi = alpha;
            continue;
        }
        if armijo(phi_a, phi0, alpha, dphi0, c1) && strong_curvature(dphi_a, dphi0, c2) {
            return (alpha, phi_a);
        }
        let armijo_holds = armijo(phi_a, phi0, alpha, dphi0, c1);
        if !armijo_holds && super::value_rounding_blocked(phi_a, phi_lo) {
            unresolved_failures += 1;
        } else {
            unresolved_failures = 0;
        }
        if super::rounding_stalled(unresolved_failures) {
            if super::resolvable_slope_crossing(dphi_a, dphi_lo, dphi0) {
                unresolved_failures = 0;
                hi = alpha;
                continue;
            }
            break;
        }
        if !armijo_holds || phi_a >= phi_lo {
            hi = alpha;
        } else {
            if dphi_a * (hi - lo) >= 0.0 {
                hi = lo;
            }
            lo = alpha;
            phi_lo = phi_a;
            dphi_lo = dphi_a;
        }
    }
    (lo, phi_lo)
}

/// Strong-Wolfe line search with Nocedal-Wright zoom (algorithm 3.5).
///
/// Initial trial is `istep`; the outer loop doubles until `alpha_max` or a
/// bracket is formed. Strong curvature permits equal rounded energies.
///
/// Wolfe, *Convergence Conditions for Ascent Methods*,
/// <https://doi.org/10.1137/1011036>.
/// Nocedal and Wright, *Numerical Optimization*,
/// <https://doi.org/10.1007/978-0-387-40065-5>.
pub(crate) fn wolfe_search<F, P>(
    oracle: &mut F,
    path: &mut P,
    istep: f64,
    c1: f64,
    c2: f64,
    maxiter: usize,
) -> (Array1<f64>, f64, f64)
where
    F: FnMut(ArrayView1<'_, f64>) -> (f64, Array1<f64>),
    P: Path + ?Sized,
{
    let pos = path.start();
    let (f0, g0) = oracle(pos.view());
    let dphi0: f64 = path.slope(0.0, pos.view(), g0.view());
    if unusable(f0, &g0, path.direction()) || !dphi0.is_finite() || dphi0 >= 0.0 {
        return (pos, f0, 0.0);
    }

    let alpha_max = 64.0_f64.max(istep.abs() * 64.0);
    let mut alpha_prev = 0.0;
    let mut phi_prev = f0;
    let mut alpha = istep.abs().max(1e-16);
    let wolfe = Wolfe {
        c1,
        c2,
        phi0: f0,
        dphi0,
    };

    for i in 0..maxiter {
        let (x, phi, dphi) = match path.point(alpha) {
            Some(x) => {
                let (phi, g) = oracle(x.view());
                let dphi = path.slope(alpha, x.view(), g.view());
                (x, scored(phi, g.view()), dphi)
            }
            None => (pos.clone(), f64::INFINITY, f64::NAN),
        };
        let fail_armijo = !phi.is_finite() || !armijo(phi, f0, alpha, dphi0, c1);
        if !fail_armijo && strong_curvature(dphi, dphi0, c2) {
            return (x, phi, alpha.abs());
        }
        if fail_armijo || !dphi.is_finite() || (i > 0 && phi >= phi_prev) {
            let zoomed = zoom_on_path(oracle, path, alpha_prev, alpha, maxiter, wolfe);
            return finish(oracle, path, &pos, zoomed, wolfe);
        }
        if dphi >= 0.0 {
            let zoomed = zoom_on_path(oracle, path, alpha, alpha_prev, maxiter, wolfe);
            return finish(oracle, path, &pos, zoomed, wolfe);
        }
        if (alpha - alpha_max).abs() < 1e-16 {
            break;
        }
        alpha_prev = alpha;
        phi_prev = phi;
        alpha = (2.0 * alpha).min(alpha_max);
    }
    // Every trial passed Armijo and still descended, so the expansion ran
    // out of budget or reached `alpha_max` with no bracket and no point
    // meeting strong curvature. A point that fails the configured
    // conditions is not returned as a step.
    (pos, f0, 0.0)
}

fn finish<F, P>(
    oracle: &mut F,
    path: &mut P,
    pos: &Array1<f64>,
    zoomed: (f64, f64),
    wolfe: Wolfe,
) -> (Array1<f64>, f64, f64)
where
    F: FnMut(ArrayView1<'_, f64>) -> (f64, Array1<f64>),
    P: Path + ?Sized,
{
    let Wolfe {
        c1,
        c2,
        phi0: f0,
        dphi0,
    } = wolfe;
    let (t, _) = zoomed;
    // A bracket endpoint can meet Armijo and still miss strong curvature.
    // That point, and any decreasing trial that only formed the bracket,
    // is not a Wolfe step.
    if t.is_finite() && t > 0.0 {
        let (ft, dphi) = phi_on_path(oracle, path, t);
        if strong_wolfe(ft, f0, t, dphi, dphi0, c1, c2)
            && let Some(x) = path.point(t)
        {
            return (x, ft, t);
        }
    }
    (pos.clone(), f0, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linesearch::Line;
    use ndarray::array;

    const C1: f64 = 1e-4;
    const C2: f64 = 0.1;

    fn linear(x: ArrayView1<'_, f64>) -> (f64, Array1<f64>) {
        (-x[0], array![-1.0])
    }

    fn quadratic(x: ArrayView1<'_, f64>) -> (f64, Array1<f64>) {
        (0.5 * x[0] * x[0], array![x[0]])
    }

    struct NanAtOne {
        start: Array1<f64>,
        direction: Array1<f64>,
    }

    impl Path for NanAtOne {
        fn start(&self) -> Array1<f64> {
            self.start.clone()
        }

        fn direction(&self) -> ArrayView1<'_, f64> {
            self.direction.view()
        }

        fn point(&mut self, t: f64) -> Option<Array1<f64>> {
            Some(array![t])
        }

        fn slope(&mut self, t: f64, _at: ArrayView1<'_, f64>, _g: ArrayView1<'_, f64>) -> f64 {
            if t == 1.0 { f64::NAN } else { t - 1.0 }
        }
    }

    #[test]
    fn zoom_contracts_away_from_an_unresolved_slope() {
        let mut path = NanAtOne {
            start: array![0.0],
            direction: array![1.0],
        };
        let mut shifted_bowl = |x: ArrayView1<'_, f64>| {
            let d = x[0] - 1.0;
            (0.5 * d * d, array![d])
        };
        let wolfe = Wolfe {
            c1: C1,
            c2: C2,
            phi0: 0.5,
            dphi0: -1.0,
        };
        let (alpha, _) = zoom_on_path(&mut shifted_bowl, &mut path, 0.0, 2.0, 20, wolfe);
        assert!(
            alpha > 0.0 && alpha < 1.0,
            "zoom crossed the unresolved slope at alpha = {alpha}"
        );
    }

    #[test]
    fn exhausted_expansion_does_not_accept_a_point_that_fails_curvature() {
        let start = array![0.0];
        let dir = array![1.0];
        let mut path = Line {
            pos: start.view(),
            dir: dir.view(),
        };
        let (x, f, alpha) = wolfe_search(&mut linear, &mut path, 1.0, C1, C2, 20);
        // The slope is -1 everywhere: curvature ratio 1 > c2 at every step.
        assert_eq!(alpha, 0.0, "accepted alpha = {alpha}, x = {x}");
        assert_eq!(x, start);
        assert_eq!(f, 0.0);
    }

    #[test]
    fn an_accepted_step_meets_both_wolfe_conditions() {
        let start = array![-3.0];
        let dir = array![1.0];
        let mut path = Line {
            pos: start.view(),
            dir: dir.view(),
        };
        let (x, f, alpha) = wolfe_search(&mut quadratic, &mut path, 1.0, C1, C2, 20);
        assert!(alpha > 0.0);
        let (f0, g0) = quadratic(start.view());
        let slope0 = g0[0] * dir[0];
        let slope = x[0] * dir[0];
        assert!(f <= f0 + C1 * alpha * slope0, "Armijo f = {f}");
        assert!(
            slope.abs() <= C2 * slope0.abs(),
            "curvature slope = {slope}"
        );
    }

    /// `phi(a) = (a - 10)^2 / 2` from 0 with direction +1.
    /// `phi(0) = 50`, `phi'(0) = -10`. Strong curvature at `c2 = 0.1`
    /// needs `|a - 10| <= 1`.
    fn bowl(x: ArrayView1<'_, f64>) -> (f64, Array1<f64>) {
        let d = x[0] - 10.0;
        (0.5 * d * d, array![d])
    }

    #[test]
    fn zoom_does_not_accept_an_armijo_point_that_misses_curvature() {
        // One bisection of the bracket [16, 0] lands at 8. Armijo holds
        // there and `|phi'(8)| = 2 > 1`. The expansion trial at 16
        // decreases and also misses curvature. Neither is returned.
        let start = array![0.0];
        let dir = array![1.0];
        let mut path = Line {
            pos: start.view(),
            dir: dir.view(),
        };
        let (x, f, alpha) = wolfe_search(&mut bowl, &mut path, 16.0, C1, C2, 1);
        assert_eq!(alpha, 0.0, "accepted alpha = {alpha}, x = {x}, f = {f}");
        assert_eq!(x, start);
        assert_eq!(f, 50.0);
    }

    #[test]
    fn zoom_accepts_a_point_that_meets_strong_wolfe() {
        let start = array![0.0];
        let dir = array![1.0];
        let mut path = Line {
            pos: start.view(),
            dir: dir.view(),
        };
        let (x, f, alpha) = wolfe_search(&mut bowl, &mut path, 16.0, C1, C2, 20);
        assert!(alpha > 0.0, "refused a step that meets both conditions");
        let slope = x[0] - 10.0;
        assert!(f <= 50.0 + C1 * alpha * -10.0, "Armijo f = {f}");
        assert!(slope.abs() <= C2 * 10.0, "curvature slope = {slope}");
    }
}
