//! Nocedal-Wright zoom (algorithm 3.6) and Wolfe search (algorithm 3.5).
//!
//! Trial interpolation is bisection (`BisectionStepSize`): the next α is the
//! midpoint of the current bracket, so every trial stays inside it.
//!
//! Wolfe, *Convergence Conditions for Ascent Methods*,
//! <https://doi.org/10.1137/1011036>.
//! Nocedal and Wright, *Numerical Optimization*,
//! <https://doi.org/10.1007/978-0-387-40065-5>.

use ndarray::{Array1, ArrayView1};

use super::conditions::{armijo, strong_curvature};
use super::{Line, Path, scored, unusable};
use crate::workspace::SearchBuffers;

/// Evaluate `phi(t) = f(x(t))` and `phi'(t) = grad f(x(t)) . x'(t)`, or
/// `(+inf, NaN)` where the path is undefined.
///
/// One oracle call yields both values so `φ` and `φ'` do not each hold a
/// mutable borrow of the same closure.
fn phi_pair<F, P>(oracle: &mut F, path: &mut P, alpha: f64, trial: &mut Array1<f64>) -> (f64, f64)
where
    F: FnMut(ArrayView1<'_, f64>) -> (f64, Array1<f64>),
    P: Path + ?Sized,
{
    if !path.point_into(alpha, trial) {
        return (f64::INFINITY, f64::NAN);
    }
    let (f, g) = oracle(trial.view());
    let dphi = path.slope(alpha, trial.view(), g.view());
    let score = scored(f, g.view());
    crate::workspace::recycle_array(g);
    (score, dphi)
}

/// Strong Wolfe constants with the measured `(phi(0), phi'(0))`.
#[derive(Clone, Copy)]
struct Wolfe {
    c1: f64,
    c2: f64,
    phi0: f64,
    dphi0: f64,
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
#[expect(
    clippy::too_many_arguments,
    reason = "public signature; grouping the arguments would break callers"
)]
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
    let path = &mut Line { pos, dir };
    let mut trial = Array1::zeros(pos.len());
    let (phi0, dphi0) = phi_pair(oracle, path, 0.0, &mut trial);
    let wolfe = Wolfe {
        c1,
        c2,
        phi0,
        dphi0,
    };
    zoom_into(oracle, path, lo, hi, maxiter, wolfe, &mut trial).0
}

/// Zoom with a precomputed `(φ(0), φ'(0))`.
fn zoom_into<F, P>(
    oracle: &mut F,
    path: &mut P,
    mut lo: f64,
    mut hi: f64,
    maxiter: usize,
    wolfe: Wolfe,
    trial: &mut Array1<f64>,
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
    let mut phi_lo = if lo == 0.0 {
        phi0
    } else {
        phi_pair(oracle, path, lo, trial).0
    };
    for _ in 0..maxiter {
        if !lo.is_finite() || !hi.is_finite() || (hi - lo).abs() < 1e-16 {
            break;
        }
        let alpha = bisect(lo, hi);
        let pair = phi_pair(oracle, path, alpha, trial);
        let phi_a = pair.0;
        let dphi_a = pair.1;
        if !phi_a.is_finite() {
            hi = alpha;
            continue;
        }
        if armijo(phi_a, phi0, alpha, dphi0, c1) && strong_curvature(dphi_a, dphi0, c2) {
            return (alpha, phi_a);
        }
        if !armijo(phi_a, phi0, alpha, dphi0, c1) || phi_a >= phi_lo {
            hi = alpha;
        } else {
            if dphi_a * (hi - lo) >= 0.0 {
                hi = lo;
            }
            lo = alpha;
            phi_lo = phi_a;
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
    bufs: &mut SearchBuffers,
) -> (f64, f64)
where
    F: FnMut(ArrayView1<'_, f64>) -> (f64, Array1<f64>),
    P: Path + ?Sized,
{
    path.start_into(&mut bufs.anchor);
    let (f0, g0) = oracle(bufs.anchor.view());
    let dphi0: f64 = path.slope(0.0, bufs.anchor.view(), g0.view());
    let bad = unusable(f0, &g0, path.direction()) || !dphi0.is_finite() || dphi0 >= 0.0;
    crate::workspace::recycle_array(g0);
    if bad {
        bufs.chosen = 0;
        return (f0, 0.0);
    }

    let alpha_max = 64.0_f64.max(istep.abs() * 64.0);
    let mut alpha_prev = 0.0;
    let mut phi_prev = f0;
    let mut alpha = istep.abs().max(1e-16);
    bufs.copy_anchor_to_best();
    let mut best_f = f0;
    let mut best_a = 0.0;
    let wolfe = Wolfe {
        c1,
        c2,
        phi0: f0,
        dphi0,
    };

    for i in 0..maxiter {
        let (phi, dphi) = if path.point_into(alpha, &mut bufs.trial) {
            let (phi, g) = oracle(bufs.trial.view());
            let dphi = path.slope(alpha, bufs.trial.view(), g.view());
            let score = scored(phi, g.view());
            crate::workspace::recycle_array(g);
            (score, dphi)
        } else {
            (f64::INFINITY, f64::NAN)
        };
        if phi.is_finite() && phi < best_f {
            best_f = phi;
            bufs.copy_trial_to_best();
            best_a = alpha;
        }
        let fail_armijo = !phi.is_finite() || !armijo(phi, f0, alpha, dphi0, c1);
        if !fail_armijo && strong_curvature(dphi, dphi0, c2) {
            bufs.chosen = 2;
            return (phi, alpha.abs());
        }
        if fail_armijo || (i > 0 && phi >= phi_prev) {
            let zoomed = zoom_into(
                oracle,
                path,
                alpha_prev,
                alpha,
                maxiter,
                wolfe,
                &mut bufs.trial,
            );
            return finish(oracle, path, zoomed, wolfe, (best_f, best_a), bufs);
        }
        if dphi >= 0.0 {
            let zoomed = zoom_into(
                oracle,
                path,
                alpha,
                alpha_prev,
                maxiter,
                wolfe,
                &mut bufs.trial,
            );
            return finish(oracle, path, zoomed, wolfe, (best_f, best_a), bufs);
        }
        if (alpha - alpha_max).abs() < 1e-16 {
            break;
        }
        alpha_prev = alpha;
        phi_prev = phi;
        alpha = (2.0 * alpha).min(alpha_max);
    }
    if best_f < f0 {
        bufs.chosen = 1;
        (best_f, best_a.abs())
    } else {
        bufs.chosen = 0;
        (f0, 0.0)
    }
}

fn finish<F, P>(
    oracle: &mut F,
    path: &mut P,
    zoomed: (f64, f64),
    wolfe: Wolfe,
    best: (f64, f64),
    bufs: &mut SearchBuffers,
) -> (f64, f64)
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
    let (best_f, best_a) = best;
    let (t, mut ft) = zoomed;
    if t.is_finite() && t.abs() > 0.0 {
        if !ft.is_finite() {
            ft = phi_pair(oracle, path, t, &mut bufs.trial).0;
        }
        let tied_wolfe = ft == f0
            && armijo(ft, f0, t, dphi0, c1)
            && strong_curvature(phi_pair(oracle, path, t, &mut bufs.trial).1, dphi0, c2);
        if ft.is_finite() && (ft < f0 || tied_wolfe) && path.point_into(t, &mut bufs.trial) {
            bufs.chosen = 2;
            return (ft, t.abs());
        }
    }
    if best_f < f0 {
        bufs.chosen = 1;
        (best_f, best_a.abs())
    } else {
        bufs.chosen = 0;
        (f0, 0.0)
    }
}
