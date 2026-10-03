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
use super::conditions::{armijo, strong_curvature};

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
    // Last evaluated pair. If the loop never interpolates, return `lo` with
    // `φ(lo)` rather than a midpoint tagged with a different value.
    let mut phi_lo = if lo == 0.0 {
        phi0
    } else {
        phi_pair(oracle, pos, dir, lo).0
    };
    let mut alpha = lo;
    let mut phi_a = phi_lo;
    for _ in 0..maxiter {
        if !lo.is_finite() || !hi.is_finite() || (hi - lo).abs() < 1e-16 {
            break;
        }
        alpha = bisect(lo, hi);
        let pair = phi_pair(oracle, pos, dir, alpha);
        phi_a = pair.0;
        let dphi_a = pair.1;
        if !phi_a.is_finite() {
            hi = alpha;
            continue;
        }
        if !armijo(phi_a, phi0, alpha, dphi0, c1) || phi_a >= phi_lo {
            hi = alpha;
        } else {
            if strong_curvature(dphi_a, dphi0, c2) {
                return (alpha, phi_a);
            }
            if dphi_a * (hi - lo) >= 0.0 {
                hi = lo;
            }
            lo = alpha;
            phi_lo = phi_a;
        }
    }
    (alpha, phi_a)
}
