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
    // The low endpoint retains a measured Armijo decrease. An infeasible
    // trial only contracts the other endpoint and cannot erase that evidence.
    let mut phi_lo = if lo == 0.0 {
        phi0
    } else {
        phi_pair(oracle, pos, dir, lo).0
    };
    for _ in 0..maxiter {
        if !lo.is_finite() || !hi.is_finite() || (hi - lo).abs() < 1e-16 {
            break;
        }
        let alpha = bisect(lo, hi);
        let pair = phi_pair(oracle, pos, dir, alpha);
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
pub(crate) fn wolfe_search<F>(
    oracle: &mut F,
    pos: ArrayView1<'_, f64>,
    dir: ArrayView1<'_, f64>,
    istep: f64,
    c1: f64,
    c2: f64,
    maxiter: usize,
) -> (Array1<f64>, f64, f64)
where
    F: FnMut(ArrayView1<'_, f64>) -> (f64, Array1<f64>),
{
    let (f0, g0) = oracle(pos);
    let dphi0: f64 = g0.iter().zip(dir.iter()).map(|(g, d)| g * d).sum();
    if !f0.is_finite() || !dphi0.is_finite() || dphi0 >= 0.0 {
        return (pos.to_owned(), f0, 0.0);
    }

    let alpha_max = 64.0_f64.max(istep.abs() * 64.0);
    let mut alpha_prev = 0.0;
    let mut phi_prev = f0;
    let mut alpha = istep.abs().max(1e-16);
    let mut best_x = pos.to_owned();
    let mut best_f = f0;
    let mut best_a = 0.0;

    for i in 0..maxiter {
        let x = axpy(pos, alpha, dir);
        let (phi, g) = oracle(x.view());
        let dphi: f64 = g.iter().zip(dir.iter()).map(|(gi, di)| gi * di).sum();
        if phi.is_finite() && phi < best_f {
            best_f = phi;
            best_x = x.clone();
            best_a = alpha;
        }
        let fail_armijo = !phi.is_finite() || !armijo(phi, f0, alpha, dphi0, c1);
        if !fail_armijo && strong_curvature(dphi, dphi0, c2) {
            return (x, phi, alpha.abs());
        }
        if fail_armijo || (i > 0 && phi >= phi_prev) {
            let zoomed = zoom_into(
                oracle, pos, dir, alpha_prev, alpha, c1, c2, maxiter, f0, dphi0,
            );
            return finish(
                oracle, pos, dir, zoomed, f0, dphi0, c1, c2, best_x, best_f, best_a,
            );
        }
        if dphi >= 0.0 {
            let zoomed = zoom_into(
                oracle, pos, dir, alpha, alpha_prev, c1, c2, maxiter, f0, dphi0,
            );
            return finish(
                oracle, pos, dir, zoomed, f0, dphi0, c1, c2, best_x, best_f, best_a,
            );
        }
        if (alpha - alpha_max).abs() < 1e-16 {
            break;
        }
        alpha_prev = alpha;
        phi_prev = phi;
        alpha = (2.0 * alpha).min(alpha_max);
    }
    if best_f < f0 {
        (best_x, best_f, best_a.abs())
    } else {
        (pos.to_owned(), f0, 0.0)
    }
}

fn finish<F>(
    oracle: &mut F,
    pos: ArrayView1<'_, f64>,
    dir: ArrayView1<'_, f64>,
    zoomed: (f64, f64),
    f0: f64,
    dphi0: f64,
    c1: f64,
    c2: f64,
    best_x: Array1<f64>,
    best_f: f64,
    best_a: f64,
) -> (Array1<f64>, f64, f64)
where
    F: FnMut(ArrayView1<'_, f64>) -> (f64, Array1<f64>),
{
    let (t, mut ft) = zoomed;
    if t.is_finite() && t.abs() > 0.0 {
        if !ft.is_finite() {
            ft = oracle(axpy(pos, t, dir).view()).0;
        }
        let tied_wolfe = ft == f0
            && armijo(ft, f0, t, dphi0, c1)
            && strong_curvature(phi_pair(oracle, pos, dir, t).1, dphi0, c2);
        if ft.is_finite() && (ft < f0 || tied_wolfe) {
            return (axpy(pos, t, dir), ft, t.abs());
        }
    }
    if best_f < f0 {
        (best_x, best_f, best_a.abs())
    } else {
        (pos.to_owned(), f0, 0.0)
    }
}
