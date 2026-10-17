//! Dense BFGS, L-BFGS, SR1, and Powell symmetric Broyden (SR2).

use eindir_core::{DifferentiableObjective, Objective};
use ndarray::{Array1, Array2};

use crate::control::Control;
use crate::error::{Error, Result};
use crate::linesearch::LineSearch;
use crate::report::Report;
use crate::step::{
    energy_noise_report, implicit_atom_cap, l2, next_istep, qn_istep, scale_step, take_step,
};
use crate::trust::{
    accept_ratio, cauchy_point, dogleg_direction, predicted_reduction, reduction_ratio,
    update_radius,
};

const CURVATURE: f64 = 1e-12;
const SR1_SKIP: f64 = 1e-8;

/// Inverse-BFGS (Nocedal-Wright 6.17).
///
/// Broyden, *The Convergence of a Class of Double-rank Minimization
/// Algorithms 1. General Considerations*,
/// <https://doi.org/10.1093/imamat/6.1.76>.
/// Fletcher, *A new approach to variable metric algorithms*,
/// <https://doi.org/10.1093/comjnl/13.3.317>.
/// Shanno, *Conditioning of quasi-Newton methods for function
/// minimization*, <https://doi.org/10.1090/s0025-5718-1970-0274029-x>.
/// Nocedal and Wright, *Numerical Optimization*,
/// <https://doi.org/10.1007/978-0-387-40065-5>.
pub fn minimize_bfgs<O>(
    obj: &O,
    init: impl Into<Array1<f64>>,
    control: &Control,
    linesearch: LineSearch,
) -> Result<Report>
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let mut pos = start(obj, init)?;
    let n = pos.len();
    let (mut value, mut grad) = obj.value_and_gradient(pos.view());
    let mut h = Array2::<f64>::eye(n);
    let mut istep = control.istep;

    for step in 0..control.maxiter {
        let gnorm = l2(&grad);
        if control.halt(gnorm, grad.len()).is_some() {
            return Ok(done(value, pos, step, gnorm, grad.len(), control, false));
        }
        let dir = -h.dot(&grad);
        let old = pos.clone();
        let gold = grad.clone();
        let t = take_step(
            obj,
            &pos,
            value,
            &grad,
            dir.view(),
            istep,
            linesearch,
            control,
            implicit_atom_cap(control),
        );
        if t.energy_noise {
            return Ok(energy_noise_report(value, pos, step + 1, gnorm));
        }
        let moved = t.moved;
        pos = t.x;
        value = t.f;
        grad = t.g;
        if moved {
            let s = &pos - &old;
            let y = &grad - &gold;
            bfgs_inverse_update(&mut h, &s, &y);
        }
        istep = qn_istep(control);
    }
    Ok(done(
        value,
        pos,
        control.maxiter,
        l2(&grad),
        grad.len(),
        control,
        true,
    ))
}

/// L-BFGS two-loop recursion (Nocedal-Wright 7.4), scaling 7.20.
///
/// Delegates the pair history and two-loop map to [`crate::Lbfgs`].
/// Nocedal, *Updating quasi-Newton matrices with limited storage*,
/// <https://doi.org/10.1090/s0025-5718-1980-0572855-7>.
/// Liu and Nocedal, *On the limited memory BFGS method for large scale
/// optimization*, <https://doi.org/10.1007/BF01589116>.
/// Nocedal and Wright, *Numerical Optimization*,
/// <https://doi.org/10.1007/978-0-387-40065-5>.
pub fn minimize_lbfgs<O>(
    obj: &O,
    init: impl Into<Array1<f64>>,
    control: &Control,
    linesearch: LineSearch,
    memory: usize,
) -> Result<Report>
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let mut solver = crate::lbfgs::Lbfgs::with_capacity(memory);
    solver.norm = crate::lbfgs::GradNorm::Euclidean;
    solver.gtol = control.gtol;
    solver.minimize_objective(obj, init, control, linesearch)
}

/// SR1 in a trust region (Nocedal-Wright 6.24 and algorithm 4.1).
///
/// The model Hessian starts at `I` and updates
/// `B+ = B + δδᵀ / (δᵀ s)`, `δ = y - Bs`, skipping
/// `|δᵀ s| < 1e-8 ||δ|| ||s||` (Nocedal-Wright 6.26). A positive-definite
/// model takes a dogleg step; an indefinite model takes the Cauchy point,
/// which stays downhill for the model when `-B^{-1} g` does not. `linesearch`
/// is accepted and ignored: a line search on the inverse SR1 Hessian is the
/// defect this replaces.
///
/// Nocedal and Wright, *Numerical Optimization*,
/// <https://doi.org/10.1007/978-0-387-40065-5>.
pub fn minimize_sr1<O>(
    obj: &O,
    init: impl Into<Array1<f64>>,
    control: &Control,
    linesearch: LineSearch,
) -> Result<Report>
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let _ = linesearch;
    minimize_sr_trust(obj, init, control, sr1_hessian_update)
}

/// Powell symmetric Broyden in the same trust region as SR1.
///
/// `B+ = B + (δ sᵀ + s δᵀ) / (sᵀ s) - (δᵀ s) s sᵀ / (sᵀ s)²`
/// with `δ = y - Bs`. This is the symmetric rank-2 least-change update:
/// `B+` is symmetric when `B` is, and `B+ s = y`. The historical
/// `sr2.hpp` map `B += δ (y + Bs)ᵀ / (δᵀ s)` is neither symmetric nor
/// secant. `linesearch` is ignored for the same reason as [`minimize_sr1`].
///
/// Powell, *A new algorithm for unconstrained optimization*, in
/// *Nonlinear Programming*, Academic Press, 1970.
pub fn minimize_sr2<O>(
    obj: &O,
    init: impl Into<Array1<f64>>,
    control: &Control,
    linesearch: LineSearch,
) -> Result<Report>
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let _ = linesearch;
    minimize_sr_trust(obj, init, control, sr2_hessian_update)
}

fn minimize_sr_trust<O>(
    obj: &O,
    init: impl Into<Array1<f64>>,
    control: &Control,
    update: fn(&mut Array2<f64>, &Array1<f64>, &Array1<f64>),
) -> Result<Report>
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let mut pos = start(obj, init)?;
    let n = pos.len();
    let (mut value, mut grad) = obj.value_and_gradient(pos.view());
    let mut b = Array2::<f64>::eye(n);
    let (mut radius, rmax) = sr_trust_limits(control.istep, n, None, control.maxmove);

    for step in 0..control.maxiter {
        let gnorm = l2(&grad);
        if control.halt(gnorm, grad.len()).is_some() {
            return Ok(done(value, pos, step, gnorm, grad.len(), control, false));
        }
        let dir = sr_trust_direction(&b, &grad, radius);
        let mut trial = &pos + &dir;
        if let Some(cap) = control.maxmove {
            scale_step(&pos, &mut trial, cap);
        }
        trial = obj.bounds().clip(trial.view());
        let p = &trial - &pos;
        let pnorm = l2(&p);
        if !(pnorm.is_finite() && pnorm > 1e-16) {
            radius = update_radius(radius, -1.0, 0.0, rmax);
            continue;
        }
        let (ft, gt) = obj.value_and_gradient(trial.view());
        if !ft.is_finite() || gt.iter().any(|v| !v.is_finite()) {
            radius = update_radius(radius, -1.0, pnorm, rmax);
            continue;
        }
        let pred = predicted_reduction(&b, &grad, &p);
        let rho = reduction_ratio(value - ft, pred);
        radius = update_radius(radius, rho, pnorm, rmax);
        if accept_ratio(rho) {
            let s = &trial - &pos;
            let y = &gt - &grad;
            update(&mut b, &s, &y);
            pos = trial;
            value = ft;
            grad = gt;
        }
    }
    Ok(done(
        value,
        pos,
        control.maxiter,
        l2(&grad),
        grad.len(),
        control,
        true,
    ))
}

/// Steepest descent: `d = -g` every iteration.
///
/// Nocedal and Wright, *Numerical Optimization*,
/// <https://doi.org/10.1007/978-0-387-40065-5>.
pub fn minimize_sd<O>(
    obj: &O,
    init: impl Into<Array1<f64>>,
    control: &Control,
    linesearch: LineSearch,
) -> Result<Report>
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let mut pos = start(obj, init)?;
    let (mut value, mut grad) = obj.value_and_gradient(pos.view());
    let mut istep = control.istep;
    for step in 0..control.maxiter {
        let gnorm = l2(&grad);
        if control.halt(gnorm, grad.len()).is_some() {
            return Ok(done(value, pos, step, gnorm, grad.len(), control, false));
        }
        let dir = grad.mapv(|g| -g);
        let t = take_step(
            obj,
            &pos,
            value,
            &grad,
            dir.view(),
            istep,
            linesearch,
            control,
            implicit_atom_cap(control),
        );
        if t.energy_noise {
            return Ok(energy_noise_report(value, pos, step + 1, gnorm));
        }
        let lsstep = t.alpha;
        pos = t.x;
        value = t.f;
        grad = t.g;
        istep = next_istep(lsstep, control);
    }
    Ok(done(
        value,
        pos,
        control.maxiter,
        l2(&grad),
        grad.len(),
        control,
        true,
    ))
}

fn start<O>(obj: &O, init: impl Into<Array1<f64>>) -> Result<Array1<f64>>
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let pos = init.into();
    if pos.len() != Objective::dim(obj) {
        return Err(Error::Dim {
            got: pos.len(),
            dim: Objective::dim(obj),
        });
    }
    Ok(obj.bounds().clip(pos.view()))
}

fn done(
    value: f64,
    coords: Array1<f64>,
    steps: usize,
    grad_norm: f64,
    n_free: usize,
    control: &Control,
    exhausted: bool,
) -> Report {
    Report {
        value,
        coords,
        steps,
        grad_norm,
        status: control.finish_status(grad_norm, n_free, exhausted),
    }
}

/// Inverse BFGS, Nocedal-Wright 6.17:
/// `H+ = (I - ρ s y^T) H (I - ρ y s^T) + ρ s s^T`, `ρ = 1/(y·s)`.
pub(crate) fn bfgs_inverse_update(h: &mut Array2<f64>, s: &Array1<f64>, y: &Array1<f64>) {
    let ys = y.dot(s);
    if ys <= CURVATURE {
        return;
    }
    let rho = 1.0 / ys;
    let hy = h.dot(y);
    let yhy = y.dot(&hy);
    let n = s.len();
    let mut hp = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let mut acc = h[(i, j)];
            acc -= rho * s[i] * hy[j];
            acc -= rho * hy[i] * s[j];
            acc += rho * rho * yhy * s[i] * s[j];
            acc += rho * s[i] * s[j];
            hp[(i, j)] = acc;
        }
    }
    *h = hp;
}

/// Powell symmetric Broyden: `B+ s = y` and `B+` stays symmetric.
pub(crate) fn sr2_hessian_update(b: &mut Array2<f64>, s: &Array1<f64>, y: &Array1<f64>) {
    if s.iter().any(|v| !v.is_finite()) || y.iter().any(|v| !v.is_finite()) {
        return;
    }
    let ss = s.dot(s);
    if !(ss.is_finite() && ss > CURVATURE) {
        return;
    }
    let delta = y - &b.dot(s);
    let ds = delta.dot(s);
    if !ds.is_finite() {
        return;
    }
    let n = s.len();
    let inv_ss = 1.0 / ss;
    let inv_ss2 = inv_ss * inv_ss;
    for i in 0..n {
        for j in 0..n {
            b[(i, j)] += (delta[i] * s[j] + s[i] * delta[j]) * inv_ss - ds * s[i] * s[j] * inv_ss2;
        }
    }
}

/// Initial trust radius and its cap.
///
/// An atom cap builds a Euclidean ball that can realize one capped
/// per-atom step (`cap * sqrt(n_atoms)`). Without one, the radius opens
/// at `max(istep, 1)` and may grow by a factor of eight.
pub(crate) fn sr_trust_limits(
    istep: f64,
    dim: usize,
    atom_maxmove: Option<f64>,
    maxmove: Option<f64>,
) -> (f64, f64) {
    let step = if istep.is_finite() && istep > 0.0 {
        istep
    } else {
        1e-8
    };
    if let Some(cap) = atom_maxmove.filter(|c| c.is_finite() && *c > 0.0) {
        let n_atoms = (dim / 3).max(1) as f64;
        let ball = (cap * n_atoms.sqrt()).max(step);
        return (ball, ball);
    }
    let radius = step.max(1.0);
    let rmax = maxmove
        .filter(|c| c.is_finite() && *c > 0.0)
        .map(|c| c.max(radius))
        .unwrap_or_else(|| (8.0 * radius).max(radius));
    (radius, rmax)
}

/// Dogleg when `B` is positive definite, otherwise the Cauchy point.
pub(crate) fn sr_trust_direction(b: &Array2<f64>, grad: &Array1<f64>, radius: f64) -> Array1<f64> {
    let rhs = grad.mapv(|g| -g);
    if solve_spd(b, &rhs).is_some() {
        dogleg_direction(b, grad, radius)
    } else {
        cauchy_point(b, grad, radius)
    }
}

/// Cholesky solve of `A x = rhs` for a symmetric positive-definite `A`.
///
/// `None` when `A` is not square of the right size, not symmetric to
/// `1e-10` relative, or a pivot `l_kk^2` is not above `1e-14` times the
/// largest diagonal entry: the factorisation is the SPD test, so a
/// matrix that would make `-A^{-1} g` an ascent direction never yields
/// one.
pub(crate) fn solve_spd(a: &Array2<f64>, rhs: &Array1<f64>) -> Option<Array1<f64>> {
    let n = rhs.len();
    if a.nrows() != n || a.ncols() != n {
        return None;
    }
    let scale = (0..n).map(|i| a[(i, i)].abs()).fold(0.0_f64, f64::max);
    if !(scale.is_finite() && scale > 0.0) {
        return None;
    }
    for i in 0..n {
        for j in 0..i {
            if (a[(i, j)] - a[(j, i)]).abs() > 1e-10 * scale {
                return None;
            }
        }
    }
    let mut l = Array2::<f64>::zeros((n, n));
    for j in 0..n {
        let mut d = a[(j, j)];
        for k in 0..j {
            d -= l[(j, k)] * l[(j, k)];
        }
        if d.is_nan() || d <= 1e-14 * scale {
            return None;
        }
        let ljj = d.sqrt();
        l[(j, j)] = ljj;
        for i in (j + 1)..n {
            let mut v = a[(i, j)];
            for k in 0..j {
                v -= l[(i, k)] * l[(j, k)];
            }
            l[(i, j)] = v / ljj;
        }
    }
    let mut y = rhs.clone();
    for i in 0..n {
        for k in 0..i {
            y[i] -= l[(i, k)] * y[k];
        }
        y[i] /= l[(i, i)];
    }
    for i in (0..n).rev() {
        for k in (i + 1)..n {
            y[i] -= l[(k, i)] * y[k];
        }
        y[i] /= l[(i, i)];
    }
    y.iter().all(|v| v.is_finite()).then_some(y)
}

#[cfg(test)]
mod update_tests {
    use super::*;
    use ndarray::array;

    /// Powell's symmetric Broyden update satisfies `B+ s = y` and stays
    /// symmetric. The historical `sr2.hpp` rank-1 map does neither.
    #[test]
    fn sr2_update_meets_the_secant_condition_and_stays_symmetric() {
        let mut b = Array2::<f64>::eye(2);
        let s = array![1.0, 0.5];
        let y = array![2.0, -0.25];
        sr2_hessian_update(&mut b, &s, &y);
        let bs = b.dot(&s);
        assert!(
            (bs[0] - y[0]).abs() < 1e-10 && (bs[1] - y[1]).abs() < 1e-10,
            "secant failed: Bs = {bs:?}, y = {y:?}, B = {b:?}"
        );
        assert!(
            (b[(0, 1)] - b[(1, 0)]).abs() < 1e-12,
            "symmetry failed: B = {b:?}"
        );
    }

    #[test]
    fn sr1_update_meets_the_secant_condition() {
        let mut b = Array2::<f64>::eye(2);
        let s = array![1.0, 0.5];
        let y = array![2.0, -0.25];
        sr1_hessian_update(&mut b, &s, &y);
        let bs = b.dot(&s);
        assert!(
            (bs[0] - y[0]).abs() < 1e-10 && (bs[1] - y[1]).abs() < 1e-10,
            "secant failed: Bs = {bs:?}"
        );
        assert!((b[(0, 1)] - b[(1, 0)]).abs() < 1e-12);
    }

    /// `-B^{-1} g` is uphill when the curvature along `g` is negative.
    /// The Cauchy point of that model is downhill.
    #[test]
    fn indefinite_model_steps_downhill() {
        let b = ndarray::array![[1.0, 0.0], [0.0, -4.0]];
        let g = array![0.0, 1.0];
        let p = sr_trust_direction(&b, &g, 0.5);
        assert!(g.dot(&p) < -1e-12, "g·p = {}, p = {p:?}", g.dot(&p));
        assert!((l2(&p) - 0.5).abs() < 1e-12);
    }
}

/// Gaussian elimination with partial pivoting for `A x = rhs`.
#[allow(dead_code)]
pub(crate) fn solve_dense(a: &Array2<f64>, rhs: &Array1<f64>) -> Option<Array1<f64>> {
    let n = rhs.len();
    let mut m = a.clone();
    let mut x = rhs.clone();
    for k in 0..n {
        let mut piv = k;
        let mut best = m[(k, k)].abs();
        for i in (k + 1)..n {
            let v = m[(i, k)].abs();
            if v > best {
                best = v;
                piv = i;
            }
        }
        if best <= CURVATURE {
            return None;
        }
        if piv != k {
            for j in 0..n {
                let tmp = m[(k, j)];
                m[(k, j)] = m[(piv, j)];
                m[(piv, j)] = tmp;
            }
            let tmp = x[k];
            x[k] = x[piv];
            x[piv] = tmp;
        }
        let akk = m[(k, k)];
        for i in (k + 1)..n {
            let f = m[(i, k)] / akk;
            m[(i, k)] = 0.0;
            for j in (k + 1)..n {
                m[(i, j)] -= f * m[(k, j)];
            }
            x[i] -= f * x[k];
        }
    }
    for i in (0..n).rev() {
        let mut acc = x[i];
        for j in (i + 1)..n {
            acc -= m[(i, j)] * x[j];
        }
        let di = m[(i, i)];
        if di.abs() <= CURVATURE {
            return None;
        }
        x[i] = acc / di;
    }
    Some(x)
}

/// Direct SR1: `B += δ δᵀ / (δᵀ s)`, `δ = y - Bs` (Nocedal-Wright 6.24).
pub(crate) fn sr1_hessian_update(b: &mut Array2<f64>, s: &Array1<f64>, y: &Array1<f64>) {
    if s.iter().any(|v| !v.is_finite()) || y.iter().any(|v| !v.is_finite()) {
        return;
    }
    let delta = y - &b.dot(s);
    let ds = delta.dot(s);
    let dn = l2(&delta);
    let sn = l2(s);
    if !ds.is_finite() || ds.abs() <= CURVATURE || ds.abs() < SR1_SKIP * dn * sn {
        return;
    }
    let n = s.len();
    for i in 0..n {
        for j in 0..n {
            b[(i, j)] += delta[i] * delta[j] / ds;
        }
    }
}
