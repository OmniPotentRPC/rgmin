//! Dense BFGS, L-BFGS, inverse SR1, and SR2.

use eindir_core::{DifferentiableObjective, Objective};
use ndarray::{Array1, Array2};

use crate::control::Control;
use crate::error::{Error, Result};
use crate::linesearch::LineSearch;
use crate::report::Report;
use crate::step::{l2, next_istep, qn_istep, take_step};

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
        if gnorm < control.gtol {
            return Ok(done(value, pos, step, gnorm));
        }
        let dir = -h.dot(&grad);
        let old = pos.clone();
        let gold = grad.clone();
        let (npos, _, _lsstep, moved) = take_step(
            obj,
            &pos,
            value,
            dir.view(),
            istep,
            linesearch,
            control,
            None,
        );
        pos = npos;
        let ev = obj.value_and_gradient(pos.view());
        value = ev.0;
        grad = ev.1;
        if moved {
            let s = &pos - &old;
            let y = &grad - &gold;
            bfgs_inverse_update(&mut h, &s, &y);
        }
        istep = qn_istep(control);
    }
    Ok(done(value, pos, control.maxiter, l2(&grad)))
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

/// Inverse SR1 (Nocedal-Wright 6.24 / 6.25).
///
/// Maintains `H ≈ B^{-1}` with `H += uu^T / (u·y)`, `u = s - Hy`.
/// The C++ `sr1.hpp` updates the Hessian `B` and inverts each step;
/// the inverse update is the same map on `H`. Skip when
/// `|u·y| < 1e-8 ||u|| ||y||` (Nocedal-Wright 6.26).
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
    let mut pos = start(obj, init)?;
    let n = pos.len();
    let (mut value, mut grad) = obj.value_and_gradient(pos.view());
    let mut h = Array2::<f64>::eye(n);
    let mut istep = control.istep;

    for step in 0..control.maxiter {
        let gnorm = l2(&grad);
        if gnorm < control.gtol {
            return Ok(done(value, pos, step, gnorm));
        }
        let dir = -h.dot(&grad);
        let old = pos.clone();
        let gold = grad.clone();
        let (npos, _, _lsstep, moved) = take_step(
            obj,
            &pos,
            value,
            dir.view(),
            istep,
            linesearch,
            control,
            None,
        );
        pos = npos;
        let ev = obj.value_and_gradient(pos.view());
        value = ev.0;
        grad = ev.1;
        if moved {
            let s = &pos - &old;
            let y = &grad - &gold;
            sr1_inverse_update(&mut h, &s, &y);
        }
        istep = qn_istep(control);
    }
    Ok(done(value, pos, control.maxiter, l2(&grad)))
}

/// SR2 Hessian update from rgmin `sr2.hpp`.
///
/// Maintains a dense Hessian `B` (starts as `I`). The search direction is
/// `-B^{-1} g`. After a move, `B += δ(y + Bs)^T / (δ^T s)` with `δ = y - Bs`.
/// This rank-1 map is the C++ `sr2.hpp` formula; it is not a Nocedal-Wright
/// numbered update.
pub fn minimize_sr2<O>(
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
    let mut b = Array2::<f64>::eye(n);
    let mut istep = control.istep;

    for step in 0..control.maxiter {
        let gnorm = l2(&grad);
        if gnorm < control.gtol {
            return Ok(done(value, pos, step, gnorm));
        }
        let rhs = grad.mapv(|g| -g);
        let dir = solve_dense(&b, &rhs).unwrap_or_else(|| rhs);
        let old = pos.clone();
        let gold = grad.clone();
        let (npos, _, _lsstep, moved) = take_step(
            obj,
            &pos,
            value,
            dir.view(),
            istep,
            linesearch,
            control,
            None,
        );
        pos = npos;
        let ev = obj.value_and_gradient(pos.view());
        value = ev.0;
        grad = ev.1;
        if moved {
            let s = &pos - &old;
            let y = &grad - &gold;
            sr2_hessian_update(&mut b, &s, &y);
        }
        istep = qn_istep(control);
    }
    Ok(done(value, pos, control.maxiter, l2(&grad)))
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
        if gnorm < control.gtol {
            return Ok(done(value, pos, step, gnorm));
        }
        let dir = grad.mapv(|g| -g);
        let (npos, _, lsstep, _) = take_step(
            obj,
            &pos,
            value,
            dir.view(),
            istep,
            linesearch,
            control,
            None,
        );
        pos = npos;
        let ev = obj.value_and_gradient(pos.view());
        value = ev.0;
        grad = ev.1;
        istep = next_istep(lsstep, control);
    }
    Ok(done(value, pos, control.maxiter, l2(&grad)))
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

fn done(value: f64, coords: Array1<f64>, steps: usize, grad_norm: f64) -> Report {
    Report {
        value,
        coords,
        steps,
        grad_norm,
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

/// `B += δ (y + Bs)^T / (δ·s)`, `δ = y - Bs` (`sr2.hpp`).
pub(crate) fn sr2_hessian_update(b: &mut Array2<f64>, s: &Array1<f64>, y: &Array1<f64>) {
    let bs = b.dot(s);
    let delta = y - &bs;
    let ybs = y + &bs;
    let denom = delta.dot(s);
    let dn = l2(&delta);
    let sn = l2(s);
    if denom.abs() < SR1_SKIP * dn * sn || denom.abs() <= CURVATURE {
        return;
    }
    let n = s.len();
    for i in 0..n {
        for j in 0..n {
            b[(i, j)] += delta[i] * ybs[j] / denom;
        }
    }
}

/// Gaussian elimination with partial pivoting for `A x = rhs`.
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

/// Inverse SR1: `H += u u^T / (u·y)`, `u = s - Hy` (Nocedal-Wright 6.25).
pub(crate) fn sr1_inverse_update(h: &mut Array2<f64>, s: &Array1<f64>, y: &Array1<f64>) {
    let hy = h.dot(y);
    let u = s - &hy;
    let uy = u.dot(y);
    let un = l2(&u);
    let yn = l2(y);
    if uy.abs() < SR1_SKIP * un * yn || uy.abs() <= CURVATURE {
        return;
    }
    let n = s.len();
    for i in 0..n {
        for j in 0..n {
            h[(i, j)] += u[i] * u[j] / uy;
        }
    }
}
