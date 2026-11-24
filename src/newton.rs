//! Shifted Newton and Banerjee RFO on a caller-supplied dense Hessian.
//!
//! This is not L-BFGS. When the Hessian is cheap (analytic pair potential,
//! model Hessian), Nocedal and Wright take a Newton step. Geometry codes
//! stabilize that step with a Levenberg shift or with rational function
//! optimization.
//!
//! Banerjee, Adams, Simons, Shepard, *Search for stationary points on
//! surfaces*, <https://doi.org/10.1021/j100247a015>.
//! Baker, *An algorithm for the location of transition states*,
//! <https://doi.org/10.1002/jcc.540070402>.
//! Nocedal and Wright, *Numerical Optimization*,
//! <https://doi.org/10.1007/978-0-387-40065-5>.

use eindir_core::{DifferentiableObjective, Objective};
use ndarray::{Array1, Array2, ArrayView1};

use crate::control::Control;
use crate::error::{Error, Result};
use crate::report::Report;
use crate::step::{apply_caps, implicit_atom_cap, l2, one_shot_report};
use crate::stop::{Counted, StopRules, counted, stop_after};

const ENERGY_RISE: f64 = 1.0e-8;
const EIG_FLOOR: f64 = 1.0e-8;
const LDLT_PIVOT: f64 = 1.0e-18;

/// Dense Hessian at a point. Row-major `n × n`, symmetric.
pub trait HessianObjective: DifferentiableObjective<f64> {
    /// Hessian `∇²f(x)`.
    fn hessian(&self, x: ArrayView1<f64>) -> Array2<f64>;
}

/// How the Newton direction is formed from `H` and `g`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NewtonKind {
    /// `d = -(H + μ I)^{-1} g` with `μ = max(0, ε - λ̂)` via LDLT.
    Shifted,
    /// Lowest mode of the Banerjee augmented Hessian.
    Rfo,
}

/// Closure adapter: fused `(f, g)` plus a dense Hessian.
pub struct HessianOracle<F, H>
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
    H: Fn(ArrayView1<f64>) -> Array2<f64> + Send + Sync,
{
    f: F,
    hess: H,
    bounds: eindir_core::Bounds<f64>,
}

impl<F, H> HessianOracle<F, H>
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
    H: Fn(ArrayView1<f64>) -> Array2<f64> + Send + Sync,
{
    /// Wide-box oracle of dimension `dim`.
    pub fn unbounded(dim: usize, f: F, hess: H) -> Self {
        const LO: f64 = -1e12;
        const HI: f64 = 1e12;
        Self {
            f,
            hess,
            bounds: eindir_core::Bounds::new(
                Array1::from_elem(dim, LO),
                Array1::from_elem(dim, HI),
                0.0,
            ),
        }
    }
}

impl<F, H> eindir_core::Objective<f64> for HessianOracle<F, H>
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
    H: Fn(ArrayView1<f64>) -> Array2<f64> + Send + Sync,
{
    fn dim(&self) -> usize {
        self.bounds.dims
    }

    fn bounds(&self) -> &eindir_core::Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        (self.f)(x).0
    }
}

impl<F, H> eindir_core::Gradient<f64> for HessianOracle<F, H>
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
    H: Fn(ArrayView1<f64>) -> Array2<f64> + Send + Sync,
{
    fn dim(&self) -> usize {
        self.bounds.dims
    }

    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        (self.f)(x).1
    }
}

impl<F, H> DifferentiableObjective<f64> for HessianOracle<F, H>
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
    H: Fn(ArrayView1<f64>) -> Array2<f64> + Send + Sync,
{
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.f)(x)
    }
}

impl<F, H> HessianObjective for HessianOracle<F, H>
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
    H: Fn(ArrayView1<f64>) -> Array2<f64> + Send + Sync,
{
    fn hessian(&self, x: ArrayView1<f64>) -> Array2<f64> {
        (self.hess)(x)
    }
}

/// First-order stationarity floor `n ε (1 + |V| + Σ |x_i|)`.
///
/// `n` is the length of `x` and `ε` is [`f64::EPSILON`]. A gradient
/// larger than this scale is not a stationary point of a
/// first-order evaluation of `V` at `x`.
pub fn backward_error_floor(value: f64, x: ArrayView1<f64>) -> f64 {
    let eps = f64::EPSILON;
    let mut sum_abs = 0.0;
    for component in x {
        sum_abs += component.abs();
    }
    (x.len() as f64) * eps * (1.0 + value.abs() + sum_abs)
}

/// Whether `p` is a backward-error Newton step at `(value, x)`.
///
/// `p` is the Newton step, or the shifted-Newton / RFO step when the
/// Hessian is indefinite. The point is accepted only when the Newton
/// decrement `|g·p| / 2` is at most `ε (1 + |V|)`, every `|p_i|` is
/// at most `ε (1 + |x_i|)`, and `||g||` is at most
/// [`backward_error_floor`], with `ε` = [`f64::EPSILON`].
///
/// A tiny step or a tiny decrement is not acceptance on its own. On an
/// ill-conditioned quadratic those two tests can pass while `||g||` is
/// still far above the floor.
pub fn newton_decrement_acceptable(
    value: f64,
    x: ArrayView1<f64>,
    grad: ArrayView1<f64>,
    step: ArrayView1<f64>,
) -> bool {
    if !roundoff_step(value, x, grad, step) {
        return false;
    }
    let gradient_norm = grad.dot(&grad).sqrt();
    at_most(gradient_norm, backward_error_floor(value, x))
}

/// Decrement and coordinate tests, without the gradient floor.
fn roundoff_step(
    value: f64,
    x: ArrayView1<f64>,
    grad: ArrayView1<f64>,
    step: ArrayView1<f64>,
) -> bool {
    if x.len() != step.len() || grad.len() != step.len() {
        return false;
    }
    let eps = f64::EPSILON;
    let decrement = grad.dot(&step).abs() / 2.0;
    if !at_most(decrement, eps * (1.0 + value.abs())) {
        return false;
    }
    for i in 0..step.len() {
        if !at_most(step[i].abs(), eps * (1.0 + x[i].abs())) {
            return false;
        }
    }
    true
}

/// `left <= right`, and false when either side is not ordered.
fn at_most(left: f64, right: f64) -> bool {
    matches!(
        left.partial_cmp(&right),
        Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
    )
}

fn newton_direction(kind: NewtonKind, hess: &Array2<f64>, grad: &Array1<f64>) -> Array1<f64> {
    match kind {
        NewtonKind::Shifted => shifted_newton(hess, grad),
        NewtonKind::Rfo => rfo_direction(hess, grad),
    }
}

/// Newton polish from `init`.
///
/// Each direction is the shifted Newton step, or the RFO step when
/// `kind` is [`NewtonKind::Rfo`]. An indefinite Hessian takes that
/// existing step. The point is kept only when
/// [`newton_decrement_acceptable`] holds, including after the last
/// step. Otherwise the report is [`crate::StopReason::PolishUnmet`].
pub fn polish_newton<O>(
    obj: &O,
    init: impl Into<Array1<f64>>,
    control: &Control,
    kind: NewtonKind,
) -> Result<Report>
where
    O: HessianObjective + ?Sized,
{
    counted(obj, control, |obj| {
        let report = polish_inner(obj, init.into(), control, kind)?;
        control.deliver(report)
    })
}

pub(crate) fn polish_inner<O>(
    obj: &O,
    mut pos: Array1<f64>,
    control: &Control,
    kind: NewtonKind,
) -> Result<Report>
where
    O: HessianObjective + ?Sized,
{
    if pos.len() != eindir_core::Objective::dim(obj) {
        return Err(Error::Dim {
            got: pos.len(),
            dim: eindir_core::Objective::dim(obj),
        });
    }
    pos = obj.bounds().clip(pos.view());
    let (mut value, mut grad) = obj.value_and_gradient(pos.view());
    crate::step::refuse_non_finite(value, &grad)?;
    let mut steps = 0usize;
    let mut stalled = false;
    for _ in 0..control.maxiter {
        let hess = obj.hessian(pos.view());
        let dir = newton_direction(kind, &hess, &grad);
        if newton_decrement_acceptable(value, pos.view(), grad.view(), dir.view()) {
            return Ok(polished_report(value, pos, steps, &grad, true));
        }
        let (npos, nval, ngrad, moved) = energy_backtrack(obj, &pos, value, &dir, control);
        steps += 1;
        if !moved {
            stalled = true;
            break;
        }
        pos = npos;
        value = nval;
        grad = ngrad;
        crate::step::refuse_non_finite(value, &grad)?;
    }
    let hess = obj.hessian(pos.view());
    let dir = newton_direction(kind, &hess, &grad);
    let accepted = newton_decrement_acceptable(value, pos.view(), grad.view(), dir.view());
    if stalled && !accepted {
        return Ok(polished_report(value, pos, steps, &grad, false)
            .with_stop_reason(crate::StopReason::LineSearchFailed));
    }
    Ok(polished_report(value, pos, steps, &grad, accepted))
}

fn polished_report(
    value: f64,
    pos: Array1<f64>,
    steps: usize,
    grad: &Array1<f64>,
    accepted: bool,
) -> Report {
    let gnorm = l2(grad);
    let gradient_ok = at_most(gnorm, backward_error_floor(value, pos.view()));
    if accepted && gradient_ok {
        one_shot_report(value, pos, steps, gnorm, crate::report::Status::Tolerance)
            .with_stop_reason(crate::StopReason::Converged)
    } else {
        // Tiny step or decrement without the gradient criterion.
        one_shot_report(value, pos, steps, gnorm, crate::report::Status::Budget)
            .with_stop_reason(crate::StopReason::PolishUnmet)
    }
}

fn finish_newton<O>(obj: &O, control: &Control, kind: NewtonKind, report: Report) -> Result<Report>
where
    O: HessianObjective + ?Sized,
{
    if !control.backward_polish {
        return control.deliver(report);
    }
    let prior = report.steps;
    let coords = report.coords.clone();
    let polished = polish_inner(obj, coords, control, kind)?;
    let steps = prior + polished.steps;
    control.deliver(polished.with_steps(steps))
}

/// Minimize with a Newton or RFO direction on the supplied Hessian.
pub fn minimize_newton<O>(
    obj: &O,
    init: impl Into<Array1<f64>>,
    control: &Control,
    kind: NewtonKind,
) -> Result<Report>
where
    O: HessianObjective + ?Sized,
{
    counted(obj, control, |obj| newton_counted(obj, init, control, kind))
}

fn newton_counted<O>(
    obj: &Counted<'_, O>,
    init: impl Into<Array1<f64>>,
    control: &Control,
    kind: NewtonKind,
) -> Result<Report>
where
    O: HessianObjective + ?Sized,
{
    let mut pos = init.into();
    if pos.len() != eindir_core::Objective::dim(obj) {
        return Err(Error::Dim {
            got: pos.len(),
            dim: eindir_core::Objective::dim(obj),
        });
    }
    pos = obj.bounds().clip(pos.view());
    let (mut value, mut grad) = obj.value_and_gradient(pos.view());
    crate::step::refuse_non_finite(value, &grad)?;
    let mut stops = StopRules::from_start(value);

    for step in 0..control.maxiter {
        let gnorm = l2(&grad);
        if let Some(status) = control.halt(gnorm, grad.len()) {
            return finish_newton(
                obj,
                control,
                kind,
                one_shot_report(value, pos, step, gnorm, status),
            );
        }
        let hess = obj.hessian(pos.view());
        let dir = newton_direction(kind, &hess, &grad);
        let (npos, nval, ngrad, moved) = energy_backtrack(obj, &pos, value, &dir, control);
        let mut accepted = moved;
        if !moved {
            let sd = grad.mapv(|g| -g);
            let (spos, sval, sgrad, sok) = energy_backtrack(obj, &pos, value, &sd, control);
            if sok {
                pos = spos;
                value = sval;
                grad = sgrad;
                accepted = true;
            } else if control.require_convergence && !control.converged(gnorm) {
                return finish_newton(
                    obj,
                    control,
                    kind,
                    one_shot_report(
                        value,
                        pos,
                        step + 1,
                        gnorm,
                        crate::report::Status::StepTolerance,
                    ),
                );
            }
        } else {
            pos = npos;
            value = nval;
            grad = ngrad;
        }
        if let Some(status) = stop_after(&mut stops, control, obj, value, accepted, &grad) {
            return control.deliver(one_shot_report(value, pos, step + 1, l2(&grad), status));
        }
    }
    let gnorm = l2(&grad);
    finish_newton(
        obj,
        control,
        kind,
        one_shot_report(
            value,
            pos,
            control.maxiter,
            gnorm,
            control.finish_status(gnorm, grad.len(), true),
        ),
    )
}

pub(crate) fn energy_backtrack<O>(
    obj: &O,
    pos: &Array1<f64>,
    value: f64,
    dir: &Array1<f64>,
    control: &Control,
) -> (Array1<f64>, f64, Array1<f64>, bool)
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let mut alpha = 1.0;
    for _ in 0..10 {
        let mut trial = pos + &(dir * alpha);
        apply_caps(
            pos,
            &mut trial,
            implicit_atom_cap(control),
            control.maxmove_cap(),
        );
        trial = obj.bounds().clip(trial.view());
        let (ft, gt) = obj.value_and_gradient(trial.view());
        if ft - value <= ENERGY_RISE {
            return (trial, ft, gt, true);
        }
        alpha *= 0.5;
    }
    (pos.clone(), value, Array1::zeros(pos.len()), false)
}

pub(crate) fn shifted_newton(h: &Array2<f64>, g: &Array1<f64>) -> Array1<f64> {
    let n = g.len();
    let mut mu = 0.0;
    for _ in 0..24 {
        let mut a = h.clone();
        for i in 0..n {
            a[(i, i)] += mu;
        }
        if let Some(d) = ldlt_solve(&a, g, false) {
            return -d;
        }
        mu = if mu == 0.0 { EIG_FLOOR } else { mu * 10.0 };
    }
    g.mapv(|v| -v)
}

pub(crate) fn rfo_direction(h: &Array2<f64>, g: &Array1<f64>) -> Array1<f64> {
    // Minimization selects the lowest eigenpair of [H g; g^T 0].
    // The stationary equation also has higher roots with uphill steps.
    crate::sella_step::rfo_get_s(h, g, 0, 1.0)
}

/// LDLT solve `A x = b`. `allow_indefinite` keeps negative pivots (RFO).
fn ldlt_solve(a: &Array2<f64>, b: &Array1<f64>, allow_indefinite: bool) -> Option<Array1<f64>> {
    let n = b.len();
    if a.nrows() != n || a.ncols() != n {
        return None;
    }
    let mut l = Array2::<f64>::zeros((n, n));
    let mut d = Array1::<f64>::zeros(n);
    for j in 0..n {
        let mut dj = a[(j, j)];
        for k in 0..j {
            dj -= l[(j, k)] * l[(j, k)] * d[k];
        }
        if dj.abs() < LDLT_PIVOT {
            return None;
        }
        if !allow_indefinite && dj <= 0.0 {
            return None;
        }
        d[j] = dj;
        l[(j, j)] = 1.0;
        for i in (j + 1)..n {
            let mut lij = a[(i, j)];
            for k in 0..j {
                lij -= l[(i, k)] * l[(j, k)] * d[k];
            }
            l[(i, j)] = lij / dj;
        }
    }
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[(i, k)] * y[k];
        }
        y[i] = s;
    }
    let mut z = Array1::<f64>::zeros(n);
    for i in 0..n {
        z[i] = y[i] / d[i];
    }
    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut s = z[i];
        for k in (i + 1)..n {
            s -= l[(k, i)] * x[k];
        }
        x[i] = s;
    }
    Some(x)
}

/// Structured failure modes for Bunch-Kaufman `P A P^T = L D L^T`
/// factorization and backward-error inertia bounds.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, thiserror::Error)]
pub enum LdlError {
    /// Matrix order `n` is zero or `a.len()` does not equal `n * n`.
    #[error("LDL^T dimension mismatch: len = {len}, n = {n}")]
    DimensionMismatch {
        /// Supplied slice length.
        len: usize,
        /// Expected square matrix order `n`.
        n: usize,
    },
    /// Matrix order `n` exceeds the IEEE-754 binary64 `γ_{2n+4} < 1` domain.
    #[error("LDL^T matrix order n = {n} exceeds binary64 rounding-bound domain")]
    OrderOverflow {
        /// Matrix order `n`.
        n: usize,
    },
    /// Input matrix contains a non-finite entry at `(row, col)`.
    #[error("LDL^T non-finite input entry at ({row}, {col})")]
    NonFiniteInput {
        /// Row index.
        row: usize,
        /// Column index.
        col: usize,
    },
    /// Intermediate Schur complement, factor entry, or rounding bound overflowed to non-finite.
    #[error("LDL^T non-finite intermediate value during {stage}")]
    NonFiniteFactor {
        /// Stage where overflow or NaN occurred.
        stage: &'static str,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum PivotBlock {
    One {
        index: usize,
        d: f64,
    },
    Two {
        index: usize,
        d00: f64,
        d01: f64,
        d11: f64,
    },
}

/// Outward-rounded Higham (2002, Eq. 3.2, Theorem 3.1, Lemma 3.3) rounding constant
///
/// `\gamma_k = k u / (1 - k u) = k u + (k u)^2 / (1 - k u)`
///
/// for IEEE-754 binary64 unit roundoff `u = 2^{-53} = 0.5 * f64::EPSILON`,
/// including the exact second-order and higher-order `O(u^2)` remainder
/// `(k u)^2 / (1 - k u)`.
fn gamma_up(k: usize) -> Option<f64> {
    let u = 0.5 * f64::EPSILON;
    let ku = ((k as f64) * u).next_up();
    if !ku.is_finite() || ku >= 1.0 {
        return None;
    }
    let denom = (1.0 - ku).next_down();
    if !denom.is_finite() || denom <= 0.0 {
        return None;
    }
    let second_and_higher = ((ku * ku).next_up() / denom).next_up();
    let total = (ku + second_and_higher).next_up();
    total.is_finite().then_some(total)
}

#[inline]
fn add_up(a: f64, b: f64) -> f64 {
    if a == 0.0 {
        b
    } else if b == 0.0 {
        a
    } else {
        (a + b).next_up()
    }
}

#[inline]
fn mul_up(a: f64, b: f64) -> f64 {
    if a == 0.0 || b == 0.0 {
        0.0
    } else if a == 1.0 {
        b
    } else if b == 1.0 {
        a
    } else {
        (a * b).next_up()
    }
}

fn pow2_checked(e: i32) -> Option<f64> {
    if !(-1022..=1023).contains(&e) {
        return None;
    }
    Some(f64::from_bits(((1023 + e) as u64) << 52))
}

fn frexp_pow2_pair(max_abs: f64) -> Option<((f64, f64), (f64, f64))> {
    if !max_abs.is_finite() || max_abs <= 0.0 {
        return None;
    }
    let bits = max_abs.to_bits();
    let biased_exp = ((bits >> 52) & 0x7ff) as i32;
    let k = if biased_exp > 0 {
        biased_exp - 1023
    } else {
        let mant = bits & ((1_u64 << 52) - 1);
        let lz = mant.leading_zeros() as i32;
        -1074 + (63 - lz)
    };
    let e1 = k / 2;
    let e2 = k - e1;
    let s1 = pow2_checked(e1)?;
    let s2 = pow2_checked(e2)?;
    let inv_s1 = pow2_checked(-e1)?;
    let inv_s2 = pow2_checked(-e2)?;
    Some(((s1, s2), (inv_s1, inv_s2)))
}

/// Compute rigorous outward-rounded IEEE-754 intervals `[(lo_1, hi_1), (lo_2, hi_2)]`
/// enclosing the two real eigenvalues `(tr - disc) / 2` and `(tr + disc) / 2` of the
/// symmetric 2x2 block `[[d00, d01], [d01, d11]]` using only IEEE-754 basic operations
/// (`+`, `-`, `*`, `sqrt`), exact power-of-two exponent scaling across the full finite
/// `f64` range (`f64::from_bits(1)` through `±f64::MAX`), and multiplication-certified
/// square-root bounds (`disc_lo^2 <= sum_sq_lo` and `disc_hi^2 >= sum_sq_hi`).
fn two_by_two_eigenvalue_intervals(d00: f64, d01: f64, d11: f64) -> Option<[(f64, f64); 2]> {
    if !d00.is_finite() || !d01.is_finite() || !d11.is_finite() {
        return None;
    }
    let abs00 = d00.abs();
    let abs01 = d01.abs();
    let abs11 = d11.abs();

    if abs01 == 0.0 {
        let (lo, hi) = if d00 <= d11 { (d00, d11) } else { (d11, d00) };
        return Some([(lo, lo), (hi, hi)]);
    }
    if d00 == 0.0 && d11 == 0.0 {
        return Some([(-abs01, -abs01), (abs01, abs01)]);
    }

    let mut max_abs = abs00;
    if abs01 > max_abs {
        max_abs = abs01;
    }
    if abs11 > max_abs {
        max_abs = abs11;
    }
    if max_abs == 0.0 {
        return Some([(0.0, 0.0), (0.0, 0.0)]);
    }

    let ((s1, s2), (inv_s1, inv_s2)) = frexp_pow2_pair(max_abs)?;
    let eta = f64::from_bits(1);

    let scale_interval = |x: f64| -> (f64, f64) {
        if x == 0.0 {
            (0.0, 0.0)
        } else {
            let sx = (x * inv_s1) * inv_s2;
            if sx.abs() >= f64::MIN_POSITIVE {
                (sx, sx)
            } else {
                (sx.next_down(), sx.next_up())
            }
        }
    };

    let (d00_lo, d00_hi) = scale_interval(d00);
    let (d01_lo, d01_hi) = scale_interval(d01);
    let (d11_lo, d11_hi) = scale_interval(d11);

    let tr_lo = (d00_lo + d11_lo).next_down();
    let tr_hi = (d00_hi + d11_hi).next_up();

    let diff_lo = (d00_lo - d11_hi).next_down();
    let diff_hi = (d00_hi - d11_lo).next_up();

    let abs_interval = |lo: f64, hi: f64| -> (f64, f64) {
        let a_lo = lo.abs();
        let a_hi = hi.abs();
        let min_a = if lo <= 0.0 && hi >= 0.0 {
            0.0
        } else if a_lo < a_hi {
            a_lo
        } else {
            a_hi
        };
        let max_a = if a_lo > a_hi { a_lo } else { a_hi };
        (min_a, max_a)
    };

    let (diff_abs_lo, diff_abs_hi) = abs_interval(diff_lo, diff_hi);
    let (d01_abs_lo, d01_abs_hi) = abs_interval(d01_lo, d01_hi);

    let two_b_lo = if d01_abs_lo == 0.0 {
        0.0
    } else {
        let v = (2.0 * d01_abs_lo).next_down();
        if v > 0.0 { v } else { 0.0 }
    };
    let two_b_hi = if d01_abs_hi == 0.0 {
        0.0
    } else {
        (2.0 * d01_abs_hi).next_up()
    };

    let sq_down = |x: f64| -> f64 {
        if x <= 0.0 {
            0.0
        } else {
            let p = (x * x).next_down();
            if p > 0.0 { p } else { 0.0 }
        }
    };
    let sq_up = |x: f64| -> f64 { if x <= 0.0 { 0.0 } else { (x * x).next_up() } };

    let s1_lo = sq_down(diff_abs_lo);
    let s1_hi = sq_up(diff_abs_hi);
    let s2_lo = sq_down(two_b_lo);
    let s2_hi = sq_up(two_b_hi);

    let sum_sq_lo = if s1_lo == 0.0 {
        s2_lo
    } else if s2_lo == 0.0 {
        s1_lo
    } else {
        let s = (s1_lo + s2_lo).next_down();
        if s > 0.0 { s } else { 0.0 }
    };
    let sum_sq_hi = if s1_hi == 0.0 && s2_hi == 0.0 {
        0.0
    } else {
        (s1_hi + s2_hi).next_up()
    };

    let disc_lo = if sum_sq_lo <= 0.0 {
        0.0
    } else {
        let mut r = sum_sq_lo.sqrt().next_down();
        if r < 0.0 {
            r = 0.0;
        }
        let mut steps = 0usize;
        while r > 0.0 && (r * r).next_up() > sum_sq_lo {
            r = r.next_down();
            if r < 0.0 {
                r = 0.0;
                break;
            }
            steps += 1;
            if steps > 8 {
                return None;
            }
        }
        r
    };
    let disc_hi = if sum_sq_hi <= 0.0 {
        0.0
    } else {
        let mut r = sum_sq_hi.sqrt().next_up();
        let mut steps = 0usize;
        while (r * r).next_down() < sum_sq_hi {
            r = r.next_up();
            if !r.is_finite() {
                return None;
            }
            steps += 1;
            if steps > 8 {
                return None;
            }
        }
        r
    };

    let eig1_scaled_lo = (0.5 * (tr_lo - disc_hi).next_down()).next_down();
    let eig1_scaled_hi = (0.5 * (tr_hi - disc_lo).next_up()).next_up();
    let eig2_scaled_lo = (0.5 * (tr_lo + disc_lo).next_down()).next_down();
    let eig2_scaled_hi = (0.5 * (tr_hi + disc_hi).next_up()).next_up();

    let eta_scaled = (eta * inv_s1) * inv_s2;

    let unscale_lo = |x: f64| -> f64 {
        if x == 0.0 {
            0.0
        } else {
            let v = (x * s2) * s1;
            if v == f64::INFINITY {
                f64::MAX
            } else if v == f64::NEG_INFINITY {
                f64::NEG_INFINITY
            } else if v.abs() >= f64::MIN_POSITIVE {
                v
            } else {
                let down = v.next_down();
                if x > 0.0 && eta_scaled > 0.0 && x >= eta_scaled && down < eta {
                    eta
                } else {
                    down
                }
            }
        }
    };
    let unscale_hi = |x: f64| -> f64 {
        if x == 0.0 {
            0.0
        } else {
            let v = (x * s2) * s1;
            if v == f64::NEG_INFINITY {
                -f64::MAX
            } else if v == f64::INFINITY {
                f64::INFINITY
            } else if v.abs() >= f64::MIN_POSITIVE {
                v
            } else {
                let up = v.next_up();
                if x < 0.0 && eta_scaled > 0.0 && x <= -eta_scaled && up > -eta {
                    -eta
                } else {
                    up
                }
            }
        }
    };

    let lo1 = unscale_lo(eig1_scaled_lo);
    let hi1 = unscale_hi(eig1_scaled_hi);
    let lo2 = unscale_lo(eig2_scaled_lo);
    let hi2 = unscale_hi(eig2_scaled_hi);
    if lo1.is_nan()
        || hi1.is_nan()
        || lo2.is_nan()
        || hi2.is_nan()
        || lo1 > hi1
        || lo2 > hi2
        || lo1 == f64::INFINITY
        || hi1 == f64::NEG_INFINITY
        || lo2 == f64::INFINITY
        || hi2 == f64::NEG_INFINITY
    {
        return None;
    }
    Some([(lo1, hi1), (lo2, hi2)])
}

/// Standard-form Bunch-Kaufman symmetric indefinite factorization
/// `P M P^T = L D L^T` where `M = (A + A^T) / 2`.
///
/// Unlike LAPACK `dsytrf`, which stores interleaved elementary permutations
/// and Gauss transformations `P_1 L_1 P_2 L_2 ...`, every pivot interchange
/// `p <-> q` during elimination simultaneously permutes rows `p` and `q`
/// across all previously computed columns `0..k` of `L`. Thus the stored unit
/// lower-triangular matrix `L` is in exact standard form `P M P^T = L D L^T`,
/// and its row sums give the exact standard-form norm `‖L‖_∞`.
#[derive(Clone, Debug)]
pub struct BunchKaufman {
    n: usize,
    perm: Vec<usize>,
    l: Vec<f64>,
    blocks: Vec<PivotBlock>,
    u_a_inf: f64,
    asym_inf: f64,
    m_max: f64,
    max_schur: f64,
}

/// Bunch-Kaufman `P M P^T = L D L^T` pivot inertia counts paired with an
/// explicit-constant backward-error bound and standard-form factor norms.
///
/// Because an unshifted factorization satisfies `P M P^T + \Delta M = \hat{L} \hat{D} \hat{L}^T`
/// with `‖\Delta M‖_∞ <= backward_error`, the pivot inertia of `\hat{D}` is the
/// exact inertia of the perturbed matrix `M + P^T \Delta M P`. For eigenvalues of `A`
/// lying inside `[-\Delta_{ldl}, +\Delta_{ldl}]`, unshifted `negative` / `positive`
/// counts are diagnostic even when `ambiguous_pivots == 0`; use
/// [`certify_shifted_ldlt_inertia`] to certify the inertia of `A` across the
/// backward-error radius via the two-sided Loewner sandwich `M \pm \Delta I`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LdlInertiaBound {
    /// Number of eigenvalues of `D` whose outward-rounded enclosure lies strictly below `0`.
    pub negative: usize,
    /// Number of eigenvalues of `D` whose outward-rounded enclosure lies strictly above `0`.
    pub positive: usize,
    /// Number of `1x1` or `2x2` block eigenvalues of `D` whose outward-rounded enclosure contains `0`.
    pub ambiguous_pivots: usize,
    /// Explicit-constant spectral/infinity backward-error bound `‖ΔA‖_2 <= ‖ΔA‖_∞`
    /// patterned after the structure of Higham (2002, Theorem 11.3, where the linear
    /// polynomial `p(n)` and `2x2` pivot constant `c` are left unspecified) and
    /// validated against exact rational reconstruction across adversarial `1x1` and
    /// `2x2` pivot matrices:
    ///
    /// `\Delta_{ldl} = \gamma_{2n+4} ‖|\hat{L}||\hat{D}||\hat{L}|^T‖_∞
    ///   + n \gamma_{2n} \rho_n ‖M‖_{\max} + ‖A - M‖_∞ + u ‖A‖_∞ + \gamma_6 ‖\hat{D}‖_∞
    ///   + n (2n + 4) \eta_{sub}`,
    ///
    /// where `\gamma_k = k u / (1 - k u) = k u + k^2 u^2 / (1 - k u)` (`u = 2^{-53}`)
    /// includes the exact second-order and higher-order `O(u^2)` remainder,
    /// `\eta_{sub} = f64::MIN_POSITIVE = 2^{-1022}` covers subnormal underflow,
    /// `\gamma_6 ‖\hat{D}‖_∞` adds a conservative block-diagonal margin, and
    /// `‖|\hat{L}||\hat{D}||\hat{L}|^T‖_∞` is computed from the standard-form `\hat{L}`.
    pub backward_error: f64,
    /// Higham (2002, Section 11.1.2, Eq. 11.5) Schur-complement pivot-growth factor
    /// `\rho_n = \max_{i,j,k} |a^{(k)}_{ij}| / \max_{i,j} |M_{ij}| >= 1`.
    pub pivot_growth: f64,
    /// Standard-form unit lower-triangular row-sum norm `‖\hat{L}‖_∞ = \max_i \sum_j |\hat{L}_{ij}|`.
    pub l_inf_norm: f64,
    /// Standard-form unit lower-triangular column-sum norm `‖\hat{L}‖_1 = \max_j \sum_i |\hat{L}_{ij}|`.
    pub l_one_norm: f64,
    /// Block-diagonal infinity norm `‖\hat{D}‖_∞`.
    pub d_inf_norm: f64,
}

fn swap_active(
    work: &mut [f64],
    l: &mut [f64],
    perm: &mut [usize],
    n: usize,
    k_done: usize,
    p: usize,
    q: usize,
) {
    if p == q {
        return;
    }
    for col in 0..n {
        work.swap(p * n + col, q * n + col);
    }
    for row in 0..n {
        work.swap(row * n + p, row * n + q);
    }
    for col in 0..k_done {
        l.swap(p * n + col, q * n + col);
    }
    perm.swap(p, q);
}

impl BunchKaufman {
    /// Factor an `n x n` row-major matrix `a` into standard-form
    /// `P M P^T = L D L^T` on `M = (A + A^T) / 2` using Bunch-Kaufman partial
    /// pivoting (`\alpha = (1 + \sqrt{17}) / 8`).
    pub fn factor(a: &[f64], n: usize) -> std::result::Result<Self, LdlError> {
        if n == 0 {
            return Err(LdlError::DimensionMismatch { len: a.len(), n });
        }
        let total = n.checked_mul(n).ok_or(LdlError::OrderOverflow { n })?;
        if a.len() != total {
            return Err(LdlError::DimensionMismatch { len: a.len(), n });
        }
        let max_gamma_idx = n
            .checked_mul(2)
            .and_then(|v| v.checked_add(4))
            .ok_or(LdlError::OrderOverflow { n })?;
        if gamma_up(max_gamma_idx).is_none() {
            return Err(LdlError::OrderOverflow { n });
        }

        let mut m = vec![0.0_f64; total];
        let mut u_a_inf = 0.0_f64;
        let mut asym_inf = 0.0_f64;
        let mut m_max = 0.0_f64;
        let u = 0.5 * f64::EPSILON;

        for r in 0..n {
            for c in 0..n {
                if !a[r * n + c].is_finite() {
                    return Err(LdlError::NonFiniteInput { row: r, col: c });
                }
            }
        }

        for r in 0..n {
            let mut u_a_row = 0.0_f64;
            let mut asym_row = 0.0_f64;
            for c in 0..n {
                let arc = a[r * n + c];
                let acr = a[c * n + r];
                let sum_ab = arc + acr;
                let sym = if sum_ab.is_finite() {
                    0.5 * sum_ab
                } else {
                    0.5 * arc + 0.5 * acr
                };
                let diff = if arc == sym {
                    0.0
                } else {
                    (arc - sym).abs().next_up()
                };
                if !sym.is_finite() || !diff.is_finite() {
                    return Err(LdlError::NonFiniteFactor {
                        stage: "symmetrization",
                    });
                }
                m[r * n + c] = sym;
                let abs_a = arc.abs();
                let abs_m = sym.abs();
                if abs_m > m_max {
                    m_max = abs_m;
                }
                u_a_row = add_up(u_a_row, mul_up(u, abs_a));
                asym_row = add_up(asym_row, diff);
                if !u_a_row.is_finite() || !asym_row.is_finite() {
                    return Err(LdlError::NonFiniteFactor {
                        stage: "input row norms",
                    });
                }
            }
            if u_a_row > u_a_inf {
                u_a_inf = u_a_row;
            }
            if asym_row > asym_inf {
                asym_inf = asym_row;
            }
        }

        let alpha = (1.0 + 17.0_f64.sqrt()) / 8.0;
        let mut perm: Vec<usize> = (0..n).collect();
        let mut l = vec![0.0_f64; total];
        for i in 0..n {
            l[i * n + i] = 1.0;
        }
        let mut work = m;
        let mut blocks = Vec::with_capacity(n);
        let mut max_schur = m_max;

        let mut k = 0usize;
        while k < n {
            if k + 1 == n {
                let d = work[k * n + k];
                if !d.is_finite() {
                    return Err(LdlError::NonFiniteFactor {
                        stage: "final 1x1 pivot",
                    });
                }
                let abs_d = d.abs();
                if abs_d > max_schur {
                    max_schur = abs_d;
                }
                blocks.push(PivotBlock::One { index: k, d });
                break;
            }

            let akk = work[k * n + k].abs();
            if !akk.is_finite() {
                return Err(LdlError::NonFiniteFactor {
                    stage: "diagonal pivot",
                });
            }
            if akk > max_schur {
                max_schur = akk;
            }

            let mut r = k + 1;
            let mut omega_1 = work[(k + 1) * n + k].abs();
            if !omega_1.is_finite() {
                return Err(LdlError::NonFiniteFactor {
                    stage: "column pivot search",
                });
            }
            for i in (k + 2)..n {
                let v = work[i * n + k].abs();
                if !v.is_finite() {
                    return Err(LdlError::NonFiniteFactor {
                        stage: "column pivot search",
                    });
                }
                if v > omega_1 {
                    omega_1 = v;
                    r = i;
                }
            }
            if omega_1 > max_schur {
                max_schur = omega_1;
            }

            if akk == 0.0 && omega_1 == 0.0 {
                blocks.push(PivotBlock::One { index: k, d: 0.0 });
                k += 1;
                continue;
            }

            let use_one = if akk >= alpha * omega_1 {
                true
            } else {
                let mut omega_2 = 0.0_f64;
                for j in k..n {
                    if j == r {
                        continue;
                    }
                    let v = if j > r {
                        work[j * n + r].abs()
                    } else {
                        work[r * n + j].abs()
                    };
                    if !v.is_finite() {
                        return Err(LdlError::NonFiniteFactor {
                            stage: "row pivot search",
                        });
                    }
                    if v > omega_2 {
                        omega_2 = v;
                    }
                }
                if omega_2 > max_schur {
                    max_schur = omega_2;
                }
                let arr = work[r * n + r].abs();
                if !arr.is_finite() {
                    return Err(LdlError::NonFiniteFactor {
                        stage: "candidate diagonal pivot",
                    });
                }
                if arr > max_schur {
                    max_schur = arr;
                }
                if akk > 0.0 && omega_2 > 0.0 && (akk / omega_1) >= alpha * (omega_1 / omega_2) {
                    true
                } else if arr >= alpha * omega_2 {
                    swap_active(&mut work, &mut l, &mut perm, n, k, k, r);
                    true
                } else {
                    swap_active(&mut work, &mut l, &mut perm, n, k, k + 1, r);
                    false
                }
            };

            if use_one {
                let d = work[k * n + k];
                if !d.is_finite() || d == 0.0 {
                    return Err(LdlError::NonFiniteFactor { stage: "1x1 pivot" });
                }
                let abs_d = d.abs();
                if abs_d > max_schur {
                    max_schur = abs_d;
                }
                for i in (k + 1)..n {
                    let lik = work[i * n + k] / d;
                    if !lik.is_finite() {
                        return Err(LdlError::NonFiniteFactor {
                            stage: "1x1 multiplier",
                        });
                    }
                    l[i * n + k] = lik;
                }
                for i in (k + 1)..n {
                    let lik = l[i * n + k];
                    if lik == 0.0 {
                        continue;
                    }
                    for j in (k + 1)..=i {
                        let updated = work[i * n + j] - lik * work[j * n + k];
                        if !updated.is_finite() {
                            return Err(LdlError::NonFiniteFactor {
                                stage: "1x1 schur update",
                            });
                        }
                        work[i * n + j] = updated;
                        work[j * n + i] = updated;
                        let abs_u = updated.abs();
                        if abs_u > max_schur {
                            max_schur = abs_u;
                        }
                    }
                }
                blocks.push(PivotBlock::One { index: k, d });
                k += 1;
            } else {
                let d00 = work[k * n + k];
                let d01 = work[(k + 1) * n + k];
                let d11 = work[(k + 1) * n + (k + 1)];
                if !d00.is_finite() || !d01.is_finite() || !d11.is_finite() || d01 == 0.0 {
                    return Err(LdlError::NonFiniteFactor { stage: "2x2 pivot" });
                }
                if k + 2 < n {
                    let mut max_d = d00.abs();
                    if d01.abs() > max_d {
                        max_d = d01.abs();
                    }
                    if d11.abs() > max_d {
                        max_d = d11.abs();
                    }
                    let (_, (inv_s1, inv_s2)) =
                        frexp_pow2_pair(max_d).ok_or(LdlError::NonFiniteFactor {
                            stage: "2x2 pivot scale",
                        })?;
                    let d00_s = (d00 * inv_s1) * inv_s2;
                    let d01_s = (d01 * inv_s1) * inv_s2;
                    let d11_s = (d11 * inv_s1) * inv_s2;
                    let d00_over_d01 = d00_s / d01_s;
                    let d11_over_d01 = d11_s / d01_s;
                    let denom_s = d00_over_d01 * d11_s - d01_s;
                    if !denom_s.is_finite() || denom_s == 0.0 {
                        return Err(LdlError::NonFiniteFactor {
                            stage: "2x2 pivot denominator",
                        });
                    }
                    for i in (k + 2)..n {
                        let aik_s = (work[i * n + k] * inv_s1) * inv_s2;
                        let aik1_s = (work[i * n + (k + 1)] * inv_s1) * inv_s2;
                        let lik = (d11_over_d01 * aik_s - aik1_s) / denom_s;
                        let lik1 = (d00_over_d01 * aik1_s - aik_s) / denom_s;
                        if !lik.is_finite() || !lik1.is_finite() {
                            return Err(LdlError::NonFiniteFactor {
                                stage: "2x2 multiplier",
                            });
                        }
                        l[i * n + k] = lik;
                        l[i * n + (k + 1)] = lik1;
                    }
                    for i in (k + 2)..n {
                        let lik = l[i * n + k];
                        let lik1 = l[i * n + (k + 1)];
                        if lik == 0.0 && lik1 == 0.0 {
                            continue;
                        }
                        for j in (k + 2)..=i {
                            let ajk = work[j * n + k];
                            let ajk1 = work[j * n + (k + 1)];
                            let updated = work[i * n + j] - lik * ajk - lik1 * ajk1;
                            if !updated.is_finite() {
                                return Err(LdlError::NonFiniteFactor {
                                    stage: "2x2 schur update",
                                });
                            }
                            work[i * n + j] = updated;
                            work[j * n + i] = updated;
                            let abs_u = updated.abs();
                            if abs_u > max_schur {
                                max_schur = abs_u;
                            }
                        }
                    }
                }
                blocks.push(PivotBlock::Two {
                    index: k,
                    d00,
                    d01,
                    d11,
                });
                k += 2;
            }
        }

        Ok(Self {
            n,
            perm,
            l,
            blocks,
            u_a_inf,
            asym_inf,
            m_max,
            max_schur,
        })
    }

    /// Matrix order `n`.
    pub fn order(&self) -> usize {
        self.n
    }

    /// Row permutation `perm` such that `(P M P^T)_{ij} = M_{perm[i], perm[j]}`.
    pub fn permutation(&self) -> &[usize] {
        &self.perm
    }

    /// Row-major `n x n` standard-form unit lower-triangular factor `L`.
    pub fn l_matrix(&self) -> &[f64] {
        &self.l
    }

    /// Compute the pivot inertia counts of `D`, standard-form factor norms,
    /// Higham (2002, Eq. 11.5) Schur-complement growth factor `\rho_n`, and
    /// explicit-constant backward-error bound.
    ///
    /// Note that unshifted `LDL^T` computes the exact inertia of a perturbed matrix
    /// `M + P^T \Delta M P` with `‖\Delta M‖_∞ <= backward_error`; for eigenvalues of
    /// `A` inside `[-\Delta_{ldl}, +\Delta_{ldl}]`, unshifted `negative` / `positive`
    /// counts are diagnostic even when `ambiguous_pivots == 0`. Use
    /// [`certify_shifted_ldlt_inertia`] to certify the inertia of `A` itself.
    pub fn inertia_with_backward_bound(&self) -> std::result::Result<LdlInertiaBound, LdlError> {
        let n = self.n;
        let gamma_6 = gamma_up(6).ok_or(LdlError::OrderOverflow { n })?;
        let gamma_2n = gamma_up(n.checked_mul(2).ok_or(LdlError::OrderOverflow { n })?)
            .ok_or(LdlError::OrderOverflow { n })?;
        let two_n_plus_4 = n
            .checked_mul(2)
            .and_then(|v| v.checked_add(4))
            .ok_or(LdlError::OrderOverflow { n })?;
        let gamma_2n_plus_4 = gamma_up(two_n_plus_4).ok_or(LdlError::OrderOverflow { n })?;

        let mut l_inf_norm = 1.0_f64;
        for r in 0..n {
            let mut row_sum = 0.0_f64;
            for c in 0..=r {
                let v = self.l[r * n + c].abs();
                if !v.is_finite() {
                    return Err(LdlError::NonFiniteFactor { stage: "L entry" });
                }
                row_sum = add_up(row_sum, v);
            }
            if !row_sum.is_finite() {
                return Err(LdlError::NonFiniteFactor {
                    stage: "L inf norm",
                });
            }
            if row_sum > l_inf_norm {
                l_inf_norm = row_sum;
            }
        }

        let mut l_col_sums = vec![0.0_f64; n];
        let mut l_one_norm = 1.0_f64;
        for c in 0..n {
            let mut col_sum = 0.0_f64;
            for r in c..n {
                let v = self.l[r * n + c].abs();
                col_sum = add_up(col_sum, v);
            }
            if !col_sum.is_finite() {
                return Err(LdlError::NonFiniteFactor {
                    stage: "L one norm",
                });
            }
            l_col_sums[c] = col_sum;
            if col_sum > l_one_norm {
                l_one_norm = col_sum;
            }
        }

        let mut d_inf_norm = 0.0_f64;
        for block in &self.blocks {
            match *block {
                PivotBlock::One { d, .. } => {
                    let v = d.abs();
                    if !v.is_finite() {
                        return Err(LdlError::NonFiniteFactor {
                            stage: "D 1x1 norm",
                        });
                    }
                    if v > d_inf_norm {
                        d_inf_norm = v;
                    }
                }
                PivotBlock::Two { d00, d01, d11, .. } => {
                    let r0 = add_up(d00.abs(), d01.abs());
                    let r1 = add_up(d01.abs(), d11.abs());
                    if !r0.is_finite() || !r1.is_finite() {
                        return Err(LdlError::NonFiniteFactor {
                            stage: "D 2x2 norm",
                        });
                    }
                    if r0 > d_inf_norm {
                        d_inf_norm = r0;
                    }
                    if r1 > d_inf_norm {
                        d_inf_norm = r1;
                    }
                }
            }
        }

        // Exact scaled row-sum norm γ_{2n+4} ‖|L̂||D̂||L̂|ᵀ‖_∞ in O(n²) via associativity,
        // pre-multiplying γ_{2n+4} into |D̂| so entries near f64::MAX do not overflow
        // before being scaled by γ_{2n+4}.
        let mut factor_term = 0.0_f64;
        for r in 0..n {
            let mut row_acc = 0.0_f64;
            for block in &self.blocks {
                match *block {
                    PivotBlock::One { index, d } => {
                        if index <= r {
                            let gd = mul_up(gamma_2n_plus_4, d.abs());
                            let w_abs = mul_up(self.l[r * n + index].abs(), gd);
                            let contrib = mul_up(w_abs, l_col_sums[index]);
                            row_acc = add_up(row_acc, contrib);
                        }
                    }
                    PivotBlock::Two {
                        index,
                        d00,
                        d01,
                        d11,
                    } => {
                        if index <= r {
                            let l0 = self.l[r * n + index].abs();
                            let l1 = if index < r {
                                self.l[r * n + index + 1].abs()
                            } else {
                                0.0
                            };
                            let gd00 = mul_up(gamma_2n_plus_4, d00.abs());
                            let gd01 = mul_up(gamma_2n_plus_4, d01.abs());
                            let gd11 = mul_up(gamma_2n_plus_4, d11.abs());
                            let w0 = add_up(mul_up(l0, gd00), mul_up(l1, gd01));
                            let w1 = add_up(mul_up(l0, gd01), mul_up(l1, gd11));
                            let c0 = mul_up(w0, l_col_sums[index]);
                            let c1 = mul_up(w1, l_col_sums[index + 1]);
                            row_acc = add_up(row_acc, add_up(c0, c1));
                        }
                    }
                }
            }
            if !row_acc.is_finite() {
                return Err(LdlError::NonFiniteFactor {
                    stage: "|L||D||L|^T norm",
                });
            }
            if row_acc > factor_term {
                factor_term = row_acc;
            }
        }

        let mut negative = 0usize;
        let mut positive = 0usize;
        let mut ambiguous_pivots = 0usize;

        for block in &self.blocks {
            match *block {
                PivotBlock::One { d, .. } => {
                    if !d.is_finite() {
                        return Err(LdlError::NonFiniteFactor {
                            stage: "1x1 eigenvalue",
                        });
                    }
                    if d < 0.0 {
                        negative += 1;
                    } else if d > 0.0 {
                        positive += 1;
                    } else {
                        ambiguous_pivots += 1;
                    }
                }
                PivotBlock::Two { d00, d01, d11, .. } => {
                    let intervals = two_by_two_eigenvalue_intervals(d00, d01, d11).ok_or(
                        LdlError::NonFiniteFactor {
                            stage: "2x2 eigenvalue",
                        },
                    )?;
                    for (lo, hi) in intervals {
                        if hi < 0.0 {
                            negative += 1;
                        } else if lo > 0.0 {
                            positive += 1;
                        } else {
                            ambiguous_pivots += 1;
                        }
                    }
                }
            }
        }

        let pivot_growth = if self.m_max > 0.0 && self.max_schur != self.m_max {
            let ratio = (self.max_schur / self.m_max).next_up();
            if !ratio.is_finite() {
                return Err(LdlError::NonFiniteFactor {
                    stage: "pivot growth ratio",
                });
            }
            ratio
        } else {
            1.0
        };

        // Explicit-constant instantiation patterned after Higham (2002) Theorem 11.3
        // (which leaves the linear polynomial p(n) and 2x2 pivot constant c unspecified):
        //   ‖P M Pᵀ - L̂ D̂ L̂ᵀ‖_∞ <= γ_{2n+4} ‖|L̂||D̂||L̂|ᵀ‖_∞ + n γ_{2n} ρ_n ‖M‖_max + γ_6 ‖D̂‖_∞ + n (2n + 4) η_sub
        // plus input symmetrization ‖A - M‖_∞ + u ‖A‖_∞:
        let schur_coeff = mul_up(mul_up(n as f64, gamma_2n), pivot_growth);
        let schur_term = mul_up(schur_coeff, self.m_max);
        let sym_term = add_up(self.asym_inf, self.u_a_inf);
        let d_term = mul_up(gamma_6, d_inf_norm);
        let underflow_ops = n
            .checked_mul(two_n_plus_4)
            .ok_or(LdlError::OrderOverflow { n })?;
        let underflow_term = mul_up(underflow_ops as f64, f64::MIN_POSITIVE);
        let backward_error = add_up(
            add_up(add_up(add_up(factor_term, schur_term), sym_term), d_term),
            underflow_term,
        );

        if !backward_error.is_finite() || backward_error < 0.0 {
            return Err(LdlError::NonFiniteFactor {
                stage: "backward error bound",
            });
        }

        Ok(LdlInertiaBound {
            negative,
            positive,
            ambiguous_pivots,
            backward_error,
            pivot_growth,
            l_inf_norm,
            l_one_norm,
            d_inf_norm,
        })
    }
}

/// Factor an `n x n` symmetric matrix with standard-form Bunch-Kaufman `LDL^T`
/// and return its pivot inertia counts paired with an explicit-constant backward-error bound.
///
/// The returned `negative` / `positive` counts reflect the exact inertia of a perturbed
/// matrix `M + P^T \Delta M P` (`‖\Delta M‖_∞ <= backward_error`) and are therefore
/// diagnostic for eigenvalues of `a` inside `[-\Delta_{ldl}, +\Delta_{ldl}]`. Use
/// [`certify_shifted_ldlt_inertia`] to certify the inertia of `a` across its backward-error radius.
pub fn ldl_inertia_with_backward_bound(
    a: &[f64],
    n: usize,
) -> std::result::Result<LdlInertiaBound, LdlError> {
    BunchKaufman::factor(a, n)?.inertia_with_backward_bound()
}

/// Certified two-shift Loewner-sandwich inertia result on a symmetric matrix
/// whose exact target `M_exact` is within `perturbation_bound` in spectral norm.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShiftedLdlCertificate {
    /// Certified negative eigenvalue count `n_-(M_exact)` when `certified_negative + certified_positive == dimension`.
    pub negative: usize,
    /// Certified positive eigenvalue count `n_+(M_exact)`.
    pub positive: usize,
    /// Matrix dimension `n`.
    pub dimension: usize,
    /// Caller-supplied spectral perturbation bound `‖M̃ - M_exact‖_2 <= perturbation_bound`.
    pub perturbation_bound: f64,
    /// Maximum `LDL^T` backward-error bound across the shifted factorizations.
    pub ldl_backward_error: f64,
    /// Effective diagonal shift applied in `M̃ ± shift_applied I`.
    pub shift_applied: f64,
    /// Maximum Schur-complement pivot-growth factor across the factorizations.
    pub pivot_growth: f64,
}

/// Structured refusal reasons for [`certify_shifted_ldlt_inertia`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, thiserror::Error)]
pub enum ShiftedLdlRefusal {
    /// Underlying `LDL^T` factorization or bound failed.
    #[error("shifted LDL^T factorization failed: {0}")]
    Factor(#[from] LdlError),
    /// Non-finite or negative `operator_scale` or `perturbation_bound`.
    #[error(
        "shifted LDL^T invalid bound input: operator_scale = {operator_scale}, perturbation_bound = {perturbation_bound}"
    )]
    InvalidBound {
        /// Supplied operator scale.
        operator_scale: f64,
        /// Supplied perturbation bound.
        perturbation_bound: f64,
    },
    /// At least one pivot eigenvalue enclosure of `D_-` or `D_+` contained `0`.
    #[error(
        "shifted LDL^T encountered {ambiguous_pivots} ambiguous pivot(s) at shift {shift_applied:.3e}"
    )]
    AmbiguousPivot {
        /// Total count of ambiguous pivots across `M̃ - ΔI` and `M̃ + ΔI`.
        ambiguous_pivots: usize,
        /// Applied diagonal shift.
        shift_applied: f64,
        /// `LDL^T` backward-error bound.
        ldl_backward_error: f64,
        /// Maximum Schur-complement growth factor.
        pivot_growth: f64,
    },
    /// Shifted factorizations violated the Loewner sandwich `n - n_+(M̃ - ΔI) < n_-(M̃ + ΔI)`.
    #[error(
        "shifted LDL^T inconsistent Loewner order: certified_negative = {certified_negative}, certified_positive = {certified_positive}, dimension = {dimension}"
    )]
    InconsistentLoewnerOrder {
        /// Lower bound on `n_-(M_exact)` from `M̃ + ΔI`.
        certified_negative: usize,
        /// Lower bound on `n_+(M_exact)` from `M̃ - ΔI`.
        certified_positive: usize,
        /// Matrix order `n`.
        dimension: usize,
        /// Applied diagonal shift.
        shift_applied: f64,
        /// `LDL^T` backward-error bound.
        ldl_backward_error: f64,
        /// Maximum Schur-complement growth factor.
        pivot_growth: f64,
    },
    /// One or more eigenvalues lie inside the uncertainty interval `[-\Delta, +\Delta]`.
    #[error(
        "shifted LDL^T unresolved inertia: certified_negative = {certified_negative}, certified_positive = {certified_positive}, unresolved = {unresolved} / {dimension} at shift {shift_applied:.3e}"
    )]
    UnresolvedInertia {
        /// Rigorous lower bound on `n_-(M_exact)`.
        certified_negative: usize,
        /// Rigorous lower bound on `n_+(M_exact)`.
        certified_positive: usize,
        /// Number of eigenvalues whose sign cannot be resolved (`dimension - certified_positive - certified_negative >= 1`).
        unresolved: usize,
        /// Matrix order `n`.
        dimension: usize,
        /// Caller-supplied perturbation bound.
        perturbation_bound: f64,
        /// `LDL^T` backward-error bound.
        ldl_backward_error: f64,
        /// Applied diagonal shift.
        shift_applied: f64,
        /// Maximum Schur-complement growth factor.
        pivot_growth: f64,
    },
}

/// Certify the exact inertia of `M_exact` from an approximate matrix `matrix` (`M̃`)
/// satisfying `‖M̃ - M_exact‖_2 <= perturbation_bound` using two shifted standard-form
/// Bunch-Kaufman `LDL^T` factorizations `M̃ - ΔI` and `M̃ + ΔI`.
///
/// By Weyl's theorem and Ostrowski's theorem, whenever
/// `Δ >= perturbation_bound + max(Δ_{ldl,-}, Δ_{ldl,+}) + δ_{diag}`,
/// the exact matrix is bracketed in the Loewner order between the exact reconstructions
/// of `M̃ - ΔI` and `M̃ + ΔI`:
/// - `n_-(M_exact) >= n_-(D_+)` (`certified_negative`),
/// - `n_+(M_exact) >= n_+(D_-)` (`certified_positive`).
///
/// When `certified_negative + certified_positive == n` and `ambiguous_pivots == 0`,
/// `n_-(M_exact) == certified_negative` is certified; otherwise a structured
/// [`ShiftedLdlRefusal`] is returned.
pub fn certify_shifted_ldlt_inertia(
    matrix: &[f64],
    n: usize,
    operator_scale: f64,
    perturbation_bound: f64,
) -> std::result::Result<ShiftedLdlCertificate, ShiftedLdlRefusal> {
    if !operator_scale.is_finite()
        || operator_scale < 0.0
        || !perturbation_bound.is_finite()
        || perturbation_bound < 0.0
    {
        return Err(ShiftedLdlRefusal::InvalidBound {
            operator_scale,
            perturbation_bound,
        });
    }
    if n == 0 {
        return Err(ShiftedLdlRefusal::Factor(LdlError::DimensionMismatch {
            len: matrix.len(),
            n,
        }));
    }
    let total = n
        .checked_mul(n)
        .ok_or(ShiftedLdlRefusal::Factor(LdlError::OrderOverflow { n }))?;
    if matrix.len() != total {
        return Err(ShiftedLdlRefusal::Factor(LdlError::DimensionMismatch {
            len: matrix.len(),
            n,
        }));
    }

    let mut max_entry = operator_scale;
    for r in 0..n {
        for c in 0..n {
            let v = matrix[r * n + c].abs();
            if !v.is_finite() {
                return Err(ShiftedLdlRefusal::Factor(LdlError::NonFiniteInput {
                    row: r,
                    col: c,
                }));
            }
            if v > max_entry {
                max_entry = v;
            }
        }
    }

    // By Sylvester's law of inertia, scaling A by a positive power of two s = s1 * s2 > 0
    // preserves the exact inertia of A while keeping extreme scales (`±f64::MAX` or
    // minimum subnormals `f64::from_bits(1)`) comfortably inside the normal range so
    // `M̃ ± ΔI` never overflows `f64::MAX` or swamps subnormal spectra in gradual underflow.
    let two_pow_500 = f64::from_bits(0x5f30_0000_0000_0000);
    let two_pow_neg_450 = f64::from_bits(0x23d0_0000_0000_0000);
    let ((s1, s2), (inv_s1, inv_s2)) = if max_entry > two_pow_500
        || (max_entry > 0.0 && max_entry < two_pow_neg_450)
    {
        frexp_pow2_pair(max_entry).ok_or(ShiftedLdlRefusal::Factor(LdlError::NonFiniteFactor {
            stage: "matrix scale",
        }))?
    } else {
        ((1.0, 1.0), (1.0, 1.0))
    };

    let (work_matrix, work_scale, work_pert) = if inv_s1 == 1.0 && inv_s2 == 1.0 {
        (matrix.to_vec(), operator_scale, perturbation_bound)
    } else {
        let mut scaled = Vec::with_capacity(total);
        let mut scale_underflow_rows = 0usize;
        for r in 0..n {
            let mut row_underflow = false;
            for c in 0..n {
                let x = matrix[r * n + c];
                let sx = (x * inv_s1) * inv_s2;
                if x != 0.0 && sx.abs() < f64::MIN_POSITIVE {
                    row_underflow = true;
                }
                scaled.push(sx);
            }
            if row_underflow {
                scale_underflow_rows += 1;
            }
        }
        let sc_op = (operator_scale * inv_s1) * inv_s2;
        let sc_pert_raw = ((perturbation_bound * inv_s1).next_up() * inv_s2).next_up();
        let sc_pert = if scale_underflow_rows > 0 {
            add_up(sc_pert_raw, mul_up(n as f64, f64::MIN_POSITIVE))
        } else {
            sc_pert_raw
        };
        if !sc_op.is_finite() || !sc_pert.is_finite() {
            return Err(ShiftedLdlRefusal::Factor(LdlError::NonFiniteFactor {
                stage: "scaled perturbation bound",
            }));
        }
        (scaled, sc_op, sc_pert)
    };

    let ldl_0 = ldl_inertia_with_backward_bound(&work_matrix, n)?;
    let u = 0.5 * f64::EPSILON;

    let mut max_diag = work_scale;
    for i in 0..n {
        let d = work_matrix[i * n + i].abs();
        if d > max_diag {
            max_diag = d;
        }
    }

    let eval_pair = |shift: f64| -> std::result::Result<
        (LdlInertiaBound, LdlInertiaBound, f64, f64),
        ShiftedLdlRefusal,
    > {
        if !shift.is_finite() {
            return Err(ShiftedLdlRefusal::Factor(LdlError::NonFiniteFactor {
                stage: "effective shift",
            }));
        }
        if shift < 0.0 {
            return Err(ShiftedLdlRefusal::InvalidBound {
                operator_scale,
                perturbation_bound: shift,
            });
        }
        let sum_ds = (max_diag + shift).next_up();
        let diag_roundoff = if sum_ds.is_finite() {
            (u * sum_ds).next_up()
        } else {
            ((u * max_diag).next_up() + (u * shift).next_up()).next_up()
        };
        if !diag_roundoff.is_finite() {
            return Err(ShiftedLdlRefusal::Factor(LdlError::NonFiniteFactor {
                stage: "diagonal shift roundoff",
            }));
        }
        let shift_eff = (shift + diag_roundoff).next_up();
        if !shift_eff.is_finite() {
            return Err(ShiftedLdlRefusal::Factor(LdlError::NonFiniteFactor {
                stage: "effective shift",
            }));
        }
        let mut m_minus = work_matrix.clone();
        let mut m_plus = work_matrix.clone();
        for i in 0..n {
            let d = work_matrix[i * n + i];
            let lo = (d - shift_eff).next_down();
            let hi = (d + shift_eff).next_up();
            if !lo.is_finite() || !hi.is_finite() {
                return Err(ShiftedLdlRefusal::Factor(LdlError::NonFiniteFactor {
                    stage: "shifted diagonal",
                }));
            }
            m_minus[i * n + i] = lo;
            m_plus[i * n + i] = hi;
        }
        let ldl_minus = ldl_inertia_with_backward_bound(&m_minus, n)?;
        let ldl_plus = ldl_inertia_with_backward_bound(&m_plus, n)?;
        Ok((ldl_minus, ldl_plus, diag_roundoff, shift_eff))
    };

    let mut shift = (work_pert + (2.0 * ldl_0.backward_error).next_up()).next_up();
    let (mut ldl_minus, mut ldl_plus, mut diag_roundoff, mut shift_eff) = eval_pair(shift)?;

    let max_three = |a: f64, b: f64, c: f64| -> f64 {
        let mut acc = a;
        if b > acc {
            acc = b;
        }
        if c > acc {
            acc = c;
        }
        acc
    };

    let mut ldl_backward_error = max_three(
        ldl_0.backward_error,
        ldl_minus.backward_error,
        ldl_plus.backward_error,
    );
    let mut pivot_growth = max_three(
        ldl_0.pivot_growth,
        ldl_minus.pivot_growth,
        ldl_plus.pivot_growth,
    );
    let required_shift = ((work_pert + ldl_backward_error).next_up() + diag_roundoff).next_up();

    if shift < required_shift {
        shift = ((work_pert + (2.0 * ldl_backward_error).next_up()).next_up()
            + (2.0 * diag_roundoff).next_up())
        .next_up();
        let retry = eval_pair(shift)?;
        ldl_minus = retry.0;
        ldl_plus = retry.1;
        diag_roundoff = retry.2;
        shift_eff = retry.3;
        ldl_backward_error = max_three(
            ldl_backward_error,
            ldl_minus.backward_error,
            ldl_plus.backward_error,
        );
        pivot_growth = max_three(pivot_growth, ldl_minus.pivot_growth, ldl_plus.pivot_growth);
        let recheck = ((work_pert + ldl_backward_error).next_up() + diag_roundoff).next_up();
        if shift < recheck {
            return Err(ShiftedLdlRefusal::Factor(LdlError::NonFiniteFactor {
                stage: "shift dominance verification",
            }));
        }
    }

    let unscale_bound =
        |x: f64, stage: &'static str| -> std::result::Result<f64, ShiftedLdlRefusal> {
            let v = if s1 == 1.0 && s2 == 1.0 {
                x
            } else {
                ((x * s2).next_up() * s1).next_up()
            };
            if v.is_finite() && v >= 0.0 {
                Ok(v)
            } else {
                Err(ShiftedLdlRefusal::Factor(LdlError::NonFiniteFactor {
                    stage,
                }))
            }
        };
    let reported_shift = unscale_bound(shift_eff, "unscaled shift bound")?;
    let reported_ldl_err = unscale_bound(ldl_backward_error, "unscaled backward error bound")?;

    let ambiguous_pivots = ldl_minus.ambiguous_pivots + ldl_plus.ambiguous_pivots;
    classify_shifted_inertia(
        (n, ldl_plus.negative, ldl_minus.positive, ambiguous_pivots),
        (
            perturbation_bound,
            reported_ldl_err,
            reported_shift,
            pivot_growth,
        ),
    )
}

fn classify_shifted_inertia(
    (n, certified_negative, certified_positive, ambiguous_pivots): (usize, usize, usize, usize),
    (perturbation_bound, reported_ldl_err, reported_shift, pivot_growth): (f64, f64, f64, f64),
) -> std::result::Result<ShiftedLdlCertificate, ShiftedLdlRefusal> {
    if ambiguous_pivots > 0 {
        return Err(ShiftedLdlRefusal::AmbiguousPivot {
            ambiguous_pivots,
            shift_applied: reported_shift,
            ldl_backward_error: reported_ldl_err,
            pivot_growth,
        });
    }
    if certified_positive > n {
        return Err(ShiftedLdlRefusal::InconsistentLoewnerOrder {
            certified_negative,
            certified_positive,
            dimension: n,
            shift_applied: reported_shift,
            ldl_backward_error: reported_ldl_err,
            pivot_growth,
        });
    }
    let below_plus = n - certified_positive;
    if below_plus < certified_negative {
        return Err(ShiftedLdlRefusal::InconsistentLoewnerOrder {
            certified_negative,
            certified_positive,
            dimension: n,
            shift_applied: reported_shift,
            ldl_backward_error: reported_ldl_err,
            pivot_growth,
        });
    }
    if below_plus > certified_negative {
        let unresolved = below_plus - certified_negative;
        return Err(ShiftedLdlRefusal::UnresolvedInertia {
            certified_negative,
            certified_positive,
            unresolved,
            dimension: n,
            perturbation_bound,
            ldl_backward_error: reported_ldl_err,
            shift_applied: reported_shift,
            pivot_growth,
        });
    }

    Ok(ShiftedLdlCertificate {
        negative: certified_negative,
        positive: certified_positive,
        dimension: n,
        perturbation_bound,
        ldl_backward_error: reported_ldl_err,
        shift_applied: reported_shift,
        pivot_growth,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn ldlt_recovers_identity() {
        let a = Array2::<f64>::eye(3);
        let b = array![1.0, 2.0, 3.0];
        let x = ldlt_solve(&a, &b, false).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-12);
        assert!((x[1] - 2.0).abs() < 1e-12);
        assert!((x[2] - 3.0).abs() < 1e-12);
    }

    struct XorShift64(u64);
    impl XorShift64 {
        fn next_f64(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        }
    }

    fn reconstruct_pmp_minus_ldlt_inf(bk: &BunchKaufman, a: &[f64]) -> (f64, f64) {
        let n = bk.order();
        let perm = bk.permutation();
        let l = bk.l_matrix();
        let mut dmat = vec![0.0_f64; n * n];
        for block in &bk.blocks {
            match *block {
                PivotBlock::One { index, d } => {
                    dmat[index * n + index] = d;
                }
                PivotBlock::Two {
                    index,
                    d00,
                    d01,
                    d11,
                } => {
                    dmat[index * n + index] = d00;
                    dmat[index * n + index + 1] = d01;
                    dmat[(index + 1) * n + index] = d01;
                    dmat[(index + 1) * n + index + 1] = d11;
                }
            }
        }
        let mut ld = vec![0.0_f64; n * n];
        for r in 0..n {
            for c in 0..n {
                let mut acc = 0.0_f64;
                for t in 0..n {
                    acc += l[r * n + t] * dmat[t * n + c];
                }
                ld[r * n + c] = acc;
            }
        }
        let mut ldlt = vec![0.0_f64; n * n];
        for r in 0..n {
            for c in 0..n {
                let mut acc = 0.0_f64;
                for t in 0..n {
                    acc += ld[r * n + t] * l[c * n + t];
                }
                ldlt[r * n + c] = acc;
            }
        }
        let mut err_inf = 0.0_f64;
        for r in 0..n {
            let mut row_err = 0.0_f64;
            for c in 0..n {
                let p_r = perm[r];
                let p_c = perm[c];
                let a_perm = a[p_r * n + p_c];
                row_err += (a_perm - ldlt[r * n + c]).abs();
            }
            if row_err > err_inf {
                err_inf = row_err;
            }
        }
        let mut true_l_inf = 0.0_f64;
        for r in 0..n {
            let mut row_sum = 0.0_f64;
            for c in 0..n {
                row_sum += l[r * n + c].abs();
            }
            if row_sum > true_l_inf {
                true_l_inf = row_sum;
            }
        }
        (err_inf, true_l_inf)
    }

    #[test]
    fn bunch_kaufman_standard_form_l_norm_and_backward_bound_hold_across_adversarial_matrices() {
        let mut rng = XorShift64(0x1234_5678_9abc_def1);
        for trial in 0..400 {
            let n = 2 + (trial % 32);
            let mut a = vec![0.0_f64; n * n];
            let kind = trial % 4;
            for r in 0..n {
                for c in 0..=r {
                    let mut v = rng.next_f64();
                    if kind == 1 {
                        v *= 10.0_f64.powf(5.0 * rng.next_f64());
                    } else if kind == 2 && r == c {
                        v *= 1.0e-10;
                    } else if kind == 3 {
                        v = if r == c {
                            1.0e-8 * rng.next_f64()
                        } else {
                            rng.next_f64() * ((r + c) % 3) as f64
                        };
                    }
                    a[r * n + c] = v;
                    a[c * n + r] = v;
                }
            }
            let bk = BunchKaufman::factor(&a, n).expect("factor");
            let bound = bk.inertia_with_backward_bound().expect("bound");
            let (actual_err_inf, true_l_inf) = reconstruct_pmp_minus_ldlt_inf(&bk, &a);
            assert!(
                bound.l_inf_norm >= true_l_inf * (1.0 - 1.0e-15),
                "trial {trial}: standard-form ||L||_inf {} < true {}",
                bound.l_inf_norm,
                true_l_inf
            );
            assert!(
                actual_err_inf <= bound.backward_error,
                "trial {trial}: actual ||P A P^T - L D L^T||_inf {actual_err_inf:e} > bound {:e}",
                bound.backward_error
            );
            assert!(bound.pivot_growth >= 1.0 && bound.pivot_growth.is_finite());
        }
    }

    #[test]
    fn bunch_kaufman_rejects_non_finite_inputs_and_resolves_2x2_pivots() {
        let anti = [0.0, 2.0, 2.0, 0.0];
        let bound = ldl_inertia_with_backward_bound(&anti, 2).expect("2x2 anti-diagonal");
        assert_eq!(bound.negative, 1);
        assert_eq!(bound.positive, 1);
        assert_eq!(bound.ambiguous_pivots, 0);

        for scale_exp in [0x5f50_0000_0000_0000_u64, 0x20b0_0000_0000_0000_u64] {
            let s = f64::from_bits(scale_exp);
            let scaled_anti = [0.0, s, s, 0.0];
            let s_bound = ldl_inertia_with_backward_bound(&scaled_anti, 2).expect("scaled 2x2");
            assert_eq!(s_bound.negative, 1);
            assert_eq!(s_bound.positive, 1);
            assert_eq!(s_bound.ambiguous_pivots, 0);
        }

        let max_one = [f64::MAX];
        let max_bound = ldl_inertia_with_backward_bound(&max_one, 1).expect("f64::MAX 1x1");
        assert_eq!(max_bound.negative, 0);
        assert_eq!(max_bound.positive, 1);
        assert_eq!(max_bound.ambiguous_pivots, 0);

        let max_shifted =
            certify_shifted_ldlt_inertia(&max_one, 1, f64::MAX, 0.0).expect("shifted f64::MAX 1x1");
        assert_eq!(max_shifted.negative, 0);
        assert_eq!(max_shifted.positive, 1);

        let neg_max_one = [-f64::MAX];
        let neg_max_shifted = certify_shifted_ldlt_inertia(&neg_max_one, 1, f64::MAX, 0.0)
            .expect("shifted -f64::MAX 1x1");
        assert_eq!(neg_max_shifted.negative, 1);
        assert_eq!(neg_max_shifted.positive, 0);

        let anti_max = [0.0, f64::MAX, f64::MAX, 0.0];
        let anti_max_intervals =
            two_by_two_eigenvalue_intervals(0.0, f64::MAX, 0.0).expect("anti_max 2x2 intervals");
        assert!(anti_max_intervals[0].0 <= -f64::MAX && anti_max_intervals[0].1 <= -f64::MAX);
        assert!(anti_max_intervals[1].0 >= f64::MAX && anti_max_intervals[1].1 >= f64::MAX);
        let anti_max_bound =
            ldl_inertia_with_backward_bound(&anti_max, 2).expect("anti_max 2x2 unshifted");
        assert_eq!(anti_max_bound.negative, 1);
        assert_eq!(anti_max_bound.positive, 1);
        assert_eq!(anti_max_bound.ambiguous_pivots, 0);
        let anti_max_cert = certify_shifted_ldlt_inertia(&anti_max, 2, f64::MAX, 0.0)
            .expect("anti_max 2x2 shifted");
        assert_eq!(anti_max_cert.negative, 1);
        assert_eq!(anti_max_cert.positive, 1);

        let gen_max_ok = [
            0.25 * f64::MAX,
            0.5 * f64::MAX,
            0.5 * f64::MAX,
            -0.25 * f64::MAX,
        ];
        let gen_max_bound = ldl_inertia_with_backward_bound(&gen_max_ok, 2)
            .expect("representable MAX 2x2 unshifted");
        assert_eq!(gen_max_bound.negative, 1);
        assert_eq!(gen_max_bound.positive, 1);
        assert_eq!(gen_max_bound.ambiguous_pivots, 0);
        assert!(gen_max_bound.d_inf_norm >= 0.75 * f64::MAX);
        assert!(
            gen_max_bound.backward_error
                >= mul_up(gamma_up(6).expect("gamma_6"), gen_max_bound.d_inf_norm)
        );

        let gen_max = [0.5 * f64::MAX, f64::MAX, f64::MAX, -0.5 * f64::MAX];
        assert_eq!(
            ldl_inertia_with_backward_bound(&gen_max, 2),
            Err(LdlError::NonFiniteFactor {
                stage: "D 2x2 norm"
            })
        );
        let gen_max_cert = certify_shifted_ldlt_inertia(&gen_max, 2, f64::MAX, 0.0)
            .expect("general MAX 2x2 shifted");
        assert_eq!(gen_max_cert.negative, 1);
        assert_eq!(gen_max_cert.positive, 1);

        let eta = f64::from_bits(1);
        let anti_sub = [0.0, eta, eta, 0.0];
        let anti_sub_intervals =
            two_by_two_eigenvalue_intervals(0.0, eta, 0.0).expect("min-subnormal 2x2 intervals");
        assert!(anti_sub_intervals[0].1 < 0.0 && anti_sub_intervals[1].0 > 0.0);
        let anti_sub_bound =
            ldl_inertia_with_backward_bound(&anti_sub, 2).expect("min-subnormal 2x2 unshifted");
        assert_eq!(anti_sub_bound.negative, 1);
        assert_eq!(anti_sub_bound.positive, 1);
        assert_eq!(anti_sub_bound.ambiguous_pivots, 0);
        let anti_sub_cert = certify_shifted_ldlt_inertia(&anti_sub, 2, eta, 0.0)
            .expect("min-subnormal 2x2 shifted");
        assert_eq!(anti_sub_cert.negative, 1);
        assert_eq!(anti_sub_cert.positive, 1);

        let gen_sub = [eta, 2.0 * eta, 2.0 * eta, -eta];
        let gen_sub_bound = ldl_inertia_with_backward_bound(&gen_sub, 2)
            .expect("general min-subnormal 2x2 unshifted");
        assert_eq!(gen_sub_bound.negative, 1);
        assert_eq!(gen_sub_bound.positive, 1);
        assert_eq!(gen_sub_bound.ambiguous_pivots, 0);
        let gen_sub_cert = certify_shifted_ldlt_inertia(&gen_sub, 2, 2.0 * eta, 0.0)
            .expect("general min-subnormal 2x2 shifted");
        assert_eq!(gen_sub_cert.negative, 1);
        assert_eq!(gen_sub_cert.positive, 1);

        let nan_mat = [1.0, f64::NAN, 0.0, 1.0];
        assert_eq!(
            ldl_inertia_with_backward_bound(&nan_mat, 2),
            Err(LdlError::NonFiniteInput { row: 0, col: 1 })
        );
        let inf_mat = [1.0, 0.0, 0.0, f64::INFINITY];
        assert_eq!(
            ldl_inertia_with_backward_bound(&inf_mat, 2),
            Err(LdlError::NonFiniteInput { row: 1, col: 1 })
        );
    }

    #[test]
    fn shifted_ldlt_inertia_certifies_separated_spectrum_and_refuses_near_zero_mode() {
        let separated = [-1.5, 0.2, 0.2, 3.0];
        let cert =
            certify_shifted_ldlt_inertia(&separated, 2, 3.2, 1.0e-6).expect("separated spectrum");
        assert_eq!(cert.negative, 1);
        assert_eq!(cert.positive, 1);

        let near_zero = [-1.5, 0.0, 0.0, 1.0e-9];
        let refusal = certify_shifted_ldlt_inertia(&near_zero, 2, 1.5, 1.0e-6)
            .expect_err("mode inside perturbation window must refuse");
        match refusal {
            ShiftedLdlRefusal::UnresolvedInertia {
                certified_negative,
                certified_positive,
                unresolved,
                dimension,
                ..
            } => {
                assert_eq!(certified_negative, 1);
                assert_eq!(certified_positive, 0);
                assert_eq!(unresolved, 1);
                assert_eq!(dimension, 2);
            }
            other => panic!("expected UnresolvedInertia, got {other:?}"),
        }
    }

    #[test]
    fn ldl_and_shifted_inertia_exercise_every_refusal_path() {
        // 1. DimensionMismatch (n == 0 and len != n * n).
        assert_eq!(
            ldl_inertia_with_backward_bound(&[], 0),
            Err(LdlError::DimensionMismatch { len: 0, n: 0 })
        );
        assert_eq!(
            ldl_inertia_with_backward_bound(&[1.0, 2.0, 3.0], 2),
            Err(LdlError::DimensionMismatch { len: 3, n: 2 })
        );
        assert_eq!(
            certify_shifted_ldlt_inertia(&[], 0, 1.0, 0.0),
            Err(ShiftedLdlRefusal::Factor(LdlError::DimensionMismatch {
                len: 0,
                n: 0,
            }))
        );
        assert_eq!(
            certify_shifted_ldlt_inertia(&[1.0, 2.0, 3.0], 2, 1.0, 0.0),
            Err(ShiftedLdlRefusal::Factor(LdlError::DimensionMismatch {
                len: 3,
                n: 2,
            }))
        );

        // 2. OrderOverflow (n * n overflow and gamma_{2n+4} domain overflow).
        assert_eq!(
            ldl_inertia_with_backward_bound(&[], usize::MAX),
            Err(LdlError::OrderOverflow { n: usize::MAX })
        );
        assert_eq!(
            certify_shifted_ldlt_inertia(&[], usize::MAX, 1.0, 0.0),
            Err(ShiftedLdlRefusal::Factor(LdlError::OrderOverflow {
                n: usize::MAX,
            }))
        );
        let huge_n = 1usize << 52;
        let synthetic_bk = BunchKaufman {
            n: huge_n,
            perm: Vec::new(),
            l: Vec::new(),
            blocks: Vec::new(),
            u_a_inf: 0.0,
            asym_inf: 0.0,
            m_max: 0.0,
            max_schur: 0.0,
        };
        assert_eq!(
            synthetic_bk.inertia_with_backward_bound(),
            Err(LdlError::OrderOverflow { n: huge_n })
        );

        // 3. NonFiniteInput on certify_shifted_ldlt_inertia.
        assert_eq!(
            certify_shifted_ldlt_inertia(&[1.0, f64::NAN, 0.0, 1.0], 2, 1.0, 0.0),
            Err(ShiftedLdlRefusal::Factor(LdlError::NonFiniteInput {
                row: 0,
                col: 1,
            }))
        );

        // 4. Intermediate NonFiniteFactor stages during factorization and bound assembly.
        let half = 0.5 * f64::MAX;
        let schur_1x1_overflow = [half, half, half, -f64::MAX];
        assert_eq!(
            ldl_inertia_with_backward_bound(&schur_1x1_overflow, 2),
            Err(LdlError::NonFiniteFactor {
                stage: "1x1 schur update",
            })
        );
        let schur_2x2_overflow = [0.0, half, half, half, 0.0, half, half, half, -f64::MAX];
        assert_eq!(
            ldl_inertia_with_backward_bound(&schur_2x2_overflow, 3),
            Err(LdlError::NonFiniteFactor {
                stage: "2x2 schur update",
            })
        );
        let asym_sym_overflow = [f64::MAX, f64::MAX, -f64::MAX, f64::MAX];
        assert_eq!(
            ldl_inertia_with_backward_bound(&asym_sym_overflow, 2),
            Err(LdlError::NonFiniteFactor {
                stage: "symmetrization",
            })
        );
        let near_max = f64::MAX.next_down();
        let asym_bound_overflow = [near_max, near_max, -near_max, near_max];
        assert_eq!(
            ldl_inertia_with_backward_bound(&asym_bound_overflow, 2),
            Err(LdlError::NonFiniteFactor {
                stage: "backward error bound",
            })
        );

        // 5. InvalidBound on negative or non-finite operator_scale / perturbation_bound.
        assert_eq!(
            certify_shifted_ldlt_inertia(&[1.0], 1, -1.0, 0.0),
            Err(ShiftedLdlRefusal::InvalidBound {
                operator_scale: -1.0,
                perturbation_bound: 0.0,
            })
        );
        assert_eq!(
            certify_shifted_ldlt_inertia(&[1.0], 1, 1.0, -0.5),
            Err(ShiftedLdlRefusal::InvalidBound {
                operator_scale: 1.0,
                perturbation_bound: -0.5,
            })
        );
        assert!(matches!(
            certify_shifted_ldlt_inertia(&[1.0], 1, f64::NAN, 0.0),
            Err(ShiftedLdlRefusal::InvalidBound { .. })
        ));
        assert_eq!(
            certify_shifted_ldlt_inertia(&[1.0], 1, 1.0, f64::INFINITY),
            Err(ShiftedLdlRefusal::InvalidBound {
                operator_scale: 1.0,
                perturbation_bound: f64::INFINITY,
            })
        );

        // 6. Effective-shift, shifted-diagonal, and unscaled-bound overflow refusal paths.
        assert_eq!(
            certify_shifted_ldlt_inertia(&[1.0], 1, 1.0, f64::MAX),
            Err(ShiftedLdlRefusal::Factor(LdlError::NonFiniteFactor {
                stage: "effective shift",
            }))
        );
        let half_max = 0.5 * f64::MAX;
        assert_eq!(
            certify_shifted_ldlt_inertia(
                &[2.0_f64.powi(500)],
                1,
                2.0_f64.powi(500),
                f64::MAX.next_down().next_down().next_down()
            ),
            Err(ShiftedLdlRefusal::Factor(LdlError::NonFiniteFactor {
                stage: "shifted diagonal",
            }))
        );
        assert_eq!(
            certify_shifted_ldlt_inertia(&[f64::MAX], 1, f64::MAX, f64::MAX),
            Err(ShiftedLdlRefusal::Factor(LdlError::NonFiniteFactor {
                stage: "unscaled shift bound",
            }))
        );
        let cert_half = certify_shifted_ldlt_inertia(&[f64::MAX], 1, f64::MAX, half_max)
            .expect("half_max perturbation is certified on f64::MAX 1x1");
        assert_eq!(cert_half.negative, 0);
        assert_eq!(cert_half.positive, 1);
        assert!(cert_half.shift_applied.is_finite() && cert_half.shift_applied > half_max);

        // 7. AmbiguousPivot when a shifted factor has an exact zero Schur pivot.
        // Construct a 2x2 matrix [[0, b], [b, 0]] with perturbation_bound = 1.0 where b
        // equals the exact +shift_eff applied by `certify_shifted_ldlt_inertia`, so
        // `M + shift_eff I = [[b, b], [b, b]]` reduces to a zero 1x1 Schur complement.
        let mut b = 1.0_f64;
        for _ in 0..4 {
            let probe = [0.0, b, b, 0.0];
            let ldl_0 = ldl_inertia_with_backward_bound(&probe, 2).expect("probe ldl_0");
            let u = 0.5 * f64::EPSILON;
            let shift = (1.0_f64 + (2.0 * ldl_0.backward_error).next_up()).next_up();
            let diag_roundoff = (u * (1.0_f64 + shift).next_up()).next_up();
            let shift_eff = (shift + diag_roundoff).next_up();
            b = (0.0_f64 + shift_eff).next_up();
        }
        let singular_shifted = [0.0, b, b, 0.0];
        let ambig = certify_shifted_ldlt_inertia(&singular_shifted, 2, 1.0, 1.0)
            .expect_err("singular shifted matrix must refuse with AmbiguousPivot");
        assert!(
            matches!(
                ambig,
                ShiftedLdlRefusal::AmbiguousPivot {
                    ambiguous_pivots: 2,
                    ..
                }
            ),
            "expected AmbiguousPivot, got {ambig:?}"
        );

        // 8. InconsistentLoewnerOrder via `classify_shifted_inertia`.
        assert_eq!(
            classify_shifted_inertia((2, 0, 3, 0), (1.0e-6, 1.0e-15, 1.0e-6, 1.0)),
            Err(ShiftedLdlRefusal::InconsistentLoewnerOrder {
                certified_negative: 0,
                certified_positive: 3,
                dimension: 2,
                shift_applied: 1.0e-6,
                ldl_backward_error: 1.0e-15,
                pivot_growth: 1.0,
            })
        );
        assert_eq!(
            classify_shifted_inertia((2, 2, 1, 0), (1.0e-6, 1.0e-15, 1.0e-6, 1.0)),
            Err(ShiftedLdlRefusal::InconsistentLoewnerOrder {
                certified_negative: 2,
                certified_positive: 1,
                dimension: 2,
                shift_applied: 1.0e-6,
                ldl_backward_error: 1.0e-15,
                pivot_growth: 1.0,
            })
        );
    }

    #[test]
    fn shifted_ldlt_inertia_reports_effective_shift() {
        let u = 0.5 * f64::EPSILON;

        // 1. Single-pass path (2 * ldl_0.backward_error > ldl_backward_error + diag_roundoff):
        // shift_applied must equal shift_eff > shift.
        let mat = [-2.0, 0.0, 0.0, 2.0];
        let op_scale = 2.0;
        let pert = 1.0e-3;
        let ldl_0 = ldl_inertia_with_backward_bound(&mat, 2).expect("ldl_0");
        let shift = (pert + (2.0 * ldl_0.backward_error).next_up()).next_up();
        let diag_roundoff = (u * (op_scale + shift).next_up()).next_up();
        let expected_shift_eff = (shift + diag_roundoff).next_up();

        let cert = certify_shifted_ldlt_inertia(&mat, 2, op_scale, pert)
            .expect("separated spectrum with single-pass shift");
        assert_eq!(cert.negative, 1);
        assert_eq!(cert.positive, 1);
        assert_eq!(cert.shift_applied, expected_shift_eff);
        assert!(cert.shift_applied > shift);

        // 2. Retry path (large operator_scale so diag_roundoff forces retry):
        // shift_applied must equal the retried effective shift, strictly exceeding the retried base shift.
        let retry_scale = 1.0e6;
        let init_shift = (2.0 * ldl_0.backward_error).next_up();
        let init_roundoff = (u * (retry_scale + init_shift).next_up()).next_up();
        let init_eff = (init_shift + init_roundoff).next_up();
        let mut m_minus = mat;
        let mut m_plus = mat;
        for i in 0..2 {
            m_minus[i * 2 + i] = (m_minus[i * 2 + i] - init_eff).next_down();
            m_plus[i * 2 + i] = (m_plus[i * 2 + i] + init_eff).next_up();
        }
        let ldl_m = ldl_inertia_with_backward_bound(&m_minus, 2).expect("ldl_m");
        let ldl_p = ldl_inertia_with_backward_bound(&m_plus, 2).expect("ldl_p");
        let max_be = ldl_0
            .backward_error
            .max(ldl_m.backward_error)
            .max(ldl_p.backward_error);
        let retry_shift = ((2.0 * max_be).next_up() + (2.0 * init_roundoff).next_up()).next_up();
        let retry_roundoff = (u * (retry_scale + retry_shift).next_up()).next_up();
        let expected_retry_eff = (retry_shift + retry_roundoff).next_up();

        let retry_cert = certify_shifted_ldlt_inertia(&mat, 2, retry_scale, 0.0)
            .expect("retry path certifies well-separated spectrum");
        assert_eq!(retry_cert.negative, 1);
        assert_eq!(retry_cert.positive, 1);
        assert_eq!(retry_cert.shift_applied, expected_retry_eff);
        assert!(retry_cert.shift_applied > retry_shift);
        assert!(retry_cert.shift_applied > init_eff);
    }
}
