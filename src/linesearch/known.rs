//! Line searches that start from a point whose value and gradient the
//! caller already holds.
//!
//! Every solver step arrives at a line search with `f(x)` and `g(x)` in
//! hand, and leaves it needing `g` at the accepted point. Paying an
//! oracle call for either is pure waste: on an atomistic surface the
//! oracle is a force call, and a quasi-Newton search that accepts the
//! unit step would otherwise cost three calls where one suffices.
//! [`LineSearch::search_from`] takes `(f0, g0)` and returns the accepted
//! point with its value and gradient, so each trial is evaluated once and
//! never again.
//!
//! The steps are bounded by `alpha_max`: a caller with a displacement cap
//! (per-atom or Euclidean) passes the largest `alpha` whose step obeys it,
//! so every evaluated trial is a point the caller can actually take and
//! no step is rescaled after it was measured.

use ndarray::{Array1, ArrayView1};

use super::conditions::{armijo, goldstein_lower, strong_curvature};
use super::interp::{clamp_between, cubic_min, quad_min};
use super::{LineSearch, LineSearchOptions, WolfeExtrapolation};

/// Accepted point of a line search, with the oracle answer at it.
#[derive(Clone, Debug)]
pub struct LineOutcome {
    /// The accepted point `x0 + alpha d`.
    pub x: Array1<f64>,
    /// `f` at [`Self::x`].
    pub f: f64,
    /// `g` at [`Self::x`].
    pub g: Array1<f64>,
    /// The accepted step length, `alpha > 0`.
    pub alpha: f64,
    /// Oracle calls the search made.
    pub evals: usize,
}

/// One evaluated trial on the line.
#[derive(Clone)]
struct Probe {
    /// Evaluation order. The lowest trial keeps the earliest one on a tie.
    seen: usize,
    alpha: f64,
    f: f64,
    dphi: f64,
    x: Array1<f64>,
    g: Array1<f64>,
}

impl Probe {
    fn into_outcome(self, evals: usize) -> LineOutcome {
        LineOutcome {
            x: self.x,
            f: self.f,
            g: self.g,
            alpha: self.alpha,
            evals,
        }
    }
}

/// Counts evaluations and keeps a copy of the lowest decreasing trial.
struct Line<'a, F> {
    oracle: &'a mut F,
    pos: ArrayView1<'a, f64>,
    dir: ArrayView1<'a, f64>,
    f0: f64,
    evals: usize,
    best: Option<Probe>,
    options: LineSearchOptions,
}

impl<'a, F> Line<'a, F>
where
    F: FnMut(ArrayView1<'_, f64>) -> (f64, Array1<f64>),
{
    fn probe(&mut self, alpha: f64) -> Probe {
        let mut x = self.pos.to_owned();
        crate::vecops::axpy(alpha, self.dir, &mut x);
        let (f, g) = (self.oracle)(x.view());
        self.evals += 1;
        let dphi = crate::vecops::dot(g.view(), self.dir);
        Probe {
            seen: self.evals,
            alpha,
            f,
            dphi,
            x,
            g,
        }
    }

    /// Remember a trial that this search will drop. A trial that is returned
    /// is not copied: the caller already owns it. Equal values keep the
    /// earlier trial, which is what updating `best` inside `probe` did.
    fn consider(&mut self, p: &Probe) {
        if !p.f.is_finite() {
            return;
        }
        let keep = match &self.best {
            None => p.f < self.f0,
            Some(b) => p.f < b.f || (p.f == b.f && p.seen < b.seen),
        };
        if keep {
            self.best = Some(p.clone());
        }
    }

    fn consider_end(&mut self, end: &End) {
        if let Some(p) = &end.p {
            self.consider(p);
        }
    }

    fn accept(&self, p: Probe) -> Option<LineOutcome> {
        Some(p.into_outcome(self.evals))
    }

    fn take_best(&mut self) -> Option<LineOutcome> {
        let evals = self.evals;
        self.best.take().map(|b| b.into_outcome(evals))
    }
}

/// One end of a zoom bracket; `p` is `None` only at `alpha = 0`.
#[derive(Clone)]
struct End {
    a: f64,
    f: f64,
    d: f64,
    p: Option<Probe>,
}

impl End {
    fn of(p: Probe) -> Self {
        Self {
            a: p.alpha,
            f: p.f,
            d: p.dphi,
            p: Some(p),
        }
    }
}

/// Next zoom trial: the safeguarded cubic, then the quadratic, then the
/// midpoint, clamped to the central 80 percent of the bracket.
///
/// Interpolation needs a model that holds across the bracket. A cubic
/// through `(lo, hi)` is trusted when
///
/// `|f(hi) - f(lo)| <= 1e3 (1 + |f(lo)| + |phi'(lo)| (hi - lo))`.
///
/// The right side is the value scale of the bracket: the start's own
/// magnitude plus the change a unit-curvature model predicts from the
/// slope over the bracket width. When the far end is non-finite or above
/// that scale by many orders, the fit carries no information about the
/// minimizer, so the trial is the midpoint.
fn zoom_trial(lo: &End, hi: &End) -> f64 {
    let w = hi.a - lo.a;
    let inner_lo = lo.a + 0.1 * w;
    let inner_hi = hi.a - 0.1 * w;
    let scale = 1.0 + lo.f.abs() + lo.d.abs() * w.abs();
    let model_holds = hi.f.is_finite() && (hi.f - lo.f).abs() <= 1e3 * scale;
    let t = if model_holds {
        cubic_min(lo.a, lo.f, lo.d, hi.a, hi.f, hi.d)
            .or_else(|| quad_min(lo.a, lo.f, lo.d, hi.a, hi.f))
            .unwrap_or(0.5 * (lo.a + hi.a))
    } else {
        0.5 * (lo.a + hi.a)
    };
    clamp_between(t, inner_lo, inner_hi)
}

impl LineSearch {
    /// Line search from a point whose value `f0` and gradient `g0` are
    /// known, along the descent direction `dir`.
    ///
    /// Trials are `pos + alpha dir` with `0 < alpha <= alpha_max`; the
    /// opening trial is `min(istep, alpha_max)`. An uphill `dir`
    /// (`g0 . dir > 0`) is searched as `-dir`. Returns `None` when no
    /// acceptable trial is found or `g0 . dir = 0`, and otherwise the
    /// accepted point with `f` and `g` there. Wolfe searches also accept
    /// approximate Wolfe slopes with strong curvature when the energy
    /// change lies within four machine epsilons times `|f0|`.
    /// The start is never re-evaluated and no trial is evaluated twice.
    ///
    /// [`LineSearch::Wolfe`] brackets and zooms (Nocedal-Wright
    /// algorithms 3.5 and 3.6) with safeguarded cubic interpolation on
    /// both bracket slopes, a quadratic fallback, and More-Thuente
    /// extrapolation bounds `[1.1, 4]` times the last increment.
    /// [`WolfeExtrapolation::Case3Average`] replaces that extrapolation
    /// and is off unless the caller opts in. A trial
    /// at `alpha_max` that satisfies Armijo and is still descending is
    /// accepted: the cap, not the curvature, ends the step. If the zoom
    /// exhausts `maxiter` the lowest Armijo point is returned.
    pub fn search_from<F>(
        &self,
        oracle: F,
        pos: ArrayView1<'_, f64>,
        f0: f64,
        g0: ArrayView1<'_, f64>,
        dir: ArrayView1<'_, f64>,
        istep: f64,
        alpha_max: f64,
    ) -> Option<LineOutcome>
    where
        F: FnMut(ArrayView1<'_, f64>) -> (f64, Array1<f64>),
    {
        self.search_from_with_options(
            oracle,
            pos,
            f0,
            g0,
            dir,
            istep,
            alpha_max,
            LineSearchOptions::default(),
        )
    }

    /// Search from cached values with an explicit objective accuracy policy.
    /// The option affects Wolfe acceptance only; all trial caps remain in force.
    pub fn search_from_with_options<F>(
        &self,
        mut oracle: F,
        pos: ArrayView1<'_, f64>,
        f0: f64,
        g0: ArrayView1<'_, f64>,
        dir: ArrayView1<'_, f64>,
        istep: f64,
        alpha_max: f64,
        options: LineSearchOptions,
    ) -> Option<LineOutcome>
    where
        F: FnMut(ArrayView1<'_, f64>) -> (f64, Array1<f64>),
    {
        let dphi0 = crate::vecops::dot(g0, dir);
        let amax = if alpha_max > 0.0 {
            alpha_max
        } else {
            f64::INFINITY
        };
        if !f0.is_finite() || !dphi0.is_finite() || amax.is_nan() || dphi0 == 0.0 {
            return None;
        }
        if dphi0 > 0.0 {
            // An uphill direction (an indefinite SR1 or SR2 model) is
            // searched backwards: -dir descends and obeys the same cap.
            let back = dir.mapv(|v| -v);
            return self.search_from_with_options(
                oracle,
                pos,
                f0,
                g0,
                back.view(),
                istep,
                alpha_max,
                options,
            );
        }
        let open = istep.abs().max(1e-16).min(amax);
        let mut line = Line {
            oracle: &mut oracle,
            pos,
            dir,
            f0,
            evals: 0,
            best: None,
            options,
        };
        match *self {
            Self::Wolfe { c1, c2, maxiter } => {
                let amax = amax.min(64.0_f64.max(64.0 * istep.abs()));
                wolfe(&mut line, f0, dphi0, open.min(amax), amax, c1, c2, maxiter)
            }
            Self::Backtracking { c, beta, maxiter } => {
                backtrack(&mut line, f0, dphi0, open, amax, c, beta, maxiter, false)
            }
            Self::Goldstein { c, beta, maxiter } => {
                backtrack(&mut line, f0, dphi0, open, amax, c, beta, maxiter, true)
            }
            Self::Brent { maxiter, tol } => brent(&mut line, f0, open, amax, maxiter, tol),
        }
    }
}

/// More-Thuente case 3: the cubic extrapolant based at the current step,
/// and the secant of the two slopes. With no bracket yet the farther of
/// the two is the trial. A cubic that does not point forward is the
/// upper bound, so that bound wins over a short secant.
fn extrapolate_case3(
    prev_a: f64,
    prev_f: f64,
    prev_d: f64,
    alpha: f64,
    f: f64,
    d: f64,
    hi_ext: f64,
) -> f64 {
    let w = alpha - prev_a;
    let theta = 3.0 * (prev_f - f) / w + prev_d + d;
    let s = theta.abs().max(prev_d.abs()).max(d.abs());
    let stpc = if w > 0.0 && s > 0.0 && s.is_finite() {
        let disc = (theta / s) * (theta / s) - (prev_d / s) * (d / s);
        if disc > 0.0 {
            let gamma = -s * disc.sqrt();
            let p = (gamma - d) + theta;
            let q = (gamma + (prev_d - d)) + gamma;
            let t = if gamma != 0.0 && q != 0.0 {
                let r = p / q;
                if r < 0.0 {
                    alpha + r * (prev_a - alpha)
                } else {
                    hi_ext
                }
            } else {
                hi_ext
            };
            if t.is_finite() && t > alpha {
                t
            } else {
                hi_ext
            }
        } else {
            hi_ext
        }
    } else {
        hi_ext
    };
    let denom = d - prev_d;
    let stpq = if denom != 0.0 && denom.is_finite() {
        let t = alpha + (d / denom) * (prev_a - alpha);
        if t.is_finite() && t > alpha { t } else { alpha }
    } else {
        alpha
    };
    if (stpc - alpha).abs() > (stpq - alpha).abs() {
        stpc
    } else {
        stpq
    }
}

#[allow(clippy::too_many_arguments)]
fn wolfe<F>(
    line: &mut Line<'_, F>,
    f0: f64,
    dphi0: f64,
    open: f64,
    amax: f64,
    c1: f64,
    c2: f64,
    maxiter: usize,
) -> Option<LineOutcome>
where
    F: FnMut(ArrayView1<'_, f64>) -> (f64, Array1<f64>),
{
    let mut prev = End {
        a: 0.0,
        f: f0,
        d: dphi0,
        p: None,
    };
    let mut alpha = open;
    for i in 0..maxiter.max(1) {
        let cur = End::of(line.probe(alpha));
        let (f, d) = (cur.f, cur.d);
        if line
            .options
            .accepts_approximate_wolfe(f, f0, d, dphi0, c1, c2)
        {
            return cur.p.and_then(|p| line.accept(p));
        }
        if !f.is_finite() || !armijo(f, f0, alpha, dphi0, c1) || (i > 0 && f >= prev.f) {
            return zoom(line, f0, dphi0, prev, cur, c1, c2, maxiter);
        }
        if strong_curvature(d, dphi0, c2) || (d < 0.0 && alpha >= amax) {
            // A strong-Wolfe point, or an Armijo point still descending
            // at the cap: the cap, not the curvature, ends the step.
            return cur.p.and_then(|p| line.accept(p));
        }
        if d >= 0.0 {
            return zoom(line, f0, dphi0, cur, prev, c1, c2, maxiter);
        }
        // More-Thuente extrapolation: the cubic's minimiser beyond the
        // current point. The default keeps it within [1.1, 4] increments.
        // Case3Average, when requested, averages that cubic with the
        // case-3 step if the case-3 step is the shorter one.
        let step = alpha - prev.a;
        let hi_ext = alpha + 4.0 * step;
        let (lo_ext, t) = if line.options.extrapolation() == WolfeExtrapolation::Case3Average {
            let interior = cubic_min(prev.a, prev.f, prev.d, alpha, f, d)
                .filter(|t| *t > alpha)
                .unwrap_or(hi_ext);
            let case3 = extrapolate_case3(prev.a, prev.f, prev.d, alpha, f, d, hi_ext);
            let t = if case3 < 0.75 * interior {
                0.5 * (interior + case3)
            } else {
                interior
            };
            (alpha + 0.5 * step, t)
        } else {
            let t = cubic_min(prev.a, prev.f, prev.d, alpha, f, d)
                .filter(|t| *t > alpha)
                .unwrap_or(hi_ext);
            (alpha + 1.1 * step, t)
        };
        line.consider_end(&prev);
        prev = cur;
        alpha = clamp_between(t, lo_ext, hi_ext).min(amax);
    }
    // Expansion budget spent while still descending: `prev` is the
    // furthest Armijo point.
    match prev.p {
        Some(p) => line.accept(p),
        None => line.take_best(),
    }
}

#[allow(clippy::too_many_arguments)]
fn zoom<F>(
    line: &mut Line<'_, F>,
    f0: f64,
    dphi0: f64,
    mut lo: End,
    mut hi: End,
    c1: f64,
    c2: f64,
    maxiter: usize,
) -> Option<LineOutcome>
where
    F: FnMut(ArrayView1<'_, f64>) -> (f64, Array1<f64>),
{
    for _ in 0..maxiter {
        let width = (hi.a - lo.a).abs();
        if width.is_nan() || width <= 4.0 * f64::EPSILON * lo.a.abs().max(hi.a.abs()) {
            break;
        }
        let alpha = zoom_trial(&lo, &hi);
        let cur = End::of(line.probe(alpha));
        let (f, d) = (cur.f, cur.d);
        if line
            .options
            .accepts_approximate_wolfe(f, f0, d, dphi0, c1, c2)
        {
            return cur.p.and_then(|p| line.accept(p));
        }
        if !f.is_finite() || !armijo(f, f0, alpha, dphi0, c1) || f >= lo.f {
            line.consider_end(&hi);
            hi = cur;
        } else if strong_curvature(d, dphi0, c2) {
            return cur.p.and_then(|p| line.accept(p));
        } else {
            if d * (hi.a - lo.a) >= 0.0 {
                line.consider_end(&hi);
                hi = lo;
            } else {
                line.consider_end(&lo);
            }
            lo = cur;
        }
    }
    // The zoom ran out: `lo` is the lowest Armijo point found. With
    // `lo` still at the origin, fall back to the lowest trial.
    match lo.p {
        Some(p) => line.accept(p),
        None => {
            line.consider_end(&hi);
            line.take_best()
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn backtrack<F>(
    line: &mut Line<'_, F>,
    f0: f64,
    dphi0: f64,
    open: f64,
    amax: f64,
    c: f64,
    beta: f64,
    maxiter: usize,
    goldstein: bool,
) -> Option<LineOutcome>
where
    F: FnMut(ArrayView1<'_, f64>) -> (f64, Array1<f64>),
{
    let mut alpha = open;
    let grow_cap = amax.min(64.0_f64.max(64.0 * open));
    for _ in 0..maxiter {
        let p = line.probe(alpha);
        let f = p.f;
        let arm = f.is_finite() && armijo(f, f0, alpha, dphi0, c);
        let accept = if goldstein {
            arm && goldstein_lower(f, f0, alpha, dphi0, c)
        } else {
            arm
        };
        if accept || (goldstein && arm && alpha >= grow_cap) {
            return line.accept(p);
        }
        line.consider(&p);
        if goldstein && arm {
            // Lower bound failed: the step is too short (Nocedal-Wright 3.11).
            alpha = (alpha / beta).min(grow_cap);
        } else {
            alpha *= beta;
        }
        if !alpha.is_finite() || alpha < 1e-16 {
            break;
        }
    }
    line.take_best()
}

/// Brent (1973) on a golden bracket of `phi(alpha)`, derivative-free in
/// `alpha`. The bracket starts at the known `phi(0) = f0` and never grows
/// past `amax`; if `phi` still decreases at `amax`, `amax` is the answer.
fn brent<F>(
    line: &mut Line<'_, F>,
    f0: f64,
    open: f64,
    amax: f64,
    maxiter: usize,
    tol: f64,
) -> Option<LineOutcome>
where
    F: FnMut(ArrayView1<'_, f64>) -> (f64, Array1<f64>),
{
    const GOLD: f64 = 1.618_034;
    const CGOLD: f64 = 0.381_966;
    let phi = |line: &mut Line<'_, F>, t: f64| {
        let p = line.probe(t);
        line.consider(&p);
        if p.f.is_finite() { p.f } else { f64::INFINITY }
    };
    // Bracket a minimum of phi on [0, c] with phi(b) < phi(0), phi(c).
    let (mut a, mut fa) = (0.0, f0);
    let (mut b, mut fb) = (open, phi(line, open));
    if fb >= fa {
        // Shrink towards 0 until the step decreases phi.
        let mut it = 0;
        while fb >= fa && it < maxiter {
            b *= CGOLD;
            if b < 1e-16 {
                return line.take_best();
            }
            fb = phi(line, b);
            it += 1;
        }
        if fb >= fa {
            return line.take_best();
        }
        // Bracket is [0, b / CGOLD], whose right end was evaluated.
        let c = b / CGOLD;
        return brent_refine(line, a, c, b, fb, tol, maxiter.max(20));
    }
    let mut c = (b + GOLD * (b - a)).min(amax);
    if c <= b {
        return line.take_best();
    }
    let mut fc = phi(line, c);
    let mut it = 0;
    while fc < fb && it < maxiter {
        if c >= amax {
            // Still decreasing at the cap: the lowest trial is `c`.
            return line.take_best();
        }
        let u = (c + GOLD * (c - b)).min(amax);
        a = b;
        fa = fb;
        b = c;
        fb = fc;
        c = u;
        fc = phi(line, c);
        it += 1;
    }
    let _ = fa;
    brent_refine(line, a, c, b, fb, tol, maxiter.max(20))
}

/// Brent's parabolic / golden refinement on `[ax, cx]` from the interior
/// point `bx` with the known value `fbx`.
fn brent_refine<F>(
    line: &mut Line<'_, F>,
    ax: f64,
    cx: f64,
    bx: f64,
    fbx: f64,
    tol: f64,
    maxiter: usize,
) -> Option<LineOutcome>
where
    F: FnMut(ArrayView1<'_, f64>) -> (f64, Array1<f64>),
{
    const CGOLD: f64 = 0.381_966;
    let mut a = ax.min(cx);
    let mut b = ax.max(cx);
    let (mut x, mut w, mut v) = (bx, bx, bx);
    let (mut fx, mut fw, mut fv) = (fbx, fbx, fbx);
    let mut e: f64 = 0.0;
    let mut d: f64 = 0.0;
    for _ in 0..maxiter {
        let xm = 0.5 * (a + b);
        let tol1 = tol * x.abs() + 1e-12;
        let tol2 = 2.0 * tol1;
        if (x - xm).abs() <= tol2 - 0.5 * (b - a) {
            break;
        }
        if e.abs() > tol1 {
            let r = (x - w) * (fx - fv);
            let mut q = (x - v) * (fx - fw);
            let mut p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            }
            q = q.abs();
            let etemp = e;
            e = d;
            if p.abs() >= (0.5 * q * etemp).abs() || p <= q * (a - x) || p >= q * (b - x) {
                e = if x >= xm { a - x } else { b - x };
                d = CGOLD * e;
            } else {
                d = p / q;
                let u = x + d;
                if u - a < tol2 || b - u < tol2 {
                    d = if xm - x >= 0.0 { tol1 } else { -tol1 };
                }
            }
        } else {
            e = if x >= xm { a - x } else { b - x };
            d = CGOLD * e;
        }
        let u = if d.abs() >= tol1 {
            x + d
        } else {
            x + if d >= 0.0 { tol1 } else { -tol1 }
        };
        let fu = {
            let p = line.probe(u);
            line.consider(&p);
            if p.f.is_finite() { p.f } else { f64::INFINITY }
        };
        if fu <= fx {
            if u >= x {
                a = x;
            } else {
                b = x;
            }
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || w == x {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        }
    }
    line.take_best()
}

#[cfg(test)]
mod tests {
    #[test]
    fn zoom_bisects_when_the_far_value_carries_no_information() {
        let lo = End {
            a: 0.0,
            f: -24.6,
            d: -1.0e3,
            p: None,
        };
        let hi = End {
            a: 1.0,
            f: 4.0e45,
            d: 1.0e46,
            p: None,
        };
        assert_eq!(zoom_trial(&lo, &hi), 0.5);
        let hi = End {
            a: 1.0,
            f: f64::INFINITY,
            d: f64::NAN,
            p: None,
        };
        assert_eq!(zoom_trial(&lo, &hi), 0.5);
    }

    #[test]
    fn zoom_interpolates_when_the_values_are_commensurate() {
        // phi(a) = (a - 0.3)^2 has its minimum at 0.3; the cubic finds it.
        let lo = End {
            a: 0.0,
            f: 0.09,
            d: -0.6,
            p: None,
        };
        let hi = End {
            a: 1.0,
            f: 0.49,
            d: 1.4,
            p: None,
        };
        let t = zoom_trial(&lo, &hi);
        assert!((t - 0.3).abs() < 1e-12, "trial {t}");
    }

    use super::*;
    use ndarray::array;
    use std::cell::Cell;

    fn wolfe() -> LineSearch {
        LineSearch::Wolfe {
            c1: 1e-4,
            c2: 0.9,
            maxiter: 20,
        }
    }

    /// `f(x) = (x - 1)^2 / 2`: along `d = 1` from 0 the unit step is exact.
    fn quad(x: ArrayView1<'_, f64>) -> (f64, Array1<f64>) {
        let z = x[0] - 1.0;
        (0.5 * z * z, array![z])
    }

    #[test]
    fn an_exact_unit_step_costs_one_evaluation() {
        let n = Cell::new(0);
        let oracle = |x: ArrayView1<'_, f64>| {
            n.set(n.get() + 1);
            quad(x)
        };
        let pos = array![0.0];
        let (f0, g0) = quad(pos.view());
        let dir = array![1.0];
        let out = wolfe()
            .search_from(
                oracle,
                pos.view(),
                f0,
                g0.view(),
                dir.view(),
                1.0,
                f64::INFINITY,
            )
            .unwrap();
        assert_eq!(n.get(), 1);
        assert_eq!(out.evals, 1);
        assert!((out.x[0] - 1.0).abs() < 1e-15);
        assert!(out.g[0].abs() < 1e-15);
        assert_eq!(out.f, 0.0);
    }

    #[test]
    fn the_cap_bounds_every_trial_and_ends_the_step() {
        let seen = Cell::new(0.0_f64);
        let oracle = |x: ArrayView1<'_, f64>| {
            seen.set(seen.get().max(x[0]));
            quad(x)
        };
        let pos = array![0.0];
        let (f0, g0) = quad(pos.view());
        let dir = array![1.0];
        for ls in [
            wolfe(),
            LineSearch::default(),
            LineSearch::Backtracking {
                c: 1e-4,
                beta: 0.5,
                maxiter: 20,
            },
            LineSearch::Goldstein {
                c: 0.25,
                beta: 0.5,
                maxiter: 20,
            },
        ] {
            seen.set(0.0);
            let out = ls
                .search_from(oracle, pos.view(), f0, g0.view(), dir.view(), 1.0, 0.25)
                .unwrap();
            assert!(seen.get() <= 0.25, "{ls:?} probed {}", seen.get());
            assert!((out.alpha - 0.25).abs() < 1e-15, "{ls:?} {}", out.alpha);
            assert!((out.g[0] + 0.75).abs() < 1e-15);
        }
    }

    #[test]
    fn an_overlong_step_zooms_with_interpolation() {
        // phi(alpha) = (alpha - 1)^2 / 2 from istep 10: Armijo fails at
        // 10, and the cubic through (0, 0.5, -1) and (10, 40.5, 9) is the
        // quadratic itself, so the first zoom trial is exact.
        let n = Cell::new(0);
        let oracle = |x: ArrayView1<'_, f64>| {
            n.set(n.get() + 1);
            quad(x)
        };
        let pos = array![0.0];
        let (f0, g0) = quad(pos.view());
        let dir = array![1.0];
        let out = wolfe()
            .search_from(
                oracle,
                pos.view(),
                f0,
                g0.view(),
                dir.view(),
                10.0,
                f64::INFINITY,
            )
            .unwrap();
        assert_eq!(n.get(), 2);
        assert!((out.x[0] - 1.0).abs() < 1e-12, "{}", out.x[0]);
    }

    #[test]
    fn an_uphill_direction_is_searched_backwards() {
        let n = Cell::new(0);
        let oracle = |x: ArrayView1<'_, f64>| {
            n.set(n.get() + 1);
            quad(x)
        };
        let pos = array![0.0];
        let (f0, g0) = quad(pos.view());
        let dir = array![-1.0];
        let out = wolfe()
            .search_from(
                oracle,
                pos.view(),
                f0,
                g0.view(),
                dir.view(),
                1.0,
                f64::INFINITY,
            )
            .unwrap();
        assert!((out.x[0] - 1.0).abs() < 1e-15);
        assert_eq!(n.get(), 1);
        // A direction orthogonal to the gradient costs nothing.
        let flat = array![0.0];
        let none = wolfe().search_from(
            oracle,
            pos.view(),
            f0,
            g0.view(),
            flat.view(),
            1.0,
            f64::INFINITY,
        );
        assert!(none.is_none());
        assert_eq!(n.get(), 1);
    }

    #[test]
    fn a_short_step_extrapolates_to_a_wolfe_point() {
        // From istep 1e-3 the expansion must reach |alpha - 1| <= 0.9
        // without re-evaluating anything.
        let pts = std::cell::RefCell::new(Vec::new());
        let oracle = |x: ArrayView1<'_, f64>| {
            pts.borrow_mut().push(x[0]);
            quad(x)
        };
        let pos = array![0.0];
        let (f0, g0) = quad(pos.view());
        let dir = array![1.0];
        let out = wolfe()
            .search_from(
                oracle,
                pos.view(),
                f0,
                g0.view(),
                dir.view(),
                1e-3,
                f64::INFINITY,
            )
            .unwrap();
        assert!(out.alpha > 0.1, "{}", out.alpha);
        let mut p = pts.borrow().clone();
        let total = p.len();
        p.sort_by(f64::total_cmp);
        p.dedup();
        assert_eq!(p.len(), total, "a trial was evaluated twice");
    }

    #[test]
    fn case3_extrapolation_recovers_a_quadratic_minimiser() {
        let f = |t: f64| 0.5 * (t - 1.0) * (t - 1.0);
        let d = |t: f64| t - 1.0;
        let t = extrapolate_case3(0.2, f(0.2), d(0.2), 0.4, f(0.4), d(0.4), 10.0);
        assert!((t - 1.0).abs() < 1e-12, "{t}");
    }

    #[test]
    fn case3_average_is_opt_in_and_changes_the_extrapolated_step() {
        assert_eq!(
            LineSearchOptions::default().extrapolation(),
            WolfeExtrapolation::Interior
        );
        // phi(t) = (t - 0.34)^2 / 2. The opening step 0.2 is still
        // steeper than c2 = 0.1 allows, so the search extrapolates.
        // The minimiser sits between half an increment and 1.1
        // increments, which is where the two safeguards disagree.
        let phi = |x: ArrayView1<'_, f64>| {
            let z = x[0] - 0.34;
            (0.5 * z * z, array![z])
        };
        let search = LineSearch::Wolfe {
            c1: 1e-4,
            c2: 0.1,
            maxiter: 20,
        };
        let pos = array![0.0];
        let (f0, g0) = phi(pos.view());
        let dir = array![1.0];
        let run = |options: LineSearchOptions| {
            let seen = std::cell::RefCell::new(Vec::new());
            let oracle = |x: ArrayView1<'_, f64>| {
                seen.borrow_mut().push(x[0]);
                phi(x)
            };
            let out = search
                .search_from_with_options(
                    oracle,
                    pos.view(),
                    f0,
                    g0.view(),
                    dir.view(),
                    0.05,
                    f64::INFINITY,
                    options,
                )
                .unwrap();
            (out, seen.into_inner())
        };
        let (plain, plain_pts) = run(LineSearchOptions::default());
        let (opted, opted_pts) =
            run(LineSearchOptions::default().with_extrapolation(WolfeExtrapolation::Case3Average));
        assert_ne!(
            plain_pts, opted_pts,
            "default {plain_pts:?} case3 {opted_pts:?}"
        );
        assert!(opted.f < f0, "case3 energy {}", opted.f);
        assert!(plain.f < f0, "default energy {}", plain.f);
    }

    #[test]
    fn the_session_uses_the_case3_average_preset() {
        use crate::control::Control;
        use crate::method::Method;
        use crate::nlcg::{Conjugacy, Restart};
        use crate::session::Solver;
        use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct Well {
            bounds: Bounds<f64>,
            calls: AtomicUsize,
        }
        impl Objective<f64> for Well {
            fn dim(&self) -> usize {
                1
            }
            fn bounds(&self) -> &Bounds<f64> {
                &self.bounds
            }
            fn eval(&self, x: ArrayView1<f64>) -> f64 {
                self.value_and_gradient(x).0
            }
        }
        impl Gradient<f64> for Well {
            fn dim(&self) -> usize {
                1
            }
            fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
                self.value_and_gradient(x).1
            }
        }
        impl DifferentiableObjective<f64> for Well {
            fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
                self.calls.fetch_add(1, Ordering::Relaxed);
                let z = x[0] - 0.34;
                (0.5 * z * z, array![z])
            }
        }
        let obj = Well {
            bounds: Bounds::new(array![-10.0], array![10.0], 0.0),
            calls: AtomicUsize::new(0),
        };
        let control = Control {
            istep: 0.2 / 0.34,
            ..Control::default()
        };
        let method = Method::Nlcg {
            conjugacy: Conjugacy::PolakRibierePlus,
            restart: Restart::njws(),
        };
        let mut x_default = array![0.0];
        let mut x_case3 = array![0.0];
        let mut plain = Solver::new(method.clone(), control.clone(), 1);
        let mut opted = Solver::new(method, control, 1);
        plain.set_atom_maxmove(-1.0);
        opted.set_atom_maxmove(-1.0);
        opted.set_wolfe_extrapolation(WolfeExtrapolation::Case3Average);
        plain.step(&obj, &mut x_default).unwrap();
        let calls_default = obj.calls.swap(0, Ordering::Relaxed);
        opted.step(&obj, &mut x_case3).unwrap();
        let calls_case3 = obj.calls.load(Ordering::Relaxed);
        assert_ne!(
            calls_default, calls_case3,
            "default calls {calls_default} at {} case3 calls {calls_case3} at {}",
            x_default[0], x_case3[0]
        );
    }
}
