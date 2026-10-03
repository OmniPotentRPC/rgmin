//! Persistent limited-memory BFGS (Nocedal-Wright 7.4, scaling 7.20).
//!
//! The production local method is L-BFGS with the strong Wolfe conditions
//! (Nocedal-Wright algorithms 3.5 and 3.6). Armijo alone accepts a step
//! that decreases the value without measuring curvature, and every later
//! direction is built from the stored pairs, so one bad pair degrades the
//! whole memory.
//!
//! Liu and Nocedal, *On the limited memory BFGS method for large scale
//! optimization*, <https://doi.org/10.1007/BF01589116>.
//! Nocedal, *Updating quasi-Newton matrices with limited storage*,
//! <https://doi.org/10.1090/s0025-5718-1980-0572855-7>.
//! Nocedal and Wright, *Numerical Optimization*,
//! <https://doi.org/10.1007/978-0-387-40065-5>.
//! Wolfe, *Convergence Conditions for Ascent Methods*,
//! <https://doi.org/10.1137/1011036>.

use ndarray::{Array1, ArrayView1};

use crate::control::Control;
use crate::error::{Error, Result};
use crate::linesearch::LineSearch;
use crate::qn::solve_spd;
use crate::report::Report;
use crate::step::{l2, qn_istep, take_step};
use eindir_core::{DifferentiableObjective, Objective};

/// How [`Lbfgs`] compares the gradient to [`Lbfgs::gtol`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GradNorm {
    /// Euclidean `||g||_2`. Matches [`crate::minimize_lbfgs`].
    Euclidean,
    /// Infinity `||g||_∞`. Matches SciPy L-BFGS-B and hopping polish.
    Infinity,
}

/// Stored curvature pair from one accepted step.
struct Pair {
    s: Array1<f64>,
    y: Array1<f64>,
    rho: f64,
}

/// L-BFGS whose curvature pairs survive between calls.
///
/// A hopping chain relaxes thousands of times from perturbations of an
/// already-relaxed structure. The curvature at a new start resembles the
/// curvature at the old minimum; a solver that forgets between calls pays
/// to rediscover it.
pub struct Lbfgs {
    memory: Vec<Pair>,
    precon_fallbacks: std::sync::atomic::AtomicUsize,
    #[cfg(feature = "highs")]
    pub(crate) coordinate_box: Option<eindir_core::Bounds<f64>>,
    /// Pairs retained; the usual choice is between five and ten.
    pub max_pairs: usize,
    /// Gradient-norm threshold that ends a relaxation.
    pub gtol: f64,
    /// Armijo sufficient-decrease constant, `c1` in the Wolfe conditions.
    pub armijo: f64,
    /// Curvature constant, `c2`. The usual choice for quasi-Newton is 0.9.
    pub curvature: f64,
    /// Line-search evaluations attempted before the direction is abandoned.
    pub max_line_evals: usize,
    /// Norm used against [`Lbfgs::gtol`].
    pub norm: GradNorm,
    /// Al-Baali extra-updates as eOn's `lbfgs_extra_updates` names them:
    /// replay the newest pair this many extra times.
    ///
    /// It has no effect, and the two-loop map ignores it: the BFGS update
    /// with `(s, y)` maps any `H` that already satisfies `H y = s` to
    /// itself, so a replayed newest pair leaves `H` unchanged in exact
    /// arithmetic (`validation/lbfgs_two_loop.py`, check 3). The field
    /// stays for source compatibility.
    pub extra_updates: usize,
    /// Li-Fukushima cautious `ε`. Zero disables the filter.
    pub cautious_eps: f64,
    /// Li-Fukushima cautious `α`.
    pub cautious_alpha: f64,
    /// When set, each direction is the HiGHS QP on the compact Hessian
    /// rather than the two-loop recursion.
    #[cfg(feature = "highs")]
    pub highs: Option<crate::lbfgs_qp::HighsStep>,
}

impl Default for Lbfgs {
    fn default() -> Self {
        Self::with_capacity(8)
    }
}

impl Lbfgs {
    /// Fresh solver with room for `max_pairs` curvature pairs.
    pub fn with_capacity(max_pairs: usize) -> Self {
        Self {
            memory: Vec::new(),
            #[cfg(feature = "highs")]
            coordinate_box: None,
            precon_fallbacks: std::sync::atomic::AtomicUsize::new(0),
            max_pairs: max_pairs.max(1),
            gtol: 1e-6,
            armijo: 1e-4,
            curvature: 0.9,
            max_line_evals: 20,
            norm: GradNorm::Infinity,
            extra_updates: 0,
            cautious_eps: 0.0,
            cautious_alpha: 0.01,
            #[cfg(feature = "highs")]
            highs: None,
        }
    }

    /// Discards the stored curvature.
    ///
    /// Called when the chain moves somewhere structurally different, where
    /// the retained pairs describe a Hessian that no longer applies.
    pub fn forget(&mut self) {
        self.memory.clear();
    }

    /// Times a supplied `H0^{-1}` (a Hessian under the Newton-preconditioned
    /// two-loop) was not symmetric positive definite and the two-loop fell
    /// back to `gamma I`.
    #[must_use]
    pub fn precon_fallbacks(&self) -> usize {
        self.precon_fallbacks
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Pairs currently held.
    pub fn len(&self) -> usize {
        self.memory.len()
    }

    /// True when no curvature is stored.
    pub fn is_empty(&self) -> bool {
        self.memory.is_empty()
    }

    /// Drops oldest pairs when `max_pairs` shrank.
    pub fn trim(&mut self) {
        let cap = self.max_pairs.max(1);
        self.max_pairs = cap;
        while self.memory.len() > cap {
            self.memory.remove(0);
        }
    }

    pub(crate) fn search_direction(&self, x: ArrayView1<f64>, g: ArrayView1<f64>) -> Array1<f64> {
        #[cfg(feature = "highs")]
        if self.highs.is_some()
            && let Ok(d) = self.highs_step(x, g)
        {
            return d;
        }
        let _ = x;
        self.direction(g)
    }

    /// Two-loop recursion: applies the inverse-Hessian approximation to `g`.
    pub(crate) fn direction(&self, g: ArrayView1<f64>) -> Array1<f64> {
        self.direction_with_precon(g, None)
    }

    /// Two-loop with optional \(H_0 = P^{-1}\). `precon` is the pair / Lindh
    /// matrix \(P\); the middle product is `solve(P, q)`, not a scalar γ.
    pub(crate) fn direction_with_precon(
        &self,
        g: ArrayView1<f64>,
        precon: Option<&ndarray::Array2<f64>>,
    ) -> Array1<f64> {
        let mut q = g.to_owned();
        let m = self.memory.len();
        let mut alpha = vec![0.0; m];
        for k in (0..m).rev() {
            let p = &self.memory[k];
            let a = p.rho * crate::vecops::dot(p.s.view(), q.view());
            alpha[k] = a;
            crate::vecops::axpy(-a, p.y.view(), &mut q);
        }
        if let Some(pmat) = precon {
            // H0 = P^{-1} keeps -H g a descent direction only for an SPD
            // P; an indefinite Hessian falls back to gamma I and is
            // counted.
            q = match solve_spd(pmat, &q) {
                Some(v) => v,
                None => {
                    self.precon_fallbacks
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    self.scale_gamma(q)
                }
            };
        } else {
            q = self.scale_gamma(q);
        }
        for (k, p) in self.memory.iter().enumerate() {
            let b = p.rho * crate::vecops::dot(p.y.view(), q.view());
            crate::vecops::axpy(alpha[k] - b, p.s.view(), &mut q);
        }
        q.mapv_inplace(|v| -v);
        q
    }

    fn scale_gamma(&self, mut q: Array1<f64>) -> Array1<f64> {
        if let Some(p) = self.memory.last() {
            let yy = p.y.dot(&p.y);
            if yy > 0.0 {
                q *= p.s.dot(&p.y) / yy;
            }
        }
        q
    }

    /// Records an accepted curvature pair (`s = x+ - x`, `y = g+ - g`).
    pub fn record(&mut self, s: Array1<f64>, y: Array1<f64>) {
        self.push(s, y);
    }

    /// Two-loop direction `d = −H g` (Nocedal-Wright 7.4).
    pub fn two_loop(&self, g: ArrayView1<f64>) -> Array1<f64> {
        self.direction(g)
    }

    pub(crate) fn transport<F>(&mut self, mut map: F)
    where
        F: FnMut(&Array1<f64>) -> Array1<f64>,
    {
        let pairs = std::mem::take(&mut self.memory);
        for pair in pairs {
            self.push_pair(map(&pair.s), map(&pair.y), None);
        }
    }

    pub(crate) fn push(&mut self, s: Array1<f64>, y: Array1<f64>) {
        self.push_pair(s, y, None);
    }

    pub(crate) fn push_pair(&mut self, s: Array1<f64>, y: Array1<f64>, gnorm: Option<f64>) {
        let sy = s.dot(&y);
        let sn = s.iter().map(|v| v * v).sum::<f64>().sqrt();
        let yn = y.iter().map(|v| v * v).sum::<f64>().sqrt();
        let ss = sn * sn;
        if self.cautious_eps > 0.0
            && let Some(g) = gnorm
        {
            let thresh = self.cautious_eps * ss * g.max(1.0e-30).powf(self.cautious_alpha);
            if sy < thresh {
                return;
            }
        }
        // Relative curvature: a tiny accepted trust step makes the
        // compact Hessian indefinite and HiGHS's QP solver does not return.
        if !sy.is_finite() || sy <= 1e-8 * sn * yn {
            return;
        }
        self.memory.push(Pair {
            s,
            y,
            rho: 1.0 / sy,
        });
        self.trim();
    }

    fn gnorm(&self, g: &Array1<f64>) -> f64 {
        // A non-finite component means the gradient is broken, not small:
        // f64::max returns its other operand against NaN, so the infinity
        // fold would report an all-NaN gradient as norm zero and the
        // relaxation would terminate as converged at a garbage point. The
        // Euclidean arm failed safe only by accident, NaN propagating into
        // a comparison that then never passes. Both arms now answer
        // infinity, which no gtol accepts; the seam's infinity norm
        // carries that guarantee itself.
        match self.norm {
            GradNorm::Euclidean => {
                if g.iter().any(|v| !v.is_finite()) {
                    return f64::INFINITY;
                }
                l2(g)
            }
            GradNorm::Infinity => crate::vecops::nrminf(g.view()),
        }
    }

    /// Strong Wolfe line search by bracketing then cubic-interpolated zoom.
    ///
    /// Nocedal and Wright algorithms 3.5 and 3.6. Returns whether a step
    /// was accepted and how many evaluations it cost.
    fn line_search<F>(
        &mut self,
        x: &mut Array1<f64>,
        f: &mut f64,
        g: &mut Array1<f64>,
        d: &Array1<f64>,
        slope: f64,
        fg: &mut F,
    ) -> (bool, usize)
    where
        F: FnMut(ArrayView1<f64>) -> Option<(f64, Array1<f64>)>,
    {
        let f0 = *f;
        let mut evals = 0usize;
        let mut scratch = x.clone();
        let probe = |a: f64, fg: &mut F, evals: &mut usize, scratch: &mut Array1<f64>| {
            scratch.assign(x);
            scratch.scaled_add(a, d);
            let r = fg(scratch.view());
            if r.is_some() {
                *evals += 1;
            }
            r
        };

        let mut a_prev = 0.0;
        let mut f_prev = f0;
        let mut slope_prev = slope;
        // A quasi-Newton direction already carries the step length. Nocedal
        // and Wright require that alpha = 1 is tried first. With no memory
        // the direction is the raw negative gradient and needs a length.
        let mut a = if self.memory.is_empty() {
            let dnorm = d.iter().fold(0.0_f64, |acc, v| acc + v * v).sqrt();
            if dnorm > 1.0 { 1.0 / dnorm } else { 1.0 }
        } else {
            1.0
        };
        let mut lo = 0.0;
        let mut f_lo = f0;
        let mut slope_lo = slope;
        let mut hi = f64::NAN;
        let mut f_hi = f64::NAN;
        let mut slope_hi = f64::NAN;
        let mut bracketed = false;

        let bracket_cap = self.max_line_evals;
        let total_cap = 2 * self.max_line_evals;

        for i in 0..bracket_cap {
            let (fa, ga) = match probe(a, fg, &mut evals, &mut scratch) {
                Some(v) => v,
                None => return (false, evals),
            };
            let slope_a = d.dot(&ga);
            if fa > f0 + self.armijo * a * slope || (i > 0 && fa >= f_prev) {
                lo = a_prev;
                f_lo = f_prev;
                slope_lo = slope_prev;
                hi = a;
                f_hi = fa;
                slope_hi = slope_a;
                bracketed = true;
                break;
            }
            if slope_a.abs() <= -self.curvature * slope {
                self.accept(x, f, g, d, a, fa, ga);
                return (true, evals);
            }
            if slope_a >= 0.0 {
                lo = a;
                f_lo = fa;
                slope_lo = slope_a;
                hi = a_prev;
                f_hi = f_prev;
                slope_hi = slope_prev;
                bracketed = true;
                break;
            }
            a_prev = a;
            f_prev = fa;
            slope_prev = slope_a;
            a *= 2.0;
        }

        if !bracketed {
            return (false, evals);
        }

        while evals < total_cap {
            let width = hi - lo;
            let mut trial = lo + 0.5 * width;
            // Cubic Hermite minimizer over the bracket (Nocedal-Wright
            // eq. 3.59), possible because both ends carry their slopes:
            // the slope at hi is information an evaluation already paid
            // for, and discarding it forced a quadratic model that the
            // doc nevertheless called cubic. More-Thuente's dcstep is
            // the reference for the guards: the discriminant clamps at
            // zero, a degenerate denominator falls back, and every
            // candidate is confined to the bracket interior so a wild
            // extrapolation costs a bisection, never a divergence.
            if slope_hi.is_finite() {
                let d1 = slope_lo + slope_hi - 3.0 * (f_lo - f_hi) / (lo - hi);
                let disc = d1 * d1 - slope_lo * slope_hi;
                if disc >= 0.0 && (lo - hi).abs() > 1e-16 {
                    let d2 = (hi - lo).signum() * disc.sqrt();
                    let denom = slope_hi - slope_lo + 2.0 * d2;
                    if denom.abs() > 1e-16 {
                        let q = hi - (hi - lo) * (slope_hi + d2 - d1) / denom;
                        if (q - lo) / width > 0.1 && (q - lo) / width < 0.9 {
                            trial = q;
                        }
                    }
                }
            } else {
                let denom = 2.0 * (f_hi - f_lo - slope_lo * width);
                if denom.abs() > 1e-16 {
                    let q = lo - slope_lo * width * width / denom;
                    if (q - lo) / width > 0.1 && (q - lo) / width < 0.9 {
                        trial = q;
                    }
                }
            }
            let (ft, gt) = match probe(trial, fg, &mut evals, &mut scratch) {
                Some(v) => v,
                None => return (false, evals),
            };
            let slope_t = d.dot(&gt);
            if ft > f0 + self.armijo * trial * slope || ft >= f_lo {
                hi = trial;
                f_hi = ft;
                slope_hi = slope_t;
            } else {
                if slope_t.abs() <= -self.curvature * slope {
                    self.accept(x, f, g, d, trial, ft, gt);
                    return (true, evals);
                }
                if slope_t * (hi - lo) >= 0.0 {
                    hi = lo;
                    f_hi = f_lo;
                    slope_hi = slope_lo;
                }
                lo = trial;
                f_lo = ft;
                slope_lo = slope_t;
            }
            if (hi - lo).abs() < 1e-14 {
                break;
            }
        }
        (false, evals)
    }

    fn accept(
        &mut self,
        x: &mut Array1<f64>,
        f: &mut f64,
        g: &mut Array1<f64>,
        d: &Array1<f64>,
        step: f64,
        f_new: f64,
        g_new: Array1<f64>,
    ) {
        let mut s = d.clone();
        s *= step;
        let mut y = g_new.clone();
        y -= &*g;
        self.push(s, y);
        x.scaled_add(step, d);
        *f = f_new;
        *g = g_new;
    }

    /// Relaxes `x0`, calling `fg` for value and gradient.
    ///
    /// `fg` returns `None` when the caller's budget is spent, which ends the
    /// relaxation where it stands. Returns the value, the point, and the
    /// number of evaluations used.
    pub fn minimize<F>(
        &mut self,
        x0: ArrayView1<f64>,
        max_iter: usize,
        fg: F,
    ) -> (f64, Array1<f64>, usize)
    where
        F: FnMut(ArrayView1<f64>) -> Option<(f64, Array1<f64>)>,
    {
        self.minimize_watched(x0, max_iter, fg, |_, _| true)
    }

    /// Relaxes `x0`, offering each accepted iterate to `watch`.
    ///
    /// `watch` receives the iteration index and the value at that iterate,
    /// and returning `false` ends the relaxation there.
    pub fn minimize_watched<F, W>(
        &mut self,
        x0: ArrayView1<f64>,
        max_iter: usize,
        mut fg: F,
        mut watch: W,
    ) -> (f64, Array1<f64>, usize)
    where
        F: FnMut(ArrayView1<f64>) -> Option<(f64, Array1<f64>)>,
        W: FnMut(usize, f64) -> bool,
    {
        self.trim();
        let mut x = x0.to_owned();
        let mut evals = 0usize;
        let (mut f, mut g) = match fg(x.view()) {
            Some(v) => v,
            None => return (f64::INFINITY, x, evals),
        };
        evals += 1;

        for it in 0..max_iter {
            if !watch(it, f) {
                break;
            }
            if self.gnorm(&g) < self.gtol {
                break;
            }
            let d = self.search_direction(x.view(), g.view());
            let slope = d.dot(&g);
            if slope >= 0.0 {
                self.forget();
                continue;
            }
            let (ok, evals_used) = self.line_search(&mut x, &mut f, &mut g, &d, slope, &mut fg);
            evals += evals_used;
            if !ok {
                if self.memory.is_empty() {
                    break;
                }
                self.forget();
            }
        }
        (f, x, evals)
    }

    /// Relaxes `x0`, consulting `recognise` at each accepted iterate.
    ///
    /// `recognise` maps an accepted iterate to a stand-in result when the
    /// caller can certify where this descent ends: a minimum already on
    /// file whose catchment the iterate has entered. Returning
    /// `Some((f_known, x_known))` ends the relaxation with that result and
    /// the evaluations spent so far, which is the refund -- the remainder
    /// of a descent whose outcome is already known is never paid for. The
    /// final `bool` reports whether the result is a stand-in, so a caller
    /// auditing its recogniser can tell refunded descents from completed
    /// ones.
    ///
    /// The hook sits where `watch` sits, at accepted iterates only, so a
    /// recogniser never sees a trial step the optimizer has not adopted.
    /// Soundness is the caller's contract: the stand-in must be the
    /// minimum this descent would have reached, and the tolerance for
    /// getting that wrong belongs to the caller's error budget, not to
    /// the solver.
    pub fn minimize_recognized<F, R>(
        &mut self,
        x0: ArrayView1<f64>,
        max_iter: usize,
        mut fg: F,
        mut recognise: R,
    ) -> (f64, Array1<f64>, usize, bool)
    where
        F: FnMut(ArrayView1<f64>) -> Option<(f64, Array1<f64>)>,
        R: FnMut(usize, f64, ArrayView1<f64>) -> Option<(f64, Array1<f64>)>,
    {
        self.trim();
        let mut x = x0.to_owned();
        let mut evals = 0usize;
        let (mut f, mut g) = match fg(x.view()) {
            Some(v) => v,
            None => return (f64::INFINITY, x, evals, false),
        };
        evals += 1;

        for it in 0..max_iter {
            if let Some((f_known, x_known)) = recognise(it, f, x.view()) {
                return (f_known, x_known, evals, true);
            }
            if self.gnorm(&g) < self.gtol {
                break;
            }
            let d = self.search_direction(x.view(), g.view());
            let slope = d.dot(&g);
            if slope >= 0.0 {
                self.forget();
                continue;
            }
            let (ok, evals_used) = self.line_search(&mut x, &mut f, &mut g, &d, slope, &mut fg);
            evals += evals_used;
            if !ok {
                if self.memory.is_empty() {
                    break;
                }
                self.forget();
            }
        }
        (f, x, evals, false)
    }

    /// Cold-start L-BFGS over an eindir objective, any [`LineSearch`].
    ///
    /// Uses Euclidean `||g||_2` against `control.gtol` so
    /// [`crate::minimize_lbfgs`] keeps its existing reports.
    pub fn minimize_objective<O>(
        &mut self,
        obj: &O,
        init: impl Into<Array1<f64>>,
        control: &Control,
        linesearch: LineSearch,
    ) -> Result<Report>
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        let mut pos = init.into();
        if pos.len() != Objective::dim(obj) {
            return Err(Error::Dim {
                got: pos.len(),
                dim: Objective::dim(obj),
            });
        }
        pos = obj.bounds().clip(pos.view());
        let (mut value, mut grad) = obj.value_and_gradient(pos.view());
        let mut istep = control.istep;

        for step in 0..control.maxiter {
            let gnorm = l2(&grad);
            if gnorm < control.gtol {
                return Ok(Report {
                    value,
                    coords: pos,
                    steps: step,
                    grad_norm: gnorm,
                });
            }
            self.step_objective(
                obj, &mut pos, &mut value, &mut grad, &mut istep, linesearch, control, None,
            );
        }
        Ok(Report {
            value,
            coords: pos,
            steps: control.maxiter,
            grad_norm: l2(&grad),
        })
    }

    /// One outer L-BFGS iteration: two-loop direction, line search, pair.
    ///
    /// `atom_maxmove` caps the largest per-atom displacement of the
    /// accepted step; `None` leaves only `control.maxmove` in force.
    pub fn step_objective<O>(
        &mut self,
        obj: &O,
        pos: &mut Array1<f64>,
        value: &mut f64,
        grad: &mut Array1<f64>,
        istep: &mut f64,
        linesearch: LineSearch,
        control: &Control,
        atom_maxmove: Option<f64>,
    ) where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        self.step_objective_with_direction(obj, pos, value, grad, istep,
                                           linesearch, control, atom_maxmove, None, true);
    }

    pub(crate) fn step_objective_with_direction<O>(
        &mut self, obj: &O, pos: &mut Array1<f64>, value: &mut f64,
        grad: &mut Array1<f64>, istep: &mut f64, linesearch: LineSearch,
        control: &Control, atom_maxmove: Option<f64>, supplied_direction: Option<Array1<f64>>,
        record_pair: bool,
    ) where O: DifferentiableObjective<f64> + ?Sized,
    {
        #[cfg(feature = "highs")]
        let allow_restart = supplied_direction.is_none();
        let dir = supplied_direction.unwrap_or_else(|| self.direction(grad.view()));
        #[cfg(feature = "highs")]
        let dir = if let Some(bounds) = &self.coordinate_box {
            if !allow_restart { dir } else {
            crate::box_objective::project_direction(bounds, pos.view(), grad.view(), dir)
            }
        } else {
            dir
        };
        // With pairs the two-loop direction is gamma-scaled and carries
        // the step length, so the search opens at the unit step
        // (Nocedal-Wright 3.5); `istep` sizes only the first, steepest
        // direction. Opening at a host's `istep` of 0.1 or 0.2 made strong
        // Wolfe (c2 = 0.9) accept a tenth of the quasi-Newton step.
        let open = if self.memory.is_empty() { *istep } else { 1.0 };
        let t = take_step(
            obj,
            pos,
            *value,
            grad,
            dir.view(),
            open,
            linesearch,
            control,
            atom_maxmove,
        );
        #[cfg(feature = "highs")]
        let t = if allow_restart && !t.moved && !self.memory.is_empty() && self.coordinate_box.is_some() {
            // A related objective may have a different curvature scale.
            self.forget();
            let direction = crate::box_objective::project_direction(
                self.coordinate_box.as_ref().unwrap(), pos.view(), grad.view(),
                self.direction(grad.view()),
            );
            take_step(obj, pos, *value, grad, direction.view(), control.istep,
                      linesearch, control, atom_maxmove)
        } else {
            t
        };
        if t.moved && record_pair {
            let s = &t.x - &*pos;
            let y = &t.g - &*grad;
            self.push(s, y);
        }
        *pos = t.x;
        *value = t.f;
        *grad = t.g;
        *istep = qn_istep(control);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    #[test]
    fn an_indefinite_hessian_falls_back_to_gamma() {
        let mut l = Lbfgs::with_capacity(4);
        l.push(array![1.0, 0.0], array![2.0, 0.0]);
        let g = array![1.0, 1.0];
        let spd = Array2::from_diag(&array![2.0, 4.0]);
        let d = l.direction_with_precon(g.view(), Some(&spd));
        assert!(d.dot(&g) < 0.0);
        assert_eq!(l.precon_fallbacks(), 0);
        // diag(1, -1): -P^{-1} g = (-1, 1) is not a descent direction.
        let indefinite = Array2::from_diag(&array![1.0, -1.0]);
        let d = l.direction_with_precon(g.view(), Some(&indefinite));
        assert!(d.dot(&g) < 0.0, "ascent {d}");
        assert_eq!(l.precon_fallbacks(), 1);
        assert_eq!(d, l.direction(g.view()));
    }
}
