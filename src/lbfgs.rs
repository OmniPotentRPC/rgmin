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
use crate::qn::solve_dense;
use crate::report::Report;
use crate::step::{l2, next_istep, take_step};
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
    /// Al-Baali extra-updates: replay the newest pair this many extra times.
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
        if self.highs.is_some() {
            if let Ok(d) = self.highs_step(x, g) {
                return d;
            }
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
        let mut idxs: Vec<usize> = (0..m).collect();
        for _ in 0..self.extra_updates {
            if m > 0 {
                idxs.push(m - 1);
            }
        }
        let mut alpha = vec![0.0; idxs.len()];
        for k in (0..idxs.len()).rev() {
            let i = idxs[k];
            let p = &self.memory[i];
            let a = p.rho * crate::vecops::dot(p.s.view(), q.view());
            alpha[k] = a;
            crate::vecops::axpy(-a, p.y.view(), &mut q);
        }
        if let Some(pmat) = precon {
            q = solve_dense(pmat, &q).unwrap_or_else(|| self.scale_gamma(q));
        } else {
            q = self.scale_gamma(q);
        }
        for k in 0..idxs.len() {
            let i = idxs[k];
            let p = &self.memory[i];
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

    pub(crate) fn push(&mut self, s: Array1<f64>, y: Array1<f64>) {
        self.push_pair(s, y, None);
    }

    /// Drop the newest pair, then push. Used after a manifold retract.
    pub(crate) fn replace_newest(&mut self, s: Array1<f64>, y: Array1<f64>, gnorm: Option<f64>) {
        let _ = self.memory.pop();
        self.push_pair(s, y, gnorm);
    }

    pub(crate) fn push_pair(&mut self, s: Array1<f64>, y: Array1<f64>, gnorm: Option<f64>) {
        let sy = s.dot(&y);
        let sn = s.iter().map(|v| v * v).sum::<f64>().sqrt();
        let yn = y.iter().map(|v| v * v).sum::<f64>().sqrt();
        let ss = sn * sn;
        if self.cautious_eps > 0.0 {
            if let Some(g) = gnorm {
                let thresh = self.cautious_eps * ss * g.max(1.0e-30).powf(self.cautious_alpha);
                if sy < thresh {
                    return;
                }
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
                obj, &mut pos, &mut value, &mut grad, &mut istep, linesearch, control,
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
    pub fn step_objective<O>(
        &mut self,
        obj: &O,
        pos: &mut Array1<f64>,
        value: &mut f64,
        grad: &mut Array1<f64>,
        istep: &mut f64,
        linesearch: LineSearch,
        control: &Control,
    ) where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        let dir = self.direction(grad.view());
        let old = pos.clone();
        let gold = grad.clone();
        let (npos, _, lsstep, moved) =
            take_step(obj, pos, *value, dir.view(), *istep, linesearch, control);
        *pos = npos;
        let ev = obj.value_and_gradient(pos.view());
        *value = ev.0;
        *grad = ev.1;
        if moved {
            self.push(&*pos - &old, &*grad - &gold);
        }
        *istep = next_istep(lsstep, control);
    }
}
