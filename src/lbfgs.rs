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

use crate::control::{Control, StopNorm};
use crate::error::{Error, Result};
use crate::linesearch::LineSearch;
use crate::qn::solve_dense;
use crate::report::Report;
use crate::step::{finite, l2, next_istep, start_answer, take_step};
use eindir_core::{DifferentiableObjective, Objective};

/// How [`Lbfgs`] compares the gradient to [`Lbfgs::gtol`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GradNorm {
    /// Euclidean `||g||_2`. Matches [`crate::minimize_lbfgs`].
    Euclidean,
    /// Infinity `||g||_∞`. Matches SciPy L-BFGS-B and hopping polish.
    Infinity,
}

/// `factr` that the L-BFGS-B 3.0 `setulb` comments give for moderate
/// accuracy (`lbfgsb.f`, Morales and Nocedal 2011,
/// <https://doi.org/10.1145/2049662.2049669>).
pub const LBFGSB_FACTR: f64 = 1.0e7;

/// The relative-decrease tolerance `factr * eps` of L-BFGS-B, SciPy's
/// default `ftol` for `method="L-BFGS-B"`, about 2.22e-9.
pub const LBFGSB_FTOL: f64 = LBFGSB_FACTR * f64::EPSILON;

/// The value and slope of a line-search probe, `(+inf, NaN)` when the
/// objective answered with a non-finite value or gradient. Every comparison
/// against NaN is false, so a raw NaN would fail no decrease test.
fn probed(f: f64, d: &Array1<f64>, g: &Array1<f64>) -> (f64, f64) {
    if finite(f, g.view()) {
        (f, d.dot(g))
    } else {
        (f64::INFINITY, f64::NAN)
    }
}

/// Stored curvature pair from one accepted step.
#[derive(Clone)]
struct Pair {
    s: Array1<f64>,
    y: Array1<f64>,
    rho: f64,
}

/// Curvature saved by [`Lbfgs::curvature`].
pub(crate) struct Curvature {
    memory: Vec<Pair>,
    last: (f64, f64, bool),
    stored: bool,
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
    /// Relative-decrease stop of the L-BFGS-B code (Zhu, Byrd, Lu and
    /// Nocedal, Algorithm 778, <https://doi.org/10.1145/279232.279236>).
    /// A line-search step from `f_k`
    /// to `f_{k+1}` ends [`Lbfgs::minimize`], [`Lbfgs::minimize_watched`]
    /// and [`Lbfgs::minimize_recognized`] at the new iterate when
    /// `f_k - f_{k+1} <= ftol * max(|f_k|, |f_{k+1}|, 1)`.
    /// Zero, the default, turns the test off, as does any value that is
    /// not positive. [`LBFGSB_FTOL`] is the SciPy default.
    pub ftol: f64,
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
    /// \(H_0 = P^{-1}\) from an assembled exponential preconditioner.
    /// A dense matrix passed to [`Self::direction_with_precon`] is the
    /// pair / Lindh hook and wins over this factor.
    h0: Option<crate::precon::Exp>,
    /// Sparse Lindh metric used as \(H_0 = P^{-1}\) when `h0` is empty.
    lindh: Option<crate::precon::Metric>,
    /// Sparse bonded metric used as \(H_0 = P^{-1}\) when `h0` and Lindh are empty.
    ff: Option<crate::precon::Metric>,
    /// Forget when the two-loop step is longer than the atom cap.
    pub distance_reset: bool,
    /// Forget when the two-loop step is not a descent direction.
    pub angle_reset: bool,
    /// Forget when the newest pair has negative curvature.
    pub curvature_reset: bool,
    /// Empty-memory scale. One leaves the two-loop unchanged.
    pub inverse_curvature: f64,
    /// `s·y` of the latest geometry step, stored even when the pair is dropped.
    pub(crate) last_sy: f64,
    /// `y·y` of the latest geometry step.
    pub(crate) last_yy: f64,
    /// [`Self::last_sy`] and [`Self::last_yy`] belong to a real step.
    pub(crate) have_curvature: bool,
    /// The latest [`Self::push_pair`] stored its pair.
    stored: bool,
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
            ftol: 0.0,
            extra_updates: 0,
            cautious_eps: 0.0,
            cautious_alpha: 0.01,
            #[cfg(feature = "highs")]
            highs: None,
            h0: None,
            lindh: None,
            ff: None,
            distance_reset: false,
            angle_reset: false,
            curvature_reset: false,
            inverse_curvature: 1.0,
            last_sy: 0.0,
            last_yy: 0.0,
            have_curvature: false,
            stored: false,
        }
    }

    /// Install or clear the exponential \(H_0 = P^{-1}\).
    pub(crate) fn set_h0(&mut self, h0: Option<crate::precon::Exp>) {
        self.h0 = h0;
    }

    /// Install or clear the Lindh \(H_0 = P^{-1}\).
    pub(crate) fn set_lindh_metric(&mut self, metric: Option<crate::precon::Metric>) {
        self.lindh = metric;
    }

    /// Install or clear the bonded \(H_0 = P^{-1}\).
    pub(crate) fn set_ff_metric(&mut self, metric: Option<crate::precon::Metric>) {
        self.ff = metric;
    }

    /// `P^{-1} v` from the Lindh metric, or the bonded metric when Lindh is absent.
    pub(crate) fn model_apply(&self, v: ArrayView1<f64>) -> Option<Array1<f64>> {
        if let Some(metric) = &self.lindh {
            return metric.solve(v);
        }
        self.ff.as_ref().and_then(|metric| metric.solve(v))
    }

    /// Discards the stored curvature.
    ///
    /// Called when the chain moves somewhere structurally different, where
    /// the retained pairs describe a Hessian that no longer applies.
    pub fn forget(&mut self) {
        self.memory.clear();
        self.stored = false;
        self.last_sy = 0.0;
        self.last_yy = 0.0;
        self.have_curvature = false;
    }

    /// The pairs and latest geometry curvature, for [`Self::restore`].
    pub(crate) fn curvature(&self) -> Curvature {
        Curvature {
            memory: self.memory.clone(),
            last: (self.last_sy, self.last_yy, self.have_curvature),
            stored: self.stored,
        }
    }

    /// Puts back what [`Self::curvature`] saved.
    pub(crate) fn restore(&mut self, saved: Curvature) {
        self.memory = saved.memory;
        (self.last_sy, self.last_yy, self.have_curvature) = saved.last;
        self.stored = saved.stored;
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
    ///
    /// An installed exponential preconditioner is \(H_0 = P^{-1}\).
    pub(crate) fn direction(&self, g: ArrayView1<f64>) -> Array1<f64> {
        self.direction_with_sources(g, None, self.h0.as_ref())
    }

    /// Two-loop with optional \(H_0 = P^{-1}\). `precon` is the pair / Lindh
    /// matrix \(P\); the middle product is `solve(P, q)`, not a scalar γ.
    /// A dense `precon` wins over an installed exponential matrix.
    pub(crate) fn direction_with_precon(
        &self,
        g: ArrayView1<f64>,
        precon: Option<&ndarray::Array2<f64>>,
    ) -> Array1<f64> {
        self.direction_with_sources(g, precon, None)
    }

    fn direction_with_sources(
        &self,
        g: ArrayView1<f64>,
        precon: Option<&ndarray::Array2<f64>>,
        exp: Option<&crate::precon::Exp>,
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
        } else if let Some(exp) = exp {
            q = exp.solve(q.view()).unwrap_or_else(|| self.scale_gamma(q));
        } else if let Some(solved) = self.model_apply(q.view()) {
            q = solved;
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

    /// `P^{-1} v` when an exponential preconditioner is installed and the solve succeeds.
    pub(crate) fn exp_apply(&self, v: ArrayView1<f64>) -> Option<Array1<f64>> {
        self.h0.as_ref().and_then(|exp| exp.solve(v))
    }

    /// Two-loop with a caller-supplied initial inverse curvature `h0`.
    /// Empty memory returns `-h0 g`. An installed exponential
    /// preconditioner replaces that scalar: the middle product is
    /// `P^{-1} q`, and a failed solve falls back to `h0`.
    pub(crate) fn direction_h0(&self, g: ArrayView1<f64>, h0: f64) -> Array1<f64> {
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
        if let Some(solved) = self.exp_apply(q.view()) {
            q = solved;
        } else if let Some(solved) = self.model_apply(q.view()) {
            q = solved;
        } else {
            q *= h0;
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

    /// Remember `s·y` and `y·y` from the step just taken.
    pub(crate) fn note_curvature(&mut self, s: &Array1<f64>, y: &Array1<f64>) {
        let sy = s.dot(y);
        let yy = y.dot(y);
        if sy.is_finite() && yy.is_finite() {
            self.last_sy = sy;
            self.last_yy = yy;
            self.have_curvature = true;
        } else {
            self.have_curvature = false;
        }
    }

    /// Store a geometry-step pair. Negative or vanishing curvature is dropped
    /// here; the caller resets memory from [`Self::note_curvature`].
    pub(crate) fn push_geometry(&mut self, s: Array1<f64>, y: Array1<f64>) {
        let sy = s.dot(&y);
        if !sy.is_finite() || sy.abs() < 1e-30 || sy <= 0.0 {
            return;
        }
        self.memory.push(Pair {
            s,
            y,
            rho: 1.0 / sy,
        });
        self.trim();
    }

    fn scale_gamma(&self, mut q: Array1<f64>) -> Array1<f64> {
        if let Some(p) = self.memory.last() {
            let yy = p.y.dot(&p.y);
            if yy > 0.0 {
                q *= p.s.dot(&p.y) / yy;
            }
        } else if self.inverse_curvature.is_finite()
            && self.inverse_curvature > 0.0
            && self.inverse_curvature != 1.0
        {
            q *= self.inverse_curvature;
        }
        q
    }

    fn positive_reset_scale(&self) -> f64 {
        if let Some(pair) = self.memory.last() {
            let yy = pair.y.dot(&pair.y);
            if yy > 0.0 {
                let h0 = pair.s.dot(&pair.y) / yy;
                if h0.is_finite() && h0 != 0.0 {
                    return h0.abs();
                }
            }
        }
        if self.inverse_curvature.is_finite() && self.inverse_curvature > 0.0 {
            self.inverse_curvature
        } else {
            1.0
        }
    }

    /// Opt-in step guards. The caller still applies the displacement cap.
    /// Flags that are off leave [`Self::direction`] as the step.
    pub(crate) fn guarded_direction(
        &mut self,
        gradient: ArrayView1<f64>,
        cap: Option<f64>,
    ) -> Array1<f64> {
        if self.curvature_reset
            && self
                .memory
                .last()
                .is_some_and(|pair| pair.s.dot(&pair.y) < 0.0)
        {
            self.forget();
            return gradient.mapv(|value| -1000.0 * value);
        }
        let scale = self.positive_reset_scale();
        let direction = self.direction(gradient);
        let too_far = self.distance_reset
            && cap.is_some_and(|limit| {
                limit > 0.0 && crate::step::max_atom_displacement(&direction) >= limit
            });
        let slope = direction.dot(&gradient);
        let wrong_angle = self.angle_reset && (!slope.is_finite() || slope >= 0.0);
        let invalid = (self.distance_reset || self.angle_reset)
            && direction.iter().any(|value| !value.is_finite());
        if too_far || wrong_angle || invalid {
            self.forget();
            return gradient.mapv(|value| -scale * value);
        }
        direction
    }

    /// Records an accepted curvature pair (`s = x+ - x`, `y = g+ - g`).
    pub fn record(&mut self, s: Array1<f64>, y: Array1<f64>) {
        self.push(s, y);
    }

    /// Two-loop direction `d = −H g` (Nocedal-Wright 7.4).
    pub fn two_loop(&self, g: ArrayView1<f64>) -> Array1<f64> {
        self.direction(g)
    }

    /// Move every retained pair to the tangent space of the next iterate.
    pub(crate) fn transport<F>(&mut self, mut map: F)
    where
        F: FnMut(&Array1<f64>) -> Array1<f64>,
    {
        let pairs = std::mem::take(&mut self.memory);
        let stored = self.stored;
        let mut kept = false;
        for pair in pairs {
            kept = self.push_pair(map(&pair.s), map(&pair.y), None);
        }
        // A newest pair the map dropped is no longer there to replace.
        self.stored = stored && kept;
    }

    /// [`Self::transport`] with a map that can fail. Every pair is mapped
    /// before any is replaced, so a failure leaves the memory as it was.
    pub(crate) fn try_transport<F, E>(&mut self, mut map: F) -> std::result::Result<(), E>
    where
        F: FnMut(&Array1<f64>) -> std::result::Result<Array1<f64>, E>,
    {
        let moved = self
            .memory
            .iter()
            .map(|pair| Ok((map(&pair.s)?, map(&pair.y)?)))
            .collect::<std::result::Result<Vec<_>, E>>()?;
        let stored = self.stored;
        self.memory.clear();
        let mut kept = false;
        for (s, y) in moved {
            kept = self.push_pair(s, y, None);
        }
        self.stored = stored && kept;
        Ok(())
    }

    /// Stored pairs, oldest first.
    #[cfg(test)]
    pub(crate) fn pairs(&self) -> impl Iterator<Item = (&Array1<f64>, &Array1<f64>)> {
        self.memory.iter().map(|p| (&p.s, &p.y))
    }

    pub(crate) fn push(&mut self, s: Array1<f64>, y: Array1<f64>) -> bool {
        self.push_pair(s, y, None)
    }

    /// Replace the newest pair after a manifold retract. Only a pair the
    /// latest push stored is popped; a dropped pair leaves the previous
    /// one in place.
    pub(crate) fn replace_newest(&mut self, s: Array1<f64>, y: Array1<f64>, gnorm: Option<f64>) {
        if self.stored {
            let _ = self.memory.pop();
        }
        self.push_pair(s, y, gnorm);
    }

    /// Store `(s, y)` unless the cautious or relative-curvature test
    /// drops it. Returns whether it was stored.
    pub(crate) fn push_pair(&mut self, s: Array1<f64>, y: Array1<f64>, gnorm: Option<f64>) -> bool {
        let sy = s.dot(&y);
        let sn = s.iter().map(|v| v * v).sum::<f64>().sqrt();
        let yn = y.iter().map(|v| v * v).sum::<f64>().sqrt();
        let ss = sn * sn;
        if self.cautious_eps > 0.0
            && let Some(g) = gnorm
        {
            let thresh = self.cautious_eps * ss * g.max(1.0e-30).powf(self.cautious_alpha);
            if sy < thresh {
                self.stored = false;
                return false;
            }
        }
        // Relative curvature: a tiny accepted trust step makes the
        // compact Hessian indefinite and HiGHS's QP solver does not return.
        // A negative pair is kept only while the curvature guard is on,
        // so the guard can see it. The guard is off unless the caller
        // asks, and then this test is the one it was.
        let keep_negative = self.curvature_reset && sy.is_finite() && sy < 0.0;
        if (!sy.is_finite() || sy <= 1e-8 * sn * yn) && !keep_negative {
            self.stored = false;
            return false;
        }
        self.memory.push(Pair {
            s,
            y,
            rho: 1.0 / sy,
        });
        self.trim();
        self.stored = true;
        true
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

    /// The step from `f_old` to `f_new` passes the [`Lbfgs::ftol`] test
    /// (`lbfgsb.f`, `mainlb`).
    fn stalled(&self, f_old: f64, f_new: f64) -> bool {
        self.ftol > 0.0 && f_old - f_new <= self.ftol * f_old.abs().max(f_new.abs()).max(1.0)
    }

    /// Strong Wolfe line search by bracketing then cubic-interpolated zoom.
    ///
    /// Nocedal and Wright algorithms 3.5 and 3.6. Returns whether a step
    /// was accepted and how many evaluations it cost. A probe answered with
    /// a non-finite value or gradient fails the decrease test, so it caps
    /// the bracket and the zoom bisects back toward the accepted end.
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
            let (fa, slope_a) = probed(fa, d, &ga);
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
                self.accept(x, f, g, d, a, (fa, ga));
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
            let (ft, slope_t) = probed(ft, d, &gt);
            if ft > f0 + self.armijo * trial * slope || ft >= f_lo {
                hi = trial;
                f_hi = ft;
                slope_hi = slope_t;
            } else {
                if slope_t.abs() <= -self.curvature * slope {
                    self.accept(x, f, g, d, trial, (ft, gt));
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
        new: (f64, Array1<f64>),
    ) {
        let (f_new, g_new) = new;
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
    /// number of evaluations used. A non-finite value or gradient at a
    /// trial step shortens the step; at `x0` it ends the relaxation there
    /// with value `+inf`, as a spent budget does.
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
        if !finite(f, g.view()) {
            return (f64::INFINITY, x, evals);
        }

        let mut stalled = false;
        for it in 0..max_iter {
            if !watch(it, f) {
                break;
            }
            if stalled || self.gnorm(&g) < self.gtol {
                break;
            }
            let f_old = f;
            let d = self.search_direction(x.view(), g.view());
            let slope = d.dot(&g);
            if !slope.is_finite() && self.memory.is_empty() {
                break;
            }
            if slope >= 0.0 || !slope.is_finite() {
                self.forget();
                continue;
            }
            let (ok, evals_used) = self.line_search(&mut x, &mut f, &mut g, &d, slope, &mut fg);
            evals += evals_used;
            stalled = ok && self.stalled(f_old, f);
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
        if !finite(f, g.view()) {
            return (f64::INFINITY, x, evals, false);
        }

        let mut stalled = false;
        for it in 0..max_iter {
            if let Some((f_known, x_known)) = recognise(it, f, x.view()) {
                return (f_known, x_known, evals, true);
            }
            if stalled || self.gnorm(&g) < self.gtol {
                break;
            }
            let f_old = f;
            let d = self.search_direction(x.view(), g.view());
            let slope = d.dot(&g);
            if !slope.is_finite() && self.memory.is_empty() {
                break;
            }
            if slope >= 0.0 || !slope.is_finite() {
                self.forget();
                continue;
            }
            let (ok, evals_used) = self.line_search(&mut x, &mut f, &mut g, &d, slope, &mut fg);
            evals += evals_used;
            stalled = ok && self.stalled(f_old, f);
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
        let (mut value, mut grad) = start_answer(obj, &pos)?;
        let mut istep = control.istep;

        for step in 0..control.maxiter {
            let gnorm = l2(&grad);
            if gnorm < control.gtol {
                return Ok(Report {
                    value,
                    coords: pos,
                    steps: step,
                    grad_norm: gnorm,
                    stop_norm: StopNorm::Two,
                    grad: None,
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
            stop_norm: StopNorm::Two,
            grad: None,
        })
    }

    /// One outer L-BFGS iteration: two-loop direction, line search, pair.
    #[expect(
        clippy::too_many_arguments,
        reason = "public signature; grouping the arguments would break callers"
    )]
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
        let dir = self.search_direction(pos.view(), grad.view());
        let old = pos.clone();
        let gold = grad.clone();
        // Curvature history supplies the quasi-Newton step length. Wolfe
        // starts at the unit step so accepted steps retain superlinear
        // convergence instead of inheriting repeated line-search shrinkage.
        let trial_step = if self.memory.is_empty() { *istep } else { 1.0 };
        let (mut npos, _, mut lsstep, mut moved) = take_step(
            obj,
            pos,
            *value,
            dir.view(),
            trial_step,
            linesearch,
            control,
        );
        if !moved && !self.memory.is_empty() {
            // Retained curvature can describe a different objective scale.
            // Retry with an empty history before reporting a stalled step.
            self.forget();
            let restart_direction = self.search_direction(pos.view(), grad.view());
            let restart = take_step(
                obj,
                pos,
                *value,
                restart_direction.view(),
                control.istep,
                linesearch,
                control,
            );
            npos = restart.0;
            lsstep = restart.2;
            moved = restart.3;
        }
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

#[cfg(test)]
mod tests {
    use ndarray::{Array1, array};

    use super::{LBFGSB_FTOL, Lbfgs};

    /// Above `|f| = 1` the threshold scales with `|f|`; below, it is
    /// `ftol` itself. A value that is not positive passes no step.
    #[test]
    fn the_decrease_test_is_relative_above_one_and_absolute_below() {
        let mut lbfgs = Lbfgs {
            ftol: LBFGSB_FTOL,
            ..Lbfgs::default()
        };
        let big = 1.0 / LBFGSB_FTOL;
        assert!(lbfgs.stalled(big, big - 0.5));
        assert!(!lbfgs.stalled(big, big - 2.0));
        let small = LBFGSB_FTOL;
        assert!(lbfgs.stalled(small, 0.5 * small));
        assert!(!lbfgs.stalled(small, -small));
        for ftol in [0.0, -LBFGSB_FTOL, f64::NAN] {
            lbfgs.ftol = ftol;
            assert!(!lbfgs.stalled(1.0, 1.0), "ftol {ftol}");
        }
    }

    fn held(lbfgs: &Lbfgs) -> Vec<(Array1<f64>, Array1<f64>)> {
        lbfgs.pairs().map(|(s, y)| (s.clone(), y.clone())).collect()
    }

    /// A transport that drops the newest pair leaves no pair of the latest
    /// push, so the next replacement keeps the older pair.
    #[test]
    fn a_transport_that_drops_the_newest_pair_keeps_the_older_one() {
        let mut lbfgs = Lbfgs::with_capacity(4);
        assert!(lbfgs.push(array![1.0, 0.0], array![2.0, 0.0]));
        assert!(lbfgs.push(array![1.0, 1.0], array![-1.0, 3.0]));
        // On the first coordinate alone the newer pair has s.y = -1.
        lbfgs.transport(|v| array![v[0], 0.0]);
        let older = (array![1.0, 0.0], array![2.0, 0.0]);
        assert_eq!(held(&lbfgs), vec![older.clone()]);
        lbfgs.replace_newest(array![0.0, 1.0], array![0.0, 3.0], None);
        assert_eq!(
            held(&lbfgs),
            vec![older, (array![0.0, 1.0], array![0.0, 3.0])]
        );
    }

    /// A map that fails on the second pair, after the first has moved,
    /// leaves both pairs as they were.
    #[test]
    fn a_failed_transport_keeps_every_pair() {
        let mut lbfgs = Lbfgs::with_capacity(4);
        assert!(lbfgs.push(array![1.0, 0.0], array![2.0, 0.0]));
        assert!(lbfgs.push(array![0.0, 1.0], array![0.0, 3.0]));
        let before = held(&lbfgs);
        let mut calls = 0;
        let moved = lbfgs.try_transport(|v| {
            calls += 1;
            if calls < 3 { Ok(v * 0.5) } else { Err(calls) }
        });
        assert_eq!(moved, Err(3));
        assert_eq!(held(&lbfgs), before);
        assert!(lbfgs.stored);
    }
    /// A fallible transport that succeeds but drops the newest pair also
    /// leaves the older pair to the next replacement.
    #[test]
    fn a_fallible_transport_that_drops_the_newest_pair_keeps_the_older_one() {
        let mut lbfgs = Lbfgs::with_capacity(4);
        assert!(lbfgs.push(array![1.0, 0.0], array![2.0, 0.0]));
        assert!(lbfgs.push(array![1.0, 1.0], array![-1.0, 3.0]));
        let moved: Result<(), ()> = lbfgs.try_transport(|v| Ok(array![v[0], 0.0]));
        assert_eq!(moved, Ok(()));
        let older = (array![1.0, 0.0], array![2.0, 0.0]);
        assert_eq!(held(&lbfgs), vec![older.clone()]);
        lbfgs.replace_newest(array![0.0, 1.0], array![0.0, 3.0], None);
        assert_eq!(
            held(&lbfgs),
            vec![older, (array![0.0, 1.0], array![0.0, 3.0])]
        );
    }
}

#[cfg(test)]
mod neb_guard_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn empty_memory_scales_by_inverse_curvature() {
        let mut solver = Lbfgs::with_capacity(4);
        let g = array![-2.0, 0.0, 0.0];
        let plain = solver.direction(g.view());
        assert!((plain[0] - 2.0).abs() < 1e-15);
        solver.inverse_curvature = 0.01;
        let scaled = solver.direction(g.view());
        assert!((scaled[0] - 0.02).abs() < 1e-15);
    }

    #[test]
    fn distance_reset_forgets_and_steps_along_h0_force() {
        let mut solver = Lbfgs::with_capacity(4);
        solver.distance_reset = true;
        solver.record(array![1.0, 0.0, 0.0], array![1.0, 1.0, 0.0]);
        let g = array![-4.0, 0.0, 0.0];
        let d = solver.guarded_direction(g.view(), Some(0.2));
        assert_eq!(solver.len(), 0);
        assert!((d[0] - 2.0).abs() < 1e-12, "d0 {d:?}");
        assert!(d[1].abs() < 1e-12, "d1 {d:?}");
        assert!(d[2].abs() < 1e-12);
    }

    #[test]
    fn negative_curvature_takes_the_force_direction_and_forgets() {
        let mut solver = Lbfgs::with_capacity(4);
        solver.curvature_reset = true;
        solver.record(array![1.0, 0.0, 0.0], array![-0.5, 0.0, 0.0]);
        assert_eq!(solver.len(), 1);
        let g = array![-1.0, 0.0, 0.0];
        let d = solver.guarded_direction(g.view(), Some(0.2));
        assert_eq!(solver.len(), 0);
        assert!((d[0] - 1000.0).abs() < 1e-9, "d {d:?}");
        assert!(d[1].abs() < 1e-12);
    }

    #[test]
    fn angle_reset_drops_a_step_that_faces_away_from_the_force() {
        let mut kept = Lbfgs::with_capacity(4);
        kept.curvature_reset = true;
        kept.record(array![1.0, 0.0, 0.0], array![-0.5, 0.0, 0.0]);
        kept.curvature_reset = false;
        kept.distance_reset = false;
        kept.angle_reset = false;
        kept.inverse_curvature = 0.01;
        let g = array![-1.0, 0.0, 0.0];
        let d_keep = kept.guarded_direction(g.view(), Some(10.0));
        assert_eq!(kept.len(), 1);
        assert!((d_keep[0] + 2.0).abs() < 1e-12, "keep {d_keep:?}");

        let mut solver = Lbfgs::with_capacity(4);
        solver.curvature_reset = true;
        solver.record(array![1.0, 0.0, 0.0], array![-0.5, 0.0, 0.0]);
        solver.curvature_reset = false;
        solver.distance_reset = false;
        solver.angle_reset = true;
        solver.inverse_curvature = 0.01;
        let d = solver.guarded_direction(g.view(), Some(10.0));
        assert_eq!(solver.len(), 0);
        assert_eq!(d, array![2.0, 0.0, 0.0]);
        assert!(d.dot(&g) < 0.0);
    }

    #[test]
    fn guards_off_keep_the_two_loop_step() {
        let mut solver = Lbfgs::with_capacity(4);
        solver.record(array![1.0, 0.0, 0.0], array![1.0, 1.0, 0.0]);
        let g = array![-4.0, 0.0, 0.0];
        let plain = solver.direction(g.view());
        let guarded = solver.guarded_direction(g.view(), Some(0.2));
        assert_eq!(solver.len(), 1);
        assert_eq!(plain, guarded);
    }
}
