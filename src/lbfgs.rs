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
use crate::linesearch::{LineSearch, LineSearchOptions};
use crate::qn::solve_spd;
use crate::report::Report;
use crate::step::{energy_noise_report, l2, qn_istep, take_step_with_options};
use eindir_core::{DifferentiableObjective, Objective};

/// Consecutive line-search failures that end an L-BFGS session.
///
/// A smooth problem accepts a step long before this bound. A run whose
/// line search rejects every trial, as f32-rounded forces do, used to
/// spend the iteration budget. Zero disables the stop.
pub const DEFAULT_LINE_FAILURES: usize = 5;

/// Absolute noise on the value and the gradient for one L-BFGS session.
///
/// Both fields default to zero. The zero pair does not change the Armijo
/// test, does not lengthen a pair, and does not allocate.
///
/// Shi, Xie, Byrd, and Nocedal, SIAM J. Optim. 32, 29 (2022),
/// <https://doi.org/10.1137/20M1373190>.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NoiseModel {
    /// Noise of the objective value, `eps_f`.
    pub eps_f: f64,
    /// Noise of one gradient component, `eps_g`.
    pub eps_g: f64,
}

impl Default for NoiseModel {
    fn default() -> Self {
        Self {
            eps_f: 0.0,
            eps_g: 0.0,
        }
    }
}

impl NoiseModel {
    /// Both components zero.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record `eps_f` and `eps_g`. Non-finite values are stored as zero.
    pub fn with_noise(eps_f: f64, eps_g: f64) -> Self {
        Self {
            eps_f: if eps_f.is_finite() && eps_f > 0.0 {
                eps_f
            } else {
                0.0
            },
            eps_g: if eps_g.is_finite() && eps_g > 0.0 {
                eps_g
            } else {
                0.0
            },
        }
    }
}

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
    /// Reset direct NEB-step history when its largest atom reaches the cap.
    pub distance_reset: bool,
    /// Reset a direct NEB direction that does not face the force.
    pub angle_reset: bool,
    /// Retain a negative newest pair for a direct NEB force-step reset.
    pub curvature_reset: bool,
    /// Empty-memory scale; a non-positive or non-finite value selects one.
    pub inverse_curvature: f64,
    /// When set, each direction is the HiGHS QP on the compact Hessian
    /// rather than the two-loop recursion.
    #[cfg(feature = "highs")]
    pub highs: Option<crate::lbfgs_qp::HighsStep>,
    /// Engine and callback policies for constrained solves.
    #[cfg(feature = "highs")]
    pub highs_options: crate::HighsOptions,
    /// \(H_0 = P^{-1}\) from an assembled geometry preconditioner.
    h0: Option<crate::precon::Metric>,
    /// Value and gradient noise. Zero leaves the exact step.
    pub noise: NoiseModel,
    /// Failed line searches in a row that report [`crate::Status::StepTolerance`].
    pub max_line_failures: usize,
    line_failures: usize,
    lengthened_evals: usize,
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
            distance_reset: false,
            angle_reset: false,
            curvature_reset: false,
            inverse_curvature: 1.0,
            #[cfg(feature = "highs")]
            highs: None,
            #[cfg(feature = "highs")]
            highs_options: crate::HighsOptions::default(),
            h0: None,
            noise: NoiseModel::default(),
            max_line_failures: DEFAULT_LINE_FAILURES,
            line_failures: 0,
            lengthened_evals: 0,
        }
    }

    /// Install or clear \(H_0 = P^{-1}\).
    pub(crate) fn set_h0(&mut self, h0: Option<crate::precon::Metric>) {
        self.h0 = h0;
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

    pub(crate) fn search_direction(
        &self,
        x: ArrayView1<f64>,
        g: ArrayView1<f64>,
    ) -> Result<Array1<f64>> {
        #[cfg(feature = "highs")]
        if self.highs.is_some() {
            return self.highs_step(x, g);
        }
        let _ = x;
        Ok(self.direction(g))
    }

    /// Two-loop recursion: applies the inverse-Hessian approximation to `g`.
    ///
    /// An installed geometry preconditioner is \(H_0 = P^{-1}\).
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
        exp: Option<&crate::precon::Metric>,
    ) -> Array1<f64> {
        let mut q = g.to_owned();
        let m = self.memory.len();
        let mut alpha = vec![0.0; m];
        for k in (0..m).rev() {
            let p = &self.memory[k];
            let a = p.rho * crate::vecops::dot_reduce(p.s.view(), q.view());
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
        } else if let Some(exp) = exp {
            q = match exp.solve(q.view()) {
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
            let b = p.rho * crate::vecops::dot_reduce(p.y.view(), q.view());
            crate::vecops::axpy(alpha[k] - b, p.s.view(), &mut q);
        }
        q.mapv_inplace(|v| -v);
        q
    }

    fn scale_gamma(&self, mut q: Array1<f64>) -> Array1<f64> {
        if let Some(p) = self.memory.last() {
            let yy = crate::vecops::dot_reduce(p.y.view(), p.y.view());
            if yy > 0.0 {
                q *= crate::vecops::dot_reduce(p.s.view(), p.y.view()) / yy;
            }
        } else if self.inverse_curvature.is_finite()
            && self.inverse_curvature > 0.0
            && self.inverse_curvature != 1.0
        {
            q *= self.inverse_curvature;
        }
        q
    }

    fn current_h0(&self) -> f64 {
        if let Some(pair) = self.memory.last() {
            let yy = pair.y.dot(&pair.y);
            if yy > 0.0 {
                return pair.s.dot(&pair.y) / yy;
            }
        }
        self.inverse_curvature
    }

    fn positive_reset_scale(&self) -> f64 {
        let h0 = self.current_h0();
        if h0.is_finite() && h0 != 0.0 {
            return h0.abs();
        }
        if self.inverse_curvature.is_finite() && self.inverse_curvature > 0.0 {
            self.inverse_curvature
        } else {
            1.0
        }
    }

    /// Opt-in direct-step guards; the caller enforces the physical step cap.
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
                limit > 0.0 && crate::step::max_atom_norm(direction.view()) >= limit
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
        if self.curvature_reset {
            if !sy.is_finite() || sy.abs() < 1e-30 {
                self.forget();
                return;
            }
            self.memory.push(Pair {
                s,
                y,
                rho: 1.0 / sy,
            });
            self.trim();
            return;
        }
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
            if fa > self.armijo_bound(f0, a, slope) || (i > 0 && fa >= f_prev) {
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
                self.commit(x, f, g, d, a, fa, ga, fg);
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
            if ft > self.armijo_bound(f0, trial, slope) || ft >= f_lo {
                hi = trial;
                f_hi = ft;
                slope_hi = slope_t;
            } else {
                if slope_t.abs() <= -self.curvature * slope {
                    self.commit(x, f, g, d, trial, ft, gt, fg);
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

    /// Armijo right-hand side. A zero `eps_f` returns the exact bound.
    fn armijo_bound(&self, f0: f64, step: f64, slope: f64) -> f64 {
        let bound = f0 + self.armijo * step * slope;
        if self.noise.eps_f == 0.0 {
            bound
        } else {
            bound + 2.0 * self.noise.eps_f
        }
    }

    fn commit<F>(
        &mut self,
        x: &mut Array1<f64>,
        f: &mut f64,
        g: &mut Array1<f64>,
        d: &Array1<f64>,
        step: f64,
        f_new: f64,
        g_new: Array1<f64>,
        fg: &mut F,
    ) where
        F: FnMut(ArrayView1<f64>) -> Option<(f64, Array1<f64>)>,
    {
        let pair = if self.noise.eps_g > 0.0 {
            let mut s = d.clone();
            s *= step;
            self.lengthen_along(x, &s, g, |point| fg(point).map(|(_, grad)| grad))
        } else {
            None
        };
        self.accept(x, f, g, d, step, f_new, g_new, pair);
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
        pair: Option<(Array1<f64>, Array1<f64>)>,
    ) {
        let (s, y) = match pair {
            Some(pair) => pair,
            None => {
                let mut s = d.clone();
                s *= step;
                let mut y = g_new.clone();
                y -= &*g;
                (s, y)
            }
        };
        self.push(s, y);
        x.scaled_add(step, d);
        *f = f_new;
        *g = g_new;
    }

    /// Gradient Lipschitz estimate `||y|| / ||s||` from the newest pair.
    ///
    /// An empty memory, or a pair whose lengths are not positive and
    /// finite, uses 1. That is the curvature of the initial `H0 = I`.
    /// The noise-safe length is `l = 2 eps_g / mu_est`: the factor 2 is
    /// the constant in Shi, Xie, Byrd and Nocedal, Algorithm 1, so the
    /// true gradient change `mu * l` is twice the gradient noise.
    fn mu_est(&self) -> f64 {
        let Some(pair) = self.memory.last() else {
            return 1.0;
        };
        let sn = crate::vecops::nrm2(pair.s.view());
        let yn = crate::vecops::nrm2(pair.y.view());
        if sn > 0.0 && yn.is_finite() && yn > 0.0 {
            let mu = yn / sn;
            if mu.is_finite() && mu > 0.0 {
                return mu;
            }
        }
        1.0
    }

    /// Replace a short `(s, y)` with a probe of length `l = 2 eps_g / mu_est`.
    ///
    /// The iterate stays at the accepted step. The stored displacement is
    /// `l s / ||s||`, and `y` is the gradient difference from `x_old` out
    /// to that point. `None` means the step is already long enough, or
    /// `eps_g` is zero, in which case the caller stores the original pair
    /// and no extra gradient is evaluated.
    pub(crate) fn lengthen_along<F>(
        &mut self,
        x_old: &Array1<f64>,
        s: &Array1<f64>,
        g_old: &Array1<f64>,
        mut grad_at: F,
    ) -> Option<(Array1<f64>, Array1<f64>)>
    where
        F: FnMut(ArrayView1<f64>) -> Option<Array1<f64>>,
    {
        let eps_g = self.noise.eps_g;
        if !(eps_g.is_finite() && eps_g > 0.0) {
            return None;
        }
        let sn = crate::vecops::nrm2(s.view());
        if !(sn.is_finite() && sn > 0.0) {
            return None;
        }
        let length = 2.0 * eps_g / self.mu_est();
        if !(length.is_finite() && length > sn) {
            return None;
        }
        let scale = length / sn;
        let mut probe = x_old.clone();
        probe.scaled_add(scale, s);
        let g_long = grad_at(probe.view())?;
        self.lengthened_evals += 1;
        if g_long.len() != g_old.len() || g_long.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let s_long = s * scale;
        let y_long = &g_long - g_old;
        Some((s_long, y_long))
    }

    /// Gradients evaluated only to lengthen a short pair.
    pub fn lengthened_evals(&self) -> usize {
        self.lengthened_evals
    }

    pub(crate) fn line_search_blocked(&self) -> bool {
        self.max_line_failures > 0 && self.line_failures >= self.max_line_failures
    }

    fn note_line_outcome(&mut self, moved: bool) {
        if moved {
            self.line_failures = 0;
        } else if self.max_line_failures > 0 {
            self.line_failures = self.line_failures.saturating_add(1);
        }
    }

    /// Relaxes `x0`, calling `fg` for value and gradient.
    ///
    /// `fg` returns `None` when the caller's budget is spent, which ends the
    /// relaxation where it stands. Returns the value, the point, and the
    /// number of evaluations used. A failed feasible-direction solve ends
    /// the relaxation at the accepted point without another objective call.
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
            let Ok(d) = self.search_direction(x.view(), g.view()) else {
                break;
            };
            let slope = d.dot(&g);
            if slope >= 0.0 {
                self.forget();
                continue;
            }
            let (ok, evals_used) = self.line_search(&mut x, &mut f, &mut g, &d, slope, &mut fg);
            evals += evals_used;
            self.note_line_outcome(ok);
            if !ok {
                if self.line_search_blocked() || self.memory.is_empty() {
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
            let Ok(d) = self.search_direction(x.view(), g.view()) else {
                break;
            };
            let slope = d.dot(&g);
            if slope >= 0.0 {
                self.forget();
                continue;
            }
            let (ok, evals_used) = self.line_search(&mut x, &mut f, &mut g, &d, slope, &mut fg);
            evals += evals_used;
            self.note_line_outcome(ok);
            if !ok {
                if self.line_search_blocked() || self.memory.is_empty() {
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
            if let Some(status) = control.halt(gnorm, grad.len()) {
                return Ok(Report {
                    value,
                    coords: pos,
                    steps: step,
                    grad_norm: gnorm,
                    status,
                });
            }
            let stall = self.step_objective(
                obj, &mut pos, &mut value, &mut grad, &mut istep, linesearch, control, None,
            );
            if stall {
                return Ok(energy_noise_report(value, pos, step + 1, gnorm));
            }
            if self.line_search_blocked() {
                return Ok(Report {
                    value,
                    coords: pos,
                    steps: step + 1,
                    grad_norm: gnorm,
                    status: crate::report::Status::StepTolerance,
                });
            }
        }
        Ok(Report {
            value,
            coords: pos,
            steps: control.maxiter,
            grad_norm: l2(&grad),
            status: control.finish_status(l2(&grad), grad.len(), true),
        })
    }

    /// One outer L-BFGS iteration: two-loop direction, line search, pair.
    ///
    /// `atom_maxmove` caps the largest per-atom displacement of the
    /// accepted step; `None` leaves only `control.maxmove` in force.
    ///
    /// Returns whether every search in this iteration found no trial
    /// whose decrease cleared `c * eps_f`. The point is then unchanged.
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
    ) -> bool
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        self.step_objective_with_direction(
            obj,
            pos,
            value,
            grad,
            istep,
            linesearch,
            control,
            atom_maxmove,
            None,
            true,
            LineSearchOptions::default(),
        )
    }

    pub(crate) fn step_objective_with_direction<O>(
        &mut self,
        obj: &O,
        pos: &mut Array1<f64>,
        value: &mut f64,
        grad: &mut Array1<f64>,
        istep: &mut f64,
        linesearch: LineSearch,
        control: &Control,
        atom_maxmove: Option<f64>,
        supplied_direction: Option<Array1<f64>>,
        record_pair: bool,
        options: LineSearchOptions,
    ) -> bool
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        let allow_restart = supplied_direction.is_none();
        let options = if self.noise.eps_f > 0.0 {
            options.with_armijo_slack(2.0 * self.noise.eps_f)
        } else {
            options
        };
        let dir = supplied_direction.unwrap_or_else(|| self.direction(grad.view()));
        #[cfg(feature = "highs")]
        let dir = if let Some(bounds) = &self.coordinate_box {
            if !allow_restart {
                dir
            } else {
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
        let t = take_step_with_options(
            obj,
            pos,
            *value,
            grad,
            dir.view(),
            open,
            linesearch,
            control,
            atom_maxmove,
            options,
        );
        // A failed search can reflect a curvature scale that no longer
        // describes the objective. Retry the gradient direction once.
        // The energy floor fires only when both searches, or the only
        // search, found no trial that cleared `c * eps_f`.
        let (t, stall) = if allow_restart && !t.moved && !self.memory.is_empty() {
            let first_noise = t.energy_noise;
            self.forget();
            let direction = self.direction(grad.view());
            #[cfg(feature = "highs")]
            let direction = if let Some(bounds) = &self.coordinate_box {
                crate::box_objective::project_direction(bounds, pos.view(), grad.view(), direction)
            } else {
                direction
            };
            let retry = take_step_with_options(
                obj,
                pos,
                *value,
                grad,
                direction.view(),
                control.istep,
                linesearch,
                control,
                atom_maxmove,
                options,
            );
            let stall = first_noise && retry.energy_noise;
            (retry, stall)
        } else {
            let stall = t.energy_noise;
            (t, stall)
        };
        if t.moved && record_pair {
            let s = &t.x - &*pos;
            let y = &t.g - &*grad;
            let stored = if self.noise.eps_g > 0.0 {
                self.lengthen_along(pos, &s, grad, |point| Some(obj.value_and_gradient(point).1))
            } else {
                None
            };
            match stored {
                Some((s_long, y_long)) => self.push(s_long, y_long),
                None => self.push(s, y),
            }
        }
        self.note_line_outcome(t.moved);
        *pos = t.x;
        *value = t.f;
        *grad = t.g;
        *istep = qn_istep(control);
        stall
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    #[test]
    fn a_short_pair_is_probed_at_twice_eps_g_over_mu() {
        let mut solver = Lbfgs::default();
        solver.noise = NoiseModel::with_noise(0.0, 1.0);
        let x = array![0.0, 0.0];
        let g = array![0.1, 0.0];
        let mut calls = 0usize;
        assert!(
            solver
                .lengthen_along(&x, &array![3.0, 0.0], &g, |_| {
                    calls += 1;
                    Some(array![0.0, 0.0])
                })
                .is_none()
        );
        assert_eq!(calls, 0);
        let (s, _) = solver
            .lengthen_along(&x, &array![0.1, 0.0], &g, |point| {
                calls += 1;
                let reach = point.dot(&point).sqrt();
                assert!((reach - 2.0).abs() < 1e-12, "probe {reach}");
                Some(point.to_owned())
            })
            .unwrap();
        assert_eq!(calls, 1);
        assert!((s.dot(&s).sqrt() - 2.0).abs() < 1e-12);
        assert!(
            Lbfgs::default()
                .lengthen_along(&x, &array![0.1, 0.0], &g, |_| { None })
                .is_none()
        );
    }

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
        // One pair: s=(1,0,0), y=(1,1,0), sy=1, yy=2, H0=1/2.
        // g=(-4,0,0). The two-loop step is (6,-2,0). H0 times the
        // force is (2,0,0). The cap is below the two-loop length.
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
    fn invalid_empty_memory_scales_select_a_finite_force_step() {
        let gradient = array![-1.0, 2.0, 0.0];
        for scale in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut solver = Lbfgs {
                angle_reset: true,
                inverse_curvature: scale,
                ..Lbfgs::default()
            };
            let direction = solver.guarded_direction(gradient.view(), Some(1.0));
            assert_eq!(direction, -&gradient);
            assert!(direction.dot(&gradient) < 0.0);
        }
    }
}
