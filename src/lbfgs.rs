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
    /// ndarray `s · y` from the step that stored the pair.
    sy: f64,
    /// ndarray `y · y` from that same step. `γ = sy / yy` when `yy > 0`.
    yy: f64,
}

impl Pair {
    fn new(s: Array1<f64>, y: Array1<f64>, sy: f64, yy: f64) -> Self {
        Self {
            s,
            y,
            rho: 1.0 / sy,
            sy,
            yy,
        }
    }
}

/// Limited-memory pairs in a ring. Once the ring is full, inserting a pair
/// moves the oldest `s` and `y` onto the spare list and the next step
/// writes the new pair into those buffers. The history does not allocate
/// again while the dimension stays the same.
struct Ring {
    slots: Vec<Pair>,
    oldest: usize,
    len: usize,
    spare: Vec<Array1<f64>>,
}

enum Middle<'a> {
    Gamma,
    Precon(&'a ndarray::Array2<f64>),
    Exp(&'a crate::precon::Metric),
    ExpOrScalar(f64),
}

fn empty_pair() -> Pair {
    Pair {
        s: Array1::zeros(0),
        y: Array1::zeros(0),
        rho: 0.0,
        sy: 0.0,
        yy: 0.0,
    }
}

impl Ring {
    fn new() -> Self {
        Self {
            slots: Vec::new(),
            oldest: 0,
            len: 0,
            spare: Vec::new(),
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn get(&self, logical: usize) -> &Pair {
        let n = self.slots.len();
        &self.slots[(self.oldest + logical) % n]
    }

    fn last(&self) -> Option<&Pair> {
        if self.len == 0 {
            None
        } else {
            Some(self.get(self.len - 1))
        }
    }

    fn take_spare(&mut self, n: usize) -> Array1<f64> {
        if let Some(i) = self.spare.iter().rposition(|a| a.len() == n) {
            self.spare.swap_remove(i)
        } else {
            Array1::zeros(n)
        }
    }

    /// Two spare vectors of length `n`, or nothing when fewer than two are waiting.
    /// A full ring parks the displaced pair here, so the next step fills those
    /// buffers instead of allocating.
    fn take_pair_buffers(&mut self, n: usize) -> Option<(Array1<f64>, Array1<f64>)> {
        let mut first = None;
        let mut second = None;
        for (i, a) in self.spare.iter().enumerate() {
            if a.len() != n {
                continue;
            }
            if first.is_none() {
                first = Some(i);
            } else {
                second = Some(i);
                break;
            }
        }
        let (Some(a), Some(b)) = (first, second) else {
            return None;
        };
        let y = self.spare.swap_remove(b);
        let s = self.spare.swap_remove(a);
        Some((s, y))
    }

    fn recycle(&mut self, a: Array1<f64>) {
        if a.is_empty() {
            return;
        }
        self.spare.push(a);
        if self.spare.len() > 32 {
            let drop_n = self.spare.len() - 32;
            self.spare.drain(0..drop_n);
        }
    }

    fn recycle_pair(&mut self, pair: Pair) {
        self.recycle(pair.s);
        self.recycle(pair.y);
    }

    fn clear(&mut self) {
        let slots = std::mem::take(&mut self.slots);
        for pair in slots {
            self.recycle_pair(pair);
        }
        self.len = 0;
        self.oldest = 0;
    }

    /// Oldest-first dense prefix. A wrapped ring is not that prefix: growing
    /// it has to unwind the ring before the next slot is appended, or the
    /// new pair lands between the survivors.
    fn linearize(&mut self) {
        if self.oldest == 0 && self.slots.len() == self.len {
            return;
        }
        let n = self.slots.len();
        let mut kept = Vec::with_capacity(self.len);
        for i in 0..self.len {
            let idx = (self.oldest + i) % n;
            kept.push(std::mem::replace(&mut self.slots[idx], empty_pair()));
        }
        self.slots = kept;
        self.len = self.slots.len();
        self.oldest = 0;
    }

    /// Logical order is oldest first. While `len < cap` the slots stay dense
    /// at the front. Once full, the next insert overwrites `oldest`.
    fn insert(&mut self, cap: usize, pair: Pair) {
        let cap = cap.max(1);
        if self.len < cap {
            self.linearize();
            debug_assert_eq!(self.oldest, 0);
            debug_assert_eq!(self.slots.len(), self.len);
            self.slots.push(pair);
            self.len += 1;
            return;
        }
        let n = self.slots.len();
        let i = self.oldest;
        let old = std::mem::replace(&mut self.slots[i], pair);
        self.recycle_pair(old);
        self.oldest = (self.oldest + 1) % n;
    }

    fn trim_to(&mut self, cap: usize) {
        let cap = cap.max(1);
        if self.len <= cap {
            return;
        }
        let start = self.len - cap;
        let mut kept = Vec::with_capacity(cap);
        let n = self.slots.len();
        for i in 0..self.len {
            let idx = (self.oldest + i) % n;
            let pair = std::mem::replace(&mut self.slots[idx], empty_pair());
            if i < start {
                self.recycle_pair(pair);
            } else {
                kept.push(pair);
            }
        }
        self.slots = kept;
        self.len = self.slots.len();
        self.oldest = 0;
    }

    fn take_all(&mut self) -> Vec<Pair> {
        if self.oldest == 0 && self.slots.len() == self.len {
            self.len = 0;
            return std::mem::take(&mut self.slots);
        }
        let n = self.slots.len();
        let mut out = Vec::with_capacity(self.len);
        for i in 0..self.len {
            let idx = (self.oldest + i) % n;
            out.push(std::mem::replace(&mut self.slots[idx], empty_pair()));
        }
        self.slots.clear();
        self.len = 0;
        self.oldest = 0;
        out
    }
}

/// L-BFGS whose curvature pairs survive between calls.
///
/// A hopping chain relaxes thousands of times from perturbations of an
/// already-relaxed structure. The curvature at a new start resembles the
/// curvature at the old minimum; a solver that forgets between calls pays
/// to rediscover it.
pub struct Lbfgs {
    memory: Ring,
    /// Reused direction buffer. Empty until the first relaxation.
    q_work: Array1<f64>,
    /// Reused two-loop coefficients.
    alpha_work: Vec<f64>,
    /// Reused line-search trial vector.
    line_scratch: Array1<f64>,
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
    /// Deprecated count kept for callers that still set eOn's
    /// `lbfgs_extra_updates`.
    ///
    /// That name replayed the newest pair. The BFGS update with `(s, y)`
    /// maps any inverse Hessian that already satisfies `H y = s` to
    /// itself (`validation/lbfgs_two_loop.py`, check 3), and the replay
    /// changed bits. The two-loop does not read this field. The value is
    /// still stored, so a host can set it and read it back.
    #[deprecated(note = "stored only; the two-loop does not replay pairs")]
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
    /// `s·y` of the latest geometry step, stored even when the pair is dropped.
    pub(crate) last_sy: f64,
    /// `y·y` of the latest geometry step.
    pub(crate) last_yy: f64,
    /// [`Self::last_sy`] and [`Self::last_yy`] belong to a real step.
    pub(crate) have_curvature: bool,
    /// \(H_0 = P^{-1}\) from an assembled geometry preconditioner.
    h0: Option<crate::precon::Metric>,
}

impl Default for Lbfgs {
    fn default() -> Self {
        Self::with_capacity(8)
    }
}

/// Euclidean step and gradient change, `s = x - x_old` and `y = g - g_old`.
///
/// Each element is the same subtraction a stored difference would hold.
/// Callers that transport or project the pair do not use this.
pub(crate) fn euclidean_pair(
    x: ArrayView1<f64>,
    old: ArrayView1<f64>,
    grad: ArrayView1<f64>,
    gold: ArrayView1<f64>,
) -> (Array1<f64>, Array1<f64>) {
    let n = x.len();
    debug_assert_eq!(old.len(), n);
    debug_assert_eq!(grad.len(), n);
    debug_assert_eq!(gold.len(), n);
    let s = Array1::from_shape_fn(n, |i| x[i] - old[i]);
    let y = Array1::from_shape_fn(n, |i| grad[i] - gold[i]);
    (s, y)
}

/// `dst[i] = a[i] - b[i]`, the subtraction [`euclidean_pair`] stores.
fn write_difference(dst: &mut Array1<f64>, a: ArrayView1<f64>, b: ArrayView1<f64>) {
    let n = dst.len();
    debug_assert_eq!(a.len(), n);
    debug_assert_eq!(b.len(), n);
    if let (Some(dst), Some(a), Some(b)) = (dst.as_slice_mut(), a.as_slice(), b.as_slice()) {
        for i in 0..n {
            dst[i] = a[i] - b[i];
        }
    } else {
        for i in 0..n {
            dst[i] = a[i] - b[i];
        }
    }
}

impl Lbfgs {
    /// Fresh solver with room for `max_pairs` curvature pairs.
    ///
    /// The process that owns the positions writes them. This type
    /// returns the accepted point through [`Lbfgs::minimize`] and
    /// [`crate::Solver::step`], and it does not open a path. Curvature
    /// pairs stay in this value. A host that resumes loads the
    /// coordinates it wrote and calls that same entry point again.
    /// The new solver starts with an empty pair list, which is the
    /// resume this library ships.
    #[allow(deprecated)]
    pub fn with_capacity(max_pairs: usize) -> Self {
        Self {
            memory: Ring::new(),
            q_work: Array1::zeros(0),
            alpha_work: Vec::new(),
            line_scratch: Array1::zeros(0),
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
            last_sy: 0.0,
            last_yy: 0.0,
            have_curvature: false,
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
        self.last_sy = 0.0;
        self.last_yy = 0.0;
        self.have_curvature = false;
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
        self.memory.trim_to(cap);
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

    /// Two-loop into the solver's `q` and alpha workspace. The returned
    /// vector is that workspace; the caller gives it back via [`Self::keep_direction`].
    /// A geometry metric still goes through [`Self::direction`], which allocates.
    fn direction_workspace(&mut self, g: ArrayView1<f64>) -> Array1<f64> {
        if self.h0.is_some() {
            return self.direction(g);
        }
        if self.q_work.len() != g.len() {
            self.q_work = g.to_owned();
        } else if let (Some(q), Some(gs)) = (self.q_work.as_slice_mut(), g.as_slice()) {
            q.copy_from_slice(gs);
        } else {
            self.q_work.assign(&g);
        }
        let mut q = std::mem::take(&mut self.q_work);
        let mut alpha = std::mem::take(&mut self.alpha_work);
        let m = self.memory.len();
        self.recurse(&mut q, &mut alpha, m, m, None, Middle::Gamma);
        self.alpha_work = alpha;
        q
    }

    fn keep_direction(&mut self, q: Array1<f64>) {
        self.q_work = q;
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
        let mut alpha = Vec::new();
        let m = self.memory.len();
        let mid = if let Some(pmat) = precon {
            Middle::Precon(pmat)
        } else if let Some(exp) = exp {
            Middle::Exp(exp)
        } else {
            Middle::Gamma
        };
        self.recurse(&mut q, &mut alpha, m, m, None, mid);
        q
    }

    /// `steps` counts loop passes, oldest first, with the newest pair repeated
    /// when `steps > unique`. `first_dot` is `s · g` for that newest pair when
    /// `q` is already a copy of `g`.
    fn recurse(
        &self,
        q: &mut Array1<f64>,
        alpha: &mut Vec<f64>,
        steps: usize,
        unique: usize,
        first_dot: Option<f64>,
        mid: Middle<'_>,
    ) {
        if steps == 0 {
            self.apply_middle(q, mid);
            q.mapv_inplace(|v| -v);
            return;
        }
        if alpha.len() < steps {
            alpha.resize(steps, 0.0);
        }
        let logical = |step: usize| -> usize { if step < unique { step } else { unique - 1 } };
        let mut k = steps - 1;
        let mut a = {
            let p = self.memory.get(logical(k));
            let dot_sq = match first_dot {
                Some(v) => v,
                None => crate::vecops::dot(p.s.view(), q.view()),
            };
            p.rho * dot_sq
        };
        alpha[k] = a;
        while k > 0 {
            let cur = logical(k);
            let prev = logical(k - 1);
            let next_dot = {
                let y = self.memory.get(cur);
                let s = self.memory.get(prev);
                crate::vecops::axpy_dot(-a, y.y.view(), q, s.s.view())
            };
            k -= 1;
            let rho = self.memory.get(logical(k)).rho;
            a = rho * next_dot;
            alpha[k] = a;
        }
        {
            let y = self.memory.get(logical(0));
            crate::vecops::axpy(-a, y.y.view(), q);
        }
        let mut ydot = self.middle_ydot(q, mid);
        let mut k = 0;
        while k + 1 < steps {
            let rho = self.memory.get(logical(k)).rho;
            let coef = alpha[k] - rho * ydot;
            let cur = logical(k);
            let next = logical(k + 1);
            ydot = {
                let s = self.memory.get(cur);
                let y = self.memory.get(next);
                crate::vecops::axpy_dot(coef, s.s.view(), q, y.y.view())
            };
            k += 1;
        }
        let rho = self.memory.get(logical(k)).rho;
        let coef = alpha[k] - rho * ydot;
        {
            let s = self.memory.get(logical(k));
            crate::vecops::axpy(coef, s.s.view(), q);
        }
        q.mapv_inplace(|v| -v);
    }

    fn cached_gamma(&self) -> Option<f64> {
        let p = self.memory.last()?;
        if p.yy > 0.0 { Some(p.sy / p.yy) } else { None }
    }

    fn scale_gamma_into(&self, q: &mut Array1<f64>) {
        if let Some(p) = self.memory.last() {
            if p.yy > 0.0 {
                *q *= p.sy / p.yy;
            }
        } else if self.inverse_curvature.is_finite()
            && self.inverse_curvature > 0.0
            && self.inverse_curvature != 1.0
        {
            *q *= self.inverse_curvature;
        }
    }

    fn apply_middle(&self, q: &mut Array1<f64>, mid: Middle<'_>) {
        match mid {
            Middle::Gamma => self.scale_gamma_into(q),
            Middle::ExpOrScalar(h0) => {
                if let Some(solved) = self.exp_apply(q.view()) {
                    q.assign(&solved);
                } else {
                    *q *= h0;
                }
            }
            Middle::Precon(pmat) => match solve_spd(pmat, q) {
                Some(v) => q.assign(&v),
                None => {
                    self.precon_fallbacks
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    self.scale_gamma_into(q);
                }
            },
            Middle::Exp(exp) => match exp.solve(q.view()) {
                Some(v) => q.assign(&v),
                None => {
                    self.precon_fallbacks
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    self.scale_gamma_into(q);
                }
            },
        }
    }

    /// Middle scaling, then `y · q` of the oldest pair. A pure scale is one pass.
    fn middle_ydot(&self, q: &mut Array1<f64>, mid: Middle<'_>) -> f64 {
        match mid {
            Middle::Gamma => {
                if let Some(gamma) = self.cached_gamma() {
                    let y = self.memory.get(0);
                    crate::vecops::scale_dot(gamma, q, y.y.view())
                } else {
                    let y = self.memory.get(0);
                    crate::vecops::dot(y.y.view(), q.view())
                }
            }
            Middle::ExpOrScalar(h0) => {
                if let Some(solved) = self.exp_apply(q.view()) {
                    q.assign(&solved);
                    let y = self.memory.get(0);
                    crate::vecops::dot(y.y.view(), q.view())
                } else {
                    let y = self.memory.get(0);
                    crate::vecops::scale_dot(h0, q, y.y.view())
                }
            }
            Middle::Precon(pmat) => {
                match solve_spd(pmat, q) {
                    Some(v) => q.assign(&v),
                    None => {
                        self.precon_fallbacks
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        self.scale_gamma_into(q);
                    }
                }
                let y = self.memory.get(0);
                crate::vecops::dot(y.y.view(), q.view())
            }
            Middle::Exp(exp) => {
                match exp.solve(q.view()) {
                    Some(v) => q.assign(&v),
                    None => {
                        self.precon_fallbacks
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        self.scale_gamma_into(q);
                    }
                }
                let y = self.memory.get(0);
                crate::vecops::dot(y.y.view(), q.view())
            }
        }
    }

    fn current_h0(&self) -> f64 {
        if let Some(pair) = self.memory.last() {
            if pair.yy > 0.0 {
                return pair.sy / pair.yy;
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
        if self.curvature_reset && self.memory.last().is_some_and(|pair| pair.sy < 0.0) {
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

    /// `P^{-1} v` when a geometry metric is installed and the solve succeeds.
    pub(crate) fn exp_apply(&self, v: ArrayView1<f64>) -> Option<Array1<f64>> {
        self.h0.as_ref().and_then(|metric| metric.solve(v))
    }

    /// Two-loop with a caller-supplied initial inverse curvature `h0`.
    /// Empty memory returns `-h0 g`. An installed geometry metric
    /// replaces that scalar: the middle product is `P^{-1} q`, and a
    /// failed solve falls back to `h0`.
    pub(crate) fn direction_h0(&self, g: ArrayView1<f64>, h0: f64) -> Array1<f64> {
        let mut q = g.to_owned();
        let mut alpha = Vec::new();
        let unique = self.memory.len();
        // `extra_updates` replays the newest pair. That replay is the
        // identity on an H that already satisfies H y = s, and the public
        // two-loop ignores the field. Replaying it here changed bits.
        self.recurse(
            &mut q,
            &mut alpha,
            unique,
            unique,
            None,
            Middle::ExpOrScalar(h0),
        );
        q
    }

    /// Remember `s·y` and `y·y` from the step just taken.
    pub(crate) fn note_curvature(&mut self, s: &Array1<f64>, y: &Array1<f64>) {
        let (sy, yy, _, _) = crate::vecops::curvature_products(s.view(), y.view());
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
        let (sy, yy, _, _) = crate::vecops::curvature_products(s.view(), y.view());
        if !sy.is_finite() || sy.abs() < 1e-30 || sy <= 0.0 {
            self.memory.recycle(s);
            self.memory.recycle(y);
            return;
        }
        self.memory.insert(self.max_pairs, Pair::new(s, y, sy, yy));
        self.trim();
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
        let pairs = self.memory.take_all();
        for pair in pairs {
            let s = map(&pair.s);
            let y = map(&pair.y);
            self.memory.recycle(pair.s);
            self.memory.recycle(pair.y);
            self.push_pair(s, y, None);
        }
    }

    pub(crate) fn push(&mut self, s: Array1<f64>, y: Array1<f64>) {
        self.push_pair(s, y, None);
    }

    /// Euclidean pair `s = x - old`, `y = g - g_old`. Once the ring is full,
    /// `s` and `y` are written into the buffers the previous insert displaced.
    pub(crate) fn push_euclidean(
        &mut self,
        x: ArrayView1<f64>,
        old: ArrayView1<f64>,
        grad: ArrayView1<f64>,
        gold: ArrayView1<f64>,
        gnorm: Option<f64>,
    ) {
        let n = x.len();
        if let Some((mut s, mut y)) = self.memory.take_pair_buffers(n) {
            write_difference(&mut s, x, old);
            write_difference(&mut y, grad, gold);
            self.push_pair(s, y, gnorm);
        } else {
            let (s, y) = euclidean_pair(x, old, grad, gold);
            self.push_pair(s, y, gnorm);
        }
    }

    pub(crate) fn push_pair(&mut self, s: Array1<f64>, y: Array1<f64>, gnorm: Option<f64>) {
        let (sy, yy, ss_sum, yy_sum) = crate::vecops::curvature_products(s.view(), y.view());
        let sn = ss_sum.sqrt();
        let yn = yy_sum.sqrt();
        let ss = sn * sn;
        let reject = |this: &mut Self, s: Array1<f64>, y: Array1<f64>| {
            this.memory.recycle(s);
            this.memory.recycle(y);
        };
        if self.curvature_reset {
            if !sy.is_finite() || sy.abs() < 1e-30 {
                self.forget();
                reject(self, s, y);
                return;
            }
            self.memory.insert(self.max_pairs, Pair::new(s, y, sy, yy));
            self.trim();
            return;
        }
        if self.cautious_eps > 0.0
            && let Some(g) = gnorm
        {
            let thresh = self.cautious_eps * ss * g.max(1.0e-30).powf(self.cautious_alpha);
            if sy < thresh {
                reject(self, s, y);
                return;
            }
        }
        // Relative curvature: a tiny accepted trust step makes the
        // compact Hessian indefinite and HiGHS's QP solver does not return.
        if !sy.is_finite() || sy <= 1e-8 * sn * yn {
            reject(self, s, y);
            return;
        }
        self.memory.insert(self.max_pairs, Pair::new(s, y, sy, yy));
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
        if self.line_scratch.len() != x.len() {
            self.line_scratch = Array1::zeros(x.len());
        }
        let mut scratch = std::mem::take(&mut self.line_scratch);
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
                None => {
                    self.line_scratch = scratch;
                    return (false, evals);
                }
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
                self.line_scratch = scratch;
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
            self.line_scratch = scratch;
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
                None => {
                    self.line_scratch = scratch;
                    return (false, evals);
                }
            };
            let slope_t = d.dot(&gt);
            if ft > f0 + self.armijo * trial * slope || ft >= f_lo {
                hi = trial;
                f_hi = ft;
                slope_hi = slope_t;
            } else {
                if slope_t.abs() <= -self.curvature * slope {
                    self.line_scratch = scratch;
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
        self.line_scratch = scratch;
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
        let mut s = self.memory.take_spare(d.len());
        s.assign(d);
        s *= step;
        let mut y = self.memory.take_spare(g.len());
        y.assign(&g_new);
        y -= &*g;
        self.push(s, y);
        x.scaled_add(step, d);
        *f = f_new;
        *g = g_new;
    }

    /// Gradient norm and `d · g`. The direction is left in `q`.
    /// `None` is a failed feasible-direction solve.
    fn assess(
        &mut self,
        x: ArrayView1<f64>,
        g: &Array1<f64>,
        q: &mut Array1<f64>,
    ) -> Option<(f64, f64)> {
        if q.len() != g.len() {
            *q = Array1::zeros(g.len());
        }
        #[cfg(feature = "highs")]
        if self.highs.is_some() {
            let gn = self.gnorm(g);
            if gn < self.gtol {
                return Some((gn, 0.0));
            }
            let d = self.highs_step(x, g.view()).ok()?;
            q.assign(&d);
            return Some((gn, q.dot(g)));
        }
        let _ = x;
        let (gn, first) = self.gnorm_into(g, q);
        if gn < self.gtol {
            return Some((gn, 0.0));
        }
        let mut alpha = std::mem::take(&mut self.alpha_work);
        let m = self.memory.len();
        self.recurse(q, &mut alpha, m, m, first, Middle::Gamma);
        self.alpha_work = alpha;
        Some((gn, q.dot(g)))
    }

    fn gnorm_into(&self, g: &Array1<f64>, q: &mut Array1<f64>) -> (f64, Option<f64>) {
        let m = self.memory.len();
        if m == 0 {
            q.assign(g);
            return (self.gnorm(g), None);
        }
        match self.norm {
            GradNorm::Infinity => {
                let s = self.memory.get(m - 1);
                let (gn, dot_sg) = crate::vecops::copy_inf_dot(g.view(), q, s.s.view());
                (gn, Some(dot_sg))
            }
            GradNorm::Euclidean => {
                let s = self.memory.get(m - 1);
                let (gg, sg, finite) = crate::vecops::copy_euclid_dot(g.view(), q, s.s.view());
                let gn = if finite { gg.sqrt() } else { f64::INFINITY };
                (gn, Some(sg))
            }
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

        let mut q = std::mem::take(&mut self.q_work);
        for it in 0..max_iter {
            if !watch(it, f) {
                break;
            }
            let Some((gn, slope)) = self.assess(x.view(), &g, &mut q) else {
                break;
            };
            if gn < self.gtol {
                break;
            }
            if slope >= 0.0 {
                self.forget();
                continue;
            }
            let (ok, evals_used) = self.line_search(&mut x, &mut f, &mut g, &q, slope, &mut fg);
            evals += evals_used;
            if !ok {
                if self.memory.is_empty() {
                    break;
                }
                self.forget();
            }
        }
        self.q_work = q;
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

        let mut q = std::mem::take(&mut self.q_work);
        for it in 0..max_iter {
            if let Some((f_known, x_known)) = recognise(it, f, x.view()) {
                self.q_work = q;
                return (f_known, x_known, evals, true);
            }
            let Some((gn, slope)) = self.assess(x.view(), &g, &mut q) else {
                break;
            };
            if gn < self.gtol {
                break;
            }
            if slope >= 0.0 {
                self.forget();
                continue;
            }
            let (ok, evals_used) = self.line_search(&mut x, &mut f, &mut g, &q, slope, &mut fg);
            evals += evals_used;
            if !ok {
                if self.memory.is_empty() {
                    break;
                }
                self.forget();
            }
        }
        self.q_work = q;
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
                obj,
                &mut pos,
                &mut value,
                &mut grad,
                &mut istep,
                linesearch,
                control,
                crate::step::implicit_atom_cap(control),
            );
            if stall {
                return Ok(energy_noise_report(value, pos, step + 1, gnorm));
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
            &mut |_| {},
        )
    }

    /// `project` acts on every trial direction before its line search:
    /// the session's rigid-motion projector, or a no-op.
    #[allow(clippy::too_many_arguments)]
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
        project: &mut dyn FnMut(&mut Array1<f64>),
    ) -> bool
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        let allow_restart = supplied_direction.is_none();
        let dir = if allow_restart {
            self.direction_workspace(grad.view())
        } else {
            supplied_direction.unwrap()
        };
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
        let mut dir = dir;
        project(&mut dir);
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
            self.keep_direction(dir);
            let direction = self.direction_workspace(grad.view());
            #[cfg(feature = "highs")]
            let direction = if let Some(bounds) = &self.coordinate_box {
                crate::box_objective::project_direction(bounds, pos.view(), grad.view(), direction)
            } else {
                direction
            };
            let mut direction = direction;
            project(&mut direction);
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
            self.keep_direction(direction);
            (retry, stall)
        } else {
            let stall = t.energy_noise;
            if allow_restart {
                self.keep_direction(dir);
            }
            (t, stall)
        };
        if t.moved && record_pair {
            self.push_euclidean(t.x.view(), pos.view(), t.g.view(), grad.view(), None);
        }
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
    fn reused_pair_storage_matches_a_fresh_pair() {
        let n = 6;
        let mut reused = Lbfgs::with_capacity(3);
        let mut fresh = Lbfgs::with_capacity(3);
        for k in 0..8 {
            let scale = (k + 1) as f64;
            let x = Array1::from_shape_fn(n, |i| scale * 0.1 * (i as f64 + 1.0));
            let old = Array1::from_shape_fn(n, |i| k as f64 * 0.1 * (i as f64 + 1.0));
            let g = Array1::from_shape_fn(n, |i| -0.3 * (i as f64 + 0.5) / scale);
            let gold = Array1::from_shape_fn(n, |i| -0.3 * (i as f64 + 0.5) / (scale + 1.0));
            reused.push_euclidean(x.view(), old.view(), g.view(), gold.view(), Some(1.0));
            let (s, y) = euclidean_pair(x.view(), old.view(), g.view(), gold.view());
            fresh.push_pair(s, y, Some(1.0));
        }
        let g = Array1::from_elem(n, 0.2);
        let left = reused.direction(g.view());
        let right = fresh.direction(g.view());
        for i in 0..n {
            assert_eq!(left[i].to_bits(), right[i].to_bits());
        }
    }

    #[test]
    fn euclidean_pair_matches_the_allocated_difference() {
        fn check(x: &Array1<f64>, old: &Array1<f64>, g: &Array1<f64>, gold: &Array1<f64>) {
            let (s, y) = euclidean_pair(x.view(), old.view(), g.view(), gold.view());
            let s_old = x - old;
            let y_old = g - gold;
            assert_eq!(s.len(), s_old.len());
            for i in 0..s.len() {
                assert_eq!(s[i].to_bits(), s_old[i].to_bits(), "s {i}");
                assert_eq!(y[i].to_bits(), y_old[i].to_bits(), "y {i}");
            }
        }
        let empty = Array1::<f64>::zeros(0);
        check(&empty, &empty, &empty, &empty);
        check(
            &array![0.0, -0.0, 1.0],
            &array![-0.0, 0.0, -0.0],
            &array![f64::NAN, f64::INFINITY, -0.0],
            &array![1.0, f64::NEG_INFINITY, 0.0],
        );
        let mut state = 0x0a1b_2c3d_4e5f_6071u64;
        let mut next = || {
            state = state
                .wrapping_mul(0x6c62_272e_07bb_0142)
                .wrapping_add(0x9e37_79b9_7f4a_7c15);
            f64::from_bits(state)
        };
        for n in [1usize, 7, 8, 72] {
            let x = Array1::from_iter((0..n).map(|_| next()));
            let old = Array1::from_iter((0..n).map(|_| next()));
            let g = Array1::from_iter((0..n).map(|_| next()));
            let gold = Array1::from_iter((0..n).map(|_| next()));
            check(&x, &old, &g, &gold);
        }
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
    fn distance_reset_treats_the_cap_itself_as_too_far() {
        let g = array![-4.0, 0.0, 0.0];
        let mut solver = Lbfgs::with_capacity(4);
        solver.record(array![1.0, 0.0, 0.0], array![1.0, 1.0, 0.0]);
        let plain = solver.direction(g.view());
        let cap = crate::step::max_atom_norm(plain.view());
        assert!(cap.is_finite() && cap > 0.0, "cap {cap}");
        solver.distance_reset = true;
        let on_cap = solver.guarded_direction(g.view(), Some(cap));
        assert_eq!(solver.len(), 0, "landing on the cap must drop the pair");
        assert!(on_cap.dot(&g) < 0.0, "reset step {on_cap:?}");

        let mut under = Lbfgs::with_capacity(4);
        under.record(array![1.0, 0.0, 0.0], array![1.0, 1.0, 0.0]);
        under.distance_reset = true;
        let kept = under.guarded_direction(g.view(), Some(cap.next_up()));
        assert_eq!(
            under.len(),
            1,
            "a cap one ulp above the step keeps the pair"
        );
        for i in 0..g.len() {
            assert_eq!(kept[i].to_bits(), plain[i].to_bits(), "d[{i}]");
        }
    }

    #[test]
    fn angle_reset_on_a_zero_gradient_returns_a_zero_step() {
        let mut solver = Lbfgs::with_capacity(2);
        solver.angle_reset = true;
        solver.record(array![1.0, 0.0, 0.0], array![1.0, 0.0, 0.0]);
        let d = solver.guarded_direction(array![0.0, 0.0, 0.0].view(), Some(0.2));
        assert_eq!(solver.len(), 0);
        assert!(d.iter().all(|v| *v == 0.0), "d {d:?}");
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

#[cfg(test)]
mod ring_tests {
    use super::*;
    use ndarray::array;

    fn pair(s: [f64; 2], y: [f64; 2]) -> (Array1<f64>, Array1<f64>) {
        (array![s[0], s[1]], array![y[0], y[1]])
    }

    /// Oldest-first history, the order [`Lbfgs::two_loop`] reads.
    fn direction_of(pairs: &[(Array1<f64>, Array1<f64>)], g: &Array1<f64>) -> Array1<f64> {
        let mut solver = Lbfgs::with_capacity(pairs.len().max(1));
        for (s, y) in pairs {
            solver.record(s.clone(), y.clone());
        }
        assert_eq!(solver.len(), pairs.len());
        solver.two_loop(g.view())
    }

    #[test]
    fn wrapped_memory_matches_the_surviving_pairs_in_order() {
        let pairs = [
            pair([1.0, 0.0], [2.0, 0.0]),
            pair([0.0, 1.0], [0.0, 3.0]),
            pair([1.0, 1.0], [1.0, 2.0]),
            pair([1.0, -1.0], [2.0, -0.5]),
            pair([-1.0, 0.5], [-2.0, 1.0]),
        ];
        let g = array![-1.5, 0.75];
        let mut wrapped = Lbfgs::with_capacity(3);
        for (s, y) in &pairs {
            wrapped.record(s.clone(), y.clone());
        }
        assert_eq!(wrapped.len(), 3);
        let kept = &pairs[pairs.len() - 3..];
        let expect = direction_of(kept, &g);
        let got = wrapped.two_loop(g.view());
        for i in 0..g.len() {
            assert_eq!(got[i].to_bits(), expect[i].to_bits(), "d[{i}]");
        }
    }

    #[test]
    fn growing_after_a_wrap_keeps_chronological_order() {
        let pairs = [
            pair([1.0, 0.0], [2.0, 0.0]),
            pair([0.0, 1.0], [0.0, 3.0]),
            pair([1.0, 1.0], [1.0, 2.0]),
            pair([1.0, -1.0], [2.0, -0.5]),
        ];
        let g = array![-1.5, 0.75];
        let mut grown = Lbfgs::with_capacity(2);
        for (s, y) in &pairs[..3] {
            grown.record(s.clone(), y.clone());
        }
        assert_eq!(grown.len(), 2);
        grown.max_pairs = 4;
        grown.record(pairs[3].0.clone(), pairs[3].1.clone());
        assert_eq!(grown.len(), 3);
        // Cap 2 dropped the oldest of the first three. Growth must append
        // the fourth after those two, not insert it in the middle.
        let expect = direction_of(&pairs[1..], &g);
        let got = grown.two_loop(g.view());
        for i in 0..g.len() {
            assert_eq!(got[i].to_bits(), expect[i].to_bits(), "d[{i}]");
        }
    }

    #[allow(deprecated)]
    fn load(extra: usize) -> Lbfgs {
        let mut solver = Lbfgs::with_capacity(3);
        solver.extra_updates = extra;
        solver.record(array![1.0, 0.0, 0.0], array![2.0, 0.1, 0.0]);
        solver.record(array![0.0, 1.0, 0.0], array![0.0, 1.5, 0.2]);
        solver.record(array![0.2, -0.3, 1.0], array![0.4, 0.1, 1.2]);
        solver
    }

    #[test]
    #[allow(deprecated)]
    fn extra_updates_do_not_change_the_geometry_direction() {
        let g = array![-1.5, 0.75, 0.25];
        let plain = load(0);
        let replay = load(5);
        let h0 = 0.37;
        let a = plain.direction_h0(g.view(), h0);
        let b = replay.direction_h0(g.view(), h0);
        for i in 0..g.len() {
            assert_eq!(a[i].to_bits(), b[i].to_bits(), "h0 d[{i}]");
        }
        let left = plain.two_loop(g.view());
        let right = replay.two_loop(g.view());
        for i in 0..g.len() {
            assert_eq!(left[i].to_bits(), right[i].to_bits(), "two-loop d[{i}]");
        }
        assert_eq!(replay.extra_updates, 5);
    }
}
