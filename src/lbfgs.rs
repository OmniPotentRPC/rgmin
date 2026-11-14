//! Persistent limited-memory BFGS (Nocedal-Wright 7.4, scaling 7.20).
//!
//! The default step is the line-search-free cap of the eOn client
//! ([`LbfgsStep::Capped`], [`EON_MAX_MOVE`], [`Lbfgs::default`]). The
//! first step estimates curvature with a finite difference of length
//! [`EON_FINITE_DIFFERENCE`] and sets `H0 = 1/C`. Later steps use
//! `(s·y)/(y·y)` when that quotient is positive, and
//! [`EON_INVERSE_CURVATURE`] otherwise. A proposal whose longest atom
//! move is at least the cap clears the memory and steps along `H0`
//! times the force (`distance_reset`). A direction that is not descent
//! does the same (`angle_reset`). Negative curvature clears the memory
//! and steps along the force at the cap. The cap only shortens a step.
//! [`LbfgsStep::Scaled`] is the ASE length control: shorten the two-loop
//! step and keep the pairs. [`LbfgsStep::Wolfe`] is the strong Wolfe
//! search, and it is opt-in.
//! [`LbfgsPairs`] and [`lbfgs_direction`] are the same two-loop with a
//! caller-supplied `H0`. They return `-H g` and do not evaluate an
//! objective.
//!
//! eOn client `LBFGS::getStep` and `ParametersSSOTDefaults.h`,
//! <https://github.com/TheochemUI/eOn>.
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
use crate::linesearch::{LineSearch, LineSearchOptions};
use crate::qn::solve_spd;
use crate::report::Report;
use crate::step::{
    cap_displacement_atom, finite, l2, max_atom_displacement, max_atom_norm, qn_istep,
    scale_displacement_to_atom_cap, take_step_with_options,
};
use eindir_core::{DifferentiableObjective, Objective};

mod cauchy;
mod direction;

pub use direction::{LbfgsH0, LbfgsPairs, lbfgs_direction};

/// eOn `[Optimizer] max_move`: the longest displacement of one atom in
/// one step, in the coordinate length. The client default is 0.2.
///
/// eOn client, <https://github.com/TheochemUI/eOn>.
pub const EON_MAX_MOVE: f64 = 0.2;

/// eOn `lbfgs_inverse_curvature`. The scalar `H0` before a positive
/// curvature pair, and the fallback when auto-scaling has no quotient.
pub const EON_INVERSE_CURVATURE: f64 = 0.01;

/// eOn `finite_difference`. The first capped step probes curvature
/// over a displacement of this length along the force.
pub const EON_FINITE_DIFFERENCE: f64 = 0.01;

/// eOn multiplies the force by this before the negative-curvature cap,
/// so the cap binds and the step follows the force.
const EON_STEEP_SCALE: f64 = 1000.0;

/// How [`Lbfgs`] chooses the length of a step.
///
/// [`LbfgsStep::Capped`] is the default: the eOn step, no line search.
/// [`LbfgsStep::Wolfe`] is the strong Wolfe search, and it is opt-in.
/// [`LbfgsStep::Scaled`] is the ASE length control.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LbfgsStep {
    /// Strong Wolfe line search. Nocedal and Wright, algorithms 3.5 and 3.6.
    /// Opt in with this variant. It is not the default.
    Wolfe,
    /// eOn `LBFGS::getStep` with the client's default switches:
    /// `auto_scale`, `distance_reset`, `angle_reset`, curvature `reset`.
    /// No line search. See the module docs.
    Capped {
        /// Longest displacement of one atom, in the coordinate length.
        max_atom: f64,
    },
    /// ASE `LBFGS` length control. No line search. Shorten the two-loop
    /// step so the longest per-atom displacement is at most `max_atom`,
    /// and keep the pairs. An empty memory follows the force and the
    /// step has that length. A direction that is not descent clears the
    /// memory. This is not the eOn step and not the default.
    Scaled {
        /// Longest displacement of one atom, in the coordinate length.
        max_atom: f64,
    },
}

impl Default for LbfgsStep {
    fn default() -> Self {
        Self::Capped {
            max_atom: EON_MAX_MOVE,
        }
    }
}

/// A positive finite cap, or the strong Wolfe search when `max_atom`
/// is not a length.
fn free_or_wolfe(capped: bool, max_atom: f64) -> LbfgsStep {
    if max_atom.is_finite() && max_atom > 0.0 {
        if capped {
            LbfgsStep::Capped { max_atom }
        } else {
            LbfgsStep::Scaled { max_atom }
        }
    } else {
        LbfgsStep::Wolfe
    }
}

/// Validated [`Lbfgs`] configuration.
///
/// [`Lbfgs::default`] is the eOn cap. [`Self::wolfe`] is the strong
/// Wolfe search. A non-positive or non-finite cap passed to
/// [`Self::capped`] or [`Self::scaled`] selects Wolfe, the same rule
/// as [`Lbfgs::capped`].
pub struct LbfgsBuilder {
    inner: Lbfgs,
}

impl LbfgsBuilder {
    /// Pairs retained. Zero is raised to one.
    #[must_use]
    pub fn memory(mut self, pairs: usize) -> Self {
        self.inner.max_pairs = pairs.max(1);
        self
    }

    /// Gradient-norm threshold.
    #[must_use]
    pub fn tolerance(mut self, gtol: f64) -> Self {
        self.inner.gtol = gtol;
        self
    }

    /// Norm compared with [`Self::tolerance`].
    #[must_use]
    pub fn norm(mut self, norm: GradNorm) -> Self {
        self.inner.norm = norm;
        self
    }

    /// Relative-decrease stop. A non-positive value turns it off.
    #[must_use]
    pub fn ftol(mut self, ftol: f64) -> Self {
        self.inner.ftol = ftol;
        self
    }

    /// Step-length rule. An unusable cap becomes [`LbfgsStep::Wolfe`].
    #[must_use]
    pub fn step(mut self, step: LbfgsStep) -> Self {
        self.inner.step = match step {
            LbfgsStep::Capped { max_atom } => free_or_wolfe(true, max_atom),
            LbfgsStep::Scaled { max_atom } => free_or_wolfe(false, max_atom),
            LbfgsStep::Wolfe => LbfgsStep::Wolfe,
        };
        self
    }

    /// Strong Wolfe search.
    #[must_use]
    pub fn wolfe(self) -> Self {
        self.step(LbfgsStep::Wolfe)
    }

    /// eOn cap. A bad `max_atom` selects Wolfe.
    #[must_use]
    pub fn capped(self, max_atom: f64) -> Self {
        self.step(LbfgsStep::Capped { max_atom })
    }

    /// ASE length control. A bad `max_atom` selects Wolfe.
    #[must_use]
    pub fn scaled(self, max_atom: f64) -> Self {
        self.step(LbfgsStep::Scaled { max_atom })
    }

    /// Fixed initial inverse-Hessian scale. A non-positive value clears it.
    #[must_use]
    pub fn fixed_scale(mut self, h0: f64) -> Self {
        self.inner.fixed_h0 = (h0.is_finite() && h0 > 0.0).then_some(h0);
        self
    }

    /// Measure the cap by the Euclidean length of the whole step.
    #[must_use]
    pub fn euclidean_cap(mut self, on: bool) -> Self {
        self.inner.euclidean_cap = on;
        self
    }

    /// Keep a long downhill model step and scale it per atom.
    #[must_use]
    pub fn model_atom(mut self, on: bool) -> Self {
        self.inner.model_atom = on;
        self
    }

    /// Replace a long two-loop step with the compact trust-region step.
    #[must_use]
    pub fn compact_trust(mut self, on: bool) -> Self {
        self.inner.compact_tr = on;
        self
    }

    /// Drop a pair with `s·y ≤ 0` instead of clearing the memory.
    #[must_use]
    pub fn skip_negative_curvature(mut self, on: bool) -> Self {
        self.inner.skip_bad = on;
        self
    }

    /// The configured solver.
    #[must_use]
    pub fn build(self) -> Lbfgs {
        self.inner
    }
}

impl LbfgsStep {
    /// Positive finite cap of a line-search-free rule.
    #[must_use]
    pub fn capped_limit(self) -> Option<f64> {
        match self {
            Self::Capped { max_atom } | Self::Scaled { max_atom }
                if max_atom.is_finite() && max_atom > 0.0 =>
            {
                Some(max_atom)
            }
            _ => None,
        }
    }
}

/// Scalar seed for one eOn step, or the negative-curvature fallback.
#[derive(Clone, Copy, Debug)]
pub(crate) enum EonSeed {
    /// Two-loop with this `H0`. A long or uphill proposal becomes `H0 f`.
    Scale(f64),
    /// Clear the memory and step along the force at the cap.
    Steepest,
}

/// Displacement proposed by [`Lbfgs::eon_step_from_seed`].
pub(crate) struct EonTake {
    pub step: Array1<f64>,
    /// The proposal is already the force fallback, so a non-finite
    /// trial is not retried along that same direction.
    pub fallback: bool,
}

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

/// Undo record for one eOn step. A successful step drops the pairs a
/// reset removed. A failed trial puts them back. Neither path clones
/// the history.
pub(crate) struct Txn {
    sy: f64,
    yy: f64,
    have: bool,
    stored: bool,
    /// Memory replaced by a reset, as it was at that reset.
    parked: Option<Vec<Pair>>,
    /// Front pairs removed before that reset, oldest first.
    pre_trimmed: Vec<Pair>,
    pre_pushed: usize,
    /// Front pairs removed since the reset, or since the step began.
    trimmed: Vec<Pair>,
    pushed: usize,
}

fn max_atom_slice(step: &[f64]) -> f64 {
    let mut max_atom = 0.0;
    let mut i = 0;
    let n = step.len();
    while i + 3 <= n {
        let r2 = step[i] * step[i] + step[i + 1] * step[i + 1] + step[i + 2] * step[i + 2];
        if r2 > max_atom * max_atom {
            max_atom = r2.sqrt();
        }
        i += 3;
    }
    if i < n {
        let mut r2 = 0.0;
        while i < n {
            r2 += step[i] * step[i];
            i += 1;
        }
        max_atom = max_atom.max(r2.sqrt());
    }
    max_atom
}

/// Two-loop into `q`, using the same dot and axpy as [`Lbfgs::direction_h0`].
#[allow(clippy::needless_range_loop)]
fn two_loop_vecops(
    memory: &[Pair],
    extra: usize,
    q: &mut Array1<f64>,
    h0: f64,
    lin: Option<&crate::lindh::LindhFactor>,
) {
    let m = memory.len();
    let total = if m == 0 { 0 } else { m + extra };
    let mut stack = [0.0; 64];
    let mut heap = Vec::new();
    let alpha: &mut [f64] = if total <= stack.len() {
        &mut stack[..total]
    } else {
        heap.resize(total, 0.0);
        heap.as_mut_slice()
    };
    for k in (0..total).rev() {
        let i = if k < m { k } else { m - 1 };
        let p = &memory[i];
        let a = p.rho * crate::vecops::dot(p.s.view(), q.view());
        alpha[k] = a;
        crate::vecops::axpy(-a, p.y.view(), q);
    }
    if let Some(lin) = lin {
        if !lin.solve_into(q) {
            *q *= h0;
        }
    } else {
        *q *= h0;
    }
    for k in 0..total {
        let i = if k < m { k } else { m - 1 };
        let p = &memory[i];
        let b = p.rho * crate::vecops::dot(p.y.view(), q.view());
        crate::vecops::axpy(alpha[k] - b, p.s.view(), q);
    }
    q.mapv_inplace(|v| -v);
}

/// The value and slope of a line-search probe, `(+inf, NaN)` when the
/// objective answered with a non-finite value or gradient, or with a
/// gradient whose length is not the direction's. Every comparison
/// against NaN is false, so a raw NaN would fail no decrease test, and
/// a short gradient must not index off the end of the direction.
/// A lower entry must be finite or `-inf`, an upper entry finite or
/// `+inf`, and the lower entry must not sit above the upper one.
/// A NaN fails every comparison, so it is not a bound.
fn bounds_ok(n: usize, lower: ArrayView1<f64>, upper: ArrayView1<f64>) -> bool {
    lower.len() == n
        && upper.len() == n
        && lower.iter().zip(upper.iter()).all(|(lo, hi)| {
            (lo.is_finite() || *lo == f64::NEG_INFINITY)
                && (hi.is_finite() || *hi == f64::INFINITY)
                && matches!(
                    lo.partial_cmp(hi),
                    Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
                )
        })
}

fn probed(f: f64, d: &Array1<f64>, g: &Array1<f64>) -> (f64, f64) {
    if g.len() == d.len() && finite(f, g.view()) {
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
    /// `s · y` at the moment the pair was stored.
    sy: f64,
    /// `y · y` at the moment the pair was stored.
    yy: f64,
}

/// `(s, y)` when the relative-curvature gate accepts it.
///
/// `sy <= 1e-8 ||s|| ||y||`, a non-finite `s · y`, and a non-positive
/// product are dropped. The stored `rho` is `1 / (s · y)`.
fn stored_pair(s: Array1<f64>, y: Array1<f64>) -> Option<Pair> {
    let sy = s.dot(&y);
    let sn = s.iter().map(|v| v * v).sum::<f64>().sqrt();
    let yn = y.iter().map(|v| v * v).sum::<f64>().sqrt();
    if !sy.is_finite() || sy <= 1e-8 * sn * yn {
        return None;
    }
    let yy = y.dot(&y);
    Some(Pair {
        s,
        y,
        rho: 1.0 / sy,
        sy,
        yy,
    })
}

/// The line along which [`Lbfgs::line_search`] steps: direction `d`,
/// slope `d·g` at the start, and the largest step `a_max`.
struct Ray<'a> {
    d: &'a Array1<f64>,
    slope: f64,
    a_max: f64,
}

/// Curvature saved by [`Lbfgs::curvature`].
pub(crate) struct Curvature {
    memory: Vec<Pair>,
    last: (f64, f64, bool),
    stored: bool,
}

enum FreeRule {
    Eon(f64),
    Scaled(f64),
}

enum CappedIter {
    Moved { stalled: bool, evals: usize },
    Stop { evals: usize },
}

enum CappedProbe {
    Moved { stalled: bool },
    NonFinite,
    Stop,
}

/// Installed \(H_0 = P^{-1}\). An exponential factor and a geometry
/// metric share the two-loop middle product.
enum InitialInverse {
    #[allow(dead_code)]
    Exp(crate::precon::Exp),
    Metric(crate::precon::Metric),
}

impl InitialInverse {
    fn solve(&self, rhs: ArrayView1<'_, f64>) -> Option<Array1<f64>> {
        match self {
            Self::Exp(exp) => exp.solve(rhs),
            Self::Metric(metric) => metric.solve(rhs),
        }
    }
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
    /// to `f_{k+1}` ends [`Lbfgs::minimize`], [`Lbfgs::minimize_watched`],
    /// [`Lbfgs::minimize_recognized`], [`Lbfgs::minimize_box`] and
    /// [`Lbfgs::minimize_box_watched`] at the new iterate when
    /// `f_k - f_{k+1} <= ftol * max(|f_k|, |f_{k+1}|, 1)`.
    /// Zero, the default, turns the test off, as does any value that is
    /// not positive. [`LBFGSB_FTOL`] is the SciPy default.
    pub ftol: f64,
    /// Step-length rule. The default is [`LbfgsStep::Capped`] at
    /// [`EON_MAX_MOVE`], the eOn step. [`LbfgsStep::Wolfe`] opts into
    /// the line search. [`LbfgsStep::Scaled`] is the ASE length control.
    pub step: LbfgsStep,
    /// Fixed initial inverse-Hessian scale. `None`, the default, uses
    /// the Nocedal-Wright scale `(s · y) / (y · y)`. `Some(1/70)` is the
    /// ASE LBFGS scale, which is not updated.
    pub fixed_h0: Option<f64>,
    /// Al-Baali extra-updates: replay the newest pair this many extra times.
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
    /// `1`, the default, leaves an empty two-loop unscaled.
    pub inverse_curvature: f64,
    /// When set, each direction is the HiGHS QP on the compact Hessian
    /// rather than the two-loop recursion.
    #[cfg(feature = "highs")]
    pub highs: Option<crate::lbfgs_qp::HighsStep>,
    /// Coordinate box copied in for a constrained two-loop projection.
    #[cfg(feature = "highs")]
    pub(crate) coordinate_box: Option<eindir_core::Bounds<f64>>,
    /// Engine and callback policies for constrained solves.
    #[cfg(feature = "highs")]
    pub highs_options: crate::HighsOptions,
    /// Times a supplied \(H_0^{-1}\) was not SPD and the two-loop fell
    /// back to `γ I`.
    precon_fallbacks: std::sync::atomic::AtomicUsize,
    /// \(H_0 = P^{-1}\) from an assembled geometry preconditioner.
    /// A dense matrix passed to [`Self::direction_with_precon`] wins.
    h0: Option<InitialInverse>,
    /// `s·y` of the latest geometry step, stored even when the pair is dropped.
    pub(crate) last_sy: f64,
    /// `y·y` of the latest geometry step.
    pub(crate) last_yy: f64,
    /// [`Self::last_sy`] and [`Self::last_yy`] belong to a real step.
    pub(crate) have_curvature: bool,
    /// The latest [`Self::push_pair`] stored its pair.
    stored: bool,
    /// Reused residual of the in-place two-loop.
    work: Array1<f64>,
    /// Curvature vectors waiting to be written again.
    pool: Vec<Array1<f64>>,
    /// The last [`Self::forget_txn`] cleared the pairs.
    pub(crate) forgot: bool,
    /// Cap the whole displacement by its Euclidean length.
    /// The default measures the longest atom.
    pub(crate) euclidean_cap: bool,
    /// A long quasi-Newton step is scaled and kept. The default clears
    /// the pairs and steps along the force.
    pub(crate) keep_long: bool,
    /// A long downhill quasi-Newton step is scaled to the per-atom cap
    /// and the pairs are cleared. The direction stays the model step.
    /// The default replaces that step with `H0` times the force.
    pub(crate) model_atom: bool,
    /// Replace a long two-loop step with the compact trust-region step.
    /// Off by default, so the eOn cap is unchanged.
    pub(crate) compact_tr: bool,
    /// Drop a pair with `s·y ≤ 0` and keep the older pairs. The default
    /// stores it and the next step clears the memory.
    pub(crate) skip_bad: bool,
    /// Lindh model Hessian. `None` keeps the scalar initial scale.
    pub(crate) lindh: Option<crate::lindh::LindhFactor>,
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
            step: LbfgsStep::default(),
            fixed_h0: None,
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
            coordinate_box: None,
            #[cfg(feature = "highs")]
            highs_options: crate::HighsOptions::default(),
            precon_fallbacks: std::sync::atomic::AtomicUsize::new(0),
            h0: None,
            last_sy: 0.0,
            last_yy: 0.0,
            have_curvature: false,
            stored: false,
            work: Array1::zeros(0),
            pool: Vec::new(),
            forgot: false,
            euclidean_cap: false,
            keep_long: false,
            model_atom: false,
            compact_tr: false,
            skip_bad: false,
            lindh: None,
        }
    }

    /// [`LbfgsStep::Capped`] with room for eight pairs. A non-positive
    /// or non-finite `max_atom` selects [`LbfgsStep::Wolfe`].
    #[must_use]
    pub fn capped(max_atom: f64) -> Self {
        Self {
            step: free_or_wolfe(true, max_atom),
            ..Self::default()
        }
    }

    /// [`LbfgsStep::Scaled`] with room for eight pairs. A non-positive
    /// or non-finite `max_atom` selects [`LbfgsStep::Wolfe`].
    #[must_use]
    pub fn scaled(max_atom: f64) -> Self {
        Self {
            step: free_or_wolfe(false, max_atom),
            ..Self::default()
        }
    }

    /// Validated configuration. The default is the eOn cap.
    /// [`LbfgsBuilder::wolfe`] opts into the strong Wolfe search.
    #[must_use]
    pub fn builder() -> LbfgsBuilder {
        LbfgsBuilder {
            inner: Self::default(),
        }
    }

    /// Install or clear a geometry-metric \(H_0 = P^{-1}\).
    pub(crate) fn set_h0(&mut self, h0: Option<crate::precon::Metric>) {
        self.h0 = h0.map(InitialInverse::Metric);
    }

    /// Install or clear an exponential \(H_0 = P^{-1}\).
    #[allow(dead_code)]
    pub(crate) fn set_exp_h0(&mut self, h0: Option<crate::precon::Exp>) {
        self.h0 = h0.map(InitialInverse::Exp);
    }

    /// Times a supplied `H0^{-1}` was not symmetric positive definite and
    /// the two-loop fell back to `gamma I`.
    #[must_use]
    pub fn precon_fallbacks(&self) -> usize {
        self.precon_fallbacks
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Install or clear the Lindh factor.
    pub(crate) fn set_lindh_factor(&mut self, factor: Option<crate::lindh::LindhFactor>) {
        self.lindh = factor;
    }

    /// Discards the stored curvature.
    ///
    /// Called when the chain moves somewhere structurally different, where
    /// the retained pairs describe a Hessian that no longer applies.
    pub fn forget(&mut self) {
        self.recycle_all();
        self.stored = false;
        self.last_sy = 0.0;
        self.last_yy = 0.0;
        self.have_curvature = false;
    }

    fn recycle_vec(&mut self, v: Array1<f64>) {
        if !v.is_empty() && self.pool.len() < self.max_pairs.saturating_mul(4).max(8) {
            self.pool.push(v);
        }
    }

    fn recycle_pair(&mut self, pair: Pair) {
        self.recycle_vec(pair.s);
        self.recycle_vec(pair.y);
    }

    fn recycle_all(&mut self) {
        let memory = std::mem::take(&mut self.memory);
        for pair in memory {
            self.recycle_pair(pair);
        }
    }

    fn acquire(&mut self, n: usize) -> Array1<f64> {
        while let Some(v) = self.pool.pop() {
            if v.len() == n {
                return v;
            }
        }
        Array1::zeros(n)
    }

    fn ensure_work(&mut self, n: usize) {
        if self.work.len() != n {
            self.work = self.acquire(n);
        }
    }

    pub(crate) fn begin_txn(&mut self) -> Txn {
        self.forgot = false;
        Txn {
            sy: self.last_sy,
            yy: self.last_yy,
            have: self.have_curvature,
            stored: self.stored,
            parked: None,
            pre_trimmed: Vec::new(),
            pre_pushed: 0,
            trimmed: Vec::new(),
            pushed: 0,
        }
    }

    /// Clear the pairs but keep them in `txn` until commit or rollback.
    pub(crate) fn forget_txn(&mut self, txn: &mut Txn) {
        self.forgot = true;
        if txn.parked.is_none() {
            txn.pre_trimmed = std::mem::take(&mut txn.trimmed);
            txn.pre_pushed = txn.pushed;
            txn.pushed = 0;
            txn.parked = Some(std::mem::take(&mut self.memory));
        } else {
            self.recycle_all();
            for pair in txn.trimmed.drain(..) {
                self.recycle_pair(pair);
            }
            txn.pushed = 0;
        }
        self.stored = false;
        self.last_sy = 0.0;
        self.last_yy = 0.0;
        self.have_curvature = false;
    }

    fn push_buf_txn(&mut self, s: Array1<f64>, y: Array1<f64>, sy: f64, yy: f64, txn: &mut Txn) {
        let cap = self.max_pairs.max(1);
        self.max_pairs = cap;
        while self.memory.len() >= cap {
            txn.trimmed.push(self.memory.remove(0));
        }
        self.memory.push(Pair {
            s,
            y,
            rho: 1.0 / sy,
            sy,
            yy,
        });
        txn.pushed += 1;
        self.stored = true;
    }

    pub(crate) fn commit_txn(&mut self, txn: Txn) {
        for pair in txn.trimmed {
            self.recycle_pair(pair);
        }
        for pair in txn.pre_trimmed {
            self.recycle_pair(pair);
        }
        if let Some(memory) = txn.parked {
            for pair in memory {
                self.recycle_pair(pair);
            }
        }
    }

    pub(crate) fn rollback_txn(&mut self, txn: Txn) {
        if txn.parked.is_some() {
            self.recycle_all();
            for pair in txn.trimmed {
                self.recycle_pair(pair);
            }
            self.memory = txn.parked.unwrap_or_default();
            for _ in 0..txn.pre_pushed {
                if let Some(pair) = self.memory.pop() {
                    self.recycle_pair(pair);
                }
            }
            for pair in txn.pre_trimmed.into_iter().rev() {
                self.memory.insert(0, pair);
            }
        } else {
            for _ in 0..txn.pushed {
                if let Some(pair) = self.memory.pop() {
                    self.recycle_pair(pair);
                }
            }
            for pair in txn.trimmed.into_iter().rev() {
                self.memory.insert(0, pair);
            }
        }
        self.last_sy = txn.sy;
        self.last_yy = txn.yy;
        self.have_curvature = txn.have;
        self.stored = txn.stored;
    }

    /// Copy `s` and `y = g_new - g_old` into a recycled slot.
    fn store_diff_txn(&mut self, s: &[f64], g_new: &[f64], g_old: &[f64], txn: &mut Txn) {
        let n = s.len();
        let mut sv = self.acquire(n);
        let mut yv = self.acquire(n);
        for i in 0..n {
            sv[i] = s[i];
            yv[i] = g_new[i] - g_old[i];
        }
        let sy = sv.dot(&yv);
        let yy = yv.dot(&yv);
        self.finish_eon_pair(sv, yv, sy, yy, txn);
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
        installed: Option<&InitialInverse>,
    ) -> Array1<f64> {
        self.direction_dense(g, precon, installed)
    }

    fn direction_dense(
        &self,
        g: ArrayView1<f64>,
        precon: Option<&ndarray::Array2<f64>>,
        installed: Option<&InitialInverse>,
    ) -> Array1<f64> {
        let (mut q, alpha, idxs) = direction::backward(&self.memory, self.extra_updates, g);
        if let Some(pmat) = precon {
            // H0 = P^{-1} keeps -H g a descent direction only for an SPD
            // P; an indefinite Hessian falls back to gamma I and is counted.
            q = match solve_spd(pmat, &q) {
                Some(v) => v,
                None => {
                    self.note_precon_fallback();
                    self.scale_gamma(q)
                }
            };
        } else if let Some(installed) = installed {
            q = match installed.solve(q.view()) {
                Some(v) => v,
                None => {
                    self.note_precon_fallback();
                    self.scale_gamma(q)
                }
            };
        } else {
            q = self.scale_gamma(q);
        }
        direction::forward(&self.memory, q, &alpha, &idxs)
    }

    fn note_precon_fallback(&self) {
        self.precon_fallbacks
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
        let (mut q, alpha, idxs) = direction::backward(&self.memory, self.extra_updates, g);
        if let Some(solved) = self.exp_apply(q.view()) {
            q = solved;
        } else {
            q *= h0;
        }
        direction::forward(&self.memory, q, &alpha, &idxs)
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
        let yy = y.dot(&y);
        self.memory.push(Pair {
            s,
            y,
            rho: 1.0 / sy,
            sy,
            yy,
        });
        self.trim();
    }

    fn fixed_scale(&self) -> Option<f64> {
        self.fixed_h0.filter(|h| h.is_finite() && *h > 0.0)
    }

    fn scale_gamma(&self, mut q: Array1<f64>) -> Array1<f64> {
        if let Some(h0) = self.fixed_scale() {
            q *= h0;
            return q;
        }
        if let Some(p) = self.memory.last() {
            if p.yy > 0.0 {
                q *= p.sy / p.yy;
            }
            return q;
        }
        if self.inverse_curvature.is_finite()
            && self.inverse_curvature > 0.0
            && self.inverse_curvature != 1.0
        {
            q *= self.inverse_curvature;
        }
        q
    }

    fn current_h0(&self) -> f64 {
        if let Some(h0) = self.fixed_scale() {
            return h0;
        }
        if let Some(pair) = self.memory.last()
            && pair.yy > 0.0
        {
            return pair.sy / pair.yy;
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
            && cap.is_some_and(|limit| limit > 0.0 && max_atom_norm(direction.view()) >= limit);
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
    #[allow(dead_code)]
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
        let ss = sn * sn;
        if self.curvature_reset {
            if !sy.is_finite() || sy.abs() < 1e-30 {
                self.forget();
                self.stored = false;
                return false;
            }
            let yy = y.dot(&y);
            self.memory.push(Pair {
                s,
                y,
                rho: 1.0 / sy,
                sy,
                yy,
            });
            self.trim();
            self.stored = true;
            return true;
        }
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
        // [`stored_pair`] repeats the products and is the store [`LbfgsPairs`] uses.
        let Some(pair) = stored_pair(s, y) else {
            self.stored = false;
            return false;
        };
        self.memory.push(pair);
        self.trim();
        self.stored = true;
        true
    }

    /// Positive finite cap of a line-search-free rule, if one is selected.
    pub(crate) fn capped_limit(&self) -> Option<f64> {
        self.step.capped_limit()
    }

    /// Line-search-free rule used by [`Self::minimize`]. An installed
    /// HiGHS step keeps the line search: the QP already enforces its
    /// trust region and box.
    fn line_search_free(&self) -> Option<FreeRule> {
        #[cfg(feature = "highs")]
        if self.highs.is_some() {
            return None;
        }
        let cap = self.capped_limit()?;
        match self.step {
            LbfgsStep::Capped { .. } => Some(FreeRule::Eon(cap)),
            LbfgsStep::Scaled { .. } => Some(FreeRule::Scaled(cap)),
            LbfgsStep::Wolfe => None,
        }
    }

    /// The first eOn step has no pair and no fixed scale, so it spends
    /// one finite-difference probe. A preconditioner supplies `H0`.
    pub(crate) fn eon_wants_fd(&self) -> bool {
        self.fixed_scale().is_none()
            && self.h0.is_none()
            && self.lindh.is_none()
            && !self.have_curvature
            && self.memory.is_empty()
    }

    /// `H0` for the next eOn step, before a finite-difference probe.
    pub(crate) fn eon_seed(&self) -> EonSeed {
        let fallback = self.fixed_scale().unwrap_or(EON_INVERSE_CURVATURE);
        if self.have_curvature {
            let sy = self.last_sy;
            let yy = self.last_yy;
            let curvature = if sy.is_finite() && yy.is_finite() && sy != 0.0 {
                yy / sy
            } else {
                -1.0
            };
            if curvature < 0.0 {
                return EonSeed::Steepest;
            }
            if let Some(fixed) = self.fixed_scale() {
                return EonSeed::Scale(fixed);
            }
            if sy > 0.0 && yy > 0.0 {
                return EonSeed::Scale(sy / yy);
            }
            return EonSeed::Scale(EON_INVERSE_CURVATURE);
        }
        if let Some(fixed) = self.fixed_scale() {
            return EonSeed::Scale(fixed);
        }
        if let Some(pair) = self.memory.last() {
            let h0 = if pair.yy > 0.0 {
                pair.sy / pair.yy
            } else {
                -1.0
            };
            if h0 > 0.0 && h0.is_finite() {
                return EonSeed::Scale(h0);
            }
            return EonSeed::Steepest;
        }
        EonSeed::Scale(fallback)
    }

    /// Displacement of one eOn step. `None` when that displacement is zero.
    /// A long or uphill quasi-Newton direction clears the memory first.
    /// The displacement is copied out of the reused workspace.
    pub(crate) fn eon_step_from_seed(
        &mut self,
        g: ArrayView1<f64>,
        max_atom: f64,
        seed: EonSeed,
        txn: &mut Txn,
    ) -> Option<EonTake> {
        let fallback = self.prepare_eon(g, max_atom, seed, txn)?;
        let step = self.copy_work();
        let motion = max_atom_displacement(&step);
        (motion.is_finite() && motion > 0.0).then_some(EonTake { step, fallback })
    }

    /// Write the eOn displacement into [`Self::work`]. `Some(fallback)`.
    fn prepare_eon(
        &mut self,
        g: ArrayView1<f64>,
        max_atom: f64,
        seed: EonSeed,
        txn: &mut Txn,
    ) -> Option<bool> {
        let n = g.len();
        self.ensure_work(n);
        match seed {
            EonSeed::Steepest => {
                self.forget_txn(txn);
                let force = g.mapv(|v| -v);
                let step = self.apply_cap(&(EON_STEEP_SCALE * &force), max_atom);
                self.work.assign(&step);
                Some(true)
            }
            EonSeed::Scale(h0) => {
                if !(h0.is_finite() && h0 > 0.0 && max_atom.is_finite() && max_atom > 0.0) {
                    return None;
                }
                if self.compact_tr && !self.memory.is_empty() && self.install_compact(g, max_atom) {
                    return Some(false);
                }
                let force = g.mapv(|v| -v);
                self.write_qn(g, h0);
                let distance = self.motion(&self.work);
                let along = self.work.dot(&force);
                let uphill = l2(&self.work) > 0.0 && l2(&force) > 0.0 && along < 0.0;
                if !distance.is_finite() {
                    return None;
                }
                let atom = max_atom_displacement(&self.work);
                if self.model_atom && !uphill && atom >= max_atom {
                    self.forget_txn(txn);
                    if atom > max_atom && atom > 0.0 {
                        self.work *= max_atom / atom;
                    }
                    return Some(false);
                }
                if uphill || (distance >= max_atom && !self.keep_long) {
                    self.forget_txn(txn);
                    let step = self.apply_cap(&(h0 * &force), max_atom);
                    self.work.assign(&step);
                    Some(true)
                } else {
                    self.cap_work(max_atom);
                    Some(false)
                }
            }
        }
    }

    fn write_qn(&mut self, g: ArrayView1<f64>, h0: f64) {
        if self.h0.is_some() && self.lindh.is_none() {
            let direction = self.direction_h0(g, h0);
            self.work.assign(&direction);
            return;
        }
        let memory = std::mem::take(&mut self.memory);
        let lindh = self.lindh.take();
        let extra = self.extra_updates;
        self.work.assign(&g);
        two_loop_vecops(&memory, extra, &mut self.work, h0, lindh.as_ref());
        self.memory = memory;
        self.lindh = lindh;
    }

    /// Compact per-atom step. False when the solve does not apply.
    fn install_compact(&mut self, g: ArrayView1<f64>, cap: f64) -> bool {
        let gvec: Vec<f64> = g.iter().copied().collect();
        let mut cols = Vec::new();
        for p in &self.memory {
            if p.s.dot(&p.y) <= 0.0 {
                continue;
            }
            cols.push((p.s.to_vec(), p.y.to_vec()));
        }
        let Some(step) = crate::compact_tr::atom_capped_step(&cols, &gvec, cap) else {
            return false;
        };
        if step.len() != self.work.len() {
            return false;
        }
        for (i, v) in step.iter().enumerate() {
            self.work[i] = *v;
        }
        true
    }

    fn copy_work(&mut self) -> Array1<f64> {
        let n = self.work.len();
        let mut out = self.acquire(n);
        if let (Some(dst), Some(src)) = (out.as_slice_mut(), self.work.as_slice()) {
            dst.copy_from_slice(src);
        } else {
            out.assign(&self.work);
        }
        out
    }

    /// Return a displacement vector to the pool.
    pub(crate) fn recycle(&mut self, v: Array1<f64>) {
        self.recycle_vec(v);
    }

    /// Length the cap compares with. The default is the longest atom.
    fn motion(&self, step: &Array1<f64>) -> f64 {
        if self.euclidean_cap {
            l2(step)
        } else {
            max_atom_displacement(step)
        }
    }

    /// Shorten `step` to `cap` in the length [`Self::motion`] uses.
    fn apply_cap(&self, step: &Array1<f64>, cap: f64) -> Array1<f64> {
        if self.euclidean_cap {
            let motion = l2(step);
            if motion > cap && motion > 0.0 && cap > 0.0 {
                step * (cap / motion)
            } else {
                step.clone()
            }
        } else {
            cap_displacement_atom(step, cap)
        }
    }

    /// Shorten `work` so its measured length is at most `cap`.
    fn cap_work(&mut self, cap: f64) {
        let motion = self.motion(&self.work);
        if motion > cap && motion > 0.0 && cap > 0.0 {
            self.work *= cap / motion;
        }
    }

    /// `s = x - prev`, `y = g_new - g_old`, stored in a recycled slot.
    pub(crate) fn store_pos_diff(
        &mut self,
        prev: &[f64],
        x: &[f64],
        g_old: &[f64],
        g_new: &[f64],
        txn: &mut Txn,
    ) {
        let n = x.len().min(prev.len());
        let mut s = self.acquire(n);
        let mut y = self.acquire(n);
        for i in 0..n {
            s[i] = x[i] - prev[i];
            y[i] = g_new[i] - g_old[i];
        }
        let sy = s.dot(&y);
        let yy = y.dot(&y);
        self.finish_eon_pair(s, y, sy, yy, txn);
    }

    fn finish_eon_pair(&mut self, s: Array1<f64>, y: Array1<f64>, sy: f64, yy: f64, txn: &mut Txn) {
        if self.skip_bad && !(sy.is_finite() && sy > 0.0 && yy.is_finite()) {
            self.recycle_vec(s);
            self.recycle_vec(y);
            return;
        }
        if sy.is_finite() && yy.is_finite() {
            self.last_sy = sy;
            self.last_yy = yy;
            self.have_curvature = true;
        } else {
            self.have_curvature = false;
            self.recycle_vec(s);
            self.recycle_vec(y);
            return;
        }
        if sy.abs() < 1e-30 {
            let note = (self.last_sy, self.last_yy, self.have_curvature);
            self.recycle_vec(s);
            self.recycle_vec(y);
            self.forget_txn(txn);
            (self.last_sy, self.last_yy, self.have_curvature) = note;
            return;
        }
        self.push_buf_txn(s, y, sy, yy, txn);
    }

    /// Two-loop direction, restarted along the force when it is not descent.
    ///
    /// The flag is true when the direction is an unscaled force and the
    /// caller must give it length `max_atom`. It is false when the
    /// Nocedal-Wright scale, or an installed exponential preconditioner,
    /// already supplies a length and the caller only shortens the step.
    pub(crate) fn capped_direction(&mut self, g: ArrayView1<f64>) -> Option<(Array1<f64>, bool)> {
        if !self.memory.is_empty() {
            let d = self.direction(g);
            let slope = crate::vecops::dot(d.view(), g);
            if slope.is_finite() && slope < 0.0 {
                return Some((d, false));
            }
            self.forget();
        }
        if let Some(pre) = self.exp_apply(g) {
            let mut d = pre;
            d.mapv_inplace(|v| -v);
            let slope = crate::vecops::dot(d.view(), g);
            if slope.is_finite() && slope < 0.0 {
                return Some((d, false));
            }
        }
        if let Some(h0) = self.fixed_scale() {
            let mut d = g.to_owned();
            d *= -h0;
            let slope = crate::vecops::dot(d.view(), g);
            if slope.is_finite() && slope < 0.0 {
                return Some((d, false));
            }
        }
        let mut d = g.to_owned();
        d.mapv_inplace(|v| -v);
        let slope = crate::vecops::dot(d.view(), g);
        (slope.is_finite() && slope < 0.0).then_some((d, true))
    }

    /// `to_cap` scales the displacement to `max_atom`. Otherwise the
    /// displacement is only shortened to that length.
    pub(crate) fn apply_atom_cap(
        direction: &Array1<f64>,
        max_atom: f64,
        to_cap: bool,
    ) -> Array1<f64> {
        if to_cap {
            scale_displacement_to_atom_cap(direction, max_atom)
        } else {
            cap_displacement_atom(direction, max_atom)
        }
    }

    /// Curvature `C = (g(x + δu) - g) · u / δ` along the unit force.
    /// A positive finite `C` is `H0 = 1/C`. A non-positive or non-finite
    /// answer is the steepest fallback. A missing answer keeps
    /// [`EON_INVERSE_CURVATURE`] and spends no evaluation.
    fn fd_seed<F>(&self, x: &Array1<f64>, g: &Array1<f64>, fg: &mut F) -> (EonSeed, usize)
    where
        F: FnMut(ArrayView1<f64>) -> Option<(f64, Array1<f64>)>,
    {
        let force = g.mapv(|v| -v);
        let norm = l2(&force);
        if !(norm.is_finite() && norm > 0.0) {
            return (EonSeed::Scale(EON_INVERSE_CURVATURE), 0);
        }
        let u = &force / norm;
        let probe = x + &(EON_FINITE_DIFFERENCE * &u);
        let Some((value, grad)) = fg(probe.view()) else {
            return (EonSeed::Scale(EON_INVERSE_CURVATURE), 0);
        };
        if grad.len() != g.len() || !finite(value, grad.view()) {
            return (EonSeed::Steepest, 1);
        }
        let curvature = (&grad - g).dot(&u) / EON_FINITE_DIFFERENCE;
        if curvature.is_finite() && curvature > 0.0 {
            (EonSeed::Scale(1.0 / curvature), 1)
        } else {
            (EonSeed::Steepest, 1)
        }
    }

    fn take_free<F>(
        &mut self,
        x: &mut Array1<f64>,
        f: &mut f64,
        g: &mut Array1<f64>,
        fg: &mut F,
    ) -> Option<CappedIter>
    where
        F: FnMut(ArrayView1<f64>) -> Option<(f64, Array1<f64>)>,
    {
        match self.line_search_free()? {
            FreeRule::Eon(max_atom) => Some(self.eon_iterate(x, f, g, max_atom, fg)),
            FreeRule::Scaled(max_atom) => Some(self.capped_iterate(x, f, g, max_atom, fg)),
        }
    }

    /// One eOn iteration. The finite-difference probe, when it runs,
    /// is one evaluation and does not move `x`. A non-finite trial of a
    /// quasi-Newton step is retried once along `H0` times the force.
    fn eon_iterate<F>(
        &mut self,
        x: &mut Array1<f64>,
        f: &mut f64,
        g: &mut Array1<f64>,
        max_atom: f64,
        fg: &mut F,
    ) -> CappedIter
    where
        F: FnMut(ArrayView1<f64>) -> Option<(f64, Array1<f64>)>,
    {
        let mut txn = self.begin_txn();
        let mut evals = 0usize;
        let mut seed = self.eon_seed();
        if self.eon_wants_fd() {
            let (probed, used) = self.fd_seed(x, g, fg);
            seed = probed;
            evals += used;
        }
        let h0 = match seed {
            EonSeed::Scale(h) => h,
            EonSeed::Steepest => EON_INVERSE_CURVATURE,
        };
        let Some(take) = self.eon_step_from_seed(g.view(), max_atom, seed, &mut txn) else {
            self.rollback_txn(txn);
            return CappedIter::Stop { evals };
        };
        match self.accept_eon(
            x,
            f,
            g,
            take.step.as_slice().unwrap(),
            &mut txn,
            &mut evals,
            fg,
        ) {
            CappedProbe::Moved { stalled } => {
                self.recycle(take.step);
                self.commit_txn(txn);
                CappedIter::Moved { stalled, evals }
            }
            CappedProbe::Stop => {
                self.recycle(take.step);
                self.rollback_txn(txn);
                CappedIter::Stop { evals }
            }
            CappedProbe::NonFinite if take.fallback => {
                self.recycle(take.step);
                self.rollback_txn(txn);
                CappedIter::Stop { evals }
            }
            CappedProbe::NonFinite => {
                self.recycle(take.step);
                self.forget_txn(&mut txn);
                self.prepare_force_cap(g.view(), h0, max_atom);
                let step = self.copy_work();
                match self.accept_eon(x, f, g, step.as_slice().unwrap(), &mut txn, &mut evals, fg) {
                    CappedProbe::Moved { stalled } => {
                        self.recycle(step);
                        self.commit_txn(txn);
                        CappedIter::Moved { stalled, evals }
                    }
                    CappedProbe::Stop | CappedProbe::NonFinite => {
                        self.recycle(step);
                        self.rollback_txn(txn);
                        CappedIter::Stop { evals }
                    }
                }
            }
        }
    }

    /// `work = cap(h0 * force)`.
    fn prepare_force_cap(&mut self, g: ArrayView1<f64>, h0: f64, max_atom: f64) {
        let n = g.len();
        self.ensure_work(n);
        if let (Some(w), Some(gv)) = (self.work.as_slice_mut(), g.as_slice()) {
            for i in 0..n {
                w[i] = h0 * -gv[i];
            }
        }
        self.cap_work(max_atom);
    }

    #[allow(clippy::too_many_arguments)]
    fn accept_eon<F>(
        &mut self,
        x: &mut Array1<f64>,
        f: &mut f64,
        g: &mut Array1<f64>,
        step: &[f64],
        txn: &mut Txn,
        evals: &mut usize,
        fg: &mut F,
    ) -> CappedProbe
    where
        F: FnMut(ArrayView1<f64>) -> Option<(f64, Array1<f64>)>,
    {
        let motion = max_atom_slice(step);
        if !motion.is_finite() || motion <= 0.0 {
            return CappedProbe::Stop;
        }
        let n = x.len().min(step.len());
        for i in 0..n {
            x[i] += step[i];
        }
        let Some((fv, gv)) = fg(x.view()) else {
            for i in 0..n {
                x[i] -= step[i];
            }
            return CappedProbe::Stop;
        };
        *evals += 1;
        if gv.len() != g.len() || step.len() != x.len() || !finite(fv, gv.view()) {
            for i in 0..n {
                x[i] -= step[i];
            }
            return CappedProbe::NonFinite;
        }
        let f_old = *f;
        if let (Some(gold), Some(gnew)) = (g.as_slice(), gv.as_slice()) {
            self.store_diff_txn(step, gnew, gold, txn);
        }
        *f = fv;
        *g = gv;
        CappedProbe::Moved {
            stalled: self.stalled(f_old, fv),
        }
    }

    /// One scaled iteration. A non-finite trial of a quasi-Newton step is
    /// retried once along the force. A second non-finite trial, a spent
    /// budget, or a zero step leaves `x` and the pairs as they were.
    fn capped_iterate<F>(
        &mut self,
        x: &mut Array1<f64>,
        f: &mut f64,
        g: &mut Array1<f64>,
        max_atom: f64,
        fg: &mut F,
    ) -> CappedIter
    where
        F: FnMut(ArrayView1<f64>) -> Option<(f64, Array1<f64>)>,
    {
        let saved = self.curvature();
        let Some((dir, full)) = self.capped_direction(g.view()) else {
            return CappedIter::Stop { evals: 0 };
        };
        let mut evals = 0usize;
        let step = Self::apply_atom_cap(&dir, max_atom, full);
        match self.accept_capped(x, f, g, &step, &mut evals, fg) {
            CappedProbe::Moved { stalled } => CappedIter::Moved { stalled, evals },
            CappedProbe::Stop => {
                self.restore(saved);
                CappedIter::Stop { evals }
            }
            CappedProbe::NonFinite if full => {
                self.restore(saved);
                CappedIter::Stop { evals }
            }
            CappedProbe::NonFinite => {
                self.forget();
                let Some((dir, full)) = self.capped_direction(g.view()) else {
                    self.restore(saved);
                    return CappedIter::Stop { evals };
                };
                let step = Self::apply_atom_cap(&dir, max_atom, full);
                match self.accept_capped(x, f, g, &step, &mut evals, fg) {
                    CappedProbe::Moved { stalled } => CappedIter::Moved { stalled, evals },
                    CappedProbe::Stop | CappedProbe::NonFinite => {
                        self.restore(saved);
                        CappedIter::Stop { evals }
                    }
                }
            }
        }
    }

    fn accept_capped<F>(
        &mut self,
        x: &mut Array1<f64>,
        f: &mut f64,
        g: &mut Array1<f64>,
        step: &Array1<f64>,
        evals: &mut usize,
        fg: &mut F,
    ) -> CappedProbe
    where
        F: FnMut(ArrayView1<f64>) -> Option<(f64, Array1<f64>)>,
    {
        let motion = max_atom_displacement(step);
        if !motion.is_finite() || motion <= 0.0 {
            return CappedProbe::Stop;
        }
        let mut trial = x.clone();
        trial.scaled_add(1.0, step);
        let Some((fv, gv)) = fg(trial.view()) else {
            return CappedProbe::Stop;
        };
        *evals += 1;
        if gv.len() != g.len() || step.len() != x.len() || !finite(fv, gv.view()) {
            return CappedProbe::NonFinite;
        }
        let s = &trial - &*x;
        let y = &gv - &*g;
        let f_old = *f;
        self.push(s, y);
        *x = trial;
        *f = fv;
        *g = gv;
        CappedProbe::Moved {
            stalled: self.stalled(f_old, fv),
        }
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
    /// No trial exceeds `a_max`; a decrease at `a_max` that has not
    /// bracketed is accepted there, as `dcsrch` does at `stpmax`.
    fn line_search<F>(
        &mut self,
        x: &mut Array1<f64>,
        f: &mut f64,
        g: &mut Array1<f64>,
        ray: Ray<'_>,
        fg: &mut F,
    ) -> (bool, usize)
    where
        F: FnMut(ArrayView1<f64>) -> Option<(f64, Array1<f64>)>,
    {
        let Ray { d, slope, a_max } = ray;
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
        }
        .min(a_max);
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
            if a >= a_max {
                self.accept(x, f, g, d, a, (fa, ga));
                return (true, evals);
            }
            a_prev = a;
            f_prev = fa;
            slope_prev = slope_a;
            a = (2.0 * a).min(a_max);
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
    ///
    /// [`LbfgsStep::Capped`] is the eOn step. The first iteration spends
    /// one finite-difference probe, then one evaluation per accepted
    /// step. A proposal at least as long as the cap clears the memory.
    /// [`LbfgsStep::Scaled`] takes one evaluation per accepted step and
    /// keeps a shortened pair. A non-finite quasi-Newton trial is
    /// retried once along the force. A second non-finite trial ends the
    /// relaxation at the last finite point.
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
        if g.len() != x.len() || !finite(f, g.view()) {
            return (f64::INFINITY, x, evals);
        }

        let mut stalled = false;
        for it in 0..max_iter {
            if g.len() != x.len() || !watch(it, f) {
                break;
            }
            if stalled || self.gnorm(&g) < self.gtol {
                break;
            }
            if let Some(taken) = self.take_free(&mut x, &mut f, &mut g, &mut fg) {
                match taken {
                    CappedIter::Moved {
                        stalled: now,
                        evals: used,
                    } => {
                        evals += used;
                        stalled = now;
                    }
                    CappedIter::Stop { evals: used } => {
                        evals += used;
                        break;
                    }
                }
                continue;
            }
            let f_old = f;
            let Ok(d) = self.search_direction(x.view(), g.view()) else {
                break;
            };
            let slope = d.dot(&g);
            if !slope.is_finite() && self.memory.is_empty() {
                break;
            }
            if slope >= 0.0 || !slope.is_finite() {
                self.forget();
                continue;
            }
            let ray = Ray {
                d: &d,
                slope,
                a_max: f64::INFINITY,
            };
            let (ok, evals_used) = self.line_search(&mut x, &mut f, &mut g, ray, &mut fg);
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
        if g.len() != x.len() || !finite(f, g.view()) {
            return (f64::INFINITY, x, evals, false);
        }

        let mut stalled = false;
        for it in 0..max_iter {
            if g.len() != x.len() {
                break;
            }
            if let Some((f_known, x_known)) = recognise(it, f, x.view()) {
                return (f_known, x_known, evals, true);
            }
            if stalled || self.gnorm(&g) < self.gtol {
                break;
            }
            if let Some(taken) = self.take_free(&mut x, &mut f, &mut g, &mut fg) {
                match taken {
                    CappedIter::Moved {
                        stalled: now,
                        evals: used,
                    } => {
                        evals += used;
                        stalled = now;
                    }
                    CappedIter::Stop { evals: used } => {
                        evals += used;
                        break;
                    }
                }
                continue;
            }
            let f_old = f;
            let Ok(d) = self.search_direction(x.view(), g.view()) else {
                break;
            };
            let slope = d.dot(&g);
            if !slope.is_finite() && self.memory.is_empty() {
                break;
            }
            if slope >= 0.0 || !slope.is_finite() {
                self.forget();
                continue;
            }
            let ray = Ray {
                d: &d,
                slope,
                a_max: f64::INFINITY,
            };
            let (ok, evals_used) = self.line_search(&mut x, &mut f, &mut g, ray, &mut fg);
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

    /// Relaxes `x0` inside the box `lower <= x <= upper` with the
    /// generalized Cauchy point and the subspace minimization of L-BFGS-B.
    ///
    /// The box path line-searches. [`Self::minimize`] keeps the configured
    /// step, whose default is the eOn cap. Each direction ends at the subspace minimizer
    /// from the Cauchy point (Byrd, Lu, Nocedal and Zhu 1995, algorithm CP
    /// and section 5.1), and the strong Wolfe search along it stops at the
    /// box. A bound may be infinite; with no finite bound the direction is
    /// the two-loop direction up to rounding. [`Self::gtol`] is compared
    /// with the projected gradient `x - clip(x - g)`. `fg` only sees points
    /// in the box, and the result is the point, value and evaluation count
    /// of [`Self::minimize`]. A bound whose length is not `x0`'s, an
    /// inverted bound, or a gradient whose length is not `x0`'s ends the
    /// call at `x0` with value `+inf` and does not panic.
    ///
    /// ```
    /// use ndarray::{ArrayView1, array};
    /// use rgmin::Lbfgs;
    ///
    /// // The minimum (2, -1) is outside the box, which stops x[0] at 1.
    /// let fg = |x: ArrayView1<f64>| {
    ///     let f = (x[0] - 2.0).powi(2) + (x[1] + 1.0).powi(2);
    ///     Some((f, array![2.0 * (x[0] - 2.0), 2.0 * (x[1] + 1.0)]))
    /// };
    /// let lower = array![f64::NEG_INFINITY, f64::NEG_INFINITY];
    /// let upper = array![1.0, f64::INFINITY];
    /// let mut opt = Lbfgs::default();
    /// let (f, x, _) = opt.minimize_box(
    ///     array![0.0, 0.0].view(),
    ///     lower.view(),
    ///     upper.view(),
    ///     100,
    ///     fg,
    /// );
    /// assert_eq!(x[0], 1.0);
    /// assert!((x[1] + 1.0).abs() < 1e-6 && (f - 1.0).abs() < 1e-12);
    /// ```
    pub fn minimize_box<F>(
        &mut self,
        x0: ArrayView1<f64>,
        lower: ArrayView1<f64>,
        upper: ArrayView1<f64>,
        max_iter: usize,
        fg: F,
    ) -> (f64, Array1<f64>, usize)
    where
        F: FnMut(ArrayView1<f64>) -> Option<(f64, Array1<f64>)>,
    {
        self.minimize_box_watched(x0, lower, upper, max_iter, fg, |_, _| true)
    }

    /// [`Self::minimize_box`], offering each accepted iterate to `watch` as
    /// [`Self::minimize_watched`] does.
    pub fn minimize_box_watched<F, W>(
        &mut self,
        x0: ArrayView1<f64>,
        lower: ArrayView1<f64>,
        upper: ArrayView1<f64>,
        max_iter: usize,
        mut fg: F,
        mut watch: W,
    ) -> (f64, Array1<f64>, usize)
    where
        F: FnMut(ArrayView1<f64>) -> Option<(f64, Array1<f64>)>,
        W: FnMut(usize, f64) -> bool,
    {
        self.trim();
        if !bounds_ok(x0.len(), lower, upper) {
            return (f64::INFINITY, x0.to_owned(), 0);
        }
        let clip = |v: ArrayView1<f64>| {
            Array1::from_shape_fn(v.len(), |i| v[i].max(lower[i]).min(upper[i]))
        };
        let mut box_fg = |v: ArrayView1<f64>| fg(clip(v).view());
        let mut x = clip(x0);
        let mut evals = 0usize;
        let (mut f, mut g) = match box_fg(x.view()) {
            Some(v) => v,
            None => return (f64::INFINITY, x, evals),
        };
        evals += 1;
        if g.len() != x.len() || !finite(f, g.view()) {
            return (f64::INFINITY, x, evals);
        }

        let mut stalled = false;
        for it in 0..max_iter {
            if g.len() != x.len() || !watch(it, f) {
                break;
            }
            let pg =
                Array1::from_shape_fn(x.len(), |i| g[i].max(x[i] - upper[i]).min(x[i] - lower[i]));
            if stalled || self.gnorm(&pg) < self.gtol {
                break;
            }
            let b = cauchy::Compact::new(x.len(), &self.memory);
            let xbar = cauchy::cauchy(x.view(), g.view(), lower, upper, &b).and_then(|(xcp, c)| {
                cauchy::subspace(x.view(), g.view(), lower, upper, &b, xcp, &c)
            });
            let d = match xbar {
                Some(xbar) => xbar - &x,
                None if self.memory.is_empty() => break,
                None => {
                    self.forget();
                    continue;
                }
            };
            let slope = d.dot(&g);
            if slope >= 0.0 || !slope.is_finite() {
                if self.memory.is_empty() {
                    break;
                }
                self.forget();
                continue;
            }
            let a_max = (0..x.len()).fold(f64::INFINITY, |a, i| match d[i] {
                di if di < 0.0 => a.min((lower[i] - x[i]) / di),
                di if di > 0.0 => a.min((upper[i] - x[i]) / di),
                _ => a,
            });
            let ray = Ray {
                d: &d,
                slope,
                a_max,
            };
            let f_old = f;
            let (ok, used) = self.line_search(&mut x, &mut f, &mut g, ray, &mut box_fg);
            evals += used;
            x = clip(x.view());
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
        crate::step::refuse_non_finite(value, &grad)?;
        let mut istep = control.istep;

        for step in 0..control.maxiter {
            let gnorm = l2(&grad);
            if let Some(status) = control.halt(gnorm, grad.len()) {
                return Ok(Report {
                    value,
                    coords: pos,
                    steps: step,
                    grad_norm: gnorm,
                    stop_norm: StopNorm::Two,
                    grad: None,
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
                return Ok(Report {
                    value,
                    coords: pos,
                    steps: step + 1,
                    grad_norm: gnorm,
                    stop_norm: StopNorm::Two,
                    grad: None,
                    status: crate::report::Status::NoiseFloor,
                });
            }
        }
        let gnorm = l2(&grad);
        Ok(Report {
            value,
            coords: pos,
            steps: control.maxiter,
            grad_norm: gnorm,
            stop_norm: StopNorm::Two,
            grad: None,
            status: control.finish_status(gnorm, grad.len(), true),
        })
    }

    /// One outer L-BFGS iteration: two-loop direction, line search, pair.
    ///
    /// `atom_maxmove` caps the largest per-atom displacement of the
    /// accepted step; `None` leaves only `control.maxmove` in force.
    ///
    /// Returns whether every search in this iteration found no trial
    /// whose decrease cleared `c * eps_f`. The point is then unchanged.
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
        let mut dir = dir;
        project(&mut dir);
        // With pairs the two-loop direction is gamma-scaled and carries
        // the step length, so the search opens at the unit step
        // (Nocedal-Wright 3.5); `istep` sizes only the first, steepest
        // direction.
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
            (retry, stall)
        } else {
            let stall = t.energy_noise;
            (t, stall)
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
        stall
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, ArrayView1, array};

    use super::{EonSeed, LBFGSB_FTOL, Lbfgs, LbfgsStep};
    use crate::step::{l2, max_atom_displacement};

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

    /// `f = 1/2 (x^2 + 4 y^2 + 9 z^2)`. One atom. The gradient is `H x`.
    fn stiff(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let g = array![x[0], 4.0 * x[1], 9.0 * x[2]];
        let f = 0.5 * (x[0] * g[0] + x[1] * g[1] + x[2] * g[2]);
        (f, g)
    }

    /// The first scaled step has length `max_atom` along the force, and
    /// the next direction is the two-loop with `H0 = (s·y)/(y·y) I`.
    /// The pair stays stored when that direction is longer than the cap.
    #[test]
    fn the_second_scaled_step_uses_the_nocedal_wright_scale_and_keeps_the_pair() {
        let cap = 0.25;
        let mut opt = Lbfgs::with_capacity(4);
        opt.step = LbfgsStep::Scaled { max_atom: cap };
        let x0 = array![1.0, 1.0, 1.0];
        let (f0, g0) = stiff(x0.view());
        let (d0, full) = opt.capped_direction(g0.view()).unwrap();
        assert!(full);
        let s = Lbfgs::apply_atom_cap(&d0, cap, true);
        let motion = max_atom_displacement(&s);
        assert!((motion - cap).abs() <= 1e-12, "motion {motion}");
        let x1 = &x0 + &s;
        let (_f1, g1) = stiff(x1.view());
        let y = &g1 - &g0;
        let sy = s.dot(&y);
        let yy = y.dot(&y);
        assert!(sy > 0.0 && yy > 0.0);
        opt.push(s.clone(), y.clone());
        assert_eq!(opt.len(), 1);

        let gamma = sy / yy;
        let rho = 1.0 / sy;
        let mut q = g1.clone();
        let alpha = rho * s.dot(&q);
        q = &q - alpha * &y;
        q *= gamma;
        let beta = rho * y.dot(&q);
        q = &q + (alpha - beta) * &s;
        let expect = -q;

        let (d1, full1) = opt.capped_direction(g1.view()).unwrap();
        assert!(!full1);
        assert_eq!(opt.len(), 1, "a long quasi-Newton step must keep the pair");
        for k in 0..3 {
            assert!(
                (d1[k] - expect[k]).abs() <= 1e-12,
                "k={k} got {} want {}",
                d1[k],
                expect[k]
            );
        }
        assert!(
            max_atom_displacement(&d1) > cap,
            "the fixture step is longer than the cap, got {}",
            max_atom_displacement(&d1)
        );
        let taken = Lbfgs::apply_atom_cap(&d1, cap, false);
        assert!((max_atom_displacement(&taken) - cap).abs() <= 1e-12);
        // The value moved, so the iteration is a real step of the driver.
        let _ = f0;
    }

    /// On `f = 1/2 ||x||^2` the eOn step probes curvature once, then
    /// moves 0.2 toward the origin until the Newton step fits in the
    /// cap. Each long proposal clears the pair it just stored.
    #[test]
    fn eon_quadratic_probes_once_and_reaches_the_origin() {
        let mut opt = Lbfgs {
            step: LbfgsStep::Capped { max_atom: 0.2 },
            gtol: 1e-8,
            norm: super::GradNorm::Euclidean,
            ..Lbfgs::default()
        };
        let mut calls = 0usize;
        let (f, x, evals) = opt.minimize(array![1.0, 0.0, 0.0].view(), 20, |v| {
            calls += 1;
            Some((0.5 * v.dot(&v), v.to_owned()))
        });
        assert_eq!(calls, evals);
        // Start, one probe, five accepted steps of 0.2 from 1 to 0.
        assert_eq!(evals, 7, "evals {evals} x {x}");
        assert!(f.abs() < 1e-16, "f {f} x {x}");
        assert!(x.iter().all(|v| v.abs() < 1e-12), "{x}");
    }

    /// A quasi-Newton direction longer than the cap is discarded.
    /// The step that replaces it follows the force, and the old pair is gone.
    #[test]
    fn a_long_eon_direction_steps_along_the_force_and_clears_the_pair() {
        let mut opt = Lbfgs::capped(super::EON_MAX_MOVE);
        let x0 = array![1.0, 1.0, 1.0];
        let (_f, x1, evals) = opt.minimize(x0.view(), 1, |v| Some(stiff(v)));
        assert_eq!(evals, 3, "start, probe, step");
        let g0 = stiff(x0.view()).1;
        let force0 = g0.mapv(|v| -v);
        let s1 = &x1 - &x0;
        let align = s1.dot(&force0) / (l2(&s1) * l2(&force0));
        assert!((align - 1.0).abs() < 1e-12, "first step align {align}");
        assert!((max_atom_displacement(&s1) - 0.2).abs() < 1e-12);
        assert_eq!(opt.len(), 1);

        let g1 = stiff(x1.view()).1;
        let h0 = opt.last_sy / opt.last_yy;
        let quasi = opt.direction_h0(g1.view(), h0);
        assert!(max_atom_displacement(&quasi) > 0.2);
        let force1 = g1.mapv(|v| -v);
        let quasi_align = quasi.dot(&force1) / (l2(&quasi) * l2(&force1));
        assert!(
            quasi_align < 0.99,
            "the fixture quasi-Newton step must leave the force, align {quasi_align}"
        );

        let (_f, x2, _) = opt.minimize(x1.view(), 1, |v| Some(stiff(v)));
        let s2 = &x2 - &x1;
        let align2 = s2.dot(&force1) / (l2(&s2) * l2(&force1));
        assert!(
            (align2 - 1.0).abs() < 1e-9,
            "distance_reset step align {align2} s {s2}"
        );
        assert_eq!(opt.len(), 1, "the quasi-Newton pair must not survive");
    }

    /// Scaled L-BFGS keeps the pair a long direction would have cleared.
    #[test]
    fn scaled_quadratic_keeps_more_than_one_pair() {
        let mut opt = Lbfgs {
            step: LbfgsStep::Scaled { max_atom: 0.2 },
            gtol: 1e-8,
            norm: super::GradNorm::Euclidean,
            ..Lbfgs::default()
        };
        let (f, x, evals) = opt.minimize(array![1.0, 1.0, 1.0].view(), 100, |v| {
            Some((0.5 * v.dot(&v), v.to_owned()))
        });
        assert!(evals > 2, "evals {evals}");
        assert!(opt.len() > 1, "pairs were dropped, len {}", opt.len());
        assert!(f < 1e-16, "f {f} x {x}");
    }

    /// A non-positive cap is not a length. The default step stays the eOn cap.
    #[test]
    fn a_non_positive_cap_is_not_a_length_and_the_default_stays_capped() {
        for max_atom in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let opt = Lbfgs {
                step: LbfgsStep::Capped { max_atom },
                ..Lbfgs::default()
            };
            assert!(opt.capped_limit().is_none(), "cap {max_atom}");
        }
        assert_eq!(
            Lbfgs::default().step,
            LbfgsStep::Capped {
                max_atom: super::EON_MAX_MOVE
            }
        );
        assert_eq!(
            Lbfgs::builder().build().step,
            LbfgsStep::Capped {
                max_atom: super::EON_MAX_MOVE
            }
        );
    }

    /// A fresh solver takes the eOn cap and does not search a line.
    #[test]
    fn the_default_step_moves_one_atom_by_the_eon_cap() {
        let mut opt = Lbfgs::default();
        let mut calls = 0usize;
        let (_f, x, evals) = opt.minimize(array![1.0, 0.0, 0.0].view(), 1, |v| {
            calls += 1;
            Some((0.5 * v.dot(&v), v.to_owned()))
        });
        assert_eq!(calls, evals);
        // The start, the curvature probe, and the accepted step.
        assert_eq!(evals, 3);
        assert!((x[0] - (1.0 - super::EON_MAX_MOVE)).abs() <= 1e-12, "{x}");
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

    /// The model direction is kept and scaled per atom. The force
    /// fallback would leave the first coordinate at zero.
    #[test]
    fn model_atom_scales_the_quasi_newton_step_and_clears_history() {
        let mut opt = Lbfgs {
            model_atom: true,
            ..Lbfgs::default()
        };
        assert!(opt.push(
            array![1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            array![2.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ));
        let g = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let mut txn = opt.begin_txn();
        let take = opt
            .eon_step_from_seed(g.view(), 0.2, EonSeed::Scale(0.5), &mut txn)
            .unwrap();
        assert!(
            take.step[0].abs() > 1e-8,
            "force fallback would be zero: {}",
            take.step
        );
        let motion = max_atom_displacement(&take.step);
        assert!((motion - 0.2).abs() < 1e-8, "max atom {motion}");
        let euclid = l2(&take.step);
        assert!(
            euclid > 0.2,
            "euclidean {euclid} should exceed the per-atom cap"
        );
        assert!(
            opt.is_empty(),
            "history should reset after a capped model step"
        );
    }

    #[test]
    fn a_non_positive_pair_is_skipped_when_asked() {
        let mut kept = Lbfgs::with_capacity(4);
        kept.skip_bad = true;
        assert!(kept.push(array![1.0, 0.0], array![2.0, 0.0]));
        let mut txn = kept.begin_txn();
        kept.store_pos_diff(
            &[0.0, 0.0],
            &[1.0, 0.0],
            &[0.0, 0.0],
            &[-1.0, 0.0],
            &mut txn,
        );
        assert_eq!(kept.len(), 1);
        assert!(matches!(kept.eon_seed(), EonSeed::Scale(_)));

        let mut wiped = Lbfgs::with_capacity(4);
        assert!(wiped.push(array![1.0, 0.0], array![2.0, 0.0]));
        let mut txn = wiped.begin_txn();
        wiped.store_pos_diff(
            &[0.0, 0.0],
            &[1.0, 0.0],
            &[0.0, 0.0],
            &[-1.0, 0.0],
            &mut txn,
        );
        assert_eq!(wiped.len(), 2);
        assert!(matches!(wiped.eon_seed(), EonSeed::Steepest));
    }

    #[test]
    fn the_compact_step_stays_inside_the_per_atom_cap() {
        let mut opt = Lbfgs {
            compact_tr: true,
            ..Lbfgs::default()
        };
        assert!(opt.push(
            array![1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            array![0.2, 0.0, 0.0, 0.0, 0.0, 0.0]
        ));
        let g = array![1.0, 0.4, 0.0, -0.3, 0.2, 0.0];
        let mut txn = opt.begin_txn();
        let take = opt
            .eon_step_from_seed(g.view(), 0.05, EonSeed::Scale(0.5), &mut txn)
            .unwrap();
        let motion = max_atom_displacement(&take.step);
        assert!(motion <= 0.05 + 1e-8, "max atom {motion}");
        assert!(take.step.dot(&g) < 0.0, "not descent");
        assert!(!opt.is_empty(), "the compact step keeps the pairs");
    }

    /// A cap that is not a length is the Wolfe search, not a capped
    /// step whose limit happens to be missing.
    #[test]
    fn a_bad_constructor_cap_selects_wolfe() {
        for max_atom in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(Lbfgs::capped(max_atom).step, LbfgsStep::Wolfe);
            assert_eq!(Lbfgs::scaled(max_atom).step, LbfgsStep::Wolfe);
            assert_eq!(
                Lbfgs::builder().capped(max_atom).build().step,
                LbfgsStep::Wolfe
            );
        }
        assert!(matches!(
            Lbfgs::capped(0.2).step,
            LbfgsStep::Capped { max_atom } if (max_atom - 0.2).abs() < 1e-15
        ));
    }

    /// A short bound is a caller error. It must not index off the end.
    #[test]
    fn minimize_box_refuses_a_short_bound() {
        let mut opt = Lbfgs::default();
        let (f, x, evals) = opt.minimize_box(
            array![0.0, 0.0].view(),
            array![0.0].view(),
            array![1.0, 1.0].view(),
            10,
            |v| Some((v.dot(&v), v.to_owned())),
        );
        assert!(f.is_infinite());
        assert_eq!(evals, 0);
        assert_eq!(x.as_slice(), Some(&[0.0, 0.0][..]));
    }

    /// An inverted bound is not a box.
    #[test]
    fn minimize_box_refuses_an_inverted_bound() {
        let mut opt = Lbfgs::default();
        let (f, _, evals) = opt.minimize_box(
            array![0.0].view(),
            array![1.0].view(),
            array![0.0].view(),
            5,
            |v| Some((v[0] * v[0], array![2.0 * v[0]])),
        );
        assert!(f.is_infinite());
        assert_eq!(evals, 0);
    }

    /// A gradient shorter than the point ends the relaxation.
    #[test]
    fn a_short_gradient_does_not_panic() {
        let mut opt = Lbfgs::default();
        let (f, x, evals) =
            opt.minimize(array![1.0, -1.0].view(), 4, |_v| Some((1.0, array![1.0])));
        assert!(f.is_infinite());
        assert_eq!(evals, 1);
        assert_eq!(x.len(), 2);

        let mut wolfe = Lbfgs::builder().wolfe().build();
        let (f, x, _) = wolfe.minimize(array![1.0, -1.0].view(), 4, |_v| Some((1.0, array![1.0])));
        assert!(f.is_infinite());
        assert_eq!(x.len(), 2);
    }

    /// The builder's default is the eOn cap, and Wolfe is explicit.
    #[test]
    fn the_builder_defaults_to_the_eon_cap() {
        let opt = Lbfgs::builder().tolerance(1e-8).memory(5).build();
        assert_eq!(opt.max_pairs, 5);
        assert!((opt.gtol - 1e-8).abs() < 1e-15);
        assert_eq!(
            opt.step,
            LbfgsStep::Capped {
                max_atom: super::EON_MAX_MOVE
            }
        );
        assert_eq!(Lbfgs::builder().wolfe().build().step, LbfgsStep::Wolfe);
    }
}

#[cfg(test)]
mod precon_fallback_tests {
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
