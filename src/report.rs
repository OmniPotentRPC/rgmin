//! Result of a minimization.

use ndarray::Array1;

use crate::control::StopNorm;

/// Why a minimization stopped.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// The gradient is still above `gtol` and above the noise floor.
    Iterating,
    /// `||g||` fell below [`crate::Control::gtol`].
    Tolerance,
    /// The stationarity measure fell below `c * sqrt(n_free) * eps_g`
    /// and was still at or above `gtol`, or the line search found no
    /// trial whose decrease cleared `c * eps_f`.
    ///
    /// One projection `g · w` with `w` uniform on the sphere has variance
    /// `tr(Σ) / n`. The Euclidean noise is `sqrt(n)` times that, so the
    /// comparison uses `sqrt(n_free)`. The energy test keeps the point
    /// from before that search.
    NoiseFloor,
    /// SCG, or an L-BFGS session stepped with no iteration budget, stopped
    /// on its step or objective test while `||g||` was still at or above
    /// `gtol`. The L-BFGS run is `max(n, 2)` accepted steps that move `x`
    /// while the value stays inside the objective roundoff window and
    /// `||g||` inside `sqrt(eps)` of the step that opened the run. One
    /// such step is an overshoot.
    StepTolerance,
    /// GROMACS steepest descent could not represent a coordinate change
    /// at its current maximum displacement.
    MachinePrecision,
    /// The iteration budget was spent above every stop.
    Budget,
    /// The energy changed by less than [`crate::Control::etol`] allows
    /// between two accepted steps, the LAMMPS `minimize etol` stop.
    EnergyTolerance,
    /// [`crate::Control::max_evals`] objective evaluations were spent above
    /// every other stop, the LAMMPS `minimize maxeval` stop.
    EvalBudget,
}

/// Typed stop of a finished minimization.
///
/// [`Status`] is the historical code. [`StopReason`] names the same
/// stops in the vocabulary a caller uses to decide whether the point
/// is stationary: [`Self::Converged`] is the `gtol` stop,
/// [`Self::MaxIter`] is a spent budget, and [`Self::LineSearchFailed`]
/// is a budget spent without accepting a step. A backward-error polish
/// that does not meet its test is [`Self::PolishUnmet`].
///
/// [`Report::converged`] is true only for [`Self::Converged`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// [`Status::Tolerance`]: the gradient norm fell below `gtol`, or a
    /// backward-error polish accepted the point.
    Converged,
    /// [`Status::Budget`]: `maxiter` was spent with the stop unmet.
    MaxIter,
    /// No step was accepted and the gradient is still above `gtol`.
    LineSearchFailed,
    /// [`Status::NoiseFloor`].
    NoiseFloor,
    /// [`Status::StepTolerance`].
    StepTolerance,
    /// [`Status::MachinePrecision`].
    MachinePrecision,
    /// An opt-in backward-error polish ended above its test.
    PolishUnmet,
    /// [`Status::Iterating`]: a step that has not stopped.
    Iterating,
    /// [`Status::EnergyTolerance`]: the relative energy change between
    /// accepted steps fell below [`crate::Control::etol`].
    EnergyTolerance,
    /// [`Status::EvalBudget`]: [`crate::Control::max_evals`] was spent.
    EvalBudget,
}

impl StopReason {
    /// The `gtol` stop, or a polish that met the backward-error test.
    pub fn converged(self) -> bool {
        matches!(self, Self::Converged)
    }

    /// A finished run whose stop test did not hold.
    ///
    /// [`Self::NoiseFloor`], [`Self::StepTolerance`] and
    /// [`Self::EnergyTolerance`] are stops that held.
    /// [`Self::Iterating`] is a step still in progress.
    pub fn is_unmet(self) -> bool {
        matches!(
            self,
            Self::MaxIter | Self::LineSearchFailed | Self::PolishUnmet | Self::EvalBudget
        )
    }

    /// Historical [`Status`] when this reason has one.
    ///
    /// [`Self::LineSearchFailed`] and [`Self::PolishUnmet`] keep
    /// [`Status::Budget`]: they are a spent stop, named more finely here.
    pub fn status(self) -> Status {
        match self {
            Self::Converged => Status::Tolerance,
            Self::MaxIter | Self::LineSearchFailed | Self::PolishUnmet => Status::Budget,
            Self::NoiseFloor => Status::NoiseFloor,
            Self::StepTolerance => Status::StepTolerance,
            Self::MachinePrecision => Status::MachinePrecision,
            Self::Iterating => Status::Iterating,
            Self::EnergyTolerance => Status::EnergyTolerance,
            Self::EvalBudget => Status::EvalBudget,
        }
    }

    fn from_status(status: Status) -> Self {
        match status {
            Status::Tolerance => Self::Converged,
            Status::Budget => Self::MaxIter,
            Status::NoiseFloor => Self::NoiseFloor,
            Status::StepTolerance => Self::StepTolerance,
            Status::MachinePrecision => Self::MachinePrecision,
            Status::Iterating => Self::Iterating,
            Status::EnergyTolerance => Self::EnergyTolerance,
            Status::EvalBudget => Self::EvalBudget,
        }
    }
}

impl std::fmt::Display for StopReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Converged => "converged",
            Self::MaxIter => "maxiter",
            Self::LineSearchFailed => "line search failed",
            Self::NoiseFloor => "noise floor",
            Self::StepTolerance => "step tolerance",
            Self::MachinePrecision => "machine precision",
            Self::PolishUnmet => "backward-error polish unmet",
            Self::Iterating => "iterating",
            Self::EnergyTolerance => "energy tolerance",
            Self::EvalBudget => "evaluation budget",
        })
    }
}

/// Final point, value, and iteration count.
///
/// Non-exhaustive, so a field added later is additive for callers that
/// read it. A host that writes its own report (a checkpoint, a replay)
/// starts from [`Report::new`] and chains the `with_*` setters.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct Report {
    /// `f(x)` at [`Report::coords`].
    pub value: f64,
    /// Accepted coordinates.
    pub coords: Array1<f64>,
    /// Outer iterations performed.
    pub steps: usize,
    /// Gradient norm at the accepted point. A Euclidean session with a box
    /// and no additional constraints reports `||x - clip(x-∇f, lo, hi)||_2`.
    /// Other sessions report the norm of their projected or raw gradient.
    /// A session measures it in [`Report::stop_norm`], `||.||_2` by default.
    /// One-shot drivers report `||g||_2`.
    pub grad_norm: f64,
    /// Norm in which [`Report::grad_norm`] is measured. Drivers that run
    /// no session report [`StopNorm::Two`].
    pub stop_norm: StopNorm,
    /// Gradient that [`Report::grad_norm`] measures, before any box
    /// projection. Sessions fill it. One-shot drivers leave it `None`.
    pub grad: Option<Array1<f64>>,
    /// Which stop produced this report.
    pub status: Status,
    /// Finer stop. `None` follows [`Report::status`].
    pub(crate) reason: Option<StopReason>,
    /// Objective evaluations the run made: calls to the objective's value,
    /// gradient or fused value-and-gradient. A one-shot driver counts its
    /// own call. A session counts every step it has taken, and
    /// [`crate::Solver::run`] counts the whole run. This is the quantity
    /// [`crate::Control::max_evals`] limits.
    pub evals: usize,
}

impl Report {
    /// A report at `coords` with value `value`: zero steps and gradient
    /// norm, [`StopNorm::Two`], no gradient, [`Status::Iterating`].
    pub fn new(value: f64, coords: Array1<f64>) -> Self {
        Self {
            value,
            coords,
            steps: 0,
            grad_norm: 0.0,
            stop_norm: StopNorm::Two,
            grad: None,
            status: Status::Iterating,
            reason: None,
            evals: 0,
        }
    }

    /// Stop named for this report.
    ///
    /// A reason set by [`Self::with_stop_reason`] wins. Otherwise
    /// [`Status::Tolerance`] is [`StopReason::Converged`] and
    /// [`Status::Budget`] is [`StopReason::MaxIter`].
    pub fn stop_reason(&self) -> StopReason {
        match self.reason {
            Some(reason) => reason,
            None => StopReason::from_status(self.status),
        }
    }

    /// Whether [`Self::stop_reason`] is [`StopReason::Converged`].
    pub fn converged(&self) -> bool {
        self.stop_reason().converged()
    }

    /// Record `reason` and the [`Status`] it corresponds to.
    #[must_use]
    pub fn with_stop_reason(mut self, reason: StopReason) -> Self {
        self.status = reason.status();
        self.reason = Some(reason);
        self
    }

    /// Replace [`Self::status`] and drop a finer [`Self::stop_reason`] override.
    #[must_use]
    pub fn with_status(mut self, status: Status) -> Self {
        self.status = status;
        self.reason = None;
        self
    }
}

with_setters!(Report {
    with_value => value: val f64,
    with_coords => coords: val Array1<f64>,
    with_steps => steps: val usize,
    with_grad_norm => grad_norm: val f64,
    with_stop_norm => stop_norm: val StopNorm,
    with_grad => grad: opt Array1<f64>,
    with_evals => evals: val usize,
});
