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
    /// SCG stopped on its step or objective test while `||g||` was still
    /// at or above `gtol`.
    StepTolerance,
    /// The iteration budget was spent above every stop.
    Budget,
}

/// Final point, value, and iteration count.
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
}
