//! Result of a minimization.

use ndarray::Array1;

/// Why a minimization stopped.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// The gradient is still above `gtol` and above the noise floor.
    Iterating,
    /// `||g||` fell below [`crate::Control::gtol`].
    Tolerance,
    /// The stationarity measure fell below `c * sqrt(n_free) * eps_g`
    /// and was still at or above `gtol`.
    ///
    /// One projection `g · w` with `w` uniform on the sphere has variance
    /// `tr(Σ) / n`. The Euclidean noise is `sqrt(n)` times that, so the
    /// comparison uses `sqrt(n_free)`.
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
    /// `||∇f||_2` at the accepted point.
    pub grad_norm: f64,
    /// Which stop produced this report.
    pub status: Status,
}
