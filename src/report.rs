//! Result of a minimization.

use ndarray::Array1;

/// Why a minimization stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// The gradient is still above `gtol` and above the noise floor.
    Iterating,
    /// `||g||` fell below [`crate::Control::gtol`].
    Tolerance,
    /// `||g||` fell below `c * eps_g` and was still above `gtol`.
    NoiseFloor,
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
