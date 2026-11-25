//! Optional per-iteration observer for minimization runs.

use ndarray::ArrayView1;

/// Decision returned by an iteration observer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObserverSignal {
    /// Continue the minimization loop.
    Continue,
    /// Stop the minimization loop with [`crate::StopReason::ObserverRequested`].
    Stop,
}

/// Read-only iteration state passed to an observer.
#[derive(Clone, Copy, Debug)]
pub struct IterationView<'a> {
    /// Zero-based iteration index.
    pub iteration: usize,
    /// Coordinates at this iteration.
    pub x: ArrayView1<'a, f64>,
    /// Objective value at `x`.
    pub value: f64,
    /// Gradient/stationarity norm at `x`.
    pub grad_norm: f64,
    /// Step length from the previous iterate to `x`.
    ///
    /// The first observed accepted iterate uses `0.0`.
    pub step_length: f64,
}

/// Called once per iteration with a read-only state snapshot.
pub trait IterationObserver {
    /// Observe one iteration and decide whether to continue.
    fn observe(&mut self, iteration: IterationView<'_>) -> ObserverSignal;
}

impl<F> IterationObserver for F
where
    F: for<'a> FnMut(IterationView<'a>) -> ObserverSignal,
{
    fn observe(&mut self, iteration: IterationView<'_>) -> ObserverSignal {
        self(iteration)
    }
}
