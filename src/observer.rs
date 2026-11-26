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
///
/// The view is a flat coordinate vector and the scalars the minimizer
/// already computed. It carries no atom layout, mass metric, or solver tag.
pub trait IterationObserver {
    /// Observe one iteration and decide whether to continue.
    fn observe(&mut self, iteration: IterationView<'_>) -> ObserverSignal;
}

/// Euclidean length of `left - right` with no temporary vector.
#[inline]
pub(crate) fn separation(left: ArrayView1<'_, f64>, right: ArrayView1<'_, f64>) -> f64 {
    let mut acc = 0.0;
    for (u, v) in left.iter().zip(right.iter()) {
        let delta = u - v;
        acc += delta * delta;
    }
    acc.sqrt()
}

/// Tell `observer` about one accepted iterate.
///
/// A vacant observer returns `false` and does not call into the trait,
/// so the unset path does no dynamic dispatch. Callers compute
/// `step_length` only when [`Option::is_some`] is true.
#[inline(always)]
pub(crate) fn request_stop(
    observer: &mut Option<&mut dyn IterationObserver>,
    iteration: usize,
    x: ArrayView1<'_, f64>,
    value: f64,
    grad_norm: f64,
    step_length: f64,
) -> bool {
    let Some(observer) = observer.as_deref_mut() else {
        return false;
    };
    observer.observe(IterationView {
        iteration,
        x,
        value,
        grad_norm,
        step_length,
    }) == ObserverSignal::Stop
}

impl<F> IterationObserver for F
where
    F: for<'a> FnMut(IterationView<'a>) -> ObserverSignal,
{
    fn observe(&mut self, iteration: IterationView<'_>) -> ObserverSignal {
        self(iteration)
    }
}
