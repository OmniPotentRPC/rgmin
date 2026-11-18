//! Stops that read the run as a whole: the relative energy change between
//! accepted steps ([`Control::etol`]) and the evaluation budget
//! ([`Control::max_evals`]).
//!
//! [`Counted`] sits between a driver and the caller's objective and counts
//! every call the objective receives. [`StopRules`] holds the energy of the
//! last accepted point and judges a finished iteration.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use eindir_core::{Bounds, DifferentiableObjective, FPair, Gradient, Objective};
use ndarray::{Array1, ArrayView1};

use crate::control::Control;
use crate::error::{Error, Result};
use crate::hvp::HessianVector;
use crate::newton::HessianObjective;
use crate::report::{Report, Status};

/// An objective behind a call counter.
///
/// One evaluation is one call to [`Objective::eval`], [`Gradient::grad`]
/// or [`DifferentiableObjective::value_and_gradient`]. A Hessian or a
/// Hessian-vector product is not an evaluation. The counter is atomic
/// because an objective is `Sync`.
pub(crate) struct Counted<'a, O: ?Sized> {
    inner: &'a O,
    evals: AtomicUsize,
    start: Option<Array1<f64>>,
    start_value: OnceLock<f64>,
}

impl<'a, O: ?Sized> Counted<'a, O> {
    pub(crate) fn new(inner: &'a O) -> Self {
        Self {
            inner,
            evals: AtomicUsize::new(0),
            start: None,
            start_value: OnceLock::new(),
        }
    }

    /// Also keep the value of the first evaluation at exactly `start`, the
    /// energy the first accepted step is compared with.
    pub(crate) fn watching(inner: &'a O, start: &Array1<f64>) -> Self {
        Self {
            start: Some(start.clone()),
            ..Self::new(inner)
        }
    }

    pub(crate) fn count(&self) -> usize {
        self.evals.load(Ordering::Relaxed)
    }

    /// The value the objective returned at the watched start, if it was
    /// evaluated there.
    pub(crate) fn start_value(&self) -> Option<f64> {
        self.start_value.get().copied()
    }

    fn note(&self, x: ArrayView1<f64>, value: Option<f64>) {
        self.evals.fetch_add(1, Ordering::Relaxed);
        let (Some(start), Some(value)) = (&self.start, value) else {
            return;
        };
        if start.len() != x.len()
            || !start
                .iter()
                .zip(x.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits())
        {
            return;
        }
        // The first evaluation at the start wins; a later one at the same point is dropped.
        self.start_value.get_or_init(|| value);
    }
}

impl<O: Objective<f64> + ?Sized> Objective<f64> for Counted<'_, O> {
    fn dim(&self) -> usize {
        Objective::dim(self.inner)
    }

    fn bounds(&self) -> &Bounds<f64> {
        self.inner.bounds()
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        let value = self.inner.eval(x);
        self.note(x, Some(value));
        value
    }

    fn global_min(&self) -> Option<&FPair<f64>> {
        self.inner.global_min()
    }
}

impl<O: DifferentiableObjective<f64> + ?Sized> Gradient<f64> for Counted<'_, O> {
    fn dim(&self) -> usize {
        Gradient::dim(self.inner)
    }

    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        let grad = self.inner.grad(x);
        self.note(x, None);
        grad
    }
}

impl<O: DifferentiableObjective<f64> + ?Sized> DifferentiableObjective<f64> for Counted<'_, O> {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let out = self.inner.value_and_gradient(x);
        self.note(x, Some(out.0));
        out
    }
}

impl<O: HessianObjective + ?Sized> HessianObjective for Counted<'_, O> {
    fn hessian(&self, x: ArrayView1<f64>) -> ndarray::Array2<f64> {
        self.inner.hessian(x)
    }
}

impl<O: HessianVector + ?Sized> HessianVector for Counted<'_, O> {
    fn hessian_vector(&self, x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
        self.inner.hessian_vector(x, v)
    }
}

/// Floor of the energy scale in the relative test: `sqrt(f64::EPSILON)`.
///
/// LAMMPS adds `EPS_ENERGY = 1.0e-8` to `|E_k| + |E_{k-1}|` so that a
/// run that ends near zero energy does not divide by zero. The same floor
/// derived from the machine epsilon is about `1.49e-8`.
fn energy_floor() -> f64 {
    f64::EPSILON.sqrt()
}

/// LAMMPS `min_cg.cpp` energy test:
/// `|E_k - E_{k-1}| < etol * 0.5 * (|E_k| + |E_{k-1}| + EPS_ENERGY)`.
///
/// A non-finite energy never passes. `etol = 0` never passes, as in LAMMPS.
pub(crate) fn energy_converged(etol: f64, previous: f64, current: f64) -> bool {
    (current - previous).abs() < etol * 0.5 * (current.abs() + previous.abs() + energy_floor())
}

/// The energy of the last accepted point, for the run-level stops of one
/// minimization.
#[derive(Clone, Debug, Default)]
pub(crate) struct StopRules {
    previous: Option<f64>,
}

impl StopRules {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// As [`Self::new`], with the energy of the start point.
    pub(crate) fn from_start(value: f64) -> Self {
        Self {
            previous: Some(value),
        }
    }

    /// Forget the recorded energy: the objective may have changed.
    pub(crate) fn reset(&mut self) {
        self.previous = None;
    }

    pub(crate) fn wants_start(&self) -> bool {
        self.previous.is_none()
    }

    pub(crate) fn seed(&mut self, value: Option<f64>) {
        if self.previous.is_none() {
            self.previous = value;
        }
    }

    /// Judge a finished iteration.
    ///
    /// `moved` says the iteration took an accepted step to energy `value`;
    /// an iteration that stayed put records nothing and cannot end the run
    /// on energy. `evals` is the count so far. `settled` is false while a
    /// damped-dynamics method is inside its delay after a velocity reset.
    /// `gradient_met` is a gradient stop (`Tolerance` or `NoiseFloor`) that
    /// holds at the new point: it is the stronger report, so no stop is
    /// returned and the caller's own test names it.
    ///
    /// The evaluation budget is read before the energy, as LAMMPS reads
    /// `maxeval` before `etol`. With neither set this returns `None` and
    /// records nothing.
    pub(crate) fn after(
        &mut self,
        control: &Control,
        value: f64,
        moved: bool,
        evals: usize,
        settled: bool,
        gradient_met: bool,
    ) -> Option<Status> {
        if control.etol.is_none() && control.max_evals.is_none() {
            return None;
        }
        let mut energy = false;
        if moved {
            if let (Some(etol), Some(previous)) = (control.etol, self.previous) {
                energy = settled && energy_converged(etol, previous, value);
            }
            self.previous = Some(value);
        }
        if gradient_met {
            return None;
        }
        if control.max_evals.is_some_and(|cap| evals >= cap) {
            return Some(Status::EvalBudget);
        }
        energy.then_some(Status::EnergyTolerance)
    }
}

/// Run `driver` on `obj` behind a counter, after `control` passes
/// [`validate`], and put the count on its report.
pub(crate) fn counted<O, F>(obj: &O, control: &Control, driver: F) -> Result<Report>
where
    O: ?Sized,
    F: FnOnce(&Counted<'_, O>) -> Result<Report>,
{
    validate(control)?;
    let counter = Counted::new(obj);
    match driver(&counter) {
        Ok(mut report) => {
            report.evals = counter.count();
            Ok(report)
        }
        Err(Error::StopUnmet { reason, mut report }) => {
            report.evals = counter.count();
            Err(Error::StopUnmet { reason, report })
        }
        Err(err) => Err(err),
    }
}

/// [`StopRules::after`] for a one-shot driver whose method has no delay:
/// the gradient stop of `grad` outranks both run-level stops.
pub(crate) fn stop_after<O: ?Sized>(
    stops: &mut StopRules,
    control: &Control,
    obj: &Counted<'_, O>,
    value: f64,
    moved: bool,
    grad: &Array1<f64>,
) -> Option<Status> {
    if control.etol.is_none() && control.max_evals.is_none() {
        return None;
    }
    let gradient_met = control.classify(crate::step::l2(grad), grad.len()) != Status::Iterating;
    stops.after(control, value, moved, obj.count(), true, gradient_met)
}

/// The `etol` and `max_evals` of `control`, refused when they are not a
/// stop a driver can read.
pub(crate) fn validate(control: &Control) -> Result<()> {
    if let Some(etol) = control.etol
        && !(etol.is_finite() && etol >= 0.0)
    {
        return Err(Error::InvalidControl {
            what: "etol must be finite and non-negative",
        });
    }
    if control.max_evals == Some(0) {
        return Err(Error::InvalidControl {
            what: "max_evals must be at least 1",
        });
    }
    Ok(())
}

/// A method with no accepted-step energy sequence cannot read `etol`.
pub(crate) fn refuse_etol(control: &Control, method: &'static str) -> Result<()> {
    if control.etol.is_some() {
        return Err(Error::StopUnsupported {
            method,
            setting: "etol",
        });
    }
    Ok(())
}

/// Whether an iteration moved the point: any coordinate differs in its bits.
pub(crate) fn moved(before: &Array1<f64>, after: &Array1<f64>) -> bool {
    before.len() != after.len()
        || before
            .iter()
            .zip(after.iter())
            .any(|(a, b)| a.to_bits() != b.to_bits())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_energy_test_is_the_lammps_formula() {
        let etol = 1e-4;
        assert!(energy_converged(etol, 100.0, 100.0 - 1e-3));
        assert!(!energy_converged(etol, 100.0, 100.0 - 1e-1));
        let floor = f64::EPSILON.sqrt();
        let tiny = 0.5 * etol * floor * 0.99;
        assert!(energy_converged(etol, 0.0, tiny));
        assert!(!energy_converged(etol, 0.0, 0.5 * etol * floor * 1.01));
    }

    #[test]
    fn zero_or_non_finite_never_passes() {
        assert!(!energy_converged(0.0, 1.0, 1.0));
        assert!(!energy_converged(1e-3, f64::NAN, 1.0));
        assert!(!energy_converged(1e-3, f64::INFINITY, f64::INFINITY));
    }

    #[test]
    fn budget_is_read_before_energy_and_a_gradient_stop_before_both() {
        let control = Control::default().with_etol(1e-3).with_max_evals(5);
        let mut rules = StopRules::from_start(1.0);
        assert_eq!(
            rules.after(&control, 1.0, true, 5, true, false),
            Some(Status::EvalBudget)
        );
        let mut rules = StopRules::from_start(1.0);
        assert_eq!(
            rules.after(&control, 1.0, true, 4, true, false),
            Some(Status::EnergyTolerance)
        );
        let mut rules = StopRules::from_start(1.0);
        assert_eq!(rules.after(&control, 1.0, true, 5, true, true), None);
    }

    #[test]
    fn a_step_that_stayed_put_cannot_end_the_run_on_energy() {
        let control = Control::default().with_etol(1e-3);
        let mut rules = StopRules::from_start(1.0);
        assert_eq!(rules.after(&control, 1.0, false, 1, true, false), None);
    }
}
