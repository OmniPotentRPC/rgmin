//! Iteration limits and gradient tolerance.

use ndarray::ArrayView1;

/// Opt-in stop for a gradient whose Euclidean noise is `sqrt(n_free) * eps_g`.
///
/// One projection `g · w` with `w` uniform on the unit sphere has
/// `E[eps_g²] = tr(Σ) / n` for any noise covariance `Σ`, because the
/// sphere average of `w wᵀ` is `I / n`. The noise vector itself has
/// `E[||e||²] = tr(Σ) = n * eps_g²`, so a test of `||g||_2` against
/// `eps_g` alone sits `sqrt(n)` too low. The stop uses
/// `c * sqrt(n_free) * eps_g`. `eps_f` is the energy noise on the same
/// stencil. A line search that finds no trial whose decrease clears
/// `c * eps_f` is the same stop.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NoiseFloor {
    /// Noise of one projected component, `eps_g`.
    pub eps_g: f64,
    /// Factor in front of `sqrt(n_free) * eps_g` and of `eps_f`. The default is 3.
    pub c: f64,
    /// Noise of the energy. `None` leaves the decrease test off.
    pub eps_f: Option<f64>,
}

impl NoiseFloor {
    /// Floor with `c = 3` and no energy test.
    pub fn new(eps_g: f64) -> Self {
        Self {
            eps_g,
            c: 3.0,
            eps_f: None,
        }
    }

    /// Replace the factor `c`.
    #[must_use]
    pub fn with_c(mut self, c: f64) -> Self {
        self.c = c;
        self
    }

    /// Replace the per-component noise `eps_g`.
    #[must_use]
    pub fn with_eps_g(mut self, eps_g: f64) -> Self {
        self.eps_g = eps_g;
        self
    }

    /// Record the energy noise from the same stencil as `eps_g`. A bare
    /// value is `Some`; `None` leaves the decrease test off.
    #[must_use]
    pub fn with_eps_f(mut self, eps_f: impl Into<Option<f64>>) -> Self {
        self.eps_f = eps_f.into();
        self
    }

    /// Whether the best finite decrease of a line search missed `c * eps_f`.
    ///
    /// A non-positive or non-finite `eps_f`, and a non-positive `c`, leave
    /// the test off. A decrease equal to `c * eps_f` clears it.
    pub fn stalled_on_energy(self, best_decrease: f64) -> bool {
        match self.eps_f {
            Some(eps) if eps.is_finite() && eps > 0.0 && self.c.is_finite() && self.c > 0.0 => {
                best_decrease < self.c * eps
            }
            _ => false,
        }
    }
}

/// Outer-loop controls for [`crate::minimize`].
///
/// Non-exhaustive: start from [`Control::default`] and chain the
/// `with_*` setters, so a knob added later does not break callers.
///
/// ```
/// let control = rgmin::Control::default()
///     .with_maxiter(500)
///     .with_gtol(1e-6)
///     .with_maxmove(0.1);
/// assert_eq!(control.maxmove, Some(0.1));
/// ```
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct Control {
    /// Iteration budget of [`crate::Solver::run`] and of the one-shot
    /// minimizers that take a control. One [`crate::Solver::step`] is one
    /// iteration. A zero budget takes no step in `Solver::run`.
    pub maxiter: usize,
    /// Stop when the gradient norm is below `gtol`: [`Self::stop_norm`]
    /// in a session, `||g||_2` elsewhere. A noise floor, when set, can
    /// stop earlier; see [`Self::classify`].
    pub gtol: f64,
    /// The opening step of a line search, and the initial SR trust radius
    /// when no per-atom cap is set (`max(istep, 1)`). BFGS and the session
    /// steepest-descent search open every line search here. L-BFGS opens
    /// here while it holds no pairs and at 1.0 once it does. `minimize_sd`
    /// opens here once and then at half the step last accepted. Nonlinear
    /// CG opens from Nocedal-Wright equation 3.60. Adam opens here once
    /// and then at the step last accepted, along a direction scaled to
    /// unit per-atom length. SR1 and SR2 do not line search.
    pub istep: f64,
    /// Optional Euclidean cap on a proposed step (xtsci `maxmove`).
    /// Only a positive finite `c` in `Some(c)` limits a step or a trust
    /// radius. Any other value is no Euclidean cap, the same as `None`,
    /// so it never scales a step and cannot reverse one. A positive
    /// value is the step limit, and the per-atom default stays off
    /// unless the caller sets a per-atom cap. A negative or infinite
    /// value also leaves the default per-atom cap off; zero and NaN keep
    /// it. On the C ABI a negative `maxmove` is that "no cap" choice;
    /// zero keeps the defaults.
    pub maxmove: Option<f64>,
    /// Norm a [`crate::Solver`] built from this control compares to
    /// [`Self::gtol`]. [`crate::Solver::set_stop_norm`] replaces it.
    /// Drivers that run no session keep `||g||_2`.
    pub stop_norm: StopNorm,
    /// Relative slack on every energy-decrease test. A trial at `ft`
    /// passes against the reference `ref_e` when
    /// `ft - ref_e <= ftol_rel * (|ref_e| + 1)`.
    ///
    /// This covers the line-search accept in every first-order method and
    /// the [`crate::Accept::Energy`] / [`crate::Accept::Nonmonotone`]
    /// backtracking. With `None`, a line search must lower the value or
    /// satisfy the Wolfe derivative check within its floating-point
    /// energy window; an accept policy tolerates a rise of at most
    /// 1e-8 absolute. Set it when the oracle returns an energy that is
    /// noisy at the level of the requested decrease, so a flat step is
    /// taken rather than refused.
    pub ftol_rel: Option<f64>,
    /// Gradient noise floor. `None` leaves the test off.
    pub noise_floor: Option<NoiseFloor>,
    /// When set, a finished run whose stop is unmet returns
    /// [`crate::Error::StopUnmet`] instead of the report. The default
    /// is off, so a spent `maxiter` is still `Ok` with
    /// [`crate::StopReason::MaxIter`].
    pub error_on_unmet: bool,
    /// When set, Newton and Hessian-vector minimizers finish with a
    /// backward-error polish. The default is off.
    pub backward_polish: bool,
    /// Stop when the energy stops changing between accepted steps.
    ///
    /// The LAMMPS `minimize etol` test (`src/min_cg.cpp`, `MinCG::iterate`),
    /// `|E_k - E_{k-1}| < etol * 0.5 * (|E_k| + |E_{k-1}| + eps_E)`, with
    /// `eps_E = sqrt(f64::EPSILON)` where LAMMPS writes `EPS_ENERGY = 1.0e-8`.
    /// `E_k` and `E_{k-1}` are the values at consecutive accepted points,
    /// so an iteration that left the point where it was is no test. SciPy
    /// `L-BFGS-B` `ftol` divides by `max(|f_k|, |f_{k+1}|, 1)`; the two agree
    /// for energies of magnitude 1 or more that change by little, and differ
    /// below 1, where LAMMPS divides by the energy and SciPy by 1. A
    /// gradient stop at the same point is reported instead, and the
    /// evaluation budget [`Self::max_evals`] is read first.
    /// [`crate::Status::EnergyTolerance`] is this stop.
    ///
    /// Must be finite and at least zero. Zero never stops, as in LAMMPS.
    /// `None`, the default, leaves the test off. A method with no energy
    /// sequence (particle swarm) refuses it with
    /// [`crate::Error::StopUnsupported`].
    pub etol: Option<f64>,
    /// Stop after this many objective evaluations.
    ///
    /// One evaluation is one call to the objective's value, gradient or
    /// fused value-and-gradient, a line-search trial and a finite-difference
    /// probe included. A Hessian or a Hessian-vector product is not one.
    /// The count is read when an iteration finishes, as LAMMPS reads
    /// `maxeval`, so the last iteration may overshoot by the evaluations
    /// one iteration costs. [`crate::Report::evals`] holds the count.
    /// [`crate::Status::EvalBudget`] is this stop. It must be at least 1.
    /// `None`, the default, leaves the budget off.
    pub max_evals: Option<usize>,
}

with_setters!(Control {
    with_maxiter => maxiter: val usize,
    with_gtol => gtol: val f64,
    with_istep => istep: val f64,
    with_maxmove => maxmove: opt f64,
    with_stop_norm => stop_norm: val StopNorm,
    with_ftol_rel => ftol_rel: opt f64,
    with_etol => etol: opt f64,
    with_max_evals => max_evals: opt usize,
});

impl Default for Control {
    fn default() -> Self {
        Self {
            maxiter: 100,
            gtol: 1e-5,
            istep: 1.0,
            maxmove: None,
            stop_norm: StopNorm::Two,
            ftol_rel: None,
            noise_floor: None,
            error_on_unmet: false,
            backward_polish: false,
            etol: None,
            max_evals: None,
        }
    }
}

/// Gradient norm a [`crate::Solver`] compares to [`Control::gtol`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StopNorm {
    /// `||g||_2`: eOn `norm`, LAMMPS `two`.
    #[default]
    Two,
    /// Largest Euclidean norm over consecutive `(x, y, z)` triples:
    /// ASE `fmax`, eOn `max_atom`, LAMMPS `max`. A trailing partial
    /// triple counts as one atom.
    MaxAtom,
    /// Largest absolute component: eOn `max_component`, LAMMPS `inf`.
    Inf,
    /// `||g||_2 / sqrt(n)`, zero for an empty `g`: eOn `rms`.
    Rms,
}

impl StopNorm {
    /// This norm of `g`. A non-finite entry gives infinity under
    /// [`Self::MaxAtom`], [`Self::Inf`] and [`Self::Rms`], so no
    /// tolerance accepts it.
    pub fn of(self, g: ArrayView1<f64>) -> f64 {
        match self {
            Self::Two => crate::vecops::nrm2(g),
            Self::Inf => crate::vecops::nrminf(g),
            Self::Rms if g.is_empty() => 0.0,
            Self::Rms if crate::vecops::nrminf(g).is_infinite() => f64::INFINITY,
            Self::Rms => crate::vecops::nrm2(g) / (g.len() as f64).sqrt(),
            Self::MaxAtom => {
                let mut worst = 0.0_f64;
                let mut i = 0;
                while i < g.len() {
                    let end = (i + 3).min(g.len());
                    let r2: f64 = (i..end).map(|k| g[k] * g[k]).sum();
                    if !r2.is_finite() {
                        return f64::INFINITY;
                    }
                    worst = worst.max(r2.sqrt());
                    i = end;
                }
                worst
            }
        }
    }
}

impl Control {
    /// [`Self::maxmove`] when it is a positive finite length, the only
    /// value that limits a step or a trust radius.
    pub(crate) fn maxmove_cap(&self) -> Option<f64> {
        self.maxmove.filter(|&cap| crate::step::binding(cap))
    }

    /// Return [`crate::Error::StopUnmet`] when a finished run misses its stop.
    ///
    /// [`crate::StopReason::Converged`], [`crate::StopReason::NoiseFloor`],
    /// [`crate::StopReason::StepTolerance`] and
    /// [`crate::StopReason::EnergyTolerance`] stay `Ok`. The default
    /// leaves this off.
    #[must_use]
    pub fn with_error_on_unmet(mut self, on: bool) -> Self {
        self.error_on_unmet = on;
        self
    }

    /// Finish a Newton or Hessian-vector minimization with a
    /// backward-error polish.
    ///
    /// The polish accepts a point only when the Newton decrement and
    /// every step component meet the roundoff test in
    /// [`crate::backward_error_floor`]'s family. An unmet polish is
    /// [`crate::StopReason::PolishUnmet`]. The default leaves this off,
    /// so `gtol` remains the stop.
    #[must_use]
    pub fn with_backward_polish(mut self, on: bool) -> Self {
        self.backward_polish = on;
        self
    }

    /// `Ok(report)`, or [`crate::Error::StopUnmet`] when
    /// [`Self::error_on_unmet`] is set and the stop did not hold.
    pub(crate) fn deliver(
        &self,
        report: crate::report::Report,
    ) -> crate::Result<crate::report::Report> {
        if self.error_on_unmet && report.stop_reason().is_unmet() {
            let reason = report.stop_reason();
            Err(crate::Error::StopUnmet {
                reason,
                report: Box::new(report),
            })
        } else {
            Ok(report)
        }
    }

    /// Opt in to the noise floor.
    ///
    /// The iteration stops when the stationarity measure is below
    /// `c * sqrt(n_free) * eps_g` and still at or above [`Self::gtol`].
    /// `n_free` is the number of coordinates that measure runs over.
    /// [`crate::Status::NoiseFloor`] is that stop.
    /// [`crate::Status::Tolerance`] remains the `gtol` stop. `None`
    /// turns it off again.
    #[must_use]
    pub fn with_noise_floor(mut self, floor: impl Into<Option<NoiseFloor>>) -> Self {
        self.noise_floor = floor.into();
        self
    }

    /// `Tolerance` when `grad` is below `gtol`, `NoiseFloor` when it is
    /// below `c * sqrt(n_free) * eps_g` and still at or above `gtol`,
    /// otherwise `Iterating`.
    ///
    /// `n_free` is the length of the gradient the stationarity measure
    /// uses, with frozen or fixed coordinates left out.
    pub fn classify(&self, grad: f64, n_free: usize) -> crate::report::Status {
        use crate::report::Status;
        if grad < self.gtol {
            return Status::Tolerance;
        }
        if let Some(floor) = self.noise_floor {
            let eps = floor.eps_g;
            let width = (n_free as f64).sqrt();
            if eps.is_finite()
                && eps >= 0.0
                && floor.c.is_finite()
                && width.is_finite()
                && grad < floor.c * width * eps
            {
                return Status::NoiseFloor;
            }
        }
        Status::Iterating
    }

    /// The stop to take now, when one is due.
    pub fn halt(&self, grad: f64, n_free: usize) -> Option<crate::report::Status> {
        match self.classify(grad, n_free) {
            status @ (crate::report::Status::Tolerance | crate::report::Status::NoiseFloor) => {
                Some(status)
            }
            _ => None,
        }
    }

    /// [`Self::classify`], with a spent budget reported as
    /// [`crate::report::Status::Budget`] when `exhausted` is set and the
    /// gradient is still above every stop.
    pub(crate) fn finish_status(
        &self,
        grad: f64,
        n_free: usize,
        exhausted: bool,
    ) -> crate::report::Status {
        match self.classify(grad, n_free) {
            crate::report::Status::Iterating if exhausted => crate::report::Status::Budget,
            other => other,
        }
    }

    /// Refuse an [`Self::etol`] that is negative or not finite and a
    /// [`Self::max_evals`] of zero, with [`crate::Error::InvalidControl`].
    ///
    /// Every driver and session step calls this before it evaluates the
    /// objective.
    pub fn validate(&self) -> crate::error::Result<()> {
        crate::stop::validate(self)
    }

    /// Whether `|current - previous| < etol * 0.5 * (|current| + |previous| + eps_E)`
    /// for the [`Self::etol`] set here, the test a driver applies between
    /// accepted steps. `false` when `etol` is `None`.
    pub fn energy_converged(&self, previous: f64, current: f64) -> bool {
        self.etol
            .is_some_and(|etol| crate::stop::energy_converged(etol, previous, current))
    }

    /// Absolute slack an energy test grants at reference `ref_e`:
    /// `ftol_rel * (|ref_e| + 1)`, or 0 when [`Self::ftol_rel`] is unset.
    pub(crate) fn ftol_slack(&self, ref_e: f64) -> f64 {
        match self.ftol_rel {
            Some(t) if t > 0.0 => t * (ref_e.abs() + 1.0),
            _ => 0.0,
        }
    }
}
