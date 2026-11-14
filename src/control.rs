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
    pub fn with_c(mut self, c: f64) -> Self {
        self.c = c;
        self
    }

    /// Record the energy noise from the same stencil as `eps_g`.
    pub fn with_eps_f(mut self, eps_f: f64) -> Self {
        self.eps_f = Some(eps_f);
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
#[derive(Clone, Debug)]
pub struct Control {
    /// Maximum CG iterations.
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
    /// `Some(c)` with `c <= 0` or NaN is no cap, the same as `None`; a
    /// non-positive cap never scales a step, so it cannot reverse one.
    /// A positive value is the step limit, and the per-atom default stays
    /// off unless the caller sets a per-atom cap. On the C ABI a negative
    /// `maxmove` is this "no cap" choice, including the default per-atom
    /// cap; zero keeps the defaults.
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
}

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
    /// Opt in to the noise floor.
    ///
    /// The iteration stops when the stationarity measure is below
    /// `c * sqrt(n_free) * eps_g` and still at or above [`Self::gtol`].
    /// `n_free` is the number of coordinates that measure runs over.
    /// [`crate::Status::NoiseFloor`] is that stop.
    /// [`crate::Status::Tolerance`] remains the `gtol` stop.
    pub fn with_noise_floor(mut self, floor: NoiseFloor) -> Self {
        self.noise_floor = Some(floor);
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

    /// Absolute slack an energy test grants at reference `ref_e`:
    /// `ftol_rel * (|ref_e| + 1)`, or 0 when [`Self::ftol_rel`] is unset.
    pub(crate) fn ftol_slack(&self, ref_e: f64) -> f64 {
        match self.ftol_rel {
            Some(t) if t > 0.0 => t * (ref_e.abs() + 1.0),
            _ => 0.0,
        }
    }
}
