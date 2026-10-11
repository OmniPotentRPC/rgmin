//! Iteration limits and gradient tolerance.

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
    /// Stop when `||g||_2 < gtol`.
    pub gtol: f64,
    /// The opening step of a line search. BFGS, SR1 and SR2 open every
    /// line search here, since their direction carries the scale and 1.0
    /// is the natural trial; L-BFGS opens here while it holds no pairs and
    /// at 1.0 once it does, whatever this is set to; steepest descent,
    /// NLCG and Adam open here once and then at half the step last
    /// accepted.
    pub istep: f64,
    /// Optional Euclidean cap on a proposed step (xtsci `maxmove`).
    /// `Some(c)` with `c <= 0` or NaN is no cap, the same as `None`; a
    /// non-positive cap never scales a step, so it cannot reverse one.
    pub maxmove: Option<f64>,
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
            ftol_rel: None,
            noise_floor: None,
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
