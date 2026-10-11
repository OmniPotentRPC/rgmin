//! Iteration limits and gradient tolerance.

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
    /// Gradient noise `eps_g`. `None` leaves the noise floor off.
    pub noise_eps: Option<f64>,
    /// Factor in the floor `c * eps_g`. The default is 3.
    pub noise_c: f64,
}

impl Default for Control {
    fn default() -> Self {
        Self {
            maxiter: 100,
            gtol: 1e-5,
            istep: 1.0,
            maxmove: None,
            ftol_rel: None,
            noise_eps: None,
            noise_c: 3.0,
        }
    }
}

impl Control {
    /// Set the noise-floor factor `c`. The floor stays off until
    /// [`Self::noise_eps`] is `Some`.
    pub fn noise_floor(mut self, c: f64) -> Self {
        self.noise_c = c;
        self
    }

    /// Opt in to the noise floor with gradient noise `eps_g`.
    ///
    /// The iteration stops when the gradient measure is below
    /// `noise_c * eps_g` and still above [`Self::gtol`]. The default
    /// factor is 3. [`crate::Status::NoiseFloor`] is that stop.
    /// [`crate::Status::Tolerance`] remains the `gtol` stop.
    pub fn with_noise_eps(mut self, eps_g: f64) -> Self {
        self.noise_eps = Some(eps_g);
        self
    }

    /// `Tolerance` when `grad` is below `gtol`, `NoiseFloor` when it is
    /// below `c * eps_g` and still above `gtol`, otherwise `Iterating`.
    pub fn classify(&self, grad: f64) -> crate::report::Status {
        use crate::report::Status;
        if grad < self.gtol {
            return Status::Tolerance;
        }
        if let Some(eps) = self.noise_eps {
            if eps.is_finite() && eps >= 0.0 && grad < self.noise_c * eps {
                return Status::NoiseFloor;
            }
        }
        Status::Iterating
    }

    /// The stop to take now, when one is due.
    pub fn halt(&self, grad: f64) -> Option<crate::report::Status> {
        match self.classify(grad) {
            status @ (crate::report::Status::Tolerance | crate::report::Status::NoiseFloor) => {
                Some(status)
            }
            _ => None,
        }
    }

    /// [`Self::classify`], with a spent budget reported as
    /// [`crate::report::Status::Budget`] when `exhausted` is set and the
    /// gradient is still above every stop.
    pub(crate) fn finish_status(&self, grad: f64, exhausted: bool) -> crate::report::Status {
        match self.classify(grad) {
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
