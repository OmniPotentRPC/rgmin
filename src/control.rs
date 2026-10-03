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
    pub maxmove: Option<f64>,
    /// Relative slack on every energy-decrease test. A trial at `ft`
    /// passes against the reference `ref_e` when
    /// `ft - ref_e <= ftol_rel * (|ref_e| + 1)`.
    ///
    /// This covers the line-search accept in every first-order method and
    /// the [`crate::Accept::Energy`] / [`crate::Accept::Nonmonotone`]
    /// backtracking. `None` keeps the strict test: a line search must
    /// lower the value, and an accept policy tolerates a rise of at most
    /// 1e-8 absolute. Set it when the oracle returns an energy that is
    /// noisy at the level of the requested decrease, so a flat step is
    /// taken rather than refused.
    pub ftol_rel: Option<f64>,
}

impl Default for Control {
    fn default() -> Self {
        Self {
            maxiter: 100,
            gtol: 1e-5,
            istep: 1.0,
            maxmove: None,
            ftol_rel: None,
        }
    }
}

impl Control {
    /// Absolute slack an energy test grants at reference `ref_e`:
    /// `ftol_rel * (|ref_e| + 1)`, or 0 when [`Self::ftol_rel`] is unset.
    pub(crate) fn ftol_slack(&self, ref_e: f64) -> f64 {
        match self.ftol_rel {
            Some(t) if t > 0.0 => t * (ref_e.abs() + 1.0),
            _ => 0.0,
        }
    }
}
