//! Errors from a minimization.

use thiserror::Error;

/// Recoverable minimization failure.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// Initial point length does not match the objective dimension.
    #[error("init length {got} != objective dim {dim}")]
    Dim {
        /// Length of the supplied start vector.
        got: usize,
        /// `Objective::dim`.
        dim: usize,
    },
    /// HiGHS rejected the L-BFGS quadratic model.
    #[error("HiGHS: {0}")]
    Highs(String),
    /// Newton / RFO / dogleg needs a Hessian oracle.
    #[error("Newton/RFO/dogleg needs a Hessian; call step_hess")]
    NeedHessian,
    /// Packed manifold rejected this ambient dimension.
    #[error("{kind} rejected dimension {got}")]
    ManifoldDim {
        /// Token (`so3`, `se3`, `rigid_quotient`, `mw_rigid`).
        kind: &'static str,
        /// Length of the working vector.
        got: usize,
    },
    /// SCG cannot make progress (non-finite objective everywhere it
    /// can step, or damping at its limit).
    #[error("SCG stalled: {what}")]
    ScgStalled {
        /// What exhausted the algorithm.
        what: &'static str,
    },
    /// FIRE 2.0 exceeded its consecutive non-positive-power limit.
    #[error("FIRE 2.0 stalled after {steps} steps")]
    FireStalled {
        /// Outer iterations completed before the limit was exceeded.
        steps: usize,
    },
    /// `Accept::Energy` or `Accept::Nonmonotone` refused the steepest
    /// restart that followed a refused step. Every later call would
    /// retry it from the same point.
    #[error("energy test refused the steepest restart after {steps} steps")]
    AcceptStalled {
        /// Outer iterations completed before the second refusal.
        steps: usize,
    },
    /// The trust radius fell below its floor. Newton-CG reaches it
    /// without an acceptable step: a non-finite gradient, a broken
    /// curvature action, or an objective that rejects every trial the
    /// model proposes. Session L-BFGS under
    /// [`crate::Solver::set_lbfgs_trust`] also returns it when an
    /// accepted ratio leaves the radius below `1e-15`, and discards that
    /// trial.
    #[error("trust region collapsed after {steps} steps")]
    TrustCollapsed {
        /// Outer iterations completed when the radius hit its floor.
        steps: usize,
    },
    /// Session L-BFGS under [`crate::Solver::set_lbfgs_trust`] found no
    /// downhill direction: the slope `g^T d` was finite and not negative
    /// even after the curvature pairs were dropped. An exactly zero
    /// gradient with `gtol <= 0` does this. A NaN or infinite slope is
    /// [`Error::Oracle`] instead.
    #[error("no descent direction after {steps} steps")]
    NoDescent {
        /// Outer iterations completed before the refusal.
        steps: usize,
    },
    /// Host oracle returned a non-finite value or gradient.
    #[error("oracle: {what}")]
    Oracle {
        /// What the host callback failed to produce.
        what: &'static str,
    },
    /// The exponential preconditioner rejected its inputs or its μ estimate.
    #[error("exp preconditioner: {what}")]
    Precon {
        /// What the assembly or the estimate refused.
        what: &'static str,
    },
    /// Named eigensolver is not linked in this build. Fail closed:
    /// the waist does not silently fall back to Lanczos.
    #[error("eigensolver {kind} is not linked in this build")]
    EigenUnavailable {
        /// Closed-enum name (`elpa`, `primme`, ...).
        kind: &'static str,
    },
    /// [`crate::Method::from_name`] did not recognize the token.
    #[error("unknown method '{name}'")]
    UnknownMethod {
        /// The text the caller passed.
        name: String,
    },
    /// [`crate::Solver::step_hvp`] was used on a session that is not
    /// Newton-CG or Newton-PCG.
    #[error("hessian-vector step requires newton-cg or newton-pcg")]
    NeedHvp,
}

/// Result alias for this crate.
pub type Result<T> = std::result::Result<T, Error>;
