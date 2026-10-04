//! Errors from a minimization.

use thiserror::Error;

/// Recoverable minimization failure.
#[derive(Debug, Error)]
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
    /// Mass table length is not one mass per atom.
    #[error("mass count {got} is not one mass per atom for dimension {dim}")]
    MassCount {
        /// Number of masses supplied.
        got: usize,
        /// Coordinate length. One mass covers three coordinates.
        dim: usize,
    },
    /// SCG cannot make progress (non-finite objective everywhere it
    /// can step, or damping at its limit).
    #[error("SCG stalled: {what}")]
    ScgStalled {
        /// What exhausted the algorithm.
        what: &'static str,
    },
    /// The trust region collapsed without an acceptable step: a
    /// non-finite gradient, a broken curvature action, or an
    /// objective that rejects every trial the model proposes.
    #[error("trust region collapsed after {steps} steps")]
    TrustCollapsed {
        /// Outer iterations completed when the radius hit its floor.
        steps: usize,
    },
    /// Host oracle returned a non-finite value or gradient.
    #[error("oracle: {what}")]
    Oracle {
        /// What the host callback failed to produce.
        what: &'static str,
    },
    /// The dense QN family cannot resolve the requested trust radius.
    #[error("restricted QN step did not converge")]
    RestrictedStep,
    /// Named eigensolver is not linked in this build.
    #[error("eigensolver {kind} is not linked in this build")]
    EigenUnavailable {
        /// Closed-enum name of the requested backend.
        kind: &'static str,
    },
    /// Linked SLEPc EPS rejected the typed configuration or the pair.
    #[error("SLEPc: {what}")]
    Slepc {
        /// What the typed EPS/ST call failed to produce.
        what: &'static str,
    },
    /// Linked PRIMME `dprimme` rejected the typed configuration or the pair.
    #[error("PRIMME: {what}")]
    Primme {
        /// What the typed `primme_params` / `dprimme` call failed to produce.
        what: &'static str,
    },
    /// Linked ChASE `dchase` rejected the assembled dense pair.
    #[error("ChASE: {what}")]
    Chase {
        /// What the typed `dchase_init_` / `dchase_` call failed to produce.
        what: &'static str,
    },
    /// Linked libkrylov `ckrylov_solve_real_equation` rejected the pair.
    #[error("libkrylov: {what}")]
    Libkrylov {
        /// What the typed ckrylov call failed to produce.
        what: &'static str,
    },
    /// Partial-spectrum window must start at the lowest pair.
    #[error("eigensolver {kind} begin must be 0, got {begin}")]
    EigenBegin {
        /// Closed-enum name (`dlaFuture`, ...).
        kind: &'static str,
        /// Requested first index.
        begin: usize,
    },
    /// Dense assembled-H backend is gated on [`crate::DENSE_EIGEN_CUTOFF`].
    #[error("eigensolver {kind} needs n >= {cutoff}, got {n}")]
    EigenDenseCutoff {
        /// Closed-enum name (`dlaFuture`, ...).
        kind: &'static str,
        /// Matrix order.
        n: usize,
        /// Cutoff (`DENSE_EIGEN_CUTOFF`).
        cutoff: usize,
    },
    /// Quick-min was asked for something its session does not hold.
    #[error("quick-min: {what}")]
    QuickMin {
        /// What the call refused.
        what: &'static str,
    },
    /// Named backend only computes the full spectrum. Partial `nev`
    /// is refused rather than silently solved as `n` and trimmed.
    #[error("eigensolver {kind} is full-spectrum only; nev {nev} < n {n}")]
    EigenFullSpectrum {
        /// Closed-enum name (`eigenExa`, ...).
        kind: &'static str,
        /// Requested pair count.
        nev: usize,
        /// Matrix order.
        n: usize,
    },
}

/// Result alias for this crate.
pub type Result<T> = std::result::Result<T, Error>;
