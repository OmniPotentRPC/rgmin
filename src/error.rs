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
    /// A manifold factory refused its shape.
    #[error("{kind} rejected shape n = {n}, p = {p}")]
    ManifoldShape {
        /// Token (`stiefel`, `grassmann`, `unitary`).
        kind: &'static str,
        /// Ambient size.
        n: usize,
        /// Column or subspace count; `n` for a square factor.
        p: usize,
    },
    /// The DLPack device tag names a backend this build does not carry.
    #[error("DLPack device {device_type} has no kernel backend in this build (CPU only)")]
    UnsupportedDevice {
        /// The DLPack `DLDeviceType` discriminant that was refused.
        device_type: i32,
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
    /// The trust region collapsed without an acceptable step: a
    /// non-finite gradient, a broken curvature action, or an
    /// objective that rejects every trial the model proposes.
    #[error("trust region collapsed after {steps} steps")]
    TrustCollapsed {
        /// Outer iterations completed when the radius hit its floor.
        steps: usize,
    },
    /// Session L-BFGS under [`crate::Solver::set_lbfgs_trust_radius`]
    /// found no downhill direction: the slope `g^T d` was finite and
    /// not negative even after the curvature pairs were dropped. An
    /// exactly zero gradient with `gtol <= 0` does this. A NaN or
    /// infinite slope is [`Error::Oracle`] instead.
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
    /// The dense QN family cannot resolve the requested trust radius.
    #[error("restricted QN step did not converge")]
    RestrictedStep,
    /// Named eigensolver is not linked in this build. Fail closed:
    /// the waist does not silently fall back to Lanczos.
    #[error("eigensolver {kind} is not linked in this build")]
    EigenUnavailable {
        /// Closed-enum name of the requested backend.
        kind: &'static str,
    },
    /// Block width is not in `1..=dim`.
    #[error("eigenpairs: k = {k} is outside 1..={dim}")]
    EigenBlock {
        /// Requested number of pairs.
        k: usize,
        /// Operator dimension.
        dim: usize,
    },
    /// A block eigenpair argument was non-finite or otherwise unusable.
    #[error("eigenpairs: {what}")]
    EigenInput {
        /// What was refused.
        what: &'static str,
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
    /// A geometry preconditioner rejected its inputs or its μ estimate.
    #[error("preconditioner: {what}")]
    Precon {
        /// What the assembly or the estimate refused.
        what: &'static str,
    },
    /// ECnoise was given a sample or a line probe it cannot run.
    #[error("ecnoise: {what}")]
    EcNoise {
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
    /// A CON checkpoint could not be written or read.
    #[error("checkpoint: {0}")]
    Checkpoint(String),
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
    /// A FIRE 2.0 preset or ABC-FIRE was set on a session that is not
    /// `fire2`.
    #[error("FIRE 2.0 preset or ABC requires fire2")]
    NeedFire2,
    /// A FIRE parameter change was asked of a session that is not FIRE.
    #[error("FIRE parameters require a FIRE session")]
    NeedFire,
    /// Capped L-BFGS was asked for where it cannot run, or a capped
    /// step could not be formed.
    #[error("capped L-BFGS: {what}")]
    LbfgsCap {
        /// What was refused.
        what: &'static str,
    },
    /// A FIRE step or an L-BFGS step under `Accept::None` on a curved
    /// kind found no retracted trial inside the box and the displacement
    /// cap.
    #[error("no trial on {kind} fits the box and the displacement cap")]
    OffManifold {
        /// Manifold token.
        kind: &'static str,
    },
    /// A distance constraint set was malformed, refused by the session,
    /// or could not be met.
    #[error("distance constraints: {0}")]
    Constraint(#[from] crate::holonomic::ConstraintError),
    /// [`crate::lbfgs_direction`] refused a gradient, a pair, or `H0`.
    #[error("L-BFGS direction: {what}")]
    LbfgsDirection {
        /// What was refused.
        what: &'static str,
    },
    /// [`crate::RiemannianManifold`] refused a shape, a point or a step.
    #[error("manifold: {0}")]
    ManifoldArgument(String),
    /// [`crate::manifold::ProductManifold::new`] refused a factor.
    #[error("product factor {index}: {what}")]
    ProductFactor {
        /// Position of the factor in the list.
        index: usize,
        /// What was refused.
        what: &'static str,
    },
    /// A setting reads raw coordinates, which the metric chart of a
    /// weighted product rescales.
    #[error("weighted product: {what} reads unscaled coordinates")]
    ProductChart {
        /// The refused setting.
        what: &'static str,
    },
    /// A session measures in the Euclidean inner product of the packed
    /// vector or of the metric chart, which a
    /// [`crate::manifold::TangentMetric`] replaces.
    /// [`crate::rtr::minimize_rtr`] runs in that metric.
    #[error("product factor {index} carries a tangent metric, which sessions do not measure in")]
    ProductTangentMetric {
        /// Position of the first such factor after flattening.
        index: usize,
    },
    /// [`crate::manifold::SubspaceSphere`] refused its basis.
    #[error("subspace sphere: {what}")]
    SubspaceBasis {
        /// What was refused.
        what: &'static str,
    },
    /// A level-set constraint \(g(x)=c\) was configured or applied with
    /// invalid data.
    #[error("level set: {what}")]
    LevelSet {
        /// What was refused.
        what: &'static str,
    },
    /// The iteration finished with the stop unmet.
    ///
    /// Opt-in via [`crate::Control::with_error_on_unmet`]. The default
    /// still returns the [`crate::Report`]. `report` holds the value,
    /// coordinates, gradient norm and step count at the stop.
    #[error(
        "stop {reason} unmet after {} steps (grad_norm {})",
        report.steps,
        report.grad_norm
    )]
    StopUnmet {
        /// Why the run stopped.
        reason: crate::report::StopReason,
        /// Point where the solver stopped.
        report: Box<crate::report::Report>,
    },
    /// Newton retraction to a level set stopped when the residual no
    /// longer contracted relative to the derived tolerance.
    #[error(
        "level set: retraction stalled after {iterations} iterations (residual {residual}, previous {previous_residual}, tolerance {tolerance})"
    )]
    LevelSetStalled {
        /// Newton steps completed.
        iterations: usize,
        /// Final residual `g(x) - c`.
        residual: f64,
        /// Previous absolute residual.
        previous_residual: f64,
        /// Derived convergence tolerance at that point.
        tolerance: f64,
    },
}

/// Result alias for this crate.
pub type Result<T> = std::result::Result<T, Error>;
