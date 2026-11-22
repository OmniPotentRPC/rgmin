//! Local first-order minimization over eindir objectives.
//!
//! This crate is the Rust rewrite of the C++ xtsci-optimize (its
//! xtensor heritage is where the old name came from). Conjugacy,
//! restart, and line search stay independent. Nonlinear CG is Nocedal and
//! Wright algorithm 5.4. Quasi-Newton methods (BFGS, L-BFGS, SR1, SR2),
//! Adam, steepest descent, and particle swarm share the same eindir
//! `DifferentiableObjective` handle. C and C++ reach these solvers
//! through an `rgmin_solver_t` session (`create` / `step` / `free`)
//! and the one-shot `rgmin_minimize` wrappers.
//!
//! The production unconstrained local method is [`Lbfgs`] with the eOn
//! cap ([`crate::LbfgsStep::Capped`] at [`EON_MAX_MOVE`]): limited-memory
//! BFGS (Nocedal-Wright 7.4) and no line search. The strong Wolfe
//! conditions (algorithms 3.5 and 3.6) are
//! [`crate::LbfgsStep::Wolfe`], and that search is opt-in.
//! [`lbfgs_direction`] applies a caller-supplied initial inverse Hessian
//! and does not evaluate an objective. [`minimize_lbfgs`] is the
//! cold-start dispatch used by [`minimize_method`].
//! [`minimize_rlbfgs`] is cautious limited-memory Riemannian BFGS. It
//! does not replace [`Lbfgs`].

#![warn(missing_docs)]
// Numerical kernels take their state as explicit arguments, and the
// small dense algebra indexes by position on purpose.
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

#[macro_use]
mod config;

/// Line-search strategies (Brent, Armijo, Goldstein, Wolfe/zoom).
pub mod linesearch;
/// Method selector (NLCG, BFGS family, Adam, steepest descent, PSO).
pub mod method;
/// Nonlinear conjugate-gradient conjugacy and restart.
pub mod nlcg;

/// Complex-contour inverse Laplace transforms (fixed-Talbot and Weeks/Laguerre)
/// and orthonormal Laguerre series evaluation.
pub mod inverse_laplace;

mod accept;
mod adam;
mod bb;
#[cfg(feature = "highs")]
mod box_objective;
/// A-posteriori Newton--Kantorovich, adjoint, and Morse-gap bounds.
pub mod certificate;
mod compact_tr;
/// CON checkpoint of an accepted point. Compiled only with `readcon`.
#[cfg(feature = "readcon")]
pub mod con_checkpoint;
mod control;
mod diis;
mod eon_cg;
mod error;
/// C ABI, gated behind the `capi` feature.
#[cfg(feature = "capi")]
#[allow(non_camel_case_types, missing_docs)]
pub mod ffi;
/// FIRE / FIRE 2.0 inertial first-order steps.
pub mod fire;
/// Gaussian-process-accelerated local geometry optimizer (GPMin).
pub mod gpmin;
/// GROMACS `integrator = steep` and `integrator = cg` rules.
pub mod gromacs;
/// Holonomic distance constraints (SHAKE, LINCS, SETTLE and RATTLE).
pub mod holonomic;
/// Matrix-free Newton: Hessian actions and Steihaug-Toint CG.
pub mod hvp;
/// Sella IRCTrustRegion / Gonzalez--Schlegel mass-weighted sphere.
pub mod irc_trust;
/// Persistent L-BFGS (Nocedal-Wright 7.4): the eOn cap by default,
/// strong Wolfe on request.
pub mod lbfgs;
mod lbfgs_plateau;
/// L-BFGS quadratic model solved by HiGHS.
#[cfg(feature = "highs")]
pub mod lbfgs_qp;
mod level_set;
/// Sparse Lindh model Hessian (`H0^{-1}` apply). The session preconditioner
/// is [`precon::Lindh`].
pub mod lindh;
/// Matrix-free lowest Hessian eigenpair and the block of k lowest pairs.
pub mod lowest_mode;
/// Sixth-order Magnus steps and the Padé matrix exponential.
pub mod magnus;
/// Matrix-free preconditioned MINRES with true-residual certification.
pub mod minres;
/// Moré and Wild computational-noise estimate and difference steps.
pub mod noise;
mod portable;

mod chase_eps;
/// DLA-Future partial-spectrum window.
pub mod dlaf_kind;
/// Closed EigenExa algorithm selection.
pub mod eigenexa_kind;
/// ELPA stage and spectrum settings.
pub mod elpa_kind;
/// Typed HiGHS solver and callback settings.
pub mod highs_kind;
#[cfg(feature = "libkrylov")]
mod libkrylov_eps;
#[cfg(feature = "primme")]
mod primme_eps;
#[cfg(feature = "slepc")]
mod slepc_eps;
/// Closed options for optional lowest-mode backends.
pub mod slepc_kind;

/// Embedded Riemannian manifolds (manopt_cpp proj / retr / transp).
pub mod manifold;
mod minimize;
/// Shifted Newton and Banerjee RFO on a dense Hessian.
pub mod newton;
/// ODE12r: adaptive first-order integration of the force.
pub mod ode12r;
mod oracle;
/// Bordered cyclic block determinants without the period product.
pub mod periodic;
/// Linear-storage factorization of periodic block-tridiagonal matrices.
pub mod periodic_block;
/// Geometry preconditioners: Exp, Lindh, and the bonded FF metric.
pub mod precon;
mod pso;
mod qn;
/// Sella QuasiNewton / QuasiNewtonIRC restricted step.
pub mod qn_irc;
mod qn_step;
/// Gauss–Legendre rules and Lobatto shape functions on `[-1, 1]`.
pub mod quadrature;
/// Quick-min projection and Euler step.
pub mod quickmin;
mod report;
mod restricted_trust;
/// Eckart projector and Horn / Coutsias alignment.
pub mod rigid;
/// Cautious limited-memory Riemannian BFGS (Huang, Absil, Gallivan).
pub mod rlbfgs;
/// Bracketed scalar roots by bisection.
pub mod root;
/// Tangent-space truncated conjugate gradients and trust-radius updates.
pub mod rtr;
/// Moller scaled conjugate gradient (damped-model step, no line search).
pub mod scg;
/// Symmetric secant updates (PSB, SR1, Bofill) of a dense Hessian model.
pub mod secant;
/// Sella RFO, P-RFO, and restricted atomic step.
pub mod sella_step;
mod session;
/// Matrix-free Krylov primitives: preconditioned CG, deflated thick-restart
/// Lanczos, Gauss/Gauss-Radau Lanczos quadrature, and deflated SLQ log-determinant.
pub mod slq;
mod sphere_lbfgs;
mod step;
/// One outer iteration. rgsaddle calls this; the host owns the loop.
pub mod stepper;
mod stop;
/// Dense dogleg and Sella trust-region steps.
pub mod trust;
/// Training-set memory and cap selection.
pub mod tune;
/// The vector seam: solver algebra behind one interface.
pub mod vecops;

pub use accept::Accept;
pub use adam::minimize_adam;
pub use certificate::{
    AdjointFunctionalBar, CertificateError, KantorovichBall, MorseGap, adjoint_functional_bar,
    certify_morse_gap, kantorovich_ball,
};
#[cfg(feature = "readcon")]
pub use con_checkpoint::{read_checkpoint, write_checkpoint};
pub use control::{Control, NoiseFloor, StopNorm};
pub use dlaf_kind::DlaFutureParams;
pub use eigenexa_kind::{EigenExaAlgo, EigenExaParams};
pub use elpa_kind::{ElpaParams, ElpaSpectrum, ElpaStage, elpa_config, elpa_spectrum};
pub use eon_cg::{EonCg, EonCgReport, minimize_eon_cg};
pub use error::{Error, Result};
pub use fire::{
    Fire2Extras, FireConfig, FireKind, FirePreset, FireTrace, FireVariant, guenole2020,
};
pub use gpmin::{
    GPMIN_DEFAULT_HYPERPARAM_MAXITER, GPMIN_DEFAULT_HYPERPARAM_TOL, GPMIN_DEFAULT_MAX_REFINEMENTS,
    GPMIN_DEFAULT_NOISE, GPMIN_DEFAULT_SCALE, GPMIN_DEFAULT_SURROGATE_MAXITER,
    GPMIN_DEFAULT_SURROGATE_MEMORY, GPMIN_DEFAULT_WEIGHT, GPMIN_UPDATED_BATCH_SIZE,
    GPMIN_UPDATED_BOUNDS, GPMIN_UPDATED_NOISE, GPMIN_UPDATED_SCALE, GPMIN_UPDATED_WEIGHT,
    GaussianProcess, GpMin, GpMinConfig, PriorStrategy, SquaredExponentialKernel,
    default_surrogate_gtol,
};
pub use gromacs::{
    GROMACS_CG_BETA_CAP, GROMACS_CG_EMSTEP_DEFAULT, GROMACS_CG_NSTCGSTEEP_DEFAULT,
    GROMACS_CG_WOLFE_C1, GROMACS_CG_WOLFE_C2, GROMACS_CG_WOLFE_MAXITER,
    GROMACS_STEEP_EMSTEP_DEFAULT, GROMACS_STEEP_GROW, GROMACS_STEEP_MAX_CONSTRAINT_SHRINKS,
    GROMACS_STEEP_MIN_USTEP, GROMACS_STEEP_SHRINK, GROMACS_STEEP_SHRINK_BUDGET_CAP, GromacsCg,
    GromacsCgConfig, GromacsSteep, GromacsSteepConfig, gromacs_steep_max_constraint_shrinks,
};
pub use highs_kind::{
    HighsCCallback, HighsCallback, HighsCallbackKind, HighsCrossover, HighsOptions, HighsSolverKind,
};
pub use holonomic::{
    ConstraintAlgorithm, ConstraintError, ConstraintReport, ConstraintSolver, DistanceConstraints,
};
pub use hvp::{
    FdHvp, HessianVector, HvpOracle, IdentityPrecond, NystromPrecond, Preconditioner,
    TryHessianVector, minimize_newton_cg, polish_hvp, steihaug_cg, steihaug_pcg,
};
pub use irc_trust::{IrcTrust, sqrt_masses_3n};
pub use lbfgs::{
    EON_FINITE_DIFFERENCE, EON_INVERSE_CURVATURE, EON_MAX_MOVE, GradNorm, LBFGSB_FACTR,
    LBFGSB_FTOL, Lbfgs, LbfgsBuilder, LbfgsH0, LbfgsPairs, LbfgsStep, lbfgs_direction,
};
#[cfg(feature = "highs")]
pub use lbfgs_qp::HighsStep;
pub use level_set::LevelSetConstraint;
pub use lindh::{LindhSpec, apply_inverse};
pub use linesearch::{LineSearch, LineSearchOptions, SearchStep, SuppliedSearch};
pub use lowest_mode::{
    ApplyBlockPreconditioner, ApplyHessian, ApplyPreconditioner, Block3Jacobi, DENSE_EIGEN_CUTOFF,
    DiagonalJacobi, EigenParams, Eigenpair, Eigenpairs, EigensolverKind, LowestMode,
    PreconditionerKind, TryApplyHessian, lowest_eigenpair, lowest_eigenpairs,
    lowest_eigenpairs_precond, lowest_mode, lowest_mode_chase, lowest_mode_dense, lowest_mode_dlaf,
    lowest_mode_eigenexa, lowest_mode_libkrylov, lowest_mode_precond, lowest_mode_precond_with,
    lowest_mode_primme, lowest_mode_slepc, lowest_mode_with, try_lowest_eigenpairs,
    try_lowest_eigenpairs_precond,
};
pub use manifold::{
    CenteredMatrix, ComplexCircle, Constant, EuclideanComplex, Grassmann, Hyperbolic, Manifold,
    ManifoldFactor, ManifoldKind, Multinomial, MultinomialDoublyStochastic, MultinomialSymmetric,
    Oblique, PoincareBall, Positive, RiemannianManifold, SkewSymmetric, Spd, SpdMetric,
    SphereComplex, StiefelNp, StiefelRetraction, Symmetric, Unitary,
};
pub use method::Method;
pub use minimize::{minimize, minimize_method, minimize_method_hess, minimize_method_hvp};
pub use minres::{
    BoundedKrylovSolve, MinresError, MinresFailure, MinresVector, minres_solve_bounded,
    minres_solve_precond,
};
pub use newton::{
    HessianObjective, HessianOracle, NewtonKind, backward_error_floor, minimize_newton,
    newton_decrement_acceptable, polish_newton,
};
pub use nlcg::{Conjugacy, ConjugacyContext, Restart};
pub use noise::{
    DEFAULT_M, NoiseEstimate, NoiseFlag, ValueGradientNoise, ecnoise, estimate_second,
    estimate_third, forward_step, gradient_dr, line_values, next_spacing, sample_gradient_noise,
    sample_gradient_noise_k, sample_noise, sample_value_gradient_noise,
};
pub use ode12r::{
    Ode12rState, Ode12rVerdict, inf_norm as ode12r_inf_norm, local_error as ode12r_local_error,
    taken_step as ode12r_taken_step,
};
pub use oracle::Oracle;
pub use precon::exp_gate_open;
pub use precon::{
    AngleTerm, BondTerm, DihedralTerm, Exp, ExpParams, Ff, FfParams, FfTerms, GeometryKind, Ghost,
    Lindh, LindhParams, MorseTerm, N_LINDH, N_PAR_ASSEMBLE, N_PAR_EXP_APPLY, N_PAR_EXP_SOLVE,
    N_PAR_LINDH_APPLY, N_PAR_LINDH_SOLVE, Neighbours, OptimizerFamily, PreconChoice, PreconSpec,
    bonded_terms, estimate_mu, max_atom_displacement, nearest_neighbour_distance, needs_rebuild,
    select_precon, sine_probe,
};
pub use pso::minimize_pso;
pub use qn::{minimize_bfgs, minimize_lbfgs, minimize_sd, minimize_sr1, minimize_sr2};
pub use qn_irc::{
    BfgsModel, bfgs_hessian_update, mw_pair, qn_get_s, qn_irc_get_s, qn_irc_restricted,
    qn_irc_restricted_identity, qn_restricted, to_mw,
};
pub use qn_step::QnStep;
pub use quickmin::{
    QuickMinOptions, QuickMinState, quickmin_baseline, quickmin_displacement, quickmin_project,
    quickmin_rescale_velocity,
};
pub use report::{Report, Status, StopReason};
pub use rigid::{RigidMode, RigidRequest, RigidWorkspace, align_rigid};
pub use rlbfgs::{
    HUANG_ARMIJO, HUANG_BACKTRACK, HUANG_CAUTIOUS_SLOPE, HUANG_GRADIENT_RATIO,
    HUANG_LINESEARCH_MAX, HUANG_MEMORY, Rlbfgs, minimize_rlbfgs,
};
pub use scg::{
    DirectionalCurvature, ScgOptions, ScgParams, ScgStepTolerance, minimize_scg,
    minimize_scg_exact, minimize_scg_exact_with_options, minimize_scg_with_options,
};
pub use sella_step::{prfo_restricted, ras_clip, rfo_get_s, rfo_restricted, ts_bfgs_update};
pub use session::Solver;
pub use session::drive::EonFeat;
pub use slepc_kind::{SlepcParams, SlepcPmat, SlepcStKind};
pub use slq::{
    DeflatedPair, ExtremePairsConfig, ExtremePairsReport, HutchPpConfig, PcgConfig, PcgResult,
    ProbeStrategy, QuadratureEstimate, SlqConfig, SlqDeflation, SlqHutchPpOptions, SlqLogDetReport,
    SlqMoments, SlqStoppingRule, TridiagEigh, gauss_radau_tridiag, lanczos_extreme_pairs,
    lanczos_quadrature_fn, lanczos_quadrature_log, pcg_solve, slq_logdet_deflated,
    slq_logdet_deflated_with_cv, slq_logdet_hutchpp, student_t_quantile, sym_eigh, tridiag_eigh,
};
pub use step::DEFAULT_ATOM_MAXMOVE;
pub use stepper::Stepper;
pub use trust::{RestrictedStep, TrustRegion};
pub use tune::{TrainRow, select_train_setting};
