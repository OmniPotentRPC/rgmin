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
//! The production unconstrained local method is [`Lbfgs`] with
//! [`LineSearch::Wolfe`]: limited-memory BFGS (Nocedal-Wright 7.4) and
//! the strong Wolfe conditions (algorithms 3.5 and 3.6). Hopping chains
//! keep the pair history on that type. [`minimize_lbfgs`] is the
//! cold-start dispatch used by [`minimize_method`].

#![warn(missing_docs)]
// Numerical kernels take their state as explicit arguments, and the
// small dense algebra indexes by position on purpose.
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

/// Line-search strategies (Brent, Armijo, Goldstein, Wolfe/zoom).
pub mod linesearch;
/// Method selector (NLCG, BFGS family, Adam, steepest descent, PSO).
pub mod method;
/// Nonlinear conjugate-gradient conjugacy and restart.
pub mod nlcg;

mod accept;
mod adam;
mod bb;
#[cfg(feature = "highs")]
mod box_objective;
/// CON checkpoint of an accepted point. Compiled only with `readcon`.
#[cfg(feature = "readcon")]
pub mod con_checkpoint;
mod control;
mod diis;
mod error;
/// C ABI, gated behind the `capi` feature.
#[cfg(feature = "capi")]
#[allow(non_camel_case_types, missing_docs)]
pub mod ffi;
/// FIRE / FIRE 2.0 inertial first-order steps.
pub mod fire;
/// Matrix-free Newton: Hessian actions and Steihaug-Toint CG.
pub mod hvp;
/// Sella IRCTrustRegion / Gonzalez--Schlegel mass-weighted sphere.
pub mod irc_trust;
/// Persistent L-BFGS (Nocedal-Wright 7.4) with strong Wolfe.
pub mod lbfgs;
/// L-BFGS quadratic model solved by HiGHS.
#[cfg(feature = "highs")]
pub mod lbfgs_qp;
/// Matrix-free lowest Hessian eigenpair.
pub mod lowest_mode;
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
/// Geometry preconditioners: Exp, Lindh, and the bonded FF metric.
pub mod precon;
mod pso;
mod qn;
/// Sella QuasiNewton / QuasiNewtonIRC restricted step.
pub mod qn_irc;
mod qn_step;
/// Quick-min projection and Euler step.
pub mod quickmin;
mod report;
mod restricted_trust;
/// Eckart projector and Horn / Coutsias alignment.
pub mod rigid;
/// Tangent-space truncated conjugate gradients and trust-radius updates.
pub mod rtr;
/// Moller scaled conjugate gradient (damped-model step, no line search).
pub mod scg;
/// Sella RFO, P-RFO, and restricted atomic step.
pub mod sella_step;
mod session;
mod sphere_lbfgs;
mod step;
/// Dense dogleg and Sella trust-region steps.
pub mod trust;
/// The vector seam: solver algebra behind one interface.
pub mod vecops;

pub use accept::Accept;
pub use adam::minimize_adam;
#[cfg(feature = "readcon")]
pub use con_checkpoint::{read_checkpoint, write_checkpoint};
pub use control::{Control, NoiseFloor};
pub use dlaf_kind::DlaFutureParams;
pub use eigenexa_kind::{EigenExaAlgo, EigenExaParams};
pub use elpa_kind::{ElpaParams, ElpaSpectrum, ElpaStage, elpa_config, elpa_spectrum};
pub use error::{Error, Result};
pub use fire::{FireKind, FireVariant};
pub use highs_kind::{
    HighsCCallback, HighsCallback, HighsCallbackKind, HighsCrossover, HighsOptions, HighsSolverKind,
};
pub use hvp::{
    FdHvp, HessianVector, HvpOracle, IdentityPrecond, NystromPrecond, Preconditioner,
    minimize_newton_cg, steihaug_cg, steihaug_pcg,
};
pub use irc_trust::{IrcTrust, sqrt_masses_3n};
pub use lbfgs::{GradNorm, Lbfgs};
#[cfg(feature = "highs")]
pub use lbfgs_qp::HighsStep;
pub use linesearch::{LineSearch, LineSearchOptions};
pub use lowest_mode::{
    ApplyHessian, ApplyPreconditioner, Block3Jacobi, DENSE_EIGEN_CUTOFF, DiagonalJacobi,
    EigenParams, EigensolverKind, LowestMode, PreconditionerKind, lowest_eigenpair, lowest_mode,
    lowest_mode_chase, lowest_mode_dense, lowest_mode_dlaf, lowest_mode_eigenexa,
    lowest_mode_libkrylov, lowest_mode_precond, lowest_mode_precond_with, lowest_mode_primme,
    lowest_mode_slepc, lowest_mode_with,
};
pub use manifold::{
    CenteredMatrix, ComplexCircle, Constant, EuclideanComplex, Grassmann, Hyperbolic, Manifold,
    ManifoldKind, Multinomial, MultinomialDoublyStochastic, MultinomialSymmetric, Oblique,
    PoincareBall, Positive, SkewSymmetric, Spd, SphereComplex, StiefelNp, Symmetric, Unitary,
};
pub use method::Method;
pub use minimize::{minimize, minimize_method, minimize_method_hess, minimize_method_hvp};
pub use newton::{HessianObjective, HessianOracle, NewtonKind, minimize_newton};
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
pub use report::{Report, Status};
pub use rigid::{RigidMode, RigidRequest, RigidWorkspace, align_rigid};
pub use scg::{
    DirectionalCurvature, ScgOptions, ScgParams, ScgStepTolerance, minimize_scg,
    minimize_scg_exact, minimize_scg_exact_with_options, minimize_scg_with_options,
};
pub use sella_step::{prfo_restricted, ras_clip, rfo_get_s, rfo_restricted, ts_bfgs_update};
pub use session::Solver;
pub use slepc_kind::{SlepcParams, SlepcPmat, SlepcStKind};
pub use step::DEFAULT_ATOM_MAXMOVE;

pub use trust::{RestrictedStep, TrustRegion};
