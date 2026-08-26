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

/// Line-search strategies (Brent, Armijo, Goldstein, Wolfe/zoom).
pub mod linesearch;
/// Method selector (NLCG, BFGS family, Adam, steepest descent, PSO).
pub mod method;
/// Nonlinear conjugate-gradient conjugacy and restart.
pub mod nlcg;

mod accept;
mod adam;
mod bb;
mod control;
mod error;
/// C ABI, gated behind the `capi` feature.
#[cfg(feature = "capi")]
#[allow(non_camel_case_types, missing_docs)]
pub mod ffi;
/// FIRE / FIRE 2.0 inertial first-order steps.
pub mod fire;
/// Persistent L-BFGS (Nocedal-Wright 7.4) with strong Wolfe.
pub mod lbfgs;
/// Closed HiGHS solver and crossover tokens.
pub mod highs_kind;
/// L-BFGS quadratic model solved by HiGHS.
#[cfg(feature = "highs")]
pub mod lbfgs_qp;
/// Embedded Riemannian manifolds (manopt_cpp proj / retr / transp).
pub mod manifold;
/// The vector seam: solver algebra behind one interface.
pub mod vecops;
mod minimize;
/// Matrix-free Newton: Hessian actions and Steihaug-Toint CG.
pub mod hvp;
/// Matrix-free lowest Hessian eigenpair (IRC kick / lambda_min).
pub mod lowest_mode;
/// Closed SLEPc ST tokens. Compile without the `slepc` feature.
pub mod slepc_kind;
/// Feature-gated SLEPc EPS (MatShell + typed EPSSet/STSet).
#[cfg(feature = "slepc")]
mod slepc_eps;
/// Shifted Newton and Banerjee RFO on a dense Hessian.
pub mod newton;
mod oracle;
mod pso;
mod qn;
mod qn_step;
mod report;
mod rigid;
/// Moller scaled conjugate gradient (damped-model step, no line search).
pub mod scg;
mod session;
mod step;
mod trust;
/// Sella IRCTrustRegion / Gonzalez--Schlegel mass-weighted sphere.
pub mod irc_trust;
/// Sella QuasiNewton / QuasiNewtonIRC restricted step.
pub mod qn_irc;
/// Sella RFO, P-RFO, and restricted atomic step.
pub mod sella_step;

pub use accept::Accept;
pub use adam::minimize_adam;
pub use control::Control;
pub use error::{Error, Result};
pub use highs_kind::{HighsCCallback, HighsCallbackKind, HighsCrossover, HighsSolverKind};
pub use fire::FireKind;
pub use irc_trust::{sqrt_masses_3n, IrcTrust};
pub use qn_irc::{
    bfgs_hessian_update, mw_pair, qn_get_s, qn_irc_get_s, qn_irc_restricted,
    qn_irc_restricted_identity, qn_restricted, to_mw, BfgsModel,
};
pub use sella_step::{prfo_restricted, ras_clip, rfo_get_s, rfo_restricted, ts_bfgs_update};
pub use lowest_mode::{
    lowest_eigenpair, lowest_mode, lowest_mode_slepc, ApplyHessian, EigenParams, EigensolverKind,
    LowestMode, DENSE_EIGEN_CUTOFF,
};
pub use slepc_kind::{SlepcParams, SlepcPmat, SlepcStKind};
pub use hvp::{
    FdHvp, HessianVector, HvpOracle, IdentityPrecond, NystromPrecond, Preconditioner,
    minimize_newton_cg, steihaug_cg, steihaug_pcg,
};
pub use lbfgs::{GradNorm, Lbfgs};
#[cfg(feature = "highs")]
pub use lbfgs_qp::HighsStep;
pub use linesearch::LineSearch;
pub use manifold::{Manifold, ManifoldKind};
pub use method::Method;
pub use minimize::{minimize, minimize_method, minimize_method_hess};
pub use newton::{HessianObjective, HessianOracle, NewtonKind, minimize_newton};
pub use nlcg::{Conjugacy, ConjugacyContext, Restart};
pub use oracle::Oracle;
pub use pso::minimize_pso;
pub use qn::{minimize_bfgs, minimize_lbfgs, minimize_sd, minimize_sr1, minimize_sr2};
pub use qn_step::QnStep;
pub use report::Report;
pub use scg::{DirectionalCurvature, ScgParams, minimize_scg, minimize_scg_exact};
pub use session::Solver;
