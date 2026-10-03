//! Møller scaled conjugate gradient: a damped-model step length for CG.
//!
//! Port of the gpr_optim SCG hyperparameter optimizer
//! (`gpr/ml/SCG.inl`), originally written by Maxim Masterov (SURF,
//! 2020) for Gaussian-process regression fits and hardened there
//! against non-finite objectives (interior-point barrier walls,
//! non-positive-definite covariances). The conjugacy coefficient and
//! restart policy come from [`crate::nlcg`]; this module contributes
//! only what makes SCG itself: the one-sided curvature probe and the
//! Levenberg-Marquardt damped model step that replaces a line search.

use eindir_core::{DifferentiableObjective, Objective};
use ndarray::Array1;

use crate::control::Control;
use crate::error::{Error, Result};
use crate::nlcg::{Conjugacy, ConjugacyContext, Restart};
use crate::report::Report;
use crate::step::l2;
use ndarray::ArrayView1;

/// Exact directional curvature `d . grad^2 f(x) . d` for SCG's step
/// pricing. gpr_optim supplies this from its analytic marginal-likelihood
/// Hessian-vector product (`ScgCurvatureMode::EXACT`); whenever it
/// returns `None` the finite-difference probe prices the step instead,
/// so partial implementations degrade gracefully.
pub trait DirectionalCurvature: DifferentiableObjective<f64> {
    /// `d . grad^2 f(x) . d`, or `None` to fall back to the probe.
    fn directional_curvature(&self, x: ArrayView1<f64>, d: ArrayView1<f64>) -> Option<f64>;
}

/// Damping schedule and convergence tolerances for [`minimize_scg`].
///
/// Defaults follow Netlab `scg.m` and the gpr_optim production values.
#[derive(Clone, Debug)]
pub struct ScgParams {
    /// Base finite-difference curvature probe step (Møller `sigma_0`).
    pub sigma0: f64,
    /// Initial Levenberg-Marquardt damping `lambda`.
    pub lambda: f64,
    /// Stop with [`Error::ScgStalled`] when `lambda` reaches this
    /// limit: the quadratic model never matches the objective, which
    /// in practice means the supplied gradient disagrees with the
    /// value.
    pub lambda_limit: f64,
    /// Solution tolerance: converged when `||alpha d||_inf` falls
    /// below this while the objective change also converges.
    pub tol_sol: f64,
    /// Relative objective-change tolerance, scaled by one plus the range
    /// of accepted values, independently of an additive objective constant.
    pub tol_func: f64,
}

impl Default for ScgParams {
    fn default() -> Self {
        Self {
            sigma0: 1e-4,
            lambda: 1.0,
            lambda_limit: 1e60,
            tol_sol: 1e-6,
            tol_func: 1e-8,
        }
    }
}

const LAMBDA_MIN: f64 = 1.0e-15;
const LAMBDA_MAX: f64 = 1.0e100;
/// A trial that stays non-finite through this many damping increases
/// is hopeless: the objective is non-finite along every step the
/// model can propose.
const MAX_NONFINITE_RETRIES: usize = 100;
/// Doubling the probe step from `sigma0/sqrt(kappa)` this many times
/// overshoots `f64::MAX`, so the cap is unreachable while any finite
/// probe exists.
const MAX_PROBE_RETRIES: usize = 2000;

/// Møller's scaled conjugate gradient (Møller, *Neural Networks* 6
/// (1993) 525, <https://doi.org/10.1016/S0893-6080(05)80056-5>).
///
/// A conjugate-gradient method whose step length comes from a
/// one-sided finite-difference curvature probe along the search
/// direction, stabilised by Levenberg-Marquardt damping that grows
/// when the quadratic model disagrees with the measured objective
/// change (Møller's comparison ratio) and shrinks when it agrees. No
/// line search: one extra gradient per iteration prices the step.
///
/// `conjugacy` supplies β (Netlab's choice is
/// [`Conjugacy::PolakRibiere`]); `restart` adds a caller policy on
/// top of Møller's intrinsic reset to steepest descent after `n`
/// consecutive successful steps (his step 9).
///
/// Non-finite objective values are first-class, as the gpr_optim
/// origin required: the curvature probe doubles its step until the
/// objective is finite, and a non-finite trial raises the damping and
/// retries, both within fixed budgets. An objective wrapped with
/// interior-point barriers therefore composes with no optimizer hook.
/// A zero `control.maxiter` leaves termination to convergence and the
/// numerical failure checks.
pub fn minimize_scg<O>(
    obj: &O,
    init: impl Into<Array1<f64>>,
    control: &Control,
    params: &ScgParams,
    conjugacy: Conjugacy,
    restart: Restart,
) -> Result<Report>
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    run_scg(
        obj,
        init.into(),
        control,
        params,
        conjugacy,
        restart,
        |_, _| None,
    )
}

/// [`minimize_scg`] with exact curvature: the probe (and its extra
/// gradient per iteration) is replaced by the objective's
/// Hessian-vector product wherever one is available, the same
/// Newton-Krylov-style pricing PETSc/TAO's `nls` obtains from
/// user HVP callbacks.
pub fn minimize_scg_exact<O>(
    obj: &O,
    init: impl Into<Array1<f64>>,
    control: &Control,
    params: &ScgParams,
    conjugacy: Conjugacy,
    restart: Restart,
) -> Result<Report>
where
    O: DirectionalCurvature + ?Sized,
{
    run_scg(
        obj,
        init.into(),
        control,
        params,
        conjugacy,
        restart,
        |x, d| obj.directional_curvature(x, d),
    )
}

fn run_scg<O>(
    obj: &O,
    init: Array1<f64>,
    control: &Control,
    params: &ScgParams,
    conjugacy: Conjugacy,
    restart: Restart,
    curvature: impl Fn(ArrayView1<f64>, ArrayView1<f64>) -> Option<f64>,
) -> Result<Report>
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let mut w = init;
    if w.len() != Objective::dim(obj) {
        return Err(Error::Dim {
            got: w.len(),
            dim: Objective::dim(obj),
        });
    }
    w = obj.bounds().clip(w.view());
    let nparams = w.len();
    let mut lambda = params.lambda;

    let (mut f_old, mut grad) = obj.value_and_gradient(w.view());
    if !f_old.is_finite() {
        return Err(Error::ScgStalled {
            what: "start point has a non-finite objective",
        });
    }
    let mut objective_min = f_old;
    let mut objective_max = f_old;
    let mut grad_old = grad.clone();
    let mut dir = -grad.clone();

    let mut success = true;
    let mut nsuccess: usize = 0;
    let mut mu = 0.0;
    let mut kappa = 0.0;
    let mut gamma = 0.0;

    let mut step = 0;
    while control.maxiter == 0 || step < control.maxiter {
        let gnorm = l2(&grad);
        if gnorm < control.gtol {
            return Ok(Report {
                value: f_old,
                coords: w,
                steps: step,
                grad_norm: gnorm,
            });
        }

        if success {
            mu = crate::vecops::dot(dir.view(), grad.view());
            if mu >= 0.0 {
                // Descent safeguard (Møller step 2): a non-descent
                // direction resets to steepest descent.
                dir = -grad.clone();
                mu = crate::vecops::dot(dir.view(), grad.view());
            }
            kappa = crate::vecops::dot(dir.view(), dir.view());
            if kappa < f64::EPSILON {
                return Ok(Report {
                    value: f_old,
                    coords: w,
                    steps: step,
                    grad_norm: gnorm,
                });
            }

            // Exact Hessian-vector curvature when the objective
            // offers one; the one-sided probe otherwise.
            if let Some(exact) = curvature(w.view(), dir.view()).filter(|c| c.is_finite()) {
                gamma = exact;
            } else {
                let mut sigma = params.sigma0 / kappa.sqrt();
                let mut probed = obj.value_and_gradient((&w + &(sigma * &dir)).view());
                let mut probe_retries = 0;
                while !(probed.0.is_finite() && probed.1.iter().all(|g| g.is_finite()))
                    && sigma.is_finite()
                    && probe_retries < MAX_PROBE_RETRIES
                {
                    sigma *= 2.0;
                    probed = obj.value_and_gradient((&w + &(sigma * &dir)).view());
                    probe_retries += 1;
                }
                if !(probed.0.is_finite() && probed.1.iter().all(|g| g.is_finite())) {
                    return Err(Error::ScgStalled {
                        what: "no finite curvature probe along the search direction",
                    });
                }
                let dg = &probed.1 - &grad;
                gamma = crate::vecops::dot(dg.view(), dir.view()) / sigma;
            }
        }

        let mut delta = gamma + lambda * kappa;
        if delta <= 0.0 {
            delta = lambda * kappa;
            lambda -= gamma / kappa;
        }
        let mut alpha = -mu / delta;

        let mut trial = &w + &(alpha * &dir);
        let (mut f_new, mut grad_new) = obj.value_and_gradient(trial.view());
        let mut fails = 0;
        while !f_new.is_finite() {
            lambda = (4.0 * lambda).min(LAMBDA_MAX);
            delta = gamma + lambda * kappa;
            if delta <= 0.0 {
                delta = lambda * kappa;
                lambda -= gamma / kappa;
            }
            alpha = -mu / delta;
            trial = &w + &(alpha * &dir);
            let ev = obj.value_and_gradient(trial.view());
            f_new = ev.0;
            grad_new = ev.1;
            fails += 1;
            if fails > MAX_NONFINITE_RETRIES {
                return Err(Error::ScgStalled {
                    what: "objective stayed non-finite through the damping budget",
                });
            }
        }

        // Møller's comparison ratio between the measured and the
        // model-predicted objective change.
        let ratio = 2.0 * (f_new - f_old) / (alpha * mu);

        success = ratio >= 0.0;
        if success {
            nsuccess += 1;
            objective_min = objective_min.min(f_new);
            objective_max = objective_max.max(f_new);
            let step_inf = alpha.abs() * crate::vecops::nrminf(dir.view());
            let solution_converged = step_inf < params.tol_sol;
            let objective_converged =
                (f_new - f_old).abs() < params.tol_func * (1.0 + (objective_max - objective_min));
            w = trial;
            if solution_converged && objective_converged {
                return Ok(Report {
                    value: f_new,
                    coords: w,
                    steps: step + 1,
                    grad_norm: l2(&grad_new),
                });
            }
            f_old = f_new;
            grad_old = std::mem::replace(&mut grad, grad_new);
            if crate::vecops::dot(grad.view(), grad.view()) < f64::EPSILON {
                return Ok(Report {
                    value: f_old,
                    coords: w,
                    steps: step + 1,
                    grad_norm: l2(&grad),
                });
            }
        }

        if ratio < 0.25 {
            lambda = (4.0 * lambda).min(LAMBDA_MAX);
        }
        if ratio > 0.75 {
            lambda = (0.5 * lambda).max(LAMBDA_MIN);
        }
        if lambda >= params.lambda_limit {
            return Err(Error::ScgStalled {
                what: "damping reached its limit; check the analytic gradient",
            });
        }

        if success {
            let ctx = ConjugacyContext {
                current_gradient: grad.view(),
                previous_gradient: grad_old.view(),
                previous_direction: dir.view(),
            };
            // Møller's intrinsic period reset after n successes, plus
            // whatever policy the caller chose.
            if nsuccess == nparams || restart.should_restart(&ctx) {
                dir = -grad.clone();
                nsuccess = 0;
            } else {
                let beta = conjugacy.beta(&ctx);
                dir = beta * &dir - &grad;
            }
        }
        step += 1;
    }

    Ok(Report {
        value: f_old,
        grad_norm: l2(&grad),
        coords: w,
        steps: control.maxiter,
    })
}
