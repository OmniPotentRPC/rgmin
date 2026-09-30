//! Dispatch Nocedal-Wright local methods on an eindir objective.

use eindir_core::{DifferentiableObjective, Objective};
use ndarray::Array1;

use crate::adam::minimize_adam;
use crate::control::Control;
use crate::error::{Error, Result};
use crate::linesearch::LineSearch;
use crate::method::Method;
use crate::newton::{HessianObjective, minimize_newton};
use crate::nlcg::{Conjugacy, ConjugacyContext, Restart};
use crate::pso::minimize_pso;
use crate::qn::{minimize_bfgs, minimize_lbfgs, minimize_sd, minimize_sr1, minimize_sr2};
use crate::report::Report;
use crate::session::Solver;
use crate::step::{l2, next_istep, take_step};

/// Minimize `obj` from `init` with the chosen conjugacy, restart, and line search.
pub fn minimize<O>(
    obj: &O,
    init: impl Into<Array1<f64>>,
    control: &Control,
    conjugacy: Conjugacy,
    restart: Restart,
    linesearch: LineSearch,
) -> Result<Report>
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    minimize_method(
        obj,
        init,
        control,
        Method::Nlcg { conjugacy, restart },
        linesearch,
    )
}

/// Minimize with any [`Method`].
pub fn minimize_method<O>(
    obj: &O,
    init: impl Into<Array1<f64>>,
    control: &Control,
    method: Method,
    linesearch: LineSearch,
) -> Result<Report>
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    match method {
        Method::Nlcg { conjugacy, restart } => {
            minimize_nlcg(obj, init, control, conjugacy, restart, linesearch)
        }
        Method::Bfgs => minimize_bfgs(obj, init, control, linesearch),
        Method::Lbfgs { memory } => minimize_lbfgs(obj, init, control, linesearch, memory),
        Method::Sr1 => minimize_sr1(obj, init, control, linesearch),
        Method::Sr2 => minimize_sr2(obj, init, control, linesearch),
        Method::Adam { beta1, beta2, eps } => {
            minimize_adam(obj, init, control, linesearch, beta1, beta2, eps)
        }
        Method::Steepest => minimize_sd(obj, init, control, linesearch),
        Method::Pso {
            n_particles,
            inertia,
            c1,
            c2,
        } => minimize_pso(obj, init, control, n_particles, inertia, c1, c2),
        Method::Newton { kind: _ } | Method::Dogleg => Err(Error::NeedHessian),
        Method::Fire { .. } | Method::Bb => run_session(obj, init, control, method),
    }
}

fn run_session<O>(
    obj: &O,
    init: impl Into<Array1<f64>>,
    control: &Control,
    method: Method,
) -> Result<Report>
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let mut pos = init.into();
    let mut solver = Solver::new(method, control.clone(), pos.len());
    let mut last = Report {
        value: 0.0,
        coords: pos.clone(),
        steps: 0,
        grad_norm: f64::INFINITY,
    };
    for _ in 0..control.maxiter {
        last = solver.step(obj, &mut pos)?;
        if last.grad_norm < control.gtol {
            break;
        }
    }
    Ok(last)
}

/// Minimize with any [`Method`], including Newton / RFO.
pub fn minimize_method_hess<O>(
    obj: &O,
    init: impl Into<Array1<f64>>,
    control: &Control,
    method: Method,
    linesearch: LineSearch,
) -> Result<Report>
where
    O: HessianObjective + ?Sized,
{
    match method {
        Method::Newton { kind } => minimize_newton(obj, init, control, kind),
        Method::Dogleg => run_session_hess(obj, init, control, method),
        other => minimize_method(obj, init, control, other, linesearch),
    }
}

fn run_session_hess<O>(
    obj: &O,
    init: impl Into<Array1<f64>>,
    control: &Control,
    method: Method,
) -> Result<Report>
where
    O: HessianObjective + ?Sized,
{
    let mut pos = init.into();
    let mut solver = Solver::new(method, control.clone(), pos.len());
    let mut last = Report {
        value: 0.0,
        coords: pos.clone(),
        steps: 0,
        grad_norm: f64::INFINITY,
    };
    for _ in 0..control.maxiter {
        last = solver.step_hess(obj, &mut pos)?;
        if last.grad_norm < control.gtol {
            break;
        }
    }
    Ok(last)
}

fn minimize_nlcg<O>(
    obj: &O,
    init: impl Into<Array1<f64>>,
    control: &Control,
    conjugacy: Conjugacy,
    restart: Restart,
    linesearch: LineSearch,
) -> Result<Report>
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let mut pos = init.into();
    if pos.len() != Objective::dim(obj) {
        return Err(Error::Dim {
            got: pos.len(),
            dim: Objective::dim(obj),
        });
    }
    pos = obj.bounds().clip(pos.view());
    let (mut value, mut grad) = obj.value_and_gradient(pos.view());
    let mut dir = grad.mapv(|g| -g);
    let mut g_old = grad.clone();
    let mut d_old = dir.clone();
    let mut istep = control.istep;

    for step in 0..control.maxiter {
        let gnorm = l2(&grad);
        if gnorm < control.gtol {
            return Ok(Report {
                value,
                coords: pos,
                steps: step,
                grad_norm: gnorm,
            });
        }
        let (npos, _, lsstep, _) = take_step(
            obj,
            &pos,
            value,
            dir.view(),
            istep,
            linesearch,
            control,
            None,
        );
        pos = npos;
        let ev = obj.value_and_gradient(pos.view());
        value = ev.0;
        grad = ev.1;
        let ctx = ConjugacyContext {
            current_gradient: grad.view(),
            previous_gradient: g_old.view(),
            previous_direction: d_old.view(),
        };
        let mut beta = conjugacy.beta(&ctx);
        if restart.should_restart(&ctx) {
            beta = 0.0;
        }
        dir = Array1::from_iter(grad.iter().zip(d_old.iter()).map(|(g, d)| -g + beta * d));
        g_old.assign(&grad);
        d_old.assign(&dir);
        istep = next_istep(lsstep, control);
    }
    Ok(Report {
        value,
        coords: pos,
        steps: control.maxiter,
        grad_norm: l2(&grad),
    })
}
