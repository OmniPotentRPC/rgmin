//! Dispatch Nocedal-Wright local methods on an eindir objective.

use eindir_core::{DifferentiableObjective, Objective};
use ndarray::Array1;

use crate::adam::minimize_adam;
use crate::control::Control;
use crate::error::{Error, Result};
use crate::hvp::HessianVector;
use crate::linesearch::LineSearch;
use crate::method::Method;
use crate::newton::{HessianObjective, minimize_newton};
use crate::nlcg::{Conjugacy, ConjugacyContext, Restart};
use crate::pso::minimize_pso;
use crate::qn::{minimize_bfgs, minimize_lbfgs, minimize_sd, minimize_sr1, minimize_sr2};
use crate::report::{Report, Status};
use crate::session::Solver;
use crate::step::{implicit_atom_cap, l2, next_istep, one_shot_report, take_step};
use crate::stop::{Counted, StopRules, counted, stop_after};

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
        Method::Fire { .. }
        | Method::Bb
        | Method::QuickMin
        | Method::Ode12r
        | Method::Diis { .. }
        | Method::NewtonCg
        | Method::NewtonPcg => run_session(obj, init, control, method),
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
    drive_session(obj, init, control, method, |solver, obj, pos| {
        solver.step(obj, pos)
    })
}

/// Step a fresh session to a stop or the end of `control.maxiter`.
///
/// A zero budget takes no step and judges the evaluated start. The first
/// step whose status is not [`Status::Iterating`], or whose gradient meets
/// [`Control::halt`], ends the run with that status; a spent budget is
/// [`Status::Budget`].
fn drive_session<O, S>(
    obj: &O,
    init: impl Into<Array1<f64>>,
    control: &Control,
    method: Method,
    mut step: S,
) -> Result<Report>
where
    O: DifferentiableObjective<f64> + ?Sized,
    S: FnMut(&mut Solver, &O, &mut Array1<f64>) -> Result<Report>,
{
    let mut pos = init.into();
    let mut solver = Solver::new(method, control.clone(), pos.len());
    if control.maxiter == 0 {
        let report = solver.assess(obj, &mut pos)?;
        return control.deliver(report);
    }
    let mut last = None;
    for _ in 0..control.maxiter {
        let mut report = step(&mut solver, obj, &mut pos)?;
        if let Some(status) = control.halt(report.grad_norm, solver.free_len(obj.bounds())) {
            report.status = status;
        }
        let stopped = report.status != Status::Iterating;
        last = Some(report);
        if stopped {
            break;
        }
    }
    let Some(mut last) = last else {
        unreachable!("a positive budget takes a step or returns its error");
    };
    if last.status == Status::Iterating {
        last.status = Status::Budget;
    }
    control.deliver(last)
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
    if matches!(method, Method::Newton { .. }) {
        return match method {
            Method::Newton { kind } => minimize_newton(obj, init, control, kind),
            _ => unreachable!("matched Newton above"),
        };
    }
    let quiet = without_polish(control);
    let report = match method {
        Method::Dogleg => run_session_hess(obj, init, &quiet, method)?,
        other => minimize_method(obj, init, &quiet, other, linesearch)?,
    };
    polish_after_hess(obj, control, report)
}

fn without_polish(control: &Control) -> Control {
    if control.backward_polish {
        control
            .clone()
            .with_backward_polish(false)
            .with_error_on_unmet(false)
    } else {
        control.clone()
    }
}

fn polish_after_hess<O>(obj: &O, control: &Control, report: Report) -> Result<Report>
where
    O: HessianObjective + ?Sized,
{
    if !control.backward_polish {
        return control.deliver(report);
    }
    let prior = report.steps;
    let polished =
        crate::newton::polish_inner(obj, report.coords, control, crate::NewtonKind::Shifted)?;
    let steps = prior + polished.steps;
    control.deliver(polished.with_steps(steps))
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
    drive_session(obj, init, control, method, |solver, obj, pos| {
        solver.step_hess(obj, pos)
    })
}

/// Minimize with any [`Method`]. Newton-CG and Newton-PCG use analytic
/// Hessian-vector products and never form a dense Hessian. Every other
/// method is [`minimize_method`].
pub fn minimize_method_hvp<O>(
    obj: &O,
    init: impl Into<Array1<f64>>,
    control: &Control,
    method: Method,
    linesearch: LineSearch,
) -> Result<Report>
where
    O: HessianVector + ?Sized,
{
    let quiet = without_polish(control);
    let report = match method {
        Method::NewtonCg | Method::NewtonPcg => run_session_hvp(obj, init, &quiet, method)?,
        other => minimize_method(obj, init, &quiet, other, linesearch)?,
    };
    if !control.backward_polish {
        return control.deliver(report);
    }
    let prior = report.steps;
    let polished = crate::hvp::polish_hvp_inner(obj, report.coords, control)?;
    let steps = prior + polished.steps;
    control.deliver(polished.with_steps(steps))
}

fn run_session_hvp<O>(
    obj: &O,
    init: impl Into<Array1<f64>>,
    control: &Control,
    method: Method,
) -> Result<Report>
where
    O: HessianVector + ?Sized,
{
    drive_session(obj, init, control, method, |solver, obj, pos| {
        solver.step_hvp(obj, pos)
    })
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
    counted(obj, control, |obj| {
        nlcg_counted(obj, init, control, conjugacy, restart, linesearch)
    })
}

fn nlcg_counted<O>(
    obj: &Counted<'_, O>,
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
    crate::step::refuse_non_finite(value, &grad)?;
    let mut dir = grad.mapv(|g| -g);
    let mut g_old = grad.clone();
    let mut d_old = dir.clone();
    let mut istep = control.istep;
    let mut stops = StopRules::from_start(value);

    for step in 0..control.maxiter {
        let gnorm = l2(&grad);
        if let Some(status) = control.halt(gnorm, grad.len()) {
            return control.deliver(one_shot_report(value, pos, step, gnorm, status));
        }
        let t = take_step(
            obj,
            &pos,
            value,
            &grad,
            dir.view(),
            istep,
            linesearch,
            control,
            implicit_atom_cap(control),
        );
        if t.energy_noise {
            return control.deliver(one_shot_report(
                value,
                pos,
                step + 1,
                gnorm,
                Status::NoiseFloor,
            ));
        }
        let lsstep = t.alpha;
        let moved = t.moved;
        pos = t.x;
        value = t.f;
        grad = t.g;
        if let Some(status) = stop_after(&mut stops, control, obj, value, moved, &grad) {
            return control.deliver(one_shot_report(value, pos, step + 1, l2(&grad), status));
        }
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
    let gnorm = l2(&grad);
    control.deliver(one_shot_report(
        value,
        pos,
        control.maxiter,
        gnorm,
        control.finish_status(gnorm, grad.len(), true),
    ))
}
