//! Dispatch Nocedal-Wright local methods on an eindir objective.

use eindir_core::{DifferentiableObjective, Objective};
use ndarray::Array1;

use crate::adam::minimize_adam_observed;
use crate::control::Control;
use crate::error::{Error, Result};
use crate::hvp::HessianVector;
use crate::linesearch::LineSearch;
use crate::method::Method;
use crate::newton::{HessianObjective, minimize_newton};
use crate::nlcg::{Conjugacy, ConjugacyContext, Restart};
use crate::observer::{IterationObserver, IterationView, ObserverSignal};
use crate::pso::minimize_pso_observed;
use crate::qn::{
    minimize_bfgs_observed, minimize_lbfgs_observed, minimize_sd_observed, minimize_sr1_observed,
    minimize_sr2_observed,
};
use crate::report::{Report, Status, StopReason};
use crate::session::Solver;
use crate::step::{implicit_atom_cap, l2, next_istep, one_shot_report, stop_if_unmoved, take_step};
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
    minimize_method_observed(obj, init, control, method, linesearch, None)
}

/// Minimize with any [`Method`], observing each accepted iterate.
pub fn minimize_method_observed<O>(
    obj: &O,
    init: impl Into<Array1<f64>>,
    control: &Control,
    method: Method,
    linesearch: LineSearch,
    observer: Option<&mut dyn IterationObserver>,
) -> Result<Report>
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    match method {
        Method::Nlcg { conjugacy, restart } => {
            minimize_nlcg(obj, init, control, conjugacy, restart, linesearch, observer)
        }
        Method::Bfgs => minimize_bfgs_observed(obj, init, control, linesearch, observer),
        Method::Lbfgs { memory } => {
            minimize_lbfgs_observed(obj, init, control, linesearch, memory, observer)
        }
        Method::Sr1 => minimize_sr1_observed(obj, init, control, linesearch, observer),
        Method::Sr2 => minimize_sr2_observed(obj, init, control, linesearch, observer),
        Method::Adam { beta1, beta2, eps } => {
            minimize_adam_observed(obj, init, control, linesearch, beta1, beta2, eps, observer)
        }
        Method::Steepest => minimize_sd_observed(obj, init, control, linesearch, observer),
        Method::Pso {
            n_particles,
            inertia,
            c1,
            c2,
        } => minimize_pso_observed(obj, init, control, n_particles, inertia, c1, c2, observer),
        Method::Newton { kind: _ } | Method::Dogleg => Err(Error::NeedHessian),
        Method::Fire { .. }
        | Method::Bb
        | Method::QuickMin
        | Method::Ode12r
        | Method::Diis { .. }
        | Method::GromacsSteep
        | Method::GpMin
        | Method::NewtonCg
        | Method::NewtonPcg => run_session(obj, init, control, method, observer),
        Method::GromacsCg => {
            run_session_linesearch(obj, init, control, method, linesearch, observer)
        }
    }
}

fn run_session<O>(
    obj: &O,
    init: impl Into<Array1<f64>>,
    control: &Control,
    method: Method,
    observer: Option<&mut dyn IterationObserver>,
) -> Result<Report>
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    drive_session(
        obj,
        init,
        control,
        method,
        None,
        observer,
        |solver, obj, pos| solver.step(obj, pos),
    )
}

fn run_session_linesearch<O>(
    obj: &O,
    init: impl Into<Array1<f64>>,
    control: &Control,
    method: Method,
    linesearch: LineSearch,
    observer: Option<&mut dyn IterationObserver>,
) -> Result<Report>
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    drive_session(
        obj,
        init,
        control,
        method,
        Some(linesearch),
        observer,
        |solver, obj, pos| solver.step(obj, pos),
    )
}

/// Step a fresh session to a stop or the end of `control.maxiter`.
///
/// A zero budget takes no step and judges the evaluated start. The first
/// step whose status is not [`Status::Iterating`], or whose gradient meets
/// [`Control::halt`], ends the run with that status; a spent budget is
/// [`Status::Budget`]. The observer sees every step that moves the
/// iterate, the one that ends the run included, and its stop replaces
/// that status.
fn drive_session<O, S>(
    obj: &O,
    init: impl Into<Array1<f64>>,
    control: &Control,
    method: Method,
    linesearch: Option<LineSearch>,
    mut observer: Option<&mut dyn IterationObserver>,
    mut step: S,
) -> Result<Report>
where
    O: DifferentiableObjective<f64> + ?Sized,
    S: FnMut(&mut Solver, &O, &mut Array1<f64>) -> Result<Report>,
{
    let mut pos = init.into();
    let mut solver = Solver::new(method, control.clone(), pos.len());
    if let Some(ls) = linesearch {
        solver.set_linesearch(ls);
    }
    if control.maxiter == 0 {
        let report = solver.assess(obj, &mut pos)?;
        return control.deliver(report);
    }
    let mut last = None;
    // `control.maxiter` counts the exact number of trial steps taken (`nsteps`),
    // matching GROMACS `do_steep` (`count == nsteps`) rather than `do_cg`'s
    // `step <= number_steps` (`nsteps + 1`); `-1` in MDP/Python maps to
    // unlimited steps (`sys.maxsize`).
    for it in 0..control.maxiter {
        let before = pos.clone();
        let mut report = step(&mut solver, obj, &mut pos)?;
        if let Some(status) = control.halt(report.grad_norm, solver.free_len(obj.bounds())) {
            report.status = status;
        }
        let dx = &report.coords - &before;
        let step_length = crate::vecops::nrm2(dx.view());
        if crate::stop::moved(&before, &report.coords)
            && let Some(obs) = observer.as_deref_mut()
            && obs.observe(IterationView {
                iteration: it,
                x: report.coords.view(),
                value: report.value,
                grad_norm: report.grad_norm,
                step_length,
            }) == ObserverSignal::Stop
        {
            report = report.with_stop_reason(StopReason::ObserverRequested);
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
            .with_require_convergence(false)
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
    let steps = report.steps;
    let evals = report.evals;
    let coords = report.coords;
    counted(obj, control, |obj| {
        let polished =
            crate::newton::polish_inner(obj, coords, control, crate::NewtonKind::Shifted)?;
        Ok(polished)
    })
    .and_then(|polished| control.deliver(carry_polish(steps, evals, polished)))
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
    drive_session(
        obj,
        init,
        control,
        method,
        None,
        None,
        |solver, obj, pos| solver.step_hess(obj, pos),
    )
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
    let steps = report.steps;
    let evals = report.evals;
    let coords = report.coords;
    counted(obj, control, |obj| {
        crate::hvp::polish_hvp_inner(obj, coords, control)
    })
    .and_then(|polished| control.deliver(carry_polish(steps, evals, polished)))
}

/// The search that ran before the polish already counted its steps and
/// evaluations. The polish report replaces that report, so both totals
/// have to be carried across or the caller sees only the polish.
fn carry_polish(prior_steps: usize, prior_evals: usize, polished: Report) -> Report {
    let steps = prior_steps + polished.steps;
    let evals = prior_evals + polished.evals;
    polished.with_steps(steps).with_evals(evals)
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
    drive_session(
        obj,
        init,
        control,
        method,
        None,
        None,
        |solver, obj, pos| solver.step_hvp(obj, pos),
    )
}

fn minimize_nlcg<O>(
    obj: &O,
    init: impl Into<Array1<f64>>,
    control: &Control,
    conjugacy: Conjugacy,
    restart: Restart,
    linesearch: LineSearch,
    observer: Option<&mut dyn IterationObserver>,
) -> Result<Report>
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    counted(obj, control, |obj| {
        nlcg_counted(obj, init, control, conjugacy, restart, linesearch, observer)
    })
}

fn nlcg_counted<O>(
    obj: &Counted<'_, O>,
    init: impl Into<Array1<f64>>,
    control: &Control,
    conjugacy: Conjugacy,
    restart: Restart,
    linesearch: LineSearch,
    mut observer: Option<&mut dyn IterationObserver>,
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
        let unmoved = !moved;
        pos = t.x;
        value = t.f;
        grad = t.g;
        if let Some(obs) = observer.as_deref_mut()
            && obs.observe(IterationView {
                iteration: step,
                x: pos.view(),
                value,
                grad_norm: l2(&grad),
                step_length: lsstep.abs() * crate::vecops::nrm2(dir.view()),
            }) == ObserverSignal::Stop
        {
            return control.deliver(
                one_shot_report(value, pos, step + 1, l2(&grad), Status::ObserverStop)
                    .with_stop_reason(StopReason::ObserverRequested),
            );
        }
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
        let next_dir = Array1::from_iter(grad.iter().zip(d_old.iter()).map(|(g, d)| -g + beta * d));
        let next = next_istep(lsstep, control);
        let fixed = unmoved && next_dir == dir && next == istep;
        if let Some(stop) = stop_if_unmoved(control, fixed, value, pos.clone(), step + 1, gnorm) {
            return stop;
        }
        dir = next_dir;
        g_old.assign(&grad);
        d_old.assign(&dir);
        istep = next;
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
