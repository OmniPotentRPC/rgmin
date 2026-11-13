//! Dispatch Nocedal-Wright local methods on an eindir objective.

#[cfg(test)]
use std::cell::Cell;

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
use crate::report::Report;
use crate::session::Solver;
use crate::step::{
    cap_alpha, energy_noise_report, implicit_atom_cap, l2, nlcg_half_opening, take_step,
};

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

/// Line search for a one-shot minimizer that was not given one.
///
/// Nonlinear CG uses strong Wolfe with `c2 = 0.1`. A caller that passes
/// its own [`LineSearch`] to [`minimize`] or [`minimize_method`] is left
/// alone, including Brent. Every other method keeps [`LineSearch::default`].
#[cfg_attr(not(feature = "capi"), allow(dead_code))]
pub(crate) fn oneshot_linesearch(method: &Method) -> LineSearch {
    if matches!(method, Method::Nlcg { .. }) {
        crate::linesearch::CG_LINESEARCH
    } else {
        LineSearch::default()
    }
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
    let mut pos = init.into();
    let mut solver = Solver::new(method, control.clone(), pos.len());
    let mut last = Report {
        value: 0.0,
        coords: pos.clone(),
        steps: 0,
        grad_norm: f64::INFINITY,
        status: crate::report::Status::Iterating,
    };
    for _ in 0..control.maxiter {
        last = solver.step(obj, &mut pos)?;
        if let Some(status) = control.halt(last.grad_norm, solver.free_len(obj.bounds())) {
            last.status = status;
            break;
        }
    }
    if last.status == crate::report::Status::Iterating {
        last.status = crate::report::Status::Budget;
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
        status: crate::report::Status::Iterating,
    };
    for _ in 0..control.maxiter {
        last = solver.step_hess(obj, &mut pos)?;
        if let Some(status) = control.halt(last.grad_norm, solver.free_len(obj.bounds())) {
            last.status = status;
            break;
        }
    }
    if last.status == crate::report::Status::Iterating {
        last.status = crate::report::Status::Budget;
    }
    Ok(last)
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
    match method {
        Method::NewtonCg | Method::NewtonPcg => run_session_hvp(obj, init, control, method),
        other => minimize_method(obj, init, control, other, linesearch),
    }
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
    let mut pos = init.into();
    let mut solver = Solver::new(method, control.clone(), pos.len());
    let mut last = Report {
        value: 0.0,
        coords: pos.clone(),
        steps: 0,
        grad_norm: f64::INFINITY,
        status: crate::report::Status::Iterating,
    };
    for _ in 0..control.maxiter {
        last = solver.step_hvp(obj, &mut pos)?;
        if let Some(status) = control.halt(last.grad_norm, solver.free_len(obj.bounds())) {
            last.status = status;
            break;
        }
    }
    if last.status == crate::report::Status::Iterating {
        last.status = crate::report::Status::Budget;
    }
    Ok(last)
}

#[cfg(test)]
thread_local! {
    static UPHILL_FIRINGS: Cell<u64> = const { Cell::new(0) };
}

#[cfg(test)]
fn note_uphill() {
    UPHILL_FIRINGS.with(|c| c.set(c.get() + 1));
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
    let mut prev_alpha = 0.0;

    for step in 0..control.maxiter {
        let gnorm = l2(&grad);
        if let Some(status) = control.halt(gnorm, grad.len()) {
            return Ok(Report {
                value,
                coords: pos,
                steps: step,
                grad_norm: gnorm,
                status,
            });
        }
        // A beta that leaves g·d >= 0 is not a descent direction. The
        // line search would flip it and then build the next beta from
        // that flipped vector. Replace it before the search.
        let gd_used = crate::vecops::dot(grad.view(), dir.view());
        if gd_used.partial_cmp(&0.0) != Some(std::cmp::Ordering::Less) {
            #[cfg(test)]
            note_uphill();
            dir = grad.mapv(|g| -g);
            d_old.assign(&dir);
        }
        let cap = cap_alpha(dir.view(), control, implicit_atom_cap(control));
        let alpha0 = nlcg_half_opening(prev_alpha, cap, control.istep);
        let t = take_step(
            obj,
            &pos,
            value,
            &grad,
            dir.view(),
            alpha0,
            linesearch,
            control,
            implicit_atom_cap(control),
        );
        if t.energy_noise {
            return Ok(energy_noise_report(value, pos, step + 1, gnorm));
        }
        pos = t.x;
        value = t.f;
        grad = t.g;
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
        let formed = crate::vecops::dot(grad.view(), dir.view());
        if formed.partial_cmp(&0.0) != Some(std::cmp::Ordering::Less) {
            #[cfg(test)]
            note_uphill();
            dir = grad.mapv(|g| -g);
        }
        g_old.assign(&grad);
        d_old.assign(&dir);
        prev_alpha = t.alpha;
    }
    Ok(Report {
        value,
        coords: pos,
        steps: control.maxiter,
        grad_norm: l2(&grad),
        status: control.finish_status(l2(&grad), grad.len(), true),
    })
}

#[cfg(test)]
mod uphill_count {
    use super::*;
    use crate::linesearch::LineSearch;
    use crate::nlcg::{Conjugacy, Restart};
    use crate::oracle::Oracle;
    use ndarray::{Array1, ArrayView1};

    fn rose(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let n = x.len();
        let mut f = -0.0;
        let mut g = Array1::zeros(n);
        for i in 0..n - 1 {
            let a = 1.0 - x[i];
            let b = x[i + 1] - x[i] * x[i];
            f += a * a + 100.0 * b * b;
            g[i] += -2.0 * a - 400.0 * x[i] * b;
            g[i + 1] += 200.0 * b;
        }
        (f, g)
    }

    fn scaled(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let n = x.len();
        let mut f = -0.0;
        let mut g = Array1::zeros(n);
        let denom = (n - 1) as f64;
        for i in 0..n {
            let lam = 10f64.powf(8.0 * (i as f64) / denom);
            f += 0.5 * lam * x[i] * x[i];
            g[i] = lam * x[i];
        }
        (f, g)
    }

    fn morse(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        const D0: f64 = 0.7102;
        const ALPHA: f64 = 1.6047;
        const R0: f64 = 2.897;
        let n = x.len() / 3;
        let rc = 7.0;
        let xc = (-ALPHA * (rc - R0)).exp();
        let shift = D0 * (xc * xc - 2.0 * xc);
        let mut energy = -0.0;
        let mut grad = Array1::zeros(x.len());
        let cut2 = rc * rc;
        for i in 0..n {
            for j in (i + 1)..n {
                let mut diff = [0.0; 3];
                let mut r2 = -0.0;
                for a in 0..3 {
                    diff[a] = x[3 * i + a] - x[3 * j + a];
                    r2 += diff[a] * diff[a];
                }
                if r2 <= 0.0 || r2 >= cut2 {
                    continue;
                }
                let r = r2.sqrt();
                let e = (-ALPHA * (r - R0)).exp();
                energy += D0 * (e * e - 2.0 * e) - shift;
                let slope = (2.0 * D0 * ALPHA * e * (1.0 - e)) / r;
                for a in 0..3 {
                    let piece = slope * diff[a];
                    grad[3 * i + a] += piece;
                    grad[3 * j + a] -= piece;
                }
            }
        }
        (energy, grad)
    }

    fn lj(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let n = x.len() / 3;
        let mut energy = -0.0;
        let mut grad = Array1::zeros(x.len());
        for i in 0..n {
            for j in (i + 1)..n {
                let mut diff = [0.0; 3];
                let mut r2 = -0.0;
                for a in 0..3 {
                    diff[a] = x[3 * i + a] - x[3 * j + a];
                    r2 += diff[a] * diff[a];
                }
                if r2 <= 0.0 {
                    continue;
                }
                let inv = 1.0 / r2;
                let sr6 = inv * inv * inv;
                let sr12 = sr6 * sr6;
                energy += 4.0 * (sr12 - sr6);
                let slope = 24.0 * (sr6 - 2.0 * sr12) * inv;
                for a in 0..3 {
                    let piece = slope * diff[a];
                    grad[3 * i + a] += piece;
                    grad[3 * j + a] -= piece;
                }
            }
        }
        (energy, grad)
    }

    struct Job {
        dim: usize,
        maxiter: usize,
        gtol: f64,
        cap: bool,
        start: Array1<f64>,
        fg: fn(ArrayView1<f64>) -> (f64, Array1<f64>),
    }

    #[test]
    fn the_uphill_fallback_does_not_fire_on_the_oracle_set() {
        let betas = [
            Conjugacy::FletcherReeves,
            Conjugacy::DaiYuan,
            Conjugacy::HestenesStiefel,
            Conjugacy::PolakRibiere,
        ];
        let linesearch = LineSearch::Wolfe {
            c1: 1e-4,
            c2: 0.1,
            maxiter: 30,
        };
        let jobs = [
            Job {
                dim: 2,
                maxiter: 400,
                gtol: 1e-6,
                cap: false,
                start: Array1::from_iter((0..2).map(|i| if i % 2 == 0 { -1.2 } else { 1.0 })),
                fg: rose,
            },
            Job {
                dim: 10,
                maxiter: 1500,
                gtol: 1e-5,
                cap: false,
                start: Array1::from_iter((0..10).map(|i| if i % 2 == 0 { -1.2 } else { 1.0 })),
                fg: rose,
            },
            Job {
                dim: 8,
                maxiter: 200,
                gtol: 1e-5,
                cap: false,
                start: Array1::from_elem(8, 0.5),
                fg: scaled,
            },
            Job {
                dim: 6,
                maxiter: 400,
                gtol: 1e-4,
                cap: true,
                start: ndarray::array![0.0, 0.0, 0.0, 3.4, 0.15, -0.05],
                fg: morse,
            },
            Job {
                dim: 9,
                maxiter: 400,
                gtol: 1e-3,
                cap: true,
                start: ndarray::array![0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.4, 1.3, 0.1],
                fg: lj,
            },
        ];
        for beta in betas {
            for job in &jobs {
                let evals = std::sync::atomic::AtomicUsize::new(0);
                let fg = job.fg;
                let obj = Oracle::unbounded(job.dim, |x: ArrayView1<f64>| {
                    evals.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    fg(x)
                });
                let control = Control {
                    maxiter: job.maxiter,
                    gtol: job.gtol,
                    istep: 1.0,
                    maxmove: if job.cap { None } else { Some(f64::INFINITY) },
                    ftol_rel: None,
                    noise_floor: None,
                };
                super::UPHILL_FIRINGS.with(|c| c.set(0));
                minimize(
                    &obj,
                    job.start.clone(),
                    &control,
                    beta.clone(),
                    Restart::njws(),
                    linesearch,
                )
                .unwrap();
                let fired = super::UPHILL_FIRINGS.with(|c| c.get());
                assert_eq!(fired, 0, "{beta:?} dim {} fired {fired}", job.dim);
            }
        }
    }
}

#[cfg(test)]
mod oneshot_tests {
    use super::oneshot_linesearch;
    use crate::linesearch::LineSearch;
    use crate::method::Method;

    #[test]
    fn a_one_shot_nlcg_uses_the_cg_curvature_and_lbfgs_keeps_brent() {
        match oneshot_linesearch(&Method::polak_ribiere()) {
            LineSearch::Wolfe { c1, c2, maxiter } => {
                assert!((c1 - 1e-4).abs() < 1e-15);
                assert!((c2 - 0.1).abs() < 1e-15);
                assert_eq!(maxiter, 20);
            }
            other => panic!("nlcg {other:?}"),
        }
        assert!(matches!(
            oneshot_linesearch(&Method::lbfgs()),
            LineSearch::Brent { maxiter: 40, tol } if (tol - 1e-10).abs() < 1e-15
        ));
    }
}
