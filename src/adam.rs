//! Kingma and Ba Adam, with a line search along the preconditioned step.

use eindir_core::{DifferentiableObjective, Objective};
use ndarray::Array1;

use crate::control::Control;
use crate::error::{Error, Result};
use crate::linesearch::LineSearch;
use crate::report::Report;
use crate::step::{energy_noise_report, implicit_atom_cap, l2, max_atom_norm, take_step};

/// Adam plus a line search along the Kingma-Ba preconditioned step.
///
/// Biased moments and bias correction follow Algorithm 1:
/// `m <- beta1 m + (1-beta1) g`, `v <- beta2 v + (1-beta2) g*g`,
/// `mhat = m / (1-beta1^t)`, `vhat = v / (1-beta2^t)`.
/// The search direction is `-mhat / (sqrt(vhat) + eps)`, scaled so its
/// largest per-atom norm is 1. The line search supplies the length.
/// The next opening is that accepted length: halving it, on top of a
/// second moment that still remembers a large early gradient, collapses
/// the step.
///
/// Kingma and Ba, *Adam: A Method for Stochastic Optimization*,
/// <https://doi.org/10.48550/arXiv.1412.6980>.
pub fn minimize_adam<O>(
    obj: &O,
    init: impl Into<Array1<f64>>,
    control: &Control,
    linesearch: LineSearch,
    beta1: f64,
    beta2: f64,
    eps: f64,
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
    let mut m = Array1::<f64>::zeros(pos.len());
    let mut v = Array1::<f64>::zeros(pos.len());
    // beta1^t and beta2^t at t = 1, then multiplied by beta after each step.
    let mut b1p = beta1;
    let mut b2p = beta2;
    let mut istep = control.istep;

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
        let m_old = m.clone();
        let v_old = v.clone();
        let raw = adam_direction(&mut m, &mut v, &grad, beta1, beta2, b1p, b2p, eps);
        let dir = scale_to_unit_atom(&raw);
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
            return Ok(energy_noise_report(value, pos, step + 1, gnorm));
        }
        if t.moved {
            pos = t.x;
            value = t.f;
            grad = t.g;
            b1p *= beta1;
            b2p *= beta2;
            if t.alpha.is_finite() && t.alpha > 0.0 {
                istep = t.alpha;
            }
        } else {
            m = m_old;
            v = v_old;
        }
    }
    Ok(Report {
        value,
        coords: pos,
        steps: control.maxiter,
        grad_norm: l2(&grad),
        status: control.finish_status(l2(&grad), grad.len(), true),
    })
}

/// Kingma-Ba moment update and bias-corrected direction (Algorithm 1).
pub(crate) fn adam_direction(
    m: &mut Array1<f64>,
    v: &mut Array1<f64>,
    grad: &Array1<f64>,
    beta1: f64,
    beta2: f64,
    beta1_pow: f64,
    beta2_pow: f64,
    eps: f64,
) -> Array1<f64> {
    let n = grad.len();
    let mut dir = Array1::<f64>::zeros(n);
    for i in 0..n {
        m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
        v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i];
        let mhat = m[i] / (1.0 - beta1_pow);
        let vhat = v[i] / (1.0 - beta2_pow);
        dir[i] = -mhat / (vhat.sqrt() + eps);
    }
    dir
}

/// Scale `dir` so the largest per-atom displacement of a unit step is 1.
///
/// Kingma-Ba's vector is a preconditioner. Leaving its Euclidean length
/// attached to `m / sqrt(v)` makes a large early gradient, stored in `v`,
/// shrink every later step.
pub(crate) fn scale_to_unit_atom(dir: &Array1<f64>) -> Array1<f64> {
    let n = max_atom_norm(dir.view());
    if n.is_finite() && n > 0.0 {
        dir.mapv(|v| v / n)
    } else {
        dir.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linesearch::LineSearch;
    use crate::oracle::Oracle;
    use ndarray::{ArrayView1, array};

    #[test]
    fn first_step_bias_correction_recovers_the_gradient() {
        let mut m = array![0.0, 0.0];
        let mut v = array![0.0, 0.0];
        let g = array![1.0, 2.0];
        let dir = adam_direction(&mut m, &mut v, &g, 0.9, 0.999, 0.9, 0.999, 1e-8);
        assert!((m[0] - 0.1).abs() < 1e-15);
        assert!((m[1] - 0.2).abs() < 1e-15);
        assert!((v[0] - 0.001).abs() < 1e-15);
        assert!((v[1] - 0.004).abs() < 1e-15);
        // mhat = g, vhat = g*g, dir_i = -g_i / (|g_i| + eps)
        assert!((dir[0] + 1.0 / (1.0 + 1e-8)).abs() < 1e-15);
        assert!((dir[1] + 2.0 / (2.0 + 1e-8)).abs() < 1e-15);
    }

    #[test]
    fn second_step_uses_beta_to_the_t() {
        let mut m = array![0.0];
        let mut v = array![0.0];
        let g1 = array![1.0];
        let _ = adam_direction(&mut m, &mut v, &g1, 0.9, 0.999, 0.9, 0.999, 0.0);
        let g2 = array![3.0];
        let dir = adam_direction(&mut m, &mut v, &g2, 0.9, 0.999, 0.81, 0.998001, 0.0);
        // m = 0.9*0.1 + 0.1*3 = 0.39; mhat = 0.39 / (1-0.81) = 2.052631...
        // v = 0.999*0.001 + 0.001*9 = 0.009999; vhat = 0.009999 / (1-0.998001)
        let mhat: f64 = 0.39 / (1.0 - 0.81);
        let vhat: f64 = 0.009999 / (1.0 - 0.998001);
        let expect = -mhat / vhat.sqrt();
        assert!((m[0] - 0.39).abs() < 1e-15);
        assert!((v[0] - 0.009999).abs() < 1e-15);
        assert!((dir[0] - expect).abs() < 1e-12);
    }

    /// Backtracking cannot lengthen a collapsed Adam direction, and halving
    /// the next opening makes that worse. A unit-atom direction whose next
    /// opening is the accepted step reaches this bowl.
    #[test]
    fn backtracking_adam_reaches_a_quadratic_bowl() {
        let obj = Oracle::unbounded(1, |x: ArrayView1<f64>| (0.5 * x[0] * x[0], array![x[0]]));
        let control = Control {
            maxiter: 30,
            gtol: 1e-8,
            istep: 1.0,
            maxmove: None,
            ftol_rel: None,
            noise_floor: None,
        };
        let linesearch = LineSearch::Backtracking {
            c: 1e-4,
            beta: 0.5,
            maxiter: 6,
        };
        let report =
            minimize_adam(&obj, array![8.0], &control, linesearch, 0.9, 0.999, 1e-8).expect("adam");
        assert!(
            report.grad_norm < 1e-6,
            "||g|| = {} at {:?} after {} steps",
            report.grad_norm,
            report.coords,
            report.steps
        );
    }
}
