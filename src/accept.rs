//! How a session takes a proposed direction.

use std::collections::VecDeque;

use eindir_core::DifferentiableObjective;
use ndarray::Array1;

use crate::control::Control;
use crate::manifold::{Manifold, ManifoldKind};
use crate::step::{finite, scale_step, scale_step_atom};

const ENERGY_RISE: f64 = 1.0e-8;
const WINDOW: usize = 5;

/// eOn `lbfgs_accept`. Default takes the clipped step (one oracle).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Accept {
    /// Take the maxmove-clipped step. One oracle at the new point.
    #[default]
    None,
    /// Refuse a rise versus the previous energy (up to 10 halvings).
    Energy,
    /// Grippo–Lampariello–Lucidi window of the last five accepted values.
    Nonmonotone,
}

fn trial_point<O>(
    obj: &O,
    pos: &Array1<f64>,
    dir: &Array1<f64>,
    alpha: f64,
    control: &Control,
    atom_maxmove: Option<f64>,
    manifold: ManifoldKind,
) -> Array1<f64>
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let step = dir * alpha;
    let mut trial = manifold.retract(pos, &step);
    if let Some(cap) = atom_maxmove {
        scale_step_atom(pos, &mut trial, cap);
    } else if let Some(cap) = control.maxmove {
        scale_step(pos, &mut trial, cap);
    }
    obj.bounds().clip(trial.view())
}

fn push_energy(hist: &mut VecDeque<f64>, energy: f64) {
    hist.push_back(energy);
    while hist.len() > WINDOW {
        hist.pop_front();
    }
}

/// Apply `dir` under `accept`. Each trial is one `value_and_gradient`.
pub(crate) fn accept_step<O>(
    obj: &O,
    pos: &Array1<f64>,
    value: f64,
    grad: &Array1<f64>,
    dir: &Array1<f64>,
    control: &Control,
    accept: Accept,
    e_hist: &mut VecDeque<f64>,
    atom_maxmove: Option<f64>,
    manifold: ManifoldKind,
) -> (Array1<f64>, f64, Array1<f64>, bool)
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    accept_step_with_fallback(
        obj,
        pos,
        value,
        grad,
        dir,
        control,
        accept,
        e_hist,
        atom_maxmove,
        manifold,
        true,
    )
}

/// Accept a direction with an optional steepest-descent fallback.
/// A constrained direction must keep the fallback disabled because it
/// cannot inherit the proposed direction's feasible set.
pub(crate) fn accept_step_with_fallback<O>(
    obj: &O,
    pos: &Array1<f64>,
    value: f64,
    grad: &Array1<f64>,
    dir: &Array1<f64>,
    control: &Control,
    accept: Accept,
    e_hist: &mut VecDeque<f64>,
    atom_maxmove: Option<f64>,
    manifold: ManifoldKind,
    steepest_fallback: bool,
) -> (Array1<f64>, f64, Array1<f64>, bool)
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    match accept {
        Accept::None => {
            let trial = trial_point(obj, pos, dir, 1.0, control, atom_maxmove, manifold);
            let (ft, gt) = obj.value_and_gradient(trial.view());
            if !ft.is_finite() || gt.iter().any(|g| !g.is_finite()) {
                return (pos.clone(), value, grad.clone(), false);
            }
            push_energy(e_hist, ft);
            (trial, ft, gt, true)
        }
        Accept::Energy | Accept::Nonmonotone => {
            let mut ref_e = value;
            if accept == Accept::Nonmonotone {
                if let Some(m) = e_hist.iter().copied().reduce(f64::max) {
                    ref_e = m;
                }
            }
            let mut alpha = 1.0;
            for _ in 0..10 {
                let trial = trial_point(obj, pos, dir, alpha, control, atom_maxmove, manifold);
                let (ft, gt) = obj.value_and_gradient(trial.view());
                if finite(ft, gt.view()) && ft - ref_e <= ENERGY_RISE {
                    push_energy(e_hist, ft);
                    return (trial, ft, gt, true);
                }
                alpha *= 0.5;
            }
            if !steepest_fallback {
                return (pos.clone(), value, grad.clone(), false);
            }
            // The fallback faces the same test it exists to satisfy. It
            // was returned as moved unconditionally, so after refusing ten
            // scaled steps the policy could accept an uphill steepest step
            // it never measured -- a silent exception to the rule the
            // caller chose. A fallback that also fails reports unmoved,
            // and the caller's stall machinery owns what happens next.
            let sd = grad.mapv(|g| -g);
            let trial = trial_point(obj, pos, &sd, 0.1, control, atom_maxmove, manifold);
            let (ft, gt) = obj.value_and_gradient(trial.view());
            if finite(ft, gt.view()) && ft - ref_e <= ENERGY_RISE {
                push_energy(e_hist, ft);
                return (trial, ft, gt, true);
            }
            (pos.clone(), value, grad.clone(), false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
    use ndarray::{ArrayView1, array};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountQuad {
        evals: AtomicUsize,
    }

    impl Objective<f64> for CountQuad {
        fn dim(&self) -> usize {
            1
        }
        fn bounds(&self) -> &Bounds<f64> {
            use std::sync::OnceLock;
            static B: OnceLock<Bounds<f64>> = OnceLock::new();
            B.get_or_init(|| Bounds::new(array![-1e6], array![1e6], 0.0))
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            self.evals.fetch_add(1, Ordering::Relaxed);
            x[0] * x[0]
        }
    }
    impl Gradient<f64> for CountQuad {
        fn dim(&self) -> usize {
            1
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            array![2.0 * x[0]]
        }
    }
    impl DifferentiableObjective<f64> for CountQuad {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            (self.eval(x), self.grad(x))
        }
    }

    fn ctrl() -> Control {
        Control {
            maxiter: 1,
            gtol: 1e-12,
            istep: 1.0,
            maxmove: None,
        }
    }

    #[test]
    fn accept_none_is_one_oracle() {
        let obj = CountQuad {
            evals: AtomicUsize::new(0),
        };
        let pos = array![1.0];
        let dir = array![1.0];
        let g = array![2.0];
        let mut hist = VecDeque::new();
        let (x, f, _, moved) = accept_step(
            &obj,
            &pos,
            1.0,
            &g,
            &dir,
            &ctrl(),
            Accept::None,
            &mut hist,
            None,
            ManifoldKind::Euclidean,
        );
        assert!(moved);
        assert!((x[0] - 2.0).abs() < 1e-15);
        assert!((f - 4.0).abs() < 1e-15);
        assert_eq!(obj.evals.load(Ordering::Relaxed), 1);
    }

    struct NanQuad;

    impl Objective<f64> for NanQuad {
        fn dim(&self) -> usize {
            1
        }
        fn bounds(&self) -> &Bounds<f64> {
            use std::sync::OnceLock;
            static B: OnceLock<Bounds<f64>> = OnceLock::new();
            B.get_or_init(|| Bounds::new(array![-1e6], array![1e6], 0.0))
        }
        fn eval(&self, _x: ArrayView1<f64>) -> f64 {
            f64::INFINITY
        }
    }
    impl Gradient<f64> for NanQuad {
        fn dim(&self) -> usize {
            1
        }
        fn grad(&self, _x: ArrayView1<f64>) -> Array1<f64> {
            array![f64::NAN]
        }
    }
    impl DifferentiableObjective<f64> for NanQuad {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            (self.eval(x), self.grad(x))
        }
    }

    #[test]
    fn accept_none_refuses_a_non_finite_oracle() {
        let pos = array![1.0];
        let dir = array![1.0];
        let g = array![2.0];
        let mut hist = VecDeque::new();
        let (x, f, _, moved) = accept_step(
            &NanQuad,
            &pos,
            1.0,
            &g,
            &dir,
            &ctrl(),
            Accept::None,
            &mut hist,
            None,
            ManifoldKind::Euclidean,
        );
        assert!(!moved);
        assert!((x[0] - 1.0).abs() < 1e-15);
        assert!((f - 1.0).abs() < 1e-15);
    }

    #[test]
    fn accept_energy_uphill_does_not_take_ten_steepest_retries() {
        let obj = CountQuad {
            evals: AtomicUsize::new(0),
        };
        let pos = array![1.0];
        let dir = array![1.0];
        let g = array![2.0];
        let mut hist = VecDeque::new();
        let _ = accept_step(
            &obj,
            &pos,
            1.0,
            &g,
            &dir,
            &ctrl(),
            Accept::Energy,
            &mut hist,
            None,
            ManifoldKind::Euclidean,
        );
        // 10 rejected halvings + one short steepest fallback.
        assert_eq!(obj.evals.load(Ordering::Relaxed), 11);
    }
}
