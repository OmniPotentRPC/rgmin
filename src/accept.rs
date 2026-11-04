//! How a session takes a proposed direction.

use std::cmp::Ordering;
use std::collections::VecDeque;

use eindir_core::DifferentiableObjective;
use ndarray::Array1;

use crate::control::Control;
use crate::manifold::{Manifold, ManifoldKind};
use crate::step::{
    cap_displacement_atom, finite, l2, max_atom_motion, scale_step, scale_step_atom,
};

pub(crate) const ENERGY_RISE: f64 = 1.0e-8;
const WINDOW: usize = 5;
/// Shrink passes after the first retraction. Each must cut the overshoot.
const SHRINK_PASSES: usize = 8;
/// Halvings of the proposed step before the steepest-descent fallback.
pub(crate) const HALVINGS: usize = 10;
/// Length of the steepest-descent fallback relative to the gradient.
pub(crate) const FALLBACK_STEP: f64 = 0.1;

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

/// Retract the tangent `step` at `pos` on a curved kind, inside the cap.
///
/// The step is capped first. A retraction that moves farther than the cap
/// shrinks the step by cap over motion and retracts again. `None` when a
/// pass does not cut the overshoot, or when the box clip would move the
/// point off the set. Returns the point and the tangent step it took.
pub(crate) fn retract_within_cap<O>(
    obj: &O,
    pos: &Array1<f64>,
    step: &Array1<f64>,
    atom_maxmove: Option<f64>,
    maxmove: Option<f64>,
    manifold: ManifoldKind,
) -> Option<(Array1<f64>, Array1<f64>)>
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let motion = |trial: &Array1<f64>| match atom_maxmove {
        Some(_) => max_atom_motion(pos, trial),
        None => l2(&(trial - pos)),
    };
    let mut step = match (atom_maxmove, maxmove) {
        (Some(cap), _) => cap_displacement_atom(step, cap),
        (None, Some(cap)) => {
            let norm = l2(step);
            if norm > cap && norm > 0.0 {
                step * (cap / norm)
            } else {
                step.clone()
            }
        }
        (None, None) => step.clone(),
    };
    let cap = atom_maxmove.or(maxmove);
    let mut trial = manifold.retract(pos, &step);
    if let Some(cap) = cap {
        let mut excess = f64::INFINITY;
        let mut passes = 0;
        loop {
            let moved = motion(&trial);
            if moved <= cap {
                break;
            }
            let over = moved - cap;
            if over.partial_cmp(&excess) != Some(Ordering::Less) || passes == SHRINK_PASSES {
                return None;
            }
            excess = over;
            passes += 1;
            step *= cap / moved;
            trial = manifold.retract(pos, &step);
        }
    }
    let bounds = obj.bounds();
    let inside = (trial.iter().zip(&bounds.low).zip(&bounds.high))
        .all(|((&t, &lo), &hi)| t.max(lo).min(hi) == t);
    if !inside {
        return None;
    }
    Some((trial, step))
}

/// The point `accept_step` evaluates for `dir * alpha`. A flat kind
/// retracts, caps the chord and clips. A curved kind projects the step onto
/// the tangent space at `pos` and uses [`retract_within_cap`], and `None`
/// refuses the trial before any oracle.
fn trial_point<O>(
    obj: &O,
    pos: &Array1<f64>,
    dir: &Array1<f64>,
    alpha: f64,
    control: &Control,
    atom_maxmove: Option<f64>,
    manifold: ManifoldKind,
) -> Option<Array1<f64>>
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let step = dir * alpha;
    if !manifold.is_flat() {
        let tangent = manifold.project(pos, &step);
        return retract_within_cap(obj, pos, &tangent, atom_maxmove, control.maxmove, manifold)
            .map(|(trial, _)| trial);
    }
    let mut trial = manifold.retract(pos, &step);
    if let Some(cap) = atom_maxmove {
        scale_step_atom(pos, &mut trial, cap);
    } else if let Some(cap) = control.maxmove {
        scale_step(pos, &mut trial, cap);
    }
    Some(obj.bounds().clip(trial.view()))
}

pub(crate) fn push_energy(hist: &mut VecDeque<f64>, energy: f64) {
    hist.push_back(energy);
    while hist.len() > WINDOW {
        hist.pop_front();
    }
}

/// Outcome of the one trial of [`Accept::None`].
pub(crate) enum Trial {
    /// Point, value and gradient of the evaluated trial.
    Accepted(Array1<f64>, f64, Array1<f64>),
    /// A curved kind found no trial inside the box and the cap. No oracle
    /// call was made.
    OffManifold,
    /// The oracle returned a non-finite value or gradient at the trial.
    NonFinite,
}

/// Take `dir` once, as [`Accept::None`] does, and say why a trial failed.
pub(crate) fn single_trial<O>(
    obj: &O,
    pos: &Array1<f64>,
    dir: &Array1<f64>,
    control: &Control,
    e_hist: &mut VecDeque<f64>,
    atom_maxmove: Option<f64>,
    manifold: ManifoldKind,
) -> Trial
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let Some(trial) = trial_point(obj, pos, dir, 1.0, control, atom_maxmove, manifold) else {
        return Trial::OffManifold;
    };
    let (ft, gt) = obj.value_and_gradient(trial.view());
    if !ft.is_finite() || gt.iter().any(|g| !g.is_finite()) {
        return Trial::NonFinite;
    }
    push_energy(e_hist, ft);
    Trial::Accepted(trial, ft, gt)
}

/// The policy a session applies to each trial step, with the energy
/// history that the nonmonotone test reads and every accepted step extends.
pub(crate) struct Acceptance<'a> {
    pub(crate) control: &'a Control,
    pub(crate) accept: Accept,
    pub(crate) e_hist: &'a mut VecDeque<f64>,
    pub(crate) atom_maxmove: Option<f64>,
    pub(crate) manifold: ManifoldKind,
}

/// Apply `dir` under `accept`. Each trial is one `value_and_gradient`.
pub(crate) fn accept_step<O>(
    obj: &O,
    pos: &Array1<f64>,
    value: f64,
    grad: &Array1<f64>,
    dir: &Array1<f64>,
    acceptance: Acceptance<'_>,
) -> (Array1<f64>, f64, Array1<f64>, bool)
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    accept_step_with_fallback(obj, pos, value, grad, dir, acceptance, true)
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
    acceptance: Acceptance<'_>,
    steepest_fallback: bool,
) -> (Array1<f64>, f64, Array1<f64>, bool)
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let Acceptance {
        control,
        accept,
        e_hist,
        atom_maxmove,
        manifold,
    } = acceptance;
    match accept {
        Accept::None => {
            match single_trial(obj, pos, dir, control, e_hist, atom_maxmove, manifold) {
                Trial::Accepted(trial, ft, gt) => (trial, ft, gt, true),
                Trial::OffManifold | Trial::NonFinite => (pos.clone(), value, grad.clone(), false),
            }
        }
        Accept::Energy | Accept::Nonmonotone => {
            let mut ref_e = value;
            if accept == Accept::Nonmonotone
                && let Some(m) = e_hist.iter().copied().reduce(f64::max)
            {
                ref_e = m;
            }
            let mut alpha = 1.0;
            for _ in 0..HALVINGS {
                if let Some(trial) =
                    trial_point(obj, pos, dir, alpha, control, atom_maxmove, manifold)
                {
                    let (ft, gt) = obj.value_and_gradient(trial.view());
                    if finite(ft, gt.view()) && ft - ref_e <= ENERGY_RISE {
                        push_energy(e_hist, ft);
                        return (trial, ft, gt, true);
                    }
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
            let Some(trial) = trial_point(
                obj,
                pos,
                &sd,
                FALLBACK_STEP,
                control,
                atom_maxmove,
                manifold,
            ) else {
                return (pos.clone(), value, grad.clone(), false);
            };
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
            Acceptance {
                control: &ctrl(),
                accept: Accept::None,
                e_hist: &mut hist,
                atom_maxmove: None,
                manifold: ManifoldKind::Euclidean,
            },
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
            Acceptance {
                control: &ctrl(),
                accept: Accept::None,
                e_hist: &mut hist,
                atom_maxmove: None,
                manifold: ManifoldKind::Euclidean,
            },
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
            Acceptance {
                control: &ctrl(),
                accept: Accept::Energy,
                e_hist: &mut hist,
                atom_maxmove: None,
                manifold: ManifoldKind::Euclidean,
            },
        );
        // 10 rejected halvings + one short steepest fallback.
        assert_eq!(obj.evals.load(Ordering::Relaxed), 11);
    }

    /// A linear objective inside an explicit box.
    struct Boxed {
        bounds: Bounds<f64>,
        evals: AtomicUsize,
    }

    impl Boxed {
        fn new(lo: Array1<f64>, hi: Array1<f64>) -> Self {
            Self {
                bounds: Bounds::new(lo, hi, 0.0),
                evals: AtomicUsize::new(0),
            }
        }
    }

    impl Objective<f64> for Boxed {
        fn dim(&self) -> usize {
            self.bounds.dims
        }
        fn bounds(&self) -> &Bounds<f64> {
            &self.bounds
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            self.evals.fetch_add(1, Ordering::Relaxed);
            x.sum()
        }
    }
    impl Gradient<f64> for Boxed {
        fn dim(&self) -> usize {
            self.bounds.dims
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            Array1::ones(x.len())
        }
    }
    impl DifferentiableObjective<f64> for Boxed {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            (self.eval(x), self.grad(x))
        }
    }

    #[test]
    fn flat_kinds_keep_retract_then_cap_then_clip() {
        let kinds = [
            (ManifoldKind::Euclidean, 6),
            (ManifoldKind::RigidQuotient, 6),
            (ManifoldKind::MwRigid, 6),
            (ManifoldKind::Symmetric, 4),
            (ManifoldKind::SkewSymmetric, 4),
            (ManifoldKind::euclidean_complex(2), 4),
            (ManifoldKind::centered_matrix(2, 2, false), 4),
            (ManifoldKind::constant(4), 4),
        ];
        for (kind, n) in kinds {
            assert!(kind.is_flat(), "{kind:?}");
            let wave = |k: usize, f: f64| Array1::from_shape_fn(n, |i| ((i + k) as f64 * f).sin());
            let obj = Boxed::new(Array1::from_elem(n, -0.6), Array1::from_elem(n, 0.6));
            for k in 0..5 {
                let pos = wave(k, 0.7) * 0.5;
                let dir = wave(k, 1.3);
                for (atom, maxmove) in [(None, None), (None, Some(0.05)), (Some(0.05), None)] {
                    let control = Control { maxmove, ..ctrl() };
                    let step = &dir * 0.5;
                    let mut expect = kind.retract(&pos, &step);
                    if let Some(cap) = atom {
                        scale_step_atom(&pos, &mut expect, cap);
                    } else if let Some(cap) = maxmove {
                        scale_step(&pos, &mut expect, cap);
                    }
                    let expect = obj.bounds().clip(expect.view());
                    let got = trial_point(&obj, &pos, &dir, 0.5, &control, atom, kind).unwrap();
                    assert_eq!(got, expect, "{kind:?} k={k} atom={atom:?} cap={maxmove:?}");
                }
            }
        }
    }

    #[test]
    fn a_box_that_cuts_the_sphere_refuses_the_trial_unevaluated() {
        let obj = Boxed::new(array![-2.0, -0.01, -2.0], array![2.0, 2.0, 2.0]);
        let pos = array![1.0, 0.0, 0.0];
        let dir = array![0.0, -0.05, 0.0];
        assert!(trial_point(&obj, &pos, &dir, 1.0, &ctrl(), None, ManifoldKind::Sphere).is_none());
        let mut hist = VecDeque::new();
        let (x, _, _, moved) = accept_step(
            &obj,
            &pos,
            1.0,
            &array![1.0, 1.0, 1.0],
            &dir,
            Acceptance {
                control: &ctrl(),
                accept: Accept::None,
                e_hist: &mut hist,
                atom_maxmove: None,
                manifold: ManifoldKind::Sphere,
            },
        );
        assert!(!moved);
        assert_eq!(x, pos);
        assert_eq!(obj.evals.load(Ordering::Relaxed), 0);
    }
}
