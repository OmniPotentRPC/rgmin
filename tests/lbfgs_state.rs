//! Persistent L-BFGS: watch hook, budget, warm start.

use ndarray::{Array1, ArrayView1};
use rgmin::Lbfgs;

fn quad(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let scales = [1.0, 10.0, 100.0, 1000.0];
    let mut f = 0.0;
    let mut g = Array1::zeros(x.len());
    for i in 0..x.len() {
        let c = scales[i % scales.len()];
        f += 0.5 * c * x[i] * x[i];
        g[i] = c * x[i];
    }
    (f, g)
}

#[test]
fn watching_without_stopping_changes_nothing() {
    let x0 = Array1::from(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    let mut a = Lbfgs::default();
    let (fa, xa, ea) = a.minimize(x0.view(), 50, |v| Some(quad(v)));
    let mut b = Lbfgs::default();
    let (fb, xb, eb) = b.minimize_watched(x0.view(), 50, |v| Some(quad(v)), |_, _| true);
    assert_eq!(ea, eb);
    assert_eq!(fa, fb);
    assert_eq!(xa, xb);
}

#[test]
fn a_hook_that_refuses_stops_at_an_accepted_point() {
    let x0 = Array1::from(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    let mut opt = Lbfgs::default();
    let mut seen = Vec::new();
    let (f, _, _) = opt.minimize_watched(
        x0.view(),
        50,
        |v| Some(quad(v)),
        |it, fv| {
            seen.push(fv);
            it < 3
        },
    );
    assert_eq!(seen.len(), 4, "hook saw {} iterates", seen.len());
    assert_eq!(f, *seen.last().unwrap());
    assert!(f > 1e-10, "stopped hook still ran to convergence");
}

#[test]
fn converges_on_an_ill_conditioned_quadratic() {
    let mut opt = Lbfgs::default();
    let x0 = Array1::from(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    let (f, x, evals) = opt.minimize(x0.view(), 200, |v| Some(quad(v)));
    assert!(f < 1e-10, "did not converge, f = {f}, evals = {evals}");
    assert!(x.iter().all(|v| v.abs() < 1e-4));
}

#[test]
fn retained_curvature_costs_fewer_evaluations() {
    let x0 = Array1::from(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

    let mut warm = Lbfgs::default();
    let (_, xa, _) = warm.minimize(x0.view(), 200, |v| Some(quad(v)));
    let mut perturbed = xa.clone();
    for (i, v) in perturbed.iter_mut().enumerate() {
        *v += if i % 2 == 0 { 0.02 } else { -0.02 };
    }
    let (_, _, warm_evals) = warm.minimize(perturbed.view(), 200, |v| Some(quad(v)));

    let mut cold = Lbfgs::default();
    let (_, _, cold_evals) = cold.minimize(perturbed.view(), 200, |v| Some(quad(v)));

    assert!(
        warm_evals < cold_evals,
        "retained curvature should cost less: warm {warm_evals}, cold {cold_evals}"
    );
}

#[test]
fn forget_clears_the_memory() {
    let mut opt = Lbfgs::default();
    let x0 = Array1::from(vec![1.0, 1.0, 1.0, 1.0]);
    opt.minimize(x0.view(), 50, |v| Some(quad(v)));
    assert!(!opt.is_empty(), "a relaxation should store curvature");
    opt.forget();
    assert!(opt.is_empty());
}

#[test]
fn stops_when_the_budget_ends() {
    let mut opt = Lbfgs::default();
    let x0 = Array1::from(vec![1.0, 1.0, 1.0, 1.0]);
    let mut left = 3;
    let (_, _, evals) = opt.minimize(x0.view(), 200, |v| {
        if left == 0 {
            return None;
        }
        left -= 1;
        Some(quad(v))
    });
    assert!(evals <= 3, "spent {evals} with a budget of 3");
}

/// A recogniser that never fires must leave the relaxation identical to
/// the plain one, or measurements taken through one form do not
/// describe the other.
#[test]
fn recognising_nothing_changes_nothing() {
    let x0 = Array1::from(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    let mut a = Lbfgs::default();
    let (fa, xa, ea) = a.minimize(x0.view(), 50, |v| Some(quad(v)));
    let mut b = Lbfgs::default();
    let (fb, xb, eb, hit) = b.minimize_recognized(x0.view(), 50, |v| Some(quad(v)), |_, _, _| None);
    assert!(!hit);
    assert_eq!(ea, eb);
    assert_eq!(fa, fb);
    assert_eq!(xa, xb);
}

/// A recogniser that fires returns exactly its stand-in, flags it, and
/// spends fewer evaluations than the full descent: the refund is the
/// point, so the test measures it rather than assuming it.
#[test]
fn a_recognised_descent_returns_the_stand_in_and_refunds_the_rest() {
    let x0 = Array1::from(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    let mut full = Lbfgs::default();
    let (_, _, evals_full) = full.minimize(x0.view(), 50, |v| Some(quad(v)));
    let known = Array1::from(vec![0.0; 8]);
    let mut opt = Lbfgs::default();
    let known_for_hook = known.clone();
    let (f, x, evals, hit) = opt.minimize_recognized(
        x0.view(),
        50,
        |v| Some(quad(v)),
        move |_, _, at| {
            let d2: f64 = at.iter().map(|v| v * v).sum();
            (d2.sqrt() < 0.5).then(|| (0.0, known_for_hook.clone()))
        },
    );
    assert!(hit, "the descent enters the ball and must be recognised");
    assert_eq!(f, 0.0, "the stand-in value is returned verbatim");
    assert_eq!(x, known, "the stand-in point is returned verbatim");
    assert!(
        evals < evals_full,
        "recognition must refund evaluations: {evals} against {evals_full}"
    );
}

/// A broken gradient must never read as converged. f64::max returns its
/// other operand against NaN, so an infinity-norm fold over an all-NaN
/// gradient is zero, which passes any tolerance: measured before the
/// guard, this test terminated in one iteration reporting success at a
/// garbage point. No direction from such a start is defined, so the
/// relaxation ends there at `+inf`, the value no caller reads as a minimum,
/// without probing along the NaN direction.
#[test]
fn a_nan_gradient_is_not_convergence() {
    for norm in [rgmin::GradNorm::Infinity, rgmin::GradNorm::Euclidean] {
        let x0 = Array1::from(vec![1.0; 4]);
        let mut opt = Lbfgs::default();
        opt.norm = norm;
        opt.gtol = 1e-6;
        let mut calls = 0usize;
        let mut probed_nan = false;
        let (f, x, evals) = opt.minimize(x0.view(), 5, |v| {
            calls += 1;
            probed_nan |= v.iter().any(|c| !c.is_finite());
            let g = Array1::from(vec![f64::NAN; v.len()]);
            Some((1.0, g))
        });
        assert_eq!(
            f,
            f64::INFINITY,
            "an all-NaN gradient must not read as a minimum of value 1 \
             (norm {norm:?}, evals {evals})"
        );
        assert!(!probed_nan, "a NaN direction was probed (norm {norm:?})");
        assert_eq!((calls, evals, x), (1, 1, x0));
    }
}

/// The zoom carries slopes at both bracket ends, so on a smooth
/// anisotropic bowl the whole relaxation must land within a bounded
/// evaluation budget: a regression here means the cubic model stopped
/// being used or stopped being trusted.
#[test]
fn the_cubic_zoom_keeps_the_evaluation_budget() {
    let x0 = Array1::from(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    let mut opt = Lbfgs::default();
    opt.gtol = 1e-8;
    opt.norm = rgmin::GradNorm::Euclidean;
    let (f, _, evals) = opt.minimize(x0.view(), 200, |v| Some(quad(v)));
    assert!(f < 1e-12, "the bowl minimum is zero, got {f}");
    assert!(
        evals < 120,
        "an anisotropic quadratic must not need {evals} evaluations"
    );
}
