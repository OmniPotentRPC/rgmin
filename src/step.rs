//! Shared accept / clip / scale for one line-search move.

use eindir_core::DifferentiableObjective;
use ndarray::{Array1, ArrayView1};

use crate::control::Control;
use crate::error::{Error, Result};
use crate::linesearch::LineSearch;

pub(crate) fn l2(g: &Array1<f64>) -> f64 {
    crate::vecops::nrm2(g.view())
}

/// An answer a decrease test may compare: value and gradient finite.
pub(crate) fn finite(f: f64, g: ArrayView1<'_, f64>) -> bool {
    f.is_finite() && g.iter().all(|v| v.is_finite())
}

pub(crate) fn non_finite() -> Error {
    Error::Oracle {
        what: "non-finite value or gradient",
    }
}

/// Value and gradient at the start of a relaxation. A non-finite answer
/// there defines no direction to search, so it is refused.
pub(crate) fn start_answer<O>(obj: &O, pos: &Array1<f64>) -> Result<(f64, Array1<f64>)>
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let (value, grad) = obj.value_and_gradient(pos.view());
    if !finite(value, grad.view()) {
        return Err(non_finite());
    }
    Ok((value, grad))
}

pub(crate) fn scale_step(origin: &Array1<f64>, trial: &mut Array1<f64>, cap: f64) {
    let mut n2 = 0.0;
    for i in 0..trial.len() {
        let d = trial[i] - origin[i];
        n2 += d * d;
    }
    let n = n2.sqrt();
    if n > cap && n > 0.0 {
        let s = cap / n;
        for i in 0..trial.len() {
            trial[i] = origin[i] + s * (trial[i] - origin[i]);
        }
    }
}

/// Largest per-atom displacement between `origin` and `trial`.
pub(crate) fn max_atom_motion(origin: &Array1<f64>, trial: &Array1<f64>) -> f64 {
    let n = trial.len().min(origin.len());
    let mut max_atom = 0.0;
    let mut i = 0;
    while i + 3 <= n {
        let dx = trial[i] - origin[i];
        let dy = trial[i + 1] - origin[i + 1];
        let dz = trial[i + 2] - origin[i + 2];
        let r = (dx * dx + dy * dy + dz * dz).sqrt();
        if r > max_atom {
            max_atom = r;
        }
        i += 3;
    }
    if i < n {
        let mut r2 = 0.0;
        while i < n {
            let d = trial[i] - origin[i];
            r2 += d * d;
            i += 1;
        }
        max_atom = max_atom.max(r2.sqrt());
    }
    max_atom
}

/// Largest per-atom length of a displacement.
pub(crate) fn max_atom_displacement(step: &Array1<f64>) -> f64 {
    let n = step.len();
    let mut max_atom = 0.0;
    let mut i = 0;
    while i + 3 <= n {
        let r = (step[i] * step[i] + step[i + 1] * step[i + 1] + step[i + 2] * step[i + 2]).sqrt();
        if r > max_atom {
            max_atom = r;
        }
        i += 3;
    }
    if i < n {
        let mut r2 = 0.0;
        while i < n {
            r2 += step[i] * step[i];
            i += 1;
        }
        max_atom = max_atom.max(r2.sqrt());
    }
    max_atom
}

/// Scale `step` so its largest per-atom displacement equals `cap`.
pub(crate) fn scale_displacement_to_atom_cap(step: &Array1<f64>, cap: f64) -> Array1<f64> {
    let motion = max_atom_displacement(step);
    if !cap.is_finite() || !motion.is_finite() || cap <= 0.0 || motion <= 0.0 {
        return Array1::zeros(step.len());
    }
    step * (cap / motion)
}

/// Shrink `step` so its largest per-atom displacement is at most `cap`.
pub(crate) fn cap_displacement_atom(step: &Array1<f64>, cap: f64) -> Array1<f64> {
    let motion = max_atom_displacement(step);
    if motion > cap && motion > 0.0 && cap > 0.0 {
        step * (cap / motion)
    } else {
        step.clone()
    }
}

/// eOn `maxAtomMotionAppliedV`: scale the whole step so the largest
/// per-atom displacement is at most `cap`.
pub(crate) fn scale_step_atom(origin: &Array1<f64>, trial: &mut Array1<f64>, cap: f64) {
    let max_atom = max_atom_motion(origin, trial);
    if max_atom > cap && max_atom > 0.0 {
        let s = cap / max_atom;
        for k in 0..trial.len() {
            trial[k] = origin[k] + s * (trial[k] - origin[k]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn atom_cap_does_not_crush_a_uniform_cluster_step() {
        // 2 atoms each move 0.15. Per-atom cap 0.2 keeps the step.
        // A Euclidean 0.2 cap would scale it down.
        let origin = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let mut trial = array![0.15, 0.0, 0.0, 1.15, 0.0, 0.0];
        scale_step_atom(&origin, &mut trial, 0.2);
        assert!((trial[0] - 0.15).abs() < 1e-15);
        assert!((trial[3] - 1.15).abs() < 1e-15);
        let mut eucl = array![0.15, 0.0, 0.0, 1.15, 0.0, 0.0];
        scale_step(&origin, &mut eucl, 0.2);
        assert!(eucl[0] < 0.15 - 1e-6);
    }
}

/// Line search, clip to bounds, optional maxmove. Returns `(x, f, |α|, moved)`.
pub(crate) fn take_step<O>(
    obj: &O,
    pos: &Array1<f64>,
    value: f64,
    dir: ArrayView1<'_, f64>,
    istep: f64,
    linesearch: LineSearch,
    control: &Control,
) -> (Array1<f64>, f64, f64, bool)
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let (npos, _, lsstep) =
        linesearch.search(|x| obj.value_and_gradient(x), pos.view(), dir, istep);
    let mut trial = obj.bounds().clip(npos.view());
    if let Some(cap) = control.maxmove {
        scale_step(pos, &mut trial, cap);
    }
    let (nval, grad) = obj.value_and_gradient(trial.view());
    let finite = nval.is_finite() && grad.iter().all(|g| g.is_finite());
    // Bounds and maxmove change the point returned by the line search.
    // Acceptance uses that point's value and directional derivative. At
    // equal rounded energies, strong curvature certifies a descent step.
    let curvature_decrease = match linesearch {
        LineSearch::Wolfe { c1, c2, .. } if finite && nval == value => {
            let (_, initial_grad) = obj.value_and_gradient(pos.view());
            let delta = &trial - pos;
            let initial_slope = initial_grad.dot(&delta);
            let trial_slope = grad.dot(&delta);
            initial_slope < 0.0
                && crate::linesearch::conditions::armijo(nval, value, 1.0, initial_slope, c1)
                && crate::linesearch::conditions::strong_curvature(trial_slope, initial_slope, c2)
        }
        _ => false,
    };
    if finite && (nval < value || curvature_decrease) {
        (trial, nval, lsstep, true)
    } else {
        (pos.clone(), value, 0.0, false)
    }
}

pub(crate) fn next_istep(lsstep: f64, control: &Control) -> f64 {
    if lsstep <= 0.0 {
        control.istep
    } else {
        lsstep * 0.5
    }
}
