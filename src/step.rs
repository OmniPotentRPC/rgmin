//! Shared accept / clip / scale for one line-search move.

use eindir_core::DifferentiableObjective;
use ndarray::{Array1, ArrayView1};

use crate::control::{Control, StopNorm};
use crate::error::{Error, Result};
use crate::linesearch::{LineSearch, LineSearchOptions};
use crate::report::Report;

pub(crate) fn l2(g: &Array1<f64>) -> f64 {
    crate::vecops::nrm2(g.view())
}

/// `bounds.clip`, returning `x` when every coordinate is strictly inside.
///
/// `f64::max` and `f64::min` keep the bits of a finite value strictly
/// between the two ends, so the clipped vector is that value. A point
/// on the boundary, a NaN, or a non-contiguous view still goes through
/// [`eindir_core::Bounds::clip`].
pub(crate) fn clip_owned(bounds: &eindir_core::Bounds<f64>, x: Array1<f64>) -> Array1<f64> {
    if point_strictly_inside(bounds, x.view()) {
        x
    } else {
        bounds.clip(x.view())
    }
}

/// [`clip_owned`] for a vector the caller already owns.
#[allow(dead_code)]
pub(crate) fn clip_mut(bounds: &eindir_core::Bounds<f64>, x: &mut Array1<f64>) {
    if point_strictly_inside(bounds, x.view()) {
        return;
    }
    let clipped = bounds.clip(x.view());
    if x.len() == clipped.len()
        && x.iter()
            .zip(clipped.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits())
    {
        return;
    }
    *x = clipped;
}

fn point_strictly_inside(bounds: &eindir_core::Bounds<f64>, x: ArrayView1<'_, f64>) -> bool {
    let (Some(xs), Some(lo), Some(hi)) =
        (x.as_slice(), bounds.low.as_slice(), bounds.high.as_slice())
    else {
        return false;
    };
    xs.len() == lo.len()
        && xs.len() == hi.len()
        && xs
            .iter()
            .zip(lo.iter())
            .zip(hi.iter())
            .all(|((v, l), h)| *v > *l && *v < *h)
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

/// A non-finite value or gradient is not a direction.
pub(crate) fn refuse_non_finite(value: f64, grad: &Array1<f64>) -> Result<()> {
    if value.is_finite() && grad.iter().all(|g| g.is_finite()) {
        Ok(())
    } else {
        Err(Error::Oracle {
            what: "non-finite value or gradient",
        })
    }
}

/// Value and gradient at the start of a relaxation. A non-finite answer
/// there defines no direction to search, so it is refused.
#[allow(dead_code)]
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

/// A cap that binds: positive and finite. Non-positive or NaN caps are
/// "no cap", the meaning `set_maxmove` and `set_atom_maxmove` give 0.
#[inline]
fn binding(cap: f64) -> bool {
    cap > 0.0 && cap.is_finite()
}

/// Default per-atom displacement cap for a method with no step control
/// of its own.
///
/// ASE `Optimizer.defaults` is `{"maxstep": 0.2}` in
/// `ase/optimize/optimize.py` at tag 3.29.0
/// (<https://gitlab.com/ase/ase/-/blob/3.29.0/ase/optimize/optimize.py>).
/// eOn `OptimizerOptions.maxMove` is `0.2` in `schema/eon_params.capnp`
/// at `4be157d4`
/// (<https://github.com/TheochemUI/eOn/blob/4be157d43265af6b1140aaac0e52e42add62693f/schema/eon_params.capnp>).
/// The number is in the coordinate unit. A geometry stored in angstroms
/// is capped at 0.2 Å. [`scale_step_atom`] limits the largest per-atom
/// displacement. It does not limit the Euclidean norm of the whole step.
///
/// BFGS, dogleg, and FIRE 2.0 do not install this value. See
/// [`crate::method::Method::installs_default_atom_cap`].
pub const DEFAULT_ATOM_MAXMOVE: f64 = 0.2;

/// Cap used when the caller has not set a per-atom value and the method
/// installs the default.
///
/// A positive finite Euclidean [`Control::maxmove`] is that caller's
/// step limit, so this returns `None` and the Euclidean cap stays in
/// force. Otherwise the cap is [`DEFAULT_ATOM_MAXMOVE`]. BFGS, dogleg,
/// and FIRE 2.0 do not call this for their default.
pub(crate) fn implicit_atom_cap(control: &Control) -> Option<f64> {
    match control.maxmove {
        Some(cap) if binding(cap) => None,
        // An infinite Euclidean cap is the caller's explicit request for
        // no cap at all: the coordinates are not atoms (hyperparameters,
        // a model's weights), so the default per-atom cap would be wrong.
        Some(cap) if cap.is_infinite() && cap > 0.0 => None,
        // A negative maxmove leaves every cap off, including this default.
        // Zero and NaN keep it. The C ABI maps a negative maxmove to
        // +infinity, which is the same "no cap" choice.
        Some(cap) if cap < 0.0 => None,
        _ => Some(DEFAULT_ATOM_MAXMOVE),
    }
}

pub(crate) fn scale_step(origin: &Array1<f64>, trial: &mut Array1<f64>, cap: f64) {
    if !binding(cap) {
        return;
    }
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

/// eOn `maxAtomMotionAppliedV`: scale the whole step so the largest
/// per-atom displacement is at most `cap`.
pub(crate) fn scale_step_atom(origin: &Array1<f64>, trial: &mut Array1<f64>, cap: f64) {
    if !binding(cap) {
        return;
    }
    let n = trial.len();
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
    if max_atom > cap && max_atom > 0.0 {
        let s = cap / max_atom;
        for k in 0..n {
            trial[k] = origin[k] + s * (trial[k] - origin[k]);
        }
    }
}

/// Largest per-atom (xyz triple) norm of `v`; a trailing partial triple
/// counts as one more atom, matching [`scale_step_atom`].
pub(crate) fn max_atom_norm(v: ArrayView1<'_, f64>) -> f64 {
    let n = v.len();
    let mut m: f64 = 0.0;
    let mut i = 0;
    while i + 3 <= n {
        let r = (v[i] * v[i] + v[i + 1] * v[i + 1] + v[i + 2] * v[i + 2]).sqrt();
        m = m.max(r);
        i += 3;
    }
    if i < n {
        let mut r2 = 0.0;
        while i < n {
            r2 += v[i] * v[i];
            i += 1;
        }
        m = m.max(r2.sqrt());
    }
    m
}

/// Largest per-atom displacement between `origin` and `trial`.
///
/// A trailing partial triple counts as one atom. The walk stops at the
/// shorter of the two vectors.
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
    max_atom_norm(step.view())
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

/// Largest `alpha` for which the step `alpha dir` obeys the cap in force:
/// the per-atom cap when set, else the Euclidean `control.maxmove`, else
/// unbounded.
pub(crate) fn cap_alpha(
    dir: ArrayView1<'_, f64>,
    control: &Control,
    atom_maxmove: Option<f64>,
) -> f64 {
    let (cap, len) = match (atom_maxmove.filter(|c| binding(*c)), control.maxmove) {
        (Some(cap), _) => (cap, max_atom_norm(dir)),
        (None, Some(cap)) if binding(cap) => (cap, crate::vecops::nrm2(dir)),
        _ => return f64::INFINITY,
    };
    if len > 0.0 && len.is_finite() {
        cap / len
    } else {
        f64::INFINITY
    }
}

/// One line-searched move, with the oracle answer at its end.
pub(crate) struct Taken {
    /// The accepted point, or the start when nothing was accepted.
    pub x: Array1<f64>,
    /// `f` at [`Self::x`].
    pub f: f64,
    /// `g` at [`Self::x`].
    pub g: Array1<f64>,
    /// Accepted step length; 0 when nothing was accepted.
    pub alpha: f64,
    /// Whether the move was accepted.
    pub moved: bool,
    /// No trial decreased the energy by `c * eps_f`. The point is the start.
    pub energy_noise: bool,
}

pub(crate) fn energy_noise_report(
    value: f64,
    coords: Array1<f64>,
    steps: usize,
    grad_norm: f64,
) -> Report {
    Report {
        value,
        coords,
        steps,
        grad_norm,
        stop_norm: StopNorm::Two,
        grad: None,
        status: crate::report::Status::NoiseFloor,
    }
}

/// Line search from the known `(value, grad)` at `pos`, inside the cap,
/// then clip to bounds.
///
/// The cap is applied before the search, as the bound `alpha_max` on its
/// trials ([`cap_alpha`]): every point the search evaluates obeys
/// `atom_maxmove` (which takes precedence) or the Euclidean
/// `control.maxmove`, and the accepted trial's value and gradient are the
/// oracle's answer at the returned point. A clip to the objective's box
/// that moves the accepted trial costs one more evaluation there.
#[allow(clippy::too_many_arguments)]
pub(crate) fn take_step<O>(
    obj: &O,
    pos: &Array1<f64>,
    value: f64,
    grad: &Array1<f64>,
    dir: ArrayView1<'_, f64>,
    istep: f64,
    linesearch: LineSearch,
    control: &Control,
    atom_maxmove: Option<f64>,
) -> Taken
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    take_step_with_options(
        obj,
        pos,
        value,
        grad,
        dir,
        istep,
        linesearch,
        control,
        atom_maxmove,
        LineSearchOptions::default(),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn take_step_with_options<O>(
    obj: &O,
    pos: &Array1<f64>,
    value: f64,
    grad: &Array1<f64>,
    dir: ArrayView1<'_, f64>,
    istep: f64,
    linesearch: LineSearch,
    control: &Control,
    atom_maxmove: Option<f64>,
    options: LineSearchOptions,
) -> Taken
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let amax = cap_alpha(dir, control, atom_maxmove);
    let mut best_decrease = f64::NEG_INFINITY;
    let mut saw_trial = false;
    let stalled = |saw: bool, best: f64| {
        saw && control
            .noise_floor
            .is_some_and(|floor| floor.stalled_on_energy(best))
    };
    let unmoved = |energy_noise: bool| Taken {
        x: pos.clone(),
        f: value,
        g: grad.clone(),
        alpha: 0.0,
        moved: false,
        energy_noise,
    };
    let Some(out) = linesearch.search_from_with_options(
        |x| {
            let (f, g) = obj.value_and_gradient(x);
            if f.is_finite() {
                saw_trial = true;
                best_decrease = best_decrease.max(value - f);
            }
            (f, g)
        },
        pos.view(),
        value,
        grad.view(),
        dir,
        istep,
        amax,
        options,
    ) else {
        return unmoved(stalled(saw_trial, best_decrease));
    };
    let trial = obj.bounds().clip(out.x.view());
    let (x, f, g) = if trial == out.x {
        (out.x, out.f, out.g)
    } else {
        let (f, g) = obj.value_and_gradient(trial.view());
        if f.is_finite() {
            saw_trial = true;
            best_decrease = best_decrease.max(value - f);
        }
        (trial, f, g)
    };
    if !f.is_finite() || g.iter().any(|value| !value.is_finite()) {
        return unmoved(stalled(saw_trial, best_decrease));
    }
    let accepted = match control.ftol_rel {
        Some(_) => f - value <= control.ftol_slack(value),
        None => {
            f < value
                || match linesearch {
                    LineSearch::Wolfe { c1, c2, .. } => {
                        // Bounds can change the line-search point. Check the
                        // measured gradient along the actual displacement.
                        let displacement = &x - pos;
                        options.accepts_approximate_wolfe(
                            f,
                            value,
                            crate::vecops::dot(g.view(), displacement.view()),
                            crate::vecops::dot(grad.view(), displacement.view()),
                            c1,
                            c2,
                        )
                    }
                    _ => false,
                }
        }
    };
    // No trial cleared `c * eps_f`. Keep the start, even if Wolfe would
    // have accepted a decrease inside that band.
    let energy_noise = stalled(saw_trial, best_decrease);
    if accepted && !energy_noise {
        Taken {
            x,
            f,
            g,
            alpha: out.alpha,
            moved: true,
            energy_noise: false,
        }
    } else {
        unmoved(energy_noise)
    }
}

/// The opening step of the next line search for a quasi-Newton method.
///
/// The direction `-H g` already carries the step's scale, so the first
/// trial is `control.istep` every iteration (Nocedal-Wright 3.5, with
/// `istep = 1`). Opening at half the last accepted step, as
/// [`next_istep`] does for a gradient direction, made the step collapse
/// geometrically under a line search that only shrinks, and L-BFGS then
/// stalled far from `gtol`.
pub(crate) fn qn_istep(control: &Control) -> f64 {
    control.istep
}

/// The opening step of the next line search when the direction is the
/// gradient (steepest descent, Adam): half the step last accepted,
/// or `control.istep` again after a step that did not move.
///
/// Nonlinear CG does not use this. Its opening step is
/// [`nlcg_opening_alpha`].
pub(crate) fn next_istep(lsstep: f64, control: &Control) -> f64 {
    if lsstep <= 0.0 {
        control.istep
    } else {
        lsstep * 0.5
    }
}

/// Opening step for nonlinear CG, Nocedal-Wright equation 3.60.
///
/// `α0 = α_{k-1} (g_{k-1}·d_{k-1}) / (g_k·d_k)`, then clamped to `cap`.
/// Both directional derivatives must be negative. Otherwise, or when the
/// previous step did not move, the result is `fallback` clamped to `cap`.
pub(crate) fn nlcg_opening_alpha(
    prev_alpha: f64,
    prev_gd: f64,
    gd: f64,
    cap: f64,
    fallback: f64,
) -> f64 {
    let cap = if cap.is_finite() && cap > 0.0 {
        cap
    } else {
        f64::INFINITY
    };
    let fb = if fallback.is_finite() && fallback > 0.0 {
        fallback.min(cap)
    } else {
        1.0_f64.min(cap)
    };
    if !(prev_alpha > 0.0 && prev_gd < 0.0 && gd < 0.0) {
        return fb;
    }
    let ratio = prev_gd / gd;
    if !ratio.is_finite() || ratio <= 0.0 {
        return fb;
    }
    let alpha = prev_alpha * ratio;
    if !alpha.is_finite() || alpha <= 0.0 {
        return fb;
    }
    alpha.min(cap)
}

#[cfg(test)]
mod tests {
    #[test]
    fn an_infinite_euclidean_cap_leaves_the_default_atom_cap_off() {
        use super::*;
        let mut control = Control::default();
        assert_eq!(implicit_atom_cap(&control), Some(DEFAULT_ATOM_MAXMOVE));
        control.maxmove = Some(f64::INFINITY);
        assert_eq!(implicit_atom_cap(&control), None);
        control.maxmove = Some(0.3);
        assert_eq!(implicit_atom_cap(&control), None);
    }

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

    #[test]
    fn cap_alpha_reaches_the_cap_exactly() {
        // Atom 1 moves (3, 4, 0) per unit alpha: norm 5. Cap 0.2 -> 0.04.
        let dir = array![3.0, 4.0, 0.0, 1.0, 0.0, 0.0];
        let ctl = Control::default();
        assert!((cap_alpha(dir.view(), &ctl, Some(0.2)) - 0.04).abs() < 1e-17);
        let euclid = Control {
            maxmove: Some(0.2),
            ..Control::default()
        };
        let n = (26.0_f64).sqrt();
        assert!((cap_alpha(dir.view(), &euclid, None) - 0.2 / n).abs() < 1e-17);
        // The per-atom cap takes precedence; no cap is unbounded.
        assert!((cap_alpha(dir.view(), &euclid, Some(0.2)) - 0.04).abs() < 1e-17);
        assert_eq!(cap_alpha(dir.view(), &ctl, None), f64::INFINITY);
        assert_eq!(
            cap_alpha(array![0.0, 0.0].view(), &ctl, Some(0.2)),
            f64::INFINITY
        );
    }

    #[test]
    fn a_non_positive_cap_never_reverses_a_step() {
        let origin = array![0.0, 0.0, 0.0];
        for cap in [0.0, -0.1, f64::NAN] {
            let mut t = array![1.0, 0.0, 0.0];
            scale_step(&origin, &mut t, cap);
            assert_eq!(t[0], 1.0, "euclid cap {cap}");
            scale_step_atom(&origin, &mut t, cap);
            assert_eq!(t[0], 1.0, "atom cap {cap}");
            let ctl = Control {
                maxmove: Some(cap),
                ..Control::default()
            };
            assert_eq!(cap_alpha(t.view(), &ctl, Some(cap)), f64::INFINITY);
        }
    }

    type Log = std::sync::Arc<std::sync::Mutex<Vec<(Array1<f64>, f64)>>>;

    /// `f = |x|^2 / 2` behind an evaluation log.
    #[allow(clippy::type_complexity)]
    fn logged() -> (
        crate::Oracle<impl Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync>,
        Log,
    ) {
        let log = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let l2 = log.clone();
        let obj = crate::Oracle::unbounded(2, move |x: ArrayView1<f64>| {
            let f = 0.5 * x.dot(&x);
            l2.lock().unwrap().push((x.to_owned(), f));
            (f, x.to_owned())
        });
        (obj, log)
    }

    #[test]
    fn the_accepted_value_is_the_oracle_value_at_the_capped_point() {
        // A long downhill step under a per-atom cap of 0.1 (a 2-vector is
        // one partial atom). The returned f must be what the oracle said
        // at the returned x, which the clamp-after-search order broke.
        let (obj, log) = logged();
        let pos = array![3.0, 4.0];
        let (f0, g0) = (12.5, pos.clone());
        let dir = array![-3.0, -4.0];
        for ls in [
            LineSearch::default(),
            LineSearch::Backtracking {
                c: 1e-4,
                beta: 0.5,
                maxiter: 20,
            },
            LineSearch::Wolfe {
                c1: 1e-4,
                c2: 0.9,
                maxiter: 20,
            },
        ] {
            log.lock().unwrap().clear();
            let t = take_step(
                &obj,
                &pos,
                f0,
                &g0,
                dir.view(),
                1.0,
                ls,
                &Control::default(),
                Some(0.1),
            );
            assert!(t.moved);
            assert!(((&t.x - &pos).dot(&(&t.x - &pos)).sqrt() - 0.1).abs() < 1e-12);
            let seen = log.lock().unwrap();
            let hit = seen
                .iter()
                .find(|(x, _)| *x == t.x)
                .expect("x never evaluated");
            assert_eq!(hit.1, t.f, "{ls:?}");
            assert_eq!(t.g, t.x, "{ls:?}");
            assert!(
                seen.iter()
                    .all(|(x, _)| (x - &pos).dot(&(x - &pos)).sqrt() <= 0.1 + 1e-12)
            );
        }
    }

    #[test]
    fn backtracking_on_an_uphill_direction_descends() {
        // At f79dd98 backtracking took the Armijo test with a positive
        // slope, which any rise smaller than c alpha slope passes.
        let (obj, _) = logged();
        let pos = array![1.0, 0.0];
        let up = array![1.0, 0.0];
        let ls = LineSearch::Backtracking {
            c: 1e-4,
            beta: 0.5,
            maxiter: 20,
        };
        let t = take_step(
            &obj,
            &pos,
            0.5,
            &pos.clone(),
            up.view(),
            1.0,
            ls,
            &Control::default(),
            None,
        );
        assert!(t.moved);
        assert!(t.f < 0.5, "rose to {}", t.f);
        assert!(t.x[0] < 1.0);
    }

    #[test]
    fn a_negative_maxmove_leaves_the_default_atom_cap_off() {
        let mut control = Control {
            maxmove: Some(-1.0),
            ..Control::default()
        };
        assert_eq!(implicit_atom_cap(&control), None);
        control.maxmove = Some(f64::NEG_INFINITY);
        assert_eq!(implicit_atom_cap(&control), None);
        control.maxmove = Some(0.0);
        assert_eq!(implicit_atom_cap(&control), Some(DEFAULT_ATOM_MAXMOVE));
        control.maxmove = Some(f64::NAN);
        assert_eq!(implicit_atom_cap(&control), Some(DEFAULT_ATOM_MAXMOVE));
    }

    #[test]
    fn an_interior_point_keeps_the_bits_clip_would_write() {
        use eindir_core::Bounds;
        let bounds = Bounds::new(
            ndarray::Array1::from_elem(4, -2.0),
            ndarray::Array1::from_elem(4, 2.0),
            0.0,
        );
        let interior = ndarray::array![-0.0, 1.0, -1.5, 0.25];
        let kept = clip_owned(&bounds, interior.clone());
        let via = bounds.clip(interior.view());
        assert_eq!(kept, interior);
        for (a, b) in kept.iter().zip(via.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        let outside = ndarray::array![3.0, -3.0, 0.0, f64::NAN];
        let owned = clip_owned(&bounds, outside.clone());
        let clipped = bounds.clip(outside.view());
        for (a, b) in owned.iter().zip(clipped.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        let mut slot = outside;
        clip_mut(&bounds, &mut slot);
        for (a, b) in slot.iter().zip(clipped.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
}
