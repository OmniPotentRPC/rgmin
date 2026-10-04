//! Shared accept / clip / scale for one line-search move.

use eindir_core::DifferentiableObjective;
use ndarray::{Array1, ArrayView1};

use crate::control::Control;
use crate::linesearch::LineSearch;

pub(crate) fn l2(g: &Array1<f64>) -> f64 {
    crate::vecops::nrm2(g.view())
}

/// A cap that binds: positive and finite. Non-positive or NaN caps are
/// "no cap", the meaning `set_maxmove` and `set_atom_maxmove` give 0.
#[inline]
fn binding(cap: f64) -> bool {
    cap > 0.0 && cap.is_finite()
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
    let amax = cap_alpha(dir, control, atom_maxmove);
    let unmoved = || Taken {
        x: pos.clone(),
        f: value,
        g: grad.clone(),
        alpha: 0.0,
        moved: false,
    };
    let Some(out) = linesearch.search_from(
        |x| obj.value_and_gradient(x),
        pos.view(),
        value,
        grad.view(),
        dir,
        istep,
        amax,
    ) else {
        return unmoved();
    };
    let trial = obj.bounds().clip(out.x.view());
    let (x, f, g) = if trial == out.x {
        (out.x, out.f, out.g)
    } else {
        let (f, g) = obj.value_and_gradient(trial.view());
        (trial, f, g)
    };
    let accepted = match control.ftol_rel {
        Some(_) => f - value <= control.ftol_slack(value),
        None => {
            f < value
                || match linesearch {
                    LineSearch::Wolfe { c1, c2, .. } => {
                        // Bounds can change the line-search point. Check the
                        // measured gradient along the actual displacement.
                        let displacement = &x - pos;
                        crate::linesearch::conditions::roundoff_strong_wolfe(
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
    if accepted {
        Taken {
            x,
            f,
            g,
            alpha: out.alpha,
            moved: true,
        }
    } else {
        unmoved()
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
/// gradient (steepest descent, NLCG, Adam): half the step last accepted,
/// or `control.istep` again after a step that did not move.
pub(crate) fn next_istep(lsstep: f64, control: &Control) -> f64 {
    if lsstep <= 0.0 {
        control.istep
    } else {
        lsstep * 0.5
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
}
