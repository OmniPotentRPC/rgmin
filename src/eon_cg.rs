//! Line-search-free conjugate gradients, the eOn client step.
//!
//! `ConjugateGradients::single_step` at TheochemUI/eOn
//! `579738a9387e8d0ea55496f45dd057812cfab892`: Polak-Ribiere on the
//! force, restarted when the forces are not sufficiently orthogonal,
//! with the step length from one curvature probe. No line search.
//!
//! Hestenes and Stiefel, *Methods of conjugate gradients for solving
//! linear systems*, <https://doi.org/10.6028/jres.049.044>.
//! Polak and Ribiere, *Note sur la convergence de methodes de directions
//! conjuguees*, Revue Francaise d'Informatique et de Recherche
//! Operationnelle 16 (1969).
//!
//! The buffers live for the whole solve. A step does not allocate.
//! An unset observer adds no allocation and no callback.

use ndarray::ArrayView1;

use crate::observer::{IterationObserver, IterationView, ObserverSignal};

/// Controls for [`minimize_eon_cg`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub struct EonCg {
    /// Longest per-atom displacement, eOn `max_move`. A non-positive or
    /// non-finite value is no cap; a step along non-positive curvature
    /// then has length [`crate::EON_MAX_MOVE`].
    pub max_move: f64,
    /// Stop when the selected gradient norm is below this value.
    pub gtol: f64,
    /// Outer iterations.
    pub max_iter: usize,
    /// `true`: max atom force. `false`: infinity norm.
    pub cluster: bool,
    /// Skip the scalar clip and keep only the per-atom cap.
    ///
    /// The eOn client clips the model length with
    /// `std::copysign(max_move, step)` and then applies the per-atom
    /// cap. On a unit direction the scalar clip already limits the
    /// Euclidean length, so the per-atom cap does not bind. Rust
    /// `step.copysign(max_move)` has the arguments the other way
    /// round: a positive model step longer than the cap is left
    /// unchanged, and only the per-atom cap scales it. The Euclidean
    /// length can then exceed `max_move`. That is what cut
    /// Rosenbrock10 from 883 force calls to 297. `atom_only` is that
    /// positive-step behaviour, without flipping the sign of a
    /// negative model step. Off by default, so the client count is
    /// unchanged.
    pub atom_only: bool,
}

with_setters!(EonCg {
    with_max_move => max_move: val f64,
    with_gtol => gtol: val f64,
    with_max_iter => max_iter: val usize,
    with_cluster => cluster: val bool,
    with_atom_only => atom_only: val bool,
});

impl Default for EonCg {
    fn default() -> Self {
        Self {
            max_move: crate::EON_MAX_MOVE,
            gtol: 1e-5,
            max_iter: 20_000,
            cluster: false,
            atom_only: false,
        }
    }
}

/// Result of [`minimize_eon_cg`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub struct EonCgReport {
    /// Energy at the returned point.
    pub energy: f64,
    /// Gradient norm in the norm selected by [`EonCg::cluster`].
    pub grad: f64,
    /// Force evaluations, including curvature probes.
    pub evals: usize,
    /// The gradient test holds at the returned point, including a start
    /// that meets it with no iteration and a last iteration that reaches it.
    pub converged: bool,
    /// The observer asked the loop to stop at the returned point.
    pub observer_stop: bool,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        s += *x * *y;
    }
    s
}

fn max_atom(v: &[f64]) -> f64 {
    let mut m = 0.0;
    let mut i = 0;
    while i + 3 <= v.len() {
        let r = (v[i] * v[i] + v[i + 1] * v[i + 1] + v[i + 2] * v[i + 2]).sqrt();
        if r > m {
            m = r;
        }
        i += 3;
    }
    if i < v.len() {
        let mut r2 = 0.0;
        while i < v.len() {
            r2 += v[i] * v[i];
            i += 1;
        }
        m = m.max(r2.sqrt());
    }
    m
}

fn grad_norm(g: &[f64], cluster: bool) -> f64 {
    if !cluster {
        let mut m = 0.0_f64;
        for v in g {
            m = m.max(v.abs());
        }
        return m;
    }
    max_atom(g)
}

/// Minimize with the eOn conjugate-gradient step.
///
/// `fill` writes the gradient of the energy it returns into `g`.
/// The solve allocates its workspace once.
pub fn minimize_eon_cg<F>(x: &mut [f64], fill: F, opts: &EonCg) -> EonCgReport
where
    F: FnMut(&[f64], &mut [f64]) -> f64,
{
    minimize_eon_cg_observed(x, fill, opts, None)
}

/// [`minimize_eon_cg`] with an optional per-iteration observer.
///
/// The unset path does not allocate and does not call the observer.
pub fn minimize_eon_cg_observed<F>(
    x: &mut [f64],
    mut fill: F,
    opts: &EonCg,
    mut observer: Option<&mut dyn IterationObserver>,
) -> EonCgReport
where
    F: FnMut(&[f64], &mut [f64]) -> f64,
{
    let n = x.len();
    let mut g = vec![0.0; n];
    let mut force = vec![0.0; n];
    let mut force_old = vec![0.0; n];
    let mut direction = vec![0.0; n];
    let mut direction_old = vec![0.0; n];
    let mut probe = vec![0.0; n];
    let mut g_probe = vec![0.0; n];
    let mut energy = fill(x, &mut g);
    let mut evals = 1usize;
    let fd = crate::EON_FINITE_DIFFERENCE;
    let cap = Some(opts.max_move).filter(|&cap| crate::step::binding(cap));
    for iteration in 0..opts.max_iter {
        let gnorm = grad_norm(&g, opts.cluster);
        if gnorm < opts.gtol {
            break;
        }
        for i in 0..n {
            force[i] = -g[i];
        }
        let b = dot(&force_old, &force_old);
        let cross = dot(&force, &force_old).abs();
        let mut gamma = 0.0;
        if b > 0.0 && cross < 0.5 * b {
            let mut num = 0.0;
            for i in 0..n {
                num += force[i] * (force[i] - force_old[i]);
            }
            gamma = num / b;
        }
        for i in 0..n {
            direction[i] = force[i] + gamma * direction_old[i];
        }
        direction_old.copy_from_slice(&direction);
        force_old.copy_from_slice(&force);
        let dn = dot(&direction, &direction).sqrt();
        let inv = if dn > 0.0 { 1.0 / dn } else { 0.0 };
        for i in 0..n {
            probe[i] = x[i] + fd * direction[i] * inv;
        }
        let _ = fill(&probe, &mut g_probe);
        evals += 1;
        let mut p1 = 0.0;
        let mut p2 = 0.0;
        for i in 0..n {
            let u = direction[i] * inv;
            p1 += force[i] * u;
            p2 += -g_probe[i] * u;
        }
        let curv = (p1 - p2) / fd;
        let modelled = curv > 0.0 && curv.is_finite();
        let mut step = if modelled {
            p1 / curv
        } else {
            cap.unwrap_or(crate::EON_MAX_MOVE)
        };
        if let Some(cap) = cap {
            // `std::copysign(max_move, step)`: magnitude of the cap, sign
            // of the model step. `atom_only` leaves a long positive model
            // step for the per-atom cap. A negative length is still clipped.
            if step < -cap {
                step = -cap;
            } else if step > cap && !opts.atom_only {
                step = cap;
            }
        }
        for d in &mut direction {
            *d *= inv * step;
        }
        if let Some(cap) = cap {
            let motion = max_atom(&direction);
            if motion > cap && motion > 0.0 {
                let s = cap / motion;
                for v in &mut direction {
                    *v *= s;
                }
            }
        }
        let step_length = if observer.is_some() {
            dot(&direction, &direction).sqrt()
        } else {
            0.0
        };
        for i in 0..n {
            x[i] += direction[i];
        }
        energy = fill(x, &mut g);
        evals += 1;
        if let Some(obs) = observer.as_deref_mut()
            && obs.observe(IterationView {
                iteration,
                x: ArrayView1::from(&x[..]),
                value: energy,
                grad_norm: grad_norm(&g, opts.cluster),
                step_length,
            }) == ObserverSignal::Stop
        {
            let grad = grad_norm(&g, opts.cluster);
            return EonCgReport {
                energy,
                grad,
                evals,
                converged: false,
                observer_stop: true,
            };
        }
        if cap.map_or(!modelled, |cap| step >= cap) {
            direction_old.fill(0.0);
            force_old.fill(0.0);
        }
    }
    let grad = grad_norm(&g, opts.cluster);
    EonCgReport {
        energy,
        grad,
        evals,
        converged: grad < opts.gtol,
        observer_stop: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rosen(x: &[f64], g: &mut [f64]) -> f64 {
        g.fill(0.0);
        let mut f = 0.0;
        for i in 0..x.len() - 1 {
            let a = 1.0 - x[i];
            let b = x[i + 1] - x[i] * x[i];
            f += a * a + 100.0 * b * b;
            g[i] += -2.0 * a - 400.0 * x[i] * b;
            g[i + 1] += 200.0 * b;
        }
        f
    }

    /// Two atoms, isotropic spring. The model step is longer than
    /// `max_move`. The displacement has that length, not the length that
    /// would put the largest atom on the cap.
    #[test]
    fn a_long_model_step_is_capped_before_the_atom_cap() {
        fn spring(x: &[f64], g: &mut [f64]) -> f64 {
            let k = 0.1;
            let mut f = 0.0;
            for i in 0..x.len() {
                g[i] = k * x[i];
                f += 0.5 * k * x[i] * x[i];
            }
            f
        }
        let mut x = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let opts = EonCg {
            max_move: 0.2,
            gtol: 1e-12,
            max_iter: 1,
            cluster: true,
            atom_only: false,
        };
        let _ = minimize_eon_cg(&mut x, spring, &opts);
        let mut n2 = 0.0;
        let start = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        for i in 0..6 {
            let d = x[i] - start[i];
            n2 += d * d;
        }
        let disp = n2.sqrt();
        assert!(
            (disp - 0.2).abs() < 1e-12,
            "scalar cap 0.2, displacement {disp} x {x:?}"
        );
    }

    #[test]
    fn rosenbrock_2_matches_the_eon_evaluation_count() {
        let mut x = [-1.2, 1.0];
        let opts = EonCg {
            gtol: 1e-5,
            cluster: false,
            ..EonCg::default()
        };
        let rep = minimize_eon_cg(&mut x, rosen, &opts);
        assert!(rep.converged, "{rep:?} x {x:?}");
        assert_eq!(rep.evals, 87, "evals {}", rep.evals);
        assert!(rep.energy < 1e-8, "{}", rep.energy);
    }

    /// Two atoms, both away from the origin. The scalar cap holds the
    /// Euclidean length at `max_move`. The per-atom cap alone lets both
    /// atoms move by `max_move`.
    #[test]
    fn per_atom_only_lets_the_euclidean_length_exceed_the_cap() {
        fn spring(x: &[f64], g: &mut [f64]) -> f64 {
            let k = 0.1;
            let mut f = 0.0;
            for i in 0..x.len() {
                g[i] = k * x[i];
                f += 0.5 * k * x[i] * x[i];
            }
            f
        }
        let start = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut x = start;
        let opts = EonCg {
            max_move: 0.2,
            gtol: 1e-12,
            max_iter: 1,
            cluster: true,
            atom_only: true,
        };
        let _ = minimize_eon_cg(&mut x, spring, &opts);
        let mut n2 = 0.0_f64;
        let mut max_a = 0.0_f64;
        for atom in 0..2 {
            let mut r2 = 0.0;
            for c in 0..3 {
                let d = x[3 * atom + c] - start[3 * atom + c];
                r2 += d * d;
                n2 += d * d;
            }
            max_a = max_a.max(r2.sqrt());
        }
        let disp = n2.sqrt();
        assert!(
            (max_a - 0.2).abs() < 1e-9,
            "per-atom cap 0.2, largest atom {max_a}"
        );
        assert!(
            disp > 0.2 + 1e-6,
            "euclidean length {disp} should exceed the scalar cap"
        );
    }

    #[test]
    fn rosenbrock_10_matches_the_eon_evaluation_count() {
        let mut x = vec![-1.2, 1.0, -1.2, 1.0, -1.2, 1.0, -1.2, 1.0, -1.2, 1.0];
        let opts = EonCg {
            gtol: 1e-5,
            cluster: false,
            ..EonCg::default()
        };
        let rep = minimize_eon_cg(&mut x, rosen, &opts);
        assert!(rep.converged, "{rep:?}");
        assert_eq!(rep.evals, 883, "evals {}", rep.evals);
    }

    #[test]
    fn per_atom_only_rosenbrock_10_uses_the_long_model_step() {
        let mut x = vec![-1.2, 1.0, -1.2, 1.0, -1.2, 1.0, -1.2, 1.0, -1.2, 1.0];
        let opts = EonCg {
            gtol: 1e-5,
            cluster: false,
            atom_only: true,
            ..EonCg::default()
        };
        let rep = minimize_eon_cg(&mut x, rosen, &opts);
        assert!(rep.converged, "{rep:?} x {x:?}");
        assert_eq!(rep.evals, 297, "evals {}", rep.evals);
    }

    fn spring(x: &[f64], g: &mut [f64]) -> f64 {
        g.copy_from_slice(x);
        0.5 * dot(x, x)
    }

    /// The report reads the gradient it returns. A stationary start with
    /// no iteration, and a last iteration that lands on the minimum,
    /// both meet the test; before, only a test at the top of a further
    /// iteration set `converged`.
    #[test]
    fn converged_reads_the_returned_gradient() {
        let mut x = [0.0, 0.0, 0.0];
        let opts = EonCg {
            max_iter: 0,
            gtol: 1e-8,
            ..EonCg::default()
        };
        let rep = minimize_eon_cg(&mut x, spring, &opts);
        assert_eq!(rep.grad, 0.0);
        assert!(rep.converged, "{rep:?}");

        let mut x = [0.1, 0.0, 0.0];
        let opts = EonCg {
            max_iter: 1,
            gtol: 1e-8,
            max_move: 0.2,
            ..EonCg::default()
        };
        let rep = minimize_eon_cg(&mut x, spring, &opts);
        assert!(rep.grad < 1e-8, "{rep:?} x {x:?}");
        assert!(rep.converged, "{rep:?}");

        let mut x = [1.0, 0.0, 0.0];
        let rep = minimize_eon_cg(&mut x, spring, &opts);
        assert!(rep.grad >= 1e-8, "{rep:?}");
        assert!(!rep.converged, "{rep:?}");
    }
}
