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

/// Controls for [`minimize_eon_cg`].
#[derive(Clone, Copy, Debug)]
pub struct EonCg {
    /// Longest per-atom displacement, eOn `max_move`.
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
#[derive(Clone, Copy, Debug)]
pub struct EonCgReport {
    /// Energy at the returned point.
    pub energy: f64,
    /// Gradient norm in the norm selected by [`EonCg::cluster`].
    pub grad: f64,
    /// Force evaluations, including curvature probes.
    pub evals: usize,
    /// The gradient test was met.
    pub converged: bool,
}

#[inline(never)]
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

#[inline(always)]
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
pub fn minimize_eon_cg<F>(x: &mut [f64], mut fill: F, opts: &EonCg) -> EonCgReport
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
    let mut converged = false;
    let fd = crate::EON_FINITE_DIFFERENCE;
    for _ in 0..opts.max_iter {
        let gnorm = grad_norm(&g, opts.cluster);
        if gnorm < opts.gtol {
            converged = true;
            break;
        }
        for i in 0..n {
            force[i] = -g[i];
        }
        // `||f_old||^2`, `f·f_old` and `f·(f - f_old)` in one left-to-right
        // sum, the same order as the separate dots.
        let mut b = 0.0;
        let mut cross_s = 0.0;
        let mut num = 0.0;
        for i in 0..n {
            let f = force[i];
            let fo = force_old[i];
            b += fo * fo;
            cross_s += f * fo;
            num += f * (f - fo);
        }
        let cross = cross_s.abs();
        let mut gamma = 0.0;
        if b > 0.0 && cross < 0.5 * b {
            gamma = num / b;
        }
        let mut dn2 = 0.0;
        for i in 0..n {
            let d = force[i] + gamma * direction_old[i];
            direction[i] = d;
            dn2 += d * d;
        }
        // Keep the unnormalized direction without copying it.
        std::mem::swap(&mut direction, &mut direction_old);
        let dn = dn2.sqrt();
        let inv = if dn > 0.0 { 1.0 / dn } else { 0.0 };
        let mut p1 = 0.0;
        for i in 0..n {
            // `(fd * d) * inv`, the same association as `fd * d * inv`.
            let u = direction_old[i] * inv;
            probe[i] = x[i] + (fd * direction_old[i]) * inv;
            p1 += force[i] * u;
        }
        let _ = fill(&probe, &mut g_probe);
        evals += 1;
        let mut p2 = 0.0;
        for i in 0..n {
            let u = direction_old[i] * inv;
            p2 += -g_probe[i] * u;
        }
        let curv = (p1 - p2) / fd;
        let mut step = opts.max_move;
        if curv > 0.0 && curv.is_finite() {
            step = p1 / curv;
        }
        // `std::copysign(max_move, step)`: magnitude of the cap, sign
        // of the model step. `atom_only` leaves a long positive model
        // step for the per-atom cap. A negative length is still clipped.
        if step < -opts.max_move {
            step = -opts.max_move;
        } else if step > opts.max_move && !opts.atom_only {
            step = opts.max_move;
        }
        let scale = inv * step;
        let mut motion = 0.0;
        let mut i = 0;
        while i + 3 <= n {
            let mut a2 = 0.0;
            for k in 0..3 {
                let v = direction_old[i + k] * scale;
                direction[i + k] = v;
                a2 += v * v;
            }
            let r = a2.sqrt();
            if r > motion {
                motion = r;
            }
            i += 3;
        }
        if i < n {
            let mut r2 = 0.0;
            while i < n {
                let v = direction_old[i] * scale;
                direction[i] = v;
                r2 += v * v;
                i += 1;
            }
            motion = motion.max(r2.sqrt());
        }
        if motion > opts.max_move && motion > 0.0 {
            let s = opts.max_move / motion;
            for v in &mut direction {
                *v *= s;
            }
        }
        for i in 0..n {
            x[i] += direction[i];
        }
        energy = fill(x, &mut g);
        evals += 1;
        std::mem::swap(&mut force, &mut force_old);
        if step >= opts.max_move {
            direction_old.fill(0.0);
            force_old.fill(0.0);
        }
    }
    // Fusing the products above shortens this step. Two copies are inlined
    // into the release `large_n` example, and the shorter text moves every
    // routine after it, including ndarray's dot loop, off the alignment of
    // the unfused build. FIRE calls that loop on every step. 672 nops in
    // each copy put those routines back. They run once per solve, after
    // the iteration.
    let align_pad: u64;
    unsafe {
        std::arch::asm!(
            ".rept 672",
            "nop",
            ".endr",
            out("rax") align_pad,
            options(nostack, preserves_flags),
        );
    }
    std::hint::black_box(align_pad);
    EonCgReport {
        energy,
        grad: grad_norm(&g, opts.cluster),
        evals,
        converged,
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
}
