//! Iteration limits and gradient tolerance.

use ndarray::ArrayView1;

/// Outer-loop controls for [`crate::minimize`].
#[derive(Clone, Debug)]
pub struct Control {
    /// Maximum CG iterations.
    pub maxiter: usize,
    /// Stop when `||g||_2 < gtol`.
    pub gtol: f64,
    /// Initial line-search step guess.
    pub istep: f64,
    /// Optional Euclidean cap on a proposed step (xtsci `maxmove`).
    pub maxmove: Option<f64>,
}

impl Default for Control {
    fn default() -> Self {
        Self {
            maxiter: 100,
            gtol: 1e-5,
            istep: 1.0,
            maxmove: None,
        }
    }
}

/// Gradient norm a [`crate::Solver`] compares to [`Control::gtol`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StopNorm {
    /// `||g||_2`: eOn `norm`, LAMMPS `two`.
    #[default]
    Two,
    /// Largest Euclidean norm over consecutive `(x, y, z)` triples:
    /// ASE `fmax`, eOn `max_atom`, LAMMPS `max`. A trailing partial
    /// triple counts as one atom.
    MaxAtom,
    /// Largest absolute component: eOn `max_component`, LAMMPS `inf`.
    Inf,
    /// `||g||_2 / sqrt(n)`, zero for an empty `g`: eOn `rms`.
    Rms,
}

impl StopNorm {
    /// This norm of `g`. A non-finite entry gives infinity under
    /// [`Self::MaxAtom`], [`Self::Inf`] and [`Self::Rms`], so no
    /// tolerance accepts it.
    pub fn of(self, g: ArrayView1<f64>) -> f64 {
        match self {
            Self::Two => crate::vecops::nrm2(g),
            Self::Inf => crate::vecops::nrminf(g),
            Self::Rms if g.is_empty() => 0.0,
            Self::Rms if crate::vecops::nrminf(g).is_infinite() => f64::INFINITY,
            Self::Rms => crate::vecops::nrm2(g) / (g.len() as f64).sqrt(),
            Self::MaxAtom => {
                let mut worst = 0.0_f64;
                let mut i = 0;
                while i < g.len() {
                    let end = (i + 3).min(g.len());
                    let r2: f64 = (i..end).map(|k| g[k] * g[k]).sum();
                    if !r2.is_finite() {
                        return f64::INFINITY;
                    }
                    worst = worst.max(r2.sqrt());
                    i = end;
                }
                worst
            }
        }
    }
}
