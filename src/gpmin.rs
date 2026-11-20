//! Gaussian-process-accelerated local geometry optimizer (GPMin).
//!
//! GPMin fits a Gaussian process (GP) surrogate to observed energies and
//! gradients using a squared-exponential covariance kernel with analytic
//! derivative blocks, then minimizes the GP posterior mean from the current
//! point with an inner L-BFGS search. When a proposed step does not lower
//! the true objective, the trial observation is added to the training set
//! and the surrogate is refined until a descent step is found or the
//! refinement budget is exhausted.
//!
//! Garijo del Río, Mortensen, and Jacobsen, *Local Bayesian optimizer for
//! atomic structures*, Phys. Rev. B 100, 104103 (2019),
//! <https://doi.org/10.1103/PhysRevB.100.104103>.
//!
//! Koistinen, Dagbjartsdóttir, Ásgeirsson, Vehtari, and Jónsson, *Nudged
//! elastic band calculations accelerated with Gaussian process regression*,
//! J. Chem. Phys. 147, 152720 (2017),
//! <https://doi.org/10.1063/1.4986787>.

use eindir_core::{DifferentiableObjective, Objective};
use ndarray::{Array1, Array2, ArrayView1};

use crate::control::Control;
use crate::error::{Error, Result};
use crate::lbfgs::{LbfgsH0, LbfgsPairs};
use crate::portable::exp as portable_exp;
use crate::report::{Report, Status};
use crate::step::{apply_caps, l2};
use crate::stepper::Stepper;

/// Default squared-exponential length scale when `update_hyperparams` is
/// `false`, matching ASE `ase.optimize.GPMin`.
pub const GPMIN_DEFAULT_SCALE: f64 = 0.4;

/// Default observation regularization (`noise`) when `update_hyperparams`
/// is `false`, matching ASE `ase.optimize.GPMin` runtime defaults.
pub const GPMIN_DEFAULT_NOISE: f64 = 0.001;

/// Default kernel amplitude (`weight`) when `update_hyperparams` is `false`,
/// matching ASE `ase.optimize.GPMin`.
pub const GPMIN_DEFAULT_WEIGHT: f64 = 1.0;

/// Default squared-exponential length scale when `update_hyperparams` is
/// `true`, matching ASE `ase.optimize.GPMin`.
pub const GPMIN_UPDATED_SCALE: f64 = 0.3;

/// Default observation regularization (`noise`) when `update_hyperparams`
/// is `true`, matching ASE `ase.optimize.GPMin` and Garijo del Río et al.
/// (2019).
pub const GPMIN_UPDATED_NOISE: f64 = 0.004;

/// Default kernel amplitude (`weight`) when `update_hyperparams` is `true`,
/// matching ASE `ase.optimize.GPMin`.
pub const GPMIN_UPDATED_WEIGHT: f64 = 2.0;

/// Default relative hyperparameter optimization bound `eps` when
/// `update_hyperparams` is `true`, matching ASE `ase.optimize.GPMin`
/// (`(1 - eps) * t0 <= t <= (1 + eps) * t0`).
pub const GPMIN_UPDATED_BOUNDS: f64 = 0.1;

/// Default batch size (update interval in observations) when
/// `update_hyperparams` is `true`, matching ASE `ase.optimize.GPMin`.
pub const GPMIN_UPDATED_BATCH_SIZE: usize = 1;

/// Default maximum surrogate refinement evaluations per outer step when a
/// proposed trial fails to lower the true energy, matching ASE
/// `ase.optimize.GPMin` (`count == 30`).
pub const GPMIN_DEFAULT_MAX_REFINEMENTS: usize = 30;

/// Default inner L-BFGS iteration limit for minimizing the GP posterior
/// mean.
pub const GPMIN_DEFAULT_SURROGATE_MAXITER: usize = 200;

/// Default L-BFGS pair memory for minimizing the GP posterior mean.
pub const GPMIN_DEFAULT_SURROGATE_MEMORY: usize = 10;

/// Default gradient tolerance for marginal log-likelihood hyperparameter
/// optimization, matching ASE `GaussianProcess.fit_hyperparameters`
/// (`tol = 0.01`).
pub const GPMIN_DEFAULT_HYPERPARAM_TOL: f64 = 0.01;

/// Default iteration cap for marginal log-likelihood hyperparameter
/// optimization.
pub const GPMIN_DEFAULT_HYPERPARAM_MAXITER: usize = 50;

/// Default inner surrogate L-BFGS gradient tolerance derived from machine
/// epsilon (`f64::EPSILON.powf(0.75)`).
#[must_use]
pub fn default_surrogate_gtol() -> f64 {
    f64::EPSILON.powf(0.75)
}

/// Strategy for updating the constant energy prior mean `m(x) = [c, 0, ..., 0]`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PriorStrategy {
    /// Update `c` to the maximum observed energy (`"maximum"`, ASE default).
    Maximum,
    /// Fix `c` to the first observed energy (`"init"`).
    Init,
    /// Update `c` to the arithmetic mean of all observed energies (`"average"`).
    Average,
    /// Hold `c` fixed at a caller-supplied finite constant (e.g. `0.0` for
    /// a zero prior).
    Fixed(f64),
}

impl PriorStrategy {
    /// Parse a strategy token (`"maximum"`, `"init"`, or `"average"`).
    pub fn from_name(name: &str) -> Result<Self> {
        let lower = name.trim().to_ascii_lowercase();
        match lower.as_str() {
            "maximum" => Ok(Self::Maximum),
            "init" => Ok(Self::Init),
            "average" => Ok(Self::Average),
            _ => Err(Error::GpMin {
                what: "update_prior_strategy must be 'maximum', 'init', or 'average'",
            }),
        }
    }
}

/// Configuration for [`GpMin`] and a `Method::GpMin` [`crate::Solver`]
/// session.
///
/// Non-exhaustive: start from [`Default`] or
/// [`Self::with_hyperparam_updates`] and chain the `with_*` setters.
///
/// [`Default`] selects ASE's fixed-hyperparameter defaults
/// (`scale = 0.4`, `noise = 0.001`, `weight = 1.0`,
/// `prior = PriorStrategy::Maximum`, `update_hyperparams = false`).
/// [`Self::with_hyperparam_updates`] selects ASE's updated-hyperparameter
/// defaults (`scale = 0.3`, `noise = 0.004`, `weight = 2.0`,
/// `batch_size = 1`, `hyperparam_bounds = Some(0.1)`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpMinConfig {
    /// Length scale `l > 0` of the squared-exponential kernel.
    pub scale: f64,
    /// Amplitude prefactor `w > 0` of the squared-exponential kernel.
    pub weight: f64,
    /// Regularization parameter `sigma_n > 0` (energy diagonal gets
    /// `(sigma_n * l)^2`, gradient diagonals get `sigma_n^2`).
    pub noise: f64,
    /// Constant or updateable prior mean strategy.
    pub prior: PriorStrategy,
    /// Whether to maximize marginal log-likelihood every `batch_size`
    /// observations.
    pub update_hyperparams: bool,
    /// Number of observations between marginal log-likelihood updates when
    /// `update_hyperparams` is `true`.
    pub batch_size: usize,
    /// Optional relative bound factor `eps` in `(0, 1)` constraining each
    /// hyperparameter `t` to `[(1 - eps) * t0, (1 + eps) * t0]` during
    /// marginal-likelihood updates. `None` leaves hyperparameters unbounded
    /// above zero.
    pub hyperparam_bounds: Option<f64>,
    /// Optional per-atom step cap applied by the standalone [`GpMin`]
    /// stepper. `None` leaves the standalone stepper uncapped; a
    /// [`crate::Solver`] session applies its own `atom_maxmove` and
    /// `maxmove` caps instead.
    pub max_step: Option<f64>,
    /// Maximum surrogate refinement evaluations per outer step when
    /// `E(r1) >= E(r0)`.
    pub max_refinements: usize,
    /// Maximum inner L-BFGS iterations when minimizing the GP posterior
    /// mean.
    pub surrogate_maxiter: usize,
    /// Number of L-BFGS curvature pairs retained during inner surrogate
    /// minimization.
    pub surrogate_memory: usize,
    /// Infinity-norm gradient tolerance for the inner L-BFGS surrogate
    /// minimization.
    pub surrogate_gtol: f64,
    /// Projected gradient infinity-norm tolerance for marginal
    /// log-likelihood hyperparameter optimization.
    pub hyperparam_tol: f64,
    /// Iteration cap for marginal log-likelihood hyperparameter
    /// optimization.
    pub hyperparam_maxiter: usize,
}

with_setters!(GpMinConfig {
    with_scale => scale: val f64,
    with_weight => weight: val f64,
    with_noise => noise: val f64,
    with_prior => prior: val PriorStrategy,
    with_update_hyperparams => update_hyperparams: val bool,
    with_batch_size => batch_size: val usize,
    with_hyperparam_bounds => hyperparam_bounds: opt f64,
    with_max_step => max_step: opt f64,
    with_max_refinements => max_refinements: val usize,
    with_surrogate_maxiter => surrogate_maxiter: val usize,
    with_surrogate_memory => surrogate_memory: val usize,
    with_surrogate_gtol => surrogate_gtol: val f64,
    with_hyperparam_tol => hyperparam_tol: val f64,
    with_hyperparam_maxiter => hyperparam_maxiter: val usize,
});

impl Default for GpMinConfig {
    fn default() -> Self {
        Self {
            scale: GPMIN_DEFAULT_SCALE,
            weight: GPMIN_DEFAULT_WEIGHT,
            noise: GPMIN_DEFAULT_NOISE,
            prior: PriorStrategy::Maximum,
            update_hyperparams: false,
            batch_size: GPMIN_UPDATED_BATCH_SIZE,
            hyperparam_bounds: None,
            max_step: None,
            max_refinements: GPMIN_DEFAULT_MAX_REFINEMENTS,
            surrogate_maxiter: GPMIN_DEFAULT_SURROGATE_MAXITER,
            surrogate_memory: GPMIN_DEFAULT_SURROGATE_MEMORY,
            surrogate_gtol: default_surrogate_gtol(),
            hyperparam_tol: GPMIN_DEFAULT_HYPERPARAM_TOL,
            hyperparam_maxiter: GPMIN_DEFAULT_HYPERPARAM_MAXITER,
        }
    }
}

impl GpMinConfig {
    /// Configuration matching ASE `GPMin(update_hyperparams=True)` defaults.
    #[must_use]
    pub fn with_hyperparam_updates() -> Self {
        Self {
            scale: GPMIN_UPDATED_SCALE,
            weight: GPMIN_UPDATED_WEIGHT,
            noise: GPMIN_UPDATED_NOISE,
            prior: PriorStrategy::Maximum,
            update_hyperparams: true,
            batch_size: GPMIN_UPDATED_BATCH_SIZE,
            hyperparam_bounds: Some(GPMIN_UPDATED_BOUNDS),
            max_step: None,
            max_refinements: GPMIN_DEFAULT_MAX_REFINEMENTS,
            surrogate_maxiter: GPMIN_DEFAULT_SURROGATE_MAXITER,
            surrogate_memory: GPMIN_DEFAULT_SURROGATE_MEMORY,
            surrogate_gtol: default_surrogate_gtol(),
            hyperparam_tol: GPMIN_DEFAULT_HYPERPARAM_TOL,
            hyperparam_maxiter: GPMIN_DEFAULT_HYPERPARAM_MAXITER,
        }
    }

    /// Validate all configuration parameters, failing closed on any
    /// non-finite or out-of-domain value.
    pub fn validate(&self) -> Result<()> {
        if !(self.scale.is_finite() && self.scale > 0.0) {
            return Err(Error::GpMin {
                what: "scale must be finite and positive",
            });
        }
        if !(self.weight.is_finite() && self.weight > 0.0) {
            return Err(Error::GpMin {
                what: "weight must be finite and positive",
            });
        }
        if !(self.noise.is_finite() && self.noise > 0.0) {
            return Err(Error::GpMin {
                what: "noise must be finite and positive",
            });
        }
        if let PriorStrategy::Fixed(c) = self.prior
            && !c.is_finite()
        {
            return Err(Error::GpMin {
                what: "fixed prior constant must be finite",
            });
        }
        if self.update_hyperparams && self.batch_size == 0 {
            return Err(Error::GpMin {
                what: "batch_size must be at least 1 when update_hyperparams is true",
            });
        }
        if let Some(bound) = self.hyperparam_bounds
            && !(bound.is_finite() && bound > 0.0 && bound < 1.0)
        {
            return Err(Error::GpMin {
                what: "hyperparam_bounds must be finite and in (0, 1)",
            });
        }
        if let Some(cap) = self.max_step
            && !(cap.is_finite() && cap > 0.0)
        {
            return Err(Error::GpMin {
                what: "max_step must be finite and positive",
            });
        }
        if self.max_refinements == 0 {
            return Err(Error::GpMin {
                what: "max_refinements must be at least 1",
            });
        }
        if self.surrogate_maxiter == 0 {
            return Err(Error::GpMin {
                what: "surrogate_maxiter must be at least 1",
            });
        }
        if self.surrogate_memory == 0 {
            return Err(Error::GpMin {
                what: "surrogate_memory must be at least 1",
            });
        }
        if !(self.surrogate_gtol.is_finite() && self.surrogate_gtol > 0.0) {
            return Err(Error::GpMin {
                what: "surrogate_gtol must be finite and positive",
            });
        }
        if !(self.hyperparam_tol.is_finite() && self.hyperparam_tol > 0.0) {
            return Err(Error::GpMin {
                what: "hyperparam_tol must be finite and positive",
            });
        }
        if self.hyperparam_maxiter == 0 {
            return Err(Error::GpMin {
                what: "hyperparam_maxiter must be at least 1",
            });
        }
        Ok(())
    }
}

/// Squared-exponential covariance kernel with analytic derivative blocks.
///
/// For points `x1, x2` in `R^D` with `d = x1 - x2` and
/// `r^2 = ||d||^2 / l^2`, the scalar energy covariance is
/// `k(x1, x2) = w^2 * exp(-0.5 * r^2)` and the `(D + 1) x (D + 1)` joint
/// covariance block for `[E, grad E]` is:
///
/// - `K[0, 0] = k`
/// - `K[0, b] = k * d_b / l^2`
/// - `K[a, 0] = -k * d_a / l^2`
/// - `K[a, b] = (k / l^2) * (delta_{ab} - d_a * d_b / l^2)`
///
/// Non-exhaustive: start from [`Default`] and chain [`Self::with_weight`]
/// and [`Self::with_scale`].
///
/// [`Default`] is the ASE `ase.optimize.GPMin` kernel for
/// `update_hyperparams = false`: weight [`GPMIN_DEFAULT_WEIGHT`] (`1.0`)
/// and scale [`GPMIN_DEFAULT_SCALE`] (`0.4`). ASE's `SquaredExponential`
/// stores no numeric default of its own; `GPMin.__init__` assigns those
/// two values in that mode (Garijo del Río, Mortensen, and Jacobsen,
/// Phys. Rev. B 100, 104103, 2019, as shipped by ASE). The
/// updated-hyperparameter preset is weight [`GPMIN_UPDATED_WEIGHT`]
/// (`2.0`) and scale [`GPMIN_UPDATED_SCALE`] (`0.3`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SquaredExponentialKernel {
    /// Amplitude prefactor `w > 0`.
    pub weight: f64,
    /// Length scale `l > 0`.
    pub scale: f64,
}

with_setters!(SquaredExponentialKernel {
    with_weight => weight: val f64,
    with_scale => scale: val f64,
});

impl Default for SquaredExponentialKernel {
    /// ASE fixed-hyperparameter kernel: weight `1.0`, scale `0.4`.
    ///
    /// Those are [`GPMIN_DEFAULT_WEIGHT`] and [`GPMIN_DEFAULT_SCALE`].
    /// `ase.optimize.gpmin.kernel.SquaredExponential` has no numeric
    /// default; `ase.optimize.GPMin.__init__` assigns `weight = 1.0` and
    /// `scale = 0.4` when `update_hyperparams` is false. The
    /// updated-hyperparameter preset (`update_hyperparams` true) is
    /// weight [`GPMIN_UPDATED_WEIGHT`] (`2.0`) and scale
    /// [`GPMIN_UPDATED_SCALE`] (`0.3`).
    fn default() -> Self {
        Self {
            weight: GPMIN_DEFAULT_WEIGHT,
            scale: GPMIN_DEFAULT_SCALE,
        }
    }
}

impl SquaredExponentialKernel {
    /// Construct a validated squared-exponential kernel.
    ///
    /// Same values as [`Default`] followed by [`Self::with_weight`] and
    /// [`Self::with_scale`], then the positivity check.
    pub fn new(weight: f64, scale: f64) -> Result<Self> {
        Self::default()
            .with_weight(weight)
            .with_scale(scale)
            .validated()
    }

    fn validated(self) -> Result<Self> {
        if !(self.weight.is_finite() && self.weight > 0.0) {
            return Err(Error::GpMin {
                what: "kernel weight must be finite and positive",
            });
        }
        if !(self.scale.is_finite() && self.scale > 0.0) {
            return Err(Error::GpMin {
                what: "kernel scale must be finite and positive",
            });
        }
        Ok(self)
    }

    /// Scalar energy-energy kernel value `w^2 * exp(-||x1 - x2||^2 / (2 l^2))`.
    #[must_use]
    pub fn value(&self, x1: ArrayView1<'_, f64>, x2: ArrayView1<'_, f64>) -> f64 {
        let inv_l2 = 1.0 / (self.scale * self.scale);
        let mut r2 = 0.0;
        for (a, b) in x1.iter().zip(x2.iter()) {
            let d = a - b;
            r2 += d * d;
        }
        r2 *= inv_l2;
        self.weight * self.weight * portable_exp(-0.5 * r2)
    }

    /// Full `(D + 1) x (D + 1)` covariance block between `[E(x1), g(x1)]`
    /// and `[E(x2), g(x2)]`.
    #[must_use]
    pub fn block(&self, x1: ArrayView1<'_, f64>, x2: ArrayView1<'_, f64>) -> Array2<f64> {
        let dim = x1.len();
        let d1 = dim + 1;
        let inv_l2 = 1.0 / (self.scale * self.scale);
        let mut r2 = 0.0;
        for i in 0..dim {
            let d = x1[i] - x2[i];
            r2 += d * d;
        }
        r2 *= inv_l2;
        let k = self.weight * self.weight * portable_exp(-0.5 * r2);

        let mut out = Array2::zeros((d1, d1));
        out[(0, 0)] = k;
        for i in 0..dim {
            let di = (x1[i] - x2[i]) * inv_l2;
            out[(0, i + 1)] = k * di;
            out[(i + 1, 0)] = -k * di;
            for j in 0..dim {
                let dj = (x1[j] - x2[j]) * inv_l2;
                let delta = if i == j { inv_l2 } else { 0.0 };
                out[(i + 1, j + 1)] = k * (delta - di * dj);
            }
        }
        out
    }

    /// Analytic derivative `dK(x1, x2) / dw` of the `(D + 1) x (D + 1)`
    /// block with respect to `weight`.
    #[must_use]
    pub fn d_weight_block(&self, x1: ArrayView1<'_, f64>, x2: ArrayView1<'_, f64>) -> Array2<f64> {
        let factor = 2.0 / self.weight;
        let mut blk = self.block(x1, x2);
        blk.mapv_inplace(|v| v * factor);
        blk
    }

    /// Analytic derivative `dK(x1, x2) / dl` of the `(D + 1) x (D + 1)`
    /// block with respect to `scale`.
    #[must_use]
    pub fn d_scale_block(&self, x1: ArrayView1<'_, f64>, x2: ArrayView1<'_, f64>) -> Array2<f64> {
        let dim = x1.len();
        let d1 = dim + 1;
        let inv_l = 1.0 / self.scale;
        let inv_l2 = inv_l * inv_l;
        let inv_l3 = inv_l2 * inv_l;
        let mut r2 = 0.0;
        for i in 0..dim {
            let d = x1[i] - x2[i];
            r2 += d * d;
        }
        r2 *= inv_l2;
        let k = self.weight * self.weight * portable_exp(-0.5 * r2);
        let one_minus_half_r2 = 1.0 - 0.5 * r2;
        let j_prefactor = -2.0 * one_minus_half_r2 * inv_l;

        let mut out = Array2::zeros((d1, d1));
        out[(0, 0)] = k * r2 * inv_l;
        for i in 0..dim {
            let di = x1[i] - x2[i];
            let grad_i = di * inv_l2;
            let dj_i = k * grad_i * j_prefactor;
            out[(0, i + 1)] = dj_i;
            out[(i + 1, 0)] = -dj_i;
            for j in 0..dim {
                let dj = x1[j] - x2[j];
                let p_ij = (di * dj) * inv_l2;
                let eye_minus_p = if i == j { 1.0 - p_ij } else { -p_ij };
                out[(i + 1, j + 1)] =
                    k * (-2.0 * inv_l3) * (one_minus_half_r2 * eye_minus_p - p_ij);
            }
        }
        out
    }

    /// Assemble the symmetric regularized covariance matrix
    /// `C = K(X, X) + diag((noise * scale)^2, noise^2, ...)` in flat
    /// row-major storage of order `n * (D + 1)`.
    fn assemble_regularized_flat(&self, xs: &[Array1<f64>], dim: usize, noise: f64) -> Vec<f64> {
        let n = xs.len();
        let d1 = dim + 1;
        let m = n * d1;
        let mut mat = vec![0.0; m * m];
        let inv_l2 = 1.0 / (self.scale * self.scale);
        let w2 = self.weight * self.weight;
        let reg_e = (noise * self.scale) * (noise * self.scale);
        let reg_g = noise * noise;

        for i in 0..n {
            let xi = xs[i].view();
            let row_i = i * d1;
            // Diagonal block (x_i == x_i): r^2 = 0, d = 0.
            mat[row_i * m + row_i] = w2 + reg_e;
            let diag_g = w2 * inv_l2 + reg_g;
            for a in 0..dim {
                let idx = row_i + 1 + a;
                mat[idx * m + idx] = diag_g;
            }
            for j in (i + 1)..n {
                let xj = xs[j].view();
                let col_j = j * d1;
                let mut r2 = 0.0;
                for a in 0..dim {
                    let d = xi[a] - xj[a];
                    r2 += d * d;
                }
                r2 *= inv_l2;
                let k = w2 * portable_exp(-0.5 * r2);

                mat[row_i * m + col_j] = k;
                mat[col_j * m + row_i] = k;
                for a in 0..dim {
                    let da = (xi[a] - xj[a]) * inv_l2;
                    let val_0a = k * da;
                    // K_ij[0, a+1] = +k * da; K_ji[a+1, 0] = +k * da
                    mat[row_i * m + (col_j + 1 + a)] = val_0a;
                    mat[(col_j + 1 + a) * m + row_i] = val_0a;
                    // K_ij[a+1, 0] = -k * da; K_ji[0, a+1] = -k * da
                    mat[(row_i + 1 + a) * m + col_j] = -val_0a;
                    mat[col_j * m + (row_i + 1 + a)] = -val_0a;

                    for b in 0..dim {
                        let db = (xi[b] - xj[b]) * inv_l2;
                        let delta = if a == b { inv_l2 } else { 0.0 };
                        let val_ab = k * (delta - da * db);
                        mat[(row_i + 1 + a) * m + (col_j + 1 + b)] = val_ab;
                        mat[(col_j + 1 + b) * m + (row_i + 1 + a)] = val_ab;
                    }
                }
            }
        }
        mat
    }

    /// Assemble the symmetric derivative matrix `dC / dw` in flat row-major
    /// storage of order `n * (D + 1)`.
    fn assemble_d_weight_flat(&self, xs: &[Array1<f64>], dim: usize) -> Vec<f64> {
        let n = xs.len();
        let d1 = dim + 1;
        let m = n * d1;
        let mut mat = vec![0.0; m * m];
        let inv_l2 = 1.0 / (self.scale * self.scale);
        let two_w = 2.0 * self.weight;

        for i in 0..n {
            let xi = xs[i].view();
            let row_i = i * d1;
            mat[row_i * m + row_i] = two_w;
            let diag_g = two_w * inv_l2;
            for a in 0..dim {
                let idx = row_i + 1 + a;
                mat[idx * m + idx] = diag_g;
            }
            for j in (i + 1)..n {
                let xj = xs[j].view();
                let col_j = j * d1;
                let mut r2 = 0.0;
                for a in 0..dim {
                    let d = xi[a] - xj[a];
                    r2 += d * d;
                }
                r2 *= inv_l2;
                let dk = two_w * portable_exp(-0.5 * r2);

                mat[row_i * m + col_j] = dk;
                mat[col_j * m + row_i] = dk;
                for a in 0..dim {
                    let da = (xi[a] - xj[a]) * inv_l2;
                    let val_0a = dk * da;
                    mat[row_i * m + (col_j + 1 + a)] = val_0a;
                    mat[(col_j + 1 + a) * m + row_i] = val_0a;
                    mat[(row_i + 1 + a) * m + col_j] = -val_0a;
                    mat[col_j * m + (row_i + 1 + a)] = -val_0a;
                    for b in 0..dim {
                        let db = (xi[b] - xj[b]) * inv_l2;
                        let delta = if a == b { inv_l2 } else { 0.0 };
                        let val_ab = dk * (delta - da * db);
                        mat[(row_i + 1 + a) * m + (col_j + 1 + b)] = val_ab;
                        mat[(col_j + 1 + b) * m + (row_i + 1 + a)] = val_ab;
                    }
                }
            }
        }
        mat
    }

    /// Assemble the symmetric derivative matrix `dC / dl` in flat row-major
    /// storage of order `n * (D + 1)`, including the `2 * noise^2 * l`
    /// derivative of the energy regularization diagonal.
    fn assemble_d_scale_flat(&self, xs: &[Array1<f64>], dim: usize, noise: f64) -> Vec<f64> {
        let n = xs.len();
        let d1 = dim + 1;
        let m = n * d1;
        let mut mat = vec![0.0; m * m];
        let inv_l = 1.0 / self.scale;
        let inv_l2 = inv_l * inv_l;
        let inv_l3 = inv_l2 * inv_l;
        let w2 = self.weight * self.weight;
        let d_reg_e = 2.0 * noise * noise * self.scale;

        for i in 0..n {
            let xi = xs[i].view();
            let row_i = i * d1;
            // Diagonal block (d = 0, r^2 = 0): dK[0,0]/dl = 0, dK[a,a]/dl = -2 w^2 / l^3.
            mat[row_i * m + row_i] = d_reg_e;
            let diag_g = -2.0 * w2 * inv_l3;
            for a in 0..dim {
                let idx = row_i + 1 + a;
                mat[idx * m + idx] = diag_g;
            }
            for j in (i + 1)..n {
                let xj = xs[j].view();
                let col_j = j * d1;
                let mut r2 = 0.0;
                for a in 0..dim {
                    let d = xi[a] - xj[a];
                    r2 += d * d;
                }
                r2 *= inv_l2;
                let k = w2 * portable_exp(-0.5 * r2);
                let one_minus_half_r2 = 1.0 - 0.5 * r2;
                let j_prefactor = -2.0 * one_minus_half_r2 * inv_l;

                let val_00 = k * r2 * inv_l;
                mat[row_i * m + col_j] = val_00;
                mat[col_j * m + row_i] = val_00;
                for a in 0..dim {
                    let da = xi[a] - xj[a];
                    let val_0a = k * (da * inv_l2) * j_prefactor;
                    mat[row_i * m + (col_j + 1 + a)] = val_0a;
                    mat[(col_j + 1 + a) * m + row_i] = val_0a;
                    mat[(row_i + 1 + a) * m + col_j] = -val_0a;
                    mat[col_j * m + (row_i + 1 + a)] = -val_0a;
                    for b in 0..dim {
                        let db = xi[b] - xj[b];
                        let p_ab = (da * db) * inv_l2;
                        let eye_minus_p = if a == b { 1.0 - p_ab } else { -p_ab };
                        let val_ab = k * (-2.0 * inv_l3) * (one_minus_half_r2 * eye_minus_p - p_ab);
                        mat[(row_i + 1 + a) * m + (col_j + 1 + b)] = val_ab;
                        mat[(col_j + 1 + b) * m + (row_i + 1 + a)] = val_ab;
                    }
                }
            }
        }
        mat
    }
}

/// Lower-triangular Cholesky factorization `A = L L^T` of a symmetric
/// positive-definite matrix of order `m` in row-major layout.
///
/// Fails closed with [`Error::GpMin`] if any pivot is non-positive or
/// non-finite; never adds silent diagonal jitter.
pub(crate) fn cholesky_lower(a: &[f64], m: usize) -> Result<Vec<f64>> {
    if a.len() != m * m {
        return Err(Error::GpMin {
            what: "Cholesky input size mismatch",
        });
    }
    let mut l = vec![0.0; m * m];
    for i in 0..m {
        let row_i = i * m;
        for j in 0..i {
            let row_j = j * m;
            let mut sum = a[row_i + j];
            for k in 0..j {
                sum -= l[row_i + k] * l[row_j + k];
            }
            let val = sum / l[row_j + j];
            if !val.is_finite() {
                return Err(Error::GpMin {
                    what: "kernel matrix is not positive definite after configured noise",
                });
            }
            l[row_i + j] = val;
        }
        let mut diag = a[row_i + i];
        for k in 0..i {
            let v = l[row_i + k];
            diag -= v * v;
        }
        if !(diag.is_finite() && diag > 0.0) {
            return Err(Error::GpMin {
                what: "kernel matrix is not positive definite after configured noise",
            });
        }
        l[row_i + i] = diag.sqrt();
    }
    Ok(l)
}

/// Solve `L L^T x = b` in place using the lower-triangular Cholesky factor
/// `L` of order `m`.
pub(crate) fn cholesky_solve_in_place(l: &[f64], m: usize, b: &mut [f64]) -> Result<()> {
    if l.len() != m * m || b.len() != m {
        return Err(Error::GpMin {
            what: "Cholesky solve dimension mismatch",
        });
    }
    // Forward substitution: L z = b
    for i in 0..m {
        let row_i = i * m;
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[row_i + k] * b[k];
        }
        let val = sum / l[row_i + i];
        if !val.is_finite() {
            return Err(Error::GpMin {
                what: "non-finite result during Cholesky forward substitution",
            });
        }
        b[i] = val;
    }
    // Backward substitution: L^T x = z
    for i in (0..m).rev() {
        let mut sum = b[i];
        for k in (i + 1)..m {
            sum -= l[k * m + i] * b[k];
        }
        let val = sum / l[i * m + i];
        if !val.is_finite() {
            return Err(Error::GpMin {
                what: "non-finite result during Cholesky backward substitution",
            });
        }
        b[i] = val;
    }
    Ok(())
}

/// Compute `C^{-1} = L^{-T} L^{-1}` from the lower-triangular Cholesky
/// factor `L` of order `m`.
fn cholesky_inverse(l: &[f64], m: usize) -> Result<Vec<f64>> {
    // Invert lower-triangular L into linv (also lower-triangular).
    let mut linv = vec![0.0; m * m];
    for j in 0..m {
        let diag = 1.0 / l[j * m + j];
        if !diag.is_finite() {
            return Err(Error::GpMin {
                what: "non-finite diagonal during Cholesky inversion",
            });
        }
        linv[j * m + j] = diag;
        for i in (j + 1)..m {
            let row_i = i * m;
            let mut sum = 0.0;
            for k in j..i {
                sum += l[row_i + k] * linv[k * m + j];
            }
            let val = -sum / l[row_i + i];
            if !val.is_finite() {
                return Err(Error::GpMin {
                    what: "non-finite entry during Cholesky inversion",
                });
            }
            linv[row_i + j] = val;
        }
    }
    // C^{-1}[i, j] = sum_{k >= max(i, j)} linv[k, i] * linv[k, j]
    let mut inv = vec![0.0; m * m];
    for i in 0..m {
        for j in i..m {
            let mut sum = 0.0;
            for k in j..m {
                sum += linv[k * m + i] * linv[k * m + j];
            }
            inv[i * m + j] = sum;
            inv[j * m + i] = sum;
        }
    }
    Ok(inv)
}

/// Gaussian process surrogate over `(x, E(x), grad E(x))` observations.
#[derive(Clone, Debug)]
pub struct GaussianProcess {
    dim: usize,
    xs: Vec<Array1<f64>>,
    energies: Vec<f64>,
    gradients: Vec<Array1<f64>>,
    prior_strategy: PriorStrategy,
    prior_value: f64,
    weight: f64,
    scale: f64,
    noise: f64,
    chol_l: Vec<f64>,
    alpha: Vec<f64>,
    trained: bool,
}

impl GaussianProcess {
    /// Create an empty GP surrogate of dimension `dim`.
    pub fn new(dim: usize, config: &GpMinConfig) -> Result<Self> {
        if dim == 0 {
            return Err(Error::GpMin {
                what: "dimension must be positive",
            });
        }
        config.validate()?;
        let initial_prior = match config.prior {
            PriorStrategy::Fixed(c) => c,
            PriorStrategy::Maximum | PriorStrategy::Init | PriorStrategy::Average => 0.0,
        };
        Ok(Self {
            dim,
            xs: Vec::new(),
            energies: Vec::new(),
            gradients: Vec::new(),
            prior_strategy: config.prior,
            prior_value: initial_prior,
            weight: config.weight,
            scale: config.scale,
            noise: config.noise,
            chol_l: Vec::new(),
            alpha: Vec::new(),
            trained: false,
        })
    }

    /// Coordinate dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of stored observations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.xs.len()
    }

    /// Whether no observations have been added yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.xs.is_empty()
    }

    /// Current constant prior mean `c`.
    #[must_use]
    pub fn prior_mean(&self) -> f64 {
        self.prior_value
    }

    /// Current `(weight, scale, noise)` hyperparameters.
    #[must_use]
    pub fn hyperparameters(&self) -> (f64, f64, f64) {
        (self.weight, self.scale, self.noise)
    }

    /// Clear all observations and reset hyperparameters to `config`.
    pub fn reset(&mut self, config: &GpMinConfig) -> Result<()> {
        config.validate()?;
        self.xs.clear();
        self.energies.clear();
        self.gradients.clear();
        self.prior_strategy = config.prior;
        self.prior_value = match config.prior {
            PriorStrategy::Fixed(c) => c,
            PriorStrategy::Maximum | PriorStrategy::Init | PriorStrategy::Average => 0.0,
        };
        self.weight = config.weight;
        self.scale = config.scale;
        self.noise = config.noise;
        self.chol_l.clear();
        self.alpha.clear();
        self.trained = false;
        Ok(())
    }

    /// Append one `(x, energy, gradient)` observation and update the prior
    /// mean according to the configured [`PriorStrategy`].
    pub fn push_observation(
        &mut self,
        x: Array1<f64>,
        energy: f64,
        gradient: Array1<f64>,
    ) -> Result<()> {
        if x.len() != self.dim || gradient.len() != self.dim {
            return Err(Error::Dim {
                got: x.len(),
                dim: self.dim,
            });
        }
        if !energy.is_finite()
            || x.iter().any(|v| !v.is_finite())
            || gradient.iter().any(|v| !v.is_finite())
        {
            return Err(Error::Oracle {
                what: "non-finite value or gradient",
            });
        }
        self.xs.push(x);
        self.energies.push(energy);
        self.gradients.push(gradient);
        self.update_prior_mean();
        self.trained = false;
        Ok(())
    }

    fn update_prior_mean(&mut self) {
        if self.energies.is_empty() {
            return;
        }
        self.prior_value = match self.prior_strategy {
            PriorStrategy::Fixed(c) => c,
            PriorStrategy::Init => self.energies[0],
            PriorStrategy::Maximum => {
                let mut best = self.energies[0];
                for &e in &self.energies[1..] {
                    if e > best {
                        best = e;
                    }
                }
                best
            }
            PriorStrategy::Average => {
                let sum: f64 = self.energies.iter().copied().sum();
                sum / (self.energies.len() as f64)
            }
        };
    }

    fn build_residual(&self) -> Vec<f64> {
        let d1 = self.dim + 1;
        let mut r = Vec::with_capacity(self.xs.len() * d1);
        for (e, g) in self.energies.iter().zip(self.gradients.iter()) {
            r.push(*e - self.prior_value);
            r.extend(g.iter().copied());
        }
        r
    }

    /// Assemble the covariance matrix, factorize it with Cholesky, and solve
    /// for the dual weights `alpha = C^{-1} (Y - m(X))`.
    pub fn train(&mut self) -> Result<()> {
        if self.xs.is_empty() {
            return Err(Error::GpMin {
                what: "cannot train GP with zero observations",
            });
        }
        let kernel = SquaredExponentialKernel::default()
            .with_weight(self.weight)
            .with_scale(self.scale)
            .validated()?;
        let m = self.xs.len() * (self.dim + 1);
        let cov = kernel.assemble_regularized_flat(&self.xs, self.dim, self.noise);
        let l = cholesky_lower(&cov, m)?;
        let mut alpha = self.build_residual();
        cholesky_solve_in_place(&l, m, &mut alpha)?;
        self.chol_l = l;
        self.alpha = alpha;
        self.trained = true;
        Ok(())
    }

    /// Predict the GP posterior mean energy and gradient `(E_hat(x), grad E_hat(x))`.
    ///
    /// Runs in `O(n * D)` without forming `(D + 1) x (D + 1)` blocks.
    pub fn predict(&self, x: ArrayView1<'_, f64>) -> Result<(f64, Array1<f64>)> {
        let (energy, grad, _) = self.predict_with_roundoff(x)?;
        Ok((energy, grad))
    }

    /// Predict `(E_hat(x), grad E_hat(x), roundoff_bound)` using Neumaier
    /// compensated summation for the energy sum and returning the forward
    /// floating-point error bound `8 * eps * (|prior| + sum_j |k_j (a_0j + q_j)|)`.
    fn predict_with_roundoff(&self, x: ArrayView1<'_, f64>) -> Result<(f64, Array1<f64>, f64)> {
        if !self.trained {
            return Err(Error::GpMin {
                what: "GP must be trained before predicting",
            });
        }
        if x.len() != self.dim {
            return Err(Error::Dim {
                got: x.len(),
                dim: self.dim,
            });
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(Error::GpMin {
                what: "non-finite query point",
            });
        }
        let inv_l2 = 1.0 / (self.scale * self.scale);
        let w2 = self.weight * self.weight;
        let d1 = self.dim + 1;

        let mut energy = self.prior_value;
        let mut comp = 0.0;
        let mut abs_sum = self.prior_value.abs();
        let mut grad: Array1<f64> = Array1::zeros(self.dim);

        for (j, xj) in self.xs.iter().enumerate() {
            let base = j * d1;
            let alpha_e = self.alpha[base];
            let alpha_g = &self.alpha[(base + 1)..(base + d1)];

            let mut r2 = 0.0;
            let mut q = 0.0;
            for a in 0..self.dim {
                let d = x[a] - xj[a];
                let u = d * inv_l2;
                r2 += d * u;
                q += u * alpha_g[a];
            }
            let k = w2 * portable_exp(-0.5 * r2);
            let combined = alpha_e + q;
            let term = k * combined;
            // Neumaier compensated summation for energy.
            let t = energy + term;
            if energy.abs() >= term.abs() {
                comp += (energy - t) + term;
            } else {
                comp += (term - t) + energy;
            }
            energy = t;
            abs_sum += term.abs();

            let k_inv_l2 = k * inv_l2;
            let k_comb_inv_l2 = k * combined * inv_l2;
            for a in 0..self.dim {
                let d = x[a] - xj[a];
                grad[a] += k_inv_l2 * alpha_g[a] - k_comb_inv_l2 * d;
            }
        }
        energy += comp;
        if !energy.is_finite() || grad.iter().any(|v| !v.is_finite()) {
            return Err(Error::GpMin {
                what: "non-finite GP posterior prediction",
            });
        }
        let roundoff = 8.0 * f64::EPSILON * (1.0 + abs_sum);
        Ok((energy, grad, roundoff))
    }

    /// Evaluate the negative log marginal likelihood and its exact analytic
    /// gradient with respect to `[weight, scale]` at the given
    /// `(weight, scale, noise)`.
    pub fn neg_log_marginal_likelihood(
        &self,
        weight: f64,
        scale: f64,
        noise: f64,
    ) -> Result<(f64, [f64; 2])> {
        if self.xs.is_empty() {
            return Err(Error::GpMin {
                what: "cannot evaluate marginal likelihood with zero observations",
            });
        }
        let kernel = SquaredExponentialKernel::default()
            .with_weight(weight)
            .with_scale(scale)
            .validated()?;
        if !(noise.is_finite() && noise > 0.0) {
            return Err(Error::GpMin {
                what: "noise must be finite and positive",
            });
        }
        let m = self.xs.len() * (self.dim + 1);
        let cov = kernel.assemble_regularized_flat(&self.xs, self.dim, noise);
        let l = cholesky_lower(&cov, m)?;
        let residual = self.build_residual();
        let mut alpha = residual.clone();
        cholesky_solve_in_place(&l, m, &mut alpha)?;

        let mut quad = 0.0;
        for i in 0..m {
            quad += residual[i] * alpha[i];
        }
        let mut log_det_half = 0.0;
        for i in 0..m {
            log_det_half += l[i * m + i].ln();
        }
        let log_p =
            -0.5 * quad - log_det_half - 0.5 * (m as f64) * (2.0 * std::f64::consts::PI).ln();

        let cinv = cholesky_inverse(&l, m)?;
        let dc_dw = kernel.assemble_d_weight_flat(&self.xs, self.dim);
        let dc_dl = kernel.assemble_d_scale_flat(&self.xs, self.dim, noise);

        let dlog_dw = marginal_grad_component(&alpha, &cinv, &dc_dw, m);
        let dlog_dl = marginal_grad_component(&alpha, &cinv, &dc_dl, m);

        let nll = -log_p;
        let grad = [-dlog_dw, -dlog_dl];
        if !nll.is_finite() || !grad[0].is_finite() || !grad[1].is_finite() {
            return Err(Error::GpMin {
                what: "non-finite marginal log-likelihood or gradient",
            });
        }
        Ok((nll, grad))
    }

    /// Fit `(weight, scale)` by minimizing the negative log marginal
    /// likelihood under optional relative box bounds, then rescale `noise`
    /// to keep `noise / weight` constant (matching ASE `GPMin.fit_to_batch`).
    pub fn fit_hyperparameters(&mut self, config: &GpMinConfig) -> Result<bool> {
        let ratio = self.noise / self.weight;
        let fixed_noise = self.noise;
        let w0 = self.weight;
        let l0 = self.scale;

        let pos_floor = f64::EPSILON.sqrt();
        let (lo, hi) = match config.hyperparam_bounds {
            Some(eps) => (
                [(1.0 - eps) * w0, (1.0 - eps) * l0],
                [(1.0 + eps) * w0, (1.0 + eps) * l0],
            ),
            None => (
                [pos_floor * w0, pos_floor * l0],
                [f64::INFINITY, f64::INFINITY],
            ),
        };

        let mut theta = [w0, l0];
        let (mut f_cur, mut g_cur) =
            self.neg_log_marginal_likelihood(theta[0], theta[1], fixed_noise)?;

        // 2x2 inverse Hessian approximation (starts at identity).
        let mut h = [[1.0, 0.0], [0.0, 1.0]];
        let ftol = 0.01 * config.hyperparam_tol;
        let c1 = 1.0 / 10_000.0;
        let mut converged = false;

        for _ in 0..config.hyperparam_maxiter {
            let pgnorm = projected_grad_inf_norm(theta, g_cur, lo, hi);
            if pgnorm <= config.hyperparam_tol {
                converged = true;
                break;
            }

            // Compute quasi-Newton direction on free variables; steepest descent
            // fallback if not descent along the active box.
            let mut dir = [
                -(h[0][0] * g_cur[0] + h[0][1] * g_cur[1]),
                -(h[1][0] * g_cur[0] + h[1][1] * g_cur[1]),
            ];
            for k in 0..2 {
                if (theta[k] <= lo[k] && dir[k] < 0.0) || (theta[k] >= hi[k] && dir[k] > 0.0) {
                    dir[k] = 0.0;
                }
            }
            let mut slope = g_cur[0] * dir[0] + g_cur[1] * dir[1];
            if !(slope.is_finite() && slope < 0.0) {
                h = [[1.0, 0.0], [0.0, 1.0]];
                dir = [-g_cur[0], -g_cur[1]];
                for k in 0..2 {
                    if (theta[k] <= lo[k] && dir[k] < 0.0) || (theta[k] >= hi[k] && dir[k] > 0.0) {
                        dir[k] = 0.0;
                    }
                }
                slope = g_cur[0] * dir[0] + g_cur[1] * dir[1];
                if !(slope.is_finite() && slope < 0.0) {
                    break;
                }
            }

            // Projected Armijo backtracking.
            let mut step = 1.0;
            let mut accepted: Option<([f64; 2], f64, [f64; 2])> = None;
            for _ in 0..30 {
                let cand = [
                    project_box(theta[0] + step * dir[0], lo[0], hi[0]),
                    project_box(theta[1] + step * dir[1], lo[1], hi[1]),
                ];
                let s = [cand[0] - theta[0], cand[1] - theta[1]];
                let gs = g_cur[0] * s[0] + g_cur[1] * s[1];
                if gs >= 0.0 {
                    step *= 0.5;
                    continue;
                }
                if let Ok((f_try, g_try)) =
                    self.neg_log_marginal_likelihood(cand[0], cand[1], fixed_noise)
                    && f_try <= f_cur + c1 * gs
                {
                    accepted = Some((cand, f_try, g_try));
                    break;
                }
                step *= 0.5;
            }

            let Some((next_theta, f_next, g_next)) = accepted else {
                break;
            };

            let s = [next_theta[0] - theta[0], next_theta[1] - theta[1]];
            let y = [g_next[0] - g_cur[0], g_next[1] - g_cur[1]];
            let df = (f_cur - f_next).abs();
            let scale_f = f_cur.abs().max(f_next.abs()).max(1.0);

            theta = next_theta;
            f_cur = f_next;
            g_cur = g_next;

            if df <= ftol * scale_f {
                converged = true;
                break;
            }

            // 2x2 BFGS inverse Hessian update when curvature is strictly positive.
            let sy = s[0] * y[0] + s[1] * y[1];
            let s_norm = (s[0] * s[0] + s[1] * s[1]).sqrt();
            let y_norm = (y[0] * y[0] + y[1] * y[1]).sqrt();
            if sy.is_finite() && sy > f64::EPSILON.sqrt() * s_norm * y_norm {
                let rho = 1.0 / sy;
                let hy = [
                    h[0][0] * y[0] + h[0][1] * y[1],
                    h[1][0] * y[0] + h[1][1] * y[1],
                ];
                let yhy = y[0] * hy[0] + y[1] * hy[1];
                let coeff = (1.0 + rho * yhy) * rho;
                for r in 0..2 {
                    for c in 0..2 {
                        h[r][c] += coeff * s[r] * s[c] - rho * (hy[r] * s[c] + s[r] * hy[c]);
                    }
                }
            }
        }

        if converged {
            self.weight = theta[0];
            self.scale = theta[1];
            self.noise = ratio * self.weight;
        }
        self.trained = false;
        Ok(converged)
    }
}

fn marginal_grad_component(alpha: &[f64], cinv: &[f64], d_cov: &[f64], m: usize) -> f64 {
    let mut quad = 0.0;
    let mut trace = 0.0;
    for i in 0..m {
        let row = i * m;
        let ai = alpha[i];
        for j in 0..m {
            let g_ij = d_cov[row + j];
            quad += ai * g_ij * alpha[j];
            trace += cinv[row + j] * g_ij;
        }
    }
    0.5 * (quad - trace)
}

fn project_box(val: f64, lo: f64, hi: f64) -> f64 {
    if val < lo {
        lo
    } else if val > hi {
        hi
    } else {
        val
    }
}

fn projected_grad_inf_norm(theta: [f64; 2], grad: [f64; 2], lo: [f64; 2], hi: [f64; 2]) -> f64 {
    let mut max_g = 0.0_f64;
    for k in 0..2 {
        let step_pt = project_box(theta[k] - grad[k], lo[k], hi[k]);
        let pg = (step_pt - theta[k]).abs();
        if pg > max_g {
            max_g = pg;
        }
    }
    max_g
}

fn inf_norm(v: &Array1<f64>) -> f64 {
    let mut best = 0.0_f64;
    for &x in v.iter() {
        let a = x.abs();
        if a > best {
            best = a;
        }
    }
    best
}

fn points_match(a: &Array1<f64>, b: &Array1<f64>) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let tol = 4.0 * f64::EPSILON;
    let mut diff2 = 0.0;
    let mut ref2 = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = x - y;
        diff2 += d * d;
        ref2 += x * x + y * y;
    }
    diff2.sqrt() <= tol * (1.0 + ref2.sqrt())
}

struct SurrogatePoint {
    x: Array1<f64>,
    f: f64,
    g: Array1<f64>,
    e_err: f64,
}

struct BestTrial {
    pt: SurrogatePoint,
    dphi: f64,
}

/// Strong Wolfe line search (Nocedal & Wright, Algorithms 3.5 & 3.6) with
/// Hager-Zhang approximate Wolfe roundoff awareness on the GP posterior mean.
fn surrogate_wolfe_search(
    gp: &GaussianProcess,
    x0: &Array1<f64>,
    f0: f64,
    g0: &Array1<f64>,
    e_err0: f64,
    dir: &Array1<f64>,
    alpha_init: f64,
) -> Result<Option<SurrogatePoint>> {
    let c1 = 1.0 / 10_000.0;
    let c2 = 0.9;
    let dphi0 = g0.dot(dir);
    if !(dphi0.is_finite() && dphi0 < 0.0) {
        return Ok(None);
    }
    let max_trials = 25usize;

    let mut alpha_prev = 0.0;
    let mut f_prev = f0;
    let mut dphi_prev = dphi0;
    let mut alpha = alpha_init;
    let mut best: Option<BestTrial> = None;

    for iter in 0..max_trials {
        let x_try = x0 + &(dir * alpha);
        let (f_try, g_try, e_err_try) = gp.predict_with_roundoff(x_try.view())?;
        let dphi_try = g_try.dot(dir);
        let roundoff = e_err0.max(e_err_try);

        if f_try <= f0 + roundoff && dphi_try > dphi0 {
            let replace = match &best {
                None => true,
                Some(b) => {
                    if f_try + roundoff < b.pt.f {
                        true
                    } else if (f_try - b.pt.f).abs() <= roundoff {
                        dphi_try.abs() < b.dphi.abs()
                    } else {
                        false
                    }
                }
            };
            if replace {
                best = Some(BestTrial {
                    pt: SurrogatePoint {
                        x: x_try.clone(),
                        f: f_try,
                        g: g_try.clone(),
                        e_err: e_err_try,
                    },
                    dphi: dphi_try,
                });
            }
        }

        if (f_try > f0 + c1 * alpha * dphi0 + roundoff) || (iter > 0 && f_try >= f_prev + roundoff)
        {
            let zoomed = surrogate_zoom(
                gp, x0, f0, e_err0, dphi0, dir, c1, c2, alpha_prev, f_prev, dphi_prev, alpha,
                f_try, dphi_try,
            )?;
            return Ok(match zoomed {
                Some(hit) => Some(hit),
                None => best.map(|b| b.pt),
            });
        }
        if dphi_try.abs() <= -c2 * dphi0 {
            return Ok(Some(SurrogatePoint {
                x: x_try,
                f: f_try,
                g: g_try,
                e_err: e_err_try,
            }));
        }
        if dphi_try >= 0.0 {
            let zoomed = surrogate_zoom(
                gp, x0, f0, e_err0, dphi0, dir, c1, c2, alpha, f_try, dphi_try, alpha_prev, f_prev,
                dphi_prev,
            )?;
            return Ok(match zoomed {
                Some(hit) => Some(hit),
                None => best.map(|b| b.pt),
            });
        }
        alpha_prev = alpha;
        f_prev = f_try;
        dphi_prev = dphi_try;
        alpha *= 2.0;
    }

    Ok(best.map(|b| b.pt))
}

#[allow(clippy::too_many_arguments)]
fn surrogate_zoom(
    gp: &GaussianProcess,
    x0: &Array1<f64>,
    f0: f64,
    e_err0: f64,
    dphi0: f64,
    dir: &Array1<f64>,
    c1: f64,
    c2: f64,
    mut a_lo: f64,
    mut f_lo: f64,
    mut dphi_lo: f64,
    mut a_hi: f64,
    mut f_hi: f64,
    mut dphi_hi: f64,
) -> Result<Option<SurrogatePoint>> {
    let min_span = 16.0 * f64::EPSILON;
    let mut best: Option<BestTrial> = None;

    for _ in 0..35 {
        if (a_hi - a_lo).abs() <= min_span * (1.0 + a_lo.abs() + a_hi.abs()) {
            break;
        }
        let a_j = cubic_or_bisect(a_lo, f_lo, dphi_lo, a_hi, f_hi, dphi_hi);
        let x_j = x0 + &(dir * a_j);
        let (f_j, g_j, e_err_j) = gp.predict_with_roundoff(x_j.view())?;
        let dphi_j = g_j.dot(dir);
        let roundoff = e_err0.max(e_err_j);

        if f_j <= f0 + roundoff && dphi_j > dphi0 {
            let replace = match &best {
                None => true,
                Some(b) => {
                    if f_j + roundoff < b.pt.f {
                        true
                    } else if (f_j - b.pt.f).abs() <= roundoff {
                        dphi_j.abs() < b.dphi.abs()
                    } else {
                        false
                    }
                }
            };
            if replace {
                best = Some(BestTrial {
                    pt: SurrogatePoint {
                        x: x_j.clone(),
                        f: f_j,
                        g: g_j.clone(),
                        e_err: e_err_j,
                    },
                    dphi: dphi_j,
                });
            }
        }

        let armijo_fail = f_j > f0 + c1 * a_j * dphi0 + roundoff;
        // Outside the energy roundoff band, enforce strict decrease below f_lo;
        // inside the roundoff band (where catastrophic cancellation masks tiny
        // real drops), fall back to the smooth analytic directional derivative.
        let worse_than_lo = if (f_j - f_lo).abs() <= roundoff {
            false
        } else {
            f_j >= f_lo
        };

        if armijo_fail || worse_than_lo {
            a_hi = a_j;
            f_hi = f_j;
            dphi_hi = dphi_j;
        } else {
            if dphi_j.abs() <= -c2 * dphi0 {
                return Ok(Some(SurrogatePoint {
                    x: x_j,
                    f: f_j,
                    g: g_j,
                    e_err: e_err_j,
                }));
            }
            if dphi_j * (a_hi - a_lo) >= 0.0 {
                a_hi = a_lo;
                f_hi = f_lo;
                dphi_hi = dphi_lo;
            }
            a_lo = a_j;
            f_lo = f_j;
            dphi_lo = dphi_j;
        }
    }

    Ok(best.map(|b| b.pt))
}

fn cubic_or_bisect(a_lo: f64, f_lo: f64, d_lo: f64, a_hi: f64, f_hi: f64, d_hi: f64) -> f64 {
    let mid = 0.5 * (a_lo + a_hi);
    let span = a_hi - a_lo;
    if !span.is_finite() || span.abs() <= f64::EPSILON {
        return mid;
    }
    let d1 = d_lo + d_hi - 3.0 * (f_lo - f_hi) / (a_lo - a_hi);
    let disc = d1 * d1 - d_lo * d_hi;
    if !(disc.is_finite() && disc >= 0.0) {
        return mid;
    }
    let d2 = disc.sqrt() * span.signum();
    let denom = d_hi - d_lo + 2.0 * d2;
    if !(denom.is_finite() && denom.abs() > f64::EPSILON) {
        return mid;
    }
    let cand = a_hi - span * ((d_hi + d2 - d1) / denom);
    let lo = a_lo.min(a_hi);
    let hi = a_lo.max(a_hi);
    let margin = 0.1 * (hi - lo);
    if cand.is_finite() && cand >= lo + margin && cand <= hi - margin {
        cand
    } else {
        mid
    }
}

/// Minimize the GP posterior mean starting from `r0` using inner L-BFGS
/// with strong Wolfe line search, then apply optional step caps while
/// preserving surrogate descent.
fn relax_surrogate(
    gp: &GaussianProcess,
    r0: &Array1<f64>,
    config: &GpMinConfig,
    atom_cap: Option<f64>,
    euclid_cap: Option<f64>,
) -> Result<Array1<f64>> {
    let dim = r0.len();
    let mut x = r0.clone();
    let (f_start, mut g, mut e_err) = gp.predict_with_roundoff(x.view())?;
    let mut f = f_start;

    let mut pairs = LbfgsPairs::new(dim, config.surrogate_memory)?;
    let mut last_gamma = 1.0;
    let curv_floor = f64::EPSILON.sqrt();
    let (_, scale, _) = gp.hyperparameters();

    let mut last_inside = r0.clone();
    let mut f_last_inside = f_start;

    for _ in 0..config.surrogate_maxiter {
        if inf_norm(&g) <= config.surrogate_gtol {
            break;
        }

        let dir = if pairs.is_empty() {
            -&g
        } else {
            let candidate = pairs.direction(g.view(), LbfgsH0::Scale(last_gamma))?;
            let slope = g.dot(&candidate);
            if slope.is_finite() && slope < 0.0 {
                candidate
            } else {
                pairs.clear();
                last_gamma = 1.0;
                -&g
            }
        };

        let d_norm = l2(&dir);
        if !(d_norm.is_finite() && d_norm > 0.0) {
            break;
        }

        // Choose initial Wolfe trial length:
        // - For a 1-point GP with constant prior, the exact surrogate minimum
        //   lies at distance `scale` along `-g / ||g||`.
        // - Otherwise start at 1.0 (quasi-Newton unit step), capped to at most
        //   4 * scale (or 1 * scale before pairs exist) so the opening trial
        //   stays inside the kernel's support and does not underflow `exp`.
        let alpha_init = if pairs.is_empty() {
            if gp.len() == 1 || d_norm > scale {
                scale / d_norm
            } else {
                1.0
            }
        } else if d_norm > 4.0 * scale {
            (4.0 * scale) / d_norm
        } else {
            1.0
        };

        let found = surrogate_wolfe_search(gp, &x, f, &g, e_err, &dir, alpha_init)?;
        let Some(SurrogatePoint {
            x: x_next,
            f: f_next,
            g: g_next,
            e_err: e_err_next,
        }) = found
        else {
            // If L-BFGS direction failed to find descent, retry once along -g.
            if !pairs.is_empty() {
                pairs.clear();
                last_gamma = 1.0;
                continue;
            }
            break;
        };

        let s = &x_next - &x;
        let y = &g_next - &g;
        let sy = s.dot(&y);
        let yy = y.dot(&y);
        let s_norm = l2(&s);
        let y_norm = l2(&y);
        if sy.is_finite() && yy.is_finite() && yy > 0.0 && sy > curv_floor * s_norm * y_norm {
            let _ = pairs.push(s.view(), y.view());
            last_gamma = sy / yy;
        }

        let x_moved = !points_match(&x_next, &x);
        x = x_next;
        f = f_next;
        g = g_next;
        e_err = e_err_next;

        // Track the best inner iterate that already lies inside the caps.
        let mut probe = x.clone();
        apply_caps(r0, &mut probe, atom_cap, euclid_cap);
        if probe == x && f < f_last_inside {
            last_inside = x.clone();
            f_last_inside = f;
        }

        if !x_moved {
            break;
        }
    }

    // Apply displacement caps relative to r0.
    let mut capped = x.clone();
    apply_caps(r0, &mut capped, atom_cap, euclid_cap);
    if capped == x {
        return Ok(x);
    }

    // Cap bound: verify the radially scaled point still decreases the
    // surrogate; if a curved valley causes an increase, fall back to the
    // best inner iterate inside the cap or backtrack along the capped ray.
    let (f_cap, _) = gp.predict(capped.view())?;
    if f_cap < f_start {
        return Ok(capped);
    }
    if f_last_inside < f_start && !points_match(&last_inside, r0) {
        return Ok(last_inside);
    }
    let ray = &capped - r0;
    let mut scale_t = 0.5;
    for _ in 0..20 {
        let trial = r0 + &(&ray * scale_t);
        let (f_t, _) = gp.predict(trial.view())?;
        if f_t < f_start {
            return Ok(trial);
        }
        scale_t *= 0.5;
    }
    Ok(capped)
}

/// Outcome of one evaluated GPMin step.
pub(crate) struct GpMinStepOutcome {
    pub x: Array1<f64>,
    pub value: f64,
    pub grad: Array1<f64>,
    pub grad_norm: f64,
    pub evals: usize,
    pub stalled_machine: bool,
}

/// Standalone GPMin stepper implementing [`Stepper`].
///
/// ```
/// use ndarray::array;
/// use rgmin::{GpMin, GpMinConfig, Oracle, Stepper};
///
/// let obj = Oracle::unbounded(2, |x: ndarray::ArrayView1<f64>| {
///     (x.dot(&x), ndarray::Array1::from_iter(x.iter().map(|v| 2.0 * v)))
/// });
/// let mut gpmin = GpMin::new(GpMinConfig::default(), 2).unwrap();
/// let mut x = array![0.4, -0.3];
/// let report = gpmin.step(&obj, &mut x).unwrap();
/// assert!(report.value < 0.25);
/// ```
#[derive(Clone, Debug)]
pub struct GpMin {
    config: GpMinConfig,
    gp: GaussianProcess,
    steps: usize,
    evals: usize,
    last_x: Option<Array1<f64>>,
    last_value: f64,
    last_grad: Array1<f64>,
}

impl GpMin {
    /// Create a new GPMin stepper for dimension `dim`.
    pub fn new(config: GpMinConfig, dim: usize) -> Result<Self> {
        let gp = GaussianProcess::new(dim, &config)?;
        Ok(Self {
            config,
            gp,
            steps: 0,
            evals: 0,
            last_x: None,
            last_value: 0.0,
            last_grad: Array1::zeros(dim),
        })
    }

    /// Coordinate dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.gp.dim()
    }

    /// Current configuration.
    #[must_use]
    pub fn config(&self) -> GpMinConfig {
        self.config
    }

    /// Replace the configuration and reset the GP training history.
    pub fn set_config(&mut self, config: GpMinConfig) -> Result<()> {
        config.validate()?;
        self.config = config;
        self.reset()
    }

    /// Drop all stored observations and reset hyperparameters and counters
    /// to the configured initial values.
    pub fn reset(&mut self) -> Result<()> {
        self.gp.reset(&self.config)?;
        self.steps = 0;
        self.evals = 0;
        self.last_x = None;
        self.last_value = 0.0;
        self.last_grad.fill(0.0);
        Ok(())
    }

    /// Number of observations currently stored in the GP surrogate.
    #[must_use]
    pub fn observations(&self) -> usize {
        self.gp.len()
    }

    /// Current `(weight, scale, noise)` hyperparameters.
    #[must_use]
    pub fn hyperparameters(&self) -> (f64, f64, f64) {
        self.gp.hyperparameters()
    }

    /// Current constant prior mean `c`.
    #[must_use]
    pub fn prior_mean(&self) -> f64 {
        self.gp.prior_mean()
    }

    /// Outer steps completed.
    #[must_use]
    pub fn steps(&self) -> usize {
        self.steps
    }

    /// Total objective evaluations performed by this stepper.
    #[must_use]
    pub fn evals(&self) -> usize {
        self.evals
    }

    /// Predict `(E_hat(x), grad E_hat(x))` from the trained GP surrogate.
    pub fn predict(&self, x: ArrayView1<'_, f64>) -> Result<(f64, Array1<f64>)> {
        self.gp.predict(x)
    }

    /// Record an observation, optionally refit hyperparameters on the
    /// configured schedule, and factorize the GP covariance matrix.
    fn observe_and_train(&mut self, x: Array1<f64>, energy: f64, grad: Array1<f64>) -> Result<()> {
        self.gp.push_observation(x, energy, grad)?;
        if self.config.update_hyperparams && self.gp.len().is_multiple_of(self.config.batch_size) {
            let _ = self.gp.fit_hyperparameters(&self.config)?;
        }
        self.gp.train()
    }

    /// Perform one GPMin step starting from an already-evaluated point
    /// `(r0, e0, g0)`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn step_from_evaluated<Eval>(
        &mut self,
        r0: &Array1<f64>,
        e0: f64,
        g0: &Array1<f64>,
        gnorm0: f64,
        atom_cap: Option<f64>,
        euclid_cap: Option<f64>,
        gtol: f64,
        mut eval_fn: Eval,
    ) -> Result<GpMinStepOutcome>
    where
        Eval: FnMut(&Array1<f64>) -> Result<(f64, Array1<f64>, f64)>,
    {
        if !e0.is_finite() || g0.iter().any(|v| !v.is_finite()) {
            return Err(Error::Oracle {
                what: "non-finite value or gradient",
            });
        }

        // Observe r0 unless it is already the most recent observation in the GP.
        let already_observed = match self.gp.xs.last() {
            Some(prev) => points_match(prev, r0),
            None => false,
        };
        if !already_observed || !self.gp.trained {
            self.observe_and_train(r0.clone(), e0, g0.clone())?;
        }

        let mut trial = relax_surrogate(&self.gp, r0, &self.config, atom_cap, euclid_cap)?;
        if points_match(&trial, r0) {
            return Ok(GpMinStepOutcome {
                x: r0.clone(),
                value: e0,
                grad: g0.clone(),
                grad_norm: gnorm0,
                evals: 0,
                stalled_machine: true,
            });
        }

        let (mut e1, mut g1, mut gnorm1) = eval_fn(&trial)?;
        let mut step_evals = 1usize;
        if !e1.is_finite() || g1.iter().any(|v| !v.is_finite()) {
            return Err(Error::Oracle {
                what: "non-finite value or gradient",
            });
        }

        let mut refinements = 0usize;
        while e1 >= e0 {
            if gtol > 0.0 && gnorm1 <= gtol {
                break;
            }
            if refinements >= self.config.max_refinements {
                return Err(Error::GpMin {
                    what: "could not build a descent surrogate within max_refinements",
                });
            }
            self.observe_and_train(trial.clone(), e1, g1.clone())?;
            let next_trial = relax_surrogate(&self.gp, r0, &self.config, atom_cap, euclid_cap)?;
            if points_match(&next_trial, r0) || points_match(&next_trial, &trial) {
                return Ok(GpMinStepOutcome {
                    x: r0.clone(),
                    value: e0,
                    grad: g0.clone(),
                    grad_norm: gnorm0,
                    evals: step_evals,
                    stalled_machine: true,
                });
            }
            trial = next_trial;
            let (e_next, g_next, gnorm_next) = eval_fn(&trial)?;
            step_evals += 1;
            if !e_next.is_finite() || g_next.iter().any(|v| !v.is_finite()) {
                return Err(Error::Oracle {
                    what: "non-finite value or gradient",
                });
            }
            e1 = e_next;
            g1 = g_next;
            gnorm1 = gnorm_next;
            refinements += 1;
        }

        Ok(GpMinStepOutcome {
            x: trial,
            value: e1,
            grad: g1,
            grad_norm: gnorm1,
            evals: step_evals,
            stalled_machine: false,
        })
    }

    /// Run GPMin until `control`'s stop criteria or iteration limit are met.
    pub fn run<O>(
        &mut self,
        objective: &O,
        x: &mut Array1<f64>,
        control: &Control,
    ) -> Result<Report>
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        control.validate()?;
        if x.len() != self.dim() || Objective::dim(objective) != self.dim() {
            return Err(Error::Dim {
                got: x.len(),
                dim: self.dim(),
            });
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(crate::step::non_finite());
        }

        let (mut value, mut grad) = self.eval_or_cached(objective, x)?;
        let mut gnorm = control.stop_norm.of(grad.view());
        let build_report = |val: f64,
                            coords: Array1<f64>,
                            g: Array1<f64>,
                            gn: f64,
                            steps: usize,
                            evals: usize,
                            status: Status| {
            Report::new(val, coords)
                .with_steps(steps)
                .with_grad_norm(gn)
                .with_stop_norm(control.stop_norm)
                .with_grad(Some(g))
                .with_evals(evals)
                .with_status(status)
        };
        if control.maxiter == 0 {
            let status = if gnorm <= control.gtol {
                Status::Tolerance
            } else {
                Status::Budget
            };
            return Ok(build_report(
                value,
                x.clone(),
                grad,
                gnorm,
                self.steps,
                self.evals,
                status,
            ));
        }
        if gnorm <= control.gtol {
            return Ok(build_report(
                value,
                x.clone(),
                grad,
                gnorm,
                self.steps,
                self.evals,
                Status::Tolerance,
            ));
        }

        let mut iters = 0usize;
        while iters < control.maxiter {
            let outcome = self.step_from_evaluated(
                x,
                value,
                &grad,
                gnorm,
                self.config.max_step,
                control.maxmove_cap(),
                control.gtol,
                |trial| {
                    let clipped = objective.bounds().clip(trial.view());
                    let (v, g) = objective.value_and_gradient(clipped.view());
                    let gn = control.stop_norm.of(g.view());
                    Ok((v, g, gn))
                },
            )?;
            self.evals += outcome.evals;
            self.steps += 1;
            iters += 1;
            *x = outcome.x;
            value = outcome.value;
            grad = outcome.grad;
            gnorm = outcome.grad_norm;
            self.last_x = Some(x.clone());
            self.last_value = value;
            self.last_grad = grad.clone();

            if gnorm <= control.gtol {
                return Ok(build_report(
                    value,
                    x.clone(),
                    grad,
                    gnorm,
                    self.steps,
                    self.evals,
                    Status::Tolerance,
                ));
            }
            if outcome.stalled_machine {
                return Ok(build_report(
                    value,
                    x.clone(),
                    grad,
                    gnorm,
                    self.steps,
                    self.evals,
                    Status::MachinePrecision,
                ));
            }
        }

        Ok(build_report(
            value,
            x.clone(),
            grad,
            gnorm,
            self.steps,
            self.evals,
            Status::Budget,
        ))
    }

    fn eval_or_cached<O>(&mut self, objective: &O, x: &Array1<f64>) -> Result<(f64, Array1<f64>)>
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        if let Some(prev) = &self.last_x
            && prev == x
        {
            return Ok((self.last_value, self.last_grad.clone()));
        }
        let (v, g) = objective.value_and_gradient(x.view());
        self.evals += 1;
        if !v.is_finite() || g.iter().any(|c| !c.is_finite()) {
            return Err(Error::Oracle {
                what: "non-finite value or gradient",
            });
        }
        self.last_x = Some(x.clone());
        self.last_value = v;
        self.last_grad = g.clone();
        Ok((v, g))
    }
}

impl Stepper for GpMin {
    fn step<O>(&mut self, objective: &O, x: &mut Array1<f64>) -> Result<Report>
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        if x.len() != self.dim() || Objective::dim(objective) != self.dim() {
            return Err(Error::Dim {
                got: x.len(),
                dim: self.dim(),
            });
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(crate::step::non_finite());
        }
        let (value, grad) = self.eval_or_cached(objective, x)?;
        let gnorm0 = l2(&grad);
        let outcome = self.step_from_evaluated(
            x,
            value,
            &grad,
            gnorm0,
            self.config.max_step,
            None,
            0.0,
            |trial| {
                let clipped = objective.bounds().clip(trial.view());
                let (v, g) = objective.value_and_gradient(clipped.view());
                let gn = l2(&g);
                Ok((v, g, gn))
            },
        )?;
        self.evals += outcome.evals;
        self.steps += 1;
        *x = outcome.x;
        self.last_x = Some(x.clone());
        self.last_value = outcome.value;
        self.last_grad = outcome.grad.clone();
        let status = if outcome.stalled_machine {
            Status::MachinePrecision
        } else {
            Status::Iterating
        };
        Ok(Report::new(outcome.value, x.clone())
            .with_steps(self.steps)
            .with_grad_norm(outcome.grad_norm)
            .with_grad(Some(outcome.grad))
            .with_evals(self.evals)
            .with_status(status))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Oracle;
    use ndarray::array;

    /// Analytic directional derivatives of the scalar squared-exponential
    /// kernel compared directly against `SquaredExponentialKernel::block`.
    #[test]
    fn kernel_derivative_blocks_match_analytic_formulas() {
        let w = 1.7;
        let l = 0.65;
        let kernel = SquaredExponentialKernel::default()
            .with_weight(w)
            .with_scale(l);
        let x1 = array![0.2, -0.5, 0.9];
        let x2 = array![-0.1, 0.3, 0.4];

        let blk = kernel.block(x1.view(), x2.view());
        let d = &x1 - &x2;
        let r2 = d.dot(&d) / (l * l);
        let k0 = w * w * (-0.5 * r2).exp();

        let tol = 32.0 * f64::EPSILON * k0.abs().max(1.0);
        // K[0, 0] = k(x1, x2)
        assert!((blk[(0, 0)] - k0).abs() <= tol);

        // K[0, b] = d/dx2_b k(x1, x2) = +k * (x1_b - x2_b) / l^2
        // K[a, 0] = d/dx1_a k(x1, x2) = -k * (x1_a - x2_a) / l^2
        for a in 0..3 {
            let expected_0a = k0 * d[a] / (l * l);
            assert!((blk[(0, a + 1)] - expected_0a).abs() <= tol);
            assert!((blk[(a + 1, 0)] + expected_0a).abs() <= tol);
        }

        // K[a, b] = d^2/(dx1_a dx2_b) k(x1, x2) = (k / l^2) * (delta_ab - d_a d_b / l^2)
        for a in 0..3 {
            for b in 0..3 {
                let delta = if a == b { 1.0 } else { 0.0 };
                let expected = (k0 / (l * l)) * (delta - d[a] * d[b] / (l * l));
                assert!((blk[(a + 1, b + 1)] - expected).abs() <= tol * (1.0 / (l * l)));
            }
        }

        // Symmetry: K(x2, x1) == K(x1, x2)^T
        let blk_rev = kernel.block(x2.view(), x1.view());
        for i in 0..4 {
            for j in 0..4 {
                assert!((blk_rev[(i, j)] - blk[(j, i)]).abs() <= tol * (1.0 / (l * l)));
            }
        }
    }

    /// Analytic hyperparameter derivatives `dK/dw` and `dK/dl` checked
    /// against closed-form expressions derived from differentiating each
    /// entry of `K(x1, x2)` explicitly.
    #[test]
    fn kernel_hyperparam_derivatives_match_closed_form() {
        let w = 2.3;
        let l = 0.55;
        let kernel = SquaredExponentialKernel::default()
            .with_weight(w)
            .with_scale(l);
        let x1 = array![0.15, -0.25, 0.45];
        let x2 = array![-0.2, 0.1, 0.05];

        let dw = kernel.d_weight_block(x1.view(), x2.view());
        let dl = kernel.d_scale_block(x1.view(), x2.view());
        let blk = kernel.block(x1.view(), x2.view());

        let d = &x1 - &x2;
        let d2 = d.dot(&d);
        let r2 = d2 / (l * l);
        let k0 = w * w * (-0.5 * r2).exp();
        let tol = 64.0 * f64::EPSILON * (k0 / (l * l * l)).abs().max(1.0);

        // dK/dw = (2 / w) K
        for i in 0..4 {
            for j in 0..4 {
                assert!((dw[(i, j)] - (2.0 / w) * blk[(i, j)]).abs() <= tol);
            }
        }

        // Exact closed-form d/dl of every entry of K(x1, x2):
        // d/dl [w^2 exp(-d^2 / (2 l^2))] = k * d^2 / l^3
        assert!((dl[(0, 0)] - k0 * d2 / (l * l * l)).abs() <= tol);

        // d/dl [k * d_a / l^2] = k * d_a * (d^2 / l^5 - 2 / l^3)
        for a in 0..3 {
            let expected = k0 * d[a] * (d2 / l.powi(5) - 2.0 / l.powi(3));
            assert!((dl[(0, a + 1)] - expected).abs() <= tol);
            assert!((dl[(a + 1, 0)] + expected).abs() <= tol);
        }

        // d/dl [k * (delta_ab / l^2 - d_a d_b / l^4)]
        // = (k * d^2 / l^3) * (delta_ab / l^2 - d_a d_b / l^4)
        //   + k * (-2 * delta_ab / l^3 + 4 * d_a d_b / l^5)
        for a in 0..3 {
            for b in 0..3 {
                let delta = if a == b { 1.0 } else { 0.0 };
                let term1 = (k0 * d2 / l.powi(3)) * (delta / (l * l) - d[a] * d[b] / l.powi(4));
                let term2 = k0 * (-2.0 * delta / l.powi(3) + 4.0 * d[a] * d[b] / l.powi(5));
                let expected = term1 + term2;
                assert!((dl[(a + 1, b + 1)] - expected).abs() <= tol);
            }
        }
    }

    /// Posterior mean gradient matches the analytic derivative of the
    /// posterior mean energy expression `m(x) = c + sum_j k_j(x) (a_0j + u_j . a_gj)`.
    #[test]
    fn posterior_mean_prediction_matches_full_block_product() {
        let cfg = GpMinConfig::default();
        let mut gp = GaussianProcess::new(3, &cfg).unwrap();
        gp.push_observation(array![0.0, 0.0, 0.0], 1.2, array![0.4, -0.3, 0.2])
            .unwrap();
        gp.push_observation(array![0.2, -0.1, 0.15], 0.85, array![0.1, -0.05, 0.08])
            .unwrap();
        gp.train().unwrap();

        let query = array![0.12, -0.04, 0.09];
        let (e_fast, g_fast) = gp.predict(query.view()).unwrap();

        // Compare with explicit block matrix-vector product.
        let kernel = SquaredExponentialKernel::default()
            .with_weight(cfg.weight)
            .with_scale(cfg.scale);
        let mut e_ref = gp.prior_mean();
        let mut g_ref: Array1<f64> = Array1::zeros(3);
        let d1 = 4;
        for (j, xj) in gp.xs.iter().enumerate() {
            let blk = kernel.block(query.view(), xj.view());
            let a_slice = &gp.alpha[(j * d1)..((j + 1) * d1)];
            for c in 0..d1 {
                e_ref += blk[(0, c)] * a_slice[c];
                for r in 0..3 {
                    g_ref[r] += blk[(r + 1, c)] * a_slice[c];
                }
            }
        }

        let tol = 64.0 * f64::EPSILON * (1.0 + e_ref.abs());
        assert!((e_fast - e_ref).abs() <= tol);
        for r in 0..3 {
            assert!((g_fast[r] - g_ref[r]).abs() <= tol);
        }
    }

    /// With one observation and constant prior `c = E_0`, the exact
    /// surrogate minimum lies at distance `scale` along `-g_0 / ||g_0||`.
    #[test]
    fn single_observation_surrogate_minimum_matches_analytic_distance() {
        let cfg = GpMinConfig::default().with_scale(0.35);
        let mut gp = GaussianProcess::new(3, &cfg).unwrap();
        let r0 = array![0.1, -0.2, 0.3];
        let g0 = array![3.0, -4.0, 0.0];
        gp.push_observation(r0.clone(), 5.0, g0.clone()).unwrap();
        gp.train().unwrap();

        let r1 = relax_surrogate(&gp, &r0, &cfg, None, None).unwrap();
        let step = &r1 - &r0;
        let expected = -&g0 * (cfg.scale / l2(&g0));
        let err = l2(&(&step - &expected));
        assert!(err <= 1024.0 * cfg.surrogate_gtol, "step error {err}");
    }

    /// Non-SPD matrix fails closed with `Error::GpMin` instead of silently
    /// adding diagonal jitter.
    #[test]
    fn cholesky_rejects_non_spd_matrix() {
        let bad = vec![1.0, 2.0, 2.0, 1.0];
        let err = cholesky_lower(&bad, 2).unwrap_err();
        assert!(
            matches!(err, Error::GpMin { .. }),
            "expected Error::GpMin, got {err:?}"
        );
    }

    /// Non-finite energy or gradient fails closed with `Error::Oracle`.
    #[test]
    fn non_finite_objective_returns_oracle_error() {
        let obj = Oracle::unbounded(2, |_x: ArrayView1<'_, f64>| (f64::NAN, array![0.0, 0.0]));
        let mut gpmin = GpMin::new(GpMinConfig::default(), 2).unwrap();
        let mut x = array![0.1, 0.2];
        let err = gpmin.step(&obj, &mut x).unwrap_err();
        assert!(
            matches!(err, Error::Oracle { .. }),
            "expected Error::Oracle, got {err:?}"
        );
    }

    /// Exact analytic marginal log-likelihood gradient matches the exact
    /// closed-form trace identity for a 1-observation system where every
    /// matrix is diagonal.
    #[test]
    fn marginal_log_likelihood_gradient_matches_exact_diagonal_1pt() {
        let cfg = GpMinConfig::default()
            .with_weight(1.5)
            .with_scale(0.45)
            .with_noise(0.02)
            .with_prior(PriorStrategy::Fixed(0.25));
        let mut gp = GaussianProcess::new(2, &cfg).unwrap();
        let e0 = 1.75;
        let g0 = array![0.6, -0.8];
        gp.push_observation(array![0.0, 0.0], e0, g0.clone())
            .unwrap();

        let (nll, grad) = gp
            .neg_log_marginal_likelihood(cfg.weight, cfg.scale, cfg.noise)
            .unwrap();

        // For 1 observation, C is diagonal:
        // c_e = w^2 + (noise * l)^2
        // c_g = w^2 / l^2 + noise^2  (repeated D=2 times)
        let w = cfg.weight;
        let l = cfg.scale;
        let n = cfg.noise;
        let ce = w * w + (n * l) * (n * l);
        let cg = w * w / (l * l) + n * n;
        let re = e0 - 0.25;
        let rg2 = g0[0] * g0[0] + g0[1] * g0[1];

        let exact_nll = 0.5 * (re * re / ce + rg2 / cg)
            + 0.5 * (ce.ln() + 2.0 * cg.ln())
            + 1.5 * (2.0 * std::f64::consts::PI).ln();

        // d(nll)/dw: dce/dw = 2w, dcg/dw = 2w/l^2
        let dce_dw = 2.0 * w;
        let dcg_dw = 2.0 * w / (l * l);
        let exact_dw = 0.5
            * (dce_dw * (1.0 / ce - re * re / (ce * ce)) + dcg_dw * (2.0 / cg - rg2 / (cg * cg)));

        // d(nll)/dl: dce/dl = 2 n^2 l, dcg/dl = -2 w^2 / l^3
        let dce_dl = 2.0 * n * n * l;
        let dcg_dl = -2.0 * w * w / (l * l * l);
        let exact_dl = 0.5
            * (dce_dl * (1.0 / ce - re * re / (ce * ce)) + dcg_dl * (2.0 / cg - rg2 / (cg * cg)));

        let tol = 64.0 * f64::EPSILON * (1.0 + exact_nll.abs() + exact_dw.abs() + exact_dl.abs());
        assert!((nll - exact_nll).abs() <= tol);
        assert!((grad[0] - exact_dw).abs() <= tol);
        assert!((grad[1] - exact_dl).abs() <= tol);
    }

    /// Public configs are built from [`Default`] and the `with_*` setters.
    #[test]
    fn public_configs_build_from_default_and_setters() {
        let cfg = GpMinConfig::default()
            .with_scale(0.35)
            .with_weight(1.5)
            .with_noise(0.002)
            .with_prior(PriorStrategy::Fixed(0.25))
            .with_update_hyperparams(true)
            .with_batch_size(2)
            .with_hyperparam_bounds(0.2)
            .with_max_step(0.15)
            .with_max_refinements(4)
            .with_surrogate_maxiter(20)
            .with_surrogate_memory(3)
            .with_surrogate_gtol(1.0e-6)
            .with_hyperparam_tol(0.02)
            .with_hyperparam_maxiter(8);
        assert_eq!(cfg.scale, 0.35);
        assert_eq!(cfg.weight, 1.5);
        assert_eq!(cfg.noise, 0.002);
        assert_eq!(cfg.prior, PriorStrategy::Fixed(0.25));
        assert!(cfg.update_hyperparams);
        assert_eq!(cfg.batch_size, 2);
        assert_eq!(cfg.hyperparam_bounds, Some(0.2));
        assert_eq!(cfg.max_step, Some(0.15));
        assert_eq!(cfg.max_refinements, 4);
        assert_eq!(cfg.surrogate_maxiter, 20);
        assert_eq!(cfg.surrogate_memory, 3);
        assert_eq!(cfg.surrogate_gtol, 1.0e-6);
        assert_eq!(cfg.hyperparam_tol, 0.02);
        assert_eq!(cfg.hyperparam_maxiter, 8);
        cfg.validate().unwrap();

        let defaults = SquaredExponentialKernel::default();
        assert_eq!(defaults.weight, GPMIN_DEFAULT_WEIGHT);
        assert_eq!(defaults.scale, GPMIN_DEFAULT_SCALE);
        let kernel = defaults.with_weight(1.5).with_scale(0.35);
        assert_eq!(kernel.weight, 1.5);
        assert_eq!(kernel.scale, 0.35);
    }
}
