//! Computational noise of a scalar function, and the finite-difference
//! steps that noise implies.
//!
//! Moré and Wild, *Estimating computational noise*,
//! SIAM J. Sci. Comput. 33 (2011), 1292-1314,
//! <https://doi.org/10.1137/100786125>.
//! Moré and Wild, *Estimating derivatives of noisy simulations*,
//! ACM Trans. Math. Softw. 38 (2012), 19:1-19:21,
//! <https://doi.org/10.1145/2168773.2168777>.
//!
//! [`ecnoise`] reads `m + 1` values at spacing `h` and builds the
//! forward-difference table. The level-`k` estimate is
//! `σ_k = sqrt(γ_k * mean_i (Δ^k f_i)^2)` with
//! `γ_k = (k!)^2 / (2k)!`. The returned noise is the smallest `k`
//! whose column changes sign and whose three estimates
//! `σ_k`, `σ_{k+1}`, `σ_{k+2}` lie within a factor of 4.
//! [`DEFAULT_M`] is 8, so the table has 9 samples.
//!
//! `h` is that sample spacing, the same `h` as in `ECnoise.m`'s
//! stencil `f(x - kh), ..., f(x + kh)`. It is the distance between
//! consecutive samples, not the half-width of the stencil.
//!
//! One projection `g · w` with `w` uniform on the unit sphere has
//! `E[eps_g²] = tr(Σ) / n` for any noise covariance `Σ`. The sphere
//! average of `w wᵀ` is `I / n`, so the identity does not need `Σ` to
//! be a multiple of `I`. The noise in `||g||_2` then has
//! `E[||e||²] = tr(Σ) = n * eps_g²`, and the floor compares the
//! stationarity measure with `c * sqrt(n_free) * eps_g`.

use eindir_core::{DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1};

use crate::error::{Error, Result};
use crate::vecops::dot_ordered;

/// Extra samples beyond the center. The table length is `m + 1`.
pub const DEFAULT_M: usize = 8;

/// Absolute piece of the lower clamp on [`forward_step`] and [`gradient_dr`].
///
/// The clamp that is applied is `max(STEP_CLAMP_MIN, 4 * ulp(point_scale))`.
/// `1e-16` is below one ulp of a coordinate near 10 Å (`ulp(10) ≈ 1.8e-15`),
/// so a step that small leaves `x + h == x`.
pub const STEP_CLAMP_MIN: f64 = 1e-16;
/// Upper clamp on [`forward_step`] and [`gradient_dr`], in the caller's length unit.
pub const STEP_CLAMP_MAX: f64 = 1.0;

/// Signal-to-noise factor in Moré and Wild, ACM TOMS 38 (2012), (17a).
const DERIV_TAU1: f64 = 100.0;
/// Relative-variation factor in the same paper, (17b).
const DERIV_TAU2: f64 = 0.1;

/// Why [`ecnoise`] accepted or refused a noise level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoiseFlag {
    /// Three consecutive levels agree within a factor of 4 and the
    /// selected column changes sign. Published `inform = 1`.
    Detected,
    /// At least half of the first differences are zero. Published
    /// `inform = 2`. The reference code retries at `100 h`.
    StepTooSmall,
    /// The samples differ in the leading digit, or no column meets
    /// the agreement test. Published `inform = 3`. The reference
    /// code retries at `h / 100`.
    StepTooLarge,
    /// [`NoiseEstimate::declared`]: a backend supplied `eps` and the
    /// table was not built.
    Declared,
}

impl NoiseFlag {
    /// Published `inform` code. A declared value is `0`.
    pub fn inform(self) -> i32 {
        match self {
            Self::Detected => 1,
            Self::StepTooSmall => 2,
            Self::StepTooLarge => 3,
            Self::Declared => 0,
        }
    }
}

/// Noise level, the per-order estimates, and the flag.
#[derive(Clone, Debug, PartialEq)]
pub struct NoiseEstimate {
    /// Accepted `σ_k`, the declared value, or 0 when the flag is a refusal.
    pub eps: f64,
    /// `σ_k` for `k = 1 .. nf - 1`. Empty when the flag is [`NoiseFlag::Declared`]
    /// or the call returns before the table is filled.
    pub level: Vec<f64>,
    /// Detection outcome.
    pub ok: NoiseFlag,
}

impl NoiseEstimate {
    /// Backend-declared noise. Sampling is not performed.
    pub fn declared(eps: f64) -> Self {
        Self {
            eps,
            level: Vec::new(),
            ok: NoiseFlag::Declared,
        }
    }
}

/// Estimate the noise level of `values`, samples of one function at equal spacing.
///
/// `values.len()` is `m + 1` and must be at least 4. Every entry must
/// be finite. The slice is left unchanged.
pub fn ecnoise(values: &[f64]) -> Result<NoiseEstimate> {
    let nf = values.len();
    if nf < 4 {
        return Err(Error::EcNoise {
            what: "need at least 4 samples",
        });
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(Error::EcNoise {
            what: "sample was not finite",
        });
    }

    let mut level = vec![0.0; nf - 1];
    let fmin = values.iter().copied().fold(f64::INFINITY, f64::min);
    let fmax = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let scale = fmax.abs().max(fmin.abs());
    if scale != 0.0 && (fmax - fmin) / scale > 0.1 {
        return Ok(refused(level, NoiseFlag::StepTooLarge));
    }

    let mut diffs = values.to_vec();
    let mut gamma = 1.0;
    let mut sign_change = vec![false; nf - 1];
    for j in 1..nf {
        let count = nf - j;
        for i in 0..count {
            diffs[i] = diffs[i + 1] - diffs[i];
        }
        if j == 1 {
            let zeros = diffs[..count].iter().filter(|d| **d == 0.0).count();
            if zeros * 2 >= nf {
                return Ok(refused(level, NoiseFlag::StepTooSmall));
            }
        }
        let jf = j as f64;
        gamma *= 0.5 * (jf / (2.0 * jf - 1.0));
        let active = &diffs[..count];
        let mean_sq = active.iter().map(|t| t * t).sum::<f64>() / count as f64;
        level[j - 1] = (gamma * mean_sq).sqrt();
        let emin = active.iter().copied().fold(f64::INFINITY, f64::min);
        let emax = active.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if emin * emax < 0.0 {
            sign_change[j - 1] = true;
        }
    }

    for k in 0..nf - 3 {
        let window = &level[k..k + 3];
        let emin = window.iter().copied().fold(f64::INFINITY, f64::min);
        let emax = window.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if emax <= 4.0 * emin && sign_change[k] {
            let eps = level[k];
            return Ok(NoiseEstimate {
                eps,
                level,
                ok: NoiseFlag::Detected,
            });
        }
    }
    Ok(refused(level, NoiseFlag::StepTooLarge))
}

fn refused(level: Vec<f64>, ok: NoiseFlag) -> NoiseEstimate {
    NoiseEstimate {
        eps: 0.0,
        level,
        ok,
    }
}

/// Next spacing after a refusal: `100 h` when the step was too small,
/// `h / 100` when it was too large.
pub fn next_spacing(h: f64, flag: NoiseFlag) -> Option<f64> {
    match flag {
        NoiseFlag::StepTooSmall => Some(h * 100.0),
        NoiseFlag::StepTooLarge => Some(h / 100.0),
        NoiseFlag::Detected | NoiseFlag::Declared => None,
    }
}

/// Sample `objective` at `m + 1` points spaced by `h` and estimate the noise.
///
/// The points are `x + (i - m/2) h u` for `i = 0 ..= m`, with `u` the
/// unit vector along `direction`. `h` is the spacing. `m` of 0 selects
/// [`DEFAULT_M`].
pub fn sample_noise<O>(
    objective: &O,
    x: ArrayView1<f64>,
    direction: ArrayView1<f64>,
    h: f64,
    m: usize,
) -> Result<NoiseEstimate>
where
    O: Objective<f64> + Sync + ?Sized,
{
    let values = line_values(objective, x, direction, h, m)?;
    ecnoise(&values)
}

/// Objective values on the stencil of [`sample_noise`], stored at the
/// sample index.
pub fn line_values<O>(
    objective: &O,
    x: ArrayView1<f64>,
    direction: ArrayView1<f64>,
    h: f64,
    m: usize,
) -> Result<Vec<f64>>
where
    O: Objective<f64> + Sync + ?Sized,
{
    let m = if m == 0 { DEFAULT_M } else { m };
    let unit = unit_direction(x, direction, h, m, Objective::dim(objective))?;
    map_indexed(m + 1, |i| {
        let shift = (i as f64 - m as f64 / 2.0) * h;
        let point = &x + &(shift * &unit);
        objective.eval(point.view())
    })
    .and_then(finite_values)
}

/// Noise of `g(x + i h u) · w`, one counter-keyed direction.
///
/// Equivalent to [`sample_gradient_noise_k`] with `k = 1`.
pub fn sample_gradient_noise<O>(
    objective: &O,
    x: ArrayView1<f64>,
    direction: ArrayView1<f64>,
    h: f64,
    m: usize,
    seed: u64,
) -> Result<NoiseEstimate>
where
    O: Gradient<f64> + Objective<f64> + Sync + ?Sized,
{
    sample_gradient_noise_k(objective, x, direction, h, m, seed, 1)
}

/// Average `eps_g²` over `k` directions.
///
/// Direction `j` is the unit vector from the counter stream
/// `(direction_seed(seed, j), i)`. `k` of 0 is treated as 1. Each
/// direction builds one ECnoise table on `g · w` with that same `w` at
/// every stencil point. The returned `eps` is the root-mean-square of
/// the detected values, which is `eps_g` for `k = 1` and a lower-variance
/// estimate of the same projection noise for `k > 1`. For any covariance
/// `Σ`, each projection has `E[eps_g²] = tr(Σ) / n` because `w` is
/// uniform on the sphere. `level` is the table of the first direction
/// that was accepted.
pub fn sample_gradient_noise_k<O>(
    objective: &O,
    x: ArrayView1<f64>,
    direction: ArrayView1<f64>,
    h: f64,
    m: usize,
    seed: u64,
    k: usize,
) -> Result<NoiseEstimate>
where
    O: Gradient<f64> + Objective<f64> + Sync + ?Sized,
{
    let k = k.max(1);
    let mut sum_sq = 0.0;
    let mut n_ok = 0usize;
    let mut level = Vec::new();
    let mut last_fail = None;
    for j in 0..k {
        let est =
            sample_gradient_noise_one(objective, x, direction, h, m, direction_seed(seed, j))?;
        if est.ok == NoiseFlag::Detected {
            sum_sq += est.eps * est.eps;
            n_ok += 1;
            if level.is_empty() {
                level = est.level;
            }
        } else {
            last_fail = Some(est);
        }
    }
    if n_ok == 0 {
        return Ok(last_fail.expect("k >= 1 produces a table"));
    }
    Ok(NoiseEstimate {
        eps: (sum_sq / n_ok as f64).sqrt(),
        level,
        ok: NoiseFlag::Detected,
    })
}

/// Energy noise and one gradient projection from a single stencil.
///
/// Each sample calls [`DifferentiableObjective::value_and_gradient`]
/// once. The energy table and the projection `g · w` are the two
/// ECnoise inputs. `w` is the counter direction of [`sample_gradient_noise`],
/// so the gradient half matches that function at `k = 1`.
pub fn sample_value_gradient_noise<O>(
    objective: &O,
    x: ArrayView1<f64>,
    direction: ArrayView1<f64>,
    h: f64,
    m: usize,
    seed: u64,
) -> Result<ValueGradientNoise>
where
    O: DifferentiableObjective<f64> + Sync + ?Sized,
{
    let m = if m == 0 { DEFAULT_M } else { m };
    let unit = unit_direction(x, direction, h, m, Objective::dim(objective))?;
    let w = counter_direction(seed, x.len());
    let samples = map_pairs(m + 1, |i| {
        let shift = (i as f64 - m as f64 / 2.0) * h;
        let point = &x + &(shift * &unit);
        let (f, g) = objective.value_and_gradient(point.view());
        (f, dot_ordered(g.view(), w.view()))
    })?;
    let values = finite_values(samples.iter().map(|sample| sample.0).collect())?;
    let projected = finite_values(samples.iter().map(|sample| sample.1).collect())?;
    Ok(ValueGradientNoise {
        value: ecnoise(&values)?,
        gradient: ecnoise(&projected)?,
    })
}

/// [`ecnoise`] on the energy and on one projection, from one stencil.
#[derive(Clone, Debug, PartialEq)]
pub struct ValueGradientNoise {
    /// Noise of `f` along the stencil.
    pub value: NoiseEstimate,
    /// Noise of `g · w` at those same points.
    pub gradient: NoiseEstimate,
}

fn sample_gradient_noise_one<O>(
    objective: &O,
    x: ArrayView1<f64>,
    direction: ArrayView1<f64>,
    h: f64,
    m: usize,
    seed: u64,
) -> Result<NoiseEstimate>
where
    O: Gradient<f64> + Objective<f64> + Sync + ?Sized,
{
    let m = if m == 0 { DEFAULT_M } else { m };
    let unit = unit_direction(x, direction, h, m, Objective::dim(objective))?;
    let w = counter_direction(seed, x.len());
    let values = map_indexed(m + 1, |i| {
        let shift = (i as f64 - m as f64 / 2.0) * h;
        let point = &x + &(shift * &unit);
        let g = objective.grad(point.view());
        dot_ordered(g.view(), w.view())
    })?;
    let values = finite_values(values)?;
    ecnoise(&values)
}

/// Stream key for direction `j`. Direction 0 keeps `seed`, so one
/// direction matches the historical counter.
fn direction_seed(seed: u64, j: usize) -> u64 {
    if j == 0 {
        return seed;
    }
    let mut z = seed ^ (j as u64).wrapping_mul(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Unit vector whose component `i` comes from the stream `(seed, i)`.
pub fn counter_direction(seed: u64, n: usize) -> Array1<f64> {
    let mut w = Array1::zeros(n.max(1));
    for i in 0..w.len() {
        w[i] = normal(seed, i);
    }
    let nrm = dot_ordered(w.view(), w.view()).sqrt();
    if nrm == 0.0 || !nrm.is_finite() {
        w[0] = 1.0;
        return w;
    }
    w.mapv_inplace(|v| v / nrm);
    w
}

fn normal(seed: u64, index: usize) -> f64 {
    let u1 = uniform(seed, (2 * index) as u64);
    let u2 = uniform(seed, (2 * index + 1) as u64);
    // libm, not the platform libm, so the bits do not depend on the target.
    (-2.0 * libm::log(u1)).sqrt() * libm::cos(std::f64::consts::TAU * u2)
}

fn uniform(seed: u64, index: u64) -> f64 {
    let mut z = seed.wrapping_add(index.wrapping_mul(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 31;
    let mant = (z >> 11) as f64 * (1.0 / ((1u64 << 53) as f64));
    mant.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON)
}

fn unit_direction(
    x: ArrayView1<f64>,
    direction: ArrayView1<f64>,
    h: f64,
    m: usize,
    dim: usize,
) -> Result<Array1<f64>> {
    if m < 3 {
        return Err(Error::EcNoise {
            what: "need at least 3 extra evaluations",
        });
    }
    if !h.is_finite() || h <= 0.0 {
        return Err(Error::EcNoise {
            what: "spacing must be finite and positive",
        });
    }
    if direction.len() != x.len() {
        return Err(Error::EcNoise {
            what: "direction length does not match the point",
        });
    }
    if x.len() != dim {
        return Err(Error::EcNoise {
            what: "point length does not match the objective",
        });
    }
    if direction.iter().any(|v| !v.is_finite()) {
        return Err(Error::EcNoise {
            what: "direction entry was not finite",
        });
    }
    let nrm = dot_ordered(direction, direction).sqrt();
    if nrm == 0.0 {
        return Err(Error::EcNoise {
            what: "sampling direction has zero length",
        });
    }
    Ok(&direction / nrm)
}

fn finite_values(values: Vec<f64>) -> Result<Vec<f64>> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(Error::EcNoise {
            what: "objective value was not finite",
        });
    }
    Ok(values)
}

fn map_indexed<F>(n: usize, f: F) -> Result<Vec<f64>>
where
    F: Fn(usize) -> f64 + Send + Sync,
{
    map_slots(n, f)
}

fn map_pairs<F>(n: usize, f: F) -> Result<Vec<(f64, f64)>>
where
    F: Fn(usize) -> (f64, f64) + Send + Sync,
{
    map_slots(n, f)
}

fn map_slots<T, F>(n: usize, f: F) -> Result<Vec<T>>
where
    T: Send,
    F: Fn(usize) -> T + Send + Sync,
{
    #[cfg(feature = "par")]
    {
        use rayon::prelude::*;
        let mut out = Vec::new();
        (0..n).into_par_iter().map(f).collect_into_vec(&mut out);
        debug_assert_eq!(out.len(), n);
        Ok(out)
    }
    #[cfg(not(feature = "par"))]
    {
        Ok((0..n).map(f).collect())
    }
}

/// `|f''|` along `p` by Moré and Wild, ACM TOMS 38 (2012), Algorithm 5.1.
///
/// The central second difference uses its own spacing, not an ECnoise
/// table. Those high differences are noise, of size `O(eps / h^k)`, so
/// a derivative read off that table is far too large.
///
/// Set `ha = eps_f^{1/4}` and `Δ(h) = |f(x−h) − 2f(x) + f(x+h)|` along
/// the unit direction of `p`. Accept `µ = Δ(h) / h²` when
/// `Δ(h) ≥ τ1 * eps_f` and
/// `|f(x±h) − f(x)| ≤ τ2 * max(|f(x)|, |f(x±h)|)`, with `τ1 = 100` and
/// `τ2 = 0.1` (paper equations 17a and 17b). If `ha` fails, one retry
/// uses `hb = (eps_f / µa)^{1/4}`. That retry is accepted when it meets
/// (17), and also when `|µa − µb| ≤ µb / 2` even if (17) fails on `hb`.
/// Otherwise the estimate is `None`.
pub fn estimate_second<O>(
    oracle: &O,
    x: ArrayView1<f64>,
    p: ArrayView1<f64>,
    eps_f: f64,
) -> Option<f64>
where
    O: Objective<f64> + ?Sized,
{
    if !(eps_f.is_finite() && eps_f > 0.0) {
        return None;
    }
    let unit = unit_vector(p)?;
    if unit.len() != x.len() {
        return None;
    }
    let f0 = oracle.eval(x);
    if !f0.is_finite() {
        return None;
    }
    let ha = eps_f.powf(0.25);
    let (mu_a, ok_a) = second_mu(oracle, x, unit.view(), f0, ha, eps_f)?;
    if ok_a {
        return Some(mu_a);
    }
    if !(mu_a.is_finite() && mu_a > 0.0) {
        return None;
    }
    let hb = (eps_f / mu_a).powf(0.25);
    if !(hb.is_finite() && hb > 0.0) {
        return None;
    }
    let (mu_b, ok_b) = second_mu(oracle, x, unit.view(), f0, hb, eps_f)?;
    if ok_b || (mu_a - mu_b).abs() <= 0.5 * mu_b {
        return Some(mu_b);
    }
    None
}

/// Third-derivative scale `|T|` along `u`, for [`gradient_dr`].
///
/// Algorithm 5.1 is applied to the directional derivative
/// `φ(t) = ∇f(x + t û) · û` instead of to `f`, and `eps_g` replaces
/// `eps_f`. The second difference of `φ` is a third derivative of `f`:
/// `Δ(h) = |φ(h) − 2φ(0) + φ(−h)|` and `|T| ≈ Δ(h) / h²`.
/// The two tests are the paper's, on `φ`:
/// `Δ(h) ≥ τ1 * eps_g`, and
/// `|φ(±h) − φ(0)| ≤ τ2 * max(|φ(0)|, |φ(±h)|)`,
/// with the same `τ1 = 100`, `τ2 = 0.1`, the same `ha = eps_g^{1/4}`,
/// and the same one retry `hb = (eps_g / µa)^{1/4}` including the
/// agreement exit `|µa − µb| ≤ µb / 2`.
pub fn estimate_third<O>(
    oracle: &O,
    x: ArrayView1<f64>,
    u: ArrayView1<f64>,
    eps_g: f64,
) -> Option<f64>
where
    O: Gradient<f64> + ?Sized,
{
    if !(eps_g.is_finite() && eps_g > 0.0) {
        return None;
    }
    let unit = unit_vector(u)?;
    if unit.len() != x.len() {
        return None;
    }
    let phi = |step: f64| -> Option<f64> {
        let point = &x + &(step * &unit);
        let g = oracle.grad(point.view());
        if g.iter().any(|v| !v.is_finite()) {
            return None;
        }
        Some(dot_ordered(g.view(), unit.view()))
    };
    let p0 = phi(0.0)?;
    let ha = eps_g.powf(0.25);
    let (mu_a, ok_a) = third_mu(&phi, p0, ha, eps_g)?;
    if ok_a {
        return Some(mu_a);
    }
    if !(mu_a.is_finite() && mu_a > 0.0) {
        return None;
    }
    let hb = (eps_g / mu_a).powf(0.25);
    if !(hb.is_finite() && hb > 0.0) {
        return None;
    }
    let (mu_b, ok_b) = third_mu(&phi, p0, hb, eps_g)?;
    if ok_b || (mu_a - mu_b).abs() <= 0.5 * mu_b {
        return Some(mu_b);
    }
    None
}

fn second_mu<O>(
    oracle: &O,
    x: ArrayView1<f64>,
    unit: ArrayView1<f64>,
    f0: f64,
    h: f64,
    eps: f64,
) -> Option<(f64, bool)>
where
    O: Objective<f64> + ?Sized,
{
    if !(h.is_finite() && h > 0.0) {
        return None;
    }
    let fp = oracle.eval((&x + &(h * &unit)).view());
    let fm = oracle.eval((&x - &(h * &unit)).view());
    if !fp.is_finite() || !fm.is_finite() {
        return None;
    }
    let delta = (fm - 2.0 * f0 + fp).abs();
    let mu = delta / (h * h);
    if !mu.is_finite() {
        return None;
    }
    Some((mu, difference_accepted(f0, fp, fm, delta, eps)))
}

fn third_mu<F>(phi: &F, p0: f64, h: f64, eps: f64) -> Option<(f64, bool)>
where
    F: Fn(f64) -> Option<f64>,
{
    if !(h.is_finite() && h > 0.0) {
        return None;
    }
    let pp = phi(h)?;
    let pm = phi(-h)?;
    let delta = (pm - 2.0 * p0 + pp).abs();
    let mu = delta / (h * h);
    if !mu.is_finite() {
        return None;
    }
    Some((mu, difference_accepted(p0, pp, pm, delta, eps)))
}

/// Paper (17a) and (17b): the second difference clears the noise, and
/// each neighbor stays within a relative factor `τ2` of the center.
fn difference_accepted(f0: f64, fp: f64, fm: f64, delta: f64, eps: f64) -> bool {
    if !(delta.is_finite() && eps.is_finite()) || delta < DERIV_TAU1 * eps {
        return false;
    }
    let near = |fa: f64| (fa - f0).abs() <= DERIV_TAU2 * f0.abs().max(fa.abs());
    near(fp) && near(fm)
}

fn unit_vector(p: ArrayView1<f64>) -> Option<Array1<f64>> {
    if p.is_empty() || p.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let nrm = dot_ordered(p, p).sqrt();
    if nrm == 0.0 || !nrm.is_finite() {
        return None;
    }
    Some(&p / nrm)
}

/// Forward-difference step `h* = 8^(1/4) sqrt(eps_f / |f''|)`.
///
/// The model is a one-sided difference whose error is about
/// `(h/2) |f''| + sqrt(2) eps_f / h`. That expression is smallest at
/// this `h`, when `f''` is the second derivative at the expansion
/// point and `eps_f` is the standard deviation of uncorrelated noise
/// in `f`. `|f''|` comes from [`estimate_second`].
///
/// `point_scale` is `max_i |x_i|`. The result is clamped to
/// `[max(STEP_CLAMP_MIN, 4 * ulp(point_scale)), STEP_CLAMP_MAX]`.
/// The lower end is four ulps of the coordinate so `x + h` moves.
/// A zero scale, and a missing estimate (`None`), with positive noise
/// return the upper clamp. A zero noise returns the lower clamp.
pub fn forward_step(eps_f: f64, second: Option<f64>, point_scale: f64) -> f64 {
    clamped_step(eps_f, second, 8.0_f64.powf(0.25), point_scale)
}

/// Gradient-difference step `dr* = 2 sqrt(eps_g / |T|)`.
///
/// The model is the curvature probe `(dr/2) |T| + 2 eps_g / dr`.
/// `|T|` comes from [`estimate_third`]. A missing estimate uses the
/// same clamp fallback as [`forward_step`].
pub fn gradient_dr(eps_g: f64, third: Option<f64>, point_scale: f64) -> f64 {
    clamped_step(eps_g, third, 2.0, point_scale)
}

fn clamped_step(eps: f64, scale: Option<f64>, factor: f64, point_scale: f64) -> f64 {
    let lo = lower_clamp(point_scale);
    if !eps.is_finite() || eps < 0.0 {
        return lo;
    }
    if eps == 0.0 {
        return lo;
    }
    let Some(scale) = scale.filter(|s| s.is_finite()) else {
        return STEP_CLAMP_MAX;
    };
    let denom = scale.abs();
    if denom == 0.0 {
        return STEP_CLAMP_MAX;
    }
    (factor * (eps / denom).sqrt()).clamp(lo, STEP_CLAMP_MAX)
}

fn lower_clamp(point_scale: f64) -> f64 {
    let scale = if point_scale.is_finite() {
        point_scale.abs()
    } else {
        0.0
    };
    STEP_CLAMP_MIN.max(4.0 * ulp(scale))
}

fn ulp(x: f64) -> f64 {
    let x = x.abs();
    if x == 0.0 || !x.is_finite() {
        return f64::from_bits(1);
    }
    let bits = x.to_bits();
    if bits >= 0x7fefffffffffffff {
        return f64::INFINITY;
    }
    f64::from_bits(bits + 1) - x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::{Control, NoiseFloor};
    use crate::linesearch::LineSearch;
    use crate::method::Method;
    use crate::minimize::minimize_method;
    use crate::oracle::Oracle;
    use crate::report::Status;
    use approx::assert_relative_eq;
    use eindir_core::DifferentiableObjective;
    use ndarray::array;

    #[test]
    fn wide_range_reports_step_too_large() {
        let samples: Vec<f64> = (0..7).map(|i| 1.0 + i as f64 / 6.0).collect();
        let est = ecnoise(&samples).unwrap();
        assert_eq!(est.ok, NoiseFlag::StepTooLarge);
        assert_eq!(est.ok.inform(), 3);
        assert_eq!(est.eps, 0.0);
        assert_eq!(est.level, vec![0.0; 6]);
    }

    #[test]
    fn constant_samples_report_step_too_small() {
        let est = ecnoise(&[3.0; 7]).unwrap();
        assert_eq!(est.ok, NoiseFlag::StepTooSmall);
        assert_eq!(est.ok.inform(), 2);
        assert_eq!(est.eps, 0.0);
        assert!(est.level.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn half_of_first_differences_zero_is_step_too_small() {
        let est = ecnoise(&[1.0, 1.0, 1.0, 1.0001]).unwrap();
        assert_eq!(est.ok, NoiseFlag::StepTooSmall);
        assert_eq!(est.eps, 0.0);
    }

    #[test]
    fn alternating_values_detect_noise_at_first_order() {
        let samples = [1.0, 1.01, 1.0, 1.01, 1.0, 1.01, 1.0];
        let est = ecnoise(&samples).unwrap();
        assert_eq!(est.ok, NoiseFlag::Detected);
        assert_relative_eq!(est.eps, 0.007071067811865481, epsilon = 1e-15);
        assert_relative_eq!(est.eps, est.level[0], epsilon = 0.0);
        let expected = [
            0.007071067811865481,
            0.008164965809277266,
            0.008944271909999166,
            0.009561828874675158,
            0.010079052613579402,
            0.010527227119356313,
        ];
        for (got, want) in est.level.iter().zip(expected) {
            assert_relative_eq!(*got, want, epsilon = 1e-15);
        }
    }

    #[test]
    fn smooth_quadratic_reports_roundoff() {
        let samples: Vec<f64> = (0..7)
            .map(|i| {
                let t = i as f64 * 1e-3;
                1.0 + t * t
            })
            .collect();
        let est = ecnoise(&samples).unwrap();
        assert_eq!(est.ok, NoiseFlag::Detected);
        assert_relative_eq!(est.eps, 3.510833468576701e-17, epsilon = 1e-30);
        assert_relative_eq!(est.eps, est.level[2], epsilon = 0.0);
    }

    #[test]
    fn declared_value_skips_the_table() {
        let est = NoiseEstimate::declared(1.25e-4);
        assert_eq!(est.ok, NoiseFlag::Declared);
        assert_eq!(est.ok.inform(), 0);
        assert_eq!(est.eps, 1.25e-4);
        assert!(est.level.is_empty());
    }

    #[test]
    fn short_or_non_finite_samples_are_refused() {
        assert!(ecnoise(&[1.0, 2.0, 3.0]).is_err());
        assert!(ecnoise(&[1.0, 1.0, 1.0, f64::NAN]).is_err());
    }

    #[test]
    fn spacing_is_the_distance_between_samples() {
        let obj = Oracle::unbounded(2, |x| {
            let f = x[0] + 2.0 * x[1];
            (f, array![1.0, 2.0])
        });
        let x = array![0.3, -0.1];
        let direction = array![3.0, 4.0];
        let h = 1e-4;
        let m = 4;
        let values = line_values(&obj, x.view(), direction.view(), h, m).unwrap();
        assert_eq!(values.len(), m + 1);
        assert_relative_eq!(values[m / 2], obj.eval(x.view()), epsilon = 1e-15);
        let unit = array![0.6, 0.8];
        let start = &x + &((-2.0 * h) * &unit);
        let end = &x + &((2.0 * h) * &unit);
        assert_relative_eq!(values[0], obj.eval(start.view()), epsilon = 1e-12);
        assert_relative_eq!(values[m], obj.eval(end.view()), epsilon = 1e-12);
        let gap = (values[1] - values[0]) / h;
        let slope = unit[0] + 2.0 * unit[1];
        assert_relative_eq!(gap, slope, epsilon = 1e-8);
        let est = sample_noise(&obj, x.view(), direction.view(), h, m).unwrap();
        let again = ecnoise(&values).unwrap();
        assert_eq!(est.ok, again.ok);
        assert_eq!(est.eps.to_bits(), again.eps.to_bits());
    }

    #[test]
    fn flags_fire_on_a_smooth_function() {
        let obj = Oracle::unbounded(1, |x| (x[0].exp(), array![x[0].exp()]));
        let x = array![0.0];
        let u = array![1.0];
        let tiny = sample_noise(&obj, x.view(), u.view(), 1e-30, DEFAULT_M).unwrap();
        assert_eq!(tiny.ok, NoiseFlag::StepTooSmall);
        let wide = sample_noise(&obj, x.view(), u.view(), 1.0, DEFAULT_M).unwrap();
        assert_eq!(wide.ok, NoiseFlag::StepTooLarge);
        let grown = next_spacing(1e-30, tiny.ok).unwrap();
        assert!((grown / 1e-30 - 100.0).abs() < 1e-6);
        assert_eq!(next_spacing(1.0, wide.ok), Some(0.01));
    }

    #[test]
    fn step_formulas_match_the_toms_minimizers() {
        let eps = 1e-8;
        let second = 4.0;
        let h = forward_step(eps, Some(second), 0.0);
        let expect = 8.0_f64.powf(0.25) * (eps / second).sqrt();
        assert_relative_eq!(h, expect, epsilon = 1e-15);
        let third = 2.5;
        let dr = gradient_dr(eps, Some(third), 0.0);
        assert_relative_eq!(dr, 2.0 * (eps / third).sqrt(), epsilon = 1e-15);
        assert_eq!(forward_step(0.0, Some(1.0), 0.0), STEP_CLAMP_MIN);
        assert_eq!(gradient_dr(1.0, Some(0.0), 0.0), STEP_CLAMP_MAX);
        assert_eq!(forward_step(1e6, Some(1e-18), 0.0), STEP_CLAMP_MAX);
        assert_eq!(forward_step(1e-8, None, 0.0), STEP_CLAMP_MAX);
        assert_eq!(gradient_dr(0.0, None, 0.0), STEP_CLAMP_MIN);
        let near_ten = forward_step(0.0, Some(1.0), 10.0);
        let four_ulps = 4.0 * ulp(10.0);
        assert!(four_ulps > STEP_CLAMP_MIN);
        assert_eq!(near_ten, four_ulps);
    }

    #[test]
    fn relative_noise_is_recovered_within_a_factor_of_two() {
        let levels = [1e-3, 1e-6, 1e-9, 1e-12];
        for &level in &levels {
            let mut ratios = Vec::new();
            let mut hits = 0usize;
            for seed in 0..200u64 {
                let Some(eps) = adapt_noise(seed, level) else {
                    continue;
                };
                let ratio = eps / (level * noise_scale());
                ratios.push(ratio);
                if (0.5..=2.0).contains(&ratio) {
                    hits += 1;
                }
            }
            let n = ratios.len();
            let mean = ratios.iter().sum::<f64>() / n.max(1) as f64;
            let var = ratios.iter().map(|r| (r - mean) * (r - mean)).sum::<f64>() / n.max(1) as f64;
            assert!(
                hits * 10 >= 200 * 9,
                "level {level}: {hits}/200 within factor 2 ({n} detected), mean ratio {mean}, std {}",
                var.sqrt()
            );
        }
    }

    fn adapt_noise(seed: u64, level: f64) -> Option<f64> {
        let obj = NoisySmooth { seed, level };
        let x = array![0.2];
        let u = array![1.0];
        // Spacing where the first difference of cos + sin sits near the
        // noise, so the table reads the noise rather than the slope.
        // Retries follow the published 100 h and h/100 steps.
        let mut h = (level * noise_scale() / 0.8).clamp(1e-8, 1e-3);
        for _ in 0..8 {
            let est = sample_noise(&obj, x.view(), u.view(), h, DEFAULT_M).unwrap();
            if est.ok == NoiseFlag::Detected {
                return Some(est.eps);
            }
            h = next_spacing(h, est.ok)?;
            if !h.is_finite() || h <= 0.0 {
                return None;
            }
        }
        None
    }

    #[test]
    fn f32_lj_gradient_noise_tracks_roundoff_times_force() {
        let x = array![0.0, 0.0, 0.0, 1.2, 0.1, -0.05];
        let u = array![1.0, 0.0, 0.0, -1.0, 0.0, 0.0];
        let f64_force = Lj { f32_arith: false };
        let f32_force = Lj { f32_arith: true };
        let g = f64_force.grad(x.view());
        let scale = g.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        assert!(scale > 0.1, "force scale {scale}");
        let h = 1e-6;
        let eps64 = sample_gradient_noise(&f64_force, x.view(), u.view(), h, DEFAULT_M, 7)
            .unwrap()
            .eps;
        let eps32 = sample_gradient_noise(&f32_force, x.view(), u.view(), h, DEFAULT_M, 7)
            .unwrap()
            .eps;
        let expect = (f32::EPSILON as f64) * scale;
        assert!(
            eps32 > eps64 * 10.0,
            "eps32 {eps32} eps64 {eps64} expect {expect}"
        );
        assert!(
            eps32 > expect * 1e-4 && eps32 < expect * 1e4,
            "eps32 {eps32} expect {expect} scale {scale}"
        );
    }

    #[test]
    fn noise_floor_never_fires_on_a_noise_free_quadratic() {
        let obj = Oracle::unbounded(1, |x| (0.5 * x[0] * x[0], array![x[0]]));
        let est = sample_gradient_noise(
            &obj,
            array![0.3].view(),
            array![1.0].view(),
            1e-6,
            DEFAULT_M,
            2,
        )
        .unwrap();
        let eps = if est.ok == NoiseFlag::Detected {
            est.eps
        } else {
            0.0
        };
        let control = Control {
            maxiter: 40,
            gtol: 1e-6,
            istep: 0.5,
            maxmove: None,
            ftol_rel: None,
            noise_floor: Some(NoiseFloor::new(eps)),
        };
        let report = minimize_method(
            &obj,
            array![0.4],
            &control,
            Method::Steepest,
            LineSearch::default(),
        )
        .unwrap();
        assert_ne!(report.status, Status::NoiseFloor);
        assert_eq!(report.status, Status::Tolerance);
        assert!(report.grad_norm < control.gtol);

        let mut loose = control.with_noise_floor(NoiseFloor::new(1e-2));
        loose.gtol = 1e-12;
        loose.istep = 0.99;
        // A full Newton step would land under gtol. Backtracking keeps
        // the opening trial, which leaves the gradient inside
        // c * sqrt(n_free) * eps_g. Here n_free is 1. That trial is
        // longer than the default per-atom cap, so a Euclidean cap above
        // it turns the per-atom default off and leaves the trial alone.
        loose.maxmove = Some(10.0);
        let early = minimize_method(
            &obj,
            array![1.5],
            &loose,
            Method::Steepest,
            LineSearch::Backtracking {
                c: 1e-4,
                beta: 0.5,
                maxiter: 8,
            },
        )
        .unwrap();
        assert_eq!(early.status, Status::NoiseFloor);
        assert!(early.grad_norm < 3.0 * 1e-2);
        assert!(early.grad_norm >= loose.gtol);
    }

    #[test]
    fn floor_tracks_sqrt_n_for_isotropic_and_anisotropic_noise() {
        for &n in &[3usize, 21, 114] {
            let sigma = 1e-3;
            let mut hits = 0usize;
            for seed in 0..200u64 {
                if floor_fires(&vec![sigma; n], seed, false, 8, 1.7).0 {
                    hits += 1;
                }
            }
            assert!(
                hits * 10 >= 200 * 9,
                "isotropic n={n}: {hits}/200 below 2 sqrt(n) sigma"
            );
            let mut exact = 0usize;
            for seed in 0..200u64 {
                if exact_eps_fires(n, sigma, seed, 2.0) {
                    exact += 1;
                }
            }
            // An exact eps_g at 2 sqrt(tr Σ) fires on 93% of draws at
            // n = 3 and on all of them at n = 21 and n = 114. The
            // measured-eps shortfall is the ECnoise spread.
            if n == 3 {
                assert!(
                    exact * 100 >= 200 * 90,
                    "exact eps n=3: {exact}/200 at 2 sqrt(tr)"
                );
            } else {
                assert_eq!(exact, 200, "exact eps n={n}: {exact}/200 at 2 sqrt(tr)");
            }
            eprintln!("exact eps n={n}: {exact}/200 at 2 sqrt(tr Σ)");
            assert!(
                !floor_fires(&vec![0.0; n], 1, true, 8, 1.7).0,
                "noise-free quadratic n={n} reported the floor"
            );
        }
        let n = 21usize;
        let sigmas: Vec<f64> = (0..n)
            .map(|i| 1e-4 * 100.0_f64.powf(i as f64 / (n - 1) as f64))
            .collect();
        let mut hits = 0usize;
        for seed in 0..200u64 {
            if floor_fires(&sigmas, seed, false, 8, 1.7).0 {
                hits += 1;
            }
        }
        assert!(
            hits * 10 >= 200 * 9,
            "anisotropic k=8: {hits}/200 below 2 sqrt(tr Σ)"
        );
        let mut once = 0usize;
        for seed in 0..200u64 {
            if floor_fires(&sigmas, seed, false, 1, 1.7).0 {
                once += 1;
            }
        }
        // A 100-fold σ spread sends one direction's factor
        // sqrt(n wᵀΣw / tr Σ) from about 0.56 to 1.45, so k = 1
        // misses the 90% bar that k = 8 clears.
        eprintln!("anisotropic k=1: {once}/200 at 1.7 sqrt(tr Σ)");
    }

    #[test]
    fn floor_stays_off_at_five_standard_deviations() {
        for &k in &[1usize, 8] {
            for &n in &[3usize, 21, 114] {
                let sigma = 1e-3;
                let mut hits = 0usize;
                for seed in 0..200u64 {
                    if floor_fires(&vec![sigma; n], seed, false, k, 5.0).0 {
                        hits += 1;
                    }
                }
                eprintln!("stays-off isotropic n={n} k={k}: {hits}/200");
                assert!(
                    hits * 100 <= 200 * 5,
                    "isotropic n={n} k={k}: {hits}/200 fired at 5 sqrt(tr)"
                );
            }
            let n = 21usize;
            let sigmas: Vec<f64> = (0..n)
                .map(|i| 1e-4 * 100.0_f64.powf(i as f64 / (n - 1) as f64))
                .collect();
            let mut hits = 0usize;
            for seed in 0..200u64 {
                if floor_fires(&sigmas, seed, false, k, 5.0).0 {
                    hits += 1;
                }
            }
            eprintln!("stays-off anisotropic k={k}: {hits}/200");
            assert!(
                hits * 100 <= 200 * 5,
                "anisotropic k={k}: {hits}/200 fired at 5 sqrt(tr)"
            );
        }
    }

    #[test]
    fn energy_floor_stops_when_no_trial_clears_eps_f() {
        let quad = NoisyQuad::new(vec![0.0], 1);
        let linesearch = LineSearch::Wolfe {
            c1: 1e-4,
            c2: 0.9,
            maxiter: 12,
        };
        let stalled = Control {
            gtol: 1e-8,
            maxiter: 20,
            noise_floor: Some(NoiseFloor::new(0.0).with_eps_f(1.0)),
            ..Control::default()
        };
        let report = minimize_method(
            &quad,
            array![0.5],
            &stalled,
            Method::Lbfgs { memory: 8 },
            linesearch,
        )
        .unwrap();
        assert_eq!(report.status, Status::NoiseFloor);
        assert!(report.steps <= 2, "steps {}", report.steps);
        assert!((report.coords[0] - 0.5).abs() < 1e-12);
        let fine = Control {
            gtol: 1e-6,
            maxiter: 40,
            noise_floor: Some(NoiseFloor::new(0.0).with_eps_f(1e-8)),
            ..Control::default()
        };
        let report = minimize_method(
            &quad,
            array![0.5],
            &fine,
            Method::Lbfgs { memory: 8 },
            linesearch,
        )
        .unwrap();
        assert_eq!(report.status, Status::Tolerance);
    }

    fn exact_eps_fires(n: usize, sigma: f64, seed: u64, radius_factor: f64) -> bool {
        let tr = n as f64 * sigma * sigma;
        let obj = NoisyQuad::new(vec![sigma; n], seed);
        let mut x = Array1::zeros(n);
        x[0] = radius_factor * tr.sqrt();
        let g = obj.grad(x.view());
        let gnorm = dot_ordered(g.view(), g.view()).sqrt();
        let control = Control {
            gtol: 1e-18,
            noise_floor: Some(NoiseFloor::new(sigma)),
            ..Control::default()
        };
        control.classify(gnorm, n) == Status::NoiseFloor
    }

    fn floor_fires(
        sigmas: &[f64],
        seed: u64,
        noise_free_minimize: bool,
        k: usize,
        radius_factor: f64,
    ) -> (bool, f64, f64) {
        let n = sigmas.len();
        let obj = NoisyQuad::new(sigmas.to_vec(), seed);
        if noise_free_minimize {
            let x = Array1::ones(n);
            let est =
                sample_gradient_noise(&obj, x.view(), Array1::ones(n).view(), 1e-6, DEFAULT_M, 3)
                    .unwrap();
            let eps = if est.ok == NoiseFlag::Detected {
                est.eps
            } else {
                0.0
            };
            let gnorm = (n as f64).sqrt();
            let control = Control {
                gtol: 1e-8,
                maxiter: 40,
                noise_floor: Some(NoiseFloor::new(eps)),
                ..Control::default()
            };
            if control.classify(gnorm, n) == Status::NoiseFloor {
                return (true, eps, gnorm);
            }
            let report =
                minimize_method(&obj, x, &control, Method::Steepest, LineSearch::default())
                    .unwrap();
            return (report.status == Status::NoiseFloor, eps, gnorm);
        }
        let tr: f64 = sigmas.iter().map(|s| s * s).sum();
        // The firing check passes 1.7. An exact eps at 2 sqrt(tr Σ)
        // fires on 93% of draws at n = 3 and on all of them at n = 21
        // and n = 114. The measured-eps shortfall is the spread of the
        // ECnoise estimate, not the chi-square tail of ||g||. The
        // stays-off check passes 5.
        let radius = radius_factor * tr.sqrt();
        let mut x = Array1::zeros(n);
        x[0] = radius;
        // Measure where the projection is large compared with the noise.
        // At the floor itself the samples differ in the leading digit and
        // ECnoise refuses the table.
        let anchor = Array1::from_elem(n, 0.5);
        let est = sample_gradient_noise_k(
            &obj,
            anchor.view(),
            Array1::ones(n).view(),
            1e-5,
            DEFAULT_M,
            seed.wrapping_add(17),
            k,
        )
        .unwrap();
        let eps = est.eps;
        let g = obj.grad(x.view());
        let gnorm = dot_ordered(g.view(), g.view()).sqrt();
        let control = Control {
            gtol: 1e-18,
            noise_floor: Some(NoiseFloor::new(eps)),
            ..Control::default()
        };
        (control.classify(gnorm, n) == Status::NoiseFloor, eps, gnorm)
    }

    #[test]
    fn second_derivative_and_forward_step_match_the_grid() {
        let levels = [1e-6_f64, 1e-8, 1e-10];
        let cases = [
            (ScalarKind::Sin, 0.7_f64),
            (ScalarKind::Exp, 0.3),
            (ScalarKind::Quartic, 0.8),
        ];
        for &level in &levels {
            for &(kind, x0) in &cases {
                let truth = kind.second(x0).abs();
                let grid: Vec<f64> = (0..=20)
                    .map(|i| 10.0_f64.powf(-6.0 + i as f64 * 0.25))
                    .collect();
                let mut curv_hits = 0usize;
                let mut star_sq = 0.0;
                let mut grid_sq = vec![0.0; grid.len()];
                let mut counted = 0usize;
                for seed in 0..200u64 {
                    let obj = NoisyScalar::new(seed, level, kind);
                    let x = array![x0];
                    let mu = estimate_second(&obj, x.view(), array![1.0].view(), level);
                    let Some(mu) = mu else { continue };
                    if (0.5..=2.0).contains(&(mu / truth)) {
                        curv_hits += 1;
                    }
                    let h_star = forward_step(level, Some(mu), x0.abs());
                    let err = forward_error(&obj, kind, x0, h_star);
                    star_sq += err * err;
                    for (i, &h) in grid.iter().enumerate() {
                        let e = forward_error(&obj, kind, x0, h);
                        grid_sq[i] += e * e;
                    }
                    counted += 1;
                }
                assert!(
                    curv_hits * 10 >= 200 * 9,
                    "{kind:?} level {level}: curvature {curv_hits}/200"
                );
                let rms_star = (star_sq / counted.max(1) as f64).sqrt();
                let rms_best = grid_sq
                    .iter()
                    .map(|s| (s / counted.max(1) as f64).sqrt())
                    .fold(f64::INFINITY, f64::min);
                assert!(
                    rms_star <= 2.0 * rms_best.max(1e-18),
                    "{kind:?} level {level}: rms derivative error {rms_star} vs best grid {rms_best}"
                );
            }
        }
    }

    fn forward_error(obj: &NoisyScalar, kind: ScalarKind, x0: f64, h: f64) -> f64 {
        let fp = obj.eval(array![x0 + h].view());
        let f0 = obj.eval(array![x0].view());
        ((fp - f0) / h - kind.deriv(x0)).abs()
    }

    #[test]
    fn counter_direction_depends_on_the_seed_and_not_on_call_order() {
        let a = counter_direction(11, 4);
        let b = counter_direction(11, 4);
        let c = counter_direction(12, 4);
        for i in 0..4 {
            assert_eq!(a[i].to_bits(), b[i].to_bits());
        }
        assert!(
            a.iter()
                .zip(c.iter())
                .any(|(p, q)| p.to_bits() != q.to_bits())
        );
        let nrm = dot_ordered(a.view(), a.view());
        assert_relative_eq!(nrm, 1.0, epsilon = 1e-12);
    }

    #[cfg(feature = "par")]
    #[test]
    fn eps_is_bit_identical_at_one_two_three_and_eight_threads() {
        let obj = Oracle::unbounded(6, |x| {
            let mut f = 0.0;
            let mut g = ndarray::Array1::zeros(6);
            for i in 0..6 {
                f += (x[i] - 0.1 * i as f64).sin();
                g[i] = (x[i] - 0.1 * i as f64).cos();
            }
            (f, g)
        });
        let x = array![0.2, -0.1, 0.4, 0.0, 0.3, -0.2];
        let u = array![1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                sample_gradient_noise(&obj, x.view(), u.view(), 1e-5, DEFAULT_M, 99).unwrap()
            })
        };
        let base = run(1);
        for n in [2usize, 3, 8] {
            let got = run(n);
            assert_eq!(got.ok, base.ok);
            assert_eq!(got.eps.to_bits(), base.eps.to_bits(), "threads {n}");
            assert_eq!(got.level.len(), base.level.len());
            for (a, b) in got.level.iter().zip(&base.level) {
                assert_eq!(a.to_bits(), b.to_bits(), "threads {n}");
            }
        }
    }

    struct NoisySmooth {
        seed: u64,
        level: f64,
    }

    impl Objective<f64> for NoisySmooth {
        fn dim(&self) -> usize {
            1
        }
        fn bounds(&self) -> &eindir_core::Bounds<f64> {
            use std::sync::OnceLock;
            static B: OnceLock<eindir_core::Bounds<f64>> = OnceLock::new();
            B.get_or_init(|| eindir_core::Bounds::new(array![-10.0], array![10.0], 0.0))
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            let t = x[0];
            let smooth = t.cos() + t.sin();
            smooth
                + self.level
                    * noise_scale()
                    * normal(self.seed ^ 0xA5A5_A5A5_A5A5_A5A5, point_index(t))
        }
    }

    impl Gradient<f64> for NoisySmooth {
        fn dim(&self) -> usize {
            1
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            array![-x[0].sin() + x[0].cos()]
        }
    }

    fn noise_scale() -> f64 {
        (0.2_f64.cos() + 0.2_f64.sin()).abs()
    }

    fn point_index(t: f64) -> usize {
        t.to_bits() as usize
    }

    struct Lj {
        f32_arith: bool,
    }

    impl Objective<f64> for Lj {
        fn dim(&self) -> usize {
            6
        }
        fn bounds(&self) -> &eindir_core::Bounds<f64> {
            use std::sync::OnceLock;
            static B: OnceLock<eindir_core::Bounds<f64>> = OnceLock::new();
            B.get_or_init(|| {
                eindir_core::Bounds::new(
                    Array1::from_elem(6, -10.0),
                    Array1::from_elem(6, 10.0),
                    0.0,
                )
            })
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            lj_pair(x, self.f32_arith).0
        }
    }

    impl Gradient<f64> for Lj {
        fn dim(&self) -> usize {
            6
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            Array1::from_vec(lj_pair(x, self.f32_arith).1)
        }
    }

    fn lj_pair(x: ArrayView1<f64>, f32_arith: bool) -> (f64, Vec<f64>) {
        let mut dx = x[3] - x[0];
        let mut dy = x[4] - x[1];
        let mut dz = x[5] - x[2];
        if f32_arith {
            dx = (dx as f32) as f64;
            dy = (dy as f32) as f64;
            dz = (dz as f32) as f64;
        }
        let r2 = dx * dx + dy * dy + dz * dz;
        let (e, df) = if f32_arith {
            let r2 = r2 as f32;
            let inv = 1.0 / r2;
            let r6 = inv * inv * inv;
            let e = 4.0 * (r6 * r6 - r6);
            let df = 24.0 * inv * (2.0 * r6 * r6 - r6);
            (e as f64, df as f64)
        } else {
            let inv = 1.0 / r2;
            let r6 = inv * inv * inv;
            let e = 4.0 * (r6 * r6 - r6);
            let df = 24.0 * inv * (2.0 * r6 * r6 - r6);
            (e, df)
        };
        let mut g = vec![0.0; 6];
        g[0] = -df * dx;
        g[1] = -df * dy;
        g[2] = -df * dz;
        g[3] = df * dx;
        g[4] = df * dy;
        g[5] = df * dz;
        if f32_arith {
            for v in &mut g {
                *v = (*v as f32) as f64;
            }
        }
        (e, g)
    }

    struct NoisyQuad {
        n: usize,
        sigmas: Vec<f64>,
        seed: u64,
        bounds: eindir_core::Bounds<f64>,
    }

    impl NoisyQuad {
        fn new(sigmas: Vec<f64>, seed: u64) -> Self {
            let n = sigmas.len();
            Self {
                n,
                sigmas,
                seed,
                bounds: eindir_core::Bounds::new(
                    Array1::from_elem(n, -10.0),
                    Array1::from_elem(n, 10.0),
                    0.0,
                ),
            }
        }
    }

    impl Objective<f64> for NoisyQuad {
        fn dim(&self) -> usize {
            self.n
        }
        fn bounds(&self) -> &eindir_core::Bounds<f64> {
            &self.bounds
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            0.5 * dot_ordered(x, x)
        }
    }

    impl DifferentiableObjective<f64> for NoisyQuad {}

    impl Gradient<f64> for NoisyQuad {
        fn dim(&self) -> usize {
            self.n
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            let key = quad_key(x);
            let mut g = x.to_owned();
            for i in 0..self.n {
                if self.sigmas[i] != 0.0 {
                    g[i] += self.sigmas[i] * normal(self.seed ^ key, i);
                }
            }
            g
        }
    }

    fn quad_key(x: ArrayView1<f64>) -> u64 {
        let mut z = 0u64;
        for v in x.iter() {
            z = z.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(v.to_bits());
        }
        z
    }

    #[derive(Clone, Copy, Debug)]
    enum ScalarKind {
        Sin,
        Exp,
        Quartic,
    }

    impl ScalarKind {
        fn value(self, t: f64) -> f64 {
            match self {
                Self::Sin => t.sin(),
                Self::Exp => t.exp(),
                Self::Quartic => t.powi(4),
            }
        }
        fn deriv(self, t: f64) -> f64 {
            match self {
                Self::Sin => t.cos(),
                Self::Exp => t.exp(),
                Self::Quartic => 4.0 * t.powi(3),
            }
        }
        fn second(self, t: f64) -> f64 {
            match self {
                Self::Sin => -t.sin(),
                Self::Exp => t.exp(),
                Self::Quartic => 12.0 * t * t,
            }
        }
    }

    struct NoisyScalar {
        seed: u64,
        level: f64,
        kind: ScalarKind,
        bounds: eindir_core::Bounds<f64>,
    }

    impl NoisyScalar {
        fn new(seed: u64, level: f64, kind: ScalarKind) -> Self {
            Self {
                seed,
                level,
                kind,
                bounds: eindir_core::Bounds::new(array![-4.0], array![4.0], 0.0),
            }
        }
    }

    impl Objective<f64> for NoisyScalar {
        fn dim(&self) -> usize {
            1
        }
        fn bounds(&self) -> &eindir_core::Bounds<f64> {
            &self.bounds
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            let t = x[0];
            self.kind.value(t) + self.level * normal(self.seed, t.to_bits() as usize)
        }
    }
}
