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

use eindir_core::{Gradient, Objective};
use ndarray::{Array1, ArrayView1};

use crate::error::{Error, Result};
use crate::vecops::dot_ordered;

/// Extra samples beyond the center. The table length is `m + 1`.
pub const DEFAULT_M: usize = 8;

/// Lower clamp on [`forward_step`] and [`gradient_dr`], in the caller's length unit.
pub const STEP_CLAMP_MIN: f64 = 1e-16;
/// Upper clamp on [`forward_step`] and [`gradient_dr`], in the caller's length unit.
pub const STEP_CLAMP_MAX: f64 = 1.0;

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

/// Noise of `g(x + i h u) · w`.
///
/// `w` is one unit vector. Component `i` is a standard normal from a
/// counter-based stream keyed by `(seed, 2i)` and `(seed, 2i+1)`, then
/// the vector is normalized with [`dot_ordered`]. The same `w` is used
/// at every stencil point. The projection is also a [`dot_ordered`] sum.
/// `h` is the sample spacing, as in [`sample_noise`].
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
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
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

/// Median absolute divided difference of `order` on samples spaced by `h`.
///
/// Order 2 is `|f''|` along the line. Order 3 is the third-derivative
/// scale `|T|` used by [`gradient_dr`]. The column is the raw table, so
/// a spacing that [`ecnoise`] already flags as noise-dominated does not
/// yield a derivative.
pub fn table_derivative(values: &[f64], h: f64, order: usize) -> Option<f64> {
    if !(h.is_finite() && h > 0.0) || values.len() <= order || order == 0 || order > 3 {
        return None;
    }
    let mut mags = Vec::with_capacity(values.len() - order);
    for i in 0..=values.len() - order - 1 {
        let d = match order {
            1 => values[i + 1] - values[i],
            2 => values[i] - 2.0 * values[i + 1] + values[i + 2],
            _ => -values[i] + 3.0 * values[i + 1] - 3.0 * values[i + 2] + values[i + 3],
        };
        mags.push((d / h.powi(order as i32)).abs());
    }
    if mags.is_empty() {
        return None;
    }
    mags.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(mags[mags.len() / 2])
}

/// Forward-difference step `h* = 8^(1/4) sqrt(eps_f / |f''|)`.
///
/// The model is a one-sided difference whose error is about
/// `(h/2) |f''| + sqrt(2) eps_f / h`. That expression is smallest at
/// this `h`, when `f''` is the second derivative at the expansion
/// point and `eps_f` is the standard deviation of uncorrelated noise
/// in `f`. The result is clamped to `[STEP_CLAMP_MIN, STEP_CLAMP_MAX]`.
/// A zero `|f''|` with positive noise returns the upper clamp. A zero
/// noise returns the lower clamp.
pub fn forward_step(eps_f: f64, second: f64) -> f64 {
    clamped_step(eps_f, second, 8.0_f64.powf(0.25))
}

/// Gradient-difference step `dr* = 2 sqrt(eps_g / |T|)`.
///
/// The model is the curvature probe `(dr/2) |T| + 2 eps_g / dr`.
/// `|T|` is a third-derivative scale, from [`table_derivative`] at
/// order 3 on the same samples that produced `eps_g`. The clamp is
/// the one documented on [`forward_step`].
pub fn gradient_dr(eps_g: f64, third: f64) -> f64 {
    clamped_step(eps_g, third, 2.0)
}

fn clamped_step(eps: f64, scale: f64, factor: f64) -> f64 {
    if !eps.is_finite() || eps < 0.0 || !scale.is_finite() {
        return STEP_CLAMP_MIN;
    }
    if eps == 0.0 {
        return STEP_CLAMP_MIN;
    }
    let denom = scale.abs();
    if denom == 0.0 {
        return STEP_CLAMP_MAX;
    }
    (factor * (eps / denom).sqrt()).clamp(STEP_CLAMP_MIN, STEP_CLAMP_MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::Control;
    use crate::linesearch::LineSearch;
    use crate::method::Method;
    use crate::minimize::minimize_method;
    use crate::oracle::Oracle;
    use crate::report::Status;
    use approx::assert_relative_eq;
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
        let h = forward_step(eps, second);
        let expect = 8.0_f64.powf(0.25) * (eps / second).sqrt();
        assert_relative_eq!(h, expect, epsilon = 1e-15);
        let third = 2.5;
        let dr = gradient_dr(eps, third);
        assert_relative_eq!(dr, 2.0 * (eps / third).sqrt(), epsilon = 1e-15);
        assert_eq!(forward_step(0.0, 1.0), STEP_CLAMP_MIN);
        assert_eq!(gradient_dr(1.0, 0.0), STEP_CLAMP_MAX);
        assert_eq!(forward_step(1e6, 1e-18), STEP_CLAMP_MAX);
    }

    #[test]
    fn table_derivative_of_a_cubic() {
        let h = 1e-3;
        let values: Vec<f64> = (0..9)
            .map(|i| {
                let t = (i as f64 - 4.0) * h;
                t + 0.5 * t * t + (1.0 / 6.0) * t * t * t
            })
            .collect();
        let second = table_derivative(&values, h, 2).unwrap();
        let third = table_derivative(&values, h, 3).unwrap();
        assert_relative_eq!(second, 1.0, epsilon = 1e-6);
        assert_relative_eq!(third, 1.0, epsilon = 1e-6);
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
            noise_eps: Some(eps),
            noise_c: 3.0,
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

        let mut loose = control.with_noise_eps(1e-2).noise_floor(3.0);
        loose.gtol = 1e-12;
        loose.istep = 0.99;
        // A full Newton step would land under gtol. Backtracking keeps
        // the opening trial, which leaves the gradient inside c * eps_g.
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
}
