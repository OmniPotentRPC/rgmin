//! Computational noise of a scalar function.
//!
//! Moré and Wild, *Estimating computational noise*,
//! SIAM J. Sci. Comput. 33 (2011), 1292-1314,
//! <https://doi.org/10.1137/100786125>.
//!
//! [`ecnoise`] takes `nf >= 4` values of a univariate function at
//! equally spaced points and builds the forward-difference table.
//! The level-`k` estimate is
//! `σ_k = sqrt(γ_k * mean_i (Δ^k f_i)^2)` with
//! `γ_k = (k!)^2 / (2k)!`. The returned noise is the smallest `k`
//! whose column changes sign and whose three estimates
//! `σ_k`, `σ_{k+1}`, `σ_{k+2}` lie within a factor of 4.
//! Seven samples are the count the paper recommends.
//!
//! [`sample_noise`] evaluates an eindir objective on the segment
//! `x + s h p` with unit `p` and `s` running from -1 to 1, then
//! calls [`ecnoise`]. [`EcNoiseStatus::StepTooSmall`] is the paper's
//! signal to repeat the sample at `100 h`, and
//! [`EcNoiseStatus::StepTooLarge`] to repeat it at `h / 100`.

use eindir_core::Objective;
use ndarray::{Array1, ArrayView1};

use crate::error::{Error, Result};

/// Why [`ecnoise`] accepted or refused a noise level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EcNoiseStatus {
    /// Three consecutive levels agree within a factor of 4 and the
    /// selected column changes sign. Published `inform = 1`.
    Detected,
    /// At least half of the first differences are zero, so the
    /// spacing does not move the function. Published `inform = 2`.
    /// The next spacing in the reference code is `100 h`.
    StepTooSmall,
    /// The samples differ in the leading digit, or no column meets
    /// the agreement test. Published `inform = 3`.
    /// The next spacing in the reference code is `h / 100`.
    StepTooLarge,
}

impl EcNoiseStatus {
    /// Published `inform` code: 1 detected, 2 grow the spacing, 3 shrink it.
    pub fn inform(self) -> i32 {
        match self {
            Self::Detected => 1,
            Self::StepTooSmall => 2,
            Self::StepTooLarge => 3,
        }
    }
}

/// Noise level and the per-order estimates that produced it.
#[derive(Clone, Debug, PartialEq)]
pub struct EcNoiseEstimate {
    /// Accepted `σ_k`, or 0 when the status is not [`EcNoiseStatus::Detected`].
    pub noise: f64,
    /// `σ_k` for `k = 1 .. nf-1`, in that order.
    /// Entries stay 0 when the call returns before the table is filled.
    pub levels: Vec<f64>,
    /// Detection outcome.
    pub status: EcNoiseStatus,
}

/// Line sample of an objective and the noise [`ecnoise`] reads off it.
#[derive(Clone, Debug, PartialEq)]
pub struct SampledNoise {
    /// Noise of the sampled values.
    pub estimate: EcNoiseEstimate,
    /// `m + 1` objective values, equally spaced in the line parameter.
    pub values: Vec<f64>,
    /// Unit direction of the sample.
    pub direction: Array1<f64>,
    /// [`EcNoiseEstimate::noise`] divided by the center sample `values[m / 2]`.
    pub relative: f64,
}

/// Estimate the noise level of `samples`, values of `f` at equal spacing.
///
/// `samples.len()` is `nf` and must be at least 4. Every entry must be
/// finite. The slice is left unchanged. On [`EcNoiseStatus::Detected`],
/// [`EcNoiseEstimate::noise`] is the lowest accepted `σ_k`.
pub fn ecnoise(samples: &[f64]) -> Result<EcNoiseEstimate> {
    let nf = samples.len();
    if nf < 4 {
        return Err(Error::EcNoise {
            what: "need at least 4 samples",
        });
    }
    if samples.iter().any(|v| !v.is_finite()) {
        return Err(Error::EcNoise {
            what: "sample was not finite",
        });
    }

    let mut levels = vec![0.0; nf - 1];
    let fmin = samples.iter().copied().fold(f64::INFINITY, f64::min);
    let fmax = samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let scale = fmax.abs().max(fmin.abs());
    if scale != 0.0 && (fmax - fmin) / scale > 0.1 {
        return Ok(undetected(levels, EcNoiseStatus::StepTooLarge));
    }

    let mut diffs = samples.to_vec();
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
                return Ok(undetected(levels, EcNoiseStatus::StepTooSmall));
            }
        }
        let jf = j as f64;
        gamma *= 0.5 * (jf / (2.0 * jf - 1.0));
        let active = &diffs[..count];
        let mean_sq = active.iter().map(|t| t * t).sum::<f64>() / count as f64;
        levels[j - 1] = (gamma * mean_sq).sqrt();
        let emin = active.iter().copied().fold(f64::INFINITY, f64::min);
        let emax = active.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if emin * emax < 0.0 {
            sign_change[j - 1] = true;
        }
    }

    for k in 0..nf - 3 {
        let window = &levels[k..k + 3];
        let emin = window.iter().copied().fold(f64::INFINITY, f64::min);
        let emax = window.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if emax <= 4.0 * emin && sign_change[k] {
            let noise = levels[k];
            return Ok(EcNoiseEstimate {
                noise,
                levels,
                status: EcNoiseStatus::Detected,
            });
        }
    }
    Ok(undetected(levels, EcNoiseStatus::StepTooLarge))
}

fn undetected(levels: Vec<f64>, status: EcNoiseStatus) -> EcNoiseEstimate {
    EcNoiseEstimate {
        noise: 0.0,
        levels,
        status,
    }
}

/// Sample `objective` at `m + 1` points from `x - h p` to `x + h p` and
/// estimate the noise.
///
/// `direction` is scaled to unit length. `m` is the number of extra
/// evaluations beyond the base point, so the table has `m + 1` entries
/// and `m` must be at least 3. The center sample sits at index `m / 2`,
/// which is `x` when `m` is even. `h` is the distance from the center
/// to either end.
pub fn sample_noise<O>(
    objective: &O,
    x: ArrayView1<f64>,
    direction: ArrayView1<f64>,
    h: f64,
    m: usize,
) -> Result<SampledNoise>
where
    O: Objective<f64> + ?Sized,
{
    if m < 3 {
        return Err(Error::EcNoise {
            what: "need at least 3 extra evaluations",
        });
    }
    if !h.is_finite() || h <= 0.0 {
        return Err(Error::EcNoise {
            what: "step must be finite and positive",
        });
    }
    if direction.len() != x.len() {
        return Err(Error::EcNoise {
            what: "direction length does not match the point",
        });
    }
    if x.len() != objective.dim() {
        return Err(Error::EcNoise {
            what: "point length does not match the objective",
        });
    }
    if direction.iter().any(|v| !v.is_finite()) {
        return Err(Error::EcNoise {
            what: "direction entry was not finite",
        });
    }
    let nrm = direction.dot(&direction).sqrt();
    if nrm == 0.0 {
        return Err(Error::EcNoise {
            what: "sampling direction has zero length",
        });
    }
    let unit = &direction / nrm;
    let mid = m / 2 + 1;
    let mut values = Vec::with_capacity(m + 1);
    for i in 0..=m {
        let s = 2.0 * ((i + 1) as f64 - mid as f64) / m as f64;
        let point = &x + &(s * h * &unit);
        let value = objective.eval(point.view());
        if !value.is_finite() {
            return Err(Error::EcNoise {
                what: "objective value was not finite",
            });
        }
        values.push(value);
    }
    let estimate = ecnoise(&values)?;
    let relative = estimate.noise / values[m / 2];
    Ok(SampledNoise {
        estimate,
        values,
        direction: unit,
        relative,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn wide_range_reports_step_too_large() {
        let samples: Vec<f64> = (0..7).map(|i| 1.0 + i as f64 / 6.0).collect();
        let est = ecnoise(&samples).unwrap();
        assert_eq!(est.status, EcNoiseStatus::StepTooLarge);
        assert_eq!(est.status.inform(), 3);
        assert_eq!(est.noise, 0.0);
        assert_eq!(est.levels, vec![0.0; 6]);
    }

    #[test]
    fn constant_samples_report_step_too_small() {
        let est = ecnoise(&[3.0; 7]).unwrap();
        assert_eq!(est.status, EcNoiseStatus::StepTooSmall);
        assert_eq!(est.status.inform(), 2);
        assert_eq!(est.noise, 0.0);
        assert!(est.levels.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn half_of_first_differences_zero_is_step_too_small() {
        let est = ecnoise(&[1.0, 1.0, 1.0, 1.0001]).unwrap();
        assert_eq!(est.status, EcNoiseStatus::StepTooSmall);
        assert_eq!(est.noise, 0.0);
    }

    #[test]
    fn alternating_values_detect_noise_at_first_order() {
        let samples = [1.0, 1.01, 1.0, 1.01, 1.0, 1.01, 1.0];
        let est = ecnoise(&samples).unwrap();
        assert_eq!(est.status, EcNoiseStatus::Detected);
        assert_eq!(est.status.inform(), 1);
        assert_relative_eq!(est.noise, 0.007071067811865481, epsilon = 1e-15);
        assert_relative_eq!(est.noise, est.levels[0], epsilon = 0.0);
        let expected = [
            0.007071067811865481,
            0.008164965809277266,
            0.008944271909999166,
            0.009561828874675158,
            0.010079052613579402,
            0.010527227119356313,
        ];
        for (got, want) in est.levels.iter().zip(expected) {
            assert_relative_eq!(*got, want, epsilon = 1e-15);
        }
        assert!(est.noise > 0.0);
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
        assert_eq!(est.status, EcNoiseStatus::Detected);
        assert_relative_eq!(est.noise, 3.510833468576701e-17, epsilon = 1e-30);
        assert_relative_eq!(est.noise, est.levels[2], epsilon = 0.0);
    }

    #[test]
    fn short_or_non_finite_samples_are_refused() {
        assert!(ecnoise(&[1.0, 2.0, 3.0]).is_err());
        assert!(ecnoise(&[1.0, 1.0, 1.0, f64::NAN]).is_err());
    }

    #[test]
    fn sample_noise_places_the_center_on_the_base_point() {
        let obj = crate::oracle::Oracle::unbounded(2, |x| {
            let f = x[0] + 2.0 * x[1];
            (f, array![1.0, 2.0])
        });
        let x = array![0.3, -0.1];
        let direction = array![3.0, 4.0];
        let h = 1e-4;
        let m = 4;
        let probed = sample_noise(&obj, x.view(), direction.view(), h, m).unwrap();
        assert_relative_eq!(probed.direction[0], 0.6, epsilon = 1e-15);
        assert_relative_eq!(probed.direction[1], 0.8, epsilon = 1e-15);
        assert_eq!(probed.values.len(), m + 1);
        let center = obj.eval(x.view());
        assert_relative_eq!(probed.values[m / 2], center, epsilon = 1e-15);
        let end = &x + &(h * &probed.direction);
        let start = &x - &(h * &probed.direction);
        assert_relative_eq!(probed.values[0], obj.eval(start.view()), epsilon = 1e-12);
        assert_relative_eq!(probed.values[m], obj.eval(end.view()), epsilon = 1e-12);
        let again = ecnoise(&probed.values).unwrap();
        assert_eq!(again.status, probed.estimate.status);
        assert_relative_eq!(again.noise, probed.estimate.noise, epsilon = 0.0);
        assert_relative_eq!(
            probed.relative,
            probed.estimate.noise / probed.values[m / 2],
            epsilon = 0.0
        );
    }

    #[test]
    fn sample_noise_refuses_a_bad_probe() {
        let obj = crate::oracle::Oracle::unbounded(1, |x| (x[0] * x[0], array![2.0 * x[0]]));
        let x = array![1.0];
        let p = array![1.0];
        assert!(sample_noise(&obj, x.view(), p.view(), 0.0, 4).is_err());
        assert!(sample_noise(&obj, x.view(), p.view(), 1e-6, 2).is_err());
        assert!(sample_noise(&obj, x.view(), array![0.0].view(), 1e-6, 4).is_err());
        assert!(sample_noise(&obj, x.view(), array![1.0, 0.0].view(), 1e-6, 4).is_err());
    }
}
