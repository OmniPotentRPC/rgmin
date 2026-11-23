//! Symmetric secant updates of a dense Hessian model.
//!
//! `B` is a symmetric `n x n` matrix, row-major. A pair `(s, y)` is a step
//! and its gradient change; every update here returns a symmetric `B+` with
//! `B+ s = y` when it changes `B`. With `r = y - B s`:
//!
//! - Powell symmetric Broyden (PSB):
//!   `B+ = B + (r sᵀ + s rᵀ)/(sᵀs) - (rᵀs) s sᵀ/(sᵀs)²`.
//! - Symmetric rank one (SR1): `B+ = B + r rᵀ/(rᵀs)`, skipped when
//!   `|rᵀs| < skip ‖r‖ ‖s‖` (Nocedal and Wright, 2nd ed., eq. 6.26).
//! - Bofill: `φ SR1 + (1 - φ) PSB` with `φ = (rᵀs)²/(‖r‖² ‖s‖²)`. The SR1
//!   part is then `(rᵀs) r rᵀ/(‖r‖² ‖s‖²)`, which has no small divisor, so
//!   the update is never skipped. Neither PSB nor Bofill keeps `B` definite,
//!   which is why they are used for saddle-point and instanton searches.
//!
//! No test compares `s` or `y` with an absolute size: every update is
//! invariant under `(s, y) -> (α s, α y)`. The step and the residual are
//! normalised to unit vectors `u = s/‖s‖` and `v = r/‖r‖`, the cosine
//! `c = (rᵀs/‖r‖)/‖s‖` stays `O(1)`, and the outer products are scaled once
//! by `‖r‖/‖s‖`. A long step with a tiny model, or a short step with a huge
//! model, therefore cannot overflow or underflow a squared Gram factor.
//! Deciding whether a pair is above the noise of the gradient is the
//! caller's job.
//!
//! No intermediate squares a norm and no intermediate forms `‖s‖` or `‖r‖`
//! itself: the step is first rescaled by a power of two to a peak in
//! `[0.5, 1)`, and `‖r‖/‖s‖` is evaluated from the mantissas and exponents of
//! the two peaks. A step or residual with components near `1e300` therefore
//! gives the same decision as the same pair scaled to `O(1)`, and the pair is
//! refused only when the correction itself is not a finite, nonzero `f64`.
//! A pair that cannot produce a finite correction returns
//! [`SecantError::NonFinite`](crate::secant::SecantError::NonFinite) or
//! [`SecantOutcome::Skipped`](crate::secant::SecantOutcome::Skipped) and leaves `B`
//! unchanged.

use thiserror::Error;

/// Which symmetric secant formula [`update_symmetric`] applies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SecantKind {
    /// Powell symmetric Broyden.
    Psb,
    /// Symmetric rank one with the relative skip test.
    ///
    /// Build this variant with [`SecantKind::sr1`]. The variant is
    /// non-exhaustive, so a struct literal cannot bypass the `[0, 1)` check
    /// on the skip threshold.
    #[non_exhaustive]
    Sr1 {
        /// The update is skipped when `|rᵀs| < skip ‖r‖ ‖s‖`.
        skip_bits: u64,
    },
    /// Bofill's convex combination of SR1 and PSB.
    Bofill,
}

impl SecantKind {
    /// SR1 with skip threshold `skip`, which must be finite and in `[0, 1)`.
    pub fn sr1(skip: f64) -> Result<Self, SecantError> {
        if !(skip.is_finite() && (0.0..1.0).contains(&skip)) {
            return Err(SecantError::Skip(skip));
        }
        Ok(Self::Sr1 {
            skip_bits: skip.to_bits(),
        })
    }
}

/// What an update did to `B`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SecantOutcome {
    /// `B` changed and now satisfies `B s = y`.
    Applied,
    /// `s` is exactly zero, so the pair carries no curvature.
    ZeroStep,
    /// `B s = y` already holds exactly; `B` is unchanged.
    Satisfied,
    /// SR1's denominator failed the relative skip test; `B` is unchanged.
    Skipped,
}

/// A refused secant update. `B` is never modified on error.
#[derive(Debug, Error, Clone, PartialEq)]
#[non_exhaustive]
pub enum SecantError {
    /// `B` is not `n x n` or a vector is not of length `n`.
    #[error("secant update dimensions: B has {matrix}, s has {step}, y has {change}, n = {n}")]
    Dim {
        /// Side of the model.
        n: usize,
        /// Length of `B`.
        matrix: usize,
        /// Length of `s`.
        step: usize,
        /// Length of `y`.
        change: usize,
    },
    /// An entry of `B`, `s` or `y`, or an intermediate, is not finite.
    #[error("secant update input is not finite")]
    NonFinite,
    /// The SR1 skip threshold is not finite or not in `[0, 1)`.
    #[error("SR1 skip threshold {0} is not in [0, 1)")]
    Skip(f64),
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `2^k` for `k` in `[-1022, 1023]`, built from its bit pattern so it is exact.
fn pow2(k: i32) -> f64 {
    debug_assert!((-1022..=1023).contains(&k));
    f64::from_bits(((1023 + i64::from(k)) as u64) << 52)
}

/// `(m, e)` with `x = m 2^e` and `m` in `[0.5, 1)`, for finite `x > 0`.
fn frexp(x: f64) -> (f64, i32) {
    debug_assert!(x.is_finite() && x > 0.0);
    // Subnormals are lifted into the normal range first; the lift is exact.
    let (x, lift) = if x < f64::MIN_POSITIVE {
        (x * pow2(54), 54)
    } else {
        (x, 0)
    };
    let e = ((x.to_bits() >> 52) & 0x7ff) as i32 - 1022;
    (ldexp(x, -e), e - lift)
}

/// `x 2^k` without forming a power of two outside the `f64` range.
///
/// The shift is applied in steps of at most `2^1000`. A step never takes a
/// normal number into the subnormals unless the final answer is subnormal
/// or zero, so the result is rounded once. Overflow gives an infinity and
/// full underflow gives zero, as a single multiplication would.
fn ldexp(x: f64, k: i32) -> f64 {
    let mut x = x;
    let mut k = k.clamp(-4400, 4400);
    while k > 1000 {
        x *= pow2(1000);
        k -= 1000;
    }
    while k < -1000 {
        x *= pow2(-1000);
        k += 1000;
    }
    x * pow2(k)
}

/// Peak-scaled direction of `x`.
///
/// `peak` is `‖x‖_∞` and `local` is `‖x‖₂/‖x‖_∞`, so `‖x‖₂ = peak * local`
/// without squaring a huge or tiny entry. The product itself is never
/// formed: `‖x‖₂` may exceed the largest finite `f64` while a ratio of two
/// such norms is still representable. `u` is `x/‖x‖₂`, formed as
/// `(x/peak)/local`, which is unchanged when `x` is scaled by a power of two.
struct Unit {
    peak: f64,
    local: f64,
    u: Vec<f64>,
}

/// `Ok(None)` is the zero vector. A non-finite entry is [`SecantError::NonFinite`].
fn unit_vector(x: &[f64]) -> Result<Option<Unit>, SecantError> {
    let mut peak = 0.0_f64;
    for v in x {
        peak = peak.max(v.abs());
    }
    if peak == 0.0 {
        return Ok(None);
    }
    if !peak.is_finite() {
        return Err(SecantError::NonFinite);
    }
    let mut sumsq = 0.0_f64;
    for v in x {
        let z = v / peak;
        sumsq += z * z;
    }
    let local = sumsq.sqrt();
    if !local.is_finite() || local == 0.0 {
        return Err(SecantError::NonFinite);
    }
    let u = x.iter().map(|v| (v / peak) / local).collect();
    Ok(Some(Unit { peak, local, u }))
}

/// `‖num‖/‖den‖`, compared in exponent space.
///
/// Each peak is split into a mantissa in `[0.5, 1)` and a power of two.
/// The mantissas and the local factors, which are all `O(1)`, are divided
/// directly and the difference of the exponents is applied once at the end.
/// Neither `peak_num/peak_den` nor `‖num‖²` is formed, so a finite ratio is
/// reported as finite however large or small the two norms are, and the
/// result is invariant under a common power-of-two scaling of both vectors.
fn norm_ratio(num: &Unit, den: &Unit) -> f64 {
    let (mn, en) = frexp(num.peak);
    let (md, ed) = frexp(den.peak);
    ldexp((mn / md) * (num.local / den.local), en - ed)
}

/// Every entry of `x` times `2^k`.
fn scale_pow2(x: &[f64], k: i32) -> Vec<f64> {
    x.iter().map(|v| ldexp(*v, k)).collect()
}

/// Coefficients of `v vᵀ`, `v uᵀ + u vᵀ` and `u uᵀ`, in units of `‖r‖/‖s‖`.
/// `None` is SR1's relative skip.
///
/// They are `O(1)` for PSB and Bofill. The caller multiplies the finished
/// `O(1)` sum of rank-one terms by `‖r‖/‖s‖` once, so a correction that fits
/// in an `f64` is not pushed out of range by a partial product.
fn unit_coefficients(kind: SecantKind, c: f64) -> Result<Option<(f64, f64, f64)>, SecantError> {
    let weights = match kind {
        SecantKind::Psb => (0.0, 1.0, -c),
        SecantKind::Sr1 { skip_bits } => {
            let skip = f64::from_bits(skip_bits);
            // `|rᵀs| < skip ‖r‖ ‖s‖` is `|c| < skip` once both vectors are units.
            if c == 0.0 || c.abs() < skip {
                return Ok(None);
            }
            (1.0 / c, 0.0, 0.0)
        }
        SecantKind::Bofill => {
            // `φ = c²`. The SR1 weight is `φ / c = c`, so a vanishing cosine
            // drops that term instead of dividing by it.
            let psb = 1.0 - c * c;
            (c, psb, -(psb * c))
        }
    };
    if weights.0.is_finite() && weights.1.is_finite() && weights.2.is_finite() {
        Ok(Some(weights))
    } else {
        Err(SecantError::NonFinite)
    }
}

/// Apply one symmetric secant update to the row-major model `b`.
///
/// `b` is read as symmetric: `B s` uses the rows of `b`. The increment
/// added to entry `(i, j)` is computed from the same products as the one
/// added to `(j, i)`, so a symmetric `b` stays bitwise symmetric.
pub fn update_symmetric(
    kind: SecantKind,
    b: &mut [f64],
    n: usize,
    s: &[f64],
    y: &[f64],
) -> Result<SecantOutcome, SecantError> {
    if n.checked_mul(n) != Some(b.len()) || s.len() != n || y.len() != n {
        return Err(SecantError::Dim {
            n,
            matrix: b.len(),
            step: s.len(),
            change: y.len(),
        });
    }
    if b.iter().chain(s).chain(y).any(|v| !v.is_finite()) {
        return Err(SecantError::NonFinite);
    }
    let Some(raw_step) = unit_vector(s)? else {
        return Ok(SecantOutcome::ZeroStep);
    };
    // The update is invariant under `(s, y) -> (α s, α y)`. A power of two is
    // exact, so `s` is brought to a peak in `[0.5, 1)` before `B s` is formed:
    // a step near `1e300` against a model near `1e-300`, or the reverse, then
    // multiplies `O(1)` numbers. `y` follows `s`, and so does `r`, which
    // leaves `‖r‖/‖s‖` and the cosine unchanged.
    let shift = -frexp(raw_step.peak).1;
    let s_scaled = scale_pow2(s, shift);
    let y_scaled = scale_pow2(y, shift);
    let step = Unit {
        peak: ldexp(raw_step.peak, shift),
        ..raw_step
    };
    if y_scaled.iter().any(|v| !v.is_finite()) {
        return Err(SecantError::NonFinite);
    }
    let r: Vec<f64> = (0..n)
        .map(|i| y_scaled[i] - dot(&b[i * n..(i + 1) * n], &s_scaled))
        .collect();
    if r.iter().any(|v| !v.is_finite()) {
        return Err(SecantError::NonFinite);
    }
    let Some(residual) = unit_vector(&r)? else {
        // `r` vanished. That is an exact fit only if the scaling did not
        // flush a nonzero entry of `y` to zero; otherwise the residual per
        // unit step is below the smallest `f64`, the correction cannot be
        // represented, and the pair is refused.
        let flushed = y
            .iter()
            .zip(&y_scaled)
            .any(|(raw, scaled)| *raw != 0.0 && *scaled == 0.0);
        return if flushed {
            Err(SecantError::NonFinite)
        } else {
            Ok(SecantOutcome::Satisfied)
        };
    };
    // `c = (rᵀs / ‖r‖) / ‖s‖` and `scale = ‖r‖ / ‖s‖`.
    let c = dot(&step.u, &residual.u);
    let scale = norm_ratio(&residual, &step);
    if !c.is_finite() || !scale.is_finite() || scale == 0.0 {
        return Err(SecantError::NonFinite);
    }
    let Some((vv, vu, uu)) = unit_coefficients(kind, c)? else {
        return Ok(SecantOutcome::Skipped);
    };
    let mut next = b.to_vec();
    for i in 0..n {
        for j in 0..n {
            let unit = vv * (residual.u[i] * residual.u[j])
                + vu * (residual.u[i] * step.u[j] + step.u[i] * residual.u[j])
                + uu * (step.u[i] * step.u[j]);
            next[i * n + j] += scale * unit;
        }
    }
    if next.iter().any(|v| !v.is_finite()) {
        return Err(SecantError::NonFinite);
    }
    b.copy_from_slice(&next);
    Ok(SecantOutcome::Applied)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matvec(b: &[f64], n: usize, s: &[f64]) -> Vec<f64> {
        (0..n).map(|i| dot(&b[i * n..(i + 1) * n], s)).collect()
    }

    /// A fixed symmetric indefinite `A`, one negative eigenvalue.
    fn target(n: usize) -> Vec<f64> {
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let base = 1.0 / (1.0 + (i as f64 - j as f64).abs());
                a[i * n + j] = if i == j { base + i as f64 } else { 0.3 * base };
            }
        }
        a[0] = -2.0;
        a
    }

    fn step(n: usize, seed: usize) -> Vec<f64> {
        (0..n)
            .map(|i| (((i + 3) * (seed + 5) * 7919) % 101) as f64 / 101.0 - 0.5)
            .collect()
    }

    fn kinds() -> Vec<SecantKind> {
        vec![
            SecantKind::Psb,
            SecantKind::sr1(1e-8).unwrap(),
            SecantKind::Bofill,
        ]
    }

    fn residual_bound(b: &[f64], n: usize, s: &[f64], y: &[f64]) -> f64 {
        let scale = b.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
            * s.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
            + y.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        (8 * n * n) as f64 * f64::EPSILON * scale
    }

    #[test]
    fn every_update_meets_the_secant_condition_and_stays_bitwise_symmetric() {
        let n = 6;
        let a = target(n);
        for kind in kinds() {
            let mut b = vec![0.0; n * n];
            for i in 0..n {
                b[i * n + i] = 1.0;
            }
            for seed in 0..4 {
                let s = step(n, seed);
                let y = matvec(&a, n, &s);
                let outcome = update_symmetric(kind, &mut b, n, &s, &y).unwrap();
                assert_eq!(outcome, SecantOutcome::Applied, "{kind:?} seed {seed}");
                let bs = matvec(&b, n, &s);
                let bound = residual_bound(&b, n, &s, &y);
                for i in 0..n {
                    assert!(
                        (bs[i] - y[i]).abs() <= bound,
                        "{kind:?} B s {} y {}",
                        bs[i],
                        y[i]
                    );
                    for j in 0..n {
                        assert_eq!(b[i * n + j].to_bits(), b[j * n + i].to_bits());
                    }
                }
            }
        }
    }

    #[test]
    fn sr1_recovers_a_quadratic_hessian_from_n_independent_steps() {
        // Nocedal and Wright, Theorem 6.1: with n independent steps and no
        // skip, SR1 on a quadratic returns its Hessian.
        let n = 5;
        let a = target(n);
        let mut b = vec![0.0; n * n];
        for k in 0..n {
            let s: Vec<f64> = (0..n)
                .map(|i| {
                    if i == k {
                        1.0
                    } else {
                        0.1 * ((i + 2 * k) as f64).sin()
                    }
                })
                .collect();
            let y = matvec(&a, n, &s);
            let outcome = update_symmetric(SecantKind::sr1(1e-8).unwrap(), &mut b, n, &s, &y);
            assert_eq!(outcome, Ok(SecantOutcome::Applied), "step {k}");
        }
        let scale = a.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        for i in 0..n * n {
            assert!(
                (b[i] - a[i]).abs() <= 1e-10 * scale,
                "entry {i}: {} vs {}",
                b[i],
                a[i]
            );
        }
    }

    #[test]
    fn updates_are_invariant_under_a_common_scale_of_the_pair() {
        let n = 4;
        let a = target(n);
        let s = step(n, 1);
        let y = matvec(&a, n, &s);
        for kind in kinds() {
            let mut unit = vec![0.0; n * n];
            let mut tiny = vec![0.0; n * n];
            for i in 0..n {
                unit[i * n + i] = 1.0;
                tiny[i * n + i] = 1.0;
            }
            let alpha = 2.0_f64.powi(-40);
            let s_tiny: Vec<f64> = s.iter().map(|v| alpha * v).collect();
            let y_tiny: Vec<f64> = y.iter().map(|v| alpha * v).collect();
            assert_eq!(
                update_symmetric(kind, &mut unit, n, &s, &y),
                Ok(SecantOutcome::Applied)
            );
            assert_eq!(
                update_symmetric(kind, &mut tiny, n, &s_tiny, &y_tiny),
                Ok(SecantOutcome::Applied)
            );
            // A power-of-two scale is exact, so every update must agree to the bit.
            let bits = |m: &[f64]| m.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&unit), bits(&tiny));
        }
    }

    #[test]
    fn degenerate_pairs_leave_the_model_unchanged() {
        let n = 3;
        let a = target(n);
        let mut b = a.clone();
        let s = step(n, 2);
        let y = matvec(&a, n, &s);
        for kind in kinds() {
            assert_eq!(
                update_symmetric(kind, &mut b, n, &[0.0; 3], &y),
                Ok(SecantOutcome::ZeroStep)
            );
        }
        // An exact model gives r = 0 only if B s reproduces y bit for bit.
        let mut exact = vec![0.0; n * n];
        for i in 0..n {
            exact[i * n + i] = 2.0;
        }
        let s2 = [1.0, 0.5, 0.25];
        let y2 = [2.0, 1.0, 0.5];
        for kind in kinds() {
            assert_eq!(
                update_symmetric(kind, &mut exact, n, &s2, &y2),
                Ok(SecantOutcome::Satisfied)
            );
        }
        // r orthogonal to s: SR1 skips, Bofill reduces to PSB.
        let mut sr1 = vec![0.0; n * n];
        let mut bofill = vec![0.0; n * n];
        let mut psb = vec![0.0; n * n];
        let s3 = [1.0, 0.0, 0.0];
        let y3 = [0.0, 1.0, 0.0];
        assert_eq!(
            update_symmetric(SecantKind::sr1(1e-8).unwrap(), &mut sr1, n, &s3, &y3),
            Ok(SecantOutcome::Skipped)
        );
        assert!(sr1.iter().all(|v| *v == 0.0));
        update_symmetric(SecantKind::Bofill, &mut bofill, n, &s3, &y3).unwrap();
        update_symmetric(SecantKind::Psb, &mut psb, n, &s3, &y3).unwrap();
        assert_eq!(bofill, psb);
        assert_eq!(b, a);
    }

    #[test]
    fn invalid_input_is_refused_without_touching_the_model() {
        let n = 2;
        let mut b = vec![1.0, 0.0, 0.0, 1.0];
        assert!(matches!(
            update_symmetric(SecantKind::Psb, &mut b, n, &[1.0], &[1.0, 2.0]),
            Err(SecantError::Dim { .. })
        ));
        assert_eq!(
            update_symmetric(SecantKind::Psb, &mut b, n, &[1.0, f64::NAN], &[1.0, 2.0]),
            Err(SecantError::NonFinite)
        );
        assert_eq!(
            update_symmetric(
                SecantKind::Psb,
                &mut b,
                n,
                &[f64::INFINITY, 1.0],
                &[1.0, 2.0]
            ),
            Err(SecantError::NonFinite)
        );
        assert_eq!(
            update_symmetric(SecantKind::Psb, &mut b, n, &[1.0, 1.0], &[1.0, f64::NAN]),
            Err(SecantError::NonFinite)
        );
        let mut bad_model = vec![1.0, f64::INFINITY, f64::INFINITY, 1.0];
        let before = bad_model.clone();
        assert_eq!(
            update_symmetric(SecantKind::Psb, &mut bad_model, n, &[1.0, 1.0], &[1.0, 2.0]),
            Err(SecantError::NonFinite)
        );
        assert_eq!(bad_model, before);
        assert_eq!(b, vec![1.0, 0.0, 0.0, 1.0]);
        assert!(matches!(SecantKind::sr1(1.0), Err(SecantError::Skip(_))));
        assert!(matches!(
            SecantKind::sr1(f64::NAN),
            Err(SecantError::Skip(_))
        ));
    }

    #[test]
    fn extreme_scales_keep_the_secant_residual_at_roundoff() {
        let n = 4;
        let a = target(n);
        let s0 = step(n, 1);
        let y0 = matvec(&a, n, &s0);
        for scale in [1e100, 1e-80] {
            let s: Vec<f64> = s0.iter().map(|v| v * scale).collect();
            let y: Vec<f64> = y0.iter().map(|v| v * scale).collect();
            for kind in kinds() {
                let mut b = vec![0.0; n * n];
                for i in 0..n {
                    b[i * n + i] = 1.0;
                }
                let outcome = update_symmetric(kind, &mut b, n, &s, &y);
                assert_eq!(
                    outcome,
                    Ok(SecantOutcome::Applied),
                    "{kind:?} scale {scale:e}"
                );
                let bs = matvec(&b, n, &s);
                let bound = residual_bound(&b, n, &s, &y);
                for i in 0..n {
                    let residual = (bs[i] - y[i]).abs();
                    assert!(
                        residual <= bound,
                        "{kind:?} scale {scale:e} component {i}: residual {residual:e} bound {bound:e}"
                    );
                }
            }
        }
    }

    /// `‖got - want‖ / ‖want‖`, with both vectors scaled by `‖want‖_∞` so a
    /// gradient change near `1e80` or `1e-80` does not overflow the squares.
    fn relative_residual(got: &[f64], want: &[f64]) -> f64 {
        let peak = want.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let mut num = 0.0_f64;
        let mut den = 0.0_f64;
        for (g, w) in got.iter().zip(want) {
            let diff = (g - w) / peak;
            let scaled = w / peak;
            num += diff * diff;
            den += scaled * scaled;
        }
        num.sqrt() / den.sqrt()
    }

    #[test]
    fn mismatched_step_and_model_scales_meet_the_secant_condition() {
        let n = 4;
        let a = target(n);
        let s0 = step(n, 1);
        let y0 = matvec(&a, n, &s0);
        // Step near 1e100 with a model near 1e-180 used to underflow the
        // second division by `sᵀs` and return Applied with a large residual.
        // Step near 1e-80 with a model near 1e160 used to overflow that
        // quotient and return NonFinite.
        let cases = [(1.0e100, 1.0e-180), (1.0e-80, 1.0e160)];
        for &(step_scale, model_scale) in &cases {
            let s: Vec<f64> = s0.iter().map(|v| v * step_scale).collect();
            let y: Vec<f64> = y0.iter().map(|v| v * step_scale * model_scale).collect();
            for kind in kinds() {
                let mut b = vec![0.0; n * n];
                for i in 0..n {
                    b[i * n + i] = model_scale;
                }
                let outcome = update_symmetric(kind, &mut b, n, &s, &y);
                assert_eq!(
                    outcome,
                    Ok(SecantOutcome::Applied),
                    "{kind:?} step {step_scale:e} model {model_scale:e}"
                );
                let bs = matvec(&b, n, &s);
                let relative = relative_residual(&bs, &y);
                assert!(
                    relative <= 1e-10,
                    "{kind:?} step {step_scale:e} model {model_scale:e}: relative residual {relative:e}"
                );
                for i in 0..n {
                    for j in 0..n {
                        assert_eq!(b[i * n + j].to_bits(), b[j * n + i].to_bits());
                    }
                }
            }
        }
    }

    fn identity(n: usize) -> Vec<f64> {
        let mut b = vec![0.0; n * n];
        for i in 0..n {
            b[i * n + i] = 1.0;
        }
        b
    }

    fn unit_with(peak: f64, local: f64) -> Unit {
        Unit {
            peak,
            local,
            u: Vec::new(),
        }
    }

    #[test]
    fn frexp_and_ldexp_round_trip_across_the_exponent_range() {
        for x in [
            f64::MAX,
            1.0e300,
            0.75,
            1.0,
            1.0e-300,
            f64::MIN_POSITIVE,
            1.0e-310,
            5.0e-324,
        ] {
            let (m, e) = frexp(x);
            assert!((0.5..1.0).contains(&m), "mantissa {m} of {x:e}");
            assert_eq!(ldexp(m, e).to_bits(), x.to_bits(), "{x:e}");
        }
        assert_eq!(ldexp(1.0, 5000), f64::INFINITY);
        assert_eq!(ldexp(1.0, -5000), 0.0);
        assert_eq!(ldexp(0.5, 1024), 2.0_f64.powi(1023));
    }

    #[test]
    fn norm_ratio_stays_finite_when_only_the_peak_quotient_overflows() {
        // `‖num‖ = MAX` and `‖den‖ = 2 * 0.9`: the quotient of the peaks is
        // `MAX / 0.9`, which overflows, while the quotient of the norms is
        // below `MAX`.
        let num = unit_with(f64::MAX, 1.0);
        let den = unit_with(0.9, 2.0);
        let naive = (num.peak / den.peak) * (num.local / den.local);
        assert!(!naive.is_finite());
        let ratio = norm_ratio(&num, &den);
        assert!(ratio.is_finite());
        let want = (f64::MAX / 1.8).to_bits();
        assert!(ratio.to_bits().abs_diff(want) <= 1, "{ratio:e}");
    }

    #[test]
    fn norm_ratio_is_invariant_under_a_common_power_of_two() {
        let base = norm_ratio(&unit_with(0.75, 1.5), &unit_with(0.625, 1.25));
        for k in [-1000, -500, 0, 500, 1000] {
            let num = unit_with(ldexp(0.75, k), 1.5);
            let den = unit_with(ldexp(0.625, k), 1.25);
            assert_eq!(norm_ratio(&num, &den).to_bits(), base.to_bits(), "k {k}");
        }
    }

    #[test]
    fn large_and_tiny_finite_norms_give_accepted_finite_updates() {
        let n = 4;
        let a = target(n);
        let s0 = step(n, 1);
        let y0 = matvec(&a, n, &s0);
        for scale in [1e154, 1e200, 1e250, 1e300, 1e-200, 1e-300] {
            let s: Vec<f64> = s0.iter().map(|v| v * scale).collect();
            let y: Vec<f64> = y0.iter().map(|v| v * scale).collect();
            for kind in kinds() {
                let mut b = identity(n);
                let outcome = update_symmetric(kind, &mut b, n, &s, &y);
                assert_eq!(
                    outcome,
                    Ok(SecantOutcome::Applied),
                    "{kind:?} scale {scale:e}"
                );
                assert!(b.iter().all(|v| v.is_finite()), "{kind:?} {scale:e}");
                let relative = relative_residual(&matvec(&b, n, &s), &y);
                assert!(
                    relative <= 1e-10,
                    "{kind:?} scale {scale:e}: relative residual {relative:e}"
                );
                for i in 0..n {
                    for j in 0..n {
                        assert_eq!(b[i * n + j].to_bits(), b[j * n + i].to_bits());
                    }
                }
            }
        }
    }

    #[test]
    fn large_and_tiny_norms_give_the_same_update_as_the_unit_pair() {
        let n = 4;
        let a = target(n);
        let s0 = step(n, 2);
        let y0 = matvec(&a, n, &s0);
        for kind in kinds() {
            let mut reference = identity(n);
            update_symmetric(kind, &mut reference, n, &s0, &y0).unwrap();
            for k in [-996, -500, 500, 996] {
                let s: Vec<f64> = s0.iter().map(|v| ldexp(*v, k)).collect();
                let y: Vec<f64> = y0.iter().map(|v| ldexp(*v, k)).collect();
                let mut b = identity(n);
                assert_eq!(
                    update_symmetric(kind, &mut b, n, &s, &y),
                    Ok(SecantOutcome::Applied),
                    "{kind:?} 2^{k}"
                );
                let bits = |m: &[f64]| m.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&b), bits(&reference), "{kind:?} 2^{k}");
            }
        }
    }

    #[test]
    fn a_step_whose_squared_norm_overflows_is_accepted_when_the_correction_is_finite() {
        // `sᵀs` is `2e600`, and `B s` against `y` leaves `r ≈ -s`, so
        // `‖r‖/‖s‖` is about one.
        let n = 2;
        let s = [1e300, 1e300];
        let y = [1.0, 2.0];
        for kind in kinds() {
            let mut b = identity(n);
            assert_eq!(
                update_symmetric(kind, &mut b, n, &s, &y),
                Ok(SecantOutcome::Applied),
                "{kind:?}"
            );
            assert!(b.iter().all(|v| v.is_finite()));
            assert_eq!(b[1].to_bits(), b[2].to_bits());
            let bs = matvec(&b, n, &s);
            let bound = residual_bound(&b, n, &s, &y);
            for i in 0..n {
                assert!((bs[i] - y[i]).abs() <= bound, "{kind:?} {} {}", bs[i], y[i]);
            }
        }
    }

    #[test]
    fn a_model_that_overflows_against_an_unscaled_step_is_accepted() {
        // `B s` is about `1e310` for the raw step, but `r / ‖s‖` is `1e10`.
        let n = 2;
        let s = [1e300, 1e300];
        let y = [0.0, 0.0];
        let mut b = vec![1e10, 0.0, 0.0, 1e10];
        assert_eq!(
            update_symmetric(SecantKind::Psb, &mut b, n, &s, &y),
            Ok(SecantOutcome::Applied)
        );
        assert!(b.iter().all(|v| v.is_finite()));
        let bs = matvec(&b, n, &[1.0, 1.0]);
        assert!(bs.iter().all(|v| v.abs() <= 1e-3), "{bs:?}");
    }

    #[test]
    fn a_correction_just_below_the_largest_double_is_accepted_by_psb() {
        // `‖r‖/‖s‖` is `1.7e308`: the quotient of the peaks, `3.4e308`,
        // overflows even though the correction itself is finite.
        let n = 2;
        let s = [0.5, 0.0];
        let y = [0.85e308, 0.0];
        let mut b = vec![0.0; n * n];
        assert_eq!(
            update_symmetric(SecantKind::Psb, &mut b, n, &s, &y),
            Ok(SecantOutcome::Applied)
        );
        assert!(b.iter().all(|v| v.is_finite()));
        assert_eq!(b[0], 1.7e308);
    }

    #[test]
    fn corrections_that_do_not_fit_in_a_double_are_still_refused() {
        let n = 2;
        // `‖r‖/‖s‖` is `1e600`.
        let huge = (vec![1e-300, 1e-300], vec![1e300, 1e300]);
        // `‖r‖/‖s‖` is `1e-600`, and `r` is flushed out of the scaled model.
        let flushed = (vec![1e300, 1e300], vec![1e-300, 1e-300]);
        for kind in kinds() {
            for (s, y) in [&huge, &flushed] {
                let mut b = vec![0.0; n * n];
                assert_eq!(
                    update_symmetric(kind, &mut b, n, s, y),
                    Err(SecantError::NonFinite),
                    "{kind:?} s {s:?} y {y:?}"
                );
                assert!(b.iter().all(|v| *v == 0.0));
            }
        }
        // A model entry that is already infinite is never repaired.
        let mut b = vec![f64::MAX, 0.0, 0.0, f64::MAX];
        let before = b.clone();
        assert_eq!(
            update_symmetric(
                SecantKind::Psb,
                &mut b,
                n,
                &[1e-300, 0.0],
                &[-f64::MAX, 0.0]
            ),
            Err(SecantError::NonFinite)
        );
        assert_eq!(b, before);
    }
}
