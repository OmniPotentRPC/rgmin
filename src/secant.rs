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
//! invariant under `(s, y) -> (α s, α y)`. Products of two Gram factors are
//! divided one factor at a time, so a very long or very short step does not
//! overflow or underflow the weight. Deciding whether a pair is above the
//! noise of the gradient is the caller's job.

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
    let ss = dot(s, s);
    if !ss.is_finite() {
        return Err(SecantError::NonFinite);
    }
    if ss == 0.0 {
        return Ok(SecantOutcome::ZeroStep);
    }
    let r: Vec<f64> = (0..n)
        .map(|i| y[i] - dot(&b[i * n..(i + 1) * n], s))
        .collect();
    let rr = dot(&r, &r);
    let rs = dot(&r, s);
    if !(rr.is_finite() && rs.is_finite()) {
        return Err(SecantError::NonFinite);
    }
    if rr == 0.0 {
        return Ok(SecantOutcome::Satisfied);
    }
    // Weights of r rᵀ, (r sᵀ + s rᵀ) and s sᵀ in the increment.
    // `ss * ss` and `rr * ss` overflow near a step of 1e100 and underflow
    // near 1e-80. Divide one square at a time: `(rs / ss) / ss`.
    let (w_rr, w_rs, w_ss) = match kind {
        SecantKind::Psb => (0.0, ss.recip(), -((rs / ss) / ss)),
        SecantKind::Sr1 { skip_bits } => {
            let skip = f64::from_bits(skip_bits);
            if rs == 0.0 || rs.abs() < skip * rr.sqrt() * ss.sqrt() {
                return Ok(SecantOutcome::Skipped);
            }
            (rs.recip(), 0.0, 0.0)
        }
        SecantKind::Bofill => {
            let phi = (rs / rr) * (rs / ss);
            let psb = 1.0 - phi;
            ((rs / rr) / ss, psb / ss, -((psb * (rs / ss)) / ss))
        }
    };
    if !(w_rr.is_finite() && w_rs.is_finite() && w_ss.is_finite()) {
        return Err(SecantError::NonFinite);
    }
    let mut next = b.to_vec();
    for i in 0..n {
        for j in 0..n {
            let increment =
                w_rr * (r[i] * r[j]) + w_rs * (r[i] * s[j] + r[j] * s[i]) + w_ss * (s[i] * s[j]);
            next[i * n + j] += increment;
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
            update_symmetric(SecantKind::Bofill, &mut b, n, &[1e300, 1e300], &[1.0, 2.0]),
            Err(SecantError::NonFinite)
        );
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
}
