//! Scalar radii-polynomial comparisons for a posteriori fixed-point bounds.
//!
//! If a map `T` satisfies `||T(x̄) - x̄|| <= epsilon` and
//! `sup_{B_r(x̄)} ||DT|| <= kappa(r)`, the strict comparison
//! `epsilon + r kappa(r) < r` proves that `T` maps the closed ball into
//! itself and is a contraction there.  This module only performs that scalar
//! comparison.  Callers remain responsible for proving the supplied bounds.

use core::fmt;

/// A failed or invalid radii-polynomial comparison.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MajorantRefusal {
    /// An input or an intermediate value was not finite.
    NonFinite,
    /// The proposed radius was not strictly positive.
    NonPositiveRadius,
    /// A quantity advertised as an upper bound was negative.
    NegativeBound,
    /// The strict self-map inequality did not hold.
    NotContracting {
        /// Upward-rounded `epsilon + r kappa(r)`.
        image_radius: f64,
        /// Proposed ball radius.
        radius: f64,
    },
}

impl fmt::Display for MajorantRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite => f.write_str("the majorant contains a non-finite value"),
            Self::NonPositiveRadius => f.write_str("the proposed radius is not positive"),
            Self::NegativeBound => f.write_str("a majorant upper bound is negative"),
            Self::NotContracting {
                image_radius,
                radius,
            } => write!(
                f,
                "the majorant image radius {image_radius:e} is not strictly below {radius:e}"
            ),
        }
    }
}

impl std::error::Error for MajorantRefusal {}

/// A successful strict radii-polynomial comparison.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RadiiCertificate {
    /// Radius of the closed ball.
    pub radius: f64,
    /// Upward-rounded bound on `kappa(radius)`.
    pub contraction: f64,
    /// Upward-rounded bound on `epsilon + radius * kappa(radius)`.
    pub image_radius: f64,
    /// Downward-safe positive margin `radius - image_radius`.
    pub margin: f64,
}

fn next_up(value: f64) -> f64 {
    if value == f64::INFINITY {
        return value;
    }
    if value == -0.0 {
        return f64::from_bits(1);
    }
    let bits = value.to_bits();
    if value >= 0.0 {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    }
}

fn add_up(left: f64, right: f64) -> Result<f64, MajorantRefusal> {
    let value = left + right;
    if !value.is_finite() {
        return Err(MajorantRefusal::NonFinite);
    }
    Ok(next_up(value))
}

fn mul_up(left: f64, right: f64) -> Result<f64, MajorantRefusal> {
    let value = left * right;
    if !value.is_finite() {
        return Err(MajorantRefusal::NonFinite);
    }
    Ok(next_up(value))
}

/// Test `epsilon + r kappa(r) < r` for a polynomial upper bound on `kappa`.
///
/// `contraction_coefficients[j]` is a nonnegative upper bound for the
/// coefficient of `r^j`.  Every arithmetic operation in the comparison is
/// rounded upward by one binary64 step.  This protects the strict comparison;
/// it does not make inexact caller-supplied coefficients rigorous.
pub fn certify_radii_polynomial(
    defect: f64,
    contraction_coefficients: &[f64],
    radius: f64,
) -> Result<RadiiCertificate, MajorantRefusal> {
    if !defect.is_finite()
        || !radius.is_finite()
        || contraction_coefficients
            .iter()
            .any(|coefficient| !coefficient.is_finite())
    {
        return Err(MajorantRefusal::NonFinite);
    }
    if radius <= 0.0 {
        return Err(MajorantRefusal::NonPositiveRadius);
    }
    if defect < 0.0
        || contraction_coefficients
            .iter()
            .any(|coefficient| *coefficient < 0.0)
    {
        return Err(MajorantRefusal::NegativeBound);
    }

    let mut contraction = 0.0;
    for coefficient in contraction_coefficients.iter().rev() {
        contraction = add_up(mul_up(contraction, radius)?, *coefficient)?;
    }
    let image_radius = add_up(defect, mul_up(radius, contraction)?)?;
    if image_radius >= radius {
        return Err(MajorantRefusal::NotContracting {
            image_radius,
            radius,
        });
    }
    let margin = radius - image_radius;
    if !margin.is_finite() || margin <= 0.0 {
        return Err(MajorantRefusal::NotContracting {
            image_radius,
            radius,
        });
    }
    Ok(RadiiCertificate {
        radius,
        contraction,
        image_radius,
        margin,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_a_strict_polynomial_self_map() {
        let certificate =
            certify_radii_polynomial(0.01, &[0.25, 0.5], 0.02).expect("strict comparison");
        assert!(certificate.image_radius < certificate.radius);
        assert!(certificate.margin > 0.0);
    }

    #[test]
    fn equality_is_not_a_certificate() {
        let refusal = certify_radii_polynomial(0.5, &[0.0], 0.5).expect_err("strict inequality");
        assert!(matches!(refusal, MajorantRefusal::NotContracting { .. }));
    }

    #[test]
    fn refuses_nan_and_negative_bounds() {
        assert_eq!(
            certify_radii_polynomial(f64::NAN, &[0.0], 1.0),
            Err(MajorantRefusal::NonFinite)
        );
        assert_eq!(
            certify_radii_polynomial(0.0, &[-0.0, -1.0], 1.0),
            Err(MajorantRefusal::NegativeBound)
        );
    }
}
