//! Scalar radii-polynomial comparisons for a posteriori fixed-point bounds.
//!
//! If a map `T` satisfies `||T(x̄) - x̄|| <= epsilon` and
//! `sup_{B_r(x̄)} ||DT|| <= kappa(r)`, the strict comparison
//! `epsilon + r kappa(r) < r` proves that `T` maps the closed ball into
//! itself and is a contraction there.  This module only performs that scalar
//! comparison.  Callers remain responsible for proving the supplied bounds.
//! `certify_radii_polynomial` is the only constructor of a
//! `RadiiCertificate`.

use core::fmt;

/// A failed or invalid radii-polynomial comparison.
///
/// The enum is non-exhaustive, so a match outside this crate needs a
/// wildcard arm.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
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
///
/// Instances are constructed by [`certify_radii_polynomial`], which enforces
/// the strict self-map inequality. The fields are private, so a certificate
/// cannot be assembled from unchecked values. Each stored value is an IEEE
/// 754 binary64 evaluation; the getters name the rounding direction.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct RadiiCertificate {
    radius: f64,
    contraction: f64,
    image_radius: f64,
    margin: f64,
}

impl RadiiCertificate {
    /// Radius of the closed ball.
    #[must_use]
    pub const fn radius(&self) -> f64 {
        self.radius
    }

    /// Upward-rounded bound on `kappa(radius)`.
    #[must_use]
    pub const fn contraction(&self) -> f64 {
        self.contraction
    }

    /// Upward-rounded bound on `epsilon + radius * kappa(radius)`.
    #[must_use]
    pub const fn image_radius(&self) -> f64 {
        self.image_radius
    }

    /// Downward-rounded positive lower bound on `radius - image_radius`.
    #[must_use]
    pub const fn margin(&self) -> f64 {
        self.margin
    }
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

fn next_down(value: f64) -> f64 {
    if value == f64::NEG_INFINITY {
        return value;
    }
    if value == 0.0 {
        return -f64::from_bits(1);
    }
    let bits = value.to_bits();
    if value > 0.0 {
        f64::from_bits(bits - 1)
    } else {
        f64::from_bits(bits + 1)
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
    // Equality was refused above, so the real gap is positive.
    debug_assert!(radius > image_radius);
    let margin = next_down(radius - image_radius);
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
        assert!(certificate.image_radius() < certificate.radius());
        assert!(certificate.margin() > 0.0);
        let gap = certificate.radius() - certificate.image_radius();
        assert_eq!(certificate.margin().to_bits(), next_down(gap).to_bits());
        assert!(certificate.margin() < gap);
    }

    #[test]
    fn equality_is_not_a_certificate() {
        // Upward rounding lands the image on the radius. That pair is not a
        // strict contraction. One ulp lower still has a positive downward
        // margin and is accepted, so the refusal is the equality itself.
        let radius = 1.0;
        let on_radius = next_down(radius);
        let refusal = certify_radii_polynomial(on_radius, &[0.0], radius)
            .expect_err("image rounded onto the radius");
        match refusal {
            MajorantRefusal::NotContracting {
                image_radius,
                radius: got,
            } => {
                assert_eq!(image_radius.to_bits(), radius.to_bits());
                assert_eq!(got.to_bits(), radius.to_bits());
            }
            other => panic!("expected NotContracting, got {other:?}"),
        }
        let below = next_down(on_radius);
        let certificate =
            certify_radii_polynomial(below, &[0.0], radius).expect("one ulp below the equal image");
        assert!(certificate.image_radius() < certificate.radius());
        assert!(certificate.margin() > 0.0);
        let gap = certificate.radius() - certificate.image_radius();
        assert_eq!(certificate.margin().to_bits(), next_down(gap).to_bits());
    }

    #[test]
    fn non_positive_radius_is_refused() {
        for radius in [0.0, -0.0, -1.0] {
            assert_eq!(
                certify_radii_polynomial(0.0, &[0.0], radius),
                Err(MajorantRefusal::NonPositiveRadius),
                "radius {radius:?}"
            );
        }
    }

    #[test]
    fn overflowing_product_is_non_finite() {
        assert_eq!(
            certify_radii_polynomial(0.0, &[f64::MAX], f64::MAX),
            Err(MajorantRefusal::NonFinite)
        );
        assert_eq!(
            certify_radii_polynomial(f64::MAX, &[f64::MAX], 2.0),
            Err(MajorantRefusal::NonFinite)
        );
    }

    #[test]
    fn directed_rounding_at_zero_and_subnormals() {
        assert_eq!(next_up(0.0).to_bits(), 1);
        assert_eq!(next_up(-0.0).to_bits(), 1);
        assert_eq!(next_down(0.0).to_bits(), 1_u64 << 63 | 1);
        assert_eq!(next_down(-0.0).to_bits(), 1_u64 << 63 | 1);

        let sub = f64::from_bits(1);
        assert_eq!(next_up(sub).to_bits(), 2);
        assert_eq!(next_down(sub).to_bits(), 0);
        assert_eq!(next_up(-sub).to_bits(), 1_u64 << 63);
        assert_eq!(next_down(-sub).to_bits(), 1_u64 << 63 | 2);
        assert_eq!(next_down(f64::from_bits(2)).to_bits(), 1);

        let lifted = certify_radii_polynomial(0.0, &[0.0], 1.0).expect("zero coefficient");
        assert_eq!(lifted.contraction().to_bits(), 2);
        assert_eq!(lifted.image_radius().to_bits(), 4);
        assert!(lifted.margin() > 0.0);

        // radius - image is one subnormal, so the downward margin is zero
        // and the comparison must refuse. Dropping the round-down would accept it.
        let radius = f64::from_bits(3);
        let refusal =
            certify_radii_polynomial(0.0, &[], radius).expect_err("margin rounds to zero");
        assert!(matches!(
            refusal,
            MajorantRefusal::NotContracting {
                image_radius,
                radius: got,
            } if image_radius.to_bits() == 2 && got.to_bits() == 3
        ));
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
