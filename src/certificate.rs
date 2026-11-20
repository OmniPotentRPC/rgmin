//! A-posteriori Newton--Kantorovich, adjoint functional, and Morse-gap bounds.
//!
//! The caller supplies operator-norm majorants in one consistent norm on the
//! primal space, together with the matching dual norm for linear functionals
//! and the induced operator norm for linear maps. This module only combines
//! those proved inputs; it does not estimate a Lipschitz constant from sampled
//! Jacobians. All arithmetic is IEEE 754 `f64` floating-point without outward
//! directed rounding or interval arithmetic, so the returned quantities are
//! floating-point evaluations of the analytical majorants rather than
//! machine-rigorous interval enclosures.
//!
//! References:
//! - L. V. Kantorovich and G. P. Akilov, *Functional Analysis*, 2nd ed.,
//!   Chapter XVIII ("Newton's Method"), Pergamon, 1982,
//!   <https://doi.org/10.1016/B978-0-08-023036-8.50024-2>;
//!   W. B. Gragg and R. A. Tapia, *Optimal Error Bounds for the
//!   Newton--Kantorovich Theorem*, SIAM J. Numer. Anal. 11 (1974) 10--13,
//!   <https://doi.org/10.1137/0711002>.
//! - R. Becker and R. Rannacher, *An optimal control approach to a posteriori
//!   error estimation in finite element methods*, Acta Numer. 10 (2001)
//!   1--102, <https://doi.org/10.1017/S0962492901000010>.
//! - H. Weyl, *Das asymptotische Verteilungsgesetz der Eigenwerte linearer
//!   partieller Differentialgleichungen*, Math. Ann. 71 (1912) 441--479,
//!   <https://doi.org/10.1007/BF01456804>.

use core::fmt;

/// A failure to prove an a-posteriori bound from the supplied majorants.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CertificateError {
    /// An input was negative or non-finite.
    InvalidMajorant {
        /// Name of the invalid input.
        input: &'static str,
        /// Supplied value.
        value: f64,
    },
    /// The Newton--Kantorovich smallness condition `h <= 1/2` failed.
    NoExistenceBall {
        /// Affine-invariant Kantorovich parameter.
        h: f64,
    },
    /// The perturbation enclosure reaches the spectral origin.
    UnresolvedMorseGap {
        /// Distance of the computed spectrum from zero.
        gap: f64,
        /// Residual-induced operator perturbation plus eigensolver roundoff.
        uncertainty: f64,
    },
}

impl fmt::Display for CertificateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMajorant { input, value } => {
                write!(f, "invalid certificate majorant {input}={value:.16e}")
            }
            Self::NoExistenceBall { h } => {
                write!(f, "Newton--Kantorovich condition failed: h={h:.16e} > 1/2")
            }
            Self::UnresolvedMorseGap { gap, uncertainty } => write!(
                f,
                "Morse gap unresolved: gap={gap:.16e}, uncertainty={uncertainty:.16e}"
            ),
        }
    }
}

impl std::error::Error for CertificateError {}

/// Certified ball containing a zero of a nonlinear stationarity map.
///
/// Instances are constructed by [`kantorovich_ball`], which validates the
/// majorants and enforces `h <= 1/2` so a negative radius or nonlinear
/// remainder cannot be forged. Bounds are evaluated in standard `f64`
/// floating-point without outward rounding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KantorovichBall {
    newton_step: f64,
    h: f64,
    radius: f64,
    nonlinear_remainder: f64,
}

impl KantorovichBall {
    /// Norm of the exact Newton correction `||J(x0)^{-1} F(x0)||`.
    #[must_use]
    pub const fn newton_step(&self) -> f64 {
        self.newton_step
    }

    /// Affine-invariant Kantorovich parameter `h = ||J(x0)^{-1}|| Lip(J) ||J(x0)^{-1} F(x0)||`.
    #[must_use]
    pub const fn h(&self) -> f64 {
        self.h
    }

    /// Radius about `x0` containing a zero.
    #[must_use]
    pub const fn radius(&self) -> f64 {
        self.radius
    }

    /// Bound on the nonlinear part of the displacement beyond the Newton step.
    #[must_use]
    pub const fn nonlinear_remainder(&self) -> f64 {
        self.nonlinear_remainder
    }
}

/// Certified first-order adjoint contribution and second-order remainder.
///
/// All quantities are evaluated in standard `f64` floating-point without
/// outward rounding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdjointFunctionalBar {
    /// Absolute first-order dual-weighted residual `|lambda^T F(x0)|`.
    pub first_order: f64,
    /// Linear displacement remainder plus the quadratic functional remainder.
    pub remainder: f64,
    /// Sum of `first_order` and `remainder`.
    pub total: f64,
}

/// Certified distance of a perturbed self-adjoint spectrum from zero.
///
/// All quantities are evaluated in standard `f64` floating-point without
/// outward rounding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MorseGap {
    /// Gap of the computed projected operator.
    pub computed: f64,
    /// Residual-induced perturbation plus eigensolver roundoff.
    pub uncertainty: f64,
    /// Strict lower bound on the exact gap.
    pub certified: f64,
}

fn nonnegative(input: &'static str, value: f64) -> Result<f64, CertificateError> {
    if value.is_finite() && value >= 0.0 {
        Ok(value)
    } else {
        Err(CertificateError::InvalidMajorant { input, value })
    }
}

fn positive(input: &'static str, value: f64) -> Result<f64, CertificateError> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(CertificateError::InvalidMajorant { input, value })
    }
}

/// Prove the classical Newton--Kantorovich existence ball.
///
/// Given a chosen primal norm `||.||` and its induced operator norm on linear
/// maps, `inverse_norm` bounds `||J(x0)^{-1}||`, `newton_step` is
/// `eta = ||J(x0)^{-1} F(x0)||`, and `jacobian_lipschitz` bounds the operator
/// Lipschitz ratio `||J(x) - J(y)|| / ||x - y||` on the ball. Since the
/// returned existence radius satisfies `r <= 2 eta` whenever `h <= 1/2`, a
/// Lipschitz bound valid on `B(x0, 2 eta)` suffices prior to evaluating `r`.
/// The returned radius uses the cancellation-free form
/// `2 eta / (1 + sqrt(1 - 2 h))` (Kantorovich and Akilov 1982,
/// <https://doi.org/10.1016/B978-0-08-023036-8.50024-2>; Gragg and Tapia
/// 1974, <https://doi.org/10.1137/0711002>). Arithmetic is standard `f64`
/// floating-point without outward rounding.
pub fn kantorovich_ball(
    newton_step: f64,
    inverse_norm: f64,
    jacobian_lipschitz: f64,
) -> Result<KantorovichBall, CertificateError> {
    let eta = nonnegative("newton_step", newton_step)?;
    let inverse = positive("inverse_norm", inverse_norm)?;
    let lipschitz = nonnegative("jacobian_lipschitz", jacobian_lipschitz)?;
    let k = inverse * lipschitz;
    if !k.is_finite() {
        return Err(CertificateError::InvalidMajorant {
            input: "inverse_norm*jacobian_lipschitz",
            value: k,
        });
    }
    let h = k * eta;
    if !h.is_finite() {
        return Err(CertificateError::InvalidMajorant {
            input: "h",
            value: h,
        });
    }
    if h > 0.5 {
        return Err(CertificateError::NoExistenceBall { h });
    }
    let radius = if eta == 0.0 {
        0.0
    } else {
        2.0 * eta / (1.0 + (1.0 - 2.0 * h).sqrt())
    };
    let nonlinear_remainder = radius - eta;
    Ok(KantorovichBall {
        newton_step: eta,
        h,
        radius,
        nonlinear_remainder,
    })
}

/// Bound a scalar functional at the zero enclosed by `ball`.
///
/// For the exact adjoint equation `J(x0)^T lambda = grad q(x0)`, `first_order`
/// is the dual-weighted residual `|lambda^T F(x0)|`, which assumes an exact
/// adjoint solve (any inexact-adjoint defect `||(J(x0)^T lambda - grad q(x0))||_*`
/// times `ball.radius()` must be added by the caller). `gradient_norm` bounds
/// `||grad q(x0)||_*` in the dual norm paired with `ball`'s primal norm, while
/// `hessian_bound` bounds `||D^2 q(x)||` in the induced operator norm from the
/// primal space to its dual throughout the existence ball (for example on
/// `B(x0, 2 eta)`). The remainder combines the linear displacement remainder
/// beyond the Newton step with the quadratic Taylor remainder of `q` (Becker
/// and Rannacher 2001, <https://doi.org/10.1017/S0962492901000010>). Bounds
/// are evaluated in standard `f64` floating-point without outward rounding.
pub fn adjoint_functional_bar(
    ball: KantorovichBall,
    first_order: f64,
    gradient_norm: f64,
    hessian_bound: f64,
) -> Result<AdjointFunctionalBar, CertificateError> {
    let first = nonnegative("first_order", first_order)?;
    let gradient = nonnegative("gradient_norm", gradient_norm)?;
    let hessian = nonnegative("hessian_bound", hessian_bound)?;
    let remainder = gradient * ball.nonlinear_remainder + 0.5 * hessian * ball.radius * ball.radius;
    if !remainder.is_finite() {
        return Err(CertificateError::InvalidMajorant {
            input: "functional_remainder",
            value: remainder,
        });
    }
    let total = first + remainder;
    if !total.is_finite() {
        return Err(CertificateError::InvalidMajorant {
            input: "functional_bar",
            value: total,
        });
    }
    Ok(AdjointFunctionalBar {
        first_order: first,
        remainder,
        total,
    })
}

/// Certify that a projected self-adjoint operator cannot change inertia.
///
/// By Weyl's eigenvalue perturbation inequality for self-adjoint operators
/// (Weyl 1912, <https://doi.org/10.1007/BF01456804>), each exact eigenvalue
/// differs from the corresponding unperturbed eigenvalue by at most the
/// spectral norm (Euclidean operator 2-norm) of the perturbation. Therefore
/// `operator_lipschitz` must bound the spectral-norm Lipschitz constant of the
/// projected self-adjoint operator over the existence ball (for example on
/// `B(x0, 2 eta)`), so that `operator_lipschitz * ball.radius()` bounds the
/// spectral-norm change between the approximate and exact stationary points.
/// `eigensolver_roundoff` is added before comparing that perturbation with the
/// computed spectral gap `computed_gap` from zero. Bounds are evaluated in
/// standard `f64` floating-point without outward rounding.
pub fn certify_morse_gap(
    computed_gap: f64,
    operator_lipschitz: f64,
    eigensolver_roundoff: f64,
    ball: KantorovichBall,
) -> Result<MorseGap, CertificateError> {
    let gap = nonnegative("computed_gap", computed_gap)?;
    let lipschitz = nonnegative("operator_lipschitz", operator_lipschitz)?;
    let roundoff = nonnegative("eigensolver_roundoff", eigensolver_roundoff)?;
    let uncertainty = lipschitz * ball.radius + roundoff;
    if !uncertainty.is_finite() {
        return Err(CertificateError::InvalidMajorant {
            input: "morse_uncertainty",
            value: uncertainty,
        });
    }
    if uncertainty >= gap {
        return Err(CertificateError::UnresolvedMorseGap { gap, uncertainty });
    }
    Ok(MorseGap {
        computed: gap,
        uncertainty,
        certified: gap - uncertainty,
    })
}
