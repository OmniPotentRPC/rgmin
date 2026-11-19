//! A-posteriori Newton--Kantorovich and adjoint functional bounds.
//!
//! The caller supplies operator-norm majorants in one consistent norm.  This
//! module only combines those proved inputs; it does not estimate a Lipschitz
//! constant from sampled Jacobians.

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
                write!(f, "invalid certificate majorant {input}={value:.8e}")
            }
            Self::NoExistenceBall { h } => {
                write!(f, "Newton--Kantorovich condition failed: h={h:.8e} > 1/2")
            }
            Self::UnresolvedMorseGap { gap, uncertainty } => write!(
                f,
                "Morse gap unresolved: gap={gap:.8e}, uncertainty={uncertainty:.8e}"
            ),
        }
    }
}

impl std::error::Error for CertificateError {}

/// Certified ball containing a zero of a nonlinear stationarity map.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KantorovichBall {
    /// Norm of the exact Newton correction `J(x0)^{-1} F(x0)`.
    pub newton_step: f64,
    /// `h = ||J(x0)^{-1}|| Lip(J) ||J(x0)^{-1} F(x0)||`.
    pub h: f64,
    /// Radius about `x0` containing a zero.
    pub radius: f64,
    /// Bound on the nonlinear part of the displacement beyond the Newton step.
    pub nonlinear_remainder: f64,
}

/// Certified first-order adjoint contribution and second-order remainder.
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
/// `inverse_norm` bounds `||J(x0)^{-1}||`, `jacobian_lipschitz` bounds
/// `||J(x)-J(y)|| / ||x-y||` on the resulting ball, and `newton_step` is
/// `||J(x0)^{-1} F(x0)||`.  The returned radius uses the cancellation-free
/// form `2 eta / (1 + sqrt(1 - 2 h))`.
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
/// If `J(x0)^T lambda = grad q(x0)`, `first_order` is
/// `|lambda^T F(x0)|`. `gradient_norm` bounds `||grad q(x0)||`, while
/// `hessian_bound` bounds `||D^2 q||` throughout the existence ball.
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
/// `operator_lipschitz * ball.radius` bounds the operator change between the
/// approximate and exact stationary points. `eigensolver_roundoff` is added
/// before comparing that perturbation with the computed gap from zero.
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
