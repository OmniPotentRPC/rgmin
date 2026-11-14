//! Line-search accept conditions (xtsci `linesearch/conditions/`).
//!
//! Armijo, weak/strong curvature, weak/strong Wolfe, and Goldstein.
//! Goldstein is Nocedal-Wright 3.11 (two-sided), not the xtsci pair of
//! upper bounds. [`goldstein_upper`](crate::linesearch::conditions::goldstein_upper)
//! keeps the xtsci named condition.

/// Armijo sufficient decrease: `φ(α) <= φ(0) + c α φ'(0)`.
///
/// Nocedal and Wright, *Numerical Optimization*,
/// <https://doi.org/10.1007/978-0-387-40065-5>.
#[inline]
pub fn armijo(phi: f64, phi0: f64, alpha: f64, dphi0: f64, c: f64) -> bool {
    phi <= phi0 + c * alpha * dphi0
}

/// Weak curvature: `φ'(α) >= c2 φ'(0)`.
///
/// Wolfe, *Convergence Conditions for Ascent Methods*,
/// <https://doi.org/10.1137/1011036>.
#[inline]
pub fn curvature(dphi: f64, dphi0: f64, c2: f64) -> bool {
    dphi >= c2 * dphi0
}

/// Strong curvature: `|φ'(α)| <= c2 |φ'(0)|`.
///
/// Wolfe, *Convergence Conditions for Ascent Methods*,
/// <https://doi.org/10.1137/1011036>.
#[inline]
pub fn strong_curvature(dphi: f64, dphi0: f64, c2: f64) -> bool {
    dphi.abs() <= c2 * dphi0.abs()
}

/// Weak Wolfe: Armijo and weak curvature.
///
/// Wolfe, *Convergence Conditions for Ascent Methods*,
/// <https://doi.org/10.1137/1011036>.
#[inline]
pub fn weak_wolfe(
    phi: f64,
    phi0: f64,
    alpha: f64,
    dphi: f64,
    dphi0: f64,
    c1: f64,
    c2: f64,
) -> bool {
    armijo(phi, phi0, alpha, dphi0, c1) && curvature(dphi, dphi0, c2)
}

/// Strong Wolfe: Armijo and strong curvature.
///
/// Wolfe, *Convergence Conditions for Ascent Methods*,
/// <https://doi.org/10.1137/1011036>.
#[inline]
pub fn strong_wolfe(
    phi: f64,
    phi0: f64,
    alpha: f64,
    dphi: f64,
    dphi0: f64,
    c1: f64,
    c2: f64,
) -> bool {
    armijo(phi, phi0, alpha, dphi0, c1) && strong_curvature(dphi, dphi0, c2)
}

/// Strong curvature plus the approximate Wolfe slope bound within a
/// relative floating-point energy window. The window is four machine
/// epsilons times `|phi0|`; it is an acceptance policy, not a bound on the
/// oracle's evaluation error. No additive energy scale is introduced.
///
/// For a quadratic restriction the slope bound is exactly Armijo. For a
/// general nonlinear restriction it is a local approximation; endpoint
/// derivatives alone do not prove exact decrease.
/// Hager and Zhang, equations (4.1)-(4.2),
/// <https://doi.org/10.1137/030601880>.
#[inline]
pub(crate) fn roundoff_strong_wolfe(
    phi: f64,
    phi0: f64,
    dphi: f64,
    dphi0: f64,
    c1: f64,
    c2: f64,
) -> bool {
    (phi - phi0).abs() <= (4.0 * f64::EPSILON) * phi0.abs()
        && approximate_strong_wolfe(phi, phi0, dphi, dphi0, c1, c2, 4.0 * f64::EPSILON)
}

/// Hager--Zhang approximate Wolfe with an explicit relative value window.
/// The slope inequalities imply Armijo for a quadratic restriction; for
/// nonlinear objectives they are a local approximation, not a decrease proof.
/// <https://doi.org/10.1137/030601880>, equations (4.1)--(4.3).
#[inline]
pub fn approximate_strong_wolfe(
    phi: f64,
    phi0: f64,
    dphi: f64,
    dphi0: f64,
    c1: f64,
    c2: f64,
    relative_error: f64,
) -> bool {
    let window = relative_error * phi0.abs();
    phi.is_finite()
        && phi0.is_finite()
        && dphi.is_finite()
        && dphi0.is_finite()
        && relative_error.is_finite()
        && relative_error >= 0.0
        && window.is_finite()
        && dphi0 < 0.0
        && 0.0 < c1
        && c1 < 0.5
        && c1 < c2
        && c2 < 1.0
        && phi - phi0 <= window
        && dphi <= (2.0 * c1 - 1.0) * dphi0
        && strong_curvature(dphi, dphi0, c2)
}

/// xtsci `GoldsteinUpperBoundCondition`: `φ(α) <= φ(0) + (1 - c) α φ'(0)`.
///
/// This is a looser sufficient-decrease test than Armijo. The Nocedal-Wright
/// (3.11) / Goldstein two-sided pair uses [`goldstein_lower`] on the `(1-c)`
/// side instead. `c` belongs in `(0, 0.5)`.
#[inline]
pub fn goldstein_upper(phi: f64, phi0: f64, alpha: f64, dphi0: f64, c: f64) -> bool {
    phi <= phi0 + (1.0 - c) * alpha * dphi0
}

/// Goldstein lower bound (Nocedal-Wright 3.11, left): `φ(α) >= φ(0) + (1 - c) α φ'(0)`.
///
/// Controls the step from below. `c` belongs in `(0, 0.5)`.
///
/// Goldstein, *Multiplier and gradient methods*,
/// <https://doi.org/10.1007/BF00927673>.
#[inline]
pub fn goldstein_lower(phi: f64, phi0: f64, alpha: f64, dphi0: f64, c: f64) -> bool {
    phi >= phi0 + (1.0 - c) * alpha * dphi0
}

/// Goldstein (Nocedal-Wright 3.11): Armijo and the `(1-c)` lower bound.
///
/// `φ(0) + (1-c) α φ'(0) <= φ(α) <= φ(0) + c α φ'(0)`.
///
/// Goldstein, *Multiplier and gradient methods*,
/// <https://doi.org/10.1007/BF00927673>.
#[inline]
pub fn goldstein(phi: f64, phi0: f64, alpha: f64, dphi0: f64, c: f64) -> bool {
    armijo(phi, phi0, alpha, dphi0, c) && goldstein_lower(phi, phi0, alpha, dphi0, c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn goldstein_is_the_two_sided_interval() {
        // φ0=10, φ'=-4, c=0.1, α=1
        // Armijo: φ <= 9.6; lower: φ >= 6.4
        assert!(goldstein(8.0, 10.0, 1.0, -4.0, 0.1));
        assert!(!goldstein(9.8, 10.0, 1.0, -4.0, 0.1));
        assert!(!goldstein(6.0, 10.0, 1.0, -4.0, 0.1));
        assert!(goldstein_lower(8.0, 10.0, 1.0, -4.0, 0.1));
        assert!(!goldstein_lower(6.0, 10.0, 1.0, -4.0, 0.1));
        assert!(goldstein_upper(6.0, 10.0, 1.0, -4.0, 0.1));
    }

    #[test]
    fn armijo_and_curvature_match_xtsci() {
        // φ0=10, φ'0=-4, c=1e-4, α=1. rhs = 10 - 4e-4 = 9.9996
        assert!(armijo(9.5, 10.0, 1.0, -4.0, 1e-4));
        assert!(!armijo(9.9997, 10.0, 1.0, -4.0, 1e-4));
        // weak: dphi >= 0.9 * (-4) = -3.6
        assert!(curvature(-3.6, -4.0, 0.9));
        assert!(!curvature(-3.61, -4.0, 0.9));
        // strong: |dphi| <= 0.9 * 4 = 3.6
        assert!(strong_curvature(-3.6, -4.0, 0.9));
        assert!(strong_curvature(3.6, -4.0, 0.9));
        assert!(!strong_curvature(3.61, -4.0, 0.9));
    }

    #[test]
    fn wolfe_accept_is_the_strong_condition() {
        // Armijo holds; dphi = +5 passes weak curvature and fails strong.
        let phi = 8.0;
        let phi0 = 10.0;
        let alpha = 1.0;
        let dphi0 = -4.0;
        assert!(weak_wolfe(phi, phi0, alpha, 5.0, dphi0, 1e-4, 0.9));
        assert!(!strong_wolfe(phi, phi0, alpha, 5.0, dphi0, 1e-4, 0.9));
        assert!(strong_wolfe(phi, phi0, alpha, -3.0, dphi0, 1e-4, 0.9));
    }
}
