//! Gauss–Legendre rules and Lobatto shape functions on `[-1, 1]`.
//!
//! The Gauss–Legendre nodes are the roots of `P_n`, found by Newton's
//! method on the three-term recurrence from the asymptotic guesses
//! `cos(π (k − 1/4) / (n + 1/2))`; the weights are
//! `2 / ((1 − x²) P_n'(x)²)` (Davis and Rabinowitz 1984, §2.7). The rule
//! integrates polynomials of degree `2n − 1` exactly.
//!
//! The Lobatto shape functions of degree `p` are the two vertex hats
//! `(1 ∓ ξ)/2` and the `p − 1` bubbles
//! `φ_k = (P_k − P_{k−2}) / √(2(2k − 1))`, `k = 2, …, p`, the normalised
//! integrals of `P_{k−1}` (Szabó and Babuška 1991, §3.1). The bubbles
//! vanish at `ξ = ±1`, their derivatives `√((2k − 1)/2) P_{k−1}` are
//! orthonormal on `[-1, 1]`, and they are orthogonal to the hat
//! derivatives, so the stiffness matrix of `−d²/dξ²` is diagonal on the
//! bubbles.

use thiserror::Error;

/// A rule or shape-function request that could not be honoured.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
#[non_exhaustive]
pub enum QuadratureError {
    /// A rule needs at least one node.
    #[error("a Gauss–Legendre rule needs at least one node")]
    EmptyRule,
    /// Newton's method did not settle on a root of `P_n`.
    #[error("Newton's method did not converge for node {node} of the {count}-point rule")]
    NodeDidNotConverge {
        /// Index of the node in the half rule.
        node: usize,
        /// Number of nodes requested.
        count: usize,
    },
    /// A node or weight came out non-finite or outside `(-1, 1)`.
    #[error("node {node} of the {count}-point rule is not a valid abscissa")]
    InvalidNode {
        /// Index of the node in the half rule.
        node: usize,
        /// Number of nodes requested.
        count: usize,
    },
    /// Lobatto shape functions need degree at least one.
    #[error("Lobatto shape functions need degree at least one")]
    ZeroDegree,
    /// The reference coordinate is not a finite point of `[-1, 1]`.
    #[error("the reference coordinate {xi} is not in [-1, 1]")]
    OutsideReference {
        /// Requested coordinate.
        xi: f64,
    },
}

/// Nodes in ascending order and their weights on `[-1, 1]`.
///
/// The struct is non-exhaustive, so a field added later is not a breaking
/// change. [`gauss_legendre`] builds it.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct GaussLegendre {
    /// Abscissae, ascending.
    pub nodes: Vec<f64>,
    /// Positive weights summing to two.
    pub weights: Vec<f64>,
}

/// `P_n(x)` and `P_{n−1}(x)` by the Bonnet recurrence.
fn legendre_pair(n: usize, x: f64) -> (f64, f64) {
    let mut previous = 1.0;
    let mut current = x;
    if n == 0 {
        return (previous, 0.0);
    }
    for k in 1..n {
        let k = k as f64;
        let next = ((2.0 * k + 1.0) * x * current - k * previous) / (k + 1.0);
        previous = current;
        current = next;
    }
    (current, previous)
}

/// The `count`-point Gauss–Legendre rule.
///
/// Nodes are mirrored, so `nodes[i] = −nodes[count − 1 − i]` and the two
/// weights agree bit for bit; the middle node of an odd rule is zero.
pub fn gauss_legendre(count: usize) -> Result<GaussLegendre, QuadratureError> {
    if count == 0 {
        return Err(QuadratureError::EmptyRule);
    }
    const NEWTON_LIMIT: usize = 100;
    let n = count as f64;
    let half = count / 2;
    let mut nodes = vec![0.0; count];
    let mut weights = vec![0.0; count];
    for node in 0..half {
        let mut x = (std::f64::consts::PI * (node as f64 + 0.75) / (n + 0.5)).cos();
        let mut converged = false;
        for _ in 0..NEWTON_LIMIT {
            let (p, q) = legendre_pair(count, x);
            let derivative = n * (x * p - q) / (x * x - 1.0);
            let step = p / derivative;
            if !step.is_finite() {
                return Err(QuadratureError::InvalidNode { node, count });
            }
            x -= step;
            if step.abs() <= f64::EPSILON {
                converged = true;
                break;
            }
        }
        if !converged {
            return Err(QuadratureError::NodeDidNotConverge { node, count });
        }
        let (p, q) = legendre_pair(count, x);
        let derivative = n * (x * p - q) / (x * x - 1.0);
        let weight = 2.0 / ((1.0 - x * x) * derivative * derivative);
        if !(x > 0.0 && x < 1.0 && weight.is_finite() && weight > 0.0) {
            return Err(QuadratureError::InvalidNode { node, count });
        }
        nodes[node] = -x;
        nodes[count - 1 - node] = x;
        weights[node] = weight;
        weights[count - 1 - node] = weight;
    }
    if count % 2 == 1 {
        let (_, q) = legendre_pair(count, 0.0);
        let derivative = n * q;
        let weight = 2.0 / (derivative * derivative);
        if !(weight.is_finite() && weight > 0.0) {
            return Err(QuadratureError::InvalidNode { node: half, count });
        }
        nodes[half] = 0.0;
        weights[half] = weight;
    }
    Ok(GaussLegendre { nodes, weights })
}

/// Values and `ξ`-derivatives of the Lobatto shape functions at one point.
///
/// Index 0 is the left hat `(1 − ξ)/2`, index 1 the right hat `(1 + ξ)/2`,
/// and index `k ≥ 2` the bubble `φ_k`.
///
/// The struct is non-exhaustive, so a field added later is not a breaking
/// change. [`lobatto_shapes`] builds it.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct LobattoShapes {
    /// Shape values, `degree + 1` of them.
    pub values: Vec<f64>,
    /// Derivatives with respect to `ξ`.
    pub derivatives: Vec<f64>,
}

/// The degree-`degree` Lobatto shape functions at `xi ∈ [-1, 1]`.
pub fn lobatto_shapes(degree: usize, xi: f64) -> Result<LobattoShapes, QuadratureError> {
    if degree == 0 {
        return Err(QuadratureError::ZeroDegree);
    }
    if !(xi.is_finite() && (-1.0..=1.0).contains(&xi)) {
        return Err(QuadratureError::OutsideReference { xi });
    }
    let mut values = Vec::with_capacity(degree + 1);
    let mut derivatives = Vec::with_capacity(degree + 1);
    values.push(0.5 * (1.0 - xi));
    values.push(0.5 * (1.0 + xi));
    derivatives.push(-0.5);
    derivatives.push(0.5);
    // legendre[k] = P_k(ξ) for k = 0, …, degree.
    let mut legendre = Vec::with_capacity(degree + 1);
    legendre.push(1.0);
    legendre.push(xi);
    for k in 1..degree {
        let kf = k as f64;
        let next = ((2.0 * kf + 1.0) * xi * legendre[k] - kf * legendre[k - 1]) / (kf + 1.0);
        legendre.push(next);
    }
    for k in 2..=degree {
        let order = (2 * k - 1) as f64;
        values.push((legendre[k] - legendre[k - 2]) / (2.0 * order).sqrt());
        derivatives.push((0.5 * order).sqrt() * legendre[k - 1]);
    }
    Ok(LobattoShapes {
        values,
        derivatives,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rules_integrate_their_degree_exactly() -> Result<(), QuadratureError> {
        for count in 1..=40 {
            let rule = gauss_legendre(count)?;
            for power in 0..(2 * count) {
                let sum: f64 = rule
                    .nodes
                    .iter()
                    .zip(&rule.weights)
                    .map(|(x, w)| w * x.powi(power as i32))
                    .sum();
                let exact = if power % 2 == 1 {
                    0.0
                } else {
                    2.0 / (power as f64 + 1.0)
                };
                assert!(
                    (sum - exact).abs() <= 64.0 * f64::EPSILON,
                    "n {count} power {power}: {sum} against {exact}"
                );
            }
            assert!(rule.nodes.windows(2).all(|pair| pair[0] < pair[1]));
        }
        Ok(())
    }

    #[test]
    fn known_three_point_rule() -> Result<(), QuadratureError> {
        let rule = gauss_legendre(3)?;
        let root = (0.6f64).sqrt();
        assert!((rule.nodes[0] + root).abs() <= 2.0 * f64::EPSILON);
        assert_eq!(rule.nodes[1], 0.0);
        assert!((rule.weights[0] - 5.0 / 9.0).abs() <= 4.0 * f64::EPSILON);
        assert!((rule.weights[1] - 8.0 / 9.0).abs() <= 4.0 * f64::EPSILON);
        Ok(())
    }

    #[test]
    fn empty_rules_and_bad_shapes_are_refused() {
        assert_eq!(gauss_legendre(0), Err(QuadratureError::EmptyRule));
        assert_eq!(lobatto_shapes(0, 0.0), Err(QuadratureError::ZeroDegree));
        assert!(matches!(
            lobatto_shapes(3, 1.5),
            Err(QuadratureError::OutsideReference { .. })
        ));
        assert!(matches!(
            lobatto_shapes(3, f64::NAN),
            Err(QuadratureError::OutsideReference { .. })
        ));
    }

    #[test]
    fn bubbles_vanish_at_the_ends_and_have_orthonormal_derivatives() -> Result<(), QuadratureError>
    {
        let degree = 9;
        for end in [-1.0, 1.0] {
            let shapes = lobatto_shapes(degree, end)?;
            for value in &shapes.values[2..] {
                assert!(value.abs() <= 8.0 * f64::EPSILON, "bubble {value} at {end}");
            }
        }
        let rule = gauss_legendre(degree + 1)?;
        let mut gram = vec![0.0; (degree + 1) * (degree + 1)];
        for (x, w) in rule.nodes.iter().zip(&rule.weights) {
            let shapes = lobatto_shapes(degree, *x)?;
            for i in 0..=degree {
                for j in 0..=degree {
                    gram[i * (degree + 1) + j] += w * shapes.derivatives[i] * shapes.derivatives[j];
                }
            }
        }
        for i in 2..=degree {
            for j in 0..=degree {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (gram[i * (degree + 1) + j] - expected).abs() <= 32.0 * f64::EPSILON,
                    "derivative Gram ({i}, {j}) = {}",
                    gram[i * (degree + 1) + j]
                );
            }
        }
        Ok(())
    }
}
