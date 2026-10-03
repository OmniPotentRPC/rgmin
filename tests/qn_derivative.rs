//! QN family derivatives include the sign of every shifted spectral mode.

use approx::assert_relative_eq;
use ndarray::{Array2, array};
use rgmin::qn_get_s;

#[test]
fn flipped_qn_mode_derivative_matches_the_exact_rational_family() {
    let eigenvalues = array![-2.0, 3.0];
    let eigenvectors = Array2::eye(2);
    let gradient = array![2.0, 3.0];
    let (step, derivative) = qn_get_s(&eigenvalues, &eigenvectors, &gradient, 1, 0.5);
    assert_relative_eq!(step[0], 4.0 / 5.0, epsilon = 1e-15);
    assert_relative_eq!(step[1], -6.0 / 7.0, epsilon = 1e-15);
    assert_relative_eq!(derivative[0], -8.0 / 25.0, epsilon = 1e-15);
    assert_relative_eq!(derivative[1], 12.0 / 49.0, epsilon = 1e-15);
}

#[test]
fn qn_mode_derivatives_match_finite_differences_after_rotation() {
    let eigenvalues = array![-2.0, 3.0];
    let eigenvectors = array![[0.6, -0.8], [0.8, 0.6]];
    let gradient = array![2.0, 3.0];
    let alpha = 0.5;
    let h = 1e-5;
    for order in 0..=2 {
        let (_, derivative) = qn_get_s(&eigenvalues, &eigenvectors, &gradient, order, alpha);
        let (plus, _) = qn_get_s(&eigenvalues, &eigenvectors, &gradient, order, alpha + h);
        let (minus, _) = qn_get_s(&eigenvalues, &eigenvectors, &gradient, order, alpha - h);
        let finite_difference = (plus - minus) / (2.0 * h);
        for k in 0..2 {
            assert_relative_eq!(derivative[k], finite_difference[k], epsilon = 1e-9);
        }
    }
}
