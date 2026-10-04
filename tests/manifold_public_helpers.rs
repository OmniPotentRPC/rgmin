use ndarray::array;
use rgmin::manifold::{
    CenterMode, CenteredMatrix, Manifold, inner_unitary, is_unitary, typical_dist_unitary,
};

#[test]
fn retained_center_mode_and_unitary_helpers_are_public() {
    let centered = CenteredMatrix::with_mode(2, 2, CenterMode::Rows);
    let x = array![1.0, 2.0, -1.0, -2.0];
    let y = centered.retract(&x, &array![2.0, 4.0, 0.0, 0.0]);
    assert_eq!(y, array![2.0, 4.0, -2.0, -4.0]);
    let unitary = array![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    assert!(is_unitary(&unitary));
    assert_eq!(inner_unitary(&unitary, &unitary), 2.0);
    assert_eq!(typical_dist_unitary(2), std::f64::consts::PI * 2.0_f64.sqrt());
}
