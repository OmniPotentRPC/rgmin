use ndarray::array;
use rgmin::manifold::{
    CenterMode, is_centered, is_centered_mode, typical_dist_centered, typical_dist_centered_mode,
};

#[test]
fn public_centered_queries_reject_nonfinite_means() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let point = array![value, 0.0, 0.0, 0.0];
        for (rows, mode) in [(false, CenterMode::Cols), (true, CenterMode::Rows)] {
            assert!(!is_centered(&point, 2, 2, rows));
            assert!(!is_centered_mode(&point, 2, 2, mode));
        }
    }
}

#[test]
fn bool_and_typed_queries_preserve_their_tolerance_boundaries() {
    let tolerance = 1e-10_f64;
    let outside = f64::from_bits(tolerance.to_bits() + 1);
    for (rows, mode) in [(false, CenterMode::Cols), (true, CenterMode::Rows)] {
        assert!(is_centered(&array![tolerance], 1, 1, rows));
        assert!(!is_centered_mode(&array![tolerance], 1, 1, mode));
        assert!(!is_centered(&array![outside], 1, 1, rows));
        assert!(!is_centered_mode(&array![outside], 1, 1, mode));
        assert!(is_centered(&array![0.0], 1, 1, rows));
        assert!(is_centered_mode(&array![0.0], 1, 1, mode));
    }
}

#[test]
fn typed_centered_queries_keep_rows_and_columns_distinct() {
    let point = array![1.0, -1.0, 2.0, -2.0];
    assert!(is_centered_mode(&point, 2, 2, CenterMode::Cols));
    assert!(!is_centered_mode(&point, 2, 2, CenterMode::Rows));
    assert_eq!(typical_dist_centered_mode(2, 3, CenterMode::Cols), 2.0);
    assert_eq!(
        typical_dist_centered_mode(2, 3, CenterMode::Rows),
        3.0_f64.sqrt()
    );
    assert_eq!(
        typical_dist_centered_mode(2, 3, CenterMode::Cols),
        typical_dist_centered(2, 3, false)
    );
    assert!(!is_centered_mode(&array![0.0], 2, 2, CenterMode::Cols));
    assert!(!is_centered(&array![0.0], 2, 2, false));
}
