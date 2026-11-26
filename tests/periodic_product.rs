#![cfg(all(feature = "periodic-product", not(feature = "capi")))]

use rgmin::periodic_product::{PeriodicProductError, periodic_product_spectrum};

#[test]
fn quarter_turn_spectrum_is_reachable_from_outside_the_crate() {
    let got = periodic_product_spectrum(&[vec![0.0, -1.0, 1.0, 0.0]], 2, 128).expect("spectrum");
    assert_eq!(got.multipliers.len(), 2);
    assert_eq!(got.precision_bits, 128);
    for multiplier in &got.multipliers {
        assert!(multiplier.log_modulus.abs() < 1e-14);
        assert!((multiplier.argument.abs() - std::f64::consts::FRAC_PI_2).abs() < 1e-14);
    }
}

#[test]
fn errors_are_matched_with_a_wildcard_arm() {
    let error = periodic_product_spectrum(&[vec![1.0, 0.0, 0.0, 1.0 + 2e-8]], 2, 128)
        .expect_err("non-reciprocal factor");
    let named = match error {
        PeriodicProductError::ReciprocityFailed => "reciprocity",
        _ => "other",
    };
    assert_eq!(named, "reciprocity");
}
