use ndarray::array;
use rgmin::{IrcTrust, sqrt_masses_3n};

#[test]
fn mass_conversion_repeats_weights_and_zeros_invalid_masses() {
    let weights = sqrt_masses_3n(&[4.0, -9.0, f64::NAN]);
    assert_eq!(
        weights.as_slice().unwrap(),
        &[2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    );
}

#[test]
fn empty_and_short_inputs_leave_unrepresented_coordinates_alone() {
    let empty = IrcTrust::new(array![], array![], 2.0);
    assert_eq!(empty.cons(&array![3.0]), 0.0);
    assert_eq!(empty.project(&array![3.0]), array![3.0]);

    let short = IrcTrust::new(array![1.0], array![2.0], 4.0);
    let projected = short.project(&array![2.0, 7.0]);
    assert_eq!(projected, array![1.0, 7.0]);
    assert_eq!(short.cons(&projected), 4.0);
}

#[test]
fn zero_radius_and_zero_weights_return_the_sphere_center() {
    let displacement = array![1.0, -2.0, 3.0];
    let zero_radius = IrcTrust::from_atom_masses(displacement.clone(), &[1.0], -4.0);
    assert_eq!(zero_radius.dx, 0.0);
    assert_eq!(
        zero_radius.project(&array![-1.0, 2.0, -3.0]),
        -&displacement
    );

    let zero_weights = IrcTrust::new(displacement.clone(), array![0.0, 0.0, 0.0], 2.0);
    assert_eq!(zero_weights.project(&array![5.0, 6.0, 7.0]), -displacement);
}

#[test]
fn degenerate_projection_uses_an_axis_with_positive_weight() {
    let trust = IrcTrust::new(array![1.0, -2.0, 3.0], array![0.0, 2.0, 3.0], 6.0);
    let projected = trust.project(&array![-1.0, 2.0, -3.0]);
    assert!(trust.on_bound(&projected, 32.0 * f64::EPSILON));
    assert!(!trust.on_bound(&array![-1.0, 2.0, -3.0], 32.0 * f64::EPSILON));
    assert_eq!(projected, array![-1.0, 5.0, -3.0]);

    let below_floor = 0.5 * 1e-16;
    let at_below_floor = IrcTrust::new(array![1.0, -2.0], array![below_floor, 2.0], 6.0);
    assert_eq!(
        at_below_floor.project(&array![-1.0, 2.0]),
        array![-1.0, 5.0]
    );
}
