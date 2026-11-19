use rgmin::{CertificateError, adjoint_functional_bar, certify_morse_gap, kantorovich_ball};

#[test]
fn affine_stationarity_has_no_nonlinear_remainder() {
    let ball = kantorovich_ball(0.25, 2.0, 0.0).expect("affine ball");
    assert_eq!(ball.radius, 0.25);
    assert_eq!(ball.nonlinear_remainder, 0.0);

    let bar = adjoint_functional_bar(ball, 0.75, 3.0, 0.0).expect("affine functional");
    assert_eq!(bar.first_order, 0.75);
    assert_eq!(bar.remainder, 0.0);
    assert_eq!(bar.total, 0.75);
}

#[test]
fn scalar_quadratic_zero_is_inside_the_kantorovich_ball() {
    // F(x) = x + x^2 - 0.1 at x0 = 0.  Here J(0) = 1,
    // Lip(J) = 2 and eta = 0.1, so h = 0.2.
    let ball = kantorovich_ball(0.1, 1.0, 2.0).expect("quadratic ball");
    let zero = (-1.0 + 1.4_f64.sqrt()) / 2.0;
    assert!(zero <= ball.radius);
    assert!((ball.h - 0.2).abs() <= f64::EPSILON);

    // q(x) = 2x + 3x^2 / 2.  Its exact change is covered by the
    // first-order adjoint term and the theorem's two remainders.
    let first_order = 0.2;
    let bar = adjoint_functional_bar(ball, first_order, 2.0, 3.0).expect("quadratic functional");
    let exact = (2.0 * zero + 1.5 * zero * zero).abs();
    assert!(exact <= bar.total);
}

#[test]
fn failed_smallness_and_morse_gap_are_typed() {
    assert!(matches!(
        kantorovich_ball(1.0, 1.0, 1.0),
        Err(CertificateError::NoExistenceBall { h }) if h == 1.0
    ));

    let ball = kantorovich_ball(0.1, 1.0, 2.0).expect("quadratic ball");
    assert!(matches!(
        certify_morse_gap(0.2, 2.0, 0.0, ball),
        Err(CertificateError::UnresolvedMorseGap { .. })
    ));
}

#[test]
fn strict_morse_gap_survives_the_residual_ball() {
    let ball = kantorovich_ball(0.01, 2.0, 1.0).expect("small ball");
    let gap = certify_morse_gap(0.5, 1.5, 1.0e-12, ball).expect("Morse gap");
    assert!(gap.certified > 0.0);
    assert_eq!(gap.certified, gap.computed - gap.uncertainty);
}
