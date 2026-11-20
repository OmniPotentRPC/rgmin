use rgmin::{CertificateError, adjoint_functional_bar, certify_morse_gap, kantorovich_ball};

#[test]
fn affine_stationarity_has_no_nonlinear_remainder() {
    let ball = kantorovich_ball(0.25, 2.0, 0.0).expect("affine ball");
    assert_eq!(ball.newton_step(), 0.25);
    assert_eq!(ball.h(), 0.0);
    assert_eq!(ball.radius(), 0.25);
    assert_eq!(ball.nonlinear_remainder(), 0.0);

    let bar = adjoint_functional_bar(ball, 0.75, 3.0, 0.0).expect("affine functional");
    assert_eq!(bar.first_order, 0.75);
    assert_eq!(bar.remainder, 0.0);
    assert_eq!(bar.total, 0.75);

    let zero_step = kantorovich_ball(0.0, 2.0, 5.0).expect("zero Newton step");
    assert_eq!(zero_step.newton_step(), 0.0);
    assert_eq!(zero_step.h(), 0.0);
    assert_eq!(zero_step.radius(), 0.0);
    assert_eq!(zero_step.nonlinear_remainder(), 0.0);
}

#[test]
fn scalar_quadratic_zero_is_inside_the_kantorovich_ball() {
    // F(x) = x + x^2 - 0.1 at x0 = 0.  Here J(0) = 1,
    // Lip(J) = 2 and eta = 0.1, so h = 0.2.
    let ball = kantorovich_ball(0.1, 1.0, 2.0).expect("quadratic ball");
    let zero = (-1.0 + 1.4_f64.sqrt()) / 2.0;
    assert!(zero <= ball.radius());
    assert!((ball.h() - 0.2).abs() <= f64::EPSILON);

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
        kantorovich_ball(0.0, 0.0, 0.0),
        Err(CertificateError::InvalidMajorant {
            input: "inverse_norm",
            value: 0.0
        })
    ));
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

#[test]
fn nonnegative_and_positive_majorants_reject_negative_nan_and_infinite_inputs() {
    for bad in [
        -1.0,
        -f64::MIN_POSITIVE,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        assert!(matches!(
            kantorovich_ball(bad, 1.0, 1.0),
            Err(CertificateError::InvalidMajorant {
                input: "newton_step",
                value
            }) if value.to_bits() == bad.to_bits()
        ));
        assert!(matches!(
            kantorovich_ball(0.1, bad, 1.0),
            Err(CertificateError::InvalidMajorant {
                input: "inverse_norm",
                value
            }) if value.to_bits() == bad.to_bits()
        ));
        assert!(matches!(
            kantorovich_ball(0.1, 1.0, bad),
            Err(CertificateError::InvalidMajorant {
                input: "jacobian_lipschitz",
                value
            }) if value.to_bits() == bad.to_bits()
        ));

        let ball = kantorovich_ball(0.1, 1.0, 1.0).expect("ball");
        assert!(matches!(
            adjoint_functional_bar(ball, bad, 1.0, 1.0),
            Err(CertificateError::InvalidMajorant {
                input: "first_order",
                value
            }) if value.to_bits() == bad.to_bits()
        ));
        assert!(matches!(
            adjoint_functional_bar(ball, 1.0, bad, 1.0),
            Err(CertificateError::InvalidMajorant {
                input: "gradient_norm",
                value
            }) if value.to_bits() == bad.to_bits()
        ));
        assert!(matches!(
            adjoint_functional_bar(ball, 1.0, 1.0, bad),
            Err(CertificateError::InvalidMajorant {
                input: "hessian_bound",
                value
            }) if value.to_bits() == bad.to_bits()
        ));
        assert!(matches!(
            certify_morse_gap(bad, 1.0, 0.0, ball),
            Err(CertificateError::InvalidMajorant {
                input: "computed_gap",
                value
            }) if value.to_bits() == bad.to_bits()
        ));
        assert!(matches!(
            certify_morse_gap(1.0, bad, 0.0, ball),
            Err(CertificateError::InvalidMajorant {
                input: "operator_lipschitz",
                value
            }) if value.to_bits() == bad.to_bits()
        ));
        assert!(matches!(
            certify_morse_gap(1.0, 1.0, bad, ball),
            Err(CertificateError::InvalidMajorant {
                input: "eigensolver_roundoff",
                value
            }) if value.to_bits() == bad.to_bits()
        ));
    }
}

#[test]
fn overflow_in_majorant_products_and_remainders_is_rejected() {
    assert_eq!(
        kantorovich_ball(1.0, f64::MAX, 2.0),
        Err(CertificateError::InvalidMajorant {
            input: "inverse_norm*jacobian_lipschitz",
            value: f64::INFINITY,
        })
    );
    assert_eq!(
        kantorovich_ball(f64::MAX, 1.0e100, 1.0e100),
        Err(CertificateError::InvalidMajorant {
            input: "h",
            value: f64::INFINITY,
        })
    );

    let large_ball = kantorovich_ball(2.0, 0.25, 1.0).expect("ball with nonlinear_remainder >= 1");
    assert_eq!(
        adjoint_functional_bar(large_ball, 0.0, f64::MAX, f64::MAX),
        Err(CertificateError::InvalidMajorant {
            input: "functional_remainder",
            value: f64::INFINITY,
        })
    );

    let half_ball = kantorovich_ball(0.25, 1.0, 2.0).expect("ball with h = 0.5");
    assert_eq!(
        adjoint_functional_bar(half_ball, f64::MAX, f64::MAX, 0.0),
        Err(CertificateError::InvalidMajorant {
            input: "functional_bar",
            value: f64::INFINITY,
        })
    );

    assert_eq!(
        certify_morse_gap(1.0, f64::MAX, f64::MAX, large_ball),
        Err(CertificateError::InvalidMajorant {
            input: "morse_uncertainty",
            value: f64::INFINITY,
        })
    );
}

#[test]
fn kantorovich_and_morse_gap_boundaries_and_display_formatting() {
    let boundary_ball = kantorovich_ball(0.25, 1.0, 2.0).expect("h == 0.5 is accepted");
    assert_eq!(boundary_ball.h(), 0.5);
    assert_eq!(boundary_ball.radius(), 0.5);
    assert_eq!(boundary_ball.nonlinear_remainder(), 0.25);

    let just_above_half = 0.5_f64.next_up();
    let err_h =
        kantorovich_ball(just_above_half, 1.0, 1.0).expect_err("nextafter(0.5, 1) exceeds 1/2");
    assert_eq!(
        err_h,
        CertificateError::NoExistenceBall { h: just_above_half }
    );
    let msg_h = err_h.to_string();
    assert_ne!(
        msg_h,
        "Newton--Kantorovich condition failed: h=5.00000000e-1 > 1/2"
    );
    assert_eq!(
        msg_h,
        "Newton--Kantorovich condition failed: h=5.0000000000000011e-1 > 1/2"
    );

    let err_gap =
        certify_morse_gap(0.5, 1.0, 0.0, boundary_ball).expect_err("uncertainty == gap is refused");
    assert_eq!(
        err_gap,
        CertificateError::UnresolvedMorseGap {
            gap: 0.5,
            uncertainty: 0.5,
        }
    );
    assert_eq!(
        err_gap.to_string(),
        "Morse gap unresolved: gap=5.0000000000000000e-1, uncertainty=5.0000000000000000e-1"
    );

    let err_majorant = CertificateError::InvalidMajorant {
        input: "newton_step",
        value: -0.25,
    };
    assert_eq!(
        err_majorant.to_string(),
        "invalid certificate majorant newton_step=-2.5000000000000000e-1"
    );
}
