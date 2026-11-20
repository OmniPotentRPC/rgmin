use rgmin::{CertificateError, adjoint_functional_bar, certify_morse_gap, kantorovich_ball};

#[derive(Clone, Copy)]
struct DoubleDouble {
    hi: f64,
    lo: f64,
}

impl DoubleDouble {
    fn new(value: f64) -> Self {
        Self { hi: value, lo: 0.0 }
    }

    fn normalize(hi: f64, lo: f64) -> Self {
        let sum = hi + lo;
        Self {
            hi: sum,
            lo: lo - (sum - hi),
        }
    }

    fn plus(self, rhs: Self) -> Self {
        let sum = self.hi + rhs.hi;
        let carried = sum - self.hi;
        let error = (self.hi - (sum - carried)) + (rhs.hi - carried) + self.lo + rhs.lo;
        Self::normalize(sum, error)
    }

    fn minus(self, rhs: Self) -> Self {
        self.plus(Self {
            hi: -rhs.hi,
            lo: -rhs.lo,
        })
    }

    fn times(self, rhs: Self) -> Self {
        let product = self.hi * rhs.hi;
        let error = self.hi.mul_add(rhs.hi, -product)
            + self.hi * rhs.lo
            + self.lo * rhs.hi
            + self.lo * rhs.lo;
        Self::normalize(product, error)
    }

    fn over(self, rhs: Self) -> Self {
        let quotient = Self::new(self.hi / rhs.hi);
        let correction = Self::new(self.minus(rhs.times(quotient)).hi / rhs.hi);
        let estimate = quotient.plus(correction);
        estimate.plus(Self::new(self.minus(rhs.times(estimate)).hi / rhs.hi))
    }

    fn sqrt(self) -> Self {
        let half = Self::new(0.5);
        let mut estimate = Self::new(self.hi.sqrt());
        for _ in 0..2 {
            estimate = estimate.plus(self.over(estimate)).times(half);
        }
        estimate
    }

    fn value(self) -> f64 {
        self.hi + self.lo
    }
}

fn high_precision_ball(eta: f64, h: f64) -> (f64, f64) {
    let one = DoubleDouble::new(1.0);
    let denominator = one.plus(
        one.minus(DoubleDouble::new(2.0).times(DoubleDouble::new(h)))
            .sqrt(),
    );
    let radius = DoubleDouble::new(eta).times(DoubleDouble::new(2.0).over(denominator));
    let remainder = DoubleDouble::new(h).times(radius.over(denominator));
    (radius.value(), remainder.value())
}

fn assert_relative(actual: f64, expected: f64, tolerance: f64) {
    let relative_error = (actual - expected).abs() / expected.abs().max(f64::MIN_POSITIVE);
    assert!(
        relative_error <= tolerance,
        "actual={actual:.17e}, expected={expected:.17e}, relative_error={relative_error:.3e}, \
         tolerance={tolerance:.3e}"
    );
}

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
fn nonnegative_majorant_of_signed_zero_is_positive_zero() {
    let positive_zero = 0.0_f64.to_bits();
    let ball = kantorovich_ball(-0.0, 1.0, -0.0).expect("signed zero is nonnegative");
    assert_eq!(ball.newton_step().to_bits(), positive_zero);
    assert_eq!(ball.h().to_bits(), positive_zero);
    assert_eq!(ball.radius().to_bits(), positive_zero);
    assert_eq!(ball.nonlinear_remainder().to_bits(), positive_zero);

    let bar = adjoint_functional_bar(ball, -0.0, -0.0, -0.0).expect("signed zero");
    assert_eq!(bar.first_order.to_bits(), positive_zero);
    assert_eq!(bar.remainder.to_bits(), positive_zero);
    assert_eq!(bar.total.to_bits(), positive_zero);

    let gap = certify_morse_gap(1.0, -0.0, -0.0, ball).expect("zero perturbation");
    assert_eq!(gap.computed.to_bits(), 1.0_f64.to_bits());
    assert_eq!(gap.uncertainty.to_bits(), positive_zero);
    assert_eq!(gap.certified.to_bits(), 1.0_f64.to_bits());

    match certify_morse_gap(-0.0, -0.0, -0.0, ball) {
        Err(CertificateError::UnresolvedMorseGap { gap, uncertainty }) => {
            assert_eq!(gap.to_bits(), positive_zero);
            assert_eq!(uncertainty.to_bits(), positive_zero);
        }
        other => panic!("a zero computed gap stays unresolved, got {other:?}"),
    }

    match kantorovich_ball(1.0, -0.0, 0.0) {
        Err(CertificateError::InvalidMajorant { input, value }) => {
            assert_eq!(input, "inverse_norm");
            assert_eq!(value.to_bits(), (-0.0_f64).to_bits());
        }
        other => panic!("inverse_norm rejects signed zero, got {other:?}"),
    }
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
            input: "h",
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
    assert_eq!(
        kantorovich_ball(f64::MAX, f64::MIN_POSITIVE, 0.125),
        Err(CertificateError::InvalidMajorant {
            input: "radius",
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

// 2^-1074, the smallest positive subnormal.
const MIN_SUBNORMAL: f64 = f64::from_bits(1);

#[test]
fn quadratic_functional_remainder_does_not_underflow_with_a_subnormal_hessian() {
    let radius = 1.0e308;
    let ball = kantorovich_ball(radius, 1.0, 0.0).expect("affine ball");
    assert_eq!(ball.radius(), radius);

    let bar = adjoint_functional_bar(ball, 0.0, 0.0, MIN_SUBNORMAL).expect("finite functional");
    // 0.5 * 2^-1074 * r = r * 2^-1000 * 2^-75 is exact and normal, so the
    // final product rounds once.
    let expected = (radius * 2.0_f64.powi(-1000) * 2.0_f64.powi(-75)) * radius;
    assert!((expected - 2.470_328_229_206_233e292).abs() <= 1.0e277);
    assert_relative(bar.remainder, expected, 2.0 * f64::EPSILON);
    assert_eq!(bar.total, bar.remainder);
}

#[test]
fn quadratic_functional_remainder_does_not_overflow_below_f64_max() {
    // 0.5 * MAX * 1.2^2 = 0.72 * MAX, although MAX * 1.2 overflows.
    let ball = kantorovich_ball(1.2, 1.0, 0.0).expect("affine ball");
    let bar = adjoint_functional_bar(ball, 0.0, 0.0, f64::MAX).expect("finite functional");
    assert_relative(bar.remainder, 0.72 * f64::MAX, 4.0 * f64::EPSILON);

    let wide = kantorovich_ball(1.0e308, 1.0, 0.0).expect("affine ball");
    assert_eq!(
        adjoint_functional_bar(wide, 0.0, 0.0, 1.0),
        Err(CertificateError::InvalidMajorant {
            input: "functional_remainder",
            value: f64::INFINITY,
        })
    );
}

#[test]
fn kantorovich_parameter_does_not_underflow_when_the_newton_step_is_large() {
    // inverse_norm * jacobian_lipschitz = 1e-400 is below every subnormal,
    // but h = 1e-100 and the nonlinear remainder is near 5e199.
    let ball = kantorovich_ball(1.0e300, 1.0e-200, 1.0e-200).expect("small h");
    assert_relative(ball.h(), 1.0e-100, 4.0 * f64::EPSILON);
    let (reference_radius, reference_remainder) = high_precision_ball(1.0e300, ball.h());
    assert_relative(ball.radius(), reference_radius, 4.0 * f64::EPSILON);
    assert_relative(
        ball.nonlinear_remainder(),
        reference_remainder,
        4.0 * f64::EPSILON,
    );
    assert!(ball.nonlinear_remainder() > 4.9e199);
}

#[test]
fn kantorovich_parameter_does_not_overflow_when_the_newton_step_is_small() {
    // inverse_norm * jacobian_lipschitz = 1e320 overflows, but
    // h = 1e320 * 2^-1074 is about 4.94e-4.
    let ball = kantorovich_ball(MIN_SUBNORMAL, 1.0e160, 1.0e160).expect("small h");
    let expected = (1.0e160 * MIN_SUBNORMAL) * 1.0e160;
    assert_relative(ball.h(), expected, 2.0 * f64::EPSILON);
    assert!((ball.h() - 4.940_656_458_412_465e-4).abs() <= 1.0e-18);
    assert!(ball.radius() >= MIN_SUBNORMAL);
}

#[test]
fn kantorovich_radius_and_remainder_match_double_double_reference() {
    // The compact double-double calculation above treats each f64 input exactly.
    // Four ulps of relative slack cover the final f64 sqrt and arithmetic.
    const RADIUS_RELATIVE_TOLERANCE: f64 = 4.0 * f64::EPSILON;
    const REMAINDER_RELATIVE_TOLERANCE: f64 = 4.0 * f64::EPSILON;

    for eta in [1.0e308, f64::MAX] {
        let ball = kantorovich_ball(eta, 1.0, 0.0).expect("finite affine radius");
        let (reference_radius, reference_remainder) = high_precision_ball(eta, 0.0);
        assert_relative(ball.radius(), reference_radius, RADIUS_RELATIVE_TOLERANCE);
        assert_eq!(ball.nonlinear_remainder(), reference_remainder);
        certify_morse_gap(f64::MAX, 0.0, 0.0, ball).expect("finite radius does not poison gap");
    }

    for h in [1.0e-17, 1.0e-12, 0.5 - f64::EPSILON] {
        let ball = kantorovich_ball(1.0, 1.0, h).expect("valid Kantorovich parameter");
        let (reference_radius, reference_remainder) = high_precision_ball(1.0, h);
        assert_relative(ball.radius(), reference_radius, RADIUS_RELATIVE_TOLERANCE);
        assert_relative(
            ball.nonlinear_remainder(),
            reference_remainder,
            REMAINDER_RELATIVE_TOLERANCE,
        );
    }
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
