use rgmin::inverse_laplace::{
    Complex64, ComplexSample, InverseLaplaceError, TalbotConfig, WeeksConfig,
    complex_matrix_inv_inf_norm, eval_orthonormal_laguerre_series, exterior_disk_modulus,
    higham_gamma, solve_complex_dense, talbot_invert, verify_laguerre_tail_decay, weeks_invert,
    weeks_mobius_w,
};

#[test]
fn complex_dense_solver_and_inverse_norm_recover_known_matrix() {
    let a = [
        Complex64::new(3.0, 1.0),
        Complex64::new(1.0, -0.5),
        Complex64::new(0.5, 0.25),
        Complex64::new(2.0, -1.0),
    ];
    let x_ref = [Complex64::new(1.25, -0.75), Complex64::new(-0.5, 1.5)];
    let b = [
        a[0] * x_ref[0] + a[1] * x_ref[1],
        a[2] * x_ref[0] + a[3] * x_ref[1],
    ];
    let x = solve_complex_dense(&a, &b, 2).expect("2x2 complex solve");
    assert!((x[0] - x_ref[0]).norm() <= 1.0e-14);
    assert!((x[1] - x_ref[1]).norm() <= 1.0e-14);

    let inv_norm = complex_matrix_inv_inf_norm(&a, 2).expect("2x2 inverse infinity norm");
    assert!(inv_norm.is_finite() && inv_norm > 0.4 && inv_norm < 2.0);
}

#[test]
fn talbot_and_weeks_invert_exponential_and_report_typed_estimates() {
    let a = 1.25_f64;
    let energy = 0.8_f64;
    let exact = (-a * energy).exp();
    let gamma4 = higham_gamma(4).expect("gamma4");

    let talbot_cfg = TalbotConfig::default()
        .with_nodes(32)
        .with_min_re_beta(-1.0e6)
        .with_tolerance(1.0e-8);
    let got_t = talbot_invert(energy, talbot_cfg, |beta| {
        let val = Complex64::new(1.0, 0.0) / (beta + Complex64::new(a, 0.0));
        Ok(ComplexSample::estimated(val, gamma4 * val.norm()))
    })
    .expect("talbot exponential");
    assert!(!got_t.is_certified());
    assert_eq!(got_t.certified_bar(), None);
    let err_t = (got_t.value - exact).abs();
    assert!(err_t <= got_t.uncertainty_estimate());
    assert!(err_t <= 1.0e-12);

    let weeks_cfg = WeeksConfig::default();
    let got_w = weeks_invert(energy, weeks_cfg, |beta| {
        let val = Complex64::new(1.0, 0.0) / (beta + Complex64::new(a, 0.0));
        Ok(ComplexSample::estimated(val, gamma4 * val.norm()))
    })
    .expect("weeks exponential");
    assert!(!got_w.is_certified());
    assert_eq!(got_w.certified_bar(), None);
    let err_w = (got_w.value - exact).abs();
    assert!(err_w <= got_w.uncertainty_estimate());
    assert!(err_w <= 1.0e-11);
}

#[test]
fn weeks_and_laguerre_refuse_slow_algebraic_tails() {
    // 1. Direct Taylor coefficients of Phi(w) = (1 - w)^alpha for alpha = 0.25 (n^-1.25) and 0.1 (n^-1.1)
    for alpha in [0.25_f64, 0.1_f64] {
        let mut coeffs = vec![0.0_f64; 64];
        coeffs[0] = 1.0;
        for n in 1..64 {
            coeffs[n] = coeffs[n - 1] * ((n as f64) - 1.0 - alpha) / (n as f64);
        }
        let res = verify_laguerre_tail_decay(&coeffs, 1.0e-14);
        assert!(
            matches!(res, Err(InverseLaplaceError::SlowAlgebraicTail { .. })),
            "alpha={alpha} must refuse slow algebraic tail, got {res:?}"
        );
    }

    // 2. End-to-end Laplace transform F(beta) = (beta + 1)^(-5/4), sigma = 0.5, b = 1.5, E = 1e-6, N = 16, r = 1
    let slow_cfg = WeeksConfig::default()
        .with_circle_radius(1.0)
        .with_laguerre_degree(16)
        .with_max_imag_beta(1.0e6)
        .with_tolerance(10.0);
    let gamma8 = higham_gamma(8).expect("gamma8");
    let res_weeks = weeks_invert(1.0e-6, slow_cfg, |beta| {
        let z = beta + Complex64::new(1.0, 0.0);
        let r = z.norm();
        let theta = z.im.atan2(z.re);
        let val = Complex64::from_polar(r.powf(-1.25), -1.25 * theta);
        Ok(ComplexSample::estimated(val, gamma8 * val.norm()))
    });
    assert!(
        matches!(
            res_weeks,
            Err(InverseLaplaceError::SlowAlgebraicTail { .. })
        ),
        "F(beta)=(beta+1)^(-5/4) must refuse with SlowAlgebraicTail, got {res_weeks:?}"
    );

    // 3. Valid O(n^-3/2) sequence (alpha = 0.5) is accepted by verify_laguerre_tail_decay
    let mut coeffs_half = vec![0.0_f64; 64];
    coeffs_half[0] = 1.0;
    for n in 1..64 {
        coeffs_half[n] = coeffs_half[n - 1] * ((n as f64) - 1.5) / (n as f64);
    }
    // The round-2 false disk: a = 0.5/beta_c, sigma = 0.5 beta_c, b = 1.5 beta_c
    // maps the pole beta = -a to w ≈ -2, which lies inside R = 3.
    let beta_c = 1.0 / (3.166_811_563e-6 * 300.0);
    let a_rate = 0.5 / beta_c;
    let sigma = 0.5 * beta_c;
    let scale_b = 1.5 * beta_c;
    let w_false = weeks_mobius_w(Complex64::new(-a_rate, 0.0), sigma, scale_b)
        .expect("mobius image of the false-disk pole");
    let false_mod = w_false.norm();
    assert!(
        false_mod > 1.9 && false_mod < 2.1,
        "false-disk |w|={false_mod}"
    );
    let refused = exterior_disk_modulus(Complex64::new(-a_rate, 0.0), sigma, scale_b, 3.0);
    assert!(
        matches!(
            refused,
            Err(InverseLaplaceError::SingularityInsideDisk { .. })
        ),
        "R=3 must refuse the pole at |w|≈2, got {refused:?}"
    );

    // A pole that really does lie outside R = 3 still does not certify the binary64 inversion.
    let a_out = 1.25_f64;
    let sigma_out = 0.5_f64;
    let b_out = 1.5_f64;
    let modulus = exterior_disk_modulus(Complex64::new(-a_out, 0.0), sigma_out, b_out, 3.0)
        .expect("pole at w=13 is outside R=3");
    assert!(modulus > 3.0);
    let exterior_cfg = WeeksConfig::default()
        .with_sigma(sigma_out)
        .with_scale_b(b_out)
        .with_laguerre_degree(32);
    let exterior = weeks_invert(0.8, exterior_cfg, |beta| {
        let val = Complex64::new(1.0, 0.0) / (beta + Complex64::new(a_out, 0.0));
        Ok(ComplexSample::estimated(val, gamma8 * val.norm()))
    })
    .expect("uncertified exterior-pole weeks");
    assert!(!exterior.is_certified());
    assert_eq!(exterior.certified_bar(), None);
    let exact_out = (-a_out * 0.8).exp();
    let err_out = (exterior.value - exact_out).abs();
    assert!(err_out <= exterior.uncertainty_estimate());
    assert!(exterior.accuracy.coefficient_shift >= 0.0);

    let ratio = verify_laguerre_tail_decay(&coeffs_half, 1.0e-14).expect("O(n^-3/2) accepted");
    assert!(ratio <= core::f64::consts::FRAC_1_SQRT_2);
    let (val, round) =
        eval_orthonormal_laguerre_series(&coeffs_half, 1.0, 0.5).expect("laguerre eval");
    assert!(val.is_finite() && round > 0.0);

    let mut nan_first = coeffs_half.clone();
    nan_first[0] = f64::NAN;
    assert!(matches!(
        eval_orthonormal_laguerre_series(&nan_first, 1.0, 0.5),
        Err(InverseLaplaceError::InvalidParameter {
            parameter: "laguerre_coeff",
            ..
        })
    ));
}

#[test]
fn tiny_well_conditioned_system_is_not_refused_by_an_absolute_pivot_floor() {
    let scale = 1.0e-18_f64;
    let a = [
        Complex64::new(scale, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(scale, 0.0),
    ];
    let b = [Complex64::new(scale, 0.0), Complex64::new(2.0 * scale, 0.0)];
    let x = solve_complex_dense(&a, &b, 2).expect("relative pivot floor");
    assert!((x[0].re - 1.0).abs() <= 1.0e-12);
    assert!((x[1].re - 2.0).abs() <= 1.0e-12);
}

#[test]
fn talbot_endpoint_tail_scales_with_the_transform() {
    let energy = 0.8_f64;
    let config = TalbotConfig::default()
        .with_nodes(8)
        .with_min_re_beta(-1.0e6)
        .with_tolerance(1.0);
    let invert = |scale: f64| {
        talbot_invert(energy, config, move |beta| {
            let val = Complex64::new(scale, 0.0) / (beta + Complex64::new(1.25, 0.0));
            Ok(ComplexSample::estimated(val, 0.0))
        })
        .expect("talbot")
    };
    let small = invert(1.0);
    let large = invert(4.0);
    let value_ratio = large.value / small.value;
    let estimate_ratio = large.uncertainty_estimate() / small.uncertainty_estimate();
    assert!(
        (value_ratio - 4.0).abs() <= 1.0e-12,
        "value ratio {value_ratio}"
    );
    assert!(
        (estimate_ratio - 4.0).abs() <= 1.0e-8,
        "estimate ratio {estimate_ratio}"
    );
}

#[test]
fn laguerre_refuses_a_non_finite_derived_scale_or_value() {
    let overflow_scale = eval_orthonormal_laguerre_series(&[1.0], f64::MAX, 0.0);
    assert!(matches!(
        overflow_scale,
        Err(InverseLaplaceError::InvalidParameter {
            parameter: "scale_b",
            ..
        })
    ));
    let overflow_argument = eval_orthonormal_laguerre_series(&[1.0], 1.0e200, 1.0e200);
    assert!(matches!(
        overflow_argument,
        Err(InverseLaplaceError::InvalidParameter {
            parameter: "laguerre_argument",
            ..
        })
    ));
    let (value, roundoff) =
        eval_orthonormal_laguerre_series(&[1.0, 0.0], 1.0, 0.25).expect("finite laguerre");
    assert!(value.is_finite() && roundoff.is_finite());
}

fn binomial_taylor(alpha: f64, len: usize) -> Vec<f64> {
    let mut coeffs = vec![0.0_f64; len];
    coeffs[0] = 1.0;
    for n in 1..len {
        coeffs[n] = coeffs[n - 1] * ((n as f64) - 1.0 - alpha) / (n as f64);
    }
    coeffs
}

#[test]
fn laguerre_tail_accepts_an_exact_three_halves_tail_at_every_length() {
    for len in 16..=40 {
        let coeffs = binomial_taylor(0.5, len);
        let ratio = verify_laguerre_tail_decay(&coeffs, 0.0)
            .unwrap_or_else(|e| panic!("N={len}: (1-w)^(1/2) refused: {e:?}"));
        assert!(
            ratio <= core::f64::consts::FRAC_1_SQRT_2,
            "N={len}: ratio {ratio}"
        );
    }
    for alpha in [0.25_f64, 0.1_f64] {
        for len in 16..=40 {
            let res = verify_laguerre_tail_decay(&binomial_taylor(alpha, len), 0.0);
            assert!(
                matches!(res, Err(InverseLaplaceError::SlowAlgebraicTail { .. })),
                "alpha={alpha} N={len}: {res:?}"
            );
        }
    }
}

#[test]
fn laguerre_tail_skips_exact_zero_terms() {
    for len in [16_usize, 17, 18, 19] {
        let mut coeffs = vec![0.0_f64; len];
        coeffs[0] = 1.0;
        assert_eq!(
            verify_laguerre_tail_decay(&coeffs, 0.0),
            Ok(0.0),
            "terminating series of length {len}"
        );
    }
}

#[test]
fn laguerre_tail_refuses_a_zero_lower_block_under_a_live_upper_block() {
    let mut coeffs = vec![0.0_f64; 16];
    coeffs[0] = 1.0;
    coeffs[8] = 1.0e-300;
    match verify_laguerre_tail_decay(&coeffs, 0.0) {
        Err(InverseLaplaceError::SlowAlgebraicTail { dyadic_ratio, .. }) => {
            assert_eq!(dyadic_ratio, f64::INFINITY);
        }
        other => panic!("zero lower block must refuse, got {other:?}"),
    }
    let mut trailing = vec![0.0_f64; 19];
    trailing[0] = 1.0;
    trailing[18] = 1.0e-300;
    assert!(matches!(
        verify_laguerre_tail_decay(&trailing, 0.0),
        Err(InverseLaplaceError::NonContractingRefinement { .. })
    ));
}

#[test]
fn weeks_degree_must_be_a_multiple_of_four() {
    let gamma4 = higham_gamma(4).expect("gamma4");
    let eval = |beta: Complex64| {
        let val = Complex64::new(1.0, 0.0) / (beta + Complex64::new(1.25, 0.0));
        Ok(ComplexSample::estimated(val, gamma4 * val.norm()))
    };
    for degree in [18_usize, 22, 30] {
        let cfg = WeeksConfig::default().with_laguerre_degree(degree);
        assert!(
            matches!(
                weeks_invert(0.8, cfg, eval),
                Err(InverseLaplaceError::InvalidParameter {
                    parameter: "laguerre_degree",
                    ..
                })
            ),
            "M={degree}"
        );
    }
    let cfg = WeeksConfig::default()
        .with_laguerre_degree(20)
        .with_tolerance(1.0e-3);
    let got = weeks_invert(0.8, cfg, eval).expect("M=20");
    assert!((got.value - (-1.25_f64 * 0.8).exp()).abs() <= got.uncertainty_estimate());
}

#[test]
fn complex_division_of_large_finite_components_stays_finite() {
    let big = 1.5e308_f64;
    let quotient = Complex64::new(big, 0.0) / Complex64::new(big, big);
    assert!(quotient.re.is_finite() && quotient.im.is_finite());
    assert!((quotient.re - 0.5).abs() <= 1.0e-12);
    assert!((quotient.im + 0.5).abs() <= 1.0e-12);

    let huge = 0.75 * f64::MAX;
    let tiny = Complex64::new(1.0, 0.0) / Complex64::new(huge, huge);
    let expect = 0.5 / huge;
    assert!(tiny.re.is_finite() && tiny.im.is_finite());
    assert!((tiny.re / expect - 1.0).abs() <= 1.0e-6);
    assert!((tiny.im / -expect - 1.0).abs() <= 1.0e-6);

    let ordinary = Complex64::new(1.0, 2.0) / Complex64::new(3.0, 4.0);
    assert!((ordinary.re - 0.44).abs() <= 1.0e-15);
    assert!((ordinary.im - 0.08).abs() <= 1.0e-15);
}
