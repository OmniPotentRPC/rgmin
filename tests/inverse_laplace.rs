use rgmin::inverse_laplace::{
    Complex64, ComplexSample, InverseLaplaceError, TalbotConfig, WeeksConfig,
    complex_matrix_inv_inf_norm, eval_orthonormal_laguerre_series, exterior_disk_modulus,
    higham_gamma, solve_complex_dense, talbot_invert, talbot_node, verify_laguerre_tail_decay,
    weeks_invert, weeks_mobius_w,
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
    // The fine upper block of a pole at w = 13 is r^{-n}-amplified roundoff.
    assert_eq!(got_w.accuracy.observed_dyadic_ratio, None);
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
    // exp(-b E) = exp(-750) underflows, so mode 400 cannot be seeded.
    let c400 = vec![1.0_f64; 400];
    assert!(matches!(
        eval_orthonormal_laguerre_series(&c400, 1.5, 500.0),
        Err(InverseLaplaceError::InvalidParameter {
            parameter: "laguerre_start",
            ..
        })
    ));
    let (edge, _) = eval_orthonormal_laguerre_series(&[1.0], 1.0, 700.0).expect("normal start");
    assert!(edge > 0.0 && edge.is_normal());
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
fn talbot_node_refuses_collapsed_or_overflowing_scales() {
    for energy in [f64::MAX, 1.0e-308] {
        assert!(
            matches!(
                talbot_node(1.0, 32, energy),
                Err(InverseLaplaceError::InvalidParameter {
                    parameter: "talbot_scale",
                    ..
                })
            ),
            "E={energy:e}"
        );
    }
    assert!(matches!(
        talbot_node(core::f64::consts::PI.next_down(), 32, 6.4e-300),
        Err(InverseLaplaceError::InvalidParameter {
            parameter: "talbot_node",
            ..
        })
    ));
}

#[test]
fn talbot_node_is_finite_and_accurate_at_tiny_angles() {
    let r = 32.0 / 5.0;
    for theta in [1.0e-200_f64, 1.0e-8, 1.0e-3] {
        let (beta, dbeta) = talbot_node(theta, 32, 1.0).expect("tiny-angle node");
        assert!(beta.is_finite() && dbeta.is_finite(), "theta={theta:e}");
        let t2 = theta * theta;
        let beta_re = r * (1.0 - t2 / 3.0 - t2 * t2 / 45.0);
        let dbeta_re = -r * theta * (2.0 / 3.0 + 4.0 * t2 / 45.0 + 4.0 * t2 * t2 / 315.0);
        assert!(
            (beta.re / beta_re - 1.0).abs() <= 1.0e-15,
            "theta={theta:e}"
        );
        assert!(
            (dbeta.re / dbeta_re - 1.0).abs() <= 1.0e-14,
            "theta={theta:e}: {} vs {dbeta_re}",
            dbeta.re
        );
        assert_eq!(beta.im, r * theta);
        assert_eq!(dbeta.im, r);
    }
    let (beta, dbeta) = talbot_node(1.0e-310, 32, 1.0).expect("subnormal angle");
    assert!(beta.is_finite() && dbeta.is_finite() && dbeta.re <= 0.0);
}

#[test]
fn talbot_shift_encloses_a_pole_right_of_the_unshifted_contour() {
    let energy = 5.0_f64;
    let exact = (2.0 * energy).exp();
    let gamma4 = higham_gamma(4).expect("gamma4");
    let eval = |beta: Complex64| {
        let val = Complex64::new(1.0, 0.0) / (beta - Complex64::new(2.0, 0.0));
        Ok(ComplexSample::estimated(val, gamma4 * val.norm()))
    };
    // Both unshifted contours cross the real axis left of the pole at 2, so the
    // refinement agrees on a value near zero. This is the documented precondition.
    let unshifted = talbot_invert(energy, TalbotConfig::default().with_tolerance(1.0), eval)
        .expect("unshifted contour still returns");
    assert!(unshifted.value.abs() < 1.0e-6 * exact);

    let shifted = talbot_invert(
        energy,
        TalbotConfig::default()
            .with_sigma(3.0)
            .with_tolerance(1.0e-2),
        eval,
    )
    .expect("shifted contour");
    let err = (shifted.value - exact).abs();
    assert!(err <= shifted.uncertainty_estimate(), "err {err:e}");
    assert!(err <= 1.0e-9 * exact, "err {err:e}");

    assert!(matches!(
        talbot_invert(energy, TalbotConfig::default().with_sigma(f64::NAN), eval),
        Err(InverseLaplaceError::InvalidParameter {
            parameter: "sigma",
            ..
        })
    ));
}

fn rational_sample(num: Complex64, den: Complex64) -> Result<ComplexSample, InverseLaplaceError> {
    let val = num / den;
    let gamma = higham_gamma(8).expect("gamma8");
    Ok(ComplexSample::estimated(val, gamma * val.norm()))
}

#[test]
fn talbot_refuses_declared_singularities_outside_the_contour() {
    // cos(20 t) at t = 10 with M = 32: the coarse contour reaches |Im| ~ 1.0, the poles
    // sit at ±20i.
    let cos20 = |beta: Complex64| rational_sample(beta, beta * beta + Complex64::new(400.0, 0.0));
    let declared = TalbotConfig::default()
        .with_tolerance(1.0)
        .with_singularity_re_max(0.0)
        .with_singularity_im_max(20.0);
    match talbot_invert(10.0, declared, cos20) {
        Err(InverseLaplaceError::ContourMissesSingularity {
            contour_re,
            contour_im_limit,
            ..
        }) => {
            assert_eq!(contour_re, f64::NEG_INFINITY);
            assert!(contour_im_limit < 20.0);
        }
        other => panic!("cos(20t) must refuse, got {other:?}"),
    }

    let sin2 = |beta: Complex64| {
        rational_sample(
            Complex64::new(2.0, 0.0),
            beta * beta + Complex64::new(4.0, 0.0),
        )
    };
    assert!(matches!(
        talbot_invert(
            10.0,
            TalbotConfig::default()
                .with_tolerance(1.0)
                .with_singularity_re_max(0.0)
                .with_singularity_im_max(2.0),
            sin2
        ),
        Err(InverseLaplaceError::ContourMissesSingularity { .. })
    ));

    let pole2 = |beta: Complex64| {
        rational_sample(Complex64::new(1.0, 0.0), beta - Complex64::new(2.0, 0.0))
    };
    match talbot_invert(
        5.0,
        TalbotConfig::default()
            .with_tolerance(1.0)
            .with_singularity_re_max(2.0),
        pole2,
    ) {
        Err(InverseLaplaceError::ContourMissesSingularity { contour_re, .. }) => {
            assert!(contour_re < 2.0);
        }
        other => panic!("1/(s-2) at E=5 must refuse, got {other:?}"),
    }
    let shifted = talbot_invert(
        5.0,
        TalbotConfig::default()
            .with_sigma(3.0)
            .with_tolerance(1.0e-2)
            .with_singularity_re_max(2.0),
        pole2,
    )
    .expect("sigma = 3 encloses the pole at 2");
    assert!((shifted.value - 10.0_f64.exp()).abs() <= 1.0e-9 * 10.0_f64.exp());

    let cos2 = |beta: Complex64| rational_sample(beta, beta * beta + Complex64::new(4.0, 0.0));
    let enclosed = talbot_invert(
        1.0,
        TalbotConfig::default()
            .with_nodes(64)
            .with_tolerance(1.0e-8)
            .with_singularity_re_max(0.0)
            .with_singularity_im_max(2.0),
        cos2,
    )
    .expect("M = 64 at t = 1 encloses ±2i");
    let err = (enclosed.value - 2.0_f64.cos()).abs();
    assert!(
        err <= enclosed.uncertainty_estimate() && err <= 1.0e-9,
        "err {err:e}"
    );

    for (re, im, parameter) in [
        (Some(f64::NAN), None, "singularity_re_max"),
        (None, Some(-1.0), "singularity_im_max"),
    ] {
        let mut cfg = TalbotConfig::default();
        cfg = cfg.with_singularity_re_max(re).with_singularity_im_max(im);
        let got = talbot_invert(1.0, cfg, cos2);
        assert!(
            matches!(got, Err(InverseLaplaceError::InvalidParameter { parameter: p, .. }) if p == parameter),
            "{parameter}: {got:?}"
        );
    }
}

#[test]
fn weeks_refuses_a_mode_that_aliases_onto_both_grids() {
    let cfg = WeeksConfig::default();
    let (sigma, b, energy) = (0.5_f64, 1.5_f64, 0.8_f64);
    for mode in [192_usize, 384] {
        // Phi(w) = w^mode, so F(beta) = (1 - w) w^mode / sqrt(2b) inverts to
        // exp(sigma E) phi_mode(E; b).
        let mut coeffs = vec![0.0_f64; mode + 1];
        coeffs[mode] = 1.0;
        let (phi_mode, _) =
            eval_orthonormal_laguerre_series(&coeffs, b, energy).expect("direct mode");
        let exact = (sigma * energy).exp() * phi_mode;
        assert!(exact.abs() > 1.0e-3, "mode {mode}: exact {exact}");
        let got = weeks_invert(energy, cfg, |beta| {
            let w = weeks_mobius_w(beta, sigma, b)?;
            let mut wn = Complex64::new(1.0, 0.0);
            for _ in 0..mode {
                wn = wn * w;
            }
            let val = (Complex64::new(1.0, 0.0) - w) * wn / (2.0 * b).sqrt();
            Ok(ComplexSample::estimated(val, 0.0))
        });
        assert!(
            matches!(got, Err(InverseLaplaceError::AliasedLaguerreModes { .. })),
            "mode {mode} (exact {exact}) must refuse, got {got:?}"
        );
    }
}

#[test]
fn complex_sqrt_and_solver_edges_stay_typed() {
    let big = Complex64::new(1.0e308, 1.0e308);
    let root = big.sqrt();
    assert!(root.is_finite());
    assert!(((root * root) - big).norm() <= 1.0e-12 * big.norm());

    let eye = [
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];
    let bad_rhs = [Complex64::new(f64::NAN, 0.0), Complex64::new(1.0, 0.0)];
    assert!(matches!(
        solve_complex_dense(&eye, &bad_rhs, 2),
        Err(InverseLaplaceError::InvalidParameter {
            parameter: "rhs_entry",
            ..
        })
    ));
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
