//! Explicit coordinate boxes for direct L-BFGS model steps.
#![cfg(feature = "highs")]
use approx::assert_relative_eq;
use ndarray::{Array1, array};
use rgmin::{HighsStep, Lbfgs};

#[test]
fn box_projection_keeps_free_components_near_a_wall() {
    use ndarray::array;

    let mut opt = Lbfgs::default();
    opt.highs = Some(HighsStep::default());
    for distance in [1e-8, 1e-12] {
        let x = array![3.0, distance];
        let gradient = array![3.0, 1.0 + distance];
        let step = opt
            .highs_step_boxed(
                x.view(),
                gradient.view(),
                Some(&[f64::NEG_INFINITY, 0.0]),
                None,
            )
            .unwrap();
        assert_relative_eq!(step[0], -3.0, epsilon = 1e-14);
        assert_relative_eq!(step[1], -distance, epsilon = 1e-16);
        assert!(gradient.dot(&step) < 0.0);
    }
}

#[test]
fn box_projection_restarts_a_non_descent_curvature_direction() {
    use ndarray::array;

    let mut opt = Lbfgs::default();
    opt.record(array![1.0, 2.0], array![1.0, 0.0]);
    opt.highs = Some(HighsStep::default());
    let x = array![0.0, 0.0];
    let gradient = array![1.0, -1.0];
    let step = opt
        .highs_step_boxed(x.view(), gradient.view(), None, Some(&[f64::INFINITY, 0.0]))
        .unwrap();
    assert!(gradient.dot(&step) < 0.0, "non-descent step: {step:?}");
    assert_relative_eq!(step[0], -1.0, epsilon = 1e-14);
    assert_eq!(step[1], 0.0);
}

#[test]
fn per_coordinate_box_is_independent() {
    let mut opt = Lbfgs::default();
    opt.highs = Some(
        HighsStep::default()
            .with_trust(None)
            .with_equalities(Vec::new())
            .with_center_axes(None),
    );
    let x = Array1::from(vec![0.0, 0.0]);
    let g = Array1::from(vec![10.0, 10.0]);
    let d = opt
        .highs_step_boxed(x.view(), g.view(), Some(&[-0.1, -10.0]), Some(&[0.1, 10.0]))
        .unwrap();
    for i in 0..2 {
        let t = x[i] + d[i];
        let (lo, hi) = if i == 0 { (-0.1, 0.1) } else { (-10.0, 10.0) };
        assert!(t >= lo - 1e-9 && t <= hi + 1e-9, "left the box: {t}");
    }
}

#[test]
fn per_coord_box_is_not_uniform() {
    let mut opt = Lbfgs::default();
    opt.highs = Some(
        HighsStep::default()
            .with_trust(None)
            .with_equalities(Vec::new())
            .with_center_axes(None),
    );
    let x = Array1::from(vec![0.0, 0.0]);
    let g = Array1::from(vec![1.0, 100.0]);
    let d = opt
        .highs_step_boxed(
            x.view(),
            g.view(),
            Some(&[-0.05, -10.0]),
            Some(&[0.05, 10.0]),
        )
        .unwrap();
    assert!(
        (x[0] + d[0]).abs() <= 0.05 + 1e-9,
        "tight axis left the box: {}",
        x[0] + d[0]
    );
    assert!(
        (x[1] + d[1]).abs() > 0.5,
        "wide axis was clipped as if uniform: {}",
        x[1] + d[1]
    );
    assert!(
        (x[1] + d[1]).abs() <= 10.0 + 1e-9,
        "wide axis left its own box: {}",
        x[1] + d[1]
    );
}

#[test]
fn null_side_is_unbounded_on_that_side() {
    let mut opt = Lbfgs::default();
    opt.highs = Some(
        HighsStep::default()
            .with_trust(None)
            .with_equalities(Vec::new())
            .with_center_axes(None),
    );
    let x = Array1::from(vec![0.5, 0.5]);
    let g = Array1::from(vec![-10.0, -1.0]);
    let d = opt
        .highs_step_boxed(x.view(), g.view(), Some(&[0.0]), None)
        .unwrap();
    assert!(
        x[0] + d[0] >= -1e-12 && x[1] + d[1] >= -1e-12,
        "lower side must hold: {:?}",
        (x[0] + d[0], x[1] + d[1])
    );
    assert!(
        d[0] > 5.0 && d[1] > 0.5,
        "unbounded upper must not clip ascent: {d:?}"
    );
}

#[test]
fn boxed_step_intersects_uniform_bounds_and_preserves_equalities() {
    let mut opt = Lbfgs::default();
    opt.highs = Some(
        HighsStep::default()
            .with_trust(0.8)
            .with_lo(-1.0)
            .with_hi(1.0)
            .with_equalities(vec![(vec![(0, 1.0), (1, 1.0)], 0.4)])
            .with_center_axes(None),
    );
    let x = array![0.0, 0.0];
    let step = opt
        .highs_step_boxed(
            x.view(),
            array![1.0, 2.0].view(),
            Some(&[-0.5]),
            Some(&[0.5]),
        )
        .unwrap();
    assert_relative_eq!(step[0], 0.5, epsilon = 1e-7);
    assert_relative_eq!(step[1], -0.1, epsilon = 1e-7);
    assert!((step.sum() - 0.4).abs() <= 1e-7);
    assert!(step.iter().all(|p| (-0.5..=0.5).contains(p)));
    opt.highs.as_mut().unwrap().equalities.clear();
    opt.highs.as_mut().unwrap().trust = Some(0.0);
    assert_eq!(
        opt.highs_step_boxed(x.view(), array![1.0, 2.0].view(), None, None)
            .unwrap(),
        x
    );
}

#[test]
fn bounded_step_rejects_malformed_domains_without_changing_history() {
    let mut opt = Lbfgs::default();
    opt.highs = Some(HighsStep::default());
    opt.record(array![1.0, 0.0], array![2.0, 0.0]);
    let x = array![0.0, 0.0];
    let gradient = array![1.0, 2.0];
    for bounds in [vec![0.0, 0.0, 0.0], vec![f64::NAN], vec![1.0]] {
        assert!(
            opt.highs_step_boxed(x.view(), gradient.view(), Some(&bounds), None)
                .is_err()
        );
        assert_eq!(opt.len(), 1);
    }
    assert!(
        opt.highs_step_boxed(array![0.0].view(), gradient.view(), None, None)
            .is_err()
    );
    assert!(
        opt.highs_step_boxed(x.view(), array![f64::NAN, 0.0].view(), None, None)
            .is_err()
    );
    assert_eq!(opt.len(), 1);
}

#[test]
fn bounded_steps_require_configuration_and_matching_dimensions() {
    let opt = Lbfgs::default();
    let err = opt
        .highs_step_boxed(array![0.0].view(), array![1.0].view(), None, None)
        .unwrap_err();
    assert!(err.to_string().contains("HighsStep"), "{err}");
    let err = opt
        .highs_step(array![0.0].view(), array![1.0].view())
        .unwrap_err();
    assert!(err.to_string().contains("HighsStep"), "{err}");

    let mut configured = Lbfgs::default();
    configured.highs = Some(HighsStep::default());
    let err = configured
        .highs_step(array![0.0].view(), array![1.0, 2.0].view())
        .unwrap_err();
    assert!(
        err.to_string().contains("length 1") && err.to_string().contains("dim 2"),
        "{err}"
    );
}

#[test]
fn bounded_steps_report_nonfinite_and_inconsistent_controls() {
    let x = array![0.0, 0.0];
    let gradient = array![1.0, 2.0];
    let invalid_steps = [
        HighsStep::default().with_trust(f64::INFINITY),
        HighsStep::default().with_trust(-1.0),
        HighsStep::default().with_lo(f64::NAN),
        HighsStep::default().with_hi(f64::NAN),
        HighsStep::default().with_center_axes((0, 2)),
        HighsStep::default().with_center_axes((1, 3)),
    ];
    for step in invalid_steps {
        let mut opt = Lbfgs::default();
        opt.highs = Some(step);
        let err = opt
            .highs_step_boxed(x.view(), gradient.view(), None, None)
            .unwrap_err();
        assert!(
            err.to_string().contains("invalid bounded-step data"),
            "{err}"
        );
    }

    let mut opt = Lbfgs::default();
    opt.highs = Some(HighsStep::default());
    for (point, grad) in [
        (array![f64::NAN, 0.0], gradient.clone()),
        (x.clone(), array![f64::INFINITY, 0.0]),
    ] {
        let err = opt
            .highs_step_boxed(point.view(), grad.view(), None, None)
            .unwrap_err();
        assert!(
            err.to_string().contains("invalid bounded-step data"),
            "{err}"
        );
    }
}

#[test]
fn bounded_steps_report_bad_coordinate_boxes() {
    let mut opt = Lbfgs::default();
    opt.highs = Some(HighsStep::default());
    let x = array![0.0, 0.0];
    let gradient = array![1.0, 2.0];

    let err = opt
        .highs_step_boxed(x.view(), gradient.view(), Some(&[f64::NAN]), None)
        .unwrap_err();
    assert!(err.to_string().contains("invalid coordinate box"), "{err}");

    let err = opt
        .highs_step_boxed(x.view(), gradient.view(), Some(&[1.0]), None)
        .unwrap_err();
    assert!(err.to_string().contains("outside coordinate box"), "{err}");
}

#[test]
fn bounded_steps_report_bad_linear_equalities() {
    let x = array![0.0, 0.0];
    let gradient = array![1.0, 2.0];
    let invalid_equalities = [
        (vec![(2, 1.0)], 0.0),
        (vec![(0, f64::NAN)], 0.0),
        (vec![(0, 1.0)], f64::INFINITY),
    ];
    for equality in invalid_equalities {
        let mut opt = Lbfgs::default();
        opt.highs = Some(HighsStep::default().with_equalities(vec![equality]));
        let err = opt
            .highs_step_boxed(x.view(), gradient.view(), None, None)
            .unwrap_err();
        assert!(err.to_string().contains("invalid linear equality"), "{err}");
    }
}
