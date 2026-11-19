use ndarray::array;
use rgmin::{GradNorm, Lbfgs, LbfgsStep};

#[test]
fn builder_normalizes_memory_caps_and_fixed_scales() {
    let configured = Lbfgs::builder()
        .memory(0)
        .tolerance(0.25)
        .norm(GradNorm::Euclidean)
        .ftol(-1.0)
        .capped(f64::NAN)
        .fixed_scale(-2.0)
        .build();
    assert_eq!(configured.max_pairs, 1);
    assert_eq!(configured.gtol, 0.25);
    assert_eq!(configured.norm, GradNorm::Euclidean);
    assert_eq!(configured.ftol, -1.0);
    assert_eq!(configured.step, LbfgsStep::Wolfe);
    assert_eq!(configured.fixed_h0, None);

    let configured = Lbfgs::builder().scaled(0.5).fixed_scale(0.25).build();
    assert_eq!(configured.step, LbfgsStep::scaled(0.5));
    assert_eq!(configured.fixed_h0, Some(0.25));
}

#[test]
fn direct_constructors_reject_unusable_caps() {
    for solver in [
        Lbfgs::capped(0.0),
        Lbfgs::capped(f64::INFINITY),
        Lbfgs::scaled(-1.0),
        Lbfgs::scaled(f64::NAN),
    ] {
        assert_eq!(solver.step, LbfgsStep::Wolfe);
    }
    assert_eq!(Lbfgs::capped(0.4).step, LbfgsStep::capped(0.4));
    assert_eq!(Lbfgs::scaled(0.3).step, LbfgsStep::scaled(0.3));
}

#[test]
fn rejected_curvature_does_not_displace_a_retained_pair() {
    let mut solver = Lbfgs::with_capacity(1);
    solver.record(array![1.0], array![2.0]);
    assert_eq!(solver.len(), 1);

    for y in [array![0.0], array![-1.0], array![f64::NAN]] {
        solver.record(array![1.0], y);
        assert_eq!(solver.len(), 1);
    }
    assert_eq!(solver.two_loop(array![3.0].view()), array![-1.5]);

    solver.record(array![2.0], array![8.0]);
    assert_eq!(solver.len(), 1);
    assert_eq!(solver.two_loop(array![4.0].view()), array![-1.0]);
}

#[test]
fn large_extra_update_counts_use_the_same_finite_direction() {
    let mut solver = Lbfgs::default();
    solver.record(array![1.0, 0.0], array![2.0, 0.0]);
    solver.extra_updates = 80;
    let gradient = array![3.0, 1.0];
    let direction = solver.two_loop(gradient.view());
    assert!(direction.iter().all(|value| value.is_finite()));
    assert!(gradient.dot(&direction) < 0.0);
}

#[test]
fn empty_memory_uses_only_positive_finite_initial_scales() {
    for scale in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let mut solver = Lbfgs::default();
        solver.inverse_curvature = scale;
        assert_eq!(solver.two_loop(array![2.0].view()), array![-2.0]);
    }

    let solver = Lbfgs::builder().fixed_scale(0.5).build();
    assert_eq!(solver.two_loop(array![2.0].view()), array![-1.0]);
}
