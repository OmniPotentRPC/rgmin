//! A fixed step that reflects through the origin, and the Armijo step
//! that does not.

use rgmin::ArmijoStepper;

/// `φ(x) = x²/2`. Steepest descent with length 2 sends `x` to `-x`.
#[test]
fn armijo_converges_where_a_fixed_step_oscillates() {
    let mut fixed: f64 = 1.0;
    let mut flips = 0;
    for _ in 0..8 {
        let previous = fixed;
        fixed -= 2.0 * fixed;
        if previous * fixed < 0.0 {
            flips += 1;
        }
    }
    assert_eq!(flips, 8, "the length-2 step should change sign every time");
    assert!(
        fixed.abs() > 0.5,
        "the fixed step is still at |x| = {fixed}"
    );

    let mut stepper = ArmijoStepper::gould();
    stepper.alpha_max = 2.0;
    // The first opening trial is the same length-2 step. Armijo has to
    // refuse it and take a shorter one.
    stepper.alpha = 1.0;
    let mut x: f64 = 1.0;
    for _ in 0..8 {
        let slope: f64 = -x * x;
        if !slope.is_finite() || slope.abs() < 1e-24 {
            break;
        }
        let phi0 = 0.5 * x * x;
        let x0 = x;
        let accepted = stepper
            .search(phi0, slope, |alpha| {
                let y = x0 + alpha * (-x0);
                Some(0.5 * y * y)
            })
            .expect("a descent step");
        assert!(
            accepted.alpha < 2.0 - 1e-12,
            "the reflecting step was accepted: {}",
            accepted.alpha
        );
        x += accepted.alpha * (-x0);
    }
    assert!(x.abs() < 1e-8, "Armijo left |x| = {x}");
}

#[test]
fn a_refused_trial_shrinks_and_the_next_opening_doubles() {
    let mut stepper = ArmijoStepper::gould();
    stepper.alpha_max = 1.0;
    assert_eq!(stepper.opening(), 1.0);
    let accepted = stepper
        .search(0.0, -1.0, |alpha| {
            // Refuse everything at least 1/2. φ(α) = -α is linear, so
            // Armijo holds for every positive α once the trial is allowed.
            if alpha >= 0.5 { None } else { Some(-alpha) }
        })
        .expect("a short trial");
    assert!(
        accepted.alpha < 0.5,
        "refused length survived: {}",
        accepted.alpha
    );
    assert_eq!(stepper.opening(), (2.0 * accepted.alpha).min(1.0));
}
