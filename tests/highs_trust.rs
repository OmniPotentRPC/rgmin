//! `set_highs_trust` stores the radius `set_trust` already uses.

use rgmin::{Control, Method, Solver};

#[test]
fn set_highs_trust_follows_set_trust() {
    let mut solver = Solver::new(Method::lbfgs(), Control::default(), 2);
    for radius in [0.25, 0.0, -1.0, f64::NAN] {
        assert_eq!(
            solver.set_highs_trust(radius),
            solver.set_trust(radius),
            "radius {radius}"
        );
    }
}
