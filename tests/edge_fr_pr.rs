//! Edge cases for Fletcher-Reeves Polak-Ribiere conjugate gradients.
//!
//! Quadratic, Rosenbrock, a zero gradient, a non-finite energy, a one-step
//! budget, and a one-dimensional parabola.

#[path = "edge_common/mod.rs"]
mod edge;

#[test]
fn reaches_the_quadratic_minimum() {
    edge::reaches_the_quadratic_minimum(&edge::fr_pr());
}

#[test]
fn reaches_the_rosenbrock_minimum() {
    edge::reaches_the_rosenbrock_minimum(&edge::fr_pr());
}

#[test]
fn a_zero_gradient_stops_at_the_known_minimum() {
    edge::a_zero_gradient_stops_at_the_known_minimum(&edge::fr_pr());
}

#[test]
fn a_nonfinite_energy_is_not_reported_as_success() {
    edge::a_nonfinite_energy_is_not_reported_as_success(&edge::fr_pr());
}

#[test]
fn one_rosenbrock_step_stops_on_the_iteration_budget() {
    edge::one_rosenbrock_step_stops_on_the_iteration_budget(&edge::fr_pr());
}

#[test]
fn solves_a_one_dimensional_parabola() {
    edge::solves_a_one_dimensional_parabola(&edge::fr_pr());
}
