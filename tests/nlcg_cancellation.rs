use approx::assert_relative_eq;
use ndarray::array;
use rgmin::{Conjugacy, ConjugacyContext};

#[test]
fn conjugacy_resolves_a_small_change_beside_a_large_common_gradient() {
    let current = array![1e8, 1.0];
    let previous = array![1e8, 0.0];
    let direction = array![-1e8, -1.0];
    let context = ConjugacyContext {
        current_gradient: current.view(),
        previous_gradient: previous.view(),
        previous_direction: direction.view(),
    };

    // The gradient change is one unit along the second coordinate.
    assert_relative_eq!(Conjugacy::LiuStorey.beta(&context), 1e-16, epsilon = 1e-30);
    assert_relative_eq!(
        Conjugacy::PolakRibiere.beta(&context),
        1e-16,
        epsilon = 1e-30
    );
    assert_relative_eq!(
        Conjugacy::HestenesStiefel.beta(&context),
        -1.0,
        epsilon = 1e-14
    );
    assert_relative_eq!(
        Conjugacy::HagerZhang.beta(&context),
        2e16,
        max_relative = 1e-14
    );
}
