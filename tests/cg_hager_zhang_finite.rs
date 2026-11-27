//! Hager-Zhang beta is zero when the formula overflows.

use ndarray::array;
use rgmin::{Conjugacy, ConjugacyContext};

#[test]
fn an_overflowed_hager_zhang_coefficient_is_a_restart() {
    let g = array![1e200, 1.0];
    let gold = array![0.0, 0.0];
    let d = array![0.0, 1.0];
    let ctx = ConjugacyContext {
        current_gradient: g.view(),
        previous_gradient: gold.view(),
        previous_direction: d.view(),
    };
    assert_eq!(Conjugacy::HagerZhang.beta(&ctx), 0.0);
    assert_eq!(
        Conjugacy::HagerZhang.beta_metric(&ctx, g.view(), gold.view()),
        0.0
    );
}
