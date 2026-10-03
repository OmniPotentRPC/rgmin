//! Exhausted standalone zoom retains its measured finite Armijo endpoint.

use ndarray::{Array1, ArrayView1, array};
use rgmin::linesearch::zoom;

fn bounded_linear(x: ArrayView1<'_, f64>) -> (f64, Array1<f64>) {
    if x[0] < -0.1 {
        (f64::INFINITY, array![f64::NAN])
    } else {
        (x[0], array![1.0])
    }
}

#[test]
fn zoom_exhaustion_retains_the_feasible_armijo_endpoint() {
    let alpha = zoom(
        &mut bounded_linear,
        array![0.0].view(),
        array![-1.0].view(),
        0.0,
        1.0,
        1e-4,
        0.9,
        6,
    );
    assert_eq!(alpha, 3.0 / 32.0);
    let (value, gradient) = bounded_linear(array![-alpha].view());
    assert!(value.is_finite() && gradient[0].is_finite());
    assert!(value <= -1e-4 * alpha);
}
