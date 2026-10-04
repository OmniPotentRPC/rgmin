use ndarray::{Array1, ArrayView1, array};
use rgmin::LineSearch;

fn bounded_linear(x: ArrayView1<'_, f64>) -> (f64, Array1<f64>) {
    if x[0] < -0.1 {
        (f64::INFINITY, array![f64::NAN])
    } else {
        (x[0], array![1.0])
    }
}

#[test]
fn wolfe_domain_boundary_keeps_a_measured_descent() {
    let search = LineSearch::Wolfe {
        c1: 1e-4,
        c2: 0.9,
        maxiter: 6,
    };
    let (point, value, alpha) =
        search.search(bounded_linear, array![0.0].view(), array![-1.0].view(), 1.0);
    assert_eq!(point[0], -3.0 / 32.0);
    assert_eq!(value, point[0]);
    assert_eq!(alpha, 3.0 / 32.0);
}
