//! `x^2 + y^2` from `(1, -2)`. The value ends below `1e-8`.

use ndarray::array;
use rgmin::Lbfgs;

fn main() {
    let mut opt = Lbfgs::default();
    let (f, x, _) = opt.minimize(array![1.0, -2.0].view(), 40, |x| {
        Some((x[0] * x[0] + x[1] * x[1], array![2.0 * x[0], 2.0 * x[1]]))
    });
    assert!(f < 1e-8, "f {f}");
    assert!(x[0].abs() < 1e-4 && x[1].abs() < 1e-4, "{x}");
    println!("quadratic {f:.3e}");
}
