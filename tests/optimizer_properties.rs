//! L-BFGS on random convex quadratics, and Rosenbrock from a box of starts.

use ndarray::{Array1, ArrayView1};
use proptest::prelude::*;
use rgmin::{Lbfgs, LbfgsStep};

fn quadratic(x: ArrayView1<f64>, target: &[f64], weight: &[f64]) -> Option<(f64, Array1<f64>)> {
    let mut f = 0.0;
    let mut g = Array1::zeros(x.len());
    for i in 0..x.len() {
        let d = x[i] - target[i];
        f += 0.5 * weight[i] * d * d;
        g[i] = weight[i] * d;
    }
    Some((f, g))
}

fn rosen(x: ArrayView1<f64>) -> Option<(f64, Array1<f64>)> {
    let a = 1.0 - x[0];
    let b = x[1] - x[0] * x[0];
    Some((
        a * a + 100.0 * b * b,
        ndarray::array![-2.0 * a - 400.0 * x[0] * b, 200.0 * b],
    ))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn capped_lbfgs_reaches_a_separable_quadratic(
        target in prop::array::uniform4(-3.0f64..3.0),
        weight in prop::array::uniform4(0.2f64..6.0),
        start in prop::array::uniform4(-3.0f64..3.0),
    ) {
        let mut opt = Lbfgs::default();
        opt.gtol = 1e-6;
        let (f, x, _) = opt.minimize(ArrayView1::from(&start), 80, |v| {
            quadratic(v, &target, &weight)
        });
        prop_assert!(f < 1e-8, "f {f}");
        for i in 0..4 {
            prop_assert!((x[i] - target[i]).abs() < 1e-4, "x {x} target {target:?}");
        }
    }

    #[test]
    fn wolfe_lbfgs_reaches_rosenbrock(
        start in prop::array::uniform2(-1.5f64..1.5),
    ) {
        let mut opt = Lbfgs::default();
        opt.step = LbfgsStep::Wolfe;
        opt.gtol = 1e-6;
        let (f, x, _) = opt.minimize(ArrayView1::from(&start), 400, rosen);
        prop_assert!(f < 1e-6, "f {f} from {start:?} at {x}");
        prop_assert!((x[0] - 1.0).abs() < 1e-3 && (x[1] - 1.0).abs() < 1e-3, "{x}");
    }
}
