//! Session stop norms under a HiGHS box measure the projected gradient.
#![cfg(feature = "highs")]

use ndarray::{Array1, ArrayView1, array};
use rgmin::{Control, Method, Oracle, Solver, StopNorm};

/// Atom 0 sits on its lower x bound with an outward gradient of 1, so the
/// projection drops that component. Unprojected, every norm is at least 1.
#[test]
fn box_stop_measures_the_projected_gradient() {
    let start = array![1.0, 0.0, 0.0, 0.3, 0.4, 0.0, 0.3, 0.0, 0.0];
    let mut lo = vec![f64::NEG_INFINITY; 9];
    lo[0] = 1.0;
    for (norm, stops_at) in [
        (StopNorm::Two, None),
        (StopNorm::MaxAtom, Some(0.5)),
        (StopNorm::Inf, Some(0.4)),
    ] {
        let obj = Oracle::unbounded(9, |x: ArrayView1<f64>| (0.5 * x.dot(&x), x.to_owned()));
        let control = Control {
            maxiter: 10,
            gtol: 0.55,
            istep: 0.1,
            maxmove: None,
            ..Default::default()
        };
        let mut s = Solver::new(Method::Lbfgs { memory: 5 }, control, 9);
        s.set_highs(true);
        assert!(s.set_box(Some(lo.clone()), None));
        s.set_stop_norm(norm);
        let mut x: Array1<f64> = start.clone();
        let rep = s.step(&obj, &mut x).unwrap();
        match stops_at {
            Some(want) => {
                assert_eq!(rep.steps, 0, "{norm:?}");
                assert_eq!(x, start, "{norm:?}");
                assert!(
                    (rep.grad_norm - want).abs() < 1e-15,
                    "{norm:?}: {}",
                    rep.grad_norm
                );
            }
            None => {
                assert_eq!(rep.steps, 1, "{norm:?}");
                assert_ne!(x, start, "{norm:?}");
            }
        }
    }
}
