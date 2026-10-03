//! A root-search tolerance does not enlarge the accepted trust ball.

use ndarray::{Array2, array};
use rgmin::{Error, TrustRegion};

#[test]
fn a_zero_trust_radius_returns_exactly_zero_even_for_a_tiny_gradient() {
    let hessian = Array2::eye(2);
    let gradient = array![1e-12, -2e-12];
    let result = TrustRegion::new(0.0).restrict_qn(&hessian, &gradient).unwrap();
    assert_eq!(result.step, array![0.0, 0.0]);
    assert_eq!(result.nrm2(), 0.0);
    assert_eq!(result.cons, 0.0);
}

#[test]
fn root_tolerance_never_authorizes_a_step_outside_the_radius() {
    let hessian = Array2::eye(2);
    for radius in [1e-12, 0.1, 1.0] {
        let gradient = array![radius + 5e-11, 0.0];
        let result = TrustRegion::new(radius).restrict_qn(&hessian, &gradient).unwrap();
        assert!(result.nrm2() <= radius, "radius={radius}, step={result:?}");
        assert_eq!(result.cons, radius);
        assert!(result.step[0] < 0.0);
    }
}

#[test]
fn dense_trust_rejects_nonfinite_values_and_malformed_controls() {
    let finite_hessian = Array2::eye(2);
    let finite_gradient = array![1.0, 2.0];
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(TrustRegion::new(bad).restrict_qn(&finite_hessian, &finite_gradient), Err(Error::RestrictedStep)));
        let mut hessian = finite_hessian.clone();
        hessian[(0, 0)] = bad;
        assert!(matches!(TrustRegion::new(1.0).restrict_qn(&hessian, &finite_gradient), Err(Error::RestrictedStep)));
        let gradient = array![bad, 2.0];
        assert!(matches!(TrustRegion::new(1.0).restrict_qn(&finite_hessian, &gradient), Err(Error::RestrictedStep)));
        let mut trust = TrustRegion::new(1.0);
        trust.delta = bad;
        assert!(matches!(trust.restrict_qn(&finite_hessian, &finite_gradient), Err(Error::RestrictedStep)));
        trust = TrustRegion::new(1.0);
        trust.tol = bad;
        assert!(matches!(trust.restrict_qn(&finite_hessian, &finite_gradient), Err(Error::RestrictedStep)));
    }
    assert!(matches!(TrustRegion::new(f64::NAN).restrict_qn(&finite_hessian, &finite_gradient), Err(Error::RestrictedStep)));
    let mut trust = TrustRegion::new(1.0);
    trust.delta = -1.0;
    assert!(matches!(trust.restrict_qn(&finite_hessian, &finite_gradient), Err(Error::RestrictedStep)));
    trust = TrustRegion::new(1.0);
    trust.tol = -1.0;
    assert!(matches!(trust.restrict_qn(&finite_hessian, &finite_gradient), Err(Error::RestrictedStep)));
}
