use ndarray::{Array1, array};
use rgmin::manifold::*;


#[test]
fn oblique_retract_stays_on_product_of_spheres() {
    let m = Oblique::new(3, 2);
    let x = array![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let v = array![0.1, 0.2, -0.3, 0.4, 0.0, -0.2];
    let y = m.retract(&x, &v);
    let n0 = (y[0] * y[0] + y[1] * y[1] + y[2] * y[2]).sqrt();
    let n1 = (y[3] * y[3] + y[4] * y[4] + y[5] * y[5]).sqrt();
    assert!((n0 - 1.0).abs() < 1e-14, "left sphere 0 {y:?}");
    assert!((n1 - 1.0).abs() < 1e-14, "left sphere 1 {y:?}");
    let t = m.project(&x, &v);
    let d0 = x[0] * t[0] + x[1] * t[1] + x[2] * t[2];
    let d1 = x[3] * t[3] + x[4] * t[4] + x[5] * t[5];
    assert!(d0.abs() < 1e-14);
    assert!(d1.abs() < 1e-14);
    let w = m.transport(&x, &y, &v);
    let p = m.project(&y, &v);
    assert!((&w - &p).mapv(f64::abs).sum() < 1e-14);
}

#[test]
fn multinomial_retract_stays_on_the_simplex() {
    let x = array![0.2, 0.3, 0.5];
    let v = array![0.1, -0.05, -0.05];
    let y = Multinomial.retract(&x, &v);
    assert!(y.iter().all(|&yi| yi > 0.0), "left the interior {y:?}");
    let s: f64 = y.iter().copied().sum();
    assert!((s - 1.0).abs() < 1e-14, "sum {s} y={y:?}");
    let t = Multinomial.project(&x, &v);
    let ts: f64 = t.iter().copied().sum();
    assert!(ts.abs() < 1e-14, "not tangent 1^T t = {ts}");
}

#[test]
fn stiefel_np_retract_stays_orthonormal() {
    let st = StiefelNp::new(4, 2).unwrap();
    let x = array![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let v = st.project(&x, &array![0.1, 0.0, 0.2, 0.0, 0.0, 0.1, 0.0, -0.2]);
    let y = st.retract(&x, &v);
    for a in 0..2 {
        for b in 0..2 {
            let mut acc = 0.0;
            for i in 0..4 {
                acc += y[i + 4 * a] * y[i + 4 * b];
            }
            let want = if a == b { 1.0 } else { 0.0 };
            assert!((acc - want).abs() < 1e-12, "Y^T Y[{a},{b}]={acc}");
        }
    }
    let nrm = y.iter().map(|a| a * a).sum::<f64>().sqrt();
    assert!((nrm - 1.0).abs() > 1e-6);
}

#[test]
fn spd_retract_stays_on_the_set() {
    let x = array![2.0, 0.2, 0.2, 3.0];
    let v = array![0.0, 0.15, 0.15, -0.1];
    let y = Spd.retract(&x, &v);
    assert!(is_spd(&y), "left the SPD set {y:?}");
    let t = Spd.project(&x, &v);
    assert!((t[1] - t[2]).abs() < 1e-15);
    let w = Spd.transport(&x, &y, &v);
    assert!((w[1] - w[2]).abs() < 1e-15);
}

#[test]
fn complex_circle_retract_stays_on_the_set() {
    let m = ComplexCircle::new(2);
    let x = array![1.0, 0.0, 0.0, 1.0];
    let v = m.project(&x, &array![0.2, -0.1, 0.3, 0.4]);
    let y = m.retract(&x, &v);
    let n0 = (y[0] * y[0] + y[1] * y[1]).sqrt();
    let n1 = (y[2] * y[2] + y[3] * y[3]).sqrt();
    assert!((n0 - 1.0).abs() < 1e-14, "left circle 0 {y:?}");
    assert!((n1 - 1.0).abs() < 1e-14, "left circle 1 {y:?}");
    let fro = y.iter().map(|a| a * a).sum::<f64>().sqrt();
    assert!((fro - 1.0).abs() > 0.3, "must not be the sphere {y:?}");
}

#[test]
fn symmetric_retract_stays_on_the_set() {
    let x = array![1.0, 0.0, 0.0, -1.0];
    let v = array![0.0, 0.2, -0.1, 0.0];
    let y = Symmetric.retract(&x, &v);
    assert!(is_symmetric(&y), "left the symmetric set {y:?}");
    assert!((y[1] - y[2]).abs() < 1e-15);
    let t = Symmetric.project(&x, &v);
    assert!((t[1] - t[2]).abs() < 1e-15);
    let w = Symmetric.transport(&x, &y, &v);
    assert!((w[1] - 0.2).abs() < 1e-15);
    assert!((w[2] + 0.1).abs() < 1e-15);
    let det = y[0] * y[3] - y[1] * y[2];
    assert!(det < 0.0, "must not force SPD {y:?}");
}

#[test]
fn skewsymmetric_retract_stays_on_the_set() {
    let x = array![0.0, 1.0, -1.0, 0.0];
    let v = array![0.2, 0.3, -0.1, -0.4];
    let y = SkewSymmetric.retract(&x, &v);
    assert!(is_skewsymmetric(&y), "left the skew-symmetric set {y:?}");
    assert!((y[0]).abs() < 1e-15);
    assert!((y[3]).abs() < 1e-15);
    assert!((y[1] + y[2]).abs() < 1e-15);
    let t = SkewSymmetric.project(&x, &v);
    assert!((t[0]).abs() < 1e-15);
    assert!((t[1] + t[2]).abs() < 1e-15);
    let w = SkewSymmetric.transport(&x, &y, &v);
    assert!((w[1] - 0.3).abs() < 1e-15);
    assert!((w[2] + 0.1).abs() < 1e-15);
}

#[test]
fn euclidean_complex_retract_stays_on_the_set() {
    let m = EuclideanComplex::new(2);
    let x = array![1.0, 0.5, -0.25, 2.0];
    let v = array![0.2, -0.1, 0.3, 0.4];
    let y = m.retract(&x, &v);
    assert!(is_euclidean_complex(&y), "left C^2 {y:?}");
    assert!((y[0] - 1.2).abs() < 1e-15);
    assert!((y[1] - 0.4).abs() < 1e-15);
    assert!((y[2] - 0.05).abs() < 1e-15);
    assert!((y[3] - 2.4).abs() < 1e-15);
    let t = m.project(&x, &v);
    assert!((t[0] - 0.2).abs() < 1e-15);
    let w = m.transport(&x, &y, &v);
    assert!((w[1] + 0.1).abs() < 1e-15);
    let fro = y.iter().map(|a| a * a).sum::<f64>().sqrt();
    assert!((fro - 1.0).abs() > 0.5, "must not be the sphere {y:?}");
}

#[test]
fn constant_retract_stays_on_the_set() {
    let m = Constant::new(3);
    let x = array![1.25, -0.5, 2.0];
    let v = array![0.3, -0.1, 4.0];
    let y = m.retract(&x, &v);
    assert!(is_constant(&y, 3), "left the singleton {y:?}");
    assert!((&y - &x).mapv(f64::abs).sum() < 1e-15, "moved off A {y:?}");
    let t = m.project(&x, &v);
    assert!(t.iter().all(|a| a.abs() < 1e-15), "nonzero tangent {t:?}");
    let w = m.transport(&x, &y, &v);
    assert!(w.iter().all(|a| a.abs() < 1e-15), "nonzero transport {w:?}");
    let fro = y.iter().map(|a| a * a).sum::<f64>().sqrt();
    assert!((fro - 1.0).abs() > 0.5, "must not be the sphere {y:?}");
}

#[test]
fn sphere_complex_retract_stays_on_the_set() {
    let m = SphereComplex::new(2);
    let x = array![1.0, 0.0, 0.0, 0.0];
    let v = array![0.0, 0.1, 0.2, -0.1];
    let y = m.retract(&x, &v);
    assert!(is_sphere_complex(&y), "left C^2 packing {y:?}");
    let fro = y.iter().map(|a| a * a).sum::<f64>().sqrt();
    assert!((fro - 1.0).abs() < 1e-14, "left the complex sphere {y:?}");
    let t = m.project(&x, &v);
    let ip: f64 = x.iter().zip(t.iter()).map(|(a, b)| a * b).sum();
    assert!(ip.abs() < 1e-14);
    let w = m.transport(&x, &y, &v);
    let p = m.project(&y, &v);
    assert!((&w - &p).mapv(f64::abs).sum() < 1e-14);
}

#[test]
fn positive_retract_stays_on_the_set() {
    let m = Positive::new(3);
    let x = array![1.0, 2.0, 0.5];
    let v = array![0.2, -0.4, 0.1];
    let y = m.retract(&x, &v);
    assert!(is_positive(&y), "left the positive orthant {y:?}");
    let t = m.project(&x, &v);
    assert!((&t - &v).mapv(f64::abs).sum() < 1e-15);
    let w = m.transport(&x, &y, &v);
    assert!((&w - &v).mapv(f64::abs).sum() < 1e-15);
    let fro = y.iter().map(|a| a * a).sum::<f64>().sqrt();
    assert!((fro - 1.0).abs() > 0.5, "must not be the sphere {y:?}");
}

#[test]
fn centered_retract_stays_on_the_set() {
    let m = CenteredMatrix::new(2, 3, false);
    let x = array![1.0, -0.5, -0.5, 2.0, 0.0, -2.0];
    let v = array![0.3, -0.1, -0.2, 0.0, 0.4, -0.4];
    let y = m.retract(&x, &v);
    assert!(is_centered(&y, 2, 3, false), "left the centered set {y:?}");
    assert!((y[0] + y[1] + y[2]).abs() < 1e-14);
    assert!((y[3] + y[4] + y[5]).abs() < 1e-14);
    let t = m.project(&x, &v);
    assert!((t[0] + t[1] + t[2]).abs() < 1e-14);
    let w = m.transport(&x, &y, &v);
    assert!((&w - &v).mapv(f64::abs).sum() < 1e-15);
    let fro = y.iter().map(|a| a * a).sum::<f64>().sqrt();
    assert!((fro - 1.0).abs() > 0.5, "must not be the sphere {y:?}");
}

#[test]
fn grassmann_p1_is_the_sphere_and_p2_rejects_3n() {
    let x = array![0.6, 0.8, 0.0];
    let v = array![0.1, -0.2, 0.5];
    let g = Grassmann { n: 3, p: 1 };
    assert!((g.project(&x, &v) - Sphere.project(&x, &v))
        .mapv(f64::abs)
        .sum()
        < 1e-14);
    assert!((g.retract(&x, &v) - Sphere.retract(&x, &v))
        .mapv(f64::abs)
        .sum()
        < 1e-14);
    assert_eq!(Grassmann { n: 4, p: 2 }.required_dim(114), Err(8));
}

#[test]
fn hyperbolic_retract_stays_on_the_sheet() {
    let x = pack(2.0_f64.sqrt(), array![1.0, 0.0].view());
    let v = Hyperbolic.project(&x, &array![0.2, -0.1, 0.4]);
    let y = Hyperbolic.retract(&x, &v);
    assert!((minkowski(y.view(), y.view()) + 1.0).abs() < 1e-12);
    assert!(y[0] > 0.0);
    assert_eq!(y.len(), 3);
}

#[test]
fn poincare_ball_is_not_the_sphere() {
    let x = array![0.3, 0.4, 0.0];
    let v = array![0.1, -0.2, 0.05];
    let yp = PoincareBall.retract(&x, &v);
    let ys = Sphere.retract(&x, &v);
    let n = yp.iter().map(|c| c * c).sum::<f64>().sqrt();
    assert!(n < 1.0, "Poincare retract must stay in the open ball, |y| = {n}");
    assert!((ys.iter().map(|c| c * c).sum::<f64>().sqrt() - 1.0).abs() < 1e-12);
    assert_ne!(yp, ys);
    assert_ne!(PoincareBall.project(&x, &v), Sphere.project(&x, &v));
}

#[test]
fn unitary_retract_stays_on_the_set() {
    let m = Unitary::new(2).unwrap();
    let mut x = Array1::zeros(8);
    x[0] = 1.0;
    x[6] = 1.0;
    let mut v = Array1::zeros(8);
    v[3] = 0.2;
    v[5] = 0.2;
    let t = m.project(&x, &v);
    let y = m.retract(&x, &t);
    assert!(is_unitary(&y), "left U(2) {y:?}");
    let uh_u00 = y[0] * y[0] + y[1] * y[1] + y[4] * y[4] + y[5] * y[5];
    let uh_u11 = y[2] * y[2] + y[3] * y[3] + y[6] * y[6] + y[7] * y[7];
    assert!((uh_u00 - 1.0).abs() < 1e-10, "U^* U[0,0] {uh_u00}");
    assert!((uh_u11 - 1.0).abs() < 1e-10, "U^* U[1,1] {uh_u11}");
    let fro = y.iter().map(|a| a * a).sum::<f64>().sqrt();
    assert!((fro - 1.0).abs() > 0.3, "must not be the sphere {y:?}");
}
