use ndarray::{Array1, array};
use rgmin::manifold::{
    CenteredMatrix, Euclidean, EuclideanComplex, MwRigid, RigidQuotient, Spd, Sphere,
    inner_cplx as inner, is_centered, is_spd, typical_dist_cplx as typical_dist,
};
use rgmin::vecops;
use rgmin::{Grassmann, IrcTrust, Manifold};

#[test]
fn irc_trust_is_not_the_unit_sphere() {
    let masses = [1.0, 4.0];
    let d1 = array![0.05, 0.0, 0.0, 0.0, 0.0, 0.0];
    let tr = IrcTrust::from_atom_masses(d1.clone(), &masses, 0.1);
    let s = array![1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let p = tr.project(&s);
    assert!(tr.on_bound(&p, 1e-12));
    let eucl = p.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        (eucl - 1.0).abs() > 1e-3,
        "must not be unit-sphere projection"
    );
    let sphere = Sphere.project(&Array1::from(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), &s);
    assert!((sphere[0] - p[0]).abs() > 1e-6 || (tr.cons(&p) - 0.1).abs() < 1e-12);
}

#[test]
fn spd_retract_stays_positive_definite_and_is_not_so3() {
    let x = array![2.0, 0.1, 0.0, 0.1, 3.0, 0.2, 0.0, 0.2, 4.0];
    let v = array![0.0, 0.05, 0.0, 0.05, 0.0, 0.1, 0.0, 0.1, 0.0];
    let y = Spd.retract(&x, &v);
    assert!(is_spd(&y), "left the SPD set {y:?}");
    let fro2: f64 = y.iter().map(|a| a * a).sum();
    assert!((fro2 - 3.0).abs() > 1.0);
}

#[test]
fn centered_matrix_retract_stays_on_the_set() {
    let m = CenteredMatrix::cols(2, 3);
    let x = array![1.0, -0.5, -0.5, 2.0, -1.0, -1.0];
    let v = array![0.3, 0.0, -0.1, -0.2, 0.4, 0.1];
    let y = m.retract(&x, &v);
    assert!(
        is_centered(&y, 2, 3, false),
        "left the centered-cols set {y:?}"
    );
    let t = m.project(&x, &v);
    assert!((t[0] + t[1] + t[2]).abs() < 1e-14);
    assert!((t[3] + t[4] + t[5]).abs() < 1e-14);
    let w = m.transport(&x, &y, &v);
    assert!((&w - &v).mapv(f64::abs).sum() < 1e-15);
    let fro = y.iter().map(|a| a * a).sum::<f64>().sqrt();
    assert!((fro - 1.0).abs() > 0.5, "must not be the sphere {y:?}");
}

#[test]
fn project_and_transport_are_identity() {
    let m = EuclideanComplex { n: 2 };
    let x = array![1.0, 0.0, 0.0, 1.0];
    let y = array![0.5, -0.5, 1.0, 0.0];
    let v = array![0.2, 0.3, -0.1, 0.4];
    let t = m.project(&x, &v);
    assert!((&t - &v).mapv(f64::abs).sum() < 1e-15);
    let w = m.transport(&x, &y, &v);
    assert!((&w - &v).mapv(f64::abs).sum() < 1e-15);
}

#[test]
fn inner_is_the_real_product() {
    let u = array![1.0, 2.0, -0.5, 0.5];
    let v = array![0.5, -1.0, 2.0, 4.0];
    // 1*0.5 + 2*(-1) + (-0.5)*2 + 0.5*4 = 0.5 - 2 - 1 + 2 = -0.5
    assert!((inner(&u, &v) + 0.5).abs() < 1e-15);
    assert!((typical_dist(4) - 2.0).abs() < 1e-15);
    assert!((vecops::nrm2(u.view()) - inner(&u, &u).sqrt()).abs() < 1e-15);
}

#[test]
fn not_the_sphere_and_not_complex_circle() {
    let m = EuclideanComplex { n: 2 };
    let x = array![2.0, 0.0, 0.0, 2.0];
    let y = m.retract(&x, &Array1::zeros(4));
    let n0 = (y[0] * y[0] + y[1] * y[1]).sqrt();
    let n1 = (y[2] * y[2] + y[3] * y[3]).sqrt();
    assert!((n0 - 1.0).abs() > 0.5, "must not force S^1 {y:?}");
    assert!((n1 - 1.0).abs() > 0.5, "must not force S^1 {y:?}");
    let fro = vecops::nrm2(y.view());
    assert!((fro - 1.0).abs() > 1.0, "must not be S^3 {y:?}");
    assert_ne!(
        rgmin::manifold::ManifoldKind::EuclideanComplex { n: 2 },
        rgmin::manifold::ManifoldKind::Sphere
    );
    assert_ne!(
        rgmin::manifold::ManifoldKind::EuclideanComplex { n: 2 },
        rgmin::manifold::ManifoldKind::ComplexCircle { n: 2 }
    );
    assert_ne!(
        rgmin::manifold::ManifoldKind::EuclideanComplex { n: 2 },
        rgmin::manifold::ManifoldKind::Euclidean
    );
}

#[test]
fn rigid_quotient_retract_stays_on_the_set() {
    let x = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let v = array![0.0, 0.1, 0.0, 0.0, -0.05, 0.05, 0.0, -0.05, -0.05];
    let t = RigidQuotient.project(&x, &v);
    let y = RigidQuotient.retract(&x, &t);
    assert_eq!(y.len(), 9);
    let inc = &y - &x;
    let re = RigidQuotient.project(&x, &inc);
    for (a, b) in inc.iter().zip(re.iter()) {
        assert!((a - b).abs() < 1e-12, "{inc:?} vs {re:?}");
    }
    let trans = array![0.2, 0.0, 0.0, 0.2, 0.0, 0.0, 0.2, 0.0, 0.0];
    let p = RigidQuotient.project(&y, &trans);
    let n = p.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(n < 1e-12, "translation of the image must vanish, |p| = {n}");
}

#[test]
fn mw_rigid_retract_stays_on_the_set() {
    let x = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let v = array![0.0, 0.1, 0.0, 0.0, -0.05, 0.05, 0.0, -0.05, -0.05];
    let geom = MwRigid::default();
    let t = geom.project(&x, &v);
    let y = geom.retract(&x, &t);
    assert_eq!(y.len(), 9);
    let inc = &y - &x;
    let re = geom.project(&x, &inc);
    for (a, b) in inc.iter().zip(re.iter()) {
        assert!((a - b).abs() < 1e-12, "{inc:?} vs {re:?}");
    }
}

#[test]
fn euclidean_retract_stays_on_the_set() {
    let x = array![1.0, -2.0, 0.5];
    let v = array![0.25, 1.0, -0.5];
    let y = Euclidean.retract(&x, &v);
    assert_eq!(y, &x + &v);
    assert_eq!(Euclidean.project(&x, &v), v);
}

#[test]
fn grassmann_retract_stays_orthonormal_and_is_not_the_sphere() {
    let gr = Grassmann::new(4, 2).unwrap();
    let x = array![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let v = array![0.0, 0.0, 0.1, -0.3, 0.0, 0.0, 0.2, 0.05];
    let y = gr.retract(&x, &v);
    let n0: f64 = y.iter().take(4).map(|a| a * a).sum();
    let n1: f64 = y.iter().skip(4).map(|a| a * a).sum();
    let d: f64 = y
        .iter()
        .take(4)
        .zip(y.iter().skip(4))
        .map(|(a, b)| a * b)
        .sum();
    assert!((n0 - 1.0).abs() < 1e-12);
    assert!((n1 - 1.0).abs() < 1e-12);
    assert!(d.abs() < 1e-12);
    let nrm = y.iter().map(|a| a * a).sum::<f64>().sqrt();
    assert!((nrm - 1.0).abs() > 1e-6);
}
