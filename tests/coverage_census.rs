//! The APIs the audit found no test had ever called.
//!
//! Coverage here is behavioural, not ceremonial: each test states the
//! contract the API promises and fails when the contract does.

use ndarray::{array, Array1, ArrayView1};
use rgmin::manifold::{
    is_spd, minkowski, pack_h, ComplexCircle, Grassmann, Hyperbolic, Manifold, MwRigid, Spd,
    Sphere, Stiefel,
};
use rgmin::IrcTrust;
use rgmin::{
    minimize_scg_exact, Conjugacy, Control, DirectionalCurvature, Restart, ScgParams,
};
use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};

/// Stiefel at p = 1 is the sphere, in all three operations, which is
/// the whole content of the type: divergence in any one of them means
/// the delegation broke.
#[test]
fn stiefel_p1_is_the_sphere_in_all_three_operations() {
    let x = array![0.6, 0.8, 0.0];
    let v = array![0.1, -0.2, 0.5];
    let y = array![0.0, 1.0, 0.0];
    assert_eq!(Stiefel.project(&x, &v), Sphere.project(&x, &v));
    assert_eq!(Stiefel.retract(&x, &v), Sphere.retract(&x, &v));
    assert_eq!(
        Stiefel.transport(&x, &y, &v),
        Sphere.transport(&x, &y, &v)
    );
}

/// Grassmann Gr(4, 2) is a 4x2 frame, not a unit sphere in R^8.
#[test]
fn grassmann_retract_stays_orthonormal_and_is_not_the_sphere() {
    let gr = Grassmann::new(4, 2);
    let x = array![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let v = array![0.0, 0.0, 0.1, -0.3, 0.0, 0.0, 0.2, 0.05];
    let y = gr.retract(&x, &v);
    let n0: f64 = y.iter().take(4).map(|a| a * a).sum();
    let n1: f64 = y.iter().skip(4).map(|a| a * a).sum();
    let d: f64 = y.iter().take(4).zip(y.iter().skip(4)).map(|(a, b)| a * b).sum();
    assert!((n0 - 1.0).abs() < 1e-12);
    assert!((n1 - 1.0).abs() < 1e-12);
    assert!(d.abs() < 1e-12);
    let nrm = y.iter().map(|a| a * a).sum::<f64>().sqrt();
    assert!((nrm - 1.0).abs() > 1e-6);
}

/// A hyperbolic retraction stays on the Lorentz sheet. This is not the
/// sphere: the quadratic form is Minkowski, and the first coordinate
/// stays positive.
#[test]
fn hyperbolic_retract_stays_on_the_sheet() {
    let x = pack_h(2.0_f64.sqrt(), array![1.0, 0.0].view());
    let v = Hyperbolic.project(&x, &array![0.2, -0.1, 0.4]);
    let y = Hyperbolic.retract(&x, &v);
    assert!((minkowski(y.view(), y.view()) + 1.0).abs() < 1e-12);
    assert!(y[0] > 0.0);
    assert_eq!(y.len(), 3);
}

/// The Eckart quotient's projection removes every rigid-body component:
/// a pure translation and a pure rotation both project to nothing, and
/// an internal distortion survives.
#[test]
fn mw_rigid_projects_out_translations_and_rotations() {
    // Four atoms off the axes so all six rigid modes are present.
    let x = array![
        0.0, 0.0, 0.0, //
        1.1, 0.0, 0.0, //
        0.0, 1.3, 0.0, //
        0.0, 0.0, 1.7
    ];
    let translation = Array1::from(vec![1.0, 0.0, 0.0].repeat(4));
    let projected = MwRigid.project(&x, &translation);
    let norm = projected.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(norm < 1e-10, "a pure translation must vanish, |p| = {norm}");

    // Infinitesimal rotation about z: v_i = omega x r_i.
    let mut rot = Array1::zeros(12);
    for atom in 0..4 {
        rot[3 * atom] = -x[3 * atom + 1];
        rot[3 * atom + 1] = x[3 * atom];
    }
    let projected = MwRigid.project(&x, &rot);
    let norm = projected.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(norm < 1e-10, "a pure rotation must vanish, |p| = {norm}");

    // A breathing distortion is internal and must survive projection.
    let breathe = x.mapv(|c| 0.01 * c);
    let projected = MwRigid.project(&x, &breathe);
    let norm = projected.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(norm > 1e-4, "an internal mode must survive, |p| = {norm}");
}

/// Gonzalez--Schlegel / Sella IRC is a sphere of radius dx about the
/// last point in the mass-weighted metric. Unit Sphere about the
/// origin is a different geometry.
#[test]
fn irc_trust_is_not_the_unit_sphere() {
    let masses = [1.0, 4.0];
    let d1 = array![0.05, 0.0, 0.0, 0.0, 0.0, 0.0];
    let tr = IrcTrust::from_atom_masses(d1.clone(), &masses, 0.1);
    let s = array![1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let p = tr.project(&s);
    assert!(tr.on_bound(&p, 1e-12));
    let eucl = p.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!((eucl - 1.0).abs() > 1e-3, "must not be unit-sphere projection");
    let sphere = Sphere.project(&Array1::from(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), &s);
    assert!((sphere[0] - p[0]).abs() > 1e-6 || (tr.cons(&p) - 0.1).abs() < 1e-12);
}

/// Affine-invariant SPD retraction lands on a symmetric positive-definite
/// matrix: the whole content of the factory is that the point stays on
/// the set.
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

/// manopt complexcirclefactory: each (re, im) pair stays on S^1.
/// The product is not the sphere in the ambient even dimension.
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

/// A quadratic bowl carrying its exact directional curvature. SCG with
/// the exact path must reach the minimum without the finite-difference
/// probe's extra gradient.
struct CurvedBowl;

impl Objective<f64> for CurvedBowl {
    fn dim(&self) -> usize {
        4
    }
    fn bounds(&self) -> &Bounds<f64> {
        static BOUNDS: std::sync::OnceLock<Bounds<f64>> = std::sync::OnceLock::new();
        BOUNDS.get_or_init(|| {
            Bounds::new(
                Array1::from_elem(4, -1e12),
                Array1::from_elem(4, 1e12),
                0.0,
            )
        })
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        x.iter()
            .enumerate()
            .map(|(i, v)| 0.5 * (i + 1) as f64 * v * v)
            .sum()
    }
}

impl Gradient<f64> for CurvedBowl {
    fn dim(&self) -> usize {
        4
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        Array1::from_iter(x.iter().enumerate().map(|(i, v)| (i + 1) as f64 * v))
    }
}

impl DifferentiableObjective<f64> for CurvedBowl {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.grad(x))
    }
}

impl DirectionalCurvature for CurvedBowl {
    fn directional_curvature(&self, _x: ArrayView1<f64>, d: ArrayView1<f64>) -> Option<f64> {
        Some(d.iter().enumerate().map(|(i, v)| (i + 1) as f64 * v * v).sum())
    }
}

#[test]
fn scg_exact_reaches_the_bowl_floor_on_supplied_curvature() {
    let rep = minimize_scg_exact(
        &CurvedBowl,
        array![1.0, -2.0, 3.0, -4.0],
        &Control { maxiter: 200, ..Control::default() },
        &ScgParams::default(),
        Conjugacy::PolakRibiere,
        Restart::Njws { threshold: 0.1 },
    )
    .expect("scg exact runs");
    assert!(
        rep.value < 1e-8,
        "the bowl floor is zero, exact-curvature SCG got {}",
        rep.value
    );
}
