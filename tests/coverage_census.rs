//! The APIs the audit found no test had ever called.
//!
//! Coverage here is behavioural, not ceremonial: each test states the
//! contract the API promises and fails when the contract does.

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};
use rgmin::manifold::{Manifold, MwRigid, Sphere, Stiefel};
use rgmin::{Conjugacy, Control, DirectionalCurvature, Restart, ScgParams, minimize_scg_exact};

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
    assert_eq!(Stiefel.transport(&x, &y, &v), Sphere.transport(&x, &y, &v));
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
    let translation = Array1::from([1.0, 0.0, 0.0].repeat(4));
    let geom = MwRigid::default();
    let projected = geom.project(&x, &translation);
    let norm = projected.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(norm < 1e-10, "a pure translation must vanish, |p| = {norm}");

    // Infinitesimal rotation about z: v_i = omega x r_i.
    let mut rot = Array1::zeros(12);
    for atom in 0..4 {
        rot[3 * atom] = -x[3 * atom + 1];
        rot[3 * atom + 1] = x[3 * atom];
    }
    let projected = geom.project(&x, &rot);
    let norm = projected.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(norm < 1e-10, "a pure rotation must vanish, |p| = {norm}");

    // A breathing distortion is internal and must survive projection.
    let breathe = x.mapv(|c| 0.01 * c);
    let projected = geom.project(&x, &breathe);
    let norm = projected.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(norm > 1e-4, "an internal mode must survive, |p| = {norm}");
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
            Bounds::new(Array1::from_elem(4, -1e12), Array1::from_elem(4, 1e12), 0.0)
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
        Some(
            d.iter()
                .enumerate()
                .map(|(i, v)| (i + 1) as f64 * v * v)
                .sum(),
        )
    }
}

#[test]
fn scg_exact_reaches_the_bowl_floor_on_supplied_curvature() {
    let rep = minimize_scg_exact(
        &CurvedBowl,
        array![1.0, -2.0, 3.0, -4.0],
        &Control {
            maxiter: 200,
            ..Control::default()
        },
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
