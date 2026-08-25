//! Persistent Session: one step is one outer iteration.

use eindir_core::objectives::Rosenbrock;
use ndarray::{Array1, array};
use rgmin::{Control, Method, Solver};

fn control() -> Control {
    Control {
        maxiter: 80,
        gtol: 1e-8,
        istep: 0.1,
        maxmove: None,
    }
}

#[test]
fn lbfgs_session_reaches_rosenbrock() {
    let obj = Rosenbrock::<2>::new();
    let mut x = array![-1.2, 1.0];
    let mut solver = Solver::new(Method::lbfgs(), control(), 2).with_gtol(1e-8);
    solver.set_accept(rgmin::Accept::Energy);
    let mut last = None;
    for _ in 0..80 {
        let rep = solver.step(&obj, &mut x).unwrap();
        last = Some(rep);
        if last.as_ref().unwrap().grad_norm < 1e-8 {
            break;
        }
    }
    let rep = last.unwrap();
    assert!(rep.value < 1e-6, "value {}", rep.value);
}

#[test]
fn lbfgs_accept_none_moves_when_energy_is_flat() {
    use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
    use ndarray::ArrayView1;
    use rgmin::Accept;
    use std::sync::OnceLock;

    struct FlatEnergyBowl;
    impl Objective<f64> for FlatEnergyBowl {
        fn dim(&self) -> usize {
            2
        }
        fn bounds(&self) -> &Bounds<f64> {
            static B: OnceLock<Bounds<f64>> = OnceLock::new();
            B.get_or_init(|| Bounds::new(array![-1e6, -1e6], array![1e6, 1e6], 0.0))
        }
        fn eval(&self, _: ArrayView1<f64>) -> f64 {
            0.0
        }
    }
    impl Gradient<f64> for FlatEnergyBowl {
        fn dim(&self) -> usize {
            2
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            &x * 2.0
        }
    }
    impl DifferentiableObjective<f64> for FlatEnergyBowl {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            (0.0, self.grad(x))
        }
    }

    let obj = FlatEnergyBowl;
    let mut x = array![3.0, -4.0];
    let start = x.clone();
    let mut ctrl = control();
    ctrl.maxmove = Some(1.0);
    let mut solver = Solver::new(Method::lbfgs(), ctrl, 2);
    solver.set_accept(Accept::None);
    solver.step(&obj, &mut x).unwrap();
    assert!(
        (x[0] - start[0]).abs() + (x[1] - start[1]).abs() > 1e-9,
        "LBFGS Accept::None stayed put {x:?}"
    );
    let n0 = (start[0] * start[0] + start[1] * start[1]).sqrt();
    let n1 = (x[0] * x[0] + x[1] * x[1]).sqrt();
    assert!(n1 < n0, "clipped two-loop step did not descend {x:?}");
}

#[test]
fn first_order_accept_none_moves_a_nonconservative_force() {
    use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
    use ndarray::ArrayView1;
    use rgmin::Accept;
    use std::sync::OnceLock;

    struct ConstantForce;
    impl Objective<f64> for ConstantForce {
        fn dim(&self) -> usize {
            2
        }
        fn bounds(&self) -> &Bounds<f64> {
            static B: OnceLock<Bounds<f64>> = OnceLock::new();
            B.get_or_init(|| Bounds::new(array![-1e6, -1e6], array![1e6, 1e6], 0.0))
        }
        fn eval(&self, _: ArrayView1<f64>) -> f64 {
            1.0
        }
    }
    impl Gradient<f64> for ConstantForce {
        fn dim(&self) -> usize {
            2
        }
        fn grad(&self, _: ArrayView1<f64>) -> Array1<f64> {
            array![0.4, 0.0]
        }
    }
    impl DifferentiableObjective<f64> for ConstantForce {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            (1.0, self.grad(x))
        }
    }

    for method in [Method::Steepest, Method::Bfgs] {
        let obj = ConstantForce;
        let mut x = array![0.0, 0.0];
        let mut ctrl = control();
        ctrl.maxmove = Some(0.2);
        let mut solver = Solver::new(method.clone(), ctrl, 2);
        solver.set_accept(Accept::None);
        solver.step(&obj, &mut x).unwrap();
        assert!(
            x[0].abs() > 1e-9,
            "{method:?} Accept::None froze on a non-conservative force {x:?}"
        );
    }
}

#[test]
fn retained_pairs_beat_a_cold_start() {
    let obj = Rosenbrock::<2>::new();
    let mut warm = Solver::new(Method::lbfgs(), control(), 2).with_gtol(1e-8);
    warm.set_accept(rgmin::Accept::Energy);
    let mut x = array![-1.2, 1.0];
    for _ in 0..80 {
        let rep = warm.step(&obj, &mut x).unwrap();
        if rep.grad_norm < 1e-8 {
            break;
        }
    }
    x[0] += 0.05;
    x[1] -= 0.05;
    let mut warm_steps = 0usize;
    for _ in 0..80 {
        warm_steps += 1;
        let rep = warm.step(&obj, &mut x).unwrap();
        if rep.grad_norm < 1e-8 {
            break;
        }
    }

    let mut cold = Solver::new(Method::lbfgs(), control(), 2).with_gtol(1e-8);
    let mut y = array![-1.2 + 0.05, 1.0 - 0.05];
    // Same start as the warm restart: first quench, then same perturbation.
    let mut z = array![-1.2, 1.0];
    for _ in 0..80 {
        let rep = cold.step(&obj, &mut z).unwrap();
        if rep.grad_norm < 1e-8 {
            break;
        }
    }
    cold.forget();
    y = z;
    y[0] += 0.05;
    y[1] -= 0.05;
    let mut cold_steps = 0usize;
    for _ in 0..80 {
        cold_steps += 1;
        let rep = cold.step(&obj, &mut y).unwrap();
        if rep.grad_norm < 1e-8 {
            break;
        }
    }
    assert!(
        warm_steps <= cold_steps,
        "warm {warm_steps} should not exceed cold {cold_steps}"
    );
}

#[test]
fn lbfgs_newton_on_a_supplied_hessian_kills_a_quadratic() {
    use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
    use ndarray::{Array2, ArrayView1};
    use rgmin::{HessianObjective, QnStep};

    struct Quad;
    impl Objective<f64> for Quad {
        fn dim(&self) -> usize {
            2
        }
        fn bounds(&self) -> &Bounds<f64> {
            use std::sync::OnceLock;
            static B: OnceLock<Bounds<f64>> = OnceLock::new();
            B.get_or_init(|| Bounds::new(array![-1e6, -1e6], array![1e6, 1e6], 0.0))
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            5.0 * x[0] * x[0] + 0.5 * x[1] * x[1]
        }
    }
    impl Gradient<f64> for Quad {
        fn dim(&self) -> usize {
            2
        }
        fn grad(&self, x: ArrayView1<f64>) -> ndarray::Array1<f64> {
            array![10.0 * x[0], x[1]]
        }
    }
    impl DifferentiableObjective<f64> for Quad {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, ndarray::Array1<f64>) {
            (self.eval(x), self.grad(x))
        }
    }
    impl HessianObjective for Quad {
        fn hessian(&self, _x: ArrayView1<f64>) -> Array2<f64> {
            Array2::from_shape_vec((2, 2), vec![10.0, 0.0, 0.0, 1.0]).unwrap()
        }
    }

    let obj = Quad;
    let mut x = array![2.0, -3.0];
    let mut solver = Solver::new(Method::lbfgs(), control(), 2).with_gtol(1e-10);
    solver.set_qn_step(QnStep::Newton);
    let rep = solver.step_hess(&obj, &mut x).unwrap();
    assert!(rep.value < 1e-12, "newton-on-P value {}", rep.value);
    assert!(x.iter().all(|v| v.abs() < 1e-6));
}

#[test]
fn sphere_rayleigh_stays_on_the_sphere() {
    use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
    use ndarray::ArrayView1;
    use rgmin::ManifoldKind;

    struct Ray;
    impl Objective<f64> for Ray {
        fn dim(&self) -> usize {
            3
        }
        fn bounds(&self) -> &Bounds<f64> {
            use std::sync::OnceLock;
            static B: OnceLock<Bounds<f64>> = OnceLock::new();
            B.get_or_init(|| Bounds::new(array![-2.0, -2.0, -2.0], array![2.0, 2.0, 2.0], 0.0))
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            0.5 * (x[0] * x[0] + 2.0 * x[1] * x[1] + 3.0 * x[2] * x[2])
        }
    }
    impl Gradient<f64> for Ray {
        fn dim(&self) -> usize {
            3
        }
        fn grad(&self, x: ArrayView1<f64>) -> ndarray::Array1<f64> {
            array![x[0], 2.0 * x[1], 3.0 * x[2]]
        }
    }
    impl DifferentiableObjective<f64> for Ray {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, ndarray::Array1<f64>) {
            (self.eval(x), self.grad(x))
        }
    }

    let obj = Ray;
    let n = (1.0_f64 + 1.0 + 1.0).sqrt();
    let mut x = array![1.0 / n, 1.0 / n, 1.0 / n];
    let mut solver = Solver::new(
        Method::Steepest,
        Control {
            maxiter: 40,
            gtol: 1e-8,
            istep: 0.2,
            maxmove: None,
        },
        3,
    );
    solver.set_manifold(ManifoldKind::Sphere);
    solver.set_accept(rgmin::Accept::None);
    for _ in 0..40 {
        let _ = solver.step(&obj, &mut x).unwrap();
        let nrm = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
        assert!((nrm - 1.0).abs() < 1e-10, "left the sphere {x:?}");
    }
}

#[test]
fn stiefel_p1_matches_sphere_retract() {
    use rgmin::{Manifold, ManifoldKind};
    let x = array![0.0, 1.0, 0.0];
    let v = array![0.1, 0.0, -0.2];
    let ys = ManifoldKind::Sphere.retract(&x, &v);
    let yv = ManifoldKind::Stiefel.retract(&x, &v);
    assert!((&ys - &yv).mapv(f64::abs).sum() < 1e-15);
    assert_eq!(ManifoldKind::Stiefel.stiefel_p(), 1);
    assert_eq!(ManifoldKind::stiefel(3, 1), ManifoldKind::Stiefel);
}

#[test]
fn stiefel_p2_retract_stays_orthonormal() {
    use rgmin::{Manifold, ManifoldKind};
    let kind = ManifoldKind::stiefel(4, 2);
    assert_eq!(kind.stiefel_p(), 2);
    assert!(kind.required_dim(8).is_ok());
    assert!(kind.required_dim(7).is_err());
    assert!(kind.required_dim(114).is_err());
    let x = array![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let raw = array![0.1, 0.0, 0.2, 0.0, 0.0, 0.1, 0.0, -0.2];
    let v = kind.project(&x, &raw);
    assert!((2.0 * v[0]).abs() < 1e-12);
    assert!((v[4] + v[1]).abs() < 1e-12);
    assert!((2.0 * v[5]).abs() < 1e-12);
    let y = kind.retract(&x, &v);
    let mut yty = [0.0; 4];
    for a in 0..2 {
        for b in 0..2 {
            let mut acc = 0.0;
            for i in 0..4 {
                acc += y[i + 4 * a] * y[i + 4 * b];
            }
            yty[a + 2 * b] = acc;
        }
    }
    assert!((yty[0] - 1.0).abs() < 1e-12);
    assert!(yty[1].abs() < 1e-12);
    assert!(yty[2].abs() < 1e-12);
    assert!((yty[3] - 1.0).abs() < 1e-12);
}

#[test]
fn oblique_rejects_a_3n_cluster() {
    let obj = Rosenbrock::<114>::new();
    let mut x = Array1::from_elem(114, 0.1);
    let mut solver = Solver::new(Method::Steepest, control(), 114);
    solver.set_oblique(3, 2);
    let err = solver.step(&obj, &mut x).unwrap_err();
    match err {
        rgmin::Error::ManifoldDim { kind, got } => {
            assert_eq!(kind, "oblique");
            assert_eq!(got, 114);
        }
        other => panic!("expected ManifoldDim, got {other:?}"),
    }
}

#[test]
fn oblique_kind_is_not_sphere() {
    assert_ne!(
        rgmin::ManifoldKind::oblique(3, 1),
        rgmin::ManifoldKind::Sphere
    );
}

#[test]
fn oblique_session_stays_on_product_of_spheres() {
    use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
    use ndarray::ArrayView1;
    use rgmin::ManifoldKind;

    struct PairRay;
    impl Objective<f64> for PairRay {
        fn dim(&self) -> usize {
            6
        }
        fn bounds(&self) -> &Bounds<f64> {
            use std::sync::OnceLock;
            static B: OnceLock<Bounds<f64>> = OnceLock::new();
            B.get_or_init(|| {
                Bounds::new(Array1::from_elem(6, -2.0), Array1::from_elem(6, 2.0), 0.0)
            })
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            0.5 * (x[0] * x[0]
                + 2.0 * x[1] * x[1]
                + 3.0 * x[2] * x[2]
                + x[3] * x[3]
                + 2.0 * x[4] * x[4]
                + 3.0 * x[5] * x[5])
        }
    }
    impl Gradient<f64> for PairRay {
        fn dim(&self) -> usize {
            6
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            array![x[0], 2.0 * x[1], 3.0 * x[2], x[3], 2.0 * x[4], 3.0 * x[5]]
        }
    }
    impl DifferentiableObjective<f64> for PairRay {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            (self.eval(x), self.grad(x))
        }
    }

    let obj = PairRay;
    let s = (0.5_f64).sqrt();
    let mut x = array![s, s, 0.0, 0.0, s, s];
    let mut solver = Solver::new(
        Method::Steepest,
        Control {
            maxiter: 30,
            gtol: 1e-8,
            istep: 0.2,
            maxmove: None,
        },
        6,
    );
    solver.set_manifold(ManifoldKind::oblique(3, 2));
    solver.set_accept(rgmin::Accept::None);
    for _ in 0..30 {
        let _ = solver.step(&obj, &mut x).unwrap();
        let n0 = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
        let n1 = (x[3] * x[3] + x[4] * x[4] + x[5] * x[5]).sqrt();
        assert!((n0 - 1.0).abs() < 1e-10, "left sphere 0 {x:?}");
        assert!((n1 - 1.0).abs() < 1e-10, "left sphere 1 {x:?}");
    }
}

#[test]
fn multinomial_session_stays_on_the_simplex() {
    use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
    use ndarray::ArrayView1;
    use rgmin::ManifoldKind;

    struct Entropy;
    impl Objective<f64> for Entropy {
        fn dim(&self) -> usize {
            3
        }
        fn bounds(&self) -> &Bounds<f64> {
            use std::sync::OnceLock;
            static B: OnceLock<Bounds<f64>> = OnceLock::new();
            B.get_or_init(|| Bounds::new(array![0.0, 0.0, 0.0], array![1.0, 1.0, 1.0], 0.0))
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            x.iter().map(|xi| xi * xi).sum()
        }
    }
    impl Gradient<f64> for Entropy {
        fn dim(&self) -> usize {
            3
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            Array1::from_iter(x.iter().map(|xi| 2.0 * xi))
        }
    }
    impl DifferentiableObjective<f64> for Entropy {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            (self.eval(x), self.grad(x))
        }
    }

    let obj = Entropy;
    let mut x = array![0.2, 0.3, 0.5];
    let mut solver = Solver::new(
        Method::Steepest,
        Control {
            maxiter: 20,
            gtol: 1e-10,
            istep: 0.1,
            maxmove: None,
        },
        3,
    );
    solver.set_manifold(ManifoldKind::Multinomial);
    solver.set_accept(rgmin::Accept::None);
    for _ in 0..20 {
        let _ = solver.step(&obj, &mut x).unwrap();
        assert!(x.iter().all(|&xi| xi > 0.0), "left the interior {x:?}");
        let s = x.iter().copied().sum::<f64>();
        assert!((s - 1.0).abs() < 1e-12, "left the simplex sum={s} x={x:?}");
    }
}

#[test]
fn multinomial_rejects_a_point() {
    let obj = Rosenbrock::<1>::new();
    let mut x = array![1.0];
    let mut solver = Solver::new(Method::Steepest, control(), 1);
    solver.set_manifold(rgmin::ManifoldKind::Multinomial);
    let err = solver.step(&obj, &mut x).unwrap_err();
    match err {
        rgmin::Error::ManifoldDim { kind, got } => {
            assert_eq!(kind, "multinomial");
            assert_eq!(got, 1);
        }
        other => panic!("expected ManifoldDim, got {other:?}"),
    }
}

#[test]
fn multinomial_ds_session_stays_doubly_stochastic() {
    use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
    use ndarray::ArrayView1;

    struct Frobenius;
    impl Objective<f64> for Frobenius {
        fn dim(&self) -> usize {
            4
        }
        fn bounds(&self) -> &Bounds<f64> {
            use std::sync::OnceLock;
            static B: OnceLock<Bounds<f64>> = OnceLock::new();
            B.get_or_init(|| {
                Bounds::new(array![0.0, 0.0, 0.0, 0.0], array![1.0, 1.0, 1.0, 1.0], 0.0)
            })
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            x.iter().map(|xi| xi * xi).sum()
        }
    }
    impl Gradient<f64> for Frobenius {
        fn dim(&self) -> usize {
            4
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            Array1::from_iter(x.iter().map(|xi| 2.0 * xi))
        }
    }
    impl DifferentiableObjective<f64> for Frobenius {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            (self.eval(x), self.grad(x))
        }
    }

    let obj = Frobenius;
    let mut x = array![0.5, 0.5, 0.5, 0.5];
    let mut solver = Solver::new(
        Method::Steepest,
        Control {
            maxiter: 20,
            gtol: 1e-10,
            istep: 0.1,
            maxmove: None,
        },
        4,
    );
    solver.set_multinomial_ds(2);
    solver.set_accept(rgmin::Accept::None);
    for _ in 0..20 {
        let _ = solver.step(&obj, &mut x).unwrap();
        assert!(x.iter().all(|&xi| xi > 0.0), "left the interior {x:?}");
        let r0 = x[0] + x[1];
        let r1 = x[2] + x[3];
        let c0 = x[0] + x[2];
        let c1 = x[1] + x[3];
        assert!((r0 - 1.0).abs() < 1e-10, "row0 {r0} x={x:?}");
        assert!((r1 - 1.0).abs() < 1e-10, "row1 {r1} x={x:?}");
        assert!((c0 - 1.0).abs() < 1e-10, "col0 {c0} x={x:?}");
        assert!((c1 - 1.0).abs() < 1e-10, "col1 {c1} x={x:?}");
    }
}

#[test]
fn multinomial_ds_rejects_a_wrong_length() {
    let obj = Rosenbrock::<1>::new();
    let mut x = array![1.0];
    let mut solver = Solver::new(Method::Steepest, control(), 1);
    solver.set_manifold(rgmin::ManifoldKind::multinomial_ds(2));
    let err = solver.step(&obj, &mut x).unwrap_err();
    match err {
        rgmin::Error::ManifoldDim { kind, got } => {
            assert_eq!(kind, "multinomialdoublystochastic");
            assert_eq!(got, 1);
        }
        other => panic!("expected ManifoldDim, got {other:?}"),
    }
}

#[test]
fn so3_session_stays_orthogonal() {
    use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
    use ndarray::ArrayView1;
    use rgmin::ManifoldKind;

    struct FrobeniusI;
    impl Objective<f64> for FrobeniusI {
        fn dim(&self) -> usize {
            9
        }
        fn bounds(&self) -> &Bounds<f64> {
            use std::sync::OnceLock;
            static B: OnceLock<Bounds<f64>> = OnceLock::new();
            B.get_or_init(|| {
                Bounds::new(Array1::from_elem(9, -2.0), Array1::from_elem(9, 2.0), 0.0)
            })
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            let i = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
            0.5 * x.iter().zip(i).map(|(a, b)| (a - b) * (a - b)).sum::<f64>()
        }
    }
    impl Gradient<f64> for FrobeniusI {
        fn dim(&self) -> usize {
            9
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            let i = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
            Array1::from_iter(x.iter().zip(i).map(|(a, b)| a - b))
        }
    }
    impl DifferentiableObjective<f64> for FrobeniusI {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            (self.eval(x), self.grad(x))
        }
    }

    let obj = FrobeniusI;
    let mut x = array![1.0, 0.1, 0.0, -0.1, 1.0, 0.0, 0.0, 0.0, 1.0];
    let mut solver = Solver::new(
        Method::Steepest,
        Control {
            maxiter: 20,
            gtol: 1e-8,
            istep: 0.1,
            maxmove: None,
        },
        9,
    );
    solver.set_manifold(ManifoldKind::So3);
    solver.set_accept(rgmin::Accept::None);
    for _ in 0..20 {
        let _ = solver.step(&obj, &mut x).unwrap();
        let mut rtr = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                rtr[i][j] = (0..3).map(|k| x[3 * k + i] * x[3 * k + j]).sum();
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (rtr[i][j] - want).abs() < 1e-10,
                    "left SO(3) rtr={rtr:?} x={x:?}"
                );
            }
        }
    }
}

#[test]
fn se3_session_keeps_rotation_and_moves_translation() {
    use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
    use ndarray::ArrayView1;
    use rgmin::ManifoldKind;

    struct Se3Target;
    impl Objective<f64> for Se3Target {
        fn dim(&self) -> usize {
            12
        }
        fn bounds(&self) -> &Bounds<f64> {
            use std::sync::OnceLock;
            static B: OnceLock<Bounds<f64>> = OnceLock::new();
            B.get_or_init(|| {
                Bounds::new(Array1::from_elem(12, -4.0), Array1::from_elem(12, 4.0), 0.0)
            })
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            0.5 * (x[9] * x[9] + x[10] * x[10] + x[11] * x[11])
        }
    }
    impl Gradient<f64> for Se3Target {
        fn dim(&self) -> usize {
            12
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            let mut g = Array1::zeros(12);
            g[9] = x[9];
            g[10] = x[10];
            g[11] = x[11];
            g
        }
    }
    impl DifferentiableObjective<f64> for Se3Target {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            (self.eval(x), self.grad(x))
        }
    }

    let obj = Se3Target;
    let mut x = array![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.5, -0.7, 0.4];
    let mut solver = Solver::new(
        Method::Steepest,
        Control {
            maxiter: 30,
            gtol: 1e-10,
            istep: 0.4,
            maxmove: None,
        },
        12,
    );
    solver.set_manifold(ManifoldKind::Se3);
    solver.set_accept(rgmin::Accept::None);
    for _ in 0..30 {
        let _ = solver.step(&obj, &mut x).unwrap();
        let mut rtr = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                rtr[i][j] = (0..3).map(|k| x[3 * k + i] * x[3 * k + j]).sum();
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((rtr[i][j] - want).abs() < 1e-10, "left SE(3) rot {rtr:?}");
            }
        }
    }
    let t2 = x[9] * x[9] + x[10] * x[10] + x[11] * x[11];
    assert!(t2 < 1e-6, "translation not killed {x:?}");
}

#[test]
fn default_manifold_is_euclidean() {
    let obj = Rosenbrock::<2>::new();
    let mut x = array![-1.2, 1.0];
    let mut solver = Solver::new(Method::lbfgs(), control(), 2).with_gtol(1e-8);
    solver.set_manifold(rgmin::ManifoldKind::Euclidean);
    let rep = solver.step(&obj, &mut x).unwrap();
    assert!(rep.grad_norm.is_finite());
}

#[test]
fn so3_rejects_a_3n_cluster() {
    let obj = Rosenbrock::<6>::new();
    let mut x = Array1::from_elem(6, 0.1);
    let mut solver = Solver::new(Method::Steepest, control(), 6);
    solver.set_manifold(rgmin::ManifoldKind::So3);
    let err = solver.step(&obj, &mut x).unwrap_err();
    match err {
        rgmin::Error::ManifoldDim { kind, got } => {
            assert_eq!(kind, "so3");
            assert_eq!(got, 6);
        }
        other => panic!("expected ManifoldDim, got {other:?}"),
    }
}

#[test]
fn se3_rejects_a_3n_cluster() {
    let obj = Rosenbrock::<114>::new();
    let mut x = Array1::from_elem(114, 0.1);
    let mut solver = Solver::new(Method::Steepest, control(), 114);
    solver.set_manifold(rgmin::ManifoldKind::Se3);
    let err = solver.step(&obj, &mut x).unwrap_err();
    match err {
        rgmin::Error::ManifoldDim { kind, got } => {
            assert_eq!(kind, "se3");
            assert_eq!(got, 114);
        }
        other => panic!("expected ManifoldDim, got {other:?}"),
    }
}

#[test]
fn complex_circle_session_stays_on_the_set() {
    use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
    use ndarray::ArrayView1;
    use rgmin::ManifoldKind;

    /// Minus the sum of real parts. Minimizer is every z_k = 1.
    struct MinusReal;
    impl Objective<f64> for MinusReal {
        fn dim(&self) -> usize {
            4
        }
        fn bounds(&self) -> &Bounds<f64> {
            use std::sync::OnceLock;
            static B: OnceLock<Bounds<f64>> = OnceLock::new();
            B.get_or_init(|| {
                Bounds::new(Array1::from_elem(4, -2.0), Array1::from_elem(4, 2.0), 0.0)
            })
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            -(x[0] + x[2])
        }
    }
    impl Gradient<f64> for MinusReal {
        fn dim(&self) -> usize {
            4
        }
        fn grad(&self, _x: ArrayView1<f64>) -> Array1<f64> {
            array![-1.0, 0.0, -1.0, 0.0]
        }
    }
    impl DifferentiableObjective<f64> for MinusReal {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            (self.eval(x), self.grad(x))
        }
    }

    let obj = MinusReal;
    let mut x = array![0.0, 1.0, -1.0, 0.0];
    let mut solver = Solver::new(
        Method::Steepest,
        Control {
            maxiter: 20,
            gtol: 1e-8,
            istep: 0.2,
            maxmove: None,
        },
        4,
    );
    solver.set_manifold(ManifoldKind::complex_circle(2));
    solver.set_accept(rgmin::Accept::None);
    for _ in 0..20 {
        let _ = solver.step(&obj, &mut x).unwrap();
        let n0 = (x[0] * x[0] + x[1] * x[1]).sqrt();
        let n1 = (x[2] * x[2] + x[3] * x[3]).sqrt();
        assert!((n0 - 1.0).abs() < 1e-10, "left circle 0 {x:?}");
        assert!((n1 - 1.0).abs() < 1e-10, "left circle 1 {x:?}");
    }
}

#[test]
fn complex_circle_rejects_a_3n_cluster() {
    let obj = Rosenbrock::<114>::new();
    let mut x = Array1::from_elem(114, 0.1);
    let mut solver = Solver::new(Method::Steepest, control(), 114);
    solver.set_manifold(rgmin::ManifoldKind::complex_circle(2));
    let err = solver.step(&obj, &mut x).unwrap_err();
    match err {
        rgmin::Error::ManifoldDim { kind, got } => {
            assert_eq!(kind, "complex_circle");
            assert_eq!(got, 114);
        }
        other => panic!("expected ManifoldDim, got {other:?}"),
    }
}

#[test]
fn symmetric_session_stays_on_the_set() {
    use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
    use ndarray::ArrayView1;
    use rgmin::ManifoldKind;
    use rgmin::manifold::is_symmetric;

    struct FrobeniusI;
    impl Objective<f64> for FrobeniusI {
        fn dim(&self) -> usize {
            4
        }
        fn bounds(&self) -> &Bounds<f64> {
            use std::sync::OnceLock;
            static B: OnceLock<Bounds<f64>> = OnceLock::new();
            B.get_or_init(|| {
                Bounds::new(Array1::from_elem(4, -4.0), Array1::from_elem(4, 4.0), 0.0)
            })
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            let i = [1.0, 0.0, 0.0, 1.0];
            0.5 * x.iter().zip(i).map(|(a, b)| (a - b) * (a - b)).sum::<f64>()
        }
    }
    impl Gradient<f64> for FrobeniusI {
        fn dim(&self) -> usize {
            4
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            let i = [1.0, 0.0, 0.0, 1.0];
            Array1::from_iter(x.iter().zip(i).map(|(a, b)| a - b))
        }
    }
    impl DifferentiableObjective<f64> for FrobeniusI {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            (self.eval(x), self.grad(x))
        }
    }

    let obj = FrobeniusI;
    let mut x = array![2.0, 0.4, -0.1, -1.5];
    let mut solver = Solver::new(
        Method::Steepest,
        Control {
            maxiter: 20,
            gtol: 1e-8,
            istep: 0.1,
            maxmove: None,
        },
        4,
    );
    solver.set_manifold(ManifoldKind::Symmetric);
    solver.set_accept(rgmin::Accept::None);
    for _ in 0..20 {
        let _ = solver.step(&obj, &mut x).unwrap();
        assert!(is_symmetric(&x), "left the symmetric set {x:?}");
        assert!((x[1] - x[2]).abs() < 1e-12, "not symmetric {x:?}");
    }
}

#[test]
fn symmetric_rejects_a_3n_cluster() {
    let obj = Rosenbrock::<114>::new();
    let mut x = Array1::from_elem(114, 0.1);
    let mut solver = Solver::new(Method::Steepest, control(), 114);
    solver.set_manifold(rgmin::ManifoldKind::Symmetric);
    let err = solver.step(&obj, &mut x).unwrap_err();
    match err {
        rgmin::Error::ManifoldDim { kind, got } => {
            assert_eq!(kind, "symmetric");
            assert_eq!(got, 114);
        }
        other => panic!("expected ManifoldDim, got {other:?}"),
    }
}

#[test]
fn skewsymmetric_session_stays_on_the_set() {
    use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
    use ndarray::ArrayView1;
    use rgmin::ManifoldKind;
    use rgmin::manifold::is_skewsymmetric;

    // Frobenius distance to J = [[0, 1], [-1, 0]]. Identity is not on the set.
    struct FrobeniusJ;
    impl Objective<f64> for FrobeniusJ {
        fn dim(&self) -> usize {
            4
        }
        fn bounds(&self) -> &Bounds<f64> {
            use std::sync::OnceLock;
            static B: OnceLock<Bounds<f64>> = OnceLock::new();
            B.get_or_init(|| {
                Bounds::new(Array1::from_elem(4, -4.0), Array1::from_elem(4, 4.0), 0.0)
            })
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            let j = [0.0, 1.0, -1.0, 0.0];
            0.5 * x.iter().zip(j).map(|(a, b)| (a - b) * (a - b)).sum::<f64>()
        }
    }
    impl Gradient<f64> for FrobeniusJ {
        fn dim(&self) -> usize {
            4
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            let j = [0.0, 1.0, -1.0, 0.0];
            Array1::from_iter(x.iter().zip(j).map(|(a, b)| a - b))
        }
    }
    impl DifferentiableObjective<f64> for FrobeniusJ {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            (self.eval(x), self.grad(x))
        }
    }

    let obj = FrobeniusJ;
    let mut x = array![0.2, 0.4, -0.1, 0.3];
    let mut solver = Solver::new(
        Method::Steepest,
        Control {
            maxiter: 20,
            gtol: 1e-8,
            istep: 0.1,
            maxmove: None,
        },
        4,
    );
    solver.set_manifold(ManifoldKind::SkewSymmetric);
    solver.set_accept(rgmin::Accept::None);
    for _ in 0..20 {
        let _ = solver.step(&obj, &mut x).unwrap();
        assert!(is_skewsymmetric(&x), "left the skew-symmetric set {x:?}");
        assert!((x[0]).abs() < 1e-12, "nonzero diagonal {x:?}");
        assert!((x[3]).abs() < 1e-12, "nonzero diagonal {x:?}");
        assert!((x[1] + x[2]).abs() < 1e-12, "not skew {x:?}");
    }
}

#[test]
fn euclidean_complex_session_stays_on_the_set() {
    use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
    use ndarray::ArrayView1;
    use rgmin::ManifoldKind;
    use rgmin::manifold::is_euclidean_complex;

    struct CplxBowl;
    impl Objective<f64> for CplxBowl {
        fn dim(&self) -> usize {
            4
        }
        fn bounds(&self) -> &Bounds<f64> {
            use std::sync::OnceLock;
            static B: OnceLock<Bounds<f64>> = OnceLock::new();
            B.get_or_init(|| {
                Bounds::new(Array1::from_elem(4, -4.0), Array1::from_elem(4, 4.0), 0.0)
            })
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            0.5 * x.iter().map(|a| a * a).sum::<f64>()
        }
    }
    impl Gradient<f64> for CplxBowl {
        fn dim(&self) -> usize {
            4
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            x.to_owned()
        }
    }
    impl DifferentiableObjective<f64> for CplxBowl {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            (self.eval(x), self.grad(x))
        }
    }

    let obj = CplxBowl;
    let mut x = array![2.0, 1.0, -1.0, 3.0];
    let mut solver = Solver::new(
        Method::Steepest,
        Control {
            maxiter: 20,
            gtol: 1e-8,
            istep: 0.1,
            maxmove: None,
        },
        4,
    );
    solver.set_manifold(ManifoldKind::euclidean_complex(2));
    solver.set_accept(rgmin::Accept::None);
    let _ = solver.step(&obj, &mut x).unwrap();
    assert!(is_euclidean_complex(&x), "left C^2 {x:?}");
    assert_eq!(x.len(), 4);
    let fro = x.iter().map(|a| a * a).sum::<f64>().sqrt();
    assert!((fro - 1.0).abs() > 0.5, "must not be the sphere {x:?}");
    let n0 = (x[0] * x[0] + x[1] * x[1]).sqrt();
    let n1 = (x[2] * x[2] + x[3] * x[3]).sqrt();
    assert!((n0 - 1.0).abs() > 0.05, "must not force S^1 {x:?}");
    assert!((n1 - 1.0).abs() > 0.05, "must not force S^1 {x:?}");
    for _ in 0..19 {
        let _ = solver.step(&obj, &mut x).unwrap();
        assert!(is_euclidean_complex(&x), "left C^2 {x:?}");
        assert_eq!(x.len(), 4);
    }
}

#[test]
fn euclidean_complex_rejects_a_3n_cluster() {
    let obj = Rosenbrock::<114>::new();
    let mut x = Array1::from_elem(114, 0.1);
    let mut solver = Solver::new(Method::Steepest, control(), 114);
    solver.set_manifold(rgmin::ManifoldKind::euclidean_complex(2));
    let err = solver.step(&obj, &mut x).unwrap_err();
    match err {
        rgmin::Error::ManifoldDim { kind, got } => {
            assert_eq!(kind, "euclidean_complex");
            assert_eq!(got, 114);
        }
        other => panic!("expected ManifoldDim, got {other:?}"),
    }
}

#[test]
fn constant_session_stays_on_the_set() {
    use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
    use ndarray::ArrayView1;
    use rgmin::ManifoldKind;
    use rgmin::manifold::is_constant;

    struct Bowl;
    impl Objective<f64> for Bowl {
        fn dim(&self) -> usize {
            3
        }
        fn bounds(&self) -> &Bounds<f64> {
            use std::sync::OnceLock;
            static B: OnceLock<Bounds<f64>> = OnceLock::new();
            B.get_or_init(|| {
                Bounds::new(Array1::from_elem(3, -4.0), Array1::from_elem(3, 4.0), 0.0)
            })
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            0.5 * x.iter().map(|a| a * a).sum::<f64>()
        }
    }
    impl Gradient<f64> for Bowl {
        fn dim(&self) -> usize {
            3
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            x.to_owned()
        }
    }
    impl DifferentiableObjective<f64> for Bowl {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            (self.eval(x), self.grad(x))
        }
    }

    let obj = Bowl;
    let start = array![1.25, -0.5, 2.0];
    let mut x = start.clone();
    let mut solver = Solver::new(
        Method::Steepest,
        Control {
            maxiter: 20,
            gtol: 1e-8,
            istep: 0.1,
            maxmove: None,
        },
        3,
    );
    solver.set_manifold(ManifoldKind::constant(3));
    solver.set_accept(rgmin::Accept::None);
    for _ in 0..20 {
        let _ = solver.step(&obj, &mut x).unwrap();
        assert!(is_constant(&x, 3), "left the singleton {x:?}");
        assert!(
            (&x - &start).mapv(f64::abs).sum() < 1e-15,
            "moved off A {x:?}"
        );
    }
    let fro = x.iter().map(|a| a * a).sum::<f64>().sqrt();
    assert!((fro - 1.0).abs() > 0.5, "must not be the sphere {x:?}");
}

#[test]
fn constant_rejects_a_3n_cluster() {
    let obj = Rosenbrock::<114>::new();
    let mut x = Array1::from_elem(114, 0.1);
    let mut solver = Solver::new(Method::Steepest, control(), 114);
    solver.set_manifold(rgmin::ManifoldKind::constant(2));
    let err = solver.step(&obj, &mut x).unwrap_err();
    match err {
        rgmin::Error::ManifoldDim { kind, got } => {
            assert_eq!(kind, "constant");
            assert_eq!(got, 114);
        }
        other => panic!("expected ManifoldDim, got {other:?}"),
    }
}

#[test]
fn skewsymmetric_rejects_a_3n_cluster() {
    let obj = Rosenbrock::<114>::new();
    let mut x = Array1::from_elem(114, 0.1);
    let mut solver = Solver::new(Method::Steepest, control(), 114);
    solver.set_manifold(rgmin::ManifoldKind::SkewSymmetric);
    let err = solver.step(&obj, &mut x).unwrap_err();
    match err {
        rgmin::Error::ManifoldDim { kind, got } => {
            assert_eq!(kind, "skewsymmetric");
            assert_eq!(got, 114);
        }
        other => panic!("expected ManifoldDim, got {other:?}"),
    }
}

#[test]
fn rigid_quotient_drops_translation_and_keeps_3n() {
    use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
    use ndarray::ArrayView1;

    struct Pair;
    impl Objective<f64> for Pair {
        fn dim(&self) -> usize {
            9
        }
        fn bounds(&self) -> &Bounds<f64> {
            use std::sync::OnceLock;
            static B: OnceLock<Bounds<f64>> = OnceLock::new();
            B.get_or_init(|| {
                Bounds::new(Array1::from_elem(9, -4.0), Array1::from_elem(9, 4.0), 0.0)
            })
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            let d01 = (x[0] - x[3]).powi(2) + (x[1] - x[4]).powi(2) + (x[2] - x[5]).powi(2);
            let d02 = (x[0] - x[6]).powi(2) + (x[1] - x[7]).powi(2) + (x[2] - x[8]).powi(2);
            0.5 * ((d01.sqrt() - 1.0).powi(2) + (d02.sqrt() - 1.0).powi(2))
        }
    }
    impl Gradient<f64> for Pair {
        fn dim(&self) -> usize {
            9
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            let e = 1e-6;
            let f0 = self.eval(x);
            let mut g = Array1::zeros(9);
            let mut y = x.to_owned();
            for i in 0..9 {
                y[i] += e;
                g[i] = (self.eval(y.view()) - f0) / e;
                y[i] = x[i];
            }
            g
        }
    }
    impl DifferentiableObjective<f64> for Pair {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            (self.eval(x), self.grad(x))
        }
    }

    let obj = Pair;
    let mut x = array![0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.0, 1.2, 0.0];
    let mut solver = Solver::new(Method::Steepest, control(), 9);
    solver.set_manifold(rgmin::ManifoldKind::RigidQuotient);
    let com0 = [(x[0] + x[3] + x[6]) / 3.0, (x[1] + x[4] + x[7]) / 3.0];
    for _ in 0..20 {
        let _ = solver.step(&obj, &mut x).unwrap();
        assert_eq!(x.len(), 9);
    }
    let com1 = [(x[0] + x[3] + x[6]) / 3.0, (x[1] + x[4] + x[7]) / 3.0];
    assert!(
        (com1[0] - com0[0]).abs() < 1e-8,
        "COM drifted {com0:?} -> {com1:?}"
    );
    assert!((com1[1] - com0[1]).abs() < 1e-8);
}

#[test]
fn nlcg_second_step_is_not_steepest() {
    let obj = Rosenbrock::<2>::new();
    let mut a = array![-1.2, 1.0];
    let mut b = array![-1.2, 1.0];
    let mut pr = Solver::new(Method::polak_ribiere(), control(), 2);
    let mut sd = Solver::new(Method::Steepest, control(), 2);
    let _ = pr.step(&obj, &mut a).unwrap();
    let _ = sd.step(&obj, &mut b).unwrap();
    let _ = pr.step(&obj, &mut a).unwrap();
    let _ = sd.step(&obj, &mut b).unwrap();
    // After two steps the conjugacy history must move PR off the SD path.
    assert!(
        (a[0] - b[0]).abs() + (a[1] - b[1]).abs() > 1e-12,
        "PR and steepest stayed together: {a:?} {b:?}"
    );
}

#[test]
fn fire_and_bb_kill_a_sphere() {
    use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
    use ndarray::ArrayView1;
    use rgmin::Accept;

    struct Sphere;
    impl Objective<f64> for Sphere {
        fn dim(&self) -> usize {
            2
        }
        fn bounds(&self) -> &Bounds<f64> {
            use std::sync::OnceLock;
            static B: OnceLock<Bounds<f64>> = OnceLock::new();
            B.get_or_init(|| Bounds::new(array![-1e6, -1e6], array![1e6, 1e6], 0.0))
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            x[0] * x[0] + x[1] * x[1]
        }
    }
    impl Gradient<f64> for Sphere {
        fn dim(&self) -> usize {
            2
        }
        fn grad(&self, x: ArrayView1<f64>) -> ndarray::Array1<f64> {
            array![2.0 * x[0], 2.0 * x[1]]
        }
    }
    impl DifferentiableObjective<f64> for Sphere {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, ndarray::Array1<f64>) {
            (self.eval(x), self.grad(x))
        }
    }

    let obj = Sphere;
    for method in [
        Method::Fire {
            kind: rgmin::FireKind::V1,
        },
        Method::Fire {
            kind: rgmin::FireKind::V2,
        },
        Method::Bb,
    ] {
        let mut x = array![1.5, -2.0];
        let mut solver = Solver::new(
            method.clone(),
            Control {
                maxiter: 200,
                gtol: 1e-8,
                istep: 0.2,
                maxmove: None,
            },
            2,
        );
        solver.set_accept(Accept::Nonmonotone);
        let mut last = None;
        for _ in 0..200 {
            let rep = solver.step(&obj, &mut x).unwrap();
            last = Some(rep);
            if last.as_ref().unwrap().grad_norm < 1e-7 {
                break;
            }
        }
        let rep = last.unwrap();
        assert!(
            rep.grad_norm < 1e-5 && rep.value < 1e-10,
            "{method:?} gnorm {} value {}",
            rep.grad_norm,
            rep.value
        );
    }
}

#[test]
fn dogleg_kills_a_quadratic() {
    use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
    use ndarray::{Array2, ArrayView1};
    use rgmin::HessianObjective;

    struct Quad;
    impl Objective<f64> for Quad {
        fn dim(&self) -> usize {
            2
        }
        fn bounds(&self) -> &Bounds<f64> {
            use std::sync::OnceLock;
            static B: OnceLock<Bounds<f64>> = OnceLock::new();
            B.get_or_init(|| Bounds::new(array![-1e6, -1e6], array![1e6, 1e6], 0.0))
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            5.0 * x[0] * x[0] + 0.5 * x[1] * x[1]
        }
    }
    impl Gradient<f64> for Quad {
        fn dim(&self) -> usize {
            2
        }
        fn grad(&self, x: ArrayView1<f64>) -> ndarray::Array1<f64> {
            array![10.0 * x[0], x[1]]
        }
    }
    impl DifferentiableObjective<f64> for Quad {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, ndarray::Array1<f64>) {
            (self.eval(x), self.grad(x))
        }
    }
    impl HessianObjective for Quad {
        fn hessian(&self, _x: ArrayView1<f64>) -> Array2<f64> {
            Array2::from_shape_vec((2, 2), vec![10.0, 0.0, 0.0, 1.0]).unwrap()
        }
    }

    let obj = Quad;
    let mut x = array![2.0, -3.0];
    let mut solver = Solver::new(
        Method::Dogleg,
        Control {
            maxiter: 10,
            gtol: 1e-10,
            istep: 4.0,
            maxmove: None,
        },
        2,
    );
    let mut last = None;
    for _ in 0..8 {
        let rep = solver.step_hess(&obj, &mut x).unwrap();
        last = Some(rep);
        if last.as_ref().unwrap().grad_norm < 1e-10 {
            break;
        }
    }
    let rep = last.unwrap();
    assert!(rep.value < 1e-12, "dogleg value {}", rep.value);
    assert!(x.iter().all(|v| v.abs() < 1e-6));
}

/// The energy policy's fallback faces the same test as the steps it
/// replaces: an oracle that rises in every direction must come back
/// unmoved rather than accepting an uphill steepest step the policy
/// spent ten halvings refusing.
#[test]
fn an_uphill_everywhere_oracle_is_refused_not_moved() {
    use ndarray::ArrayView1;
    let obj = rgmin::Oracle::unbounded(6, |x: ArrayView1<f64>| {
        // A cone rising away from the origin: every step from the
        // origin increases the value, and the gradient at the origin
        // pretends to point somewhere useful.
        let r = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        let g = if r > 1e-12 {
            x.mapv(|v| v / r)
        } else {
            Array1::from(vec![1.0; x.len()])
        };
        (r, g)
    });
    let mut solver = rgmin::Solver::new(rgmin::Method::Steepest, rgmin::Control::default(), 6);
    solver.set_accept(rgmin::Accept::Energy);
    let mut x = Array1::from(vec![0.0; 6]);
    let rep = solver.step(&obj, &mut x).expect("step runs");
    assert!(
        rep.value <= 1e-12,
        "refused steps must not report an uphill point as progress, got {}",
        rep.value
    );
    assert!(
        x.iter().all(|v| v.abs() <= 1e-12),
        "a refused step must leave the position where it stood"
    );
}
