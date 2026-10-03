use eindir_core::objectives::Rosenbrock;
use ndarray::{Array1, array};
use rgmin::{Control, Method, Solver};

fn control() -> Control {
    Control {
        maxiter: 80,
        gtol: 1e-8,
        istep: 0.1,
        maxmove: None,
        ftol_rel: None,
    }
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
            ftol_rel: None,
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
            ftol_rel: None,
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
            ftol_rel: None,
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
fn multinomial_sym_session_stays_symmetric_doubly_stochastic() {
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
            ftol_rel: None,
        },
        4,
    );
    solver.set_multinomial_sym(2);
    solver.set_accept(rgmin::Accept::None);
    for _ in 0..20 {
        let _ = solver.step(&obj, &mut x).unwrap();
        assert!(x.iter().all(|&xi| xi > 0.0), "left the interior {x:?}");
        assert!((x[1] - x[2]).abs() < 1e-12, "not symmetric {x:?}");
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
fn multinomial_sym_rejects_a_wrong_length() {
    let obj = Rosenbrock::<1>::new();
    let mut x = array![1.0];
    let mut solver = Solver::new(Method::Steepest, control(), 1);
    solver.set_manifold(rgmin::ManifoldKind::multinomial_sym(2));
    let err = solver.step(&obj, &mut x).unwrap_err();
    match err {
        rgmin::Error::ManifoldDim { kind, got } => {
            assert_eq!(kind, "multinomialsymmetric");
            assert_eq!(got, 1);
        }
        other => panic!("expected ManifoldDim, got {other:?}"),
    }
}

#[test]
fn sphere_complex_session_stays_on_the_set() {
    use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
    use ndarray::ArrayView1;

    struct FirstImag;
    impl Objective<f64> for FirstImag {
        fn dim(&self) -> usize {
            4
        }
        fn bounds(&self) -> &Bounds<f64> {
            use std::sync::OnceLock;
            static B: OnceLock<Bounds<f64>> = OnceLock::new();
            B.get_or_init(|| {
                Bounds::new(Array1::from_elem(4, -1e12), Array1::from_elem(4, 1e12), 0.0)
            })
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            x[2]
        }
    }
    impl Gradient<f64> for FirstImag {
        fn dim(&self) -> usize {
            4
        }
        fn grad(&self, _x: ArrayView1<f64>) -> Array1<f64> {
            array![0.0, 0.0, 1.0, 0.0]
        }
    }
    impl DifferentiableObjective<f64> for FirstImag {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            (self.eval(x), self.grad(x))
        }
    }

    let obj = FirstImag;
    let mut x = array![1.0, 0.0, 0.0, 0.0];
    let mut solver = Solver::new(
        Method::Steepest,
        Control {
            maxiter: 20,
            gtol: 1e-10,
            istep: 0.1,
            maxmove: None,
            ftol_rel: None,
        },
        4,
    );
    solver.set_sphere_complex(2);
    solver.set_accept(rgmin::Accept::None);
    for _ in 0..8 {
        let _ = solver.step(&obj, &mut x).unwrap();
        let nrm = x.iter().map(|xi| xi * xi).sum::<f64>().sqrt();
        assert!((nrm - 1.0).abs() < 1e-12, "left the complex sphere {x:?}");
    }
}

#[test]
fn positive_session_stays_on_the_set() {
    use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
    use ndarray::ArrayView1;
    use rgmin::manifold::is_positive;

    struct SumLin;
    impl Objective<f64> for SumLin {
        fn dim(&self) -> usize {
            3
        }
        fn bounds(&self) -> &Bounds<f64> {
            use std::sync::OnceLock;
            static B: OnceLock<Bounds<f64>> = OnceLock::new();
            B.get_or_init(|| {
                Bounds::new(Array1::from_elem(3, 1e-12), Array1::from_elem(3, 1e12), 0.0)
            })
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            x.iter().sum()
        }
    }
    impl Gradient<f64> for SumLin {
        fn dim(&self) -> usize {
            3
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            Array1::ones(x.len())
        }
    }
    impl DifferentiableObjective<f64> for SumLin {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            (self.eval(x), self.grad(x))
        }
    }

    let obj = SumLin;
    let mut x = array![2.0, 0.5, 4.0];
    let mut solver = Solver::new(
        Method::Steepest,
        Control {
            maxiter: 20,
            gtol: 1e-10,
            istep: 0.1,
            maxmove: None,
            ftol_rel: None,
        },
        3,
    );
    solver.set_positive(3);
    solver.set_accept(rgmin::Accept::None);
    let start = x.clone();
    let _ = solver.step(&obj, &mut x).unwrap();
    assert!(is_positive(&x), "left the positive orthant {x:?}");
    let fro = x.iter().map(|xi| xi * xi).sum::<f64>().sqrt();
    assert!((fro - 1.0).abs() > 0.5, "must not be the sphere {x:?}");
    assert!(
        (&x - &start).mapv(f64::abs).sum() > 1e-12,
        "expected a retraction step {x:?}"
    );
    for _ in 0..7 {
        let _ = solver.step(&obj, &mut x).unwrap();
        assert!(is_positive(&x), "left the positive orthant {x:?}");
    }
}

#[test]
fn centered_matrix_session_stays_on_the_set() {
    use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
    use ndarray::ArrayView1;
    use rgmin::manifold::is_centered;

    struct WeightedLin;
    impl Objective<f64> for WeightedLin {
        fn dim(&self) -> usize {
            4
        }
        fn bounds(&self) -> &Bounds<f64> {
            use std::sync::OnceLock;
            static B: OnceLock<Bounds<f64>> = OnceLock::new();
            B.get_or_init(|| {
                Bounds::new(Array1::from_elem(4, -1e12), Array1::from_elem(4, 1e12), 0.0)
            })
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            x.iter()
                .enumerate()
                .map(|(i, xi)| (i as f64 + 1.0) * xi)
                .sum()
        }
    }
    impl Gradient<f64> for WeightedLin {
        fn dim(&self) -> usize {
            4
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            Array1::from_iter((0..x.len()).map(|i| i as f64 + 1.0))
        }
    }
    impl DifferentiableObjective<f64> for WeightedLin {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            (self.eval(x), self.grad(x))
        }
    }

    let obj = WeightedLin;
    let mut x = array![1.0, -1.0, 2.0, -2.0];
    let mut solver = Solver::new(
        Method::Steepest,
        Control {
            maxiter: 20,
            gtol: 1e-10,
            istep: 0.1,
            maxmove: None,
            ftol_rel: None,
        },
        4,
    );
    solver.set_centered_matrix(2, 2, false);
    solver.set_accept(rgmin::Accept::None);
    let start = x.clone();
    let _ = solver.step(&obj, &mut x).unwrap();
    assert!(is_centered(&x, 2, 2, false), "left the centered set {x:?}");
    let fro = x.iter().map(|xi| xi * xi).sum::<f64>().sqrt();
    assert!((fro - 1.0).abs() > 0.5, "must not be the sphere {x:?}");
    assert!(
        (&x - &start).mapv(f64::abs).sum() > 1e-12,
        "expected a retraction step {x:?}"
    );
    for _ in 0..7 {
        let _ = solver.step(&obj, &mut x).unwrap();
        assert!(is_centered(&x, 2, 2, false), "left the centered set {x:?}");
    }
}

#[test]
fn positive_rejects_a_3n_cluster() {
    let obj = Rosenbrock::<114>::new();
    let mut x = Array1::from_elem(114, 0.1);
    let mut solver = Solver::new(Method::Steepest, control(), 114);
    solver.set_manifold(rgmin::ManifoldKind::positive(2));
    let err = solver.step(&obj, &mut x).unwrap_err();
    match err {
        rgmin::Error::ManifoldDim { kind, got } => {
            assert_eq!(kind, "positive");
            assert_eq!(got, 114);
        }
        other => panic!("expected ManifoldDim, got {other:?}"),
    }
}

#[test]
fn centered_matrix_rejects_a_3n_cluster() {
    let obj = Rosenbrock::<114>::new();
    let mut x = Array1::from_elem(114, 0.1);
    let mut solver = Solver::new(Method::Steepest, control(), 114);
    solver.set_manifold(rgmin::ManifoldKind::centered_matrix(2, 2, false));
    let err = solver.step(&obj, &mut x).unwrap_err();
    match err {
        rgmin::Error::ManifoldDim { kind, got } => {
            assert_eq!(kind, "centeredmatrix");
            assert_eq!(got, 114);
        }
        other => panic!("expected ManifoldDim, got {other:?}"),
    }
}

#[test]
fn sphere_complex_rejects_a_wrong_length() {
    let obj = Rosenbrock::<1>::new();
    let mut x = array![1.0];
    let mut solver = Solver::new(Method::Steepest, control(), 1);
    solver.set_manifold(rgmin::ManifoldKind::sphere_complex(2));
    let err = solver.step(&obj, &mut x).unwrap_err();
    match err {
        rgmin::Error::ManifoldDim { kind, got } => {
            assert_eq!(kind, "spherecomplex");
            assert_eq!(got, 1);
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
            ftol_rel: None,
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
            ftol_rel: None,
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
            ftol_rel: None,
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
            ftol_rel: None,
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
            ftol_rel: None,
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
fn grassmann_session_stays_on_gr_4_2() {
    use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
    use ndarray::ArrayView1;
    use rgmin::{Grassmann, Manifold, ManifoldKind};

    struct Brockett;
    impl Objective<f64> for Brockett {
        fn dim(&self) -> usize {
            8
        }
        fn bounds(&self) -> &Bounds<f64> {
            use std::sync::OnceLock;
            static B: OnceLock<Bounds<f64>> = OnceLock::new();
            B.get_or_init(|| {
                Bounds::new(Array1::from_elem(8, -4.0), Array1::from_elem(8, 4.0), 0.0)
            })
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            // f(X) = 1*||col0||^2 + 2*||col1||^2 on the first two
            // coordinates of each column: a linear-in-X^T A X stand-in
            // whose Euclidean gradient is not already horizontal.
            0.5 * (1.0 * x[0] * x[0] + 2.0 * x[1] * x[1] + 3.0 * x[4] * x[4] + 4.0 * x[5] * x[5])
        }
    }
    impl Gradient<f64> for Brockett {
        fn dim(&self) -> usize {
            8
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            let mut g = Array1::zeros(8);
            g[0] = x[0];
            g[1] = 2.0 * x[1];
            g[4] = 3.0 * x[4];
            g[5] = 4.0 * x[5];
            g
        }
    }
    impl DifferentiableObjective<f64> for Brockett {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            (self.eval(x), self.grad(x))
        }
    }

    let g = Grassmann { n: 4, p: 2 };
    let mut x = array![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let mut solver = Solver::new(
        Method::Steepest,
        Control {
            maxiter: 25,
            gtol: 1e-10,
            istep: 0.15,
            maxmove: None,
            ftol_rel: None,
        },
        8,
    );
    solver.set_manifold(ManifoldKind::Grassmann);
    solver.set_factor_shape(4, 2);
    solver.set_accept(rgmin::Accept::None);
    for _ in 0..25 {
        let _ = solver.step(&Brockett, &mut x).unwrap();
        let yty = {
            let cols = g.unpack(&x).unwrap();
            [
                cols[0].dot(&cols[0]),
                cols[0].dot(&cols[1]),
                cols[1].dot(&cols[1]),
            ]
        };
        assert!((yty[0] - 1.0).abs() < 1e-10, "col0 {yty:?} x={x:?}");
        assert!(yty[1].abs() < 1e-10, "not orthogonal {yty:?} x={x:?}");
        assert!((yty[2] - 1.0).abs() < 1e-10, "col1 {yty:?} x={x:?}");
        let t = g.project(&x, &Array1::from_elem(8, 0.3));
        let xtt0 = g.unpack(&x).unwrap()[0].dot(&g.unpack(&t).unwrap()[0]);
        let xtt1 = g.unpack(&x).unwrap()[1].dot(&g.unpack(&t).unwrap()[1]);
        assert!(xtt0.abs() < 1e-10 && xtt1.abs() < 1e-10);
    }
}

#[test]
fn grassmann_shape_rejects_a_3n_cluster() {
    let obj = Rosenbrock::<114>::new();
    let mut x = Array1::from_elem(114, 0.1);
    let mut solver = Solver::new(Method::Steepest, control(), 114);
    solver.set_manifold(rgmin::ManifoldKind::Grassmann);
    solver.set_factor_shape(4, 2);
    let err = solver.step(&obj, &mut x).unwrap_err();
    match err {
        rgmin::Error::ManifoldDim { kind, got } => {
            assert_eq!(kind, "grassmann");
            assert_eq!(got, 114);
        }
        other => panic!("expected ManifoldDim, got {other:?}"),
    }
}

#[test]
fn hyperbolic_session_stays_on_the_hyperboloid() {
    use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
    use ndarray::ArrayView1;
    use rgmin::ManifoldKind;
    use rgmin::manifold::{minkowski, pack as pack_h};

    // Spatial-radius squared on H^2. The origin (1, 0, 0) is the min.
    struct SpatialRadius;
    impl Objective<f64> for SpatialRadius {
        fn dim(&self) -> usize {
            3
        }
        fn bounds(&self) -> &Bounds<f64> {
            use std::sync::OnceLock;
            static B: OnceLock<Bounds<f64>> = OnceLock::new();
            B.get_or_init(|| Bounds::new(array![-8.0, -8.0, -8.0], array![8.0, 8.0, 8.0], 0.0))
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            0.5 * (x[1] * x[1] + x[2] * x[2])
        }
    }
    impl Gradient<f64> for SpatialRadius {
        fn dim(&self) -> usize {
            3
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            array![0.0, x[1], x[2]]
        }
    }
    impl DifferentiableObjective<f64> for SpatialRadius {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            (self.eval(x), self.grad(x))
        }
    }

    let obj = SpatialRadius;
    let mut x = pack_h((2.0_f64).sqrt(), array![1.0, 0.0].view());
    let mut solver = Solver::new(
        Method::Steepest,
        Control {
            maxiter: 40,
            gtol: 1e-8,
            istep: 0.2,
            maxmove: None,
            ftol_rel: None,
        },
        3,
    );
    solver.set_manifold(ManifoldKind::Hyperbolic);
    solver.set_accept(rgmin::Accept::None);
    for _ in 0..40 {
        let _ = solver.step(&obj, &mut x).unwrap();
        let q = minkowski(x.view(), x.view());
        assert!(
            (q + 1.0).abs() < 1e-10,
            "left the hyperboloid minkowski={q} x={x:?}"
        );
        assert!(x[0] > 0.0, "left the positive sheet {x:?}");
    }
}

#[test]
fn poincare_session_stays_inside_the_open_ball() {
    use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
    use ndarray::ArrayView1;
    use rgmin::ManifoldKind;

    struct Drift;
    impl Objective<f64> for Drift {
        fn dim(&self) -> usize {
            3
        }
        fn bounds(&self) -> &Bounds<f64> {
            use std::sync::OnceLock;
            static B: OnceLock<Bounds<f64>> = OnceLock::new();
            B.get_or_init(|| Bounds::new(array![-2.0, -2.0, -2.0], array![2.0, 2.0, 2.0], 0.0))
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            -x[0]
        }
    }
    impl Gradient<f64> for Drift {
        fn dim(&self) -> usize {
            3
        }
        fn grad(&self, _x: ArrayView1<f64>) -> Array1<f64> {
            array![-1.0, 0.0, 0.0]
        }
    }
    impl DifferentiableObjective<f64> for Drift {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            (self.eval(x), self.grad(x))
        }
    }

    let obj = Drift;
    let mut x = array![0.1, 0.0, 0.0];
    let mut solver = Solver::new(
        Method::Steepest,
        Control {
            maxiter: 40,
            gtol: 1e-12,
            istep: 0.4,
            maxmove: None,
            ftol_rel: None,
        },
        3,
    );
    solver.set_manifold(ManifoldKind::PoincareBall);
    solver.set_accept(rgmin::Accept::None);
    for _ in 0..40 {
        let _ = solver.step(&obj, &mut x).unwrap();
        let nrm = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
        assert!(nrm < 1.0, "left the open ball {x:?}");
    }
}

#[test]
fn unitary_session_stays_on_the_set() {
    use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
    use ndarray::ArrayView1;
    use rgmin::ManifoldKind;
    use rgmin::manifold::is_unitary;

    /// Minus Re(tr U) on packed U(2). Minimizer is the identity.
    struct MinusReTrace;
    impl Objective<f64> for MinusReTrace {
        fn dim(&self) -> usize {
            8
        }
        fn bounds(&self) -> &Bounds<f64> {
            use std::sync::OnceLock;
            static B: OnceLock<Bounds<f64>> = OnceLock::new();
            B.get_or_init(|| {
                Bounds::new(Array1::from_elem(8, -2.0), Array1::from_elem(8, 2.0), 0.0)
            })
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            -(x[0] + x[6])
        }
    }
    impl Gradient<f64> for MinusReTrace {
        fn dim(&self) -> usize {
            8
        }
        fn grad(&self, _x: ArrayView1<f64>) -> Array1<f64> {
            let mut g = Array1::zeros(8);
            g[0] = -1.0;
            g[6] = -1.0;
            g
        }
    }
    impl DifferentiableObjective<f64> for MinusReTrace {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            (self.eval(x), self.grad(x))
        }
    }

    let obj = MinusReTrace;
    let s = 0.5_f64.sqrt();
    let mut x = array![s, 0.0, s, 0.0, 0.0, s, 0.0, -s];
    let mut solver = Solver::new(
        Method::Steepest,
        Control {
            maxiter: 20,
            gtol: 1e-8,
            istep: 0.2,
            maxmove: None,
            ftol_rel: None,
        },
        8,
    );
    solver.set_manifold(ManifoldKind::unitary(2));
    solver.set_accept(rgmin::Accept::None);
    for _ in 0..20 {
        let _ = solver.step(&obj, &mut x).unwrap();
        assert!(is_unitary(&x), "left U(2) {x:?}");
    }
}

#[test]
fn unitary_rejects_a_3n_cluster() {
    let obj = Rosenbrock::<114>::new();
    let mut x = Array1::from_elem(114, 0.1);
    let mut solver = Solver::new(Method::Steepest, control(), 114);
    solver.set_manifold(rgmin::ManifoldKind::unitary(2));
    let err = solver.step(&obj, &mut x).unwrap_err();
    match err {
        rgmin::Error::ManifoldDim { kind, got } => {
            assert_eq!(kind, "unitary");
            assert_eq!(got, 114);
        }
        other => panic!("expected ManifoldDim, got {other:?}"),
    }
}
