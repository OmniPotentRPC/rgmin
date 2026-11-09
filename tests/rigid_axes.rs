//! Rigid-axis masks on a session.
//!
//! A zero projected gradient stops under `Control::gtol`. Empty masks
//! keep the projection `set_project_rigid` already used. `set_periodic`
//! still only sets the cell flag.

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};
use rgmin::{Control, Method, RigidMode, Solver};

/// `E = 1/2 (y0 + y1 + y2)^2`. The gradient is a pure y translation.
struct SlideY;

impl Objective<f64> for SlideY {
    fn dim(&self) -> usize {
        9
    }
    fn bounds(&self) -> &Bounds<f64> {
        use std::sync::OnceLock;
        static B: OnceLock<Bounds<f64>> = OnceLock::new();
        B.get_or_init(|| Bounds::new(Array1::from_elem(9, -1e6), Array1::from_elem(9, 1e6), 0.0))
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        let s = x[1] + x[4] + x[7];
        0.5 * s * s
    }
}

impl Gradient<f64> for SlideY {
    fn dim(&self) -> usize {
        9
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        let s = x[1] + x[4] + x[7];
        let mut g = Array1::zeros(9);
        g[1] = s;
        g[4] = s;
        g[7] = s;
        g
    }
}

impl DifferentiableObjective<f64> for SlideY {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.grad(x))
    }
}

fn start() -> Array1<f64> {
    array![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]
}

fn solver() -> Solver {
    Solver::new(
        Method::lbfgs(),
        Control {
            gtol: 1e-8,
            istep: 0.1,
            ..Control::default()
        },
        9,
    )
}

#[test]
fn the_existing_projection_removes_a_translation() {
    let mut session = solver();
    session.set_project_rigid(true);
    let mut x = start();
    let before = x.clone();
    let rep = session.step(&SlideY, &mut x).unwrap();
    assert!(rep.grad_norm < 1e-8, "gnorm {}", rep.grad_norm);
    assert_eq!(x, before);
}

#[test]
fn a_periodic_x_axis_keeps_the_y_translation() {
    let mut session = solver();
    session.set_pbc_axes(true, false, false);
    let mut x = start();
    let rep = session.step(&SlideY, &mut x).unwrap();
    assert!(rep.grad_norm > 1.0, "gnorm {}", rep.grad_norm);
    assert!(x[1] < 1.0);
}

#[test]
fn a_broken_y_translation_is_not_removed() {
    let mut session = solver();
    session.set_project_rigid(true);
    session.set_broken_axes(false, true, false, false, false, false);
    let mut x = start();
    let rep = session.step(&SlideY, &mut x).unwrap();
    assert!(rep.grad_norm > 1.0, "gnorm {}", rep.grad_norm);
}

#[test]
fn a_frozen_coordinate_does_not_move() {
    struct Pull;
    impl Objective<f64> for Pull {
        fn dim(&self) -> usize {
            9
        }
        fn bounds(&self) -> &Bounds<f64> {
            use std::sync::OnceLock;
            static B: OnceLock<Bounds<f64>> = OnceLock::new();
            B.get_or_init(|| {
                Bounds::new(Array1::from_elem(9, -1e6), Array1::from_elem(9, 1e6), 0.0)
            })
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            0.5 * x[0] * x[0]
        }
    }
    impl Gradient<f64> for Pull {
        fn dim(&self) -> usize {
            9
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            let mut g = Array1::zeros(9);
            g[0] = x[0];
            g
        }
    }
    impl DifferentiableObjective<f64> for Pull {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            (self.eval(x), self.grad(x))
        }
    }

    let mut session = solver();
    let mut frozen = vec![0u8; 9];
    frozen[0] = 1;
    session.set_frozen(&frozen);
    let mut x = array![1.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.4, 0.2, 0.0];
    let rep = session.step(&Pull, &mut x).unwrap();
    assert!(rep.grad_norm < 1e-8, "gnorm {}", rep.grad_norm);
    assert_eq!(x[0], 1.0);
}

#[test]
fn off_clears_the_projection_and_periodic_stays_a_flag() {
    let mut session = solver();
    session.set_rigid_mode(RigidMode::Translation);
    session.set_rigid_mode(RigidMode::Off);
    session.set_periodic(true);
    let mut x = start();
    let rep = session.step(&SlideY, &mut x).unwrap();
    assert!(rep.grad_norm > 1.0, "gnorm {}", rep.grad_norm);
}

#[test]
fn the_mass_metric_still_removes_a_uniform_translation() {
    let mut session = solver();
    session.set_project_rigid(true);
    session.set_masses(array![16.0, 1.0, 1.0]);
    session.set_rigid_mass_metric(true);
    let mut x = start();
    let before = x.clone();
    let rep = session.step(&SlideY, &mut x).unwrap();
    assert!(rep.grad_norm < 1e-8, "gnorm {}", rep.grad_norm);
    assert_eq!(x, before);
}
