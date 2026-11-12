//! Projected L-BFGS on the complement of one mode, and a restart when
//! the position and that direction repeat.

use ndarray::{Array1, array};
use rgmin::{Control, Method, Solver};

fn session() -> Solver {
    Solver::new(Method::lbfgs(), Control::default(), 3)
}

#[test]
fn the_complement_direction_drops_the_mode() {
    let mut solver = session();
    assert!(solver.set_complement_mode(Some(array![1.0, 0.0, 0.0])));
    let step = solver
        .complement_direction(array![0.2, 0.0, 0.0].view(), array![4.0, -1.5, 0.5].view())
        .unwrap();
    assert!(step[0].abs() < 1e-12, "{step:?}");
    assert!((step[1] - 1.5).abs() < 1e-12, "{step:?}");
    assert!((step[2] + 0.5).abs() < 1e-12, "{step:?}");
}

#[test]
fn a_pair_along_the_mode_is_not_stored() {
    let mut solver = session();
    solver.set_complement_mode(Some(array![0.0, 1.0, 0.0]));
    assert!(solver.push_pair(array![0.0, 0.4, 0.0].view(), array![0.0, 1.0, 0.0].view()));
    assert_eq!(solver.pair_count(), 0);
    assert!(solver.push_pair(array![0.2, 0.4, 0.0].view(), array![1.0, 9.0, 0.0].view()));
    assert_eq!(solver.pair_count(), 1);
    let step = solver
        .complement_direction(array![0.0, 0.0, 0.0].view(), array![0.0, 3.0, 0.0].view())
        .unwrap();
    assert!(step.dot(&array![0.0, 1.0, 0.0]).abs() < 1e-8, "{step:?}");
}

#[test]
fn a_repeated_position_and_direction_restarts() {
    let mut solver = session();
    solver.set_complement_mode(Some(array![1.0, 0.0, 0.0]));
    let x = array![0.4, 0.2, -0.1];
    let g = array![0.0, 1.0, -0.25];
    let first = solver.complement_direction(x.view(), g.view()).unwrap();
    let second = solver.complement_direction(x.view(), g.view()).unwrap();
    let align = first.dot(&second) / (first.dot(&first).sqrt() * second.dot(&second).sqrt());
    assert!(
        align < 0.5,
        "repeated state kept its direction: {first:?} then {second:?}"
    );
    assert!(second.dot(&array![1.0, 0.0, 0.0]).abs() < 1e-8 * second.dot(&second).sqrt());
}

#[test]
fn an_unset_mode_keeps_steepest_descent() {
    let mut solver = session();
    let g = array![1.0, -2.0, 0.5];
    let step = solver
        .complement_direction(Array1::zeros(3).view(), g.view())
        .unwrap();
    assert_eq!(step, -&g);
}
