//! One slot for Exp, Lindh, and the bonded metric.

use ndarray::array;
use rgmin::{
    BondTerm, Control, FfParams, FfTerms, FireKind, Ghost, LindhParams, Method, Neighbours,
    OptimizerFamily, PreconSpec, Solver,
};

fn session(method: Method, dim: usize) -> Solver {
    Solver::new(
        method,
        Control {
            maxiter: 1,
            gtol: 1e-8,
            istep: 0.2,
            maxmove: None,
            ..Control::default()
        },
        dim,
    )
}

#[test]
fn set_precon_replaces_the_slot_and_reports_the_scale() {
    let pos = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let mut neigh = Neighbours::default();
    neigh.i.push(0);
    neigh.j.push(1);
    neigh.r.push(1.0);
    let exp = rgmin::ExpParams {
        mu: Some(1.5),
        r_nn: Some(1.0),
        ..rgmin::ExpParams::default()
    };
    let mut solver = session(Method::Lbfgs { memory: 4 }, pos.len());
    assert_eq!(solver.optimizer_family(), OptimizerFamily::QuasiNewton);
    assert!(solver.precon_scale().is_none());
    solver
        .set_precon(PreconSpec::Exp {
            positions: pos.view(),
            neighbours: &neigh,
            params: &exp,
        })
        .unwrap();
    assert_eq!(solver.precon_scale(), Some(1.5));
    assert_eq!(solver.exp_mu(), Some(1.5));

    let lindh = LindhParams {
        mu: Some(2.0),
        ..LindhParams::default()
    };
    solver
        .set_precon(PreconSpec::Lindh {
            positions: pos.view(),
            z: &[8, 1],
            ghosts: &[] as &[Ghost],
            params: &lindh,
        })
        .unwrap();
    assert_eq!(solver.precon_scale(), Some(2.0));
    assert_eq!(solver.lindh_mu(), Some(2.0));
    assert!(solver.exp_mu().is_none());

    let bonds = [BondTerm { i: 0, j: 1, k: 4.0 }];
    let terms = FfTerms {
        bonds: &bonds,
        ..FfTerms::default()
    };
    let ff = FfParams {
        mu: Some(3.0),
        ..FfParams::default()
    };
    solver.set_ff(pos.view(), terms, &ff).unwrap();
    assert_eq!(solver.precon_scale(), Some(3.0));
    assert_eq!(solver.ff_mu(), Some(3.0));
    assert!(solver.lindh_mu().is_none());

    solver.clear_precon();
    assert!(solver.precon_scale().is_none());
    assert!(solver.ff_mu().is_none());
}

#[test]
fn optimizer_family_follows_the_method() {
    let lbfgs = session(Method::lbfgs(), 2);
    assert_eq!(lbfgs.optimizer_family(), OptimizerFamily::QuasiNewton);
    let cg = session(Method::polak_ribiere(), 2);
    assert_eq!(cg.optimizer_family(), OptimizerFamily::Cg);
    let fire = session(Method::Fire { kind: FireKind::V1 }, 2);
    assert_eq!(fire.optimizer_family(), OptimizerFamily::Fire);
}

#[test]
fn geometry_step_defaults_on_and_can_be_cleared() {
    let mut solver = session(Method::lbfgs(), 3);
    solver.set_geometry_step(false);
    solver.set_geometry_step(true);
}
