//! A session on the extended cell layout: relaxation of a strained
//! periodic crystal under both L-BFGS step rules, with and without the
//! exponential preconditioner and its cell block.

mod cell_support;

use std::sync::atomic::{AtomicUsize, Ordering};

use cell_support::{evaluate, fcc, neighbours, rattle, strain_positions, strained};
use ndarray::{Array1, ArrayView1};
use rgmin::{
    Cell, CellEvaluation, CellFilter, CellFilterConfig, Control, Error, ExpParams, LbfgsStep,
    Method, PreconSpec, Solver, Status,
};

const STRAIN: [[f64; 3]; 3] = [
    [0.03, 0.01, -0.006],
    [0.01, -0.025, 0.008],
    [-0.006, 0.008, 0.02],
];
const R_CUT: f64 = 3.0;
const GTOL: f64 = 1.0e-5;

struct Crystal {
    positions: Array1<f64>,
    cell: Cell,
    orig: Cell,
    natoms: usize,
}

fn crystal() -> Crystal {
    let (mut positions, orig) = fcc(2, 3.61);
    rattle(&mut positions, 0.03, 11);
    let cell = strained(&orig, STRAIN);
    let positions = strain_positions(&positions, STRAIN);
    let natoms = positions.len() / 3;
    Crystal {
        positions,
        cell,
        orig,
        natoms,
    }
}

fn max_abs(v: ArrayView1<f64>) -> f64 {
    v.iter().fold(0.0_f64, |m, a| m.max(a.abs()))
}

fn session(step: LbfgsStep, dim: usize, maxiter: usize) -> Solver {
    let control = Control::default().with_maxiter(maxiter).with_gtol(GTOL);
    let mut solver = Solver::new(Method::lbfgs_with_memory(20), control, dim);
    solver.set_lbfgs_step(step).unwrap();
    solver
}

struct Run {
    report: rgmin::Report,
    positions: Array1<f64>,
    cell: Cell,
    evaluations: usize,
}

fn relax(
    c: &Crystal,
    config: CellFilterConfig,
    step: LbfgsStep,
    exp: Option<ExpParams>,
    maxiter: usize,
) -> (Run, Solver) {
    let filter = CellFilter::new(config, c.orig, c.natoms).unwrap();
    let mut solver = session(step, filter.dim(), maxiter);
    solver.set_cell_filter(filter.clone()).unwrap();
    let mut positions = c.positions.clone();
    let mut cell = c.cell;
    if let Some(params) = exp {
        let x = filter.pack(positions.view(), &cell).unwrap();
        let nb = neighbours(positions.view(), &cell, R_CUT);
        solver
            .set_precon(PreconSpec::Exp {
                positions: x.view(),
                neighbours: &nb,
                params: &params,
            })
            .unwrap();
    }
    let calls = AtomicUsize::new(0);
    let eval = |p: ArrayView1<f64>, cell: &Cell| -> CellEvaluation {
        calls.fetch_add(1, Ordering::Relaxed);
        evaluate(p, cell)
    };
    let mut report = None;
    for _ in 0..maxiter {
        let r = solver.step_cell(&eval, &mut positions, &mut cell).unwrap();
        let done = r.grad_norm < GTOL || r.status != Status::Iterating;
        report = Some(r);
        if done {
            break;
        }
    }
    let run = Run {
        report: report.unwrap(),
        positions,
        cell,
        evaluations: calls.load(Ordering::Relaxed),
    };
    (run, solver)
}

fn steps() -> [(&'static str, LbfgsStep); 2] {
    [
        ("capped", LbfgsStep::default()),
        ("wolfe", LbfgsStep::Wolfe),
    ]
}

#[test]
fn both_step_rules_relax_the_cell_to_zero_stress() {
    let c = crystal();
    for (name, step) in steps() {
        let (run, _) = relax(&c, CellFilterConfig::default(), step, None, 600);
        assert!(
            run.report.grad_norm < GTOL,
            "{name}: grad {} after {} steps",
            run.report.grad_norm,
            run.report.steps
        );
        let ev = evaluate(run.positions.view(), &run.cell);
        assert!(max_abs(ev.forces.view()) < 1.0e-4, "{name} forces");
        for row in ev.stress {
            for s in row {
                assert!(s.abs() < 1.0e-4, "{name}: stress {s}");
            }
        }
        eprintln!(
            "smoke {name}: {} steps, {} evaluations",
            run.report.steps, run.evaluations
        );
    }
}

#[test]
fn the_exponential_cell_block_relaxes_under_both_step_rules() {
    let c = crystal();
    for (name, step) in steps() {
        let params = ExpParams::default().with_r_cut(R_CUT);
        let (run, solver) = relax(&c, CellFilterConfig::default(), step, Some(params), 600);
        assert!(
            run.report.grad_norm < GTOL,
            "{name}: grad {} after {} steps",
            run.report.grad_norm,
            run.report.steps
        );
        let mu_c = solver.exp_mu_c().unwrap();
        assert!(mu_c >= 1.0, "{name}: mu_c {mu_c}");
        eprintln!(
            "smoke {name} + exp: {} steps, {} evaluations, mu_c {mu_c}",
            run.report.steps, run.evaluations
        );
    }
}

#[test]
fn a_given_mu_c_is_kept() {
    let c = crystal();
    let params = ExpParams::default().with_r_cut(R_CUT).with_mu_c(3.25);
    let (_, solver) = relax(
        &c,
        CellFilterConfig::default(),
        LbfgsStep::default(),
        Some(params),
        3,
    );
    assert_eq!(solver.exp_mu_c(), Some(3.25));
}

#[test]
fn external_pressure_sets_the_final_stress() {
    let c = crystal();
    let pressure = 0.002;
    let config = CellFilterConfig::default()
        .with_pressure(pressure)
        .with_hydrostatic(true);
    for (name, step) in steps() {
        let (run, _) = relax(&c, config.clone(), step, None, 600);
        assert!(run.report.grad_norm < GTOL, "{name}");
        let ev = evaluate(run.positions.view(), &run.cell);
        let trace = (ev.stress[0][0] + ev.stress[1][1] + ev.stress[2][2]) / 3.0;
        assert!(
            (trace + pressure).abs() < 1.0e-4,
            "{name}: mean stress {trace}"
        );
    }
}

#[test]
fn constant_volume_keeps_the_volume() {
    let c = crystal();
    let config = CellFilterConfig::default().with_constant_volume(true);
    let before = CellFilter::volume(&c.cell);
    for (name, step) in steps() {
        let (run, _) = relax(&c, config.clone(), step, None, 600);
        assert!(run.report.grad_norm < GTOL, "{name}");
        let after = CellFilter::volume(&run.cell);
        assert!((after - before).abs() <= 1.0e-9 * before, "{name}");
    }
}

#[test]
fn the_session_needs_a_filter_of_its_own_size() {
    let c = crystal();
    let filter = CellFilter::new(CellFilterConfig::default(), c.orig, c.natoms).unwrap();
    let mut plain = session(LbfgsStep::default(), 3 * c.natoms, 1);
    assert!(matches!(
        plain.set_cell_filter(filter.clone()),
        Err(Error::Dim { .. })
    ));
    let mut positions = c.positions.clone();
    let mut cell = c.cell;
    let eval = |p: ArrayView1<f64>, cell: &Cell| evaluate(p, cell);
    assert!(matches!(
        plain.step_cell(&eval, &mut positions, &mut cell),
        Err(Error::Cell { .. })
    ));
}

#[test]
fn the_filter_comes_before_the_preconditioner() {
    let c = crystal();
    let filter = CellFilter::new(CellFilterConfig::default(), c.orig, c.natoms).unwrap();
    let mut solver = session(LbfgsStep::default(), filter.dim(), 1);
    let x = filter.pack(c.positions.view(), &c.cell).unwrap();
    let nb = neighbours(c.positions.view(), &c.cell, R_CUT);
    let params = ExpParams::default().with_r_cut(R_CUT);
    solver
        .set_precon(PreconSpec::Exp {
            positions: x.view(),
            neighbours: &nb,
            params: &params,
        })
        .unwrap();
    assert!(matches!(
        solver.set_cell_filter(filter),
        Err(Error::Cell { .. })
    ));
}

#[test]
fn clearing_the_filter_returns_the_plain_layout() {
    let c = crystal();
    let filter = CellFilter::new(CellFilterConfig::default(), c.orig, c.natoms).unwrap();
    let mut solver = session(LbfgsStep::default(), filter.dim(), 1);
    solver.set_cell_filter(filter).unwrap();
    assert!(solver.cell_filter().is_some());
    solver.clear_cell_filter();
    assert!(solver.cell_filter().is_none());
    assert_eq!(solver.exp_mu_c(), None);
}

#[test]
fn mu_c_is_fixed_and_released_after_the_preconditioner_is_installed() {
    let c = crystal();
    let filter = CellFilter::new(CellFilterConfig::default(), c.orig, c.natoms).unwrap();
    let mut solver = session(LbfgsStep::default(), filter.dim(), 3);
    assert!(matches!(
        solver.set_exp_mu_c(Some(2.0)),
        Err(Error::Cell { .. })
    ));
    solver.set_cell_filter(filter.clone()).unwrap();
    assert!(matches!(
        solver.set_exp_mu_c(Some(2.0)),
        Err(Error::Cell { .. })
    ));
    let x = filter.pack(c.positions.view(), &c.cell).unwrap();
    let nb = neighbours(c.positions.view(), &c.cell, R_CUT);
    let params = ExpParams::default().with_r_cut(R_CUT);
    solver
        .set_precon(PreconSpec::Exp {
            positions: x.view(),
            neighbours: &nb,
            params: &params,
        })
        .unwrap();
    assert!(matches!(
        solver.set_exp_mu_c(Some(-1.0)),
        Err(Error::Cell { .. })
    ));
    solver.set_exp_mu_c(Some(2.5)).unwrap();
    assert_eq!(solver.exp_mu_c(), Some(2.5));
    let mut positions = c.positions.clone();
    let mut cell = c.cell;
    let eval = |p: ArrayView1<f64>, lattice: &Cell| evaluate(p, lattice);
    solver.step_cell(&eval, &mut positions, &mut cell).unwrap();
    assert_eq!(solver.exp_mu_c(), Some(2.5));
    solver.set_exp_mu_c(None).unwrap();
    assert_eq!(solver.exp_mu_c(), None);
    solver.step_cell(&eval, &mut positions, &mut cell).unwrap();
    assert!(solver.exp_mu_c().unwrap() >= 1.0);
}

#[test]
fn lindh_and_ff_have_no_cell_block() {
    let c = crystal();
    let filter = CellFilter::new(CellFilterConfig::default(), c.orig, c.natoms).unwrap();
    let mut solver = session(LbfgsStep::default(), filter.dim(), 1);
    solver.set_cell_filter(filter.clone()).unwrap();
    let x = filter.pack(c.positions.view(), &c.cell).unwrap();
    let z = vec![29; c.natoms];
    let params = rgmin::LindhParams::default();
    let result = solver.set_precon(PreconSpec::Lindh {
        positions: x.view(),
        z: &z,
        ghosts: &[],
        params: &params,
    });
    assert!(matches!(result, Err(Error::Cell { .. })));
}
