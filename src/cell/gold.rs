//! Replay of ASE 3.29.0 FIRE2 on `FrechetCellFilter`, from
//! `tests/data/ase_cell_gold.txt` (written by `tests/ase_cell_gold.py`).
//!
//! ASE evaluated its own Tersoff (bulk Si) and EMT (bulk Cu), so the
//! forces and stresses are ASE's. Three replays:
//!
//! 1. the filter gradient, extended coordinates and enthalpy of rgmin at
//!    every ASE evaluation;
//! 2. one rgmin FIRE2 step from the ASE state at every step, including
//!    the half-step-back evaluation;
//! 3. a free-running [`crate::Solver`] whose objective answers with the
//!    recorded ASE evaluations in call order, and which must ask for the
//!    points ASE asked for.
//!
//! Tolerances are `K` times the unit roundoff times a scale taken from
//! the data. `K` counts operations: [`GRADIENT_OPS`] bounds the products
//! and sums between a force or stress entry and a gradient entry plus the
//! backward error of SciPy's `logm` and `expm_frechet` and of the matrix
//! functions in `matrix.rs`, each a modest multiple of `u` times the
//! norm of its argument.

use std::sync::Mutex;

use ndarray::{Array1, ArrayView1};

use super::{Cell, CellEvaluation, CellFilter, CellFilterConfig};
use crate::fire::{
    FirePreparation, FirePreset, FireState, fire_displacement_into, fire_prepare_v2,
};
use crate::{Control, FireKind, Method, Solver};

const U: f64 = f64::EPSILON / 2.0;
/// Operation count of one gradient entry against ASE: three nested
/// 3x3 products and a 9-term Fréchet contraction (about 60 operations),
/// two matrix-function backward errors of about 100 u each on the
/// condition number of a near-identity deformation, rounded up.
const GRADIENT_OPS: f64 = 512.0;
/// Operation count of one coordinate against ASE: SciPy `logm` and
/// `expm` each carry a backward error of a few tens of `u` times the
/// norm of the cell block, and ASE re-derives its `x` through them at
/// every step.
const FUNCTION_OPS: f64 = 64.0;

fn gamma(m: f64) -> f64 {
    m * U / (1.0 - m * U)
}

struct Call {
    positions: Array1<f64>,
    cell: Cell,
    energy: f64,
    forces: Array1<f64>,
    stress: Cell,
}

struct Step {
    x: Array1<f64>,
    g: Array1<f64>,
    h: f64,
    calls: Vec<Call>,
    out_x: Array1<f64>,
    out_v: Array1<f64>,
    out_dt: f64,
    out_alpha: f64,
    out_n_pos: usize,
}

struct Case {
    name: String,
    natoms: usize,
    ecf: f64,
    config: CellFilterConfig,
    orig: Cell,
    steps: Vec<Step>,
}

#[derive(Clone, Copy)]
struct Start {
    dt: f64,
    dtmax: f64,
    n_min: usize,
}

fn numbers(tokens: &[&str]) -> Vec<f64> {
    tokens.iter().map(|t| t.parse().unwrap()).collect()
}

fn cell3(v: &[f64]) -> Cell {
    [[v[0], v[1], v[2]], [v[3], v[4], v[5]], [v[6], v[7], v[8]]]
}

fn load() -> (Start, Vec<Case>) {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/ase_cell_gold.txt");
    let text = std::fs::read_to_string(path).unwrap();
    let mut lines = text.lines().filter(|l| !l.starts_with('#')).peekable();
    let head: Vec<&str> = lines.next().unwrap().split_whitespace().collect();
    let start = Start {
        dt: head[1].parse().unwrap(),
        dtmax: head[3].parse().unwrap(),
        n_min: head[7].parse().unwrap(),
    };
    let mut cases = Vec::new();
    while let Some(line) = lines.next() {
        let t: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(t[0], "case");
        let natoms: usize = t[2].parse().unwrap();
        let nsteps: usize = t[3].parse().unwrap();
        let ecf: f64 = t[4].parse().unwrap();
        let mask_values = numbers(&t[8..17]);
        let mut mask = [[false; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                mask[i][j] = mask_values[3 * i + j] != 0.0;
            }
        }
        let config = CellFilterConfig::default()
            .with_exp_cell_factor(ecf)
            .with_pressure(t[5].parse().unwrap())
            .with_hydrostatic(t[6] == "1")
            .with_constant_volume(t[7] == "1")
            .with_mask(mask);
        let orig_line: Vec<&str> = lines.next().unwrap().split_whitespace().collect();
        let orig = cell3(&numbers(&orig_line[1..]));
        let _rollbacks = lines.next().unwrap();
        let mut steps = Vec::new();
        for _ in 0..nsteps {
            let head: Vec<&str> = lines.next().unwrap().split_whitespace().collect();
            let ncalls: usize = head[2].parse().unwrap();
            let row = |line: &str| -> Vec<f64> {
                let t: Vec<&str> = line.split_whitespace().collect();
                numbers(&t[1..])
            };
            let x = Array1::from(row(lines.next().unwrap()));
            let g = Array1::from(row(lines.next().unwrap()));
            let h = row(lines.next().unwrap())[0];
            let mut calls = Vec::new();
            for _ in 0..ncalls {
                let v = row(lines.next().unwrap());
                let n3 = 3 * natoms;
                calls.push(Call {
                    positions: Array1::from(v[..n3].to_vec()),
                    cell: cell3(&v[n3..n3 + 9]),
                    energy: v[n3 + 9],
                    forces: Array1::from(v[n3 + 10..2 * n3 + 10].to_vec()),
                    stress: cell3(&v[2 * n3 + 10..2 * n3 + 19]),
                });
            }
            let out = row(lines.next().unwrap());
            let dim = 3 * natoms + 9;
            steps.push(Step {
                x,
                g,
                h,
                calls,
                out_x: Array1::from(out[..dim].to_vec()),
                out_v: Array1::from(out[dim..2 * dim].to_vec()),
                out_dt: out[2 * dim],
                out_alpha: out[2 * dim + 1],
                out_n_pos: out[2 * dim + 2] as usize,
            });
        }
        cases.push(Case {
            name: t[1].to_owned(),
            natoms,
            ecf,
            config,
            orig,
            steps,
        });
    }
    (start, cases)
}

fn max_abs(v: ArrayView1<f64>) -> f64 {
    v.iter().fold(0.0_f64, |m, a| m.max(a.abs()))
}

/// ASE stores `ecf logm(N F)`, which is rgmin's cell block plus
/// `ecf ln N` on the diagonal.
fn ase_offset(case: &Case) -> Array1<f64> {
    let mut off = Array1::zeros(3 * case.natoms + 9);
    let base = 3 * case.natoms;
    for d in 0..3 {
        off[base + 4 * d] = case.ecf * (case.natoms as f64).ln();
    }
    off
}

fn filter(case: &Case) -> CellFilter {
    CellFilter::new(case.config.clone(), case.orig, case.natoms).unwrap()
}

fn label(case: &Case, k: usize) -> String {
    format!("{} step {k}", case.name)
}

#[test]
fn the_filter_matches_ase_at_every_evaluation() {
    let (_, cases) = load();
    let mut worst = 0.0_f64;
    for case in &cases {
        let filter = filter(case);
        let offset = ase_offset(case);
        for (k, step) in case.steps.iter().enumerate() {
            let call = &step.calls[0];
            let want_x = &step.x - &offset;
            let x = filter.pack(call.positions.view(), &call.cell).unwrap();
            let scale = max_abs(step.x.view());
            let tol = FUNCTION_OPS * U * scale;
            for i in 0..x.len() {
                assert!(
                    (x[i] - want_x[i]).abs() <= tol,
                    "{} x[{i}]: {} vs {}",
                    label(case, k),
                    x[i],
                    want_x[i]
                );
            }
            let (back, back_cell) = filter.unpack(want_x.view()).unwrap();
            for i in 0..back.len() {
                assert!(
                    (back[i] - call.positions[i]).abs() <= tol,
                    "{} pos[{i}]",
                    label(case, k)
                );
            }
            for i in 0..3 {
                for j in 0..3 {
                    assert!(
                        (back_cell[i][j] - call.cell[i][j]).abs() <= tol,
                        "{} cell[{i}][{j}]",
                        label(case, k)
                    );
                }
            }
            let g = filter
                .gradient(call.forces.view(), &call.stress, &call.cell)
                .unwrap();
            let scale = gradient_scale(case, call, &step.g);
            let tol = GRADIENT_OPS * U * scale;
            for i in 0..g.len() {
                let diff = (g[i] - step.g[i]).abs();
                worst = worst.max(diff / (U * scale));
                assert!(
                    diff <= tol,
                    "{} g[{i}]: {} vs {}",
                    label(case, k),
                    g[i],
                    step.g[i]
                );
            }
            let h = filter.enthalpy(call.energy, &call.cell);
            let tol = GRADIENT_OPS * U * (step.h.abs() + call.energy.abs());
            assert!((h - step.h).abs() <= tol, "{} enthalpy", label(case, k));
        }
    }
    eprintln!("gradient: worst difference {worst:.3} u scale");
}

/// Magnitude the entries of a gradient are formed from: the forces and
/// the virial `V (|stress| + p) / ecf` of the cell block.
fn gradient_scale(case: &Case, call: &Call, g: &Array1<f64>) -> f64 {
    let stress = call
        .stress
        .iter()
        .flatten()
        .fold(0.0_f64, |m, s| m.max(s.abs()));
    let virial = CellFilter::volume(&call.cell) * (stress + case.config.pressure.abs()) / case.ecf;
    max_abs(g.view())
        .max(max_abs(call.forces.view()))
        .max(virial)
}

fn fire_state(start: Start, dim: usize) -> FireState {
    let mut state = FireState::new(FireKind::V2, dim, start.dt);
    state.set_preset(FirePreset::Guenole2020);
    assert_eq!(state.dt_max, start.dtmax);
    assert_eq!(state.n_min, start.n_min);
    state
}

fn vel_tol(kick: f64) -> f64 {
    2.0 * gamma(14.0) * kick
}

#[test]
fn fire2_steps_from_the_ase_state_reproduce_ase() {
    let (start, cases) = load();
    let mut rollbacks = 0;
    for case in &cases {
        let filter = filter(case);
        let offset = ase_offset(case);
        let dim = filter.dim();
        let mut state = fire_state(start, dim);
        let mut worst = 0.0_f64;
        for (k, step) in case.steps.iter().enumerate() {
            let mut x = &step.x - &offset;
            let call = &step.calls[0];
            let g = filter
                .gradient(call.forces.view(), &call.stress, &call.cell)
                .unwrap();
            let mut force = -&g;
            if let FirePreparation::Rollback(c) = fire_prepare_v2(&mut state, &force) {
                rollbacks += 1;
                assert_eq!(step.calls.len(), 2, "{}", label(case, k));
                x += &c;
                let again = &step.calls[1];
                let packed = filter.pack(again.positions.view(), &again.cell).unwrap();
                let tol = FUNCTION_OPS * U * max_abs(step.x.view());
                for i in 0..dim {
                    assert!(
                        (packed[i] - x[i]).abs() <= tol,
                        "{} rollback point [{i}]",
                        label(case, k)
                    );
                }
                let g = filter
                    .gradient(again.forces.view(), &again.stress, &again.cell)
                    .unwrap();
                force = -&g;
            } else {
                assert_eq!(step.calls.len(), 1, "{}", label(case, k));
            }
            let kick = {
                let kicked = &state.vel + &(&force * state.dt);
                crate::vecops::nrm2(kicked.view())
            };
            let mut dx = Array1::zeros(dim);
            fire_displacement_into(&mut state, &force, &mut dx);
            assert_eq!(
                (state.dt, state.alpha, state.n_pos),
                (step.out_dt, step.out_alpha, step.out_n_pos),
                "{}",
                label(case, k)
            );
            let tol_v =
                vel_tol(kick) + state.dt * GRADIENT_OPS * U * gradient_scale(case, call, &step.g);
            for i in 0..dim {
                let tol_x = state.dt * tol_v
                    + 2.0 * gamma(2.0) * (step.out_x[i].abs() + dx[i].abs())
                    + FUNCTION_OPS * U * max_abs(step.out_x.view());
                let diff_v = (state.vel[i] - step.out_v[i]).abs();
                assert!(diff_v <= tol_v, "{} v[{i}]", label(case, k));
                let next = x[i] + dx[i] + offset[i];
                let diff_x = (next - step.out_x[i]).abs();
                worst = worst.max(diff_x / tol_x);
                assert!(
                    diff_x <= tol_x,
                    "{} x[{i}]: {next} vs {}",
                    label(case, k),
                    step.out_x[i]
                );
            }
            state.vel.assign(&step.out_v);
        }
        eprintln!(
            "{}: worst x difference {worst:.3} of its tolerance",
            case.name
        );
    }
    assert!(
        rollbacks >= cases.len(),
        "every case takes a half-step back"
    );
}

struct Replay<'a> {
    calls: Vec<&'a Call>,
    next: Mutex<usize>,
    worst: Mutex<f64>,
    scale: f64,
    ops: f64,
}

impl Replay<'_> {
    fn answer(&self, positions: ArrayView1<f64>, cell: &Cell) -> CellEvaluation {
        let mut next = self.next.lock().unwrap();
        let call = self.calls[*next];
        let mut worst = self.worst.lock().unwrap();
        let tol = self.ops * U * self.scale;
        for i in 0..positions.len() {
            let diff = (positions[i] - call.positions[i]).abs();
            *worst = worst.max(diff / (U * self.scale));
            assert!(diff <= tol, "call {} position[{i}]", *next);
        }
        for i in 0..3 {
            for j in 0..3 {
                let diff = (cell[i][j] - call.cell[i][j]).abs();
                assert!(diff <= tol, "call {} cell[{i}][{j}]", *next);
            }
        }
        *next += 1;
        CellEvaluation::new(call.energy, call.forces.clone(), call.stress)
    }
}

#[test]
fn a_free_running_session_asks_for_the_points_ase_asked_for() {
    let (start, cases) = load();
    for case in &cases {
        let filter = filter(case);
        let dim = filter.dim();
        let calls: Vec<&Call> = case.steps.iter().flat_map(|s| s.calls.iter()).collect();
        let scale = case
            .steps
            .iter()
            .fold(0.0_f64, |m, s| m.max(max_abs(s.x.view())));
        let steps = case.steps.len() - 1;
        let replay = Replay {
            calls,
            next: Mutex::new(0),
            worst: Mutex::new(0.0),
            scale,
            // Rounding differences per step feed the next step through
            // the force: one gradient budget per step taken.
            ops: GRADIENT_OPS * steps as f64,
        };
        let control = Control {
            maxiter: steps,
            gtol: 0.0,
            istep: start.dt,
            maxmove: None,
            ..Default::default()
        };
        let mut solver = Solver::new(Method::Fire { kind: FireKind::V2 }, control, dim);
        solver.set_fire_preset(FirePreset::Guenole2020).unwrap();
        solver.set_cell_filter(filter.clone()).unwrap();
        let first = &case.steps[0].calls[0];
        let mut positions = first.positions.clone();
        let mut cell = first.cell;
        let eval = |p: ArrayView1<f64>, c: &Cell| replay.answer(p, c);
        for _ in 0..steps {
            solver.step_cell(&eval, &mut positions, &mut cell).unwrap();
        }
        let used = *replay.next.lock().unwrap();
        assert_eq!(used, replay.calls.len(), "{}: evaluations", case.name);
        let last = &case.steps[steps];
        let want = &last.x - &ase_offset(case);
        let x = filter.pack(positions.view(), &cell).unwrap();
        let tol = replay.ops * U * scale;
        for i in 0..dim {
            assert!((x[i] - want[i]).abs() <= tol, "{} final x[{i}]", case.name);
        }
        eprintln!(
            "{}: {} evaluations, worst point difference {:.3} u scale",
            case.name,
            used,
            *replay.worst.lock().unwrap()
        );
    }
}
