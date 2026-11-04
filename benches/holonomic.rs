//! Constrained minima and their cost against GROMACS and ASE.
//!
//! `benches/holonomic/gromacs_em.py` runs `gmx_d` energy minimisation
//! with constraints and writes `benches/holonomic/data/gromacs_em.txt`:
//! the water, ethane and LJ13 systems of
//! `tests/data/ase_fix_bond_lengths.txt` (ASE `FixBondLengths`, written
//! by `tests/ase_fix_bond_lengths.py`) and 216 SPC/E waters kept rigid
//! by SETTLE. This bench evaluates the same energies, checks them
//! against GROMACS at its end points, and runs rgmin from the same
//! starts to the same `emtol` on the largest per-atom projected force,
//! with unit masses and with real ones.
//!
//! Every end point, from any program, is scored with the same energy,
//! the exact unit-mass projected force (what GROMACS reports as `Fmax`,
//! since it sets every mass to one in minimisation, but projected
//! exactly rather than by a truncated LINCS expansion), and the largest
//! relative bond residual `|d^2 - r^2| / (2 d^2)`. `E_min` is the lowest
//! energy among the end points on the set to its tolerance, so an end
//! point off the set can lie below it. rgmin works on
//! unwrapped coordinates and the energy takes the minimum image, so the
//! session needs no periodic cell; GROMACS end points are unwrapped
//! against the start.
//!
//! The toy systems take the ASE settings of the test data (initial step
//! and FIRE `dt` 0.1, largest atom move 0.2), so their rows compare with
//! ASE's. SPC/E takes the GROMACS `emstep` for both. On SPC/E the tight
//! `emtol` lies below what an energy test can resolve, and GROMACS stops
//! at machine precision; rgmin stops there at the smallest force GROMACS
//! reached, and at the GROMACS default `emtol`. A run ends after 50000
//! steps or force calls, the GROMACS `nsteps`. Force calls count every
//! oracle call. Every rgmin row is timed in `ROUNDS` rounds, on one
//! thread like `mdrun -nt 1`, and each round runs every row once in
//! table order, so the runs of one row interleave with all the others.
//! The ms columns give the median and the type-7 quartiles (Hyndman and
//! Fan 1996) of those times. GROMACS and ASE rows give the one time their
//! data file records, with no quartiles.
//! A second table runs steepest descent, Polak-Ribiere CG and L-BFGS
//! under the energy test from the toy starts without the set, so a
//! method that stalls there stalls for a reason the set does not add.
//! A third table gives every time of every rgmin row in round order, to
//! the nanosecond, so the quartiles can be recomputed.
//! `cargo bench --bench holonomic`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1};
use rgmin::{
    Accept, ConstraintSolver, Control, DistanceConstraints, FireKind, Method, Oracle, Solver,
    StopNorm,
};

/// `tests/ase_fix_bond_lengths.py`: FIRE `dt` and the largest step.
const ASE_DT: f64 = 0.1;
const ASE_MAXSTEP: f64 = 0.2;
/// The GROMACS `nsteps` of the reference runs, a budget only.
const BUDGET: usize = 50_000;
/// Timed rounds over every rgmin row, at least six for the quartiles.
const ROUNDS: usize = 7;

struct Counted<O> {
    inner: O,
    calls: AtomicUsize,
}

impl<O: Objective<f64>> Objective<f64> for Counted<O> {
    fn dim(&self) -> usize {
        self.inner.dim()
    }
    fn bounds(&self) -> &Bounds<f64> {
        self.inner.bounds()
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.inner.eval(x)
    }
}

impl<O: DifferentiableObjective<f64>> Gradient<f64> for Counted<O> {
    fn dim(&self) -> usize {
        Objective::dim(&self.inner)
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.inner.grad(x)
    }
}

impl<O: DifferentiableObjective<f64>> DifferentiableObjective<f64> for Counted<O> {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.inner.value_and_gradient(x)
    }
}

/// `1 / (4 pi epsilon_0)` in kJ mol^-1 nm e^-2, from the CODATA 2018
/// values of `api/legacy/include/gromacs/math/units.h`.
fn one_4pi_eps0() -> f64 {
    let e = 1.602176634e-19_f64;
    let eps0 = 8.8541878128e-12 * 1e-9 * 1e3 / (e * e * 6.02214076e23);
    1.0 / (4.0 * std::f64::consts::PI * eps0)
}

/// The GROMACS force switch of `r^-p` from `rsw` to `rc`
/// (`force_switch_constants` in `src/gromacs/mdtypes/interaction_const.cpp`
/// and the reference kernel `nbnxm/kernels_reference/kernel_ref_inner.h`):
/// with `s = max(r - rsw, 0)` the potential is
/// `r^-p - p (c2 s^3 / 3 + c3 s^4 / 4) + cpot`, zero with its force at `rc`.
#[derive(Clone, Copy)]
struct Switch {
    p: i32,
    rsw: f64,
    c2: f64,
    c3: f64,
    cpot: f64,
}

impl Switch {
    fn new(p: i32, rsw: f64, rc: f64) -> Self {
        let q = f64::from(p);
        let w = rc - rsw;
        let c2 = ((q + 1.0) * rsw - (q + 4.0) * rc) / (rc.powi(p + 2) * w * w);
        let c3 = -((q + 1.0) * rsw - (q + 3.0) * rc) / (rc.powi(p + 2) * w.powi(3));
        let cpot = -rc.powi(-p) + q * c2 / 3.0 * w.powi(3) + q * c3 / 4.0 * w.powi(4);
        Self {
            p,
            rsw,
            c2,
            c3,
            cpot,
        }
    }

    /// `(V, dV/dr)`.
    fn eval(&self, r: f64) -> (f64, f64) {
        let q = f64::from(self.p);
        let s = (r - self.rsw).max(0.0);
        let rp = r.powi(-self.p);
        let v = rp - q * (self.c2 / 3.0 + self.c3 / 4.0 * s) * s.powi(3) + self.cpot;
        let dv = -q * (rp / r + (self.c2 + self.c3 * s) * s * s);
        (v, dv)
    }
}

enum Kind {
    /// `C6 = C12 = 4` on every pair: epsilon = sigma = 1.
    Lj,
    /// `(i, j, r0)` harmonic bonds of unit stiffness.
    Springs(Vec<(usize, usize, f64)>),
    /// SPC/E with reaction field at `epsilon_rf = 0`.
    Spce,
}

/// SPC/E of `oplsaa.ff/spce.itp`, as in `gromacs_em.py`.
const SPCE_Q: [f64; 3] = [-0.8476, 0.4238, 0.4238];
const SPCE_SIGMA: f64 = 3.16557e-01;
const SPCE_EPS: f64 = 6.50194e-01;

/// The energy of one system as GROMACS evaluates it.
struct Model {
    cell: f64,
    rc: f64,
    kind: Kind,
    rep: Switch,
    disp: Switch,
}

impl Model {
    /// The minimum image by subtraction: `f64::round` is a libm call on
    /// the baseline target, and coordinates stay within a few cells.
    fn image(&self, x: &[f64], i: usize, j: usize) -> [f64; 3] {
        let half = 0.5 * self.cell;
        [0, 1, 2].map(|c| {
            let mut d = x[3 * i + c] - x[3 * j + c];
            while d > half {
                d -= self.cell;
            }
            while d < -half {
                d += self.cell;
            }
            d
        })
    }

    fn energy(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let owned;
        let x = match x.as_slice() {
            Some(s) => s,
            None => {
                owned = x.to_vec();
                &owned
            }
        };
        let n = x.len() / 3;
        let rc2 = self.rc * self.rc;
        let mut e = 0.0;
        let mut g = vec![0.0; x.len()];
        let push = |g: &mut [f64], i: usize, j: usize, d: [f64; 3], r: f64, dv: f64| {
            for c in 0..3 {
                g[3 * i + c] += dv * d[c] / r;
                g[3 * j + c] -= dv * d[c] / r;
            }
        };
        let lj = |r: f64, c6: f64, c12: f64| {
            let (v12, d12) = self.rep.eval(r);
            let (v6, d6) = self.disp.eval(r);
            (c12 * v12 - c6 * v6, c12 * d12 - c6 * d6)
        };
        match &self.kind {
            Kind::Springs(springs) => {
                for &(i, j, r0) in springs {
                    let d = self.image(x, i, j);
                    let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                    e += 0.5 * (r - r0) * (r - r0);
                    push(&mut g, i, j, d, r, r - r0);
                }
            }
            Kind::Lj => {
                for i in 0..n {
                    for j in i + 1..n {
                        let d = self.image(x, i, j);
                        let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                        if r2 < rc2 {
                            let r = r2.sqrt();
                            let (v, dv) = lj(r, 4.0, 4.0);
                            e += v;
                            push(&mut g, i, j, d, r, dv);
                        }
                    }
                }
            }
            Kind::Spce => {
                // `rf_util.cpp` calc_rffac at epsilon_rf = 0.
                let f = one_4pi_eps0();
                let krf = 1.0 / (2.0 * self.rc.powi(3));
                let crf = 1.0 / self.rc + krf * rc2;
                let c6 = 4.0 * SPCE_EPS * SPCE_SIGMA.powi(6);
                let c12 = 4.0 * SPCE_EPS * SPCE_SIGMA.powi(12);
                // Every atom's self-exclusion, `kernel_ref_outer.h`.
                e -= 0.5 * f * crf * (n / 3) as f64 * SPCE_Q.iter().map(|q| q * q).sum::<f64>();
                for i in 0..n {
                    for j in i + 1..n {
                        let d = self.image(x, i, j);
                        let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                        if r2 >= rc2 {
                            continue;
                        }
                        let r = r2.sqrt();
                        let qq = f * SPCE_Q[i % 3] * SPCE_Q[j % 3];
                        if i / 3 == j / 3 {
                            // An excluded pair keeps the reaction-field terms.
                            e += qq * (krf * r2 - crf);
                            push(&mut g, i, j, d, r, qq * 2.0 * krf * r);
                            continue;
                        }
                        let mut v = qq * (1.0 / r + krf * r2 - crf);
                        let mut dv = qq * (2.0 * krf * r - 1.0 / r2);
                        if i % 3 == 0 && j % 3 == 0 {
                            let (u, du) = lj(r, c6, c12);
                            v += u;
                            dv += du;
                        }
                        e += v;
                        push(&mut g, i, j, d, r, dv);
                    }
                }
            }
        }
        (e, Array1::from(g))
    }
}

struct Reference {
    program: &'static str,
    label: String,
    emtol: f64,
    outcome: String,
    steps: usize,
    calls: usize,
    energy: Option<f64>,
    fmax: f64,
    seconds: f64,
    end: Array1<f64>,
}

struct System {
    name: String,
    model: Model,
    start: Array1<f64>,
    pairs: Vec<(usize, usize)>,
    lengths: Option<Vec<f64>>,
    masses: Option<Vec<f64>>,
    references: Vec<Reference>,
    /// `(label, emtol, message)` of the GROMACS runs that stopped.
    refused: Vec<(String, f64, String)>,
}

impl System {
    fn unit_set(&self) -> DistanceConstraints {
        match &self.lengths {
            Some(l) => DistanceConstraints::new(self.pairs.clone(), l.clone()).unwrap(),
            None => DistanceConstraints::at(self.pairs.clone(), self.start.view()).unwrap(),
        }
    }

    /// `end` with every coordinate within half a cell of the start.
    fn unwrap(&self, end: &Array1<f64>) -> Array1<f64> {
        let l = self.model.cell;
        Array1::from_iter(self.start.iter().zip(end).map(|(s, e)| {
            let d = e - s;
            s + d - l * (d / l).round()
        }))
    }
}

fn floats(w: &[&str]) -> Vec<f64> {
    w.iter().map(|v| v.parse().unwrap()).collect()
}

/// `benches/holonomic/data/gromacs_em.txt`: `(emstep, systems)`.
fn gromacs() -> (f64, Vec<System>) {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/benches/holonomic/data/gromacs_em.txt"
    );
    let text = std::fs::read_to_string(path).expect("gromacs_em.txt");
    let mut emstep = f64::NAN;
    let mut systems: Vec<System> = Vec::new();
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let w: Vec<&str> = line.split_whitespace().collect();
        match (w[0], systems.last_mut()) {
            ("emstep", _) => emstep = w[1].parse().unwrap(),
            ("system", _) => {
                let [cell, rc, rsw] = [3, 4, 5].map(|k| w[k].parse::<f64>().unwrap());
                let kind = match w[6] {
                    "lj" => Kind::Lj,
                    "springs" => Kind::Springs(Vec::new()),
                    "spce" => Kind::Spce,
                    other => panic!("unknown kind {other}"),
                };
                systems.push(System {
                    name: w[1].to_string(),
                    model: Model {
                        cell,
                        rc,
                        kind,
                        rep: Switch::new(12, rsw, rc),
                        disp: Switch::new(6, rsw, rc),
                    },
                    start: Array1::zeros(0),
                    pairs: Vec::new(),
                    lengths: None,
                    masses: None,
                    references: Vec::new(),
                    refused: Vec::new(),
                });
            }
            ("start", Some(s)) => s.start = Array1::from(floats(&w[1..])),
            ("pairs", Some(s)) => {
                let ids: Vec<usize> = w[1..].iter().map(|v| v.parse().unwrap()).collect();
                s.pairs = ids.chunks(2).map(|p| (p[0], p[1])).collect();
            }
            ("settles", Some(s)) => {
                let [doh, dhh] = [1, 2].map(|k| w[k].parse::<f64>().unwrap());
                let waters = s.start.len() / 9;
                s.pairs = (0..waters)
                    .flat_map(|m| {
                        [
                            (3 * m, 3 * m + 1),
                            (3 * m, 3 * m + 2),
                            (3 * m + 1, 3 * m + 2),
                        ]
                    })
                    .collect();
                s.lengths = Some((0..waters).flat_map(|_| [doh, doh, dhh]).collect());
            }
            ("springs", Some(s)) => {
                let t = &w[1..];
                let springs = (0..t.len() / 3)
                    .map(|k| {
                        let id = |o: usize| t[3 * k + o].parse::<usize>().unwrap();
                        (id(0), id(1), t[3 * k + 2].parse().unwrap())
                    })
                    .collect();
                s.model.kind = Kind::Springs(springs);
            }
            ("masses", Some(s)) => s.masses = Some(floats(&w[1..])),
            ("run", Some(s)) => s.references.push(Reference {
                program: "gmx_d",
                label: w[1].to_string(),
                emtol: w[2].parse().unwrap(),
                outcome: w[3].to_string(),
                steps: w[4].parse().unwrap(),
                calls: w[5].parse().unwrap(),
                energy: Some(w[6].parse().unwrap()),
                fmax: w[7].parse().unwrap(),
                seconds: w[8].parse().unwrap(),
                end: Array1::zeros(0),
            }),
            ("end", Some(s)) => {
                let end = s.unwrap(&Array1::from(floats(&w[1..])));
                s.references.last_mut().unwrap().end = end;
            }
            ("failed", Some(s)) => {
                let emtol = w[2].parse().unwrap();
                s.refused.push((w[1].to_string(), emtol, w[4..].join(" ")));
            }
            _ => panic!("unexpected line {line}"),
        }
    }
    (emstep, systems)
}

/// The runs of `tests/data/ase_fix_bond_lengths.txt`, by system name.
fn ase() -> Vec<(String, Vec<Reference>)> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/ase_fix_bond_lengths.txt"
    );
    let text = std::fs::read_to_string(path).expect("ase_fix_bond_lengths.txt");
    let mut out: Vec<(String, Vec<Reference>)> = Vec::new();
    let mut fmax = f64::NAN;
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let w: Vec<&str> = line.split_whitespace().collect();
        match (w[0], out.last_mut()) {
            ("fmax", _) => fmax = w[1].parse().unwrap(),
            ("system", _) => out.push((w[1].to_string(), Vec::new())),
            ("run", Some((_, runs))) => runs.push(Reference {
                program: "ase",
                label: w[1].to_string(),
                emtol: fmax,
                outcome: if w[4] == "nan" {
                    "rattle-failed"
                } else {
                    "fmax"
                }
                .to_string(),
                steps: w[2].parse().unwrap(),
                calls: w[3].parse().unwrap(),
                energy: None,
                fmax: w[5].parse().unwrap(),
                seconds: w[9].parse().unwrap(),
                end: Array1::zeros(0),
            }),
            ("end", Some((_, runs))) => {
                runs.last_mut().unwrap().end = Array1::from(floats(&w[1..]))
            }
            _ => {}
        }
    }
    out
}

fn max_atom(g: &Array1<f64>) -> f64 {
    g.exact_chunks(3)
        .into_iter()
        .map(|f| f.dot(&f).sqrt())
        .fold(0.0, f64::max)
}

/// Energy, exact unit-mass projected `Fmax`, bond residual, and whether
/// `x` is on the set to its tolerance.
fn score(sys: &System, unit: &DistanceConstraints, x: &Array1<f64>) -> (f64, f64, f64, bool) {
    let (e, mut g) = sys.model.energy(x.view());
    unit.project_gradient(x.view(), &mut g).unwrap();
    let on = unit.excess(x.view()) <= 1.0;
    (e, max_atom(&g), unit.residual(x.view()), on)
}

#[derive(Clone, Copy)]
struct Arm {
    name: &'static str,
    method: fn() -> Method,
    accept: Accept,
    solver: ConstraintSolver,
    settle: bool,
}

struct Run {
    outcome: String,
    steps: usize,
    calls: usize,
    x: Array1<f64>,
}

fn solve(
    sys: &System,
    set: Option<&DistanceConstraints>,
    arm: Arm,
    control: &Control,
    cap: f64,
) -> Run {
    let obj = Counted {
        inner: Oracle::unbounded(sys.start.len(), |x: ArrayView1<f64>| sys.model.energy(x)),
        calls: AtomicUsize::new(0),
    };
    let mut solver = Solver::new((arm.method)(), control.clone(), sys.start.len());
    solver.set_accept(arm.accept);
    solver.set_stop_norm(StopNorm::MaxAtom);
    solver.set_atom_maxmove(cap);
    if let Some(set) = set {
        let set = set.clone().with_solver(arm.solver).with_settle(arm.settle);
        solver.set_distance_constraints(set).unwrap();
    }
    let mut x = sys.start.clone();
    let mut outcome = "budget".to_string();
    let mut steps = 0;
    while steps < BUDGET && obj.calls.load(Ordering::Relaxed) < BUDGET {
        steps += 1;
        match solver.step(&obj, &mut x) {
            Ok(rep) if rep.grad_norm < control.gtol => {
                outcome = "fmax".into();
                break;
            }
            Ok(_) => {}
            Err(e) => {
                let text = e.to_string();
                outcome = text.split(':').next().unwrap_or(&text).replace(' ', "-");
                break;
            }
        }
    }
    Run {
        outcome,
        steps,
        calls: obj.calls.load(Ordering::Relaxed),
        x,
    }
}

/// One table row.
struct Line {
    program: String,
    label: String,
    stop: f64,
    masses: String,
    outcome: String,
    steps: usize,
    calls: usize,
    energy: f64,
    reported: f64,
    exact: f64,
    residual: f64,
    on_set: bool,
    ms: f64,
    /// Type-7 first and third quartiles of the timed rounds, in ms.
    quartiles: Option<(f64, f64)>,
}

impl Line {
    fn print(&self, sys: &str, e_min: f64) {
        let (q1, q3) = self.quartiles.map_or(("-".into(), "-".into()), |(a, b)| {
            (format!("{a:.1}"), format!("{b:.1}"))
        });
        println!(
            "| {sys} | {} | {} | {:.2e} | {} | {} | {} | {} | {:.2e} | {:.2e} | {:.2e} | {:.1e} | {:.1} | {q1} | {q3} |",
            self.program,
            self.label,
            self.stop,
            self.masses,
            self.outcome,
            self.steps,
            self.calls,
            self.energy - e_min,
            self.reported,
            self.exact,
            self.residual,
            self.ms
        );
    }
}

fn arms(spce: bool) -> Vec<Arm> {
    let auto = ConstraintSolver::Auto;
    let arm = |name, method: fn() -> Method, accept| Arm {
        name,
        method,
        accept,
        solver: auto,
        settle: true,
    };
    let mut out = vec![
        arm("lbfgs", Method::lbfgs, Accept::Energy),
        arm("lbfgs", Method::lbfgs, Accept::Nonmonotone),
        arm("lbfgs", Method::lbfgs, Accept::None),
        arm("cg-pr", Method::polak_ribiere, Accept::Energy),
        arm("steepest", || Method::Steepest, Accept::Energy),
        arm("bfgs", || Method::Bfgs, Accept::Energy),
        arm("fire", || Method::Fire { kind: FireKind::V1 }, Accept::None),
        arm(
            "fire2",
            || Method::Fire { kind: FireKind::V2 },
            Accept::None,
        ),
        arm("newton-cg", || Method::NewtonCg, Accept::None),
    ];
    if spce {
        out.push(Arm {
            name: "lbfgs shake, no settle",
            settle: false,
            solver: ConstraintSolver::Shake,
            ..arm("", Method::lbfgs, Accept::Energy)
        });
    } else {
        for (name, solver) in [
            ("lbfgs shake", ConstraintSolver::Shake),
            ("lbfgs lincs", ConstraintSolver::Lincs),
        ] {
            out.push(Arm {
                name,
                solver,
                ..arm("", Method::lbfgs, Accept::Energy)
            });
        }
    }
    out
}

fn hardware() -> String {
    let info = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let model = info
        .lines()
        .find(|l| l.starts_with("model name"))
        .and_then(|l| l.split(':').nth(1))
        .map_or("unknown CPU".to_string(), |m| m.trim().to_string());
    let cpus = info.lines().filter(|l| l.starts_with("processor")).count();
    format!("{model}, {cpus} logical CPUs, one thread")
}

/// Hyndman and Fan (1996) type 7 quantile of sorted `x`.
fn quantile(x: &[f64], p: f64) -> f64 {
    let h = (x.len() - 1) as f64 * p;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(x.len() - 1);
    x[lo] + (h - lo as f64) * (x[hi] - x[lo])
}

/// One timed rgmin row: what it runs, and where its line sits.
struct Cell {
    sys: usize,
    row: usize,
    set: DistanceConstraints,
    arm: Arm,
    control: Control,
    cap: f64,
    steps: usize,
    calls: usize,
    times: Vec<f64>,
}

fn main() {
    let (emstep, systems) = gromacs();
    let ase = ase();
    println!("{}", hardware());
    println!();
    println!(
        "| system | program | method | stop | masses | outcome | steps | force calls | E - E_min | reported Fmax | exact Fmax | bond residual | ms | ms Q1 | ms Q3 |"
    );
    println!("|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    let mut controls = Vec::new();
    let mut tables = Vec::new();
    let mut cells = Vec::new();
    for (index, sys) in systems.iter().enumerate() {
        let unit = sys.unit_set();
        let spce = matches!(sys.model.kind, Kind::Spce);
        let (istep, cap) = if spce {
            (emstep, emstep)
        } else {
            (ASE_DT, ASE_MAXSTEP)
        };
        let (e0, f0, r0, _) = score(sys, &unit, &sys.start);
        let t0 = Instant::now();
        let calls = (ROUNDS * 10) as u32;
        for _ in 0..calls {
            std::hint::black_box(sys.model.energy(sys.start.view()));
        }
        let call_ms = t0.elapsed().as_secs_f64() * 1e3 / f64::from(calls);
        eprintln!("{}: {call_ms:.3} ms per energy and force call", sys.name);
        let mut rows = vec![Line {
            program: "start".into(),
            label: "-".into(),
            stop: f64::NAN,
            masses: "-".into(),
            outcome: "-".into(),
            steps: 0,
            calls: 0,
            energy: e0,
            reported: f0,
            exact: f0,
            residual: r0,
            on_set: false,
            ms: 0.0,
            quartiles: None,
        }];
        let gmx = sys.references.iter();
        let theirs = ase.iter().filter(|(name, _)| *name == sys.name);
        for r in gmx.chain(theirs.flat_map(|(_, runs)| runs)) {
            let (e, fmax, res, on_set) = score(sys, &unit, &r.end);
            if let Some(reported) = r.energy {
                let scale = reported.abs().max(1.0);
                assert!(
                    (e - reported).abs() <= 1e-6 * scale,
                    "{} {}: energy {e} at the GROMACS end, GROMACS {reported}",
                    sys.name,
                    r.label
                );
            }
            let (label, masses) = match r.label.strip_suffix("-mass") {
                Some(l) => (l.to_string(), "real"),
                None => (r.label.clone(), "unit"),
            };
            rows.push(Line {
                program: r.program.into(),
                label,
                stop: r.emtol,
                masses: masses.into(),
                outcome: r.outcome.clone(),
                steps: r.steps,
                calls: r.calls,
                energy: e,
                reported: r.fmax,
                exact: fmax,
                residual: res,
                on_set,
                ms: r.seconds * 1e3,
                quartiles: None,
            });
        }
        let mut stops: Vec<f64> = sys.references.iter().map(|r| r.emtol).collect();
        stops.sort_by(f64::total_cmp);
        stops.dedup();
        if spce {
            // The smallest force GROMACS reaches before machine precision.
            let floor = sys
                .references
                .iter()
                .filter(|r| r.emtol == stops[0])
                .map(|r| r.fmax)
                .fold(f64::INFINITY, f64::min);
            stops[0] = floor;
        }
        if !spce {
            let control = Control {
                maxiter: BUDGET,
                gtol: stops[0],
                istep,
                maxmove: None,
            };
            let free = ["steepest", "cg-pr", "lbfgs"];
            for arm in arms(false)
                .into_iter()
                .filter(|a| free.contains(&a.name) && a.accept == Accept::Energy)
            {
                let run = solve(sys, None, arm, &control, cap);
                let fmax = max_atom(&sys.model.energy(run.x.view()).1);
                controls.push(format!(
                    "| {} | {} energy | {:.2e} | {} | {} | {} | {fmax:.2e} |",
                    sys.name, arm.name, stops[0], run.outcome, run.steps, run.calls
                ));
                eprintln!("{} {} without the set: {}", sys.name, arm.name, run.outcome);
            }
        }
        let mut sets = vec![("unit", unit.clone())];
        if let Some(m) = &sys.masses {
            sets.push(("real", unit.clone().with_masses(m.clone()).unwrap()));
        }
        for &stop in &stops {
            let control = Control {
                maxiter: BUDGET,
                gtol: stop,
                istep,
                maxmove: None,
            };
            for (masses, set) in &sets {
                for arm in arms(spce) {
                    // The first round keeps the end point for the table.
                    let t0 = Instant::now();
                    let run = solve(sys, Some(set), arm, &control, cap);
                    let seconds = t0.elapsed().as_secs_f64();
                    let (e, fmax, res, on_set) = score(sys, &unit, &run.x);
                    let reported = {
                        let mut g = sys.model.energy(run.x.view()).1;
                        set.project_gradient(run.x.view(), &mut g).unwrap();
                        max_atom(&g)
                    };
                    cells.push(Cell {
                        sys: index,
                        row: rows.len(),
                        set: set.clone(),
                        arm,
                        control: control.clone(),
                        cap,
                        steps: run.steps,
                        calls: run.calls,
                        times: vec![seconds],
                    });
                    rows.push(Line {
                        program: "rgmin".into(),
                        label: format!("{} {:?}", arm.name, arm.accept).to_lowercase(),
                        stop,
                        masses: (*masses).into(),
                        outcome: run.outcome,
                        steps: run.steps,
                        calls: run.calls,
                        energy: e,
                        reported,
                        exact: fmax,
                        residual: res,
                        on_set,
                        ms: 0.0,
                        quartiles: None,
                    });
                    let last = rows.last().unwrap();
                    eprintln!(
                        "{} {} {stop:.2e} {masses}: {} {} steps, {seconds:.1} s",
                        sys.name, last.label, last.outcome, last.steps
                    );
                }
            }
        }
        tables.push(rows);
    }
    eprintln!("round 1 of {ROUNDS} done");
    for round in 2..=ROUNDS {
        for cell in &mut cells {
            let sys = &systems[cell.sys];
            let t0 = Instant::now();
            let run = solve(sys, Some(&cell.set), cell.arm, &cell.control, cell.cap);
            cell.times.push(t0.elapsed().as_secs_f64());
            if (run.steps, run.calls) != (cell.steps, cell.calls) {
                eprintln!(
                    "{} {}: round {round} took {} steps and {} calls, round 1 {} and {}",
                    sys.name, cell.arm.name, run.steps, run.calls, cell.steps, cell.calls
                );
            }
        }
        eprintln!("round {round} of {ROUNDS} done");
    }
    let mut samples = Vec::new();
    for cell in &cells {
        let line = &tables[cell.sys][cell.row];
        let times: Vec<String> = cell
            .times
            .iter()
            .map(|t| format!("{:.6}", t * 1e3))
            .collect();
        samples.push(format!(
            "| {} | {} | {:.2e} | {} | {} |",
            systems[cell.sys].name,
            line.label,
            line.stop,
            line.masses,
            times.join(" | ")
        ));
    }
    for cell in &mut cells {
        cell.times.sort_by(f64::total_cmp);
        let ms = |p| quantile(&cell.times, p) * 1e3;
        let line = &mut tables[cell.sys][cell.row];
        line.ms = ms(0.5);
        line.quartiles = Some((ms(0.25), ms(0.75)));
    }
    for (sys, rows) in systems.iter().zip(&tables) {
        let e_min = rows
            .iter()
            .filter(|r| r.on_set)
            .map(|r| r.energy)
            .fold(f64::INFINITY, f64::min);
        for r in rows {
            r.print(&sys.name, e_min);
        }
        for (label, emtol, why) in &sys.refused {
            println!(
                "| {} | gmx_d | {label} | {emtol:.2e} | unit | stopped: {why} | | | | | | | | | |",
                sys.name
            );
        }
    }
    println!();
    println!("The same starts and stops without the set:");
    println!();
    println!("| system | method | stop | outcome | steps | force calls | Fmax |");
    println!("|---|---|---:|---|---:|---:|---:|");
    for line in &controls {
        println!("{line}");
    }
    println!();
    println!("Round times of every rgmin row, in ms:");
    println!();
    let rounds: Vec<String> = (1..=ROUNDS).map(|r| format!("round {r}")).collect();
    println!(
        "| system | method | stop | masses | {} |",
        rounds.join(" | ")
    );
    println!("|---|---|---:|---|{}", "---:|".repeat(ROUNDS));
    for line in &samples {
        println!("{line}");
    }
}
