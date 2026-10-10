//! rgmin on real rgpot potentials.
//!
//! The chain is the one an atomistic host drives: an rgpot C++ / Fortran
//! potential behind the rgpot-core callback, wrapped by
//! `rgpot_potential_new_eindir` into an eindir objective, reduced to the
//! free degrees of freedom, and handed to an `rgmin::Solver` session
//! (or the persistent `rgmin::Lbfgs`). Every force call is counted and
//! timed inside the rgpot callback, so the solver's own cost is the wall
//! time minus the potential time.
//!
//! One CSV row per run:
//! `system,method,seed,converged,iters,calls,energy,fmax,wall_s,pot_s,overhead_us_per_iter`.

use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use dlpk::sys::{
    DLDataType, DLDataTypeCode, DLDevice, DLDeviceType, DLManagedTensorVersioned, DLPackVersion,
    DLTensor,
};
use eindir_core::ffi::{eindir_objective_eval, eindir_objective_grad, eindir_status_t};
use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1};
use rgmin::{
    Accept, Conjugacy, Control, ExpParams, FireKind, Lbfgs, LineSearch, Method, Neighbours,
    Restart, Solver,
};
use rgpot_core::eindir::{
    rgpot_potential_free_eindir, rgpot_potential_new_eindir, rgpot_potential_t,
};
use rgpot_core::status::rgpot_status_t;
use rgpot_core::types::{rgpot_force_input_t, rgpot_force_out_t};

unsafe extern "C" {
    fn bench_force(
        kind: i32,
        natoms: usize,
        pos: *const f64,
        atmnrs: *const i32,
        cell: *const f64,
        forces: *mut f64,
        energy: *mut f64,
    ) -> i32;
}

/// Callback context: which kernel, a force buffer, and the counters.
struct Kernel {
    kind: i32,
    forces: Vec<f64>,
    calls: AtomicUsize,
    nanos: AtomicU64,
}

unsafe extern "C" fn force_callback(
    user: *mut c_void,
    input: *const rgpot_force_input_t,
    output: *mut rgpot_force_out_t,
) -> rgpot_status_t {
    let k = unsafe { &mut *(user as *mut Kernel) };
    let input = unsafe { &*input };
    let pos = unsafe { &(*input.positions).dl_tensor };
    let natoms = unsafe { *(pos.shape as *const i64) } as usize;
    let z = unsafe { &(*input.atomic_numbers).dl_tensor };
    let cell = unsafe { &(*input.box_matrix).dl_tensor };
    let mut energy = 0.0;
    let t0 = Instant::now();
    let st = unsafe {
        bench_force(
            k.kind,
            natoms,
            pos.data as *const f64,
            z.data as *const i32,
            cell.data as *const f64,
            k.forces.as_mut_ptr(),
            &mut energy,
        )
    };
    k.nanos
        .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    k.calls.fetch_add(1, Ordering::Relaxed);
    if st != 0 {
        return rgpot_status_t::RGPOT_INTERNAL_ERROR;
    }
    let output = unsafe { &mut *output };
    output.energy = energy;
    output.forces = unsafe {
        rgpot_core::tensor::rgpot_tensor_cpu_f64_2d(k.forces.as_mut_ptr(), natoms as i64, 3)
    };
    rgpot_status_t::RGPOT_SUCCESS
}

/// Borrowed rank-1 CPU f64 DLPack view over `len` doubles at `data`.
struct Shell {
    managed: Box<DLManagedTensorVersioned>,
    _shape: Box<[i64; 1]>,
    _strides: Box<[i64; 1]>,
}

impl Shell {
    fn new(data: *mut f64, len: usize) -> Self {
        let mut shape = Box::new([len as i64]);
        let mut strides = Box::new([1i64]);
        let managed = Box::new(DLManagedTensorVersioned {
            version: DLPackVersion { major: 1, minor: 0 },
            manager_ctx: std::ptr::null_mut(),
            deleter: None,
            flags: 0,
            dl_tensor: DLTensor {
                data: data.cast(),
                device: DLDevice {
                    device_type: DLDeviceType::kDLCPU,
                    device_id: 0,
                },
                ndim: 1,
                dtype: DLDataType {
                    code: DLDataTypeCode::kDLFloat,
                    bits: 64,
                    lanes: 1,
                },
                shape: shape.as_mut_ptr(),
                strides: strides.as_mut_ptr(),
                byte_offset: 0,
            },
        });
        Self {
            managed,
            _shape: shape,
            _strides: strides,
        }
    }
}

/// The rgpot eindir objective restricted to the free coordinates.
struct FreeObjective {
    pot: *mut rgpot_potential_t,
    kernel: *mut Kernel,
    /// Full 3N coordinate and gradient buffers (frozen atoms keep their
    /// reference positions).
    full: Mutex<(Vec<f64>, Vec<f64>)>,
    free: Vec<usize>,
    bounds: Bounds<f64>,
    /// Last few evaluations, so the driver reads the gradient at an
    /// accepted iterate without paying a force call.
    seen: Mutex<Vec<(Array1<f64>, f64, Array1<f64>)>>,
}

// SAFETY: the benchmark is single-threaded; the raw pointers are owned
// by this struct for its lifetime and every access goes through the
// Mutex-guarded buffers.
unsafe impl Send for FreeObjective {}
unsafe impl Sync for FreeObjective {}

impl FreeObjective {
    fn eval(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let mut guard = self.full.lock().unwrap();
        let (pos, grad) = &mut *guard;
        for (k, &i) in self.free.iter().enumerate() {
            pos[i] = x[k];
        }
        let n = pos.len();
        let xs = Shell::new(pos.as_mut_ptr(), n);
        let mut gs = Shell::new(grad.as_mut_ptr(), n);
        let mut value = f64::NAN;
        let base = self.pot as *const eindir_core::ffi::eindir_objective_t;
        let st = unsafe { eindir_objective_eval(base, &*xs.managed, &mut value) };
        let sg = unsafe { eindir_objective_grad(base, &*xs.managed, &mut *gs.managed) };
        if st != eindir_status_t::EINDIR_SUCCESS || sg != eindir_status_t::EINDIR_SUCCESS {
            return (f64::INFINITY, Array1::zeros(x.len()));
        }
        let g = Array1::from_iter(self.free.iter().map(|&i| grad[i]));
        drop(guard);
        let mut seen = self.seen.lock().unwrap();
        if seen.len() >= 64 {
            seen.remove(0);
        }
        seen.push((x.to_owned(), value, g.clone()));
        (value, g)
    }

    fn lookup(&self, x: &Array1<f64>) -> Option<(f64, Array1<f64>)> {
        let seen = self.seen.lock().unwrap();
        seen.iter()
            .rev()
            .find(|(p, _, _)| p == x)
            .map(|(_, f, g)| (*f, g.clone()))
    }

    fn calls(&self) -> usize {
        unsafe { (*self.kernel).calls.load(Ordering::Relaxed) }
    }

    fn pot_seconds(&self) -> f64 {
        unsafe { (*self.kernel).nanos.load(Ordering::Relaxed) as f64 * 1e-9 }
    }
}

impl Drop for FreeObjective {
    fn drop(&mut self) {
        unsafe {
            rgpot_potential_free_eindir(self.pot);
            drop(Box::from_raw(self.kernel));
        }
    }
}

impl Objective<f64> for FreeObjective {
    fn dim(&self) -> usize {
        self.free.len()
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        FreeObjective::eval(self, x).0
    }
}

impl Gradient<f64> for FreeObjective {
    fn dim(&self) -> usize {
        self.free.len()
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        FreeObjective::eval(self, x).1
    }
}

impl DifferentiableObjective<f64> for FreeObjective {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        FreeObjective::eval(self, x)
    }
}

/// A system: full positions, atomic numbers, cell, frozen mask.
struct System {
    kind: i32,
    pos: Vec<f64>,
    z: Vec<i32>,
    cell: [f64; 9],
    fixed: Vec<bool>,
}

fn read_con(path: &str, z: i32, kind: i32) -> System {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let lines: Vec<&str> = text.lines().collect();
    let lengths: Vec<f64> = lines[2]
        .split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect();
    let mut pos = Vec::new();
    let mut fixed = Vec::new();
    for line in &lines[10..] {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() != 5 {
            continue;
        }
        for v in &f[..3] {
            pos.push(v.parse::<f64>().unwrap());
        }
        fixed.push(f[3] == "1");
    }
    let n = fixed.len();
    System {
        kind,
        pos,
        z: vec![z; n],
        cell: [
            lengths[0], 0.0, 0.0, 0.0, lengths[1], 0.0, 0.0, 0.0, lengths[2],
        ],
        fixed,
    }
}

/// splitmix64: a seeded, dependency-free generator.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// LJ38 from a random packing in a sphere: no pair closer than 0.9.
fn lj38(seed: u64) -> System {
    let n = 38;
    let radius = 2.4;
    let mut rng = Rng(seed.wrapping_mul(7919) + 17);
    let mut pos: Vec<f64> = Vec::with_capacity(3 * n);
    while pos.len() < 3 * n {
        let p = [
            (2.0 * rng.uniform() - 1.0) * radius,
            (2.0 * rng.uniform() - 1.0) * radius,
            (2.0 * rng.uniform() - 1.0) * radius,
        ];
        if p.iter().map(|v| v * v).sum::<f64>() > radius * radius {
            continue;
        }
        let close = pos.chunks(3).any(|q| {
            let d: f64 = (0..3).map(|k| (q[k] - p[k]).powi(2)).sum();
            d < 0.81
        });
        if !close {
            pos.extend_from_slice(&p);
        }
    }
    System {
        kind: 0,
        pos,
        z: vec![18; n],
        cell: [100.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 100.0],
        fixed: vec![false; n],
    }
}

const DIAMOND: [[f64; 3]; 8] = [
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.0],
    [0.5, 0.0, 0.5],
    [0.0, 0.5, 0.5],
    [0.25, 0.25, 0.25],
    [0.75, 0.75, 0.25],
    [0.75, 0.25, 0.75],
    [0.25, 0.75, 0.75],
];

fn diamond_cell(nx: usize, ny: usize, nz: usize, a: f64) -> Vec<f64> {
    let mut pos = Vec::new();
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                for b in DIAMOND {
                    pos.push((ix as f64 + b[0]) * a);
                    pos.push((iy as f64 + b[1]) * a);
                    pos.push((iz as f64 + b[2]) * a);
                }
            }
        }
    }
    pos
}

/// 160-atom Stillinger-Weber silicon slab, Packwood et al. section III.
/// A 1x1x20 diamond cell, free along z, with the upper half shifted down
/// by 0.5 Å.
fn si_slab() -> System {
    let a = 5.431;
    let nz = 20;
    let mut pos = diamond_cell(1, 1, nz, a);
    let zmid = 0.5 * nz as f64 * a;
    for z in pos.iter_mut().skip(2).step_by(3) {
        if *z >= zmid {
            *z -= 0.5;
        }
    }
    let n = pos.len() / 3;
    let vacuum = 12.0;
    System {
        kind: 3,
        z: vec![14; n],
        cell: [a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, nz as f64 * a + vacuum],
        pos,
        fixed: vec![false; n],
    }
}

/// 256-atom Stillinger-Weber wire: 32x1x1 diamond, 0.5% compression,
/// plus a 0.1 Å random displacement. Packwood et al. section III.
fn si_strain() -> System {
    let a0 = 5.431;
    let a = a0 * 0.995;
    let nx = 32;
    let mut pos = diamond_cell(nx, 1, 1, a);
    let mut rng = Rng(17);
    for p in &mut pos {
        *p += 0.1 * rng.normal();
    }
    let n = pos.len() / 3;
    System {
        kind: 3,
        z: vec![14; n],
        cell: [nx as f64 * a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a],
        pos,
        fixed: vec![false; n],
    }
}

/// FCC copper with one vacancy, Asap EMT. 5x5x5 conventional cells,
/// 499 atoms. `a` is the energy minimum of this EMT build: a perfect
/// crystal there has zero force and an energy near 0 eV/atom, because
/// the potential subtracts E0.
fn cu_vacancy() -> System {
    let a = 3.595;
    let ncell = 5;
    let basis = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ];
    let mut pos = Vec::new();
    for ix in 0..ncell {
        for iy in 0..ncell {
            for iz in 0..ncell {
                for b in basis {
                    pos.push((ix as f64 + b[0]) * a);
                    pos.push((iy as f64 + b[1]) * a);
                    pos.push((iz as f64 + b[2]) * a);
                }
            }
        }
    }
    // Drop the atom nearest the cell centre.
    let centre = 0.5 * ncell as f64 * a;
    let mut drop = 0;
    let mut best = f64::INFINITY;
    for i in 0..pos.len() / 3 {
        let d = (0..3)
            .map(|k| pos[3 * i + k] - centre)
            .map(|v| v * v)
            .sum::<f64>();
        if d < best {
            best = d;
            drop = i;
        }
    }
    pos.drain(3 * drop..3 * drop + 3);
    let n = pos.len() / 3;
    let l = ncell as f64 * a;
    System {
        kind: 4,
        z: vec![29; n],
        cell: [l, 0.0, 0.0, 0.0, l, 0.0, 0.0, 0.0, l],
        pos,
        fixed: vec![false; n],
    }
}

fn build(sys: &System, sigma: f64, seed: u64) -> (FreeObjective, Array1<f64>) {
    let n = sys.fixed.len();
    let kernel = Box::into_raw(Box::new(Kernel {
        kind: sys.kind,
        forces: vec![0.0; 3 * n],
        calls: AtomicUsize::new(0),
        nanos: AtomicU64::new(0),
    }));
    let lo = vec![-1e6; 3 * n];
    let hi = vec![1e6; 3 * n];
    let pot = unsafe {
        rgpot_potential_new_eindir(
            force_callback,
            kernel.cast(),
            None,
            n,
            sys.z.as_ptr(),
            sys.cell.as_ptr(),
            lo.as_ptr(),
            hi.as_ptr(),
        )
    };
    assert!(!pot.is_null());
    let free: Vec<usize> = (0..n)
        .filter(|&a| !sys.fixed[a])
        .flat_map(|a| [3 * a, 3 * a + 1, 3 * a + 2])
        .collect();
    let mut rng = Rng(seed.wrapping_mul(104_729) + 3);
    let x0 = Array1::from_iter(free.iter().map(|&i| sys.pos[i] + sigma * rng.normal()));
    let m = free.len();
    let obj = FreeObjective {
        pot,
        kernel,
        full: Mutex::new((sys.pos.clone(), vec![0.0; 3 * n])),
        free,
        bounds: Bounds::new(Array1::from_elem(m, -1e6), Array1::from_elem(m, 1e6), 0.0),
        seen: Mutex::new(Vec::new()),
    };
    (obj, x0)
}

/// Largest per-atom force norm.
fn fmax(g: &Array1<f64>) -> f64 {
    g.as_slice()
        .unwrap()
        .chunks(3)
        .map(|c| c.iter().map(|v| v * v).sum::<f64>().sqrt())
        .fold(0.0, f64::max)
}

fn wolfe() -> LineSearch {
    LineSearch::Wolfe {
        c1: 1e-4,
        c2: 0.9,
        maxiter: 20,
    }
}

/// `(method, accept, linesearch, istep)` for a session name.
fn session(name: &str) -> Option<(Method, Accept, LineSearch, f64)> {
    let lbfgs = Method::Lbfgs { memory: 10 };
    let pr = Method::Nlcg {
        conjugacy: Conjugacy::PolakRibiere,
        restart: Restart::njws(),
    };
    let back = LineSearch::Backtracking {
        c: 1e-4,
        beta: 0.5,
        maxiter: 20,
    };
    Some(match name {
        "lbfgs-brent" => (lbfgs, Accept::None, LineSearch::default(), 1.0),
        "lbfgs-wolfe" => (lbfgs, Accept::None, wolfe(), 1.0),
        "lbfgs-backtrack" => (lbfgs, Accept::None, back, 1.0),
        "lbfgs-step" => (lbfgs, Accept::Step, wolfe(), 1.0),
        "fire" => (
            Method::Fire { kind: FireKind::V1 },
            Accept::None,
            wolfe(),
            0.1,
        ),
        "fire2" | "fire2g" => (
            Method::Fire { kind: FireKind::V2 },
            Accept::None,
            wolfe(),
            0.1,
        ),
        "bb" => (Method::Bb, Accept::None, wolfe(), 0.01),
        "bb-nm" => (Method::Bb, Accept::Nonmonotone, wolfe(), 0.01),
        "cg-wolfe" | "cg-exp" => (pr, Accept::None, wolfe(), 0.01),
        "cg-brent" => (pr, Accept::None, LineSearch::default(), 0.01),
        "lbfgs-exp" => (lbfgs, Accept::None, wolfe(), 1.0),
        "fire-exp" => (
            Method::Fire { kind: FireKind::V1 },
            Accept::None,
            wolfe(),
            0.1,
        ),
        _ => return None,
    })
}

fn max_atom_move(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.as_slice()
        .unwrap_or(&[])
        .chunks(3)
        .zip(b.as_slice().unwrap_or(&[]).chunks(3))
        .map(|(u, v)| {
            let s: f64 = (0..3).map(|k| (u[k] - v[k]).powi(2)).sum();
            s.sqrt()
        })
        .fold(0.0, f64::max)
}

fn mic(dx: f64, length: f64) -> f64 {
    if !(length.is_finite() && length > 0.0) {
        return dx;
    }
    dx - length * (dx / length).round()
}

fn pair_distance(pos: &[f64], a: usize, b: usize, lx: f64, ly: f64, lz: f64) -> f64 {
    let dx = mic(pos[3 * a] - pos[3 * b], lx);
    let dy = mic(pos[3 * a + 1] - pos[3 * b + 1], ly);
    let dz = mic(pos[3 * a + 2] - pos[3 * b + 2], lz);
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Neighbours of the free atoms, minimum image, cutoff `2 r_nn`.
/// Bonds to frozen atoms are anchors. The second value is `r_nn`.
fn exp_neighbours(sys: &System, x: &Array1<f64>) -> (Neighbours, f64) {
    let free: Vec<usize> = (0..sys.fixed.len()).filter(|&a| !sys.fixed[a]).collect();
    let nall = sys.fixed.len();
    let mut pos = sys.pos.clone();
    for (k, &atom) in free.iter().enumerate() {
        for a in 0..3 {
            pos[3 * atom + a] = x[3 * k + a];
        }
    }
    let lx = sys.cell[0];
    let ly = sys.cell[4];
    let lz = sys.cell[8];
    let half = lx.min(ly).min(lz) * 0.5;
    // A = 3 keeps neighbours out to 2 r_nn. The search has to reach that
    // shell. MIC stores one image of each pair.
    let search = if half.is_finite() && half > 0.0 {
        half.max(6.0).min(12.0)
    } else {
        8.0
    };
    let mut min_d = vec![f64::INFINITY; nall];
    for a in 0..nall {
        for b in (a + 1)..nall {
            let r = pair_distance(&pos, a, b, lx, ly, lz);
            if r > 1e-8 && r < search {
                min_d[a] = min_d[a].min(r);
                min_d[b] = min_d[b].min(r);
            }
        }
    }
    let r_nn = min_d
        .iter()
        .copied()
        .filter(|d| d.is_finite())
        .fold(0.0_f64, f64::max);
    let r_nn = if r_nn > 0.0 { r_nn } else { 1.0 };
    let r_cut = 2.0 * r_nn;
    let mut packed = vec![None; nall];
    for (k, &atom) in free.iter().enumerate() {
        packed[atom] = Some(k);
    }
    let mut neigh = Neighbours::default();
    for a in 0..nall {
        for b in (a + 1)..nall {
            let r = pair_distance(&pos, a, b, lx, ly, lz);
            if r <= 1e-8 || r >= r_cut {
                continue;
            }
            match (packed[a], packed[b]) {
                (Some(i), Some(j)) => {
                    neigh.i.push(i);
                    neigh.j.push(j);
                    neigh.r.push(r);
                }
                (Some(i), None) => neigh.anchor.push((i, r)),
                (None, Some(j)) => neigh.anchor.push((j, r)),
                (None, None) => {}
            }
        }
    }
    (neigh, r_nn)
}

struct Outcome {
    converged: bool,
    iters: usize,
    energy: f64,
    fmax: f64,
}

fn run_session(
    obj: &FreeObjective,
    mut x: Array1<f64>,
    sys: &System,
    name: &str,
    tol: f64,
    maxiter: usize,
    maxmove: f64,
) -> Outcome {
    let (method, accept, ls, istep) = session(name).expect("unknown method");
    let control = Control {
        maxiter: 1,
        gtol: 0.0,
        istep,
        maxmove: None,
        ftol_rel: None,
    };
    let mut solver = Solver::new(method, control, x.len());
    solver.set_accept(accept);
    solver.set_linesearch(ls);
    solver.set_atom_maxmove(maxmove);
    if name == "fire2g" {
        solver.set_fire_variant(rgmin::FireVariant::Guenole2020);
    }
    let exp = name.ends_with("-exp");
    let mut mu = None;
    let mut built_at: Option<Array1<f64>> = None;
    let mut r_nn_gate = 1.0;
    let mut out = Outcome {
        converged: false,
        iters: 0,
        energy: f64::NAN,
        fmax: f64::INFINITY,
    };
    for it in 0..maxiter {
        if exp {
            // Packwood et al. rebuild P when the largest atomic move since
            // the last assembly exceeds r_nn / 2. A fixed P keeps the CG
            // metric and the FIRE velocity in one inner product.
            let disp = built_at
                .as_ref()
                .map(|p| max_atom_move(p, &x))
                .unwrap_or(f64::INFINITY);
            if disp > 0.5 * r_nn_gate {
                let (neigh, r_nn) = exp_neighbours(sys, &x);
                r_nn_gate = r_nn;
                let mut params = ExpParams::default();
                params.mu = mu;
                // The cluster sits in a vacuum box, so the sine probe uses
                // the span of the positions. Periodic cells pass their lengths.
                if sys.kind != 0 {
                    params.lengths = Some([sys.cell[0], sys.cell[4], sys.cell[8]]);
                }
                if solver.set_exp(x.view(), &neigh, &params).is_err() {
                    eprintln!("{name}: exp assembly failed");
                    break;
                }
                built_at = Some(x.clone());
            }
        }
        let rep = match solver.step(obj, &mut x) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("{name}: {e}");
                break;
            }
        };
        out.iters = it + 1;
        if exp && mu.is_none() {
            mu = solver.exp_mu();
        }
        let (f, g) = obj.lookup(&x).unwrap_or_else(|| {
            // Not counted against the solver: the driver's own check.
            let calls = unsafe { &(*obj.kernel).calls };
            let before = calls.load(Ordering::Relaxed);
            let r = obj.eval(x.view());
            calls.store(before, Ordering::Relaxed);
            r
        });
        out.energy = f;
        out.fmax = fmax(&g);
        let _ = rep;
        if out.fmax < tol {
            out.converged = true;
            break;
        }
        if !f.is_finite() {
            break;
        }
    }
    out
}

fn run_hop(obj: &FreeObjective, x: Array1<f64>, tol: f64, maxiter: usize) -> Outcome {
    let mut solver = Lbfgs::with_capacity(10);
    solver.gtol = 0.0;
    let mut converged = false;
    let mut iters = 0;
    let (f, xf, _) = solver.minimize_watched(
        x.view(),
        maxiter,
        |p| Some(obj.eval(p)),
        |it, _| {
            iters = it;
            true
        },
    );
    let _ = &mut converged;
    let (_, g) = obj.lookup(&xf).unwrap_or_else(|| obj.eval(xf.view()));
    let fm = fmax(&g);
    Outcome {
        converged: fm < tol,
        iters,
        energy: f,
        fmax: fm,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: {} <lj38|pt7|al> <method>[,<method>...] [seeds] [sigma] [data_dir]",
            args[0]
        );
        std::process::exit(2);
    }
    let system = args[1].as_str();
    let methods: Vec<&str> = args[2].split(',').collect();
    let seeds: u64 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(5);
    let sigma: f64 = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(0.1);
    let data = args.get(5).cloned().unwrap_or_else(|| "data".into());
    let tol = 0.01;
    let maxiter = 2000;
    let maxmove = 0.2;
    println!(
        "system,method,seed,converged,iters,calls,energy,fmax,wall_s,pot_s,overhead_us_per_iter"
    );
    for seed in 0..seeds {
        let sys = match system {
            "lj38" => lj38(seed),
            "pt7" => read_con(&format!("{data}/pt_heptamer.con"), 78, 1),
            "al" => read_con(&format!("{data}/al_slab.con"), 13, 2),
            "si-slab" => si_slab(),
            "si-strain" => si_strain(),
            "cu-vac" => cu_vacancy(),
            other => panic!("unknown system {other}"),
        };
        let sig = if system == "lj38" || system.starts_with("si-") || system == "cu-vac" {
            0.0
        } else {
            sigma
        };
        for &m in &methods {
            let (obj, x0) = build(&sys, sig, seed);
            let t0 = Instant::now();
            let out = if m == "lbfgs-hop" {
                run_hop(&obj, x0, tol, maxiter)
            } else {
                run_session(&obj, x0, &sys, m, tol, maxiter, maxmove)
            };
            let wall = t0.elapsed().as_secs_f64();
            let pot = obj.pot_seconds();
            let iters = out.iters.max(1);
            println!(
                "{system},{m},{seed},{},{},{},{:.10},{:.3e},{:.6},{:.6},{:.3}",
                out.converged as u8,
                out.iters,
                obj.calls(),
                out.energy,
                out.fmax,
                wall,
                pot,
                (wall - pot) / iters as f64 * 1e6
            );
        }
    }
}
