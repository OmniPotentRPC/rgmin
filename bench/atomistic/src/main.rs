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
    Accept, Conjugacy, Control, ExpParams, FfParams, FfTerms, FireKind, GeometryKind, Ghost, Lbfgs,
    LindhParams, LineSearch, Method, Neighbours, PreconChoice, QuickMinOptions, QuickMinUphill,
    Restart, Solver, bonded_terms, select_precon,
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
    let st = if k.kind >= 5 {
        let pos_s = unsafe { std::slice::from_raw_parts(pos.data as *const f64, 3 * natoms) };
        molecule_forces(k.kind, pos_s, &mut k.forces, &mut energy)
    } else {
        unsafe {
            bench_force(
                k.kind,
                natoms,
                pos.data as *const f64,
                z.data as *const i32,
                cell.data as *const f64,
                k.forces.as_mut_ptr(),
                &mut energy,
            )
        }
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

/// Harmonic water (`kind` 5) and methane (`kind` 6).
///
/// The buffer written into `forces` is the physical force, the same
/// sign the rgpot kernels use. `0` is success.
fn molecule_forces(kind: i32, pos: &[f64], forces: &mut [f64], energy: &mut f64) -> i32 {
    if forces.len() < pos.len() {
        return 2;
    }
    forces[..pos.len()].fill(0.0);
    let mut grad = vec![0.0; pos.len()];
    let mut e = 0.0;
    let ok = match kind {
        5 => water_energy(pos, &mut grad, &mut e),
        6 => methane_energy(pos, &mut grad, &mut e),
        _ => false,
    };
    if !ok {
        return 2;
    }
    for (f, g) in forces.iter_mut().zip(grad) {
        *f = -g;
    }
    *energy = e;
    0
}

fn add_bond(pos: &[f64], i: usize, j: usize, r0: f64, k: f64, energy: &mut f64, grad: &mut [f64]) {
    let mut d = [0.0; 3];
    let mut r2 = 0.0;
    for a in 0..3 {
        d[a] = pos[3 * i + a] - pos[3 * j + a];
        r2 += d[a] * d[a];
    }
    let r = r2.sqrt();
    if r < 1e-12 {
        return;
    }
    let dr = r - r0;
    *energy += 0.5 * k * dr * dr;
    let scale = k * dr / r;
    for a in 0..3 {
        grad[3 * i + a] += scale * d[a];
        grad[3 * j + a] -= scale * d[a];
    }
}

fn add_angle(
    pos: &[f64],
    i: usize,
    vertex: usize,
    k: usize,
    theta0: f64,
    k_theta: f64,
    energy: &mut f64,
    grad: &mut [f64],
) {
    let mut u = [0.0; 3];
    let mut v = [0.0; 3];
    for a in 0..3 {
        u[a] = pos[3 * i + a] - pos[3 * vertex + a];
        v[a] = pos[3 * k + a] - pos[3 * vertex + a];
    }
    let du = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
    let dv = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if du < 1e-12 || dv < 1e-12 {
        return;
    }
    let eu = [u[0] / du, u[1] / du, u[2] / du];
    let ev = [v[0] / dv, v[1] / dv, v[2] / dv];
    let c = (eu[0] * ev[0] + eu[1] * ev[1] + eu[2] * ev[2]).clamp(-1.0, 1.0);
    let s = (1.0 - c * c).sqrt();
    if s <= 1e-8 {
        return;
    }
    let theta = c.acos();
    let dtheta = theta - theta0;
    *energy += 0.5 * k_theta * dtheta * dtheta;
    let scale = k_theta * dtheta;
    let qu = [ev[0] - c * eu[0], ev[1] - c * eu[1], ev[2] - c * eu[2]];
    let qv = [eu[0] - c * ev[0], eu[1] - c * ev[1], eu[2] - c * ev[2]];
    let mut gi = [0.0; 3];
    let mut gk = [0.0; 3];
    for a in 0..3 {
        gi[a] = -qu[a] / (du * s);
        gk[a] = -qv[a] / (dv * s);
        grad[3 * i + a] += scale * gi[a];
        grad[3 * k + a] += scale * gk[a];
        grad[3 * vertex + a] -= scale * (gi[a] + gk[a]);
    }
}

fn water_energy(pos: &[f64], grad: &mut [f64], energy: &mut f64) -> bool {
    if pos.len() != 9 {
        return false;
    }
    let theta0 = 104.5_f64.to_radians();
    add_bond(pos, 0, 1, 0.957, 25.0, energy, grad);
    add_bond(pos, 0, 2, 0.957, 25.0, energy, grad);
    add_angle(pos, 1, 0, 2, theta0, 2.0, energy, grad);
    true
}

fn methane_energy(pos: &[f64], grad: &mut [f64], energy: &mut f64) -> bool {
    if pos.len() != 15 {
        return false;
    }
    let theta0 = (-1.0_f64 / 3.0).acos();
    for h in 1..5 {
        add_bond(pos, 0, h, 1.09, 15.0, energy, grad);
    }
    for a in 1..5 {
        for b in (a + 1)..5 {
            add_angle(pos, a, 0, b, theta0, 1.2, energy, grad);
        }
    }
    true
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

/// Stillinger-Weber wire: `nx` x 1 x 1 diamond, 0.5% compression, plus a
/// 0.1 Å random displacement. Packwood et al. section III, the
/// `N x 1 x 1` series (32 to 512 atoms).
fn si_wire(nx: usize) -> System {
    let a0 = 5.431;
    let a = a0 * 0.995;
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

/// FCC copper with one vacancy, Asap EMT, plus a 0.1 Å random
/// displacement. `a` is the energy minimum of this EMT build: a perfect
/// crystal there has zero force and an energy near 0 eV/atom, because
/// the potential subtracts E0. The ideal vacancy alone has a force below
/// 0.11 eV/Å, under the benchmark tolerance, so the cell is displaced.
fn cu_vacancy() -> System {
    let a = 3.595;
    let ncell = 8;
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
    let mut rng = Rng(17);
    for p in &mut pos {
        *p += 0.1 * rng.normal();
    }
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

/// Water, OH = 0.957 Å, HOH = 104.5°, displaced so the start is not the well.
fn h2o() -> System {
    let ang = 104.5_f64.to_radians();
    let r = 0.957;
    let pos = vec![
        0.0,
        0.04,
        -0.02,
        r - 0.03,
        0.02,
        0.01,
        r * ang.cos(),
        r * ang.sin() + 0.05,
        -0.02,
    ];
    System {
        kind: 5,
        pos,
        z: vec![8, 1, 1],
        cell: [30.0, 0.0, 0.0, 0.0, 30.0, 0.0, 0.0, 0.0, 30.0],
        fixed: vec![false; 3],
    }
}

/// Methane on a tetrahedron, CH = 1.09 Å, displaced off the well.
fn ch4() -> System {
    let r = 1.09;
    let dirs: [[f64; 3]; 4] = [
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ];
    let mut pos = vec![0.03, -0.02, 0.04];
    for d in dirs {
        let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        pos.push(r * d[0] / n);
        pos.push(r * d[1] / n);
        pos.push(r * d[2] / n);
    }
    pos[3] += 0.04;
    pos[7] -= 0.03;
    pos[11] += 0.02;
    System {
        kind: 6,
        pos,
        z: vec![6, 1, 1, 1, 1],
        cell: [30.0, 0.0, 0.0, 0.0, 30.0, 0.0, 0.0, 0.0, 30.0],
        fixed: vec![false; 5],
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

/// Nocedal-Wright section 5.2: CG uses a tighter curvature constant than
/// the quasi-Newton default. The same search is used with and without a metric.
fn cg_wolfe() -> LineSearch {
    LineSearch::Wolfe {
        c1: 1e-4,
        c2: 0.1,
        maxiter: 20,
    }
}

/// Quick-min options from `QM_X`, a comma list over
/// [`QuickMinOptions::rgmin`]: `eon` (start from the eOn options),
/// `fixed`, `norescale`, `peratom`, `restart`, `backtrack`, `curv=<c>`,
/// `dtmax=<factor>`.
fn quickmin_from_env() -> QuickMinOptions {
    let mut o = QuickMinOptions::rgmin();
    let spec = std::env::var("QM_X").unwrap_or_default();
    for tok in spec.split(',').filter(|t| !t.is_empty()) {
        let (key, val) = tok.split_once('=').unwrap_or((tok, ""));
        match key {
            "eon" => o = QuickMinOptions::eon(),
            "fixed" => o.adapt_dt = false,
            "norescale" => o.rescale_on_clamp = false,
            "peratom" => o.per_atom = true,
            "restart" => o.uphill = QuickMinUphill::Restart,
            "backtrack" => o.uphill = QuickMinUphill::Backtrack,
            "curv" => o.curvature_cap = Some(val.parse().expect("curv=<c>")),
            "dtmax" => o.dt_max_factor = val.parse().expect("dtmax=<f>"),
            other => panic!("QM_X: unknown option {other}"),
        }
    }
    o
}

/// `(method, accept, linesearch, istep)` for a session name.
fn session(name: &str) -> Option<(Method, Accept, LineSearch, f64)> {
    let lbfgs = Method::Lbfgs { memory: 10 };
    let pr = Method::Nlcg {
        conjugacy: Conjugacy::PolakRibierePlus,
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
        "cg-wolfe" | "cg" => (pr, Accept::None, cg_wolfe(), 0.01),
        // Library change alone: signed Polak-Ribiere and the session
        // Wolfe constant c2 = 0.9. The opening trial is still equation 3.60.
        "cg-ablation" => (
            Method::Nlcg {
                conjugacy: Conjugacy::PolakRibiere,
                restart: Restart::njws(),
            },
            Accept::None,
            wolfe(),
            0.01,
        ),
        "cg-brent" => (pr, Accept::None, LineSearch::default(), 0.01),
        "lbfgs" => (lbfgs, Accept::None, wolfe(), 1.0),
        "qm" | "qm-eon" | "qmx" => (Method::QuickMin, Accept::None, wolfe(), 0.1),
        _ => return None,
    })
}

enum PreconKind {
    None,
    Exp,
    Lindh,
    Ff,
    Auto,
}

/// Split a `lbfgs-lindh` style name into the solver and the metric.
///
/// `auto` is [`select_precon`] with no force field. The `-ff` arm is the
/// bonded geometric metric, which that rule picks only for a molecule
/// or a cluster when the caller has a field.
fn classify(name: &str) -> Option<(&str, PreconKind)> {
    if let Some(base) = name.strip_suffix("-auto") {
        Some((base, PreconKind::Auto))
    } else if let Some(base) = name.strip_suffix("-lindh") {
        Some((base, PreconKind::Lindh))
    } else if let Some(base) = name.strip_suffix("-ff") {
        Some((base, PreconKind::Ff))
    } else if let Some(base) = name.strip_suffix("-exp") {
        Some((base, PreconKind::Exp))
    } else {
        Some((name, PreconKind::None))
    }
}

fn geometry_of(sys: &System) -> GeometryKind {
    match sys.kind {
        2 | 3 => GeometryKind::Surface,
        4 => GeometryKind::Bulk,
        5 | 6 => GeometryKind::Molecule,
        _ => GeometryKind::Cluster,
    }
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
    fallbacks: u64,
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
    let (base, which) = classify(name).expect("unknown method");
    let (method, accept, ls, istep) = session(base).expect("unknown method");
    let control = Control {
        maxiter: 1,
        gtol: 0.0,
        istep,
        maxmove: None,
        ftol_rel: None,
        noise_floor: None,
    };
    let mut solver = Solver::new(method, control, x.len());
    solver.set_accept(accept);
    solver.set_linesearch(ls);
    solver.set_atom_maxmove(maxmove);
    match name {
        "qm-eon" => solver.set_quickmin_options(QuickMinOptions::eon()),
        "qmx" => solver.set_quickmin_options(quickmin_from_env()),
        _ => {}
    }
    if name == "fire2g" {
        solver.set_fire_variant(rgmin::FireVariant::Guenole2020);
    }
    let auto = if matches!(which, PreconKind::Auto) {
        let free = free_index(sys);
        let z: Vec<i32> = free.iter().map(|&a| sys.z[a]).collect();
        Some(select_precon(
            geometry_of(sys),
            &z,
            false,
            solver.optimizer_family(),
        ))
    } else {
        None
    };
    let exp = matches!(which, PreconKind::Exp) || matches!(auto, Some(PreconChoice::Exp));
    let mut mu = None;
    let mut built_at: Option<Array1<f64>> = None;
    let mut r_nn_gate = 1.0;
    let mut out = Outcome {
        converged: false,
        iters: 0,
        energy: f64::NAN,
        fmax: f64::INFINITY,
        fallbacks: 0,
    };
    let want_lindh =
        matches!(which, PreconKind::Lindh) || matches!(auto, Some(PreconChoice::Lindh));
    let want_ff = matches!(which, PreconKind::Ff) || matches!(auto, Some(PreconChoice::Ff));
    if want_lindh && !install_lindh(&mut solver, sys, &x) {
        eprintln!("{name}: lindh assembly failed");
        return out;
    }
    if want_ff && !install_ff(&mut solver, sys, &x) {
        eprintln!("{name}: ff assembly failed");
        return out;
    }
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
            // The sine-probe gradient is an extra `grad` inside `step`.
            // `calls` counts it. The fmax check below is refunded.
            mu = solver.precon_scale();
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
    out.fallbacks = solver.nlcg_descent_fallbacks();
    out
}

fn cell_lengths(sys: &System) -> Option<[f64; 3]> {
    if sys.kind == 0 || sys.kind >= 5 {
        None
    } else {
        Some([sys.cell[0], sys.cell[4], sys.cell[8]])
    }
}

fn free_index(sys: &System) -> Vec<usize> {
    (0..sys.fixed.len()).filter(|&a| !sys.fixed[a]).collect()
}

fn install_lindh(solver: &mut Solver, sys: &System, x: &Array1<f64>) -> bool {
    let free = free_index(sys);
    let z: Vec<i32> = free.iter().map(|&a| sys.z[a]).collect();
    let mut ghosts = Vec::new();
    for (a, fixed) in sys.fixed.iter().enumerate() {
        if *fixed {
            ghosts.push(Ghost {
                z: sys.z[a],
                position: [sys.pos[3 * a], sys.pos[3 * a + 1], sys.pos[3 * a + 2]],
            });
        }
    }
    let mut params = LindhParams::default();
    params.lengths = cell_lengths(sys);
    solver.set_lindh(x.view(), &z, &ghosts, &params).is_ok()
}

/// Bonded geometric metric: pairs within 3.5 Å, unit bond curvature,
/// angle curvature 0.1. These are not the Makri, Ortner, and Kermode
/// 2019 parameters.
fn install_ff(solver: &mut Solver, sys: &System, x: &Array1<f64>) -> bool {
    let lengths = cell_lengths(sys);
    let Ok((bonds, angles)) = bonded_terms(x.view(), 3.5, 1.0, 0.1, lengths) else {
        return false;
    };
    if bonds.is_empty() && angles.is_empty() {
        return false;
    }
    let terms = FfTerms {
        bonds: &bonds,
        angles: &angles,
        ..FfTerms::default()
    };
    let mut params = FfParams::default();
    params.r_cut = Some(3.5);
    params.lengths = lengths;
    solver.set_ff(x.view(), terms, &params).is_ok()
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
        fallbacks: 0,
    }
}

#[cfg(test)]
mod force_check {
    use super::{methane_energy, water_energy};

    fn fd(kind: &str, pos: &[f64]) {
        let mut grad = vec![0.0; pos.len()];
        let mut e0 = 0.0;
        let ok = if kind == "h2o" {
            water_energy(pos, &mut grad, &mut e0)
        } else {
            methane_energy(pos, &mut grad, &mut e0)
        };
        assert!(ok);
        let eps = 1e-6;
        for i in 0..pos.len() {
            let mut plus = pos.to_vec();
            let mut minus = pos.to_vec();
            plus[i] += eps;
            minus[i] -= eps;
            let mut gp = vec![0.0; pos.len()];
            let mut gm = vec![0.0; pos.len()];
            let mut ep = 0.0;
            let mut em = 0.0;
            if kind == "h2o" {
                water_energy(&plus, &mut gp, &mut ep);
                water_energy(&minus, &mut gm, &mut em);
            } else {
                methane_energy(&plus, &mut gp, &mut ep);
                methane_energy(&minus, &mut gm, &mut em);
            }
            let num = (ep - em) / (2.0 * eps);
            assert!(
                (num - grad[i]).abs() < 1e-5,
                "{kind} coord {i}: fd {num} analytic {}",
                grad[i]
            );
        }
    }

    #[test]
    fn analytic_gradients_match_a_central_difference() {
        let water = super::h2o();
        let methane = super::ch4();
        fd("h2o", &water.pos);
        fd("ch4", &methane.pos);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: {} <lj38|pt7|al|si-slab|si-strain|si-wire|cu-vac|h2o|ch4> <method>[,<method>...] [seeds] [sigma] [data_dir] [repeats]",
            args[0]
        );
        std::process::exit(2);
    }
    let system = args[1].as_str();
    let methods: Vec<&str> = args[2].split(',').collect();
    let seeds: u64 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(5);
    let sigma: f64 = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(0.1);
    let data = args.get(5).cloned().unwrap_or_else(|| "data".into());
    let repeats: u64 = args.get(6).map(|s| s.parse().unwrap()).unwrap_or(1);
    let tol = 0.01;
    let maxiter = 2000;
    let maxmove = 0.2;
    println!(
        "system,method,seed,converged,iters,calls,energy,fmax,wall_s,pot_s,overhead_us_per_iter,rep,fallbacks"
    );
    let seed0: u64 = args.get(7).map(|s| s.parse().unwrap()).unwrap_or(0);
    let forced = std::env::var("RGMIN_BENCH_FORCE_SIGMA")
        .ok()
        .and_then(|s| s.parse::<f64>().ok());
    for rep in 0..repeats {
        for seed in seed0..seed0 + seeds {
            let sys = match system {
                "lj38" => lj38(seed),
                "pt7" => read_con(&format!("{data}/pt_heptamer.con"), 78, 1),
                "al" => read_con(&format!("{data}/al_slab.con"), 13, 2),
                "si-slab" => si_slab(),
                "si-strain" => si_wire(32),
                "si-wire" => si_wire(64),
                "cu-vac" => cu_vacancy(),
                "h2o" => h2o(),
                "ch4" => ch4(),
                other => panic!("unknown system {other}"),
            };
            let sig = if let Some(sigma) = forced {
                sigma
            } else if system == "lj38"
                || system.starts_with("si-")
                || system == "cu-vac"
                || system == "h2o"
                || system == "ch4"
            {
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
                    "{system},{m},{seed},{},{},{},{:.10},{:.3e},{:.6},{:.6},{:.3},{rep},{}",
                    out.converged as u8,
                    out.iters,
                    obj.calls(),
                    out.energy,
                    out.fmax,
                    wall,
                    pot,
                    (wall - pot) / iters as f64 * 1e6,
                    out.fallbacks
                );
            }
        }
    }
}
