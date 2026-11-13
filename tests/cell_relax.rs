//! Variable-cell relaxation on Stillinger-Weber silicon and Sutton-Chen copper.
//!
//! The stress the filter sees is the tensor
//! `rgpot_potential_calculate_mechanical` returns (rgpot-core
//! `fdd5ae53`). The kernels below are the 1985 Stillinger-Weber silicon
//! parameters and the Sutton-Chen copper set, evaluated here and reported
//! through that call.

use std::os::raw::c_void;
use std::process::Command;

use rgmin::{
    CellFilter, CellMask, Control, FireKind, LineSearch, Mechanical, Mechanics, Method, StopNorm,
    minimize_method,
};
use rgpot_core::c_api::potential::{
    rgpot_mechanical_slot, rgpot_potential_calculate_mechanical, rgpot_potential_free,
    rgpot_potential_new,
};
use rgpot_core::status::rgpot_status_t;
use rgpot_core::tensor::{
    rgpot_tensor_cpu_f64_2d, rgpot_tensor_cpu_f64_matrix3, rgpot_tensor_cpu_i32_1d,
    rgpot_tensor_data, rgpot_tensor_free,
};
use rgpot_core::types::{rgpot_force_input_t, rgpot_force_out_t};

const SW_EPSILON: f64 = 3.4723e-12 * 1.0e-7 / 1.602176634e-19;
const SW_SIGMA: f64 = 2.0951;
const SW_A: f64 = 7.049556277;
const SW_B: f64 = 0.6022245584;
const SW_P: f64 = 4.0;
const SW_CUT: f64 = 1.80;
const SW_LAMBDA: f64 = 21.0;
const SW_GAMMA: f64 = 1.20;

/// Sutton and Chen, Phil. Mag. Lett. 61, 139 (1990), copper row.
const SC_EPSILON: f64 = 1.2382e-2;
const SC_C: f64 = 39.432;
const SC_A: f64 = 3.61;
const SC_N: i32 = 9;
const SC_M: i32 = 6;
const SC_CUTOFF: f64 = 7.0;

/// 1 GPa in eV/Å³. 1 eV/Å³ = 160.2176634 GPa.
const GPA: f64 = 1.0 / 160.2176634;

#[derive(Clone, Copy)]
enum Kind {
    Sw,
    Sc,
}

struct Kernel {
    z: i32,
    pot: *mut rgpot_core::eindir::rgpot_potential_t,
}

unsafe impl Send for Kernel {}
unsafe impl Sync for Kernel {}

impl Kernel {
    fn new(kind: Kind) -> Self {
        let z = match kind {
            Kind::Sw => 14,
            Kind::Sc => 29,
        };
        let state = Box::new(KindState {
            kind,
            forces: Vec::new(),
        });
        let pot = unsafe {
            rgpot_potential_new(
                kernel_callback,
                Box::into_raw(state) as *mut c_void,
                Some(free_state),
            )
        };
        assert!(!pot.is_null());
        Self { z, pot }
    }
}

impl Drop for Kernel {
    fn drop(&mut self) {
        unsafe { rgpot_potential_free(self.pot) };
    }
}

struct KindState {
    kind: Kind,
    forces: Vec<f64>,
}

unsafe extern "C" fn free_state(p: *mut c_void) {
    if !p.is_null() {
        drop(unsafe { Box::from_raw(p.cast::<KindState>()) });
    }
}

unsafe extern "C" fn kernel_callback(
    user: *mut c_void,
    input: *const rgpot_force_input_t,
    output: *mut rgpot_force_out_t,
) -> rgpot_status_t {
    let state = unsafe { &mut *user.cast::<KindState>() };
    let inp = unsafe { &*input };
    let n = unsafe { inp.n_atoms() }.unwrap_or(0);
    let pos = unsafe {
        std::slice::from_raw_parts(rgpot_tensor_data(inp.positions) as *const f64, 3 * n)
    };
    let mut box9 = [0.0; 9];
    let raw =
        unsafe { std::slice::from_raw_parts(rgpot_tensor_data(inp.box_matrix) as *const f64, 9) };
    box9.copy_from_slice(raw);
    let (energy, forces, virial, stress) = match state.kind {
        Kind::Sw => sw_eval(pos, &box9),
        Kind::Sc => sc_eval(pos, &box9),
    };
    state.forces = forces;
    let out = unsafe { &mut *output };
    out.energy = energy;
    out.variance = 0.0;
    out.forces = unsafe { rgpot_tensor_cpu_f64_2d(state.forces.as_mut_ptr(), n as i64, 3) };
    let slot = unsafe { rgpot_mechanical_slot() };
    if !slot.is_null() {
        let m = unsafe { &mut *slot };
        m.virial = virial;
        m.stress = stress;
        m.has_virial = 1;
        m.has_stress = 1;
    }
    rgpot_status_t::RGPOT_SUCCESS
}

impl Mechanical for Kernel {
    fn evaluate(&self, positions: &[f64], cell: &[f64; 9]) -> rgmin::Result<Mechanics> {
        let n = positions.len() / 3;
        let mut pos = positions.to_vec();
        let mut z = vec![self.z; n];
        let mut box9 = *cell;
        let input = rgpot_force_input_t {
            positions: unsafe { rgpot_tensor_cpu_f64_2d(pos.as_mut_ptr(), n as i64, 3) },
            atomic_numbers: unsafe { rgpot_tensor_cpu_i32_1d(z.as_mut_ptr(), n as i64) },
            box_matrix: unsafe { rgpot_tensor_cpu_f64_matrix3(box9.as_mut_ptr()) },
        };
        let mut output = rgpot_force_out_t {
            forces: std::ptr::null_mut(),
            energy: 0.0,
            variance: 0.0,
        };
        let mut mechanical = rgpot_core::types::rgpot_mechanical_t {
            virial: [0.0; 9],
            stress: [0.0; 9],
            has_virial: 0,
            has_stress: 0,
        };
        let status = unsafe {
            rgpot_potential_calculate_mechanical(self.pot, &input, &mut output, &mut mechanical)
        };
        unsafe {
            rgpot_tensor_free(input.positions);
            rgpot_tensor_free(input.atomic_numbers);
            rgpot_tensor_free(input.box_matrix);
        }
        if status != rgpot_status_t::RGPOT_SUCCESS || mechanical.has_stress == 0 {
            unsafe { rgpot_tensor_free(output.forces) };
            return Err(rgmin::Error::Oracle {
                what: "mechanical evaluate failed",
            });
        }
        let forces = unsafe {
            std::slice::from_raw_parts(rgpot_tensor_data(output.forces) as *const f64, 3 * n)
                .to_vec()
        };
        unsafe { rgpot_tensor_free(output.forces) };
        Ok(Mechanics {
            energy: output.energy,
            forces,
            stress: mechanical.stress,
        })
    }
}

fn sw_pair(r: f64) -> Option<(f64, f64, f64, f64)> {
    let u = r / SW_SIGMA;
    let gap = u - SW_CUT;
    if gap >= 0.0 || gap > -1.0 / 700.0 {
        return None;
    }
    let s = 1.0 / gap;
    let inv_u = 1.0 / u;
    let u_m4 = inv_u.powi(4);
    let g = SW_B * u_m4 - 1.0;
    let es = s.exp();
    let f2 = SW_A * g * es;
    let dgdu = -SW_P * SW_B * u_m4 * inv_u;
    let dfdu = SW_A * es * (dgdu - g * s * s);
    let dedr = SW_EPSILON * dfdu / SW_SIGMA;
    let gexp = (SW_GAMMA * s).exp();
    let dgdr = gexp * (-SW_GAMMA * s * s) / SW_SIGMA;
    Some((f2, dedr, gexp, dgdr))
}

fn sw_eval(pos: &[f64], cell: &[f64; 9]) -> (f64, Vec<f64>, [f64; 9], [f64; 9]) {
    let n = pos.len() / 3;
    let cut = SW_CUT * SW_SIGMA;
    let mut energy = 0.0;
    let mut force = vec![0.0; 3 * n];
    let mut virial = [0.0; 9];
    let mut bonds = vec![Vec::new(); n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let d = min_image(
                *cell,
                [
                    pos[3 * i] - pos[3 * j],
                    pos[3 * i + 1] - pos[3 * j + 1],
                    pos[3 * i + 2] - pos[3 * j + 2],
                ],
            );
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            if r <= 0.0 || r >= cut {
                continue;
            }
            let Some((f2, dedr, gexp, dgdr)) = sw_pair(r) else {
                continue;
            };
            // Store r_j - r_i = -d, matching the C++ bond.
            bonds[i].push((-d[0], -d[1], -d[2], r, dedr, gexp, dgdr, f2, j));
        }
    }
    for i in 0..n {
        for b in &bonds[i] {
            if b.8 <= i {
                continue;
            }
            energy += SW_EPSILON * b.7;
            let inv_r = 1.0 / b.3;
            let fx = b.4 * b.0 * inv_r;
            let fy = b.4 * b.1 * inv_r;
            let fz = b.4 * b.2 * inv_r;
            force[3 * i] += fx;
            force[3 * i + 1] += fy;
            force[3 * i + 2] += fz;
            force[3 * b.8] -= fx;
            force[3 * b.8 + 1] -= fy;
            force[3 * b.8 + 2] -= fz;
            add_virial(&mut virial, -b.0, -b.1, -b.2, fx, fy, fz);
        }
        let m = bonds[i].len();
        for a in 0..m {
            for c in (a + 1)..m {
                let bj = bonds[i][a];
                let bk = bonds[i][c];
                let inv_r1 = 1.0 / bj.3;
                let inv_r2 = 1.0 / bk.3;
                let inv = inv_r1 * inv_r2;
                let mut cth = (bj.0 * bk.0 + bj.1 * bk.1 + bj.2 * bk.2) * inv;
                cth = cth.clamp(-1.0, 1.0);
                let ang = cth + 1.0 / 3.0;
                let ang2 = ang * ang;
                energy += SW_EPSILON * SW_LAMBDA * bj.5 * bk.5 * ang2;
                let pref = SW_EPSILON * SW_LAMBDA;
                let d_edcth = pref * bj.5 * bk.5 * 2.0 * ang;
                let d_edr1 = pref * bj.6 * bk.5 * ang2;
                let d_edr2 = pref * bj.5 * bk.6 * ang2;
                let g1 = [
                    d_edr1 * bj.0 * inv_r1 + d_edcth * (bk.0 * inv - cth * bj.0 * inv_r1 * inv_r1),
                    d_edr1 * bj.1 * inv_r1 + d_edcth * (bk.1 * inv - cth * bj.1 * inv_r1 * inv_r1),
                    d_edr1 * bj.2 * inv_r1 + d_edcth * (bk.2 * inv - cth * bj.2 * inv_r1 * inv_r1),
                ];
                let g2 = [
                    d_edr2 * bk.0 * inv_r2 + d_edcth * (bj.0 * inv - cth * bk.0 * inv_r2 * inv_r2),
                    d_edr2 * bk.1 * inv_r2 + d_edcth * (bj.1 * inv - cth * bk.1 * inv_r2 * inv_r2),
                    d_edr2 * bk.2 * inv_r2 + d_edcth * (bj.2 * inv - cth * bk.2 * inv_r2 * inv_r2),
                ];
                let j = bj.8;
                let k = bk.8;
                force[3 * j] -= g1[0];
                force[3 * j + 1] -= g1[1];
                force[3 * j + 2] -= g1[2];
                force[3 * k] -= g2[0];
                force[3 * k + 1] -= g2[1];
                force[3 * k + 2] -= g2[2];
                force[3 * i] += g1[0] + g2[0];
                force[3 * i + 1] += g1[1] + g2[1];
                force[3 * i + 2] += g1[2] + g2[2];
                add_virial(&mut virial, bj.0, bj.1, bj.2, -g1[0], -g1[1], -g1[2]);
                add_virial(&mut virial, bk.0, bk.1, bk.2, -g2[0], -g2[1], -g2[2]);
            }
        }
    }
    let volume = volume_of(cell);
    let mut stress = [0.0; 9];
    for i in 0..9 {
        stress[i] = -virial[i] / volume;
    }
    (energy, force, virial, stress)
}

fn sc_eval(pos: &[f64], cell: &[f64; 9]) -> (f64, Vec<f64>, [f64; 9], [f64; 9]) {
    let n = pos.len() / 3;
    let cut2 = SC_CUTOFF * SC_CUTOFF;
    let mut rho = vec![0.0; n];
    let mut energy = 0.0;
    let mut pairs = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let d = min_image(
                *cell,
                [
                    pos[3 * i] - pos[3 * j],
                    pos[3 * i + 1] - pos[3 * j + 1],
                    pos[3 * i + 2] - pos[3 * j + 2],
                ],
            );
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if r2 <= 0.0 || r2 > cut2 {
                continue;
            }
            let ar2 = (SC_A * SC_A) / r2;
            let phi = ar2.powf(SC_M as f64 / 2.0);
            let rep = ar2.powf(SC_N as f64 / 2.0);
            rho[i] += phi;
            rho[j] += phi;
            energy += SC_EPSILON * rep;
            pairs.push((i, j, d, r2, phi, rep));
        }
    }
    let mut weight = vec![0.0; n];
    for i in 0..n {
        if rho[i] > 0.0 {
            let root = rho[i].sqrt();
            energy -= SC_EPSILON * SC_C * root;
            weight[i] = SC_EPSILON * SC_C / (2.0 * root);
        }
    }
    let mut force = vec![0.0; 3 * n];
    let mut acc = [0.0; 6];
    for (i, j, d, r2, phi, rep) in pairs {
        let fscale =
            (SC_EPSILON * SC_N as f64 * rep - SC_M as f64 * phi * (weight[i] + weight[j])) / r2;
        let f = [fscale * d[0], fscale * d[1], fscale * d[2]];
        for k in 0..3 {
            force[3 * i + k] += f[k];
            force[3 * j + k] -= f[k];
        }
        acc[0] += -fscale * d[0] * d[0];
        acc[1] += -fscale * d[1] * d[1];
        acc[2] += -fscale * d[2] * d[2];
        acc[3] += -fscale * d[1] * d[2];
        acc[4] += -fscale * d[0] * d[2];
        acc[5] += -fscale * d[0] * d[1];
    }
    let volume = volume_of(cell);
    let virial = [
        -acc[0], -acc[5], -acc[4], -acc[5], -acc[1], -acc[3], -acc[4], -acc[3], -acc[2],
    ];
    let mut stress = [0.0; 9];
    for i in 0..9 {
        stress[i] = -virial[i] / volume;
    }
    (energy, force, virial, stress)
}

fn add_virial(w: &mut [f64; 9], rx: f64, ry: f64, rz: f64, fx: f64, fy: f64, fz: f64) {
    w[0] += rx * fx;
    w[1] += rx * fy;
    w[2] += rx * fz;
    w[3] += ry * fx;
    w[4] += ry * fy;
    w[5] += ry * fz;
    w[6] += rz * fx;
    w[7] += rz * fy;
    w[8] += rz * fz;
}

fn volume_of(cell: &[f64; 9]) -> f64 {
    let (a, b, c) = (&cell[0..3], &cell[3..6], &cell[6..9]);
    (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0]))
        .abs()
}

fn min_image(cell: [f64; 9], d: [f64; 3]) -> [f64; 3] {
    let m = [
        [cell[0], cell[1], cell[2]],
        [cell[3], cell[4], cell[5]],
        [cell[6], cell[7], cell[8]],
    ];
    let det = volume_of(&cell);
    let s = 1.0 / det;
    let inv = [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * s,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * s,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * s,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * s,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * s,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * s,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * s,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * s,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * s,
        ],
    ];
    // s = d_row * M^{-1} = M^{-T} d_col.
    let mut frac = [0.0; 3];
    for i in 0..3 {
        frac[i] = inv[0][i] * d[0] + inv[1][i] * d[1] + inv[2][i] * d[2];
        frac[i] -= frac[i].round();
    }
    [
        frac[0] * m[0][0] + frac[1] * m[1][0] + frac[2] * m[2][0],
        frac[0] * m[0][1] + frac[1] * m[1][1] + frac[2] * m[2][1],
        frac[0] * m[0][2] + frac[1] * m[1][2] + frac[2] * m[2][2],
    ]
}

fn diamond(a: f64) -> (Vec<f64>, [f64; 9]) {
    let basis = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [0.25, 0.25, 0.25],
        [0.75, 0.75, 0.25],
        [0.75, 0.25, 0.75],
        [0.25, 0.75, 0.75],
    ];
    let mut pos = Vec::new();
    for b in basis {
        pos.extend_from_slice(&[b[0] * a, b[1] * a, b[2] * a]);
    }
    (pos, [a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a])
}

fn fcc(a: f64) -> (Vec<f64>, [f64; 9]) {
    let basis = [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ];
    let mut pos = Vec::new();
    for b in basis {
        pos.extend_from_slice(&[b[0] * a, b[1] * a, b[2] * a]);
    }
    (pos, [a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a])
}

/// Pair-minimum diamond lattice constant. The three-body term is zero at
/// the tetrahedral angle, so the cubic crystal sits at the pair minimum.
fn sw_a0() -> f64 {
    let mut u = 1.12;
    for _ in 0..40 {
        let s = 1.0 / (u - SW_CUT);
        let inv = 1.0 / u;
        let um4 = inv.powi(4);
        let g = SW_B * um4 - 1.0;
        let es = s.exp();
        let dgdu = -SW_P * SW_B * um4 * inv;
        let df = SW_A * es * (dgdu - g * s * s);
        let h = 1e-6;
        let df_at = |uu: f64| {
            let s = 1.0 / (uu - SW_CUT);
            let inv = 1.0 / uu;
            let um4 = inv.powi(4);
            let g = SW_B * um4 - 1.0;
            let es = s.exp();
            let dgdu = -SW_P * SW_B * um4 * inv;
            SW_A * es * (dgdu - g * s * s)
        };
        u -= df / ((df_at(u + h) - df_at(u - h)) / (2.0 * h));
    }
    u * SW_SIGMA * 4.0 / 3.0_f64.sqrt()
}

fn mean_length(cell: &[f64; 9]) -> f64 {
    let rows = [
        (cell[0] * cell[0] + cell[1] * cell[1] + cell[2] * cell[2]).sqrt(),
        (cell[3] * cell[3] + cell[4] * cell[4] + cell[5] * cell[5]).sqrt(),
        (cell[6] * cell[6] + cell[7] * cell[7] + cell[8] * cell[8]).sqrt(),
    ];
    (rows[0] + rows[1] + rows[2]) / 3.0
}

fn max_abs_stress(stress: &[f64; 9], pressure: f64) -> f64 {
    let mut m = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            let target = if i == j { -pressure } else { 0.0 };
            m = m.max((stress[3 * i + j] - target).abs());
        }
    }
    m
}

struct Relax {
    mask: CellMask,
    pressure: f64,
    method: Method,
    gtol: f64,
    maxiter: usize,
}

fn relax(kernel: &Kernel, pos: &[f64], cell: [f64; 9], run: Relax) -> (f64, [f64; 9], [f64; 9]) {
    let n = pos.len() / 3;
    let filter = CellFilter::new(kernel, n, cell, None)
        .unwrap()
        .mask(run.mask)
        .pressure(run.pressure);
    let x0 = filter.coordinates(pos, &cell).unwrap();
    let control = Control {
        maxiter: run.maxiter,
        gtol: run.gtol,
        istep: 0.05,
        stop_norm: StopNorm::MaxAtom,
        ..Control::default()
    };
    let linesearch = LineSearch::Wolfe {
        c1: 1e-4,
        c2: 0.9,
        maxiter: 12,
    };
    let report = minimize_method(&filter, x0, &control, run.method, linesearch).unwrap();
    let (_, relaxed) = filter.geometry(report.coords.view()).unwrap();
    let stress = filter.stress(report.coords.view()).unwrap();
    (report.grad_norm, relaxed, stress)
}

#[test]
fn filter_gradient_matches_a_central_difference() {
    let kernel = Kernel::new(Kind::Sw);
    let (mut pos, mut cell) = diamond(5.50);
    cell[3] = 0.15;
    cell[7] = -0.1;
    pos[0] += 0.04;
    let filter = CellFilter::new(&kernel, 8, cell, None)
        .unwrap()
        .pressure(0.001);
    let x = filter.coordinates(&pos, &cell).unwrap();
    let (_h, g) = filter.enthalpy_gradient(x.view()).unwrap();
    let h = 1e-6;
    for k in [0usize, 4, 3 * 8, 3 * 8 + 4] {
        let mut plus = x.clone();
        let mut minus = x.clone();
        plus[k] += h;
        minus[k] -= h;
        let gp = filter.enthalpy_gradient(plus.view()).unwrap().0;
        let gm = filter.enthalpy_gradient(minus.view()).unwrap().0;
        let fd = (gp - gm) / (2.0 * h);
        assert!(
            (g[k] - fd).abs() < 1e-5,
            "component {k}: grad {} fd {fd}",
            g[k]
        );
    }
}

#[test]
fn silicon_relaxes_to_the_stillinger_weber_lattice() {
    let a0 = sw_a0();
    let kernel = Kernel::new(Kind::Sw);
    let (pos, mut cell) = diamond(5.55);
    cell[3] = 0.2;
    cell[6] = 0.05;
    cell[7] = -0.12;
    let lammps = lammps_sw_lattice(&pos, &cell);
    for method in [Method::lbfgs(), Method::Fire { kind: FireKind::V2 }] {
        let (_g, relaxed, stress) = relax(
            &kernel,
            &pos,
            cell,
            Relax {
                mask: CellMask::Full,
                pressure: 0.0,
                method: method.clone(),
                gtol: 2e-3,
                maxiter: 80,
            },
        );
        let a = mean_length(&relaxed);
        assert!(
            (a - a0).abs() < 2e-3,
            "{method:?} a {a} a0 {a0} stress {stress:?}"
        );
        assert!(
            (a - lammps).abs() < 5e-3,
            "{method:?} a {a} lammps {lammps}"
        );
        assert!(max_abs_stress(&stress, 0.0) < 5e-3, "{stress:?}");
    }
}

#[test]
fn tighter_stress_tolerance_lowers_the_residual() {
    let kernel = Kernel::new(Kind::Sw);
    let (pos, cell) = diamond(5.6);
    let loose = relax(
        &kernel,
        &pos,
        cell,
        Relax {
            mask: CellMask::FixedAngles,
            pressure: 0.0,
            method: Method::lbfgs(),
            gtol: 5e-2,
            maxiter: 8,
        },
    );
    let tight = relax(
        &kernel,
        &pos,
        cell,
        Relax {
            mask: CellMask::FixedAngles,
            pressure: 0.0,
            method: Method::lbfgs(),
            gtol: 1e-4,
            maxiter: 60,
        },
    );
    let loose_s = max_abs_stress(&loose.2, 0.0);
    let tight_s = max_abs_stress(&tight.2, 0.0);
    assert!(tight_s < loose_s, "tight {tight_s} loose {loose_s}");
    assert!(tight_s < 1e-3, "tight residual {tight_s}");
}

#[test]
fn copper_under_one_gigapascal_matches_the_enthalpy_fit() {
    let kernel = Kernel::new(Kind::Sc);
    let pressure = GPA;
    let (pos, cell) = fcc(3.70);
    let (_g, relaxed, stress) = relax(
        &kernel,
        &pos,
        cell,
        Relax {
            mask: CellMask::Hydrostatic,
            pressure,
            method: Method::lbfgs(),
            gtol: 5e-4,
            maxiter: 40,
        },
    );
    let volume = volume_of(&relaxed);
    let fitted = enthalpy_minimum(&kernel, pressure);
    assert!(
        (volume - fitted).abs() / fitted < 2e-3,
        "relaxed {volume} fit {fitted} stress {stress:?}"
    );
    assert!(max_abs_stress(&stress, pressure) < 2e-3, "{stress:?}");
}

/// Volume where a central difference of E(V) equals -p.
fn enthalpy_minimum(kernel: &Kernel, pressure: f64) -> f64 {
    let dedv = |a: f64| {
        let da = 1.0e-3;
        let energy = |scale: f64| {
            let (pos, cell) = fcc(scale);
            kernel.evaluate(&pos, &cell).unwrap().energy
        };
        let vp = volume_of(&fcc(a + da).1);
        let vm = volume_of(&fcc(a - da).1);
        (energy(a + da) - energy(a - da)) / (vp - vm)
    };
    let mut lo = 3.15;
    let mut hi = 3.55;
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        if dedv(mid) + pressure > 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    volume_of(&fcc(0.5 * (lo + hi)).1)
}

/// LAMMPS `box/relax` lattice constant of the same Stillinger-Weber cell.
fn lammps_sw_lattice(pos: &[f64], cell: &[f64; 9]) -> f64 {
    let dir = std::env::temp_dir().join("rgmin-sw-relax");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let n = pos.len() / 3;
    let mut data = format!(
        "SW relax\n\n{n} atoms\n1 atom types\n\n0 {lx} xlo xhi\n0 {ly} ylo yhi\n0 {lz} zlo zhi\n{xy} {xz} {yz} xy xz yz\n\nMasses\n\n1 28.0855\n\nAtoms # atomic\n\n",
        lx = cell[0],
        ly = cell[4],
        lz = cell[8],
        xy = cell[3],
        xz = cell[6],
        yz = cell[7],
    );
    for i in 0..n {
        data.push_str(&format!(
            "{} 1 {} {} {}\n",
            i + 1,
            pos[3 * i],
            pos[3 * i + 1],
            pos[3 * i + 2]
        ));
    }
    std::fs::write(dir.join("data.si"), data).unwrap();
    let sw = format!(
        "Si Si Si {SW_EPSILON:.16e} {SW_SIGMA:.16e} {SW_CUT:.16e} {SW_LAMBDA:.16e} {SW_GAMMA:.16e} {} {SW_A:.16e} {SW_B:.16e} {SW_P:.16e} 0 0\n",
        -1.0 / 3.0
    );
    std::fs::write(dir.join("Si.sw"), sw).unwrap();
    std::fs::write(
        dir.join("in.sw"),
        "units metal\natom_style atomic\nboundary p p p\nread_data data.si\npair_style sw\npair_coeff * * Si.sw Si\nmin_style cg\nminimize 0 1e-8 200 2000\nfix 1 all box/relax tri 0.0\nminimize 0 1e-10 400 4000\nvariable lx equal lx\nvariable ly equal ly\nvariable lz equal lz\nprint \"RGPOT_BOX ${lx} ${ly} ${lz}\"\n",
    )
    .unwrap();
    let script = "from lammps import lammps\nL = lammps(cmdargs=['-screen', 'none', '-log', 'none'])\nL.file('in.sw')\nprint('RGPOT_BOX', L.get_thermo('lx'), L.get_thermo('ly'), L.get_thermo('lz'))\n";
    std::fs::write(dir.join("run.py"), script).unwrap();
    let out = Command::new("python3")
        .arg("run.py")
        .current_dir(&dir)
        .output()
        .expect("python3");
    let text = String::from_utf8_lossy(&out.stdout);
    let err = String::from_utf8_lossy(&out.stderr);
    let line = text
        .lines()
        .chain(err.lines())
        .rev()
        .find(|l| l.contains("RGPOT_BOX"))
        .unwrap_or_else(|| panic!("LAMMPS failed\n{text}\n{err}"));
    let nums: Vec<f64> = line
        .split_whitespace()
        .skip(1)
        .filter_map(|s| s.parse().ok())
        .collect();
    assert_eq!(nums.len(), 3, "{line}");
    (nums[0] + nums[1] + nums[2]) / 3.0
}
