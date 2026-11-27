//! Sparse Lindh model Hessian, used as `H₀⁻¹ = P⁻¹`.
//!
//! Stretch and bend constants are Lindh, Bernhardsson, Karlström, and
//! Malmqvist, Chem. Phys. Lett. 241, 423 (1995),
//! <https://doi.org/10.1016/0009-2614(95)00646-L>, as tabulated by
//! Bakken and Helgaker, J. Chem. Phys. 117, 9160 (2002),
//! <https://doi.org/10.1063/1.1515483>. The Cartesian block of a
//! coordinate `ξ` is `k (∂ξ/∂x)(∂ξ/∂x)ᵀ`. Torsions are left out.
//!
//! Pairs come from a linked cell list, the same organisation as a
//! short-range potential: each atom is binned once and only the
//! neighbouring bins are searched. The cutoff is the distance at which
//! `ρ` falls through `STRETCH_RHO`, so the list is the same set an
//! all-pairs scan would have kept.
//!
//! `P⁻¹ rhs` is Jacobi-preconditioned conjugate gradients. The iteration
//! stops at [`crate::lindh::SOLVE_RTOL`] or after `SOLVE_MAXIT` steps, both
//! independent of the atom count, so one application is linear in the
//! number of atoms. An incomplete factor of this sparsity cost more
//! than the matvecs it saved at a few thousand atoms, so the solve
//! stays matrix-free.
//!
//! `ρᵢⱼ = exp(αᵢⱼ (r_ref² − r²))` with `r` in bohr. An atomic number
//! past argon uses the third row of the table. `c_stab I` keeps `P`
//! positive definite. `μ` scales it into the objective's units.

/// Å to bohr. `ρ` is defined on the bohr length in the Lindh table.
const BOHR_PER_ANGSTROM: f64 = 1.889_726_124_626;

const ALPHA: [[f64; 3]; 3] = [
    [1.0, 0.3949, 0.3949],
    [0.3949, 0.28, 0.28],
    [0.3949, 0.28, 0.28],
];

const RREF: [[f64; 3]; 3] = [[1.35, 2.10, 2.53], [2.10, 2.87, 3.40], [2.53, 3.40, 3.40]];

const K_STRETCH: f64 = 0.45;
const K_BEND: f64 = 0.15;

/// Drop stretches softer than this. The bend list uses a higher cut.
const STRETCH_RHO: f64 = 1.0e-4;
const BEND_RHO: f64 = 0.05;

/// Relative residual at which the preconditioner solve stops.
pub const SOLVE_RTOL: f64 = 1.0e-2;

/// Iteration cap. It does not grow with the number of atoms.
const SOLVE_MAXIT: usize = 32;

/// Atomic numbers and the ridge used to build the model.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct LindhSpec {
    /// Length `N`. Empty means every atom is period 1 (H, He).
    pub z: Vec<u8>,
    /// Diagonal ridge in the unscaled matrix. Must be positive.
    pub c_stab: f64,
    /// Periodic lengths. A non-positive entry is open.
    pub cell: [f64; 3],
}

impl LindhSpec {
    /// Atomic numbers `z`, ridge `c_stab`, and periodic lengths `cell`.
    pub fn new(z: Vec<u8>, c_stab: f64, cell: [f64; 3]) -> Self {
        Self { z, c_stab, cell }
    }
}

with_setters!(LindhSpec {
    with_z => z: val Vec<u8>,
    with_c_stab => c_stab: val f64,
    with_cell => cell: val [f64; 3],
});

/// Block-sparse model. `diag[i]` is the 3×3 diagonal block of atom `i`.
#[derive(Clone, Debug)]
pub(crate) struct LindhFactor {
    diag_block: Vec<[[f64; 3]; 3]>,
    offs: Vec<Vec<Off>>,
    /// Diagonal entries, length `3 N`, for Jacobi.
    diag: Vec<f64>,
}

/// `vᵀ P v` at `μ = 1`, including `c_stab`.
#[must_use]
pub(crate) fn shape_quad(x: &[f64], z: &[u8], c_stab: f64, cell: [f64; 3], v: &[f64]) -> f64 {
    if x.len() != v.len() || c_stab <= 0.0 {
        return 0.0;
    }
    let Some(model) = build_model(x, z, c_stab, cell, 1.0) else {
        return 0.0;
    };
    let mut pv = vec![0.0; v.len()];
    model.mul(v, &mut pv);
    dot(v, &pv)
}

/// Build `μ P` and replace `rhs` with an approximate `P⁻¹ rhs`.
///
/// False when the lengths differ or a scale is not positive. The residual
/// is at most [`SOLVE_RTOL`] times `‖rhs‖` when the iteration runs to
/// completion.
#[must_use]
pub fn apply_inverse(
    x: &[f64],
    z: &[u8],
    c_stab: f64,
    cell: [f64; 3],
    mu: f64,
    rhs: &mut [f64],
) -> bool {
    let Some(factor) = factor(x, z, c_stab, cell, mu) else {
        return false;
    };
    let mut q = ndarray::Array1::from(rhs.to_vec());
    if !factor.solve_into(&mut q) {
        return false;
    }
    let Some(solved) = q.as_slice() else {
        return false;
    };
    rhs.copy_from_slice(solved);
    true
}

/// Factor `μ P`. `None` only when the dimension is not a multiple of 3
/// or a scale is not positive. The factor is linear in the atom count.
#[must_use]
pub(crate) fn factor(
    x: &[f64],
    z: &[u8],
    c_stab: f64,
    cell: [f64; 3],
    mu: f64,
) -> Option<LindhFactor> {
    if !(mu.is_finite() && mu > 0.0 && c_stab.is_finite() && c_stab > 0.0) {
        return None;
    }
    build_model(x, z, c_stab, cell, mu)
}

impl LindhFactor {
    /// `P⁻¹ rhs`, to [`SOLVE_RTOL`]. False when the lengths differ or
    /// the iterate is non-finite.
    pub(crate) fn solve_into(&self, q: &mut ndarray::Array1<f64>) -> bool {
        self.solve_tol(q, SOLVE_RTOL, SOLVE_MAXIT)
    }

    fn solve_tol(&self, q: &mut ndarray::Array1<f64>, rtol: f64, maxit: usize) -> bool {
        let Some(slice) = q.as_slice_mut() else {
            return false;
        };
        solve_borrowed(self, slice, rtol, maxit).is_some() && q.iter().all(|v| v.is_finite())
    }
}

fn solve_borrowed(factor: &LindhFactor, q: &mut [f64], rtol: f64, maxit: usize) -> Option<()> {
    let n = factor.diag.len();
    if q.len() != n {
        return None;
    }
    let bnorm = nrm2(q);
    if bnorm == 0.0 {
        return Some(());
    }
    if factor.diag.iter().any(|d| !(d.is_finite() && *d > 0.0)) {
        return None;
    }
    let rhs = q.to_vec();
    let mut x = vec![0.0; n];
    let mut r = rhs.clone();
    let mut z = vec![0.0; n];
    let mut p = vec![0.0; n];
    let mut ap = vec![0.0; n];
    apply_precond(factor, &r, &mut z);
    p.copy_from_slice(&z);
    let mut rz = dot(&r, &z);
    if !(rz.is_finite() && rz > 0.0) {
        // Jacobi of a positive diagonal is the safe preconditioner.
        for i in 0..n {
            q[i] = rhs[i] / factor.diag[i];
        }
        return Some(());
    }
    let tol = rtol * bnorm;
    for _ in 0..maxit {
        if nrm2(&r) <= tol {
            q.copy_from_slice(&x);
            return Some(());
        }
        factor.mul(&p, &mut ap);
        let pap = dot(&p, &ap);
        if !(pap.is_finite() && pap > 0.0) {
            break;
        }
        let alpha = rz / pap;
        if !alpha.is_finite() {
            break;
        }
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        apply_precond(factor, &r, &mut z);
        let rz_new = dot(&r, &z);
        if !(rz_new.is_finite() && rz_new > 0.0) {
            break;
        }
        let beta = rz_new / rz;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        rz = rz_new;
    }
    if x.iter().all(|v| v.is_finite()) {
        q.copy_from_slice(&x);
        Some(())
    } else {
        None
    }
}

fn apply_precond(factor: &LindhFactor, r: &[f64], z: &mut [f64]) {
    for i in 0..r.len() {
        z[i] = r[i] / factor.diag[i];
    }
}

impl LindhFactor {
    #[allow(clippy::needless_range_loop)]
    fn mul(&self, x: &[f64], y: &mut [f64]) {
        let natoms = self.diag_block.len();
        for i in 0..natoms {
            let d = &self.diag_block[i];
            for a in 0..3 {
                let mut s = d[a][0] * x[3 * i] + d[a][1] * x[3 * i + 1] + d[a][2] * x[3 * i + 2];
                for off in &self.offs[i] {
                    let base = 3 * off.j;
                    s += off.h[a][0] * x[base]
                        + off.h[a][1] * x[base + 1]
                        + off.h[a][2] * x[base + 2];
                }
                y[3 * i + a] = s;
            }
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn nrm2(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

#[derive(Clone, Debug)]
struct Off {
    j: usize,
    h: [[f64; 3]; 3],
}

#[allow(clippy::needless_range_loop)]
fn add_block(offs: &mut Vec<Off>, j: usize, h: [[f64; 3]; 3]) {
    if let Some(slot) = offs.iter_mut().find(|o| o.j == j) {
        for a in 0..3 {
            for b in 0..3 {
                slot.h[a][b] += h[a][b];
            }
        }
    } else {
        offs.push(Off { j, h });
    }
}

fn add_stretch(
    diag: &mut [[[f64; 3]; 3]],
    offs: &mut [Vec<Off>],
    i: usize,
    j: usize,
    k: f64,
    u: [f64; 3],
) {
    for a in 0..3 {
        for b in 0..3 {
            let uu = k * u[a] * u[b];
            diag[i][a][b] += uu;
            diag[j][a][b] += uu;
        }
    }
    let mut neg = [[0.0; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            neg[a][b] = -k * u[a] * u[b];
        }
    }
    add_block(&mut offs[i], j, neg);
    add_block(&mut offs[j], i, neg);
}

fn build_model(x: &[f64], z: &[u8], c_stab: f64, cell: [f64; 3], mu: f64) -> Option<LindhFactor> {
    let n = x.len();
    let natoms = n / 3;
    if natoms == 0 || n != 3 * natoms || !(mu.is_finite() && mu > 0.0) {
        return None;
    }
    let bonds = cell_bonds(x, z, cell);
    let mut diag_b = vec![[[0.0; 3]; 3]; natoms];
    for block in &mut diag_b {
        block[0][0] = c_stab;
        block[1][1] = c_stab;
        block[2][2] = c_stab;
    }
    let mut offs = vec![Vec::new(); natoms];
    let mut bend: Vec<Vec<(usize, f64)>> = vec![Vec::new(); natoms];
    let scale = BOHR_PER_ANGSTROM * BOHR_PER_ANGSTROM;
    for (i, j, rho, u) in bonds {
        add_stretch(&mut diag_b, &mut offs, i, j, K_STRETCH * rho * scale, u);
        if rho > BEND_RHO {
            bend[i].push((j, rho));
            bend[j].push((i, rho));
        }
    }
    for (vertex, list) in bend.iter().enumerate() {
        for a in 0..list.len() {
            for b in (a + 1)..list.len() {
                add_angle_blocks(
                    &mut diag_b,
                    &mut offs,
                    x,
                    cell,
                    list[a].0,
                    vertex,
                    list[b].0,
                    K_BEND * list[a].1 * list[b].1,
                );
            }
        }
    }
    let mut diag = Vec::with_capacity(n);
    for block in &mut diag_b {
        for row in block.iter_mut() {
            for v in row.iter_mut() {
                *v *= mu;
            }
        }
        diag.push(block[0][0]);
        diag.push(block[1][1]);
        diag.push(block[2][2]);
    }
    for list in &mut offs {
        for off in list.iter_mut() {
            for row in &mut off.h {
                for v in row.iter_mut() {
                    *v *= mu;
                }
            }
        }
    }
    if diag.iter().any(|d| !(d.is_finite() && *d > 0.0)) {
        return None;
    }
    Some(LindhFactor {
        diag_block: diag_b,
        offs,
        diag,
    })
}

/// `(i, j, ρ, unit vector from i to j)` with `i < j`.
#[allow(clippy::needless_range_loop)]
fn cell_bonds(x: &[f64], z: &[u8], cell: [f64; 3]) -> Vec<(usize, usize, f64, [f64; 3])> {
    let natoms = x.len() / 3;
    let cutoff = stretch_cutoff(z);
    if !(cutoff.is_finite() && cutoff > 0.0) || natoms == 0 {
        return Vec::new();
    }
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for i in 0..natoms {
        for a in 0..3 {
            let p = x[3 * i + a];
            lo[a] = lo[a].min(p);
            hi[a] = hi[a].max(p);
        }
    }
    let periodic = [cell[0] > 0.0, cell[1] > 0.0, cell[2] > 0.0];
    let mut length = [0.0; 3];
    let mut origin = [0.0; 3];
    let mut ncell = [1usize; 3];
    for a in 0..3 {
        if periodic[a] {
            length[a] = cell[a];
            origin[a] = 0.0;
        } else {
            length[a] = (hi[a] - lo[a]).max(cutoff);
            origin[a] = lo[a];
        }
        let n = (length[a] / cutoff).floor() as usize;
        ncell[a] = n.clamp(1, 512);
    }
    let nbin = ncell[0] * ncell[1] * ncell[2];
    let inv = [
        ncell[0] as f64 / length[0],
        ncell[1] as f64 / length[1],
        ncell[2] as f64 / length[2],
    ];
    // Coordinates in the bin frame: periodic axes wrapped into the cell,
    // open axes shifted so the minimum sits at 0. The stencil image is
    // applied to these, not to the original values.
    let mut frame = vec![0.0; x.len()];
    for i in 0..natoms {
        for a in 0..3 {
            let mut t = x[3 * i + a];
            if periodic[a] {
                let c = cell[a];
                t -= c * (t / c).floor();
                if t < 0.0 {
                    t += c;
                }
            } else {
                t -= lo[a];
            }
            frame[3 * i + a] = t;
        }
    }
    let bin_periodic = [false; 3];
    let bin_origin = [0.0; 3];
    let mut head = vec![usize::MAX; nbin];
    let mut next = vec![usize::MAX; natoms];
    for i in 0..natoms {
        let b = bin_of(&frame, i, bin_origin, inv, ncell, bin_periodic, cell);
        next[i] = head[b];
        head[b] = i;
    }
    let cutoff2 = cutoff * cutoff;
    if natoms >= 2048 {
        return cell_bonds_parallel(
            &frame, z, cell, natoms, cutoff2, &head, &next, bin_origin, inv, ncell, periodic,
        );
    }
    let mut bonds = Vec::with_capacity(natoms.saturating_mul(8));
    for i in 0..natoms {
        let bi = bin_index(&frame, i, bin_origin, inv, ncell, bin_periodic, cell);
        for dz in -1isize..=1 {
            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    let off = [dx, dy, dz];
                    let mut nb = [0usize; 3];
                    let mut image = [0.0; 3];
                    let mut ok = true;
                    for a in 0..3 {
                        let raw = bi[a] + off[a];
                        if raw < 0 {
                            if !periodic[a] {
                                ok = false;
                                break;
                            }
                            nb[a] = (raw + ncell[a] as isize) as usize;
                            image[a] = -cell[a];
                        } else if raw >= ncell[a] as isize {
                            if !periodic[a] {
                                ok = false;
                                break;
                            }
                            nb[a] = (raw - ncell[a] as isize) as usize;
                            image[a] = cell[a];
                        } else {
                            nb[a] = raw as usize;
                        }
                    }
                    if !ok {
                        continue;
                    }
                    let b = (nb[0] * ncell[1] + nb[1]) * ncell[2] + nb[2];
                    let mut j = head[b];
                    while j != usize::MAX {
                        if j > i {
                            let d0 = frame[3 * j] + image[0] - frame[3 * i];
                            let d1 = frame[3 * j + 1] + image[1] - frame[3 * i + 1];
                            let d2 = frame[3 * j + 2] + image[2] - frame[3 * i + 2];
                            let r2 = d0 * d0 + d1 * d1 + d2 * d2;
                            if r2 > 1.0e-16 && r2 <= cutoff2 {
                                push_bond(z, i, j, [d0, d1, d2], r2, &mut bonds);
                            }
                        }
                        j = next[j];
                    }
                }
            }
        }
    }
    bonds
}

#[allow(clippy::too_many_arguments)]
fn cell_bonds_parallel(
    x: &[f64],
    z: &[u8],
    cell: [f64; 3],
    natoms: usize,
    cutoff2: f64,
    head: &[usize],
    next: &[usize],
    origin: [f64; 3],
    inv: [f64; 3],
    ncell: [usize; 3],
    periodic: [bool; 3],
) -> Vec<(usize, usize, f64, [f64; 3])> {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, 8);
    let chunk = natoms.div_ceil(threads);
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for t in 0..threads {
            let start = t * chunk;
            let end = (start + chunk).min(natoms);
            if start >= end {
                continue;
            }
            handles.push(scope.spawn(move || {
                let mut bonds = Vec::new();
                for i in start..end {
                    let bi = bin_index(x, i, origin, inv, ncell, periodic, cell);
                    for dz in -1isize..=1 {
                        for dy in -1isize..=1 {
                            for dx in -1isize..=1 {
                                let off = [dx, dy, dz];
                                let mut nb = [0usize; 3];
                                let mut image = [0.0; 3];
                                let mut ok = true;
                                for a in 0..3 {
                                    let raw = bi[a] + off[a];
                                    if raw < 0 {
                                        if !periodic[a] {
                                            ok = false;
                                            break;
                                        }
                                        nb[a] = (raw + ncell[a] as isize) as usize;
                                        image[a] = -cell[a];
                                    } else if raw >= ncell[a] as isize {
                                        if !periodic[a] {
                                            ok = false;
                                            break;
                                        }
                                        nb[a] = (raw - ncell[a] as isize) as usize;
                                        image[a] = cell[a];
                                    } else {
                                        nb[a] = raw as usize;
                                    }
                                }
                                if !ok {
                                    continue;
                                }
                                let b = (nb[0] * ncell[1] + nb[1]) * ncell[2] + nb[2];
                                let mut j = head[b];
                                while j != usize::MAX {
                                    if j > i {
                                        let d0 = x[3 * j] + image[0] - x[3 * i];
                                        let d1 = x[3 * j + 1] + image[1] - x[3 * i + 1];
                                        let d2 = x[3 * j + 2] + image[2] - x[3 * i + 2];
                                        let r2 = d0 * d0 + d1 * d1 + d2 * d2;
                                        if r2 > 1.0e-16 && r2 <= cutoff2 {
                                            push_bond(z, i, j, [d0, d1, d2], r2, &mut bonds);
                                        }
                                    }
                                    j = next[j];
                                }
                            }
                        }
                    }
                }
                bonds
            }));
        }
        let mut bonds = Vec::new();
        for handle in handles {
            bonds.extend(handle.join().unwrap_or_default());
        }
        bonds
    })
}

fn push_bond(
    z: &[u8],
    i: usize,
    j: usize,
    d: [f64; 3],
    d2: f64,
    bonds: &mut Vec<(usize, usize, f64, [f64; 3])>,
) {
    let r = d2.sqrt();
    let r_bohr = r * BOHR_PER_ANGSTROM;
    let alpha = ALPHA[row_of(z, i)][row_of(z, j)];
    let rref = RREF[row_of(z, i)][row_of(z, j)];
    let rho = (alpha * (rref * rref - r_bohr * r_bohr)).exp();
    if rho.is_finite() && rho > STRETCH_RHO {
        let inv = 1.0 / r;
        bonds.push((i, j, rho, [d[0] * inv, d[1] * inv, d[2] * inv]));
    }
}

fn stretch_cutoff(z: &[u8]) -> f64 {
    let mut rows = [false; 3];
    if z.is_empty() {
        rows[0] = true;
    } else {
        for &zi in z {
            rows[row_of_z(zi)] = true;
        }
    }
    let mut max_r = 0.0_f64;
    for (i, row_i) in rows.iter().enumerate() {
        if !row_i {
            continue;
        }
        for (j, row_j) in rows.iter().enumerate() {
            if !row_j {
                continue;
            }
            let alpha = ALPHA[i][j];
            let rref = RREF[i][j];
            let r2 = rref * rref - STRETCH_RHO.ln() / alpha;
            if r2 > 0.0 {
                max_r = max_r.max(r2.sqrt() / BOHR_PER_ANGSTROM);
            }
        }
    }
    max_r * 1.001 + 1.0e-8
}

fn bin_of(
    x: &[f64],
    i: usize,
    origin: [f64; 3],
    inv: [f64; 3],
    ncell: [usize; 3],
    periodic: [bool; 3],
    cell: [f64; 3],
) -> usize {
    let b = bin_index(x, i, origin, inv, ncell, periodic, cell);
    (b[0] as usize * ncell[1] + b[1] as usize) * ncell[2] + b[2] as usize
}

fn bin_index(
    x: &[f64],
    i: usize,
    origin: [f64; 3],
    inv: [f64; 3],
    ncell: [usize; 3],
    periodic: [bool; 3],
    cell: [f64; 3],
) -> [isize; 3] {
    let mut b = [0isize; 3];
    for a in 0..3 {
        let mut t = x[3 * i + a] - origin[a];
        if periodic[a] {
            let c = cell[a];
            t -= c * (t / c).floor();
            if t < 0.0 {
                t += c;
            }
        }
        let mut k = (t * inv[a]).floor() as isize;
        if k < 0 {
            k = 0;
        }
        if k >= ncell[a] as isize {
            k = ncell[a] as isize - 1;
        }
        b[a] = k;
    }
    b
}

#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn add_angle_blocks(
    diag: &mut [[[f64; 3]; 3]],
    offs: &mut [Vec<Off>],
    x: &[f64],
    cell: [f64; 3],
    i: usize,
    j: usize,
    k: usize,
    kconst: f64,
) {
    if !(kconst.is_finite() && kconst > 0.0) {
        return;
    }
    let a = delta(x, j, i, cell);
    let b = delta(x, j, k, cell);
    let la = norm(&a);
    let lb = norm(&b);
    if la < 1e-8 || lb < 1e-8 {
        return;
    }
    let c = (dot3(&a, &b) / (la * lb)).clamp(-1.0 + 1.0e-12, 1.0 - 1.0e-12);
    let sin = (1.0 - c * c).sqrt();
    if sin < 1e-6 {
        return;
    }
    let inv = 1.0 / (la * lb);
    let mut gi = [0.0; 3];
    let mut gk = [0.0; 3];
    for t in 0..3 {
        let dc_i = b[t] * inv - c * a[t] / (la * la);
        let dc_k = a[t] * inv - c * b[t] / (lb * lb);
        gi[t] = -dc_i / sin;
        gk[t] = -dc_k / sin;
    }
    let gj = [-gi[0] - gk[0], -gi[1] - gk[1], -gi[2] - gk[2]];
    let atoms = [i, j, k];
    let g = [gi, gj, gk];
    for p in 0..3 {
        for q in 0..3 {
            let mut h = [[0.0; 3]; 3];
            for ia in 0..3 {
                for ib in 0..3 {
                    h[ia][ib] = kconst * g[p][ia] * g[q][ib];
                }
            }
            if p == q {
                for ia in 0..3 {
                    for ib in 0..3 {
                        diag[atoms[p]][ia][ib] += h[ia][ib];
                    }
                }
            } else {
                add_block(&mut offs[atoms[p]], atoms[q], h);
            }
        }
    }
}

fn row_of(z: &[u8], atom: usize) -> usize {
    if atom < z.len() { row_of_z(z[atom]) } else { 0 }
}

fn row_of_z(z: u8) -> usize {
    match z {
        0..=2 => 0,
        3..=10 => 1,
        _ => 2,
    }
}

fn delta(x: &[f64], i: usize, j: usize, cell: [f64; 3]) -> [f64; 3] {
    let mut d = [0.0; 3];
    for a in 0..3 {
        let mut v = x[3 * j + a] - x[3 * i + a];
        if cell[a] > 0.0 {
            v -= cell[a] * (v / cell[a]).round();
        }
        d[a] = v;
    }
    d
}

fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: &[f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    fn brute_pairs(x: &[f64], z: &[u8], cell: [f64; 3]) -> BTreeSet<(usize, usize)> {
        let cutoff = stretch_cutoff(z);
        let n = x.len() / 3;
        let mut out = BTreeSet::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let d = delta(x, i, j, cell);
                let r = norm(&d);
                if r <= 1e-8 || r > cutoff {
                    continue;
                }
                let r_bohr = r * BOHR_PER_ANGSTROM;
                let alpha = ALPHA[row_of(z, i)][row_of(z, j)];
                let rref = RREF[row_of(z, i)][row_of(z, j)];
                let rho = (alpha * (rref * rref - r_bohr * r_bohr)).exp();
                if rho > STRETCH_RHO {
                    out.insert((i, j));
                }
            }
        }
        out
    }

    #[test]
    fn the_cell_list_matches_the_all_pairs_scan() {
        let mut x = Vec::new();
        for i in 0..12 {
            x.extend_from_slice(&[
                0.4 * (i % 3) as f64,
                0.5 * ((i / 3) % 2) as f64,
                0.3 * (i / 6) as f64,
            ]);
        }
        // A neighbour across the periodic boundary.
        x[0] = 0.05;
        x[3] = 3.95;
        let z = vec![1u8; 12];
        let cell = [4.0, 0.0, 0.0];
        let brute = brute_pairs(&x, &z, cell);
        let list = cell_bonds(&x, &z, cell);
        let got: BTreeSet<_> = list.iter().map(|(i, j, _, _)| (*i, *j)).collect();
        assert_eq!(list.len(), got.len(), "duplicate cell-list pairs");
        assert_eq!(got, brute);
        assert!(got.contains(&(0, 1)), "periodic pair missing: {got:?}");
    }

    #[test]
    fn the_cell_list_matches_a_periodic_slab() {
        let a = 2.556 * std::f64::consts::SQRT_2;
        let (nx, ny, nz) = (6usize, 6, 3);
        let mut x = Vec::new();
        let basis = [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ];
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    for b in basis {
                        let p = [
                            (ix as f64 + b[0]) * a,
                            (iy as f64 + b[1]) * a,
                            (iz as f64 + b[2]) * a,
                        ];
                        let k = x.len() / 3;
                        for (c, v) in p.iter().enumerate() {
                            x.push(v + 0.04 * ((k * 3 + c) as f64 * 1.7).sin());
                        }
                    }
                }
            }
        }
        let z = vec![29u8; x.len() / 3];
        let cell = [nx as f64 * a, ny as f64 * a, 0.0];
        let brute = brute_pairs(&x, &z, cell);
        let list = cell_bonds(&x, &z, cell);
        let got: BTreeSet<_> = list.iter().map(|(i, j, _, _)| (*i, *j)).collect();
        assert_eq!(list.len(), got.len(), "duplicate pairs");
        if got.len() != brute.len() {
            let missing: Vec<_> = brute.difference(&got).take(3).copied().collect::<Vec<_>>();
            let (i, j) = missing[0];
            let d = delta(&x, i, j, cell);
            let r = norm(&d);
            panic!(
                "cell {} brute {} missing {:?} r {r} i {:?} j {:?}",
                got.len(),
                brute.len(),
                missing,
                &x[3 * i..3 * i + 3],
                &x[3 * j..3 * j + 3]
            );
        }
        assert_eq!(got, brute);
    }

    #[test]
    fn a_tight_solve_matches_a_dense_product() {
        let x = [0.0, 0.0, 0.0, 0.74, 0.0, 0.0, 0.2, 0.9, 0.05];
        let z = [1u8, 1, 8];
        let factor = factor(&x, &z, 0.1, [0.0; 3], 1.7).unwrap();
        let rhs = [0.2, -0.1, 0.05, 0.3, 0.0, -0.2, 0.1, 0.4, -0.3];
        let mut q = ndarray::Array1::from(rhs.to_vec());
        assert!(factor.solve_tol(&mut q, 1e-10, 64));
        let mut pv = vec![0.0; 9];
        factor.mul(q.as_slice().unwrap(), &mut pv);
        for i in 0..9 {
            assert!(
                (pv[i] - rhs[i]).abs() < 1e-6,
                "{i}: {} vs {}",
                pv[i],
                rhs[i]
            );
        }
    }

    #[test]
    fn the_loose_solve_stops_inside_its_tolerance() {
        let x = [0.0, 0.0, 0.0, 0.74, 0.0, 0.0, 0.2, 0.9, 0.05];
        let factor = factor(&x, &[1, 1, 8], 0.1, [0.0; 3], 1.0).unwrap();
        let rhs = [0.3, -0.2, 0.1, 0.0, 0.4, -0.1, 0.2, 0.0, -0.3];
        let mut q = ndarray::Array1::from(rhs.to_vec());
        assert!(factor.solve_into(&mut q));
        let mut pv = vec![0.0; rhs.len()];
        factor.mul(q.as_slice().unwrap(), &mut pv);
        let mut r2 = 0.0;
        let mut b2 = 0.0;
        for i in 0..rhs.len() {
            let d = pv[i] - rhs[i];
            r2 += d * d;
            b2 += rhs[i] * rhs[i];
        }
        assert!(
            r2.sqrt() <= SOLVE_RTOL * b2.sqrt() * 1.01,
            "residual {} tol {}",
            r2.sqrt(),
            SOLVE_RTOL * b2.sqrt()
        );
    }

    #[test]
    fn a_bond_is_stiffer_than_a_perpendicular_shift() {
        let x = [0.0, 0.0, 0.0, 0.74, 0.0, 0.0];
        let along = [-1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let perp = [0.0, -1.0, 0.0, 0.0, 1.0, 0.0];
        let qa = shape_quad(&x, &[1, 1], 0.05, [0.0; 3], &along);
        let qp = shape_quad(&x, &[1, 1], 0.05, [0.0; 3], &perp);
        assert!(qa > qp * 2.0, "bond {qa} perp {qp}");
    }

    #[test]
    fn a_lone_atom_solves_as_the_ridge() {
        let x = [0.0, 0.0, 0.0];
        let f = factor(&x, &[1], 2.0, [0.0; 3], 1.0).unwrap();
        let mut q = ndarray::Array1::from(vec![2.0, 0.0, -2.0]);
        assert!(f.solve_into(&mut q));
        assert!((q[0] - 1.0).abs() < 1e-8, "{}", q[0]);
        assert!(q[1].abs() < 1e-8, "{}", q[1]);
        assert!((q[2] + 1.0).abs() < 1e-8, "{}", q[2]);
    }

    #[test]
    fn a_wide_lattice_still_factors() {
        let side = 16usize;
        let n = side * side * side;
        let mut x = Vec::with_capacity(3 * n);
        for iz in 0..side {
            for iy in 0..side {
                for ix in 0..side {
                    x.push(1.2 * ix as f64);
                    x.push(1.2 * iy as f64);
                    x.push(1.2 * iz as f64);
                }
            }
        }
        let f = factor(&x, &[], 0.1, [0.0; 3], 1.0);
        assert!(f.is_some(), "cell-list factor refused {n} atoms");
        let f = f.unwrap();
        let mut q = ndarray::Array1::zeros(x.len());
        q[0] = 1.0;
        q[3] = -0.5;
        assert!(f.solve_into(&mut q));
        assert!(q.iter().all(|v| v.is_finite()));
    }
}
