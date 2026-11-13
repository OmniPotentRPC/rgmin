//! Horizontal space of \(R^{3N}/\mathrm{SE}(3)\).
//!
//! Sella Cartesian PES (`fix_translation` + `fix_rotation`; Hermes,
//! Sarsfield, Zádor, JCTC 2022, doi:10.1021/acs.jctc.2c00395). Masses
//! turn the inner product into the Page–McIver / Eckart metric used
//! by Sella IRC and gpr_optim `IRCDriver` (Page and McIver 1988,
//! doi:10.1063/1.454172).
//!
//! The six generators are the translations and \(\omega \times (r - r_{\mathrm{com}})\).
//! A frozen atom cannot move, so a generator is admissible only in
//! combinations that are already zero on frozen atoms. Those columns are
//! zeroed and only then ranked. Column-pivoted Householder QR decides the
//! rank from the diagonal of \(R\): an entry is kept when it exceeds
//! \(\max(3N, k)\,ε\,\max(|R_{11}|, \lVert x \rVert_\infty)\). The leading
//! columns of \(Q\) are the orthonormal basis that is projected out.
//!
//! Pivoted QR is the factorization that fits. The matrix is \(3N \times k\)
//! with \(k \le 6\), so one thin QR both reveals the rank and orthonormalizes
//! the kept columns. A Gram SVD on \(A^\top A\) squares every singular value
//! and will drop a real rotation of a near-linear molecule, or keep a
//! roundoff column, because the gap is tested after that squaring. A thin
//! SVD of \(A\) itself is a sharper rank test, but on six columns the gap is
//! either roundoff or a geometric lever arm, and the same relative cutoff
//! on the largest singular value decides it. QR already returns the basis.

use ndarray::{Array1, ArrayView1};

/// eOn `projectOutRotTrans`. Unit-mass Eckart. Isolated molecule (T+R).
pub(crate) fn project_out_rot_trans(
    vec: &mut Array1<f64>,
    pos: ArrayView1<f64>,
    frozen: Option<&[bool]>,
) {
    project_horizontal(vec, pos, None, true, frozen);
}

/// Horizontal space of \(R^{3N}/\mathrm{SE}(3)\) or \(R^{3N}/T(3)\).
///
/// `rotate` is Sella `proj_rot`: false under PBC (the cell kills
/// rotational invariance). Masses are per atom (length N). `frozen`
/// is per atom; a frozen atom is zero in every generator before the
/// rank test, and stays zero in `vec`.
pub(crate) fn project_horizontal(
    vec: &mut Array1<f64>,
    pos: ArrayView1<f64>,
    masses: Option<&[f64]>,
    rotate: bool,
    frozen: Option<&[bool]>,
) {
    let n = pos.len();
    if n < 3 || !n.is_multiple_of(3) || vec.len() != n {
        return;
    }
    let nat = n / 3;
    let frozen = frozen.filter(|mask| mask.len() == nat);
    zero_frozen_array(vec, frozen, nat);

    let k = if rotate { 6 } else { 3 };
    let com = centre_of_mass(pos, masses, nat);
    let mut cols = rigid_columns(pos, nat, rotate, com);
    let coord = coordinate_scale(pos);
    cols = admissible(cols, n, k, frozen, nat, coord);
    let ncols = if cols.is_empty() { 0 } else { cols.len() / n };
    if ncols == 0 {
        zero_frozen_array(vec, frozen, nat);
        return;
    }
    let scale = mass_scale(masses, nat, n);
    for j in 0..ncols {
        for i in 0..n {
            cols[j * n + i] *= scale[i];
        }
    }
    let qr = col_pivoted_qr(cols, n, ncols, coord);
    let q = thin_q(&qr);
    apply_basis(vec, &q, &scale, qr.rank);
    zero_frozen_array(vec, frozen, nat);
}

/// Mass-weighted centre of mass. Unit mass when `masses` is absent.
fn centre_of_mass(pos: ArrayView1<f64>, masses: Option<&[f64]>, nat: usize) -> [f64; 3] {
    let mut com = [0.0; 3];
    let mut total = 0.0;
    for atom in 0..nat {
        let mass = masses
            .filter(|m| m.len() == nat)
            .map(|m| m[atom].max(0.0))
            .unwrap_or(1.0);
        total += mass;
        com[0] += mass * pos[3 * atom];
        com[1] += mass * pos[3 * atom + 1];
        com[2] += mass * pos[3 * atom + 2];
    }
    if total > 0.0 {
        com[0] /= total;
        com[1] /= total;
        com[2] /= total;
    }
    com
}

fn coordinate_scale(pos: ArrayView1<f64>) -> f64 {
    pos.iter().fold(0.0, |scale, value| scale.max(value.abs()))
}

/// Translations, then \(\omega \times (r - r_{\mathrm{com}})\).
///
/// Centering is required. \(\omega \times r\) at a far origin is a
/// translation of size \(\lVert r \rVert\) plus a rotation of molecular
/// size, and column-pivoted QR then loses a real mode in that
/// cancellation. The column space is the same: \(\omega \times r_{\mathrm{com}}\)
/// is a translation.
fn rigid_columns(pos: ArrayView1<f64>, nat: usize, rotate: bool, com: [f64; 3]) -> Vec<f64> {
    let n = nat * 3;
    let k = if rotate { 6 } else { 3 };
    let mut a = vec![0.0; n * k];
    for axis in 0..3 {
        for atom in 0..nat {
            a[axis * n + 3 * atom + axis] = 1.0;
        }
    }
    if rotate {
        for atom in 0..nat {
            let x = pos[3 * atom] - com[0];
            let y = pos[3 * atom + 1] - com[1];
            let z = pos[3 * atom + 2] - com[2];
            let base = 3 * atom;
            // ω = e_x, e_y, e_z.
            a[3 * n + base + 1] = -z;
            a[3 * n + base + 2] = y;
            a[4 * n + base] = z;
            a[4 * n + base + 2] = -x;
            a[5 * n + base] = -y;
            a[5 * n + base + 1] = x;
        }
    }
    a
}

/// Columns of `A Z`, where `Z` spans the kernel of the frozen rows of `A`.
///
/// With no frozen atom this is `A`. Otherwise the frozen rows are a
/// constraint matrix: a rigid motion that moves a frozen atom is not a
/// motion of the free system. The kernel combinations are zero on those
/// atoms by construction; they are zeroed again so the roundoff of the
/// combination cannot inflate the rank that follows.
fn admissible(
    cols: Vec<f64>,
    n: usize,
    k: usize,
    frozen: Option<&[bool]>,
    nat: usize,
    coord: f64,
) -> Vec<f64> {
    let Some(mask) = frozen else {
        return cols;
    };
    let mut rows = Vec::new();
    for (atom, &is_frozen) in mask.iter().enumerate().take(nat) {
        if is_frozen {
            rows.extend([3 * atom, 3 * atom + 1, 3 * atom + 2]);
        }
    }
    if rows.is_empty() {
        return cols;
    }
    let mf = rows.len();
    let mut constraint = vec![0.0; mf * k];
    for j in 0..k {
        for (r, &i) in rows.iter().enumerate() {
            constraint[j * mf + r] = cols[j * n + i];
        }
    }
    let qr = col_pivoted_qr(constraint, mf, k, coord);
    let nullity = k - qr.rank;
    if nullity == 0 {
        return Vec::new();
    }
    let z = kernel_basis(&qr);
    let mut kept = vec![0.0; n * nullity];
    for j in 0..nullity {
        for t in 0..k {
            let coef = z[j * k + t];
            if coef == 0.0 {
                continue;
            }
            for i in 0..n {
                kept[j * n + i] += coef * cols[t * n + i];
            }
        }
        for &i in &rows {
            kept[j * n + i] = 0.0;
        }
    }
    kept
}

/// `sqrt(m)` on each coordinate. Unit mass when `masses` is absent or
/// the wrong length. A non-positive mass contributes nothing to the
/// Eckart product.
fn mass_scale(masses: Option<&[f64]>, nat: usize, n: usize) -> Vec<f64> {
    let mut scale = vec![1.0; n];
    let Some(masses) = masses.filter(|m| m.len() == nat) else {
        return scale;
    };
    for atom in 0..nat {
        let s = masses[atom].max(0.0).sqrt();
        scale[3 * atom] = s;
        scale[3 * atom + 1] = s;
        scale[3 * atom + 2] = s;
    }
    scale
}

fn zero_frozen_array(v: &mut Array1<f64>, frozen: Option<&[bool]>, nat: usize) {
    let Some(mask) = frozen else {
        return;
    };
    for (atom, &is_frozen) in mask.iter().enumerate().take(nat) {
        if is_frozen {
            for c in 0..3 {
                let i = 3 * atom + c;
                if i < v.len() {
                    v[i] = 0.0;
                }
            }
        }
    }
}

/// Subtract the mass-orthonormal columns of `q` from `vec`.
///
/// `q` is orthonormal in the scaled chart `w = sqrt(m) v`. Dividing
/// back by `sqrt(m)` is the Euclidean representation of that chart, so
/// the subtraction is the Eckart (mass) projection.
fn apply_basis(vec: &mut Array1<f64>, q: &[f64], scale: &[f64], rank: usize) {
    let n = vec.len();
    if rank == 0 || n == 0 {
        return;
    }
    let mut w = vec![0.0; n];
    for i in 0..n {
        w[i] = vec[i] * scale[i];
    }
    for j in 0..rank {
        let col = j * n;
        let mut dot = 0.0;
        for i in 0..n {
            dot += w[i] * q[col + i];
        }
        for i in 0..n {
            w[i] -= dot * q[col + i];
        }
    }
    for i in 0..n {
        if scale[i] > 0.0 {
            vec[i] = w[i] / scale[i];
        }
    }
}

/// Column-pivoted Householder QR of a column-major `m × n` matrix.
struct Qr {
    /// `R` in the upper triangle, Householder vectors below the diagonal.
    a: Vec<f64>,
    tau: Vec<f64>,
    /// `pivot[j]` is the original column now stored at index `j`.
    pivot: Vec<usize>,
    m: usize,
    n: usize,
    /// Leading diagonals of `R` above the relative cutoff.
    rank: usize,
}

/// Rank-revealing QR.
///
/// At step `k` the remaining column of largest norm from row `k` down is
/// swapped in (Golub and Van Loan, Matrix Computations, algorithm 5.4.1).
/// A diagonal entry is kept when
/// `|R_ii| > max(m, n) · ε · max(|R_11|, coord_scale)`. `|R_11|` is the
/// largest diagonal after pivoting. `coord_scale` is `‖x‖_∞`, the absolute
/// error of a centred coordinate. `max(m, n) ε` is the dimension factor
/// `numpy.linalg.matrix_rank` uses on the largest singular value.
#[allow(
    clippy::needless_range_loop,
    reason = "k indexes the column, the pivot and the reflector together"
)]
fn col_pivoted_qr(mut a: Vec<f64>, m: usize, n: usize, coord_scale: f64) -> Qr {
    let mut pivot: Vec<usize> = (0..n).collect();
    let mut tau = vec![0.0; n];
    let steps = m.min(n);
    for k in 0..steps {
        let mut best_j = k;
        let mut best = -1.0;
        for j in k..n {
            let mut nrm2 = 0.0;
            for i in k..m {
                let v = a[j * m + i];
                nrm2 += v * v;
            }
            if nrm2 > best {
                best = nrm2;
                best_j = j;
            }
        }
        if best_j != k {
            swap_columns(&mut a, m, k, best_j);
            pivot.swap(k, best_j);
        }
        tau[k] = householder_column(&mut a, m, k);
        for j in k + 1..n {
            apply_reflector(&mut a, m, k, tau[k], j);
        }
    }
    let mut rmax = 0.0_f64;
    for i in 0..steps {
        rmax = rmax.max(a[i * m + i].abs());
    }
    // `|R_11|` is the largest diagonal after pivoting. Centering pulls
    // that below the absolute error of `x - x_com`, which is set by
    // `‖x‖_∞`. The cutoff uses the larger of the two, then the usual
    // factor `max(m, n) ε`. A far cluster would otherwise keep the
    // roundoff column of a null rotation, and an uncentered generator
    // would drop a real translation instead.
    let tol = (m.max(n) as f64) * f64::EPSILON * rmax.max(coord_scale);
    let mut rank = 0;
    while rank < steps && a[rank * m + rank].abs() > tol {
        rank += 1;
    }
    Qr {
        a,
        tau,
        pivot,
        m,
        n,
        rank,
    }
}

/// LAPACK `dlarfg` on the tail of column `k`, which is contiguous in
/// column-major storage. `v_0 = 1` is implicit; the rest of `v` overwrites
/// the column below the diagonal, and the diagonal becomes `R_kk`.
fn householder_column(a: &mut [f64], m: usize, k: usize) -> f64 {
    let start = k * m + k;
    householder(&mut a[start..k * m + m])
}

fn householder(x: &mut [f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let mut sigma = 0.0;
    for v in x.iter().skip(1) {
        sigma += v * v;
    }
    let alpha = x[0];
    if sigma == 0.0 {
        if alpha >= 0.0 {
            return 0.0;
        }
        x[0] = -alpha;
        return 2.0;
    }
    let beta = if alpha >= 0.0 {
        -alpha.hypot(sigma.sqrt())
    } else {
        alpha.hypot(sigma.sqrt())
    };
    let tau = (beta - alpha) / beta;
    let inv = 1.0 / (alpha - beta);
    for v in x.iter_mut().skip(1) {
        *v *= inv;
    }
    x[0] = beta;
    tau
}

fn apply_reflector(a: &mut [f64], m: usize, k: usize, tau: f64, j: usize) {
    if tau == 0.0 {
        return;
    }
    let col_k = k * m;
    let col_j = j * m;
    let mut dot = a[col_j + k];
    for i in k + 1..m {
        dot += a[col_k + i] * a[col_j + i];
    }
    let t = tau * dot;
    a[col_j + k] -= t;
    for i in k + 1..m {
        a[col_j + i] -= t * a[col_k + i];
    }
}

fn swap_columns(a: &mut [f64], m: usize, j: usize, k: usize) {
    if j == k {
        return;
    }
    for i in 0..m {
        a.swap(j * m + i, k * m + i);
    }
}

/// Orthonormal columns of `Q` for the kept rank. Reflectors are applied
/// from the last down, so `Q = H_1 H_2 … H_r I_{:, 1:r}`.
fn thin_q(qr: &Qr) -> Vec<f64> {
    let m = qr.m;
    let rank = qr.rank;
    let mut q = vec![0.0; m * rank];
    for j in 0..rank {
        q[j * m + j] = 1.0;
    }
    for k in (0..rank).rev() {
        let tau = qr.tau[k];
        if tau == 0.0 {
            continue;
        }
        for j in 0..rank {
            let col = j * m;
            let mut dot = q[col + k];
            for i in k + 1..m {
                dot += qr.a[k * m + i] * q[col + i];
            }
            let t = tau * dot;
            q[col + k] -= t;
            for i in k + 1..m {
                q[col + i] -= t * qr.a[k * m + i];
            }
        }
    }
    q
}

/// Basis of `ker(B)` from the pivoted QR of `B`, one column per free variable.
///
/// With rank `r`, `R_11 y = R_12[:, j]` and the pivoted null vector is
/// `(-y, e_j)`. Unpermuting by the column pivots puts it back in the
/// generator coordinates `(t, ω)`.
#[allow(
    clippy::needless_range_loop,
    reason = "back-substitution indexes R and the unknown together"
)]
fn kernel_basis(qr: &Qr) -> Vec<f64> {
    let k = qr.n;
    let r = qr.rank;
    let nullity = k - r;
    let mut z = vec![0.0; k * nullity];
    if nullity == 0 {
        return z;
    }
    if r == 0 {
        for i in 0..k {
            z[i * k + qr.pivot[i]] = 1.0;
        }
        return z;
    }
    for j in 0..nullity {
        let pcol = r + j;
        let mut y = vec![0.0; r];
        for i in (0..r).rev() {
            let mut sum = qr.a[pcol * qr.m + i];
            for c in i + 1..r {
                sum -= qr.a[c * qr.m + i] * y[c];
            }
            y[i] = sum / qr.a[i * qr.m + i];
        }
        let col = j * k;
        for i in 0..r {
            z[col + qr.pivot[i]] = -y[i];
        }
        z[col + qr.pivot[pcol]] = 1.0;
    }
    z
}

#[cfg(test)]
fn l2(v: &Array1<f64>) -> f64 {
    crate::vecops::nrm2(v.view())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn drops_a_pure_translation() {
        let pos = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut v = array![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        project_out_rot_trans(&mut v, pos.view(), None);
        assert!(l2(&v) < 1e-12);
    }

    #[test]
    fn unequal_masses_still_kill_translation() {
        let pos = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut v = array![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let masses = [16.0, 1.0, 1.0];
        project_horizontal(&mut v, pos.view(), Some(&masses), true, None);
        assert!(l2(&v) < 1e-12);
    }

    #[test]
    fn unequal_masses_kill_mass_weighted_rotation() {
        // Infinitesimal rotation about z of a planar triangle.
        let pos = array![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0];
        let mut v = array![0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0];
        let masses = [12.0, 1.0, 1.0];
        project_horizontal(&mut v, pos.view(), Some(&masses), true, None);
        assert!(l2(&v) < 1e-10, "{v:?}");
    }

    #[test]
    fn periodic_keeps_rotation_drops_translation() {
        let pos = array![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0];
        let rot = array![0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0];
        let mut v = rot.clone();
        project_horizontal(&mut v, pos.view(), None, false, None);
        assert!(l2(&v) > 0.5, "rotation was removed under PBC {v:?}");
        let mut t = array![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        project_horizontal(&mut t, pos.view(), None, false, None);
        assert!(l2(&t) < 1e-12, "translation survived under PBC {t:?}");
    }

    #[test]
    fn a_single_atom_is_pure_translation() {
        let pos = array![1.0, 2.0, -0.5];
        let mut v = array![0.3, -0.4, 0.5];
        project_out_rot_trans(&mut v, pos.view(), None);
        assert!(l2(&v) < 1e-12, "{v:?}");
    }

    #[test]
    fn near_linear_keeps_the_small_rotation() {
        let delta = 1e-10;
        let pos = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, delta, 0.0];
        let com_y = delta / 3.0;
        // ω = e_x about the centroid: v = (0, -z, y - com_y).
        let mut v = array![0.0, 0.0, -com_y, 0.0, 0.0, -com_y, 0.0, 0.0, delta - com_y];
        let before = l2(&v);
        project_out_rot_trans(&mut v, pos.view(), None);
        assert!(
            l2(&v) < 1e-8 * before,
            "small rotation survived: {} -> {}",
            before,
            l2(&v)
        );
    }

    #[test]
    fn frozen_slab_does_not_remove_the_adsorbate() {
        let pos = array![0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.5, 0.5, 2.0];
        let frozen = [true, true, true, false];
        let mut v = array![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, -0.1, 0.4];
        project_out_rot_trans(&mut v, pos.view(), Some(&frozen));
        let slab = v.iter().take(9).map(|c| c * c).sum::<f64>().sqrt();
        assert!(slab < 1e-12, "slab moved {v:?}");
        assert!((v[9] - 0.2).abs() < 1e-10, "{v:?}");
        assert!((v[10] + 0.1).abs() < 1e-10, "{v:?}");
        assert!((v[11] - 0.4).abs() < 1e-10, "{v:?}");
    }
}
