//! Sparse symmetric metric, separate from the exponential preconditioner.
//!
//! The matrix is a block CSR. `BlockAccum` records the sparsity, then
//! scatters rank-1 terms into those stored entries. The solve is
//! Jacobi-preconditioned conjugate gradients.

use std::sync::Arc;

use ndarray::{Array1, Array2, ArrayView1};

use crate::error::{Error, Result};

/// Atom counts for the parallel loops. Each gate is an atom count.
/// A kernel stays serial below its gate in every build.
///
/// Measured on bulk diamond Si, 4 Angstrom cutoff, release, five
/// interleaved runs, threads 1/2/4/8. A gate is the smallest size in
/// {160, 2_000, 8_000, 32_000, 100_000} where two threads beat one by
/// more than the wider run-to-run range. `usize::MAX` means that bar
/// was never cleared, so the kernel stays serial. The gate does not
/// read the thread count.
///
/// Lindh and bonded-metric matvec, eight applies.
/// 160 atoms: 1 thread 0.493 ms, 2 threads 0.361 ms, gap 0.132 ms,
/// range 0.344 ms, no.
/// 2,000 atoms: 1 thread 8.360 ms, 2 threads 4.751 ms, gap 3.609 ms,
/// range 1.052 ms, yes.
pub(crate) const N_PAR_LINDH_APPLY: usize = 2_000;
/// Lindh and bonded-metric Jacobi PCG, eight solves.
/// 160 atoms: 1 thread 31.355 ms, 2 threads 16.752 ms, gap 14.602 ms,
/// range 17.571 ms, no.
/// 2,000 atoms: 1 thread 0.914 s, 2 threads 0.467 s, gap 0.447 s,
/// range 0.035 s, yes.
pub(crate) const N_PAR_LINDH_SOLVE: usize = 2_000;

/// Fixed length of one dot-product chunk. The same chunks are used
/// with and without the `par` feature.
const DOT_CHUNK: usize = 1024;

#[derive(Clone, Debug)]
pub(crate) struct Csr {
    n: usize,
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    val: Vec<f64>,
    diag: Vec<f64>,
    /// Atom count at which `apply` may use the `par` pool.
    apply_gate: usize,
    /// Atom count at which the solve's products may use the `par` pool.
    solve_gate: usize,
}

/// Sparse SPD metric. `apply` is `P v`. `solve` is `P x = rhs`.
#[derive(Clone, Debug)]
pub(crate) struct Metric {
    csr: Arc<Csr>,
}

impl Metric {
    pub(crate) fn from_csr(csr: Csr) -> Self {
        Self { csr: Arc::new(csr) }
    }

    pub(crate) fn dim(&self) -> usize {
        self.csr.n
    }

    pub(crate) fn apply(&self, v: ArrayView1<f64>) -> Array1<f64> {
        let mut y = vec![0.0; self.csr.n];
        if v.len() == self.csr.n {
            matvec(
                &self.csr,
                v.as_slice().unwrap_or(&v.to_vec()),
                &mut y,
                self.csr.apply_gate,
            );
        }
        Array1::from_vec(y)
    }

    pub(crate) fn solve(&self, rhs: ArrayView1<f64>) -> Option<Array1<f64>> {
        self.solve_counted(rhs).map(|(x, _)| x)
    }

    pub(crate) fn solve_counted(&self, rhs: ArrayView1<f64>) -> Option<(Array1<f64>, usize)> {
        if rhs.len() != self.csr.n {
            return None;
        }
        let rhs_vec = rhs
            .as_slice()
            .map(<[f64]>::to_vec)
            .unwrap_or_else(|| rhs.to_vec());
        solve_pcg(&self.csr, &rhs_vec).map(|(x, iters)| (Array1::from_vec(x), iters))
    }

    pub(crate) fn to_dense(&self) -> Array2<f64> {
        let n = self.csr.n;
        let mut a = Array2::zeros((n, n));
        for i in 0..n {
            for k in self.csr.row_ptr[i]..self.csr.row_ptr[i + 1] {
                a[(i, self.csr.col_idx[k])] = self.csr.val[k];
            }
        }
        a
    }

    /// `mu P`. A model assembled at scale 1 applies the energy scale here.
    pub(crate) fn scaled(&self, mu: f64) -> Self {
        let mut csr = (*self.csr).clone();
        for v in &mut csr.val {
            *v *= mu;
        }
        for d in &mut csr.diag {
            *d *= mu;
        }
        Self::from_csr(csr)
    }
}

/// `mu = v^T (grad E(x+v) - grad E(x)) / v^T P v` for a model at scale 1.
///
/// The probe is [`super::sine_probe`]. A nonpositive ratio is an error.
/// The session then keeps the previous scale, or 1 if it has none.
pub(crate) fn model_scale<G>(
    metric: &Metric,
    positions: ArrayView1<f64>,
    r_nn: f64,
    lengths: Option<[f64; 3]>,
    grad_here: &Array1<f64>,
    grad: G,
) -> Result<f64>
where
    G: Fn(ArrayView1<f64>) -> Array1<f64>,
{
    if grad_here.len() != positions.len() {
        return Err(Error::Precon {
            what: "gradient length does not match positions",
        });
    }
    let v = super::sine_probe(positions, r_nn, lengths);
    let mut shifted = positions.to_owned();
    shifted += &v;
    let g1 = grad(shifted.view());
    if g1.len() != positions.len() {
        return Err(Error::Precon {
            what: "gradient length does not match positions",
        });
    }
    let mut num = 0.0;
    for i in 0..positions.len() {
        num += (g1[i] - grad_here[i]) * v[i];
    }
    let den = v.dot(&metric.apply(v.view()));
    if !(den.is_finite() && den.abs() > 1e-30 && num.is_finite()) {
        return Err(Error::Precon {
            what: "mu estimate denominator vanished",
        });
    }
    let mu = num / den;
    if !(mu.is_finite() && mu > 0.0) {
        return Err(Error::Precon {
            what: "mu estimate is not positive",
        });
    }
    Ok(mu)
}

/// Record which free atoms share a 3 by 3 block, then scatter into that
/// pattern. Memory is the stored nonzeros. Each row is summed from the
/// terms that touch it, in the order those terms are presented, so a
/// later parallel fill of the rows does not change the sum.
pub(crate) struct BlockAccum {
    n_atoms: usize,
    partners: Vec<Vec<usize>>,
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    val: Vec<f64>,
}

impl BlockAccum {
    /// `partners[i]` lists other free atoms. `i` itself is added here.
    pub(crate) fn new(n_atoms: usize, mut partners: Vec<Vec<usize>>) -> Self {
        for (i, p) in partners.iter_mut().enumerate() {
            p.push(i);
            p.sort_unstable();
            p.dedup();
        }
        let mut row_ptr = Vec::with_capacity(3 * n_atoms + 1);
        let mut col_idx = Vec::new();
        row_ptr.push(0);
        for partner in &partners {
            let cols: Vec<usize> = partner
                .iter()
                .flat_map(|&b| [3 * b, 3 * b + 1, 3 * b + 2])
                .collect();
            for _d in 0..3 {
                col_idx.extend_from_slice(&cols);
                row_ptr.push(col_idx.len());
            }
        }
        let nnz = col_idx.len();
        Self {
            n_atoms,
            partners,
            row_ptr,
            col_idx,
            val: vec![0.0; nnz],
        }
    }

    /// Add `scale * g g^T` on the free atoms. A `None` index is a ghost.
    pub(crate) fn add_rank1(&mut self, atoms: &[(Option<usize>, [f64; 3])], scale: f64) {
        if !(scale.is_finite() && scale != 0.0) {
            return;
        }
        let free: Vec<(usize, [f64; 3])> = atoms
            .iter()
            .copied()
            .filter_map(|(i, g)| i.map(|i| (i, g)))
            .collect();
        let partners = &self.partners;
        let row_ptr = &self.row_ptr;
        let val = &mut self.val;
        for &(ia, ga) in &free {
            let plist = &partners[ia];
            for d in 0..3 {
                let start = row_ptr[3 * ia + d];
                for &(ib, gb) in &free {
                    let p = plist
                        .binary_search(&ib)
                        .expect("rank-1 atom is missing from the sparsity pattern");
                    let base = start + 3 * p;
                    for e in 0..3 {
                        val[base + e] += scale * ga[d] * gb[e];
                    }
                }
            }
        }
    }

    /// Add `c` to every diagonal after the rank-1 terms.
    pub(crate) fn add_stabilization(&mut self, c: f64) {
        for atom in 0..self.n_atoms {
            let p = self.partners[atom]
                .binary_search(&atom)
                .expect("diagonal block");
            for d in 0..3 {
                let base = self.row_ptr[3 * atom + d] + 3 * p;
                self.val[base + d] += c;
            }
        }
    }

    pub(crate) fn finish(self, apply_gate: usize, solve_gate: usize) -> Csr {
        let n = 3 * self.n_atoms;
        let mut diag = vec![0.0; n];
        for (i, diag_i) in diag.iter_mut().enumerate() {
            let s = self.row_ptr[i];
            let e = self.row_ptr[i + 1];
            if let Ok(k) = self.col_idx[s..e].binary_search(&i) {
                *diag_i = self.val[s + k];
            }
        }
        Csr {
            n,
            row_ptr: self.row_ptr,
            col_idx: self.col_idx,
            val: self.val,
            diag,
            apply_gate,
            solve_gate,
        }
    }
}

/// Push every free atom of a rank-1 term onto the partner lists.
pub(crate) fn note_block(
    partners: &mut [Vec<usize>],
    atoms: &[(Option<usize>, [f64; 3])],
    scale: f64,
) {
    if !(scale.is_finite() && scale != 0.0) {
        return;
    }
    let ids: Vec<usize> = atoms.iter().filter_map(|a| a.0).collect();
    for &i in &ids {
        for &j in &ids {
            partners[i].push(j);
        }
    }
}

fn row_dot(a: &Csr, x: &[f64], i: usize) -> f64 {
    let mut s = 0.0;
    for k in a.row_ptr[i]..a.row_ptr[i + 1] {
        s += a.val[k] * x[a.col_idx[k]];
    }
    s
}

fn matvec(a: &Csr, x: &[f64], y: &mut [f64], gate: usize) {
    let parallel = gate < usize::MAX && a.n / 3 >= gate && a.n >= 2;
    #[cfg(not(feature = "par"))]
    let _ = parallel;
    #[cfg(feature = "par")]
    {
        use rayon::prelude::*;
        // Each row is an independent product. Writing `y[i]` from that
        // row does not reorder any sum, so the vector matches the
        // serial loop at every thread count.
        if parallel {
            y.par_iter_mut().enumerate().for_each(|(i, yi)| {
                *yi = row_dot(a, x, i);
            });
            return;
        }
    }
    for (i, yi) in y.iter_mut().enumerate() {
        *yi = row_dot(a, x, i);
    }
}

/// Dot product. Chunks of [`DOT_CHUNK`] are summed left to right from
/// `0.0` in every build. The `par` feature evaluates the chunks
/// concurrently when `atoms >= gate`; the fold of the partials stays
/// in chunk order.
fn dot(a: &[f64], b: &[f64], gate: usize) -> f64 {
    if a.is_empty() {
        return 0.0;
    }
    let nchunks = a.len().div_ceil(DOT_CHUNK);
    let atoms = a.len() / 3;
    let partials = super::ordered_map(nchunks, atoms, gate, |c| {
        let start = c * DOT_CHUNK;
        let end = (start + DOT_CHUNK).min(a.len());
        let mut acc = 0.0;
        for i in start..end {
            acc += a[i] * b[i];
        }
        acc
    });
    partials.into_iter().fold(0.0, |acc, v| acc + v)
}

fn nrm2(a: &[f64], gate: usize) -> f64 {
    dot(a, a, gate).sqrt()
}

fn solve_pcg(a: &Csr, b: &[f64]) -> Option<(Vec<f64>, usize)> {
    let n = a.n;
    let gate = a.solve_gate;
    if b.len() != n {
        return None;
    }
    let bnorm = nrm2(b, gate);
    if bnorm == 0.0 {
        return Some((vec![0.0; n], 0));
    }
    if a.diag.iter().any(|d| !(d.is_finite() && *d > 0.0)) {
        return None;
    }
    let mut x = vec![0.0; n];
    let mut r = b.to_vec();
    let mut z = vec![0.0; n];
    let mut p = vec![0.0; n];
    let mut ap = vec![0.0; n];
    jacobi(a, &r, &mut z);
    p.copy_from_slice(&z);
    let mut rz = dot(&r, &z, gate);
    if !(rz.is_finite() && rz > 0.0) {
        return None;
    }
    let tol = 1e-12 * bnorm;
    let maxit = (8 * n).max(64);
    let mut iters = 0usize;
    for _ in 0..maxit {
        if nrm2(&r, gate) <= tol {
            return finite_vec(x).map(|v| (v, iters));
        }
        iters += 1;
        matvec(a, &p, &mut ap, gate);
        let pap = dot(&p, &ap, gate);
        if !(pap.is_finite() && pap > 0.0) {
            return None;
        }
        let alpha = rz / pap;
        if !alpha.is_finite() {
            return None;
        }
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        if nrm2(&r, gate) <= tol {
            return finite_vec(x).map(|v| (v, iters));
        }
        jacobi(a, &r, &mut z);
        let rz_new = dot(&r, &z, gate);
        if !(rz_new.is_finite() && rz_new > 0.0) {
            return None;
        }
        let beta = rz_new / rz;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        rz = rz_new;
    }
    if nrm2(&r, gate) <= 1e-8 * bnorm {
        return finite_vec(x).map(|v| (v, iters));
    }
    None
}

fn jacobi(a: &Csr, r: &[f64], z: &mut [f64]) {
    for i in 0..a.n {
        z[i] = r[i] / a.diag[i];
    }
}

fn finite_vec(x: Vec<f64>) -> Option<Vec<f64>> {
    x.iter().all(|v| v.is_finite()).then_some(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn fnv1a(x: &[f64]) -> u64 {
        let mut h = 0xcbf29ce484222325u64;
        for v in x {
            for b in v.to_le_bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        h
    }

    #[test]
    fn pcg_solution_hash_is_the_same_in_every_build() {
        // 512 atoms, several chunks of 1024, a banded SPD matrix.
        let n = 1536;
        let mut row_ptr = vec![0usize];
        let mut col_idx = Vec::new();
        let mut val = Vec::new();
        let mut diag = vec![0.0; n];
        for (i, diag_i) in diag.iter_mut().enumerate() {
            let mut cols = Vec::new();
            if i >= 5 {
                cols.push((i - 5, -0.25));
            }
            if i > 0 {
                cols.push((i - 1, -1.0));
            }
            let d = 4.0 + (i % 7) as f64;
            cols.push((i, d));
            *diag_i = d;
            if i + 1 < n {
                cols.push((i + 1, -1.0));
            }
            if i + 5 < n {
                cols.push((i + 5, -0.25));
            }
            for (c, v) in cols {
                col_idx.push(c);
                val.push(v);
            }
            row_ptr.push(col_idx.len());
        }
        let csr = Csr {
            n,
            row_ptr,
            col_idx,
            val,
            diag,
            apply_gate: usize::MAX,
            solve_gate: usize::MAX,
        };
        let rhs: Vec<f64> = (0..n).map(|i| ((i * 17) % 11) as f64 - 5.0).collect();
        let (sol, iters) = solve_pcg(&csr, &rhs).expect("pcg");
        let hash = fnv1a(&sol);
        assert!(
            iters > 1 && iters <= n,
            "iters {iters} should take more than one step and at most n"
        );
        assert_eq!(
            hash, 0xf9a8d91302c9cb63,
            "committed hash drifted: iters {iters} hash {hash:#x}"
        );
    }

    #[test]
    fn a_rank1_block_is_symmetric_and_matches_a_dense_solve() {
        let atoms = [(Some(0), [1.0, 0.2, -0.1]), (Some(1), [-0.4, 0.7, 0.3])];
        let mut partners = vec![Vec::new(); 2];
        note_block(&mut partners, &atoms, 1.5);
        let mut acc = BlockAccum::new(2, partners);
        acc.add_rank1(&atoms, 1.5);
        acc.add_stabilization(0.2);
        let metric = Metric::from_csr(acc.finish(usize::MAX, usize::MAX));
        let dense = metric.to_dense();
        for i in 0..6 {
            for j in 0..i {
                assert!(
                    (dense[(i, j)] - dense[(j, i)]).abs() < 1e-12,
                    "asym {i} {j}"
                );
            }
        }
        let rhs = array![0.2, -0.1, 0.05, 0.0, 0.3, -0.2];
        let sparse = metric.solve(rhs.view()).expect("pcg");
        let reference = crate::qn::solve_dense(&dense, &rhs).expect("dense");
        let err = (&sparse - &reference)
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        assert!(err < 1e-8, "sparse vs dense {err}");
    }

    #[test]
    fn model_scale_recovers_a_constant_multiple() {
        let atoms = [(Some(0), [1.0, 0.0, 0.0]), (Some(1), [-1.0, 0.0, 0.0])];
        let mut partners = vec![Vec::new(); 2];
        note_block(&mut partners, &atoms, 1.0);
        let mut acc = BlockAccum::new(2, partners);
        acc.add_rank1(&atoms, 1.0);
        acc.add_stabilization(0.4);
        let metric = Metric::from_csr(acc.finish(usize::MAX, usize::MAX));
        let pos = array![0.2, 0.4, 0.1, 1.1, -0.3, 0.5];
        let scale = 2.5;
        let grad = |x: ArrayView1<f64>| scale * metric.apply(x);
        let mu = model_scale(
            &metric,
            pos.view(),
            1.0,
            Some([4.0, 4.0, 4.0]),
            &grad(pos.view()),
            grad,
        )
        .unwrap();
        assert!((mu - scale).abs() < 1e-8, "mu {mu}");
    }
}
