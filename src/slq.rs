//! Matrix-free Krylov primitives for symmetric log-determinants and linear systems.
//!
//! Provides:
//! - [`tridiag_eigh`] and [`sym_eigh`]: pure-Rust implicit-QL (Wilkinson-shift)
//!   symmetric tridiagonal and dense eigensolvers.
//! - [`gauss_radau_tridiag`] and [`lanczos_quadrature_log`]: Golub-Welsch Gauss
//!   Lanczos quadrature for `v^T ln(A) v`, with optional left Gauss-Radau
//!   two-sided brackets (`GR_m(a) <= v^T ln(A) v <= G_m`) when the caller
//!   supplies a valid positive lower bound `0 < a <= lambda_min(A)`, or
//!   adaptive Gauss convergence estimation when `lower_bound` is `None`.
//! - [`lanczos_quadrature_fn`]: adaptive Lanczos quadrature for general smooth
//!   scalar functions `v^T f(A) v`.
//! - [`pcg_solve`]: preconditioned conjugate gradient on an optional orthogonal
//!   complement, with true-residual verification and indefiniteness refusal.
//! - [`lanczos_extreme_pairs`]: deflated thick-restart Lanczos with full two-pass
//!   reorthogonalization to locate negative and extreme positive eigenpairs and
//!   estimate the bottom of the positive spectrum.
//! - [`crate::slq::student_t_quantile`]: exact two-sided Student-t quantiles via the
//!   regularized incomplete beta function.
//! - [`slq_logdet_deflated`], [`slq_logdet_deflated_with_cv`], and
//!   [`slq_logdet_hutchpp`]: deflated stochastic Lanczos quadrature (Hutchinson
//!   Rademacher or strided-color probing, optional control variate, polynomial
//!   moment subtraction, and Hutch++ range sketching) for `ln |det'(A)|` with
//!   fixed-sample or sequentially valid Bonferroni alpha-spending confidence
//!   intervals. Stochastic error estimates are probabilistic, not deterministic
//!   enclosures.

use crate::error::{Error, Result};

fn slq_err(what: &'static str) -> Error {
    Error::Slq { what }
}

fn dot_slice(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        sum += x * y;
    }
    sum
}

fn norm2_slice(a: &[f64]) -> f64 {
    let mut scale = 0.0_f64;
    let mut ssq = 1.0_f64;
    for &x in a {
        let ax = x.abs();
        if ax == 0.0 {
            continue;
        }
        if !ax.is_finite() {
            return f64::NAN;
        }
        if scale < ax {
            let r = scale / ax;
            ssq = 1.0 + ssq * r * r;
            scale = ax;
        } else {
            let r = ax / scale;
            ssq += r * r;
        }
    }
    if scale == 0.0 {
        0.0
    } else {
        scale * ssq.sqrt()
    }
}

fn all_finite(a: &[f64]) -> bool {
    a.iter().all(|v| v.is_finite())
}

fn project_out_basis(v: &mut [f64], basis: &[Vec<f64>]) {
    for _ in 0..2 {
        for q in basis {
            let c = dot_slice(q, v);
            for (slot, &qi) in v.iter_mut().zip(q.iter()) {
                *slot -= c * qi;
            }
        }
    }
}

fn orthonormalize_vectors(vectors: &[Vec<f64>], dim: usize) -> Result<Vec<Vec<f64>>> {
    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(vectors.len());
    for v in vectors {
        if v.len() != dim {
            return Err(slq_err("deflation vector dimension mismatch"));
        }
        if !all_finite(v) {
            return Err(slq_err("deflation vector contains non-finite entries"));
        }
        let mut q = v.clone();
        project_out_basis(&mut q, &basis);
        let nrm = norm2_slice(&q);
        if !nrm.is_finite() {
            return Err(slq_err(
                "non-finite norm while orthonormalizing deflation basis",
            ));
        }
        if nrm > 1e-12 {
            for slot in &mut q {
                *slot /= nrm;
            }
            basis.push(q);
        }
    }
    Ok(basis)
}

/// Deterministic SplitMix64 generator for reproducible Rademacher probes.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn next_rademacher(&mut self) -> f64 {
        if (self.next_u64() & 1) == 0 {
            1.0
        } else {
            -1.0
        }
    }

    fn next_unit_f64(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        let u = (bits as f64) * (1.0 / ((1u64 << 53) as f64));
        2.0 * u - 1.0
    }
}

/// Symmetric eigendecomposition output: ascending eigenvalues and row-major
/// eigenvector columns (`vectors[row * m + col]`).
#[derive(Debug, Clone)]
pub struct TridiagEigh {
    /// Eigenvalues sorted in ascending order.
    pub values: Vec<f64>,
    /// Row-major `m x m` orthogonal eigenvector matrix; column `j` corresponds
    /// to `values[j]`.
    pub vectors: Vec<f64>,
}

/// Implicit-QL (Wilkinson-shift) eigendecomposition of a real symmetric
/// tridiagonal matrix with diagonal `diag` (`m`) and subdiagonal `offdiag`
/// (`m - 1`).
///
/// Refuses mismatched lengths, non-finite entries, or non-convergence.
pub fn tridiag_eigh(diag: &[f64], offdiag: &[f64]) -> Result<TridiagEigh> {
    let m = diag.len();
    if m == 0 {
        if !offdiag.is_empty() {
            return Err(slq_err("offdiag must be empty when diag is empty"));
        }
        return Ok(TridiagEigh {
            values: Vec::new(),
            vectors: Vec::new(),
        });
    }
    if offdiag.len() + 1 != m {
        return Err(slq_err("offdiag length must equal diag.len() - 1"));
    }
    if !all_finite(diag) || !all_finite(offdiag) {
        return Err(slq_err("tridiagonal matrix contains non-finite entries"));
    }
    let mut d = diag.to_vec();
    let mut e = vec![0.0; m];
    e[..m - 1].copy_from_slice(offdiag);

    // Column-major accumulation for cache-friendly Givens rotations; transposed
    // to row-major at return.
    let mut z = vec![0.0; m * m];
    for i in 0..m {
        z[i * m + i] = 1.0;
    }

    if m > 1 {
        let max_iter = 60usize;
        for l in 0..m {
            let mut iter = 0usize;
            loop {
                // Find smallest index mm >= l with negligible e[mm].
                let mut mm = l;
                while mm + 1 < m {
                    let dd = d[mm].abs() + d[mm + 1].abs();
                    if e[mm].abs() <= f64::EPSILON * dd {
                        break;
                    }
                    mm += 1;
                }
                if mm == l {
                    break;
                }
                if iter >= max_iter {
                    return Err(slq_err("tridiagonal QL did not converge"));
                }
                iter += 1;

                // Wilkinson shift from leading 2x2 block at (l, l+1).
                let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
                let r = g.hypot(1.0);
                let signed_r = if g >= 0.0 { r } else { -r };
                g = d[mm] - d[l] + e[l] / (g + signed_r);

                let mut s = 1.0;
                let mut c = 1.0;
                let mut p = 0.0;
                let mut broke_underflow = false;

                for i in (l..mm).rev() {
                    let f = s * e[i];
                    let b = c * e[i];
                    let r = f.hypot(g);
                    e[i + 1] = r;
                    if r == 0.0 {
                        d[i + 1] -= p;
                        e[mm] = 0.0;
                        broke_underflow = true;
                        break;
                    }
                    s = f / r;
                    c = g / r;
                    g = d[i + 1] - p;
                    let t = (d[i] - g) * s + 2.0 * c * b;
                    p = s * t;
                    d[i + 1] = g + p;
                    g = c * t - b;

                    let col_i = i * m;
                    let col_next = (i + 1) * m;
                    for k in 0..m {
                        let zi = z[col_i + k];
                        let zn = z[col_next + k];
                        z[col_next + k] = s * zi + c * zn;
                        z[col_i + k] = c * zi - s * zn;
                    }
                }
                if !broke_underflow {
                    d[l] -= p;
                    e[l] = g;
                    e[mm] = 0.0;
                }
            }
        }
    }

    if !all_finite(&d) || !all_finite(&z) {
        return Err(slq_err("tridiagonal QL produced non-finite output"));
    }

    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| d[a].total_cmp(&d[b]));

    let mut values = Vec::with_capacity(m);
    let mut vectors = vec![0.0; m * m];
    for (new_col, &old_col) in order.iter().enumerate() {
        values.push(d[old_col]);
        let src = old_col * m;
        for row in 0..m {
            vectors[row * m + new_col] = z[src + row];
        }
    }
    Ok(TridiagEigh { values, vectors })
}

/// Dense symmetric eigendecomposition via Householder tridiagonalization and
/// [`tridiag_eigh`]. `a` is row-major `m x m` (symmetrized internally).
///
/// Intended for small Krylov projected matrices (`m <= 256`).
pub fn sym_eigh(a: &[f64], m: usize) -> Result<TridiagEigh> {
    if m == 0 {
        if !a.is_empty() {
            return Err(slq_err("matrix must be empty when m == 0"));
        }
        return Ok(TridiagEigh {
            values: Vec::new(),
            vectors: Vec::new(),
        });
    }
    let Some(cells) = m.checked_mul(m) else {
        return Err(slq_err("dense eigensolver dimension overflowed"));
    };
    if a.len() != cells {
        return Err(slq_err("dense matrix length must equal m * m"));
    }
    if !all_finite(a) {
        return Err(slq_err(
            "dense symmetric matrix contains non-finite entries",
        ));
    }
    if m == 1 {
        return Ok(TridiagEigh {
            values: vec![a[0]],
            vectors: vec![1.0],
        });
    }

    // Symmetrize into a working copy (row-major).
    let mut q = vec![0.0; cells];
    for i in 0..m {
        q[i * m + i] = a[i * m + i];
        for j in (i + 1)..m {
            let s = 0.5 * (a[i * m + j] + a[j * m + i]);
            q[i * m + j] = s;
            q[j * m + i] = s;
        }
    }

    // Householder reduction to tridiagonal form (tred2).
    let mut diag = vec![0.0; m];
    let mut off = vec![0.0; m];
    for i in (1..m).rev() {
        let l = i - 1;
        let mut h = 0.0;
        let mut scale = 0.0;
        if l > 0 {
            for k in 0..=l {
                scale += q[i * m + k].abs();
            }
            if scale == 0.0 {
                off[i] = q[i * m + l];
            } else {
                for k in 0..=l {
                    let v = q[i * m + k] / scale;
                    q[i * m + k] = v;
                    h += v * v;
                }
                let f = q[i * m + l];
                let g = if f >= 0.0 { -h.sqrt() } else { h.sqrt() };
                off[i] = scale * g;
                h -= f * g;
                q[i * m + l] = f - g;
                let mut f_acc = 0.0;
                for j in 0..=l {
                    q[j * m + i] = q[i * m + j] / h;
                    let mut g_acc = 0.0;
                    for k in 0..=j {
                        g_acc += q[j * m + k] * q[i * m + k];
                    }
                    for k in (j + 1)..=l {
                        g_acc += q[k * m + j] * q[i * m + k];
                    }
                    off[j] = g_acc / h;
                    f_acc += off[j] * q[i * m + j];
                }
                let hh = f_acc / (h + h);
                for j in 0..=l {
                    let f_j = q[i * m + j];
                    let g_j = off[j] - hh * f_j;
                    off[j] = g_j;
                    for k in 0..=j {
                        q[j * m + k] -= f_j * off[k] + g_j * q[i * m + k];
                    }
                }
            }
        } else {
            off[i] = q[i * m + l];
        }
        diag[i] = h;
    }
    diag[0] = 0.0;
    off[0] = 0.0;

    // Accumulate Householder transformations into Q.
    for i in 0..m {
        if diag[i] != 0.0 {
            for j in 0..i {
                let mut g = 0.0;
                for k in 0..i {
                    g += q[i * m + k] * q[k * m + j];
                }
                for k in 0..i {
                    q[k * m + j] -= g * q[k * m + i];
                }
            }
        }
        diag[i] = q[i * m + i];
        q[i * m + i] = 1.0;
        for j in 0..i {
            q[j * m + i] = 0.0;
            q[i * m + j] = 0.0;
        }
    }

    let tri = tridiag_eigh(&diag, &off[1..m])?;
    let mut vectors = vec![0.0; cells];
    for row in 0..m {
        for col in 0..m {
            let mut sum = 0.0;
            for k in 0..m {
                sum += q[row * m + k] * tri.vectors[k * m + col];
            }
            vectors[row * m + col] = sum;
        }
    }
    Ok(TridiagEigh {
        values: tri.values,
        vectors,
    })
}

/// Extend a Lanczos tridiagonal `(diag, offdiag)` of size `m` with last
/// residual `beta_last` to the `(m + 1) x (m + 1)` Gauss-Radau tridiagonal
/// having `fixed_node` as an exact eigenvalue (Golub & Meurant, Theorem 6.2).
///
/// When `fixed_node < lambda_min(T_m)`, Sylvester's law of inertia certifies
/// `T_m - fixed_node * I > 0` during the `O(m)` pivot sweep. Refuses any
/// `fixed_node` that does not lie strictly outside the spectrum of `T_m` on the
/// requested side.
pub fn gauss_radau_tridiag(
    diag: &[f64],
    offdiag: &[f64],
    beta_last: f64,
    fixed_node: f64,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let m = diag.len();
    if m == 0 || offdiag.len() + 1 != m {
        return Err(slq_err("invalid tridiagonal dimensions for Gauss-Radau"));
    }
    if !all_finite(diag)
        || !all_finite(offdiag)
        || !beta_last.is_finite()
        || beta_last < 0.0
        || !fixed_node.is_finite()
    {
        return Err(slq_err("non-finite or negative beta in Gauss-Radau"));
    }
    // LDL^T pivot sweep for T_m - fixed_node * I.
    let mut pivot = diag[0] - fixed_node;
    if !pivot.is_finite() || pivot == 0.0 {
        return Err(slq_err("Gauss-Radau node hits leading diagonal entry"));
    }
    let sign_positive = pivot > 0.0;
    for i in 1..m {
        let b = offdiag[i - 1];
        pivot = (diag[i] - fixed_node) - (b * b) / pivot;
        if !pivot.is_finite() || pivot == 0.0 || (pivot > 0.0) != sign_positive {
            return Err(slq_err(
                "Gauss-Radau node is not strictly outside the tridiagonal spectrum",
            ));
        }
    }
    let alpha_ext = fixed_node + (beta_last * beta_last) / pivot;
    if !alpha_ext.is_finite() {
        return Err(slq_err("Gauss-Radau diagonal extension overflowed"));
    }
    let mut ext_diag = Vec::with_capacity(m + 1);
    ext_diag.extend_from_slice(diag);
    ext_diag.push(alpha_ext);
    let mut ext_off = Vec::with_capacity(m);
    ext_off.extend_from_slice(offdiag);
    ext_off.push(beta_last);
    Ok((ext_diag, ext_off))
}

/// Output of one Lanczos quadrature evaluation of `v^T f(A) v`.
#[derive(Debug, Clone, Copy)]
pub struct QuadratureEstimate {
    /// Point estimate (midpoint of Gauss and Gauss-Radau when bracketed, or
    /// the Gauss quadrature value otherwise).
    pub value: f64,
    /// Lower endpoint (left Gauss-Radau when `bracketed` is true; estimated
    /// lower endpoint otherwise).
    pub lower: f64,
    /// Upper endpoint (Gauss quadrature value for `ln`; estimated upper
    /// endpoint for general functions).
    pub upper: f64,
    /// Non-negative quadrature error estimate (`0.5 * (upper - lower)` when
    /// bracketed; successive-step Gauss difference otherwise).
    pub error_estimate: f64,
    /// True if a caller-supplied `lower_bound` produced a Gauss / left
    /// Gauss-Radau bracket (which is a true two-sided enclosure in exact
    /// arithmetic only if `0 < lower_bound <= lambda_min(A)`).
    pub bracketed: bool,
    /// Exact first moment `v^T A v` (`||v||^2 * alpha_0`) captured on step 0.
    pub first_moment: f64,
    /// Exact second moment `v^T A^2 v` (`||v||^2 * (alpha_0^2 + beta_0^2)`) captured on step 0.
    pub second_moment: f64,
    /// Number of Lanczos steps taken.
    pub steps: usize,
    /// Number of operator applications performed.
    pub matvecs: usize,
    /// True if Lanczos terminated at an invariant subspace (`beta_j == 0`).
    pub invariant_breakdown: bool,
}

fn eval_gauss_ln(diag: &[f64], offdiag: &[f64], norm_sq: f64) -> Result<(f64, f64)> {
    let eig = tridiag_eigh(diag, offdiag)?;
    let m = diag.len();
    let mut sum = 0.0;
    for j in 0..m {
        let lam = eig.values[j];
        if !(lam > 0.0 && lam.is_finite()) {
            return Err(slq_err(
                "Lanczos tridiagonal produced a non-positive Ritz value for ln",
            ));
        }
        let w = eig.vectors[j];
        sum += w * w * lam.ln();
    }
    let val = norm_sq * sum;
    if !val.is_finite() {
        return Err(slq_err("Gauss quadrature for ln was not finite"));
    }
    Ok((val, eig.values[0]))
}

/// Adaptive Lanczos quadrature for `v^T ln(A) v` on a symmetric positive
/// definite operator `A`.
///
/// - When `lower_bound` is `Some(a)`, computes Golub-Welsch Gauss (upper) and
///   left Gauss-Radau at `a` (lower). Because every even derivative of `ln(x)`
///   is strictly negative and every odd derivative is strictly positive on
///   `(0, inf)`, `GR_m(a) <= v^T ln(A) v <= G_m` holds in exact arithmetic
///   whenever `0 < a <= lambda_min(A)`. Returns a typed error if `a` is not
///   strictly below the smallest Ritz value.
/// - When `lower_bound` is `None`, computes adaptive Golub-Welsch Gauss
///   quadrature `G_m` (still refusing any non-positive Ritz value) and
///   estimates convergence from successive monotone Gauss differences
///   `|G_m - G_{m_prev}|`.
///
/// Uses full two-pass modified Gram-Schmidt reorthogonalization so finite
/// precision does not introduce ghost Ritz values.
pub fn lanczos_quadrature_log<F>(
    v: &[f64],
    op: &mut F,
    lower_bound: Option<f64>,
    max_steps: usize,
    quad_tol: f64,
    check_every: usize,
) -> Result<QuadratureEstimate>
where
    F: FnMut(&[f64], &mut [f64]) -> Result<()>,
{
    lanczos_quadrature_log_inner(
        v,
        op,
        lower_bound,
        max_steps,
        quad_tol,
        check_every,
        v.len(),
    )
}

fn lanczos_quadrature_log_inner<F>(
    v: &[f64],
    op: &mut F,
    lower_bound: Option<f64>,
    max_steps: usize,
    quad_tol: f64,
    check_every: usize,
    effective_dim: usize,
) -> Result<QuadratureEstimate>
where
    F: FnMut(&[f64], &mut [f64]) -> Result<()>,
{
    let n = v.len();
    if n == 0 {
        return Err(slq_err("empty probe vector"));
    }
    let eff_dim = effective_dim.min(n).max(1);
    if !all_finite(v) {
        return Err(slq_err("probe vector contains non-finite entries"));
    }
    if let Some(lb) = lower_bound
        && !(lb > 0.0 && lb.is_finite())
    {
        return Err(slq_err(
            "lower_bound for ln quadrature must be finite and positive",
        ));
    }
    if !(quad_tol >= 0.0 && quad_tol.is_finite()) {
        return Err(slq_err("quad_tol must be finite and non-negative"));
    }
    let norm = norm2_slice(v);
    if !norm.is_finite() {
        return Err(slq_err("probe norm is not finite"));
    }
    if norm == 0.0 {
        return Ok(QuadratureEstimate {
            value: 0.0,
            lower: 0.0,
            upper: 0.0,
            error_estimate: 0.0,
            bracketed: lower_bound.is_some(),
            first_moment: 0.0,
            second_moment: 0.0,
            steps: 0,
            matvecs: 0,
            invariant_breakdown: true,
        });
    }
    let steps_cap = max_steps.min(eff_dim).max(1);
    let check_stride = check_every.max(1);
    let norm_sq = norm * norm;

    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(steps_cap);
    let q0: Vec<f64> = v.iter().map(|&x| x / norm).collect();
    basis.push(q0);

    let mut alpha = Vec::with_capacity(steps_cap);
    let mut beta = Vec::with_capacity(steps_cap);
    let mut w = vec![0.0; n];
    let mut first_moment = 0.0;
    let mut second_moment = 0.0;

    let mut prev_gauss: Option<f64> = None;
    let mut last_est: Option<QuadratureEstimate> = None;

    for j in 0..steps_cap {
        w.fill(0.0);
        op(&basis[j], &mut w)?;
        let matvecs = j + 1;
        if !all_finite(&w) {
            return Err(slq_err(
                "operator produced non-finite output in Lanczos quadrature",
            ));
        }
        let a_j = dot_slice(&basis[j], &w);
        if !a_j.is_finite() {
            return Err(slq_err("non-finite Lanczos alpha"));
        }
        alpha.push(a_j);

        for (slot, &qj) in w.iter_mut().zip(basis[j].iter()) {
            *slot -= a_j * qj;
        }
        if j > 0 {
            let b_prev = beta[j - 1];
            for (slot, &qprev) in w.iter_mut().zip(basis[j - 1].iter()) {
                *slot -= b_prev * qprev;
            }
        }
        project_out_basis(&mut w, &basis);
        let b_j = norm2_slice(&w);
        if !b_j.is_finite() {
            return Err(slq_err("non-finite Lanczos beta"));
        }
        if j == 0 {
            first_moment = norm_sq * a_j;
            second_moment = norm_sq * (a_j * a_j + b_j * b_j);
        }

        let m = j + 1;
        let breakdown_tol = (n as f64).sqrt() * f64::EPSILON * (1.0 + a_j.abs());
        let invariant = b_j <= breakdown_tol || m == eff_dim;

        if invariant {
            let (gauss, _) = eval_gauss_ln(&alpha, &beta[..m - 1], norm_sq)?;
            return Ok(QuadratureEstimate {
                value: gauss,
                lower: gauss,
                upper: gauss,
                error_estimate: 0.0,
                bracketed: true,
                first_moment,
                second_moment,
                steps: m,
                matvecs,
                invariant_breakdown: true,
            });
        }

        let is_last = m == steps_cap;
        let should_check = is_last || (m >= 2 && m % check_stride == 0);
        if should_check {
            let (gauss, ritz_min) = eval_gauss_ln(&alpha, &beta[..m - 1], norm_sq)?;
            if let Some(lb) = lower_bound {
                if !ritz_min.is_finite() || lb >= ritz_min {
                    return Err(slq_err("lower_bound is not strictly below Ritz minimum"));
                }
                let (gr_d, gr_e) = gauss_radau_tridiag(&alpha, &beta[..m - 1], b_j, lb)?;
                let (radau, _) = eval_gauss_ln(&gr_d, &gr_e, norm_sq)?;
                if !radau.is_finite() || !gauss.is_finite() {
                    return Err(slq_err("non-finite Gauss or Gauss-Radau quadrature value"));
                }
                let (lower, upper) = if radau <= gauss {
                    (radau, gauss)
                } else if radau - gauss <= 1e-11 * (1.0 + gauss.abs()) {
                    (gauss, radau)
                } else {
                    return Err(slq_err(
                        "Gauss-Radau lower bound exceeded Gauss upper bound",
                    ));
                };
                let value = 0.5 * (lower + upper);
                let error_estimate = 0.5 * (upper - lower);
                let est = QuadratureEstimate {
                    value,
                    lower,
                    upper,
                    error_estimate,
                    bracketed: true,
                    first_moment,
                    second_moment,
                    steps: m,
                    matvecs,
                    invariant_breakdown: false,
                };
                if error_estimate <= quad_tol || is_last {
                    return Ok(est);
                }
                last_est = Some(est);
            } else {
                if let Some(prev) = prev_gauss {
                    let diff = (prev - gauss).abs();
                    let est = QuadratureEstimate {
                        value: gauss,
                        lower: gauss - diff,
                        upper: gauss,
                        error_estimate: diff,
                        bracketed: false,
                        first_moment,
                        second_moment,
                        steps: m,
                        matvecs,
                        invariant_breakdown: false,
                    };
                    if diff <= quad_tol || is_last {
                        return Ok(est);
                    }
                    last_est = Some(est);
                } else if is_last {
                    return Ok(QuadratureEstimate {
                        value: gauss,
                        lower: gauss,
                        upper: gauss,
                        error_estimate: f64::INFINITY,
                        bracketed: false,
                        first_moment,
                        second_moment,
                        steps: m,
                        matvecs,
                        invariant_breakdown: false,
                    });
                }
                prev_gauss = Some(gauss);
            }
        }

        beta.push(b_j);
        let q_next: Vec<f64> = w.iter().map(|&x| x / b_j).collect();
        basis.push(q_next);
    }

    last_est.ok_or_else(|| slq_err("Lanczos quadrature completed with no estimate"))
}

/// Adaptive Lanczos quadrature for `v^T f(A) v` on a symmetric operator `A` and
/// arbitrary scalar function `f` (e.g. Matsubara `sinh_tail`).
///
/// Estimates convergence from successive Lanczos Gauss quadratures or exact
/// invariant-subspace breakdown.
pub fn lanczos_quadrature_fn<F, G>(
    v: &[f64],
    op: &mut F,
    mut func: G,
    max_steps: usize,
    quad_tol: f64,
) -> Result<QuadratureEstimate>
where
    F: FnMut(&[f64], &mut [f64]) -> Result<()>,
    G: FnMut(f64) -> Result<f64>,
{
    let n = v.len();
    if n == 0 {
        return Err(slq_err("empty probe vector"));
    }
    if !all_finite(v) {
        return Err(slq_err("probe vector contains non-finite entries"));
    }
    if !(quad_tol >= 0.0 && quad_tol.is_finite()) {
        return Err(slq_err("quad_tol must be finite and non-negative"));
    }
    let norm = norm2_slice(v);
    if !norm.is_finite() {
        return Err(slq_err("probe norm is not finite"));
    }
    if norm == 0.0 {
        return Ok(QuadratureEstimate {
            value: 0.0,
            lower: 0.0,
            upper: 0.0,
            error_estimate: 0.0,
            bracketed: false,
            first_moment: 0.0,
            second_moment: 0.0,
            steps: 0,
            matvecs: 0,
            invariant_breakdown: true,
        });
    }
    let steps_cap = max_steps.min(n).max(1);
    let norm_sq = norm * norm;

    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(steps_cap);
    basis.push(v.iter().map(|&x| x / norm).collect());
    let mut alpha = Vec::with_capacity(steps_cap);
    let mut beta = Vec::with_capacity(steps_cap);
    let mut w = vec![0.0; n];
    let mut prev_val: Option<f64> = None;
    let mut first_moment = 0.0;
    let mut second_moment = 0.0;

    for j in 0..steps_cap {
        w.fill(0.0);
        op(&basis[j], &mut w)?;
        let matvecs = j + 1;
        if !all_finite(&w) {
            return Err(slq_err(
                "operator produced non-finite output in Lanczos quadrature",
            ));
        }
        let a_j = dot_slice(&basis[j], &w);
        if !a_j.is_finite() {
            return Err(slq_err("non-finite Lanczos alpha"));
        }
        alpha.push(a_j);
        for (slot, &qj) in w.iter_mut().zip(basis[j].iter()) {
            *slot -= a_j * qj;
        }
        if j > 0 {
            let b_prev = beta[j - 1];
            for (slot, &qprev) in w.iter_mut().zip(basis[j - 1].iter()) {
                *slot -= b_prev * qprev;
            }
        }
        project_out_basis(&mut w, &basis);
        let b_j = norm2_slice(&w);
        if !b_j.is_finite() {
            return Err(slq_err("non-finite Lanczos beta"));
        }
        if j == 0 {
            first_moment = norm_sq * a_j;
            second_moment = norm_sq * (a_j * a_j + b_j * b_j);
        }
        let m = j + 1;
        let breakdown_tol = (n as f64).sqrt() * f64::EPSILON * (1.0 + a_j.abs());
        let invariant = b_j <= breakdown_tol || m == n;

        let eig = tridiag_eigh(&alpha, &beta[..m - 1])?;
        let mut sum = 0.0;
        for col in 0..m {
            let fv = func(eig.values[col])?;
            if !fv.is_finite() {
                return Err(slq_err("scalar function returned non-finite value"));
            }
            let wt = eig.vectors[col];
            sum += wt * wt * fv;
        }
        let val = norm_sq * sum;
        if !val.is_finite() {
            return Err(slq_err("Lanczos quadrature sum was not finite"));
        }

        if invariant {
            return Ok(QuadratureEstimate {
                value: val,
                lower: val,
                upper: val,
                error_estimate: 0.0,
                bracketed: true,
                first_moment,
                second_moment,
                steps: m,
                matvecs,
                invariant_breakdown: true,
            });
        }

        if let Some(prev) = prev_val {
            let diff = (val - prev).abs();
            if diff <= quad_tol || m == steps_cap {
                return Ok(QuadratureEstimate {
                    value: val,
                    lower: val - diff,
                    upper: val + diff,
                    error_estimate: diff,
                    bracketed: false,
                    first_moment,
                    second_moment,
                    steps: m,
                    matvecs,
                    invariant_breakdown: false,
                });
            }
        } else if m == steps_cap {
            return Ok(QuadratureEstimate {
                value: val,
                lower: val,
                upper: val,
                error_estimate: f64::INFINITY,
                bracketed: false,
                first_moment,
                second_moment,
                steps: m,
                matvecs,
                invariant_breakdown: false,
            });
        }
        prev_val = Some(val);
        beta.push(b_j);
        basis.push(w.iter().map(|&x| x / b_j).collect());
    }

    Err(slq_err(
        "Lanczos quadrature exhausted without producing an estimate",
    ))
}

/// Configuration for [`pcg_solve`].
#[derive(Debug, Clone, Copy)]
pub struct PcgConfig {
    /// Target relative true residual `||b - A x||_2 / ||b||_2`.
    pub rel_tol: f64,
    /// Maximum CG iterations.
    pub max_iter: usize,
}

/// Outcome of [`pcg_solve`].
#[derive(Debug, Clone)]
pub struct PcgResult {
    /// Converged solution vector `x`.
    pub solution: Vec<f64>,
    /// Number of CG iterations performed.
    pub iterations: usize,
    /// Total operator applications (including final true-residual certification).
    pub matvecs: usize,
    /// Certified relative true residual `||b - A x||_2 / ||b||_2`.
    pub rel_residual: f64,
}

/// Preconditioned Conjugate Gradient for symmetric positive definite `A x = b`
/// on the orthogonal complement of `deflate`.
///
/// Refuses non-positive curvature (`p^T A p <= 0`), non-positive preconditioner
/// inner products (`r^T z <= 0`), non-finite outputs, or failure to meet the
/// certified true residual `rel_tol` within `max_iter` steps.
pub fn pcg_solve<F, P>(
    rhs: &[f64],
    op: &mut F,
    precond: &mut P,
    deflate: &[Vec<f64>],
    config: PcgConfig,
) -> Result<PcgResult>
where
    F: FnMut(&[f64], &mut [f64]) -> Result<()>,
    P: FnMut(&[f64], &mut [f64]) -> Result<()>,
{
    let n = rhs.len();
    if n == 0 {
        return Err(slq_err("PCG right-hand side is empty"));
    }
    if !all_finite(rhs) {
        return Err(slq_err("PCG right-hand side is not finite"));
    }
    if !(config.rel_tol > 0.0 && config.rel_tol.is_finite()) || config.max_iter == 0 {
        return Err(slq_err("invalid PCG tolerance or iteration budget"));
    }
    for q in deflate {
        if q.len() != n {
            return Err(slq_err("PCG deflation vector length mismatch"));
        }
    }

    let mut b = rhs.to_vec();
    if !deflate.is_empty() {
        project_out_basis(&mut b, deflate);
    }
    let b_norm = norm2_slice(&b);
    if !b_norm.is_finite() {
        return Err(slq_err("PCG projected RHS norm is not finite"));
    }
    if b_norm == 0.0 {
        return Ok(PcgResult {
            solution: vec![0.0; n],
            iterations: 0,
            matvecs: 0,
            rel_residual: 0.0,
        });
    }

    let mut x = vec![0.0; n];
    let mut r = b.clone();
    let mut z = vec![0.0; n];
    precond(&r, &mut z)?;
    if !all_finite(&z) {
        return Err(slq_err("PCG preconditioner produced non-finite output"));
    }
    if !deflate.is_empty() {
        project_out_basis(&mut z, deflate);
    }
    let mut p = z.clone();
    let mut rz = dot_slice(&r, &z);
    if !(rz > 0.0 && rz.is_finite()) {
        return Err(slq_err(
            "PCG preconditioner is not positive definite on residual",
        ));
    }

    let mut ap = vec![0.0; n];
    let mut matvecs = 0usize;

    for iter in 1..=config.max_iter {
        ap.fill(0.0);
        op(&p, &mut ap)?;
        matvecs += 1;
        if !all_finite(&ap) {
            return Err(slq_err("PCG operator produced non-finite output"));
        }
        if !deflate.is_empty() {
            project_out_basis(&mut ap, deflate);
        }
        let pap = dot_slice(&p, &ap);
        if !(pap > 0.0 && pap.is_finite()) {
            return Err(slq_err("PCG encountered non-positive operator curvature"));
        }
        let alpha = rz / pap;
        if !alpha.is_finite() {
            return Err(slq_err("PCG step length overflowed"));
        }
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        if !deflate.is_empty() {
            project_out_basis(&mut r, deflate);
        }
        let r_norm = norm2_slice(&r);
        if !r_norm.is_finite() {
            return Err(slq_err("PCG residual became non-finite"));
        }
        if r_norm <= config.rel_tol * b_norm {
            // Certify true residual explicitly.
            ap.fill(0.0);
            op(&x, &mut ap)?;
            matvecs += 1;
            if !all_finite(&ap) {
                return Err(slq_err("PCG true-residual matvec was not finite"));
            }
            if !deflate.is_empty() {
                project_out_basis(&mut ap, deflate);
            }
            let mut true_r = vec![0.0; n];
            for i in 0..n {
                true_r[i] = b[i] - ap[i];
            }
            let true_norm = norm2_slice(&true_r);
            let rel = true_norm / b_norm;
            if rel.is_finite() && rel <= config.rel_tol {
                return Ok(PcgResult {
                    solution: x,
                    iterations: iter,
                    matvecs,
                    rel_residual: rel,
                });
            }
            // Replace drifted recurrence residual and continue.
            r = true_r;
        }

        z.fill(0.0);
        precond(&r, &mut z)?;
        if !all_finite(&z) {
            return Err(slq_err("PCG preconditioner produced non-finite output"));
        }
        if !deflate.is_empty() {
            project_out_basis(&mut z, deflate);
        }
        let rz_new = dot_slice(&r, &z);
        if !(rz_new >= 0.0 && rz_new.is_finite()) {
            return Err(slq_err("PCG preconditioner lost positive definiteness"));
        }
        let beta = if rz > 0.0 { rz_new / rz } else { 0.0 };
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        rz = rz_new;
    }

    Err(slq_err(
        "PCG exhausted iteration budget before meeting rel_tol",
    ))
}

/// One deflated approximate eigenpair `(eigenvalue, vector)` with verified
/// residual norm `||A v - eigenvalue * v||_2`.
#[derive(Debug, Clone)]
pub struct DeflatedPair {
    /// Ritz eigenvalue.
    pub eigenvalue: f64,
    /// Unit eigenvector orthogonal to the kernel and other deflated pairs.
    pub vector: Vec<f64>,
    /// Verified residual norm `||P_kernel A v - eigenvalue * v||_2`.
    pub residual: f64,
}

/// Configuration for [`lanczos_extreme_pairs`].
#[derive(Debug, Clone, Copy)]
pub struct ExtremePairsConfig {
    /// Eigenvalues strictly below this threshold are classified as negative
    /// (default `0.0`).
    pub neg_threshold: f64,
    /// Maximum absolute residual for negative eigenvalues.
    pub neg_abs_tol: f64,
    /// Target relative error on `ln |lambda|` for negative eigenvalues; requires
    /// `residual^2 / (|lambda| * gap) <= neg_log_tol`.
    pub neg_log_tol: f64,
    /// Number of lowest positive eigenpairs to keep and converge (at least 1 so
    /// a positive lower bound can be certified).
    pub num_low_positive: usize,
    /// Number of highest positive eigenpairs to keep and converge for deflation.
    pub num_high_positive: usize,
    /// Target residual for converged positive deflation pairs.
    pub pos_residual_tol: f64,
    /// Krylov subspace size before thick restart (`subspace_dim > kept`).
    pub subspace_dim: usize,
    /// Maximum number of thick-restart cycles.
    pub max_restarts: usize,
    /// Deterministic seed for initial and restart vectors.
    pub seed: u64,
}

/// Output of [`lanczos_extreme_pairs`].
#[derive(Debug, Clone)]
pub struct ExtremePairsReport {
    /// Converged negative eigenpairs (`eigenvalue < neg_threshold`), sorted ascending.
    pub negative: Vec<DeflatedPair>,
    /// Converged positive eigenpairs for optional SLQ variance deflation.
    pub positive_deflation: Vec<DeflatedPair>,
    /// Heuristic Ritz-minus-residual estimate (`min_positive_ritz - residual`)
    /// of the smallest positive eigenvalue on `kernel^perp`. This is a
    /// spectral scale estimate, not a guaranteed lower bound on `lambda_min`
    /// unless `span_exhausted` is true.
    pub positive_lower_estimate: f64,
    /// Smallest positive Ritz value observed.
    pub min_positive_ritz: f64,
    /// Largest positive Ritz value observed.
    pub max_positive_ritz: f64,
    /// True if Lanczos spanned the entire `dim - kernel.len()` subspace.
    pub span_exhausted: bool,
    /// Total operator applications used.
    pub matvecs: usize,
    /// True if negative modes, lowest-positive residual ratio, and target
    /// positive deflation residuals all converged within `max_restarts`.
    pub converged: bool,
}

fn fresh_orthogonal_unit(
    n: usize,
    rng: &mut SplitMix64,
    kernel: &[Vec<f64>],
    basis: &[Vec<f64>],
) -> Option<Vec<f64>> {
    for _ in 0..16 {
        let mut v: Vec<f64> = (0..n).map(|_| rng.next_unit_f64()).collect();
        if !kernel.is_empty() {
            project_out_basis(&mut v, kernel);
        }
        if !basis.is_empty() {
            project_out_basis(&mut v, basis);
        }
        let nrm = norm2_slice(&v);
        if nrm.is_finite() && nrm > 1e-12 {
            for slot in &mut v {
                *slot /= nrm;
            }
            return Some(v);
        }
    }
    // Deterministic fallback sweep over coordinate axes.
    for axis in 0..n {
        let mut v = vec![0.0; n];
        v[axis] = 1.0;
        if !kernel.is_empty() {
            project_out_basis(&mut v, kernel);
        }
        if !basis.is_empty() {
            project_out_basis(&mut v, basis);
        }
        let nrm = norm2_slice(&v);
        if nrm.is_finite() && nrm > (n as f64) * f64::EPSILON {
            for slot in &mut v {
                *slot /= nrm;
            }
            return Some(v);
        }
    }
    None
}

/// Deflated thick-restart Lanczos on the orthogonal complement of `kernel`.
///
/// Finds negative eigenvalues (with residual checks on `ln |lambda|`),
/// estimates the bottom of the positive spectrum, and collects converged
/// extreme positive eigenpairs for SLQ variance deflation.
pub fn lanczos_extreme_pairs<F>(
    dim: usize,
    op: &mut F,
    kernel: &[Vec<f64>],
    config: ExtremePairsConfig,
) -> Result<ExtremePairsReport>
where
    F: FnMut(&[f64], &mut [f64]) -> Result<()>,
{
    if dim == 0 {
        return Err(slq_err("operator dimension must be positive"));
    }
    for q in kernel {
        if q.len() != dim || !all_finite(q) {
            return Err(slq_err("invalid kernel basis vector"));
        }
    }
    let effective_dim = dim
        .checked_sub(kernel.len())
        .ok_or_else(|| slq_err("kernel dimension exceeds operator dimension"))?;
    if effective_dim == 0 {
        return Ok(ExtremePairsReport {
            negative: Vec::new(),
            positive_deflation: Vec::new(),
            positive_lower_estimate: 1.0,
            min_positive_ritz: 1.0,
            max_positive_ritz: 1.0,
            span_exhausted: true,
            matvecs: 0,
            converged: true,
        });
    }

    let mut rng = SplitMix64::new(config.seed);
    let m_max = config.subspace_dim.max(6).min(effective_dim);
    let want_low = config.num_low_positive.max(1);
    let want_high = config.num_high_positive;

    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(m_max + 1);
    let start = fresh_orthogonal_unit(dim, &mut rng, kernel, &[])
        .ok_or_else(|| slq_err("failed to construct start vector orthogonal to kernel"))?;
    basis.push(start);

    let mut kept = 0usize;
    let mut kept_theta: Vec<f64> = Vec::new();
    let mut kept_sigma: Vec<f64> = Vec::new();
    let mut matvecs = 0usize;
    let mut w = vec![0.0; dim];

    let mut last_values: Vec<f64> = Vec::new();
    let mut last_vecs: Vec<Vec<f64>> = Vec::new();
    let mut last_res: Vec<f64> = Vec::new();
    let mut final_neg_ok = false;
    let mut final_low_bound_ok = false;
    let mut final_target_pos_ok = false;
    let mut final_span_exhausted = false;

    for cycle in 0..=config.max_restarts {
        // Build projected matrix B of size m x m via thick-restart Lanczos.
        let mut alphas = Vec::with_capacity(m_max);
        let mut betas = Vec::with_capacity(m_max);
        let mut restart_couplings = Vec::with_capacity(kept);
        let mut final_beta = 0.0;
        let mut next_q: Option<Vec<f64>> = None;

        for j in kept..m_max {
            w.fill(0.0);
            op(&basis[j], &mut w)?;
            matvecs += 1;
            if !all_finite(&w) {
                return Err(slq_err(
                    "operator produced non-finite output in thick-restart Lanczos",
                ));
            }
            if !kernel.is_empty() {
                project_out_basis(&mut w, kernel);
            }

            if j == kept && kept > 0 {
                restart_couplings.clear();
                for i in 0..kept {
                    let s = dot_slice(&basis[i], &w);
                    restart_couplings.push(s);
                    for (slot, &qi) in w.iter_mut().zip(basis[i].iter()) {
                        *slot -= s * qi;
                    }
                }
            }

            let a_j = dot_slice(&basis[j], &w);
            if !a_j.is_finite() {
                return Err(slq_err("non-finite Lanczos diagonal entry"));
            }
            alphas.push(a_j);
            for (slot, &qj) in w.iter_mut().zip(basis[j].iter()) {
                *slot -= a_j * qj;
            }
            if j > kept {
                let b_prev = betas[j - kept - 1];
                for (slot, &qprev) in w.iter_mut().zip(basis[j - 1].iter()) {
                    *slot -= b_prev * qprev;
                }
            }
            if !kernel.is_empty() {
                project_out_basis(&mut w, kernel);
            }
            project_out_basis(&mut w, &basis);
            let b_j = norm2_slice(&w);
            if !b_j.is_finite() {
                return Err(slq_err("non-finite Lanczos subdiagonal entry"));
            }

            let built = j + 1;
            let breakdown_tol = (dim as f64).sqrt() * f64::EPSILON * (1.0 + a_j.abs());
            if built == m_max || built == effective_dim {
                if b_j > breakdown_tol && built < effective_dim {
                    final_beta = b_j;
                    next_q = Some(w.iter().map(|&x| x / b_j).collect());
                } else {
                    final_beta = 0.0;
                    next_q = None;
                }
                break;
            }

            if b_j <= breakdown_tol {
                betas.push(0.0);
                if let Some(fresh) = fresh_orthogonal_unit(dim, &mut rng, kernel, &basis) {
                    basis.push(fresh);
                } else {
                    final_beta = 0.0;
                    next_q = None;
                    break;
                }
            } else {
                betas.push(b_j);
                basis.push(w.iter().map(|&x| x / b_j).collect());
            }
        }

        let m = basis.len();
        let mut b_mat = vec![0.0; m * m];
        for i in 0..kept {
            b_mat[i * m + i] = kept_theta[i];
            if kept < m {
                let sig = if i < restart_couplings.len() {
                    0.5 * (kept_sigma[i] + restart_couplings[i])
                } else {
                    kept_sigma[i]
                };
                b_mat[i * m + kept] = sig;
                b_mat[kept * m + i] = sig;
            }
        }
        for (idx, &a_val) in alphas.iter().enumerate() {
            let row = kept + idx;
            b_mat[row * m + row] = a_val;
            if idx < betas.len() && row + 1 < m {
                let b_val = betas[idx];
                b_mat[row * m + (row + 1)] = b_val;
                b_mat[(row + 1) * m + row] = b_val;
            }
        }

        let eig = sym_eigh(&b_mat, m)?;
        let mut ritz_vecs = Vec::with_capacity(m);
        let mut ritz_res = Vec::with_capacity(m);
        for col in 0..m {
            let mut rv = vec![0.0; dim];
            for row in 0..m {
                let coeff = eig.vectors[row * m + col];
                if coeff == 0.0 {
                    continue;
                }
                for d_idx in 0..dim {
                    rv[d_idx] += coeff * basis[row][d_idx];
                }
            }
            let r_est = final_beta * eig.vectors[(m - 1) * m + col].abs();
            ritz_vecs.push(rv);
            ritz_res.push(r_est);
        }

        // Check convergence of all negative eigenvalues and the lowest positive eigenvalue.
        let mut neg_indices = Vec::new();
        let mut pos_indices = Vec::new();
        for idx in 0..m {
            if eig.values[idx] < config.neg_threshold {
                neg_indices.push(idx);
            } else {
                pos_indices.push(idx);
            }
        }

        let min_pos_val = pos_indices.first().map(|&i| eig.values[i]).unwrap_or(1.0);
        let min_pos_res = pos_indices.first().map(|&i| ritz_res[i]).unwrap_or(0.0);

        let mut neg_ok = true;
        for &idx in &neg_indices {
            let lam = eig.values[idx];
            let res = ritz_res[idx];
            let gap = (min_pos_val - lam).abs();
            let denom = lam.abs() * gap;
            if denom <= 0.0 || !denom.is_finite() {
                neg_ok = false;
                continue;
            }
            let log_err = (res * res) / denom;
            if !(res <= config.neg_abs_tol && log_err <= config.neg_log_tol) {
                neg_ok = false;
            }
        }

        let lower_bound_cand = min_pos_val - min_pos_res;
        let low_bound_ok = lower_bound_cand > 0.0 && min_pos_res <= 0.25 * min_pos_val;

        // Check whether desired positive deflation pairs converged.
        let mut target_pos_ok = true;
        let check_low = want_low.min(pos_indices.len());
        for &idx in pos_indices.iter().take(check_low) {
            if ritz_res[idx] > config.pos_residual_tol {
                target_pos_ok = false;
            }
        }
        let check_high = want_high.min(pos_indices.len().saturating_sub(check_low));
        for &idx in pos_indices.iter().rev().take(check_high) {
            if ritz_res[idx] > config.pos_residual_tol {
                target_pos_ok = false;
            }
        }

        last_values = eig.values;
        last_vecs = ritz_vecs;
        last_res = ritz_res;

        let span_exhausted = m == effective_dim || next_q.is_none();
        final_neg_ok = neg_ok;
        final_low_bound_ok = low_bound_ok;
        final_target_pos_ok = target_pos_ok;
        final_span_exhausted = span_exhausted;
        if (neg_ok && low_bound_ok && target_pos_ok)
            || span_exhausted
            || cycle == config.max_restarts
        {
            break;
        }

        // Select Ritz pairs to retain across the thick restart:
        // all negative modes + lowest positive + highest positive, capped strictly below m_max - 2.
        let max_keep = m_max.saturating_sub(4).max(1).min(m - 1);
        let mut selected: Vec<usize> = Vec::new();
        for &idx in &neg_indices {
            if selected.len() < max_keep {
                selected.push(idx);
            }
        }
        let low_keep = (want_low + 4).min(pos_indices.len());
        for &idx in pos_indices.iter().take(low_keep) {
            if selected.len() < max_keep && !selected.contains(&idx) {
                selected.push(idx);
            }
        }
        let high_keep = (want_high + 2).min(pos_indices.len());
        for &idx in pos_indices.iter().rev().take(high_keep) {
            if selected.len() < max_keep && !selected.contains(&idx) {
                selected.push(idx);
            }
        }
        selected.sort_unstable();

        kept = selected.len();
        kept_theta = selected.iter().map(|&idx| last_values[idx]).collect();
        kept_sigma = selected
            .iter()
            .map(|&idx| final_beta * eig.vectors[(m - 1) * m + idx])
            .collect();

        let mut new_basis = Vec::with_capacity(m_max + 1);
        for &idx in &selected {
            let mut v = last_vecs[idx].clone();
            if !kernel.is_empty() {
                project_out_basis(&mut v, kernel);
            }
            project_out_basis(&mut v, &new_basis);
            let nrm = norm2_slice(&v);
            if nrm > 0.0 && nrm.is_finite() {
                for slot in &mut v {
                    *slot /= nrm;
                }
            }
            new_basis.push(v);
        }
        let mut q_restart = match next_q {
            Some(q) => q,
            None => break,
        };
        if !kernel.is_empty() {
            project_out_basis(&mut q_restart, kernel);
        }
        project_out_basis(&mut q_restart, &new_basis);
        let qn = norm2_slice(&q_restart);
        if !(qn > 1e-12 && qn.is_finite()) {
            break;
        }
        for slot in &mut q_restart {
            *slot /= qn;
        }
        new_basis.push(q_restart);
        basis = new_basis;
    }

    // Build final report from last_values / last_vecs / last_res.
    let mut negative = Vec::new();
    let mut pos_candidates = Vec::new();
    for idx in 0..last_values.len() {
        let pair = DeflatedPair {
            eigenvalue: last_values[idx],
            vector: last_vecs[idx].clone(),
            residual: last_res[idx],
        };
        if pair.eigenvalue < config.neg_threshold {
            negative.push(pair);
        } else {
            pos_candidates.push(pair);
        }
    }

    let min_positive_ritz = pos_candidates.first().map(|p| p.eigenvalue).unwrap_or(1.0);
    let min_positive_res = pos_candidates.first().map(|p| p.residual).unwrap_or(0.0);
    let max_positive_ritz = pos_candidates.last().map(|p| p.eigenvalue).unwrap_or(1.0);

    let positive_lower_estimate = if pos_candidates.is_empty() {
        1.0
    } else {
        min_positive_ritz - min_positive_res
    };

    // Honor all convergence criteria including the restart limit.
    let mut converged = positive_lower_estimate > 0.0
        && positive_lower_estimate.is_finite()
        && final_neg_ok
        && final_low_bound_ok
        && final_target_pos_ok;
    for pair in &negative {
        let gap = (min_positive_ritz - pair.eigenvalue).abs();
        let denom = pair.eigenvalue.abs() * gap;
        if denom <= 0.0 || !denom.is_finite() {
            converged = false;
            continue;
        }
        let log_err = (pair.residual * pair.residual) / denom;
        if !(pair.residual <= config.neg_abs_tol && log_err <= config.neg_log_tol) {
            converged = false;
        }
    }

    // Keep positive Ritz pairs that are converged well enough for variance deflation
    // (those furthest from 1.0 in |ln(lambda)|).
    let mut positive_deflation = Vec::new();
    if positive_lower_estimate > 0.0 {
        let mut eligible: Vec<DeflatedPair> = pos_candidates
            .into_iter()
            .filter(|p| {
                p.eigenvalue > 0.0
                    && p.residual <= config.pos_residual_tol
                    && (p.residual * p.residual)
                        / (2.0 * positive_lower_estimate * positive_lower_estimate)
                        <= config.neg_log_tol
            })
            .collect();
        eligible.sort_by(|a, b| b.eigenvalue.ln().abs().total_cmp(&a.eigenvalue.ln().abs()));
        let cap = want_low.saturating_add(want_high);
        eligible.truncate(cap);
        positive_deflation = eligible;
    }

    Ok(ExtremePairsReport {
        negative,
        positive_deflation,
        positive_lower_estimate,
        min_positive_ritz,
        max_positive_ritz,
        span_exhausted: final_span_exhausted,
        matvecs,
        converged,
    })
}

/// Probe distribution for [`slq_logdet_deflated`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeStrategy {
    /// Standard i.i.d. Rademacher `{-1, +1}^n` probes projected onto the
    /// non-deflated subspace.
    Rademacher,
    /// Strided-color randomized probing: partitions indices `0..n` into
    /// `stride` disjoint color classes grouped by `block_size` (e.g. `block_size = d`
    /// for Fourier modes so modes in the same color satisfy `|k - k'| >= stride`).
    /// Each sample sums one independent Rademacher probe per color class,
    /// eliminating all couplings at distance `< stride` from the variance while
    /// remaining strictly unbiased.
    StridedColors {
        /// Number of disjoint color classes (`>= 1`).
        stride: usize,
        /// Number of contiguous coordinates per block (`>= 1`).
        block_size: usize,
    },
}

/// Stopping rule for stochastic trace sampling in [`slq_logdet_deflated`] and
/// [`slq_logdet_hutchpp`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlqStoppingRule {
    /// Draw exactly `max_samples` independent samples without intermediate
    /// peeking, applying the two-sided Student-t quantile at significance level
    /// `alpha = 1 - confidence_level`.
    FixedSamples,
    /// Test stopping only at geometric checkpoints
    /// `min_samples, 2*min_samples, ..., max_samples` (`K` checkpoints total)
    /// using Bonferroni alpha-spending `alpha_k = (1 - confidence_level) / K`.
    /// Under normal sampling, this preserves nominal simultaneous coverage
    /// `>= confidence_level` across all stopping checkpoints.
    AlphaSpending,
}

/// Configuration for [`slq_logdet_deflated`] and [`slq_logdet_hutchpp`].
#[derive(Debug, Clone, Copy)]
pub struct SlqConfig {
    /// Probe distribution.
    pub strategy: ProbeStrategy,
    /// Stopping rule controlling Student-t quantile calibration.
    pub stopping_rule: SlqStoppingRule,
    /// Minimum independent samples before the first stopping check (`>= 2`).
    /// Ignored when `stopping_rule == SlqStoppingRule::FixedSamples`.
    pub min_samples: usize,
    /// Maximum independent samples before stopping (`>= min_samples`).
    pub max_samples: usize,
    /// Maximum Lanczos steps per probe vector.
    pub max_lanczos_steps: usize,
    /// Quadrature check stride inside [`lanczos_quadrature_log`].
    pub quad_check_every: usize,
    /// Target total error estimate on `ln |det'(A)|`.
    pub target_tol: f64,
    /// Nominal two-sided confidence level in `(0.5, 1.0)` (e.g. `0.995`).
    /// Exact Student-t quantiles are computed for `df = samples - 1` (with
    /// Bonferroni alpha-spending when `stopping_rule == AlphaSpending`).
    pub confidence_level: f64,
    /// Deterministic PRNG seed.
    pub seed: u64,
}

/// Output of [`slq_logdet_deflated`] and [`slq_logdet_hutchpp`].
///
/// The error fields represent a-posteriori stochastic confidence intervals
/// combined with quadrature and first-order deflation error estimates; they
/// are probabilistic estimates, not deterministic enclosures.
#[derive(Debug, Clone)]
pub struct SlqLogDetReport {
    /// Estimated `ln |det'(A)|` on `kernel^perp`.
    pub logabs_det: f64,
    /// Contribution `sum ln |lambda_j^-|` of deflated negative modes.
    pub negative_logabs: f64,
    /// Contribution `sum ln(theta_i^+)` of deflated positive modes.
    pub deflated_positive_logabs: f64,
    /// SLQ trace estimate on the non-deflated positive complement.
    pub stochastic_trace: f64,
    /// First-order deflation approximation error estimate (negative + positive
    /// Ritz residual terms).
    pub deflation_error_estimate: f64,
    /// Quadrature error estimate averaged over samples plus any deterministic
    /// Hutch++ subspace quadrature error.
    pub quadrature_error_estimate: f64,
    /// Sample standard error `s_m / sqrt(m)` across independent samples.
    pub stochastic_se: f64,
    /// Student-t confidence half-width `t_{1 - alpha_eff/2, m-1} * stochastic_se`.
    pub ci_halfwidth: f64,
    /// Combined error estimate
    /// `deflation_error_estimate + quadrature_error_estimate + ci_halfwidth`.
    pub total_error_estimate: f64,
    /// Stated nominal simultaneous confidence level in `(0.5, 1.0)`.
    pub confidence_level: f64,
    /// Stopping rule used to calibrate `ci_halfwidth`.
    pub stopping_rule: SlqStoppingRule,
    /// Number of independent probe passes completed.
    pub samples: usize,
    /// Total probe vectors evaluated (`samples * num_colors + hutchpp_basis`).
    pub probes: usize,
    /// Total operator applications used by SLQ.
    pub matvecs: usize,
    /// True if `total_error_estimate <= config.target_tol`.
    /// This is a probabilistic acceptance check, not a deterministic guarantee.
    pub accepted: bool,
}

/// Continued fraction for the regularized incomplete beta function `I_x(a, b)`
/// (modified Lentz algorithm, Numerical Recipes §6.4).
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const MAX_IT: usize = 200;
    const FPMIN: f64 = 1e-300;
    const EPS: f64 = 3e-15;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0_f64;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = d.recip();
    let mut h = d;
    for m in 1..=MAX_IT {
        let mf = m as f64;
        let m2 = 2.0 * mf;
        let aa = mf * (b - mf) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = d.recip();
        h *= d * c;

        let aa2 = -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa2 * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa2 / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = d.recip();
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() <= EPS {
            break;
        }
    }
    h
}

/// Regularized incomplete beta function `I_x(a, b)` for `0 <= x <= 1`, `a > 0`, `b > 0`.
fn reg_inc_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let ln_bt =
        libm::lgamma(a + b) - libm::lgamma(a) - libm::lgamma(b) + a * x.ln() + b * (1.0 - x).ln();
    let bt = ln_bt.exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(a, b, x) / a
    } else {
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    }
}

/// Exact two-sided Student-t tail probability `P(|T_df| >= t)` for `t >= 0`.
fn student_t_two_sided_tail(df: f64, t: f64) -> f64 {
    if t <= 0.0 {
        return 1.0;
    }
    let x = df / (df + t * t);
    reg_inc_beta(x, 0.5 * df, 0.5)
}

/// Exact two-sided Student-t critical value `t > 0` satisfying
/// `P(|T_df| >= t) = alpha` (equivalently the one-sided `1 - alpha / 2`
/// quantile of Student's t distribution with `df >= 1` degrees of freedom).
pub fn student_t_quantile(df: usize, alpha: f64) -> Result<f64> {
    if df == 0 {
        return Err(slq_err("Student-t degrees of freedom must be >= 1"));
    }
    if !(alpha > 0.0 && alpha < 1.0 && alpha.is_finite()) {
        return Err(slq_err("Student-t alpha must lie strictly in (0, 1)"));
    }
    if df == 1 {
        let q = (0.5 * std::f64::consts::PI * alpha).tan().recip();
        if !q.is_finite() || q <= 0.0 {
            return Err(slq_err("Student-t quantile overflowed for df = 1"));
        }
        return Ok(q);
    }
    if df == 2 {
        let denom = alpha * (2.0 - alpha);
        let q = (2.0 * (1.0 - alpha) * (1.0 - alpha) / denom).sqrt();
        if !q.is_finite() || q <= 0.0 {
            return Err(slq_err("Student-t quantile overflowed for df = 2"));
        }
        return Ok(q);
    }
    let df_f = df as f64;
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    while student_t_two_sided_tail(df_f, hi) > alpha {
        hi *= 2.0;
        if !hi.is_finite() || hi > 1e12 {
            return Err(slq_err("Student-t quantile bracketing failed"));
        }
    }
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if student_t_two_sided_tail(df_f, mid) > alpha {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo <= 1e-13 * (1.0 + hi) {
            break;
        }
    }
    let q = 0.5 * (lo + hi);
    if !(q > 0.0 && q.is_finite()) {
        return Err(slq_err("Student-t quantile was not finite"));
    }
    Ok(q)
}

/// Build the predetermined checkpoint schedule for `SlqStoppingRule`.
fn build_checkpoints(min_samples: usize, max_samples: usize, rule: SlqStoppingRule) -> Vec<usize> {
    match rule {
        SlqStoppingRule::FixedSamples => vec![max_samples],
        SlqStoppingRule::AlphaSpending => {
            let mut out = Vec::new();
            let mut cur = min_samples;
            while cur < max_samples {
                out.push(cur);
                let next = cur.saturating_mul(2);
                if next <= cur {
                    break;
                }
                cur = next;
            }
            out.push(max_samples);
            out
        }
    }
}

/// Deflation subspace and spectral-scale inputs for [`slq_logdet_deflated`],
/// [`slq_logdet_deflated_with_cv`], and [`slq_logdet_hutchpp`].
#[derive(Debug, Clone, Copy)]
pub struct SlqDeflation<'a> {
    /// Orthonormalized kernel basis vectors projected out before probing.
    pub kernel: &'a [Vec<f64>],
    /// Converged negative eigenpairs deflated analytically.
    pub negative_pairs: &'a [DeflatedPair],
    /// Converged positive eigenpairs deflated analytically.
    pub positive_deflation: &'a [DeflatedPair],
    /// Positive spectral scale estimate (`> 0`) used in first-order deflation
    /// residual estimates.
    pub deflation_lower_estimate: f64,
    /// Optional valid lower bound `0 < a <= lambda_min(A_rem)` enabling left
    /// Gauss-Radau two-sided brackets. Pass `None` to use adaptive Gauss
    /// Lanczos quadrature without a lower-bound assumption.
    pub radau_lower_bound: Option<f64>,
}

/// Optional exact polynomial moments `tr(A_rem)` and `tr(A_rem^2)` on the
/// non-deflated subspace for zero-extra-matvec variance reduction.
#[derive(Debug, Clone, Copy, Default)]
pub struct SlqMoments {
    /// Exact `tr(A_rem)` on the complement of the deflated subspace.
    pub trace: Option<f64>,
    /// Exact `tr(A_rem^2)` on the complement of the deflated subspace.
    pub trace_sq: Option<f64>,
}

/// Combined Hutch++, polynomial-moment, and control-variate trace options for
/// [`slq_logdet_hutchpp`].
#[derive(Debug, Clone, Copy)]
pub struct SlqHutchPpOptions {
    /// Exact moments `tr(A_rem)` and `tr(A_rem^2)` when available.
    pub moments: SlqMoments,
    /// Full trace `tr(L_cv)` of the symmetric control variate (`0.0` if unused).
    pub cv_full_trace: f64,
    /// Hutch++ randomized range sketch parameters (`sketch_dim = 0` disables).
    pub hutchpp: HutchPpConfig,
}

/// Deflated Stochastic Lanczos Quadrature for `ln |det'(A)|` on `kernel^perp`.
///
/// Deflates `kernel`, `negative_pairs`, and `positive_deflation` by projecting
/// onto their orthogonal complement `P_rem = I - U_all U_all^T`, evaluates
/// `sum ln |lambda_j^-| + sum ln(theta_i^+) + tr(ln(P_rem A P_rem |_{range(P_rem)}))`,
/// and reports a-posteriori stochastic confidence intervals plus quadrature
/// and first-order deflation error estimates.
pub fn slq_logdet_deflated<F>(
    dim: usize,
    op: &mut F,
    deflation: SlqDeflation<'_>,
    config: SlqConfig,
) -> Result<SlqLogDetReport>
where
    F: FnMut(&[f64], &mut [f64]) -> Result<()>,
{
    let mut zero_cv = |_z: &[f64]| -> Result<f64> { Ok(0.0) };
    slq_logdet_deflated_with_cv(dim, op, deflation, 0.0, &mut zero_cv, config)
}

/// Configuration for Hutch++ randomized subspace variance reduction inside
/// [`slq_logdet_hutchpp`].
#[derive(Debug, Clone, Copy)]
pub struct HutchPpConfig {
    /// Number of independent Rademacher sketch vectors `r` used to build the
    /// deterministic trace subspace `Q` (`0` disables Hutch++).
    pub sketch_dim: usize,
    /// Number of power iterations (`>= 1`) applied to the sketch operator
    /// (`1` forms `Y = B S`, `2` forms `Y = B^2 S`, etc.).
    pub power_iters: usize,
    /// Independent PRNG seed for the sketch matrix `S`.
    pub sketch_seed: u64,
}

/// Deflated Stochastic Lanczos Quadrature for `ln |det'(A)|` on `kernel^perp`
/// with an optional symmetric control variate `L_cv`.
///
/// Given `cv_full_trace = tr(L_cv)` and `cv_quad(z) = z^T L_cv z`, this uses
/// the exact unbiased identity
/// `tr(ln(M_rem)) = tr(P_rem L_cv P_rem) + E_z[z^T ln(M_rem) z - z^T L_cv z]`,
/// where `tr(P_rem L_cv P_rem) = tr(L_cv) - sum_{u in U_all} u^T L_cv u`.
pub fn slq_logdet_deflated_with_cv<F, C>(
    dim: usize,
    op: &mut F,
    deflation: SlqDeflation<'_>,
    cv_full_trace: f64,
    cv_quad: &mut C,
    config: SlqConfig,
) -> Result<SlqLogDetReport>
where
    F: FnMut(&[f64], &mut [f64]) -> Result<()>,
    C: FnMut(&[f64]) -> Result<f64>,
{
    let mut id_matvec = |x: &[f64], y: &mut [f64]| -> Result<()> {
        y.copy_from_slice(x);
        Ok(())
    };
    slq_logdet_hutchpp(
        dim,
        op,
        deflation,
        cv_quad,
        &mut id_matvec,
        SlqHutchPpOptions {
            moments: SlqMoments::default(),
            cv_full_trace,
            hutchpp: HutchPpConfig {
                sketch_dim: 0,
                power_iters: 1,
                sketch_seed: 0,
            },
        },
        config,
    )
}

/// Deflated Hutch++ Stochastic Lanczos Quadrature for `ln |det'(A)|` on
/// `kernel^perp`, combining:
/// 1. Exact deflation of `kernel`, `negative_pairs`, and `positive_deflation`;
/// 2. Optional exact polynomial moment subtraction (`moments.trace = tr(A_rem)`
///    and optional `moments.trace_sq = tr(A_rem^2)`):
///    uses `tr(ln(A_rem)) = tr(A_rem - I) - 0.5 * tr((A_rem - I)^2) + tr(R_3(A_rem))`,
///    cancelling both `O(A_rem - I)` and `O((A_rem - I)^2)` at zero extra matvecs
///    via `QuadratureEstimate::first_moment` and `second_moment`;
/// 3. An optional symmetric control variate `L_cv` (`options.cv_full_trace`, `cv_quad`);
/// 4. An optional Hutch++ randomized range sketch `Q = orth(P_rem (A - M_cv)^p S)`
///    (Meyer-Musco-Musco-Woodruff 2021) where `cv_matvec` applies `M_cv` (or
///    `I` when no control variate is used). `tr(Q^T B Q)` is evaluated
///    deterministically via `lanczos_quadrature_log` (zero extra deflation error)
///    and the orthogonal complement `P_rem - Q Q^T` is estimated by independent
///    SLQ probes under `config.stopping_rule`.
pub fn slq_logdet_hutchpp<F, C, M>(
    dim: usize,
    op: &mut F,
    deflation: SlqDeflation<'_>,
    cv_quad: &mut C,
    cv_matvec: &mut M,
    options: SlqHutchPpOptions,
    config: SlqConfig,
) -> Result<SlqLogDetReport>
where
    F: FnMut(&[f64], &mut [f64]) -> Result<()>,
    C: FnMut(&[f64]) -> Result<f64>,
    M: FnMut(&[f64], &mut [f64]) -> Result<()>,
{
    let kernel = deflation.kernel;
    let negative_pairs = deflation.negative_pairs;
    let positive_deflation = deflation.positive_deflation;
    let deflation_lower_estimate = deflation.deflation_lower_estimate;
    let radau_lower_bound = deflation.radau_lower_bound;
    let exact_trace_rem = options.moments.trace;
    let exact_trace_sq_rem = options.moments.trace_sq;
    let cv_full_trace = options.cv_full_trace;
    let hutchpp = options.hutchpp;
    if dim == 0 {
        return Err(slq_err("SLQ dimension must be positive"));
    }
    if !(deflation_lower_estimate > 0.0 && deflation_lower_estimate.is_finite()) {
        return Err(slq_err(
            "SLQ deflation_lower_estimate must be finite and positive",
        ));
    }
    if let Some(lb) = radau_lower_bound
        && !(lb > 0.0 && lb.is_finite())
    {
        return Err(slq_err("SLQ radau_lower_bound must be finite and positive"));
    }
    if !cv_full_trace.is_finite() {
        return Err(slq_err("SLQ control-variate trace must be finite"));
    }
    if !(config.target_tol > 0.0 && config.target_tol.is_finite()) {
        return Err(slq_err("SLQ target_tol must be finite and positive"));
    }
    if !(config.confidence_level > 0.5
        && config.confidence_level < 1.0
        && config.confidence_level.is_finite())
    {
        return Err(slq_err(
            "SLQ confidence_level must lie strictly in (0.5, 1.0)",
        ));
    }
    let min_samples = config.min_samples.max(2);
    let max_samples = config.max_samples.max(min_samples);
    let checkpoints = build_checkpoints(min_samples, max_samples, config.stopping_rule);
    let alpha_total = 1.0 - config.confidence_level;
    let alpha_per_check = alpha_total / (checkpoints.len() as f64);

    let (stride, block_size) = match config.strategy {
        ProbeStrategy::Rademacher => (1usize, 1usize),
        ProbeStrategy::StridedColors { stride, block_size } => {
            if stride == 0 || block_size == 0 {
                return Err(slq_err(
                    "StridedColors requires stride >= 1 and block_size >= 1",
                ));
            }
            (stride, block_size)
        }
    };

    // Contribution + first-order deflation residual estimate of deflated negative modes.
    let mut negative_logabs = 0.0;
    let mut deflation_error_estimate = 0.0;
    for pair in negative_pairs {
        if !(pair.eigenvalue < 0.0 && pair.eigenvalue.is_finite()) {
            return Err(slq_err("negative_pairs contains a non-negative eigenvalue"));
        }
        negative_logabs += pair.eigenvalue.abs().ln();
        let gap = (deflation_lower_estimate - pair.eigenvalue).abs();
        let denom_eig = pair.eigenvalue.abs() * gap;
        let denom_schur = pair.eigenvalue.abs() * deflation_lower_estimate;
        if denom_eig <= 0.0
            || !denom_eig.is_finite()
            || denom_schur <= 0.0
            || !denom_schur.is_finite()
        {
            return Err(slq_err(
                "degenerate negative eigenvalue deflation denominator",
            ));
        }
        deflation_error_estimate += (pair.residual * pair.residual) / denom_eig
            + (pair.residual * pair.residual) / denom_schur;
    }

    // Contribution + first-order deflation residual estimate of deflated positive modes.
    let mut deflated_positive_logabs = 0.0;
    for pair in positive_deflation {
        if !(pair.eigenvalue > 0.0 && pair.eigenvalue.is_finite()) {
            return Err(slq_err(
                "positive_deflation contains a non-positive eigenvalue",
            ));
        }
        deflated_positive_logabs += pair.eigenvalue.ln();
        let denom_schur = pair.eigenvalue * deflation_lower_estimate;
        let denom_self = 2.0 * pair.eigenvalue * pair.eigenvalue;
        if denom_schur <= 0.0
            || !denom_schur.is_finite()
            || denom_self <= 0.0
            || !denom_self.is_finite()
        {
            return Err(slq_err(
                "degenerate positive eigenvalue deflation denominator",
            ));
        }
        deflation_error_estimate += (pair.residual * pair.residual) / denom_schur
            + (pair.residual * pair.residual) / denom_self;
    }

    // Orthonormalize the full deflated basis: kernel + negative + positive_deflation.
    let mut raw_deflate: Vec<Vec<f64>> =
        Vec::with_capacity(kernel.len() + negative_pairs.len() + positive_deflation.len());
    for q in kernel {
        raw_deflate.push(q.clone());
    }
    for p in negative_pairs {
        raw_deflate.push(p.vector.clone());
    }
    for p in positive_deflation {
        raw_deflate.push(p.vector.clone());
    }
    let probe_deflate = orthonormalize_vectors(&raw_deflate, dim)?;

    let total_deflated = probe_deflate.len();
    if total_deflated >= dim {
        let logabs_det = negative_logabs + deflated_positive_logabs;
        let accepted = deflation_error_estimate <= config.target_tol;
        return Ok(SlqLogDetReport {
            logabs_det,
            negative_logabs,
            deflated_positive_logabs,
            stochastic_trace: 0.0,
            deflation_error_estimate,
            quadrature_error_estimate: 0.0,
            stochastic_se: 0.0,
            ci_halfwidth: 0.0,
            total_error_estimate: deflation_error_estimate,
            confidence_level: config.confidence_level,
            stopping_rule: config.stopping_rule,
            samples: 0,
            probes: 0,
            matvecs: 0,
            accepted,
        });
    }

    // Symmetric positive-definite projected operator on range(P_rem):
    // A_rem x = P_rem A P_rem x, where P_rem = I - U_all U_all^T.
    let mut tmp_in = vec![0.0; dim];
    let mut a_pos = |x: &[f64], y: &mut [f64]| -> Result<()> {
        tmp_in.copy_from_slice(x);
        if !probe_deflate.is_empty() {
            project_out_basis(&mut tmp_in, &probe_deflate);
        }
        op(&tmp_in, y)?;
        if !probe_deflate.is_empty() {
            project_out_basis(y, &probe_deflate);
        }
        Ok(())
    };

    // Exact projected trace of the symmetric control variate:
    // tr(P_rem L_cv P_rem) = tr(L_cv) - sum_{u in probe_deflate} u^T L_cv u.
    let mut cv_projected_trace = cv_full_trace;
    for u in &probe_deflate {
        let u_val = cv_quad(u)?;
        if !u_val.is_finite() {
            return Err(slq_err(
                "control variate returned non-finite value on deflation vector",
            ));
        }
        cv_projected_trace -= u_val;
    }

    // Exact polynomial moment subtraction:
    // - linear only:    tr(A_rem - I_rem) = tr_a_rem - rem_dim
    // - linear + quad:  tr(A_rem - I_rem) - 0.5 * tr((A_rem - I_rem)^2)
    let rem_dim = dim - total_deflated;
    let rem_dim_f = rem_dim as f64;
    let (use_linear_sub, use_quad_sub, linear_offset) = match (exact_trace_rem, exact_trace_sq_rem)
    {
        (Some(tr_a), Some(tr_a2)) => {
            if !tr_a.is_finite() || !tr_a2.is_finite() {
                return Err(slq_err(
                    "exact_trace_rem and exact_trace_sq_rem must be finite",
                ));
            }
            let tr_e = tr_a - rem_dim_f;
            let tr_e2 = tr_a2 - 2.0 * tr_a + rem_dim_f;
            (true, true, tr_e - 0.5 * tr_e2)
        }
        (Some(tr_a), None) => {
            if !tr_a.is_finite() {
                return Err(slq_err("exact_trace_rem must be finite"));
            }
            (true, false, tr_a - rem_dim_f)
        }
        (None, Some(_)) => {
            return Err(slq_err("exact_trace_sq_rem requires exact_trace_rem"));
        }
        (None, None) => (false, false, 0.0),
    };

    // Optional Hutch++ randomized subspace Q inside range(P_rem).
    let sketch_r = hutchpp.sketch_dim.min(rem_dim);
    let power_iters = hutchpp.power_iters.max(1);
    let mut q_basis: Vec<Vec<f64>> = Vec::with_capacity(sketch_r);
    let mut matvecs = 0usize;
    let mut probes = 0usize;
    let mut det_q_trace = 0.0_f64;
    let mut det_quad_err = 0.0_f64;

    if sketch_r > 0 {
        let mut s_rng = SplitMix64::new(hutchpp.sketch_seed);
        let mut s_cur = vec![0.0; dim];
        let mut s_nxt = vec![0.0; dim];
        let mut s_cv = vec![0.0; dim];
        for _ in 0..sketch_r {
            for slot in &mut s_cur {
                *slot = s_rng.next_rademacher();
            }
            if !probe_deflate.is_empty() {
                project_out_basis(&mut s_cur, &probe_deflate);
            }
            for _ in 0..power_iters {
                s_nxt.fill(0.0);
                a_pos(&s_cur, &mut s_nxt)?;
                s_cv.fill(0.0);
                cv_matvec(&s_cur, &mut s_cv)?;
                for (dst, &cv_v) in s_nxt.iter_mut().zip(s_cv.iter()) {
                    *dst -= cv_v;
                }
                matvecs += 1;
                if !all_finite(&s_nxt) {
                    return Err(slq_err(
                        "Hutch++ sketch operator returned non-finite output",
                    ));
                }
                if !probe_deflate.is_empty() {
                    project_out_basis(&mut s_nxt, &probe_deflate);
                }
                let nrm = norm2_slice(&s_nxt);
                if !(nrm.is_finite() && nrm > 1e-14) {
                    s_cur.fill(0.0);
                    break;
                }
                for (dst, &val) in s_cur.iter_mut().zip(s_nxt.iter()) {
                    *dst = val / nrm;
                }
            }
            if !probe_deflate.is_empty() {
                project_out_basis(&mut s_cur, &probe_deflate);
            }
            if !q_basis.is_empty() {
                project_out_basis(&mut s_cur, &q_basis);
            }
            let nrm = norm2_slice(&s_cur);
            if nrm.is_finite() && nrm > 1e-12 {
                let q_vec: Vec<f64> = s_cur.iter().map(|&v| v / nrm).collect();
                q_basis.push(q_vec);
            }
        }

        if !q_basis.is_empty() {
            let per_q_quad_tol = (0.05 * config.target_tol) / (q_basis.len() as f64);
            for q in &q_basis {
                let est = lanczos_quadrature_log_inner(
                    q,
                    &mut a_pos,
                    radau_lower_bound,
                    config.max_lanczos_steps,
                    per_q_quad_tol,
                    config.quad_check_every,
                    rem_dim,
                )?;
                let cv_val = cv_quad(q)?;
                if !cv_val.is_finite() {
                    return Err(slq_err(
                        "control variate returned non-finite value on Hutch++ basis",
                    ));
                }
                let lin_val = if use_quad_sub {
                    let z2 = dot_slice(q, q);
                    let e1 = est.first_moment - z2;
                    let e2 = est.second_moment - 2.0 * est.first_moment + z2;
                    e1 - 0.5 * e2
                } else if use_linear_sub {
                    est.first_moment - dot_slice(q, q)
                } else {
                    0.0
                };
                det_q_trace += est.value - lin_val - cv_val;
                det_quad_err += est.error_estimate;
                probes += 1;
                matvecs += est.matvecs;
            }
        }
    }

    // If Q + probe_deflate spans the entire space, no stochastic trace remains.
    if total_deflated + q_basis.len() >= dim {
        let stochastic_trace = linear_offset + cv_projected_trace + det_q_trace;
        let logabs_det = negative_logabs + deflated_positive_logabs + stochastic_trace;
        let total_err = deflation_error_estimate + det_quad_err;
        return Ok(SlqLogDetReport {
            logabs_det,
            negative_logabs,
            deflated_positive_logabs,
            stochastic_trace,
            deflation_error_estimate,
            quadrature_error_estimate: det_quad_err,
            stochastic_se: 0.0,
            ci_halfwidth: 0.0,
            total_error_estimate: total_err,
            confidence_level: config.confidence_level,
            stopping_rule: config.stopping_rule,
            samples: 0,
            probes,
            matvecs,
            accepted: total_err <= config.target_tol,
        });
    }

    let mut stochastic_project: Vec<Vec<f64>> =
        Vec::with_capacity(probe_deflate.len() + q_basis.len());
    for u in &probe_deflate {
        stochastic_project.push(u.clone());
    }
    for q in &q_basis {
        stochastic_project.push(q.clone());
    }

    let per_color_quad_tol = (0.1 * config.target_tol) / (stride as f64);

    let mut rng = SplitMix64::new(config.seed);
    let mut mean = 0.0_f64;
    let mut m2 = 0.0_f64;
    let mut quad_err_sum = 0.0_f64;
    let mut samples = 0usize;
    let mut probe = vec![0.0; dim];
    let mut next_checkpoint_idx = 0usize;

    for sample_idx in 1..=max_samples {
        let mut sample_val = 0.0;
        let mut sample_quad_err = 0.0;
        for color in 0..stride {
            probe.fill(0.0);
            for idx in 0..dim {
                let blk = idx / block_size;
                if blk % stride == color {
                    probe[idx] = rng.next_rademacher();
                }
            }
            if !stochastic_project.is_empty() {
                project_out_basis(&mut probe, &stochastic_project);
            }
            let est = lanczos_quadrature_log_inner(
                &probe,
                &mut a_pos,
                radau_lower_bound,
                config.max_lanczos_steps,
                per_color_quad_tol,
                config.quad_check_every,
                rem_dim,
            )?;
            let cv_val = cv_quad(&probe)?;
            if !cv_val.is_finite() {
                return Err(slq_err(
                    "control variate returned non-finite value on probe",
                ));
            }
            let lin_val = if use_quad_sub {
                let z2 = dot_slice(&probe, &probe);
                let e1 = est.first_moment - z2;
                let e2 = est.second_moment - 2.0 * est.first_moment + z2;
                e1 - 0.5 * e2
            } else if use_linear_sub {
                est.first_moment - dot_slice(&probe, &probe)
            } else {
                0.0
            };
            sample_val += est.value - lin_val - cv_val;
            sample_quad_err += est.error_estimate;
            probes += 1;
            matvecs += est.matvecs;
        }
        samples = sample_idx;
        quad_err_sum += sample_quad_err;
        let delta = sample_val - mean;
        mean += delta / (samples as f64);
        let delta2 = sample_val - mean;
        m2 += delta * delta2;

        let is_checkpoint =
            next_checkpoint_idx < checkpoints.len() && samples == checkpoints[next_checkpoint_idx];
        if is_checkpoint {
            next_checkpoint_idx += 1;
            let variance = m2 / ((samples - 1) as f64);
            let se = (variance / (samples as f64)).sqrt();
            let t_mult = student_t_quantile(samples - 1, alpha_per_check)?;
            let ci_halfwidth = t_mult * se;
            let quad_err = det_quad_err + quad_err_sum / (samples as f64);
            let total_err = deflation_error_estimate + quad_err + ci_halfwidth;
            if total_err <= config.target_tol || samples == max_samples {
                let stochastic_trace = linear_offset + cv_projected_trace + det_q_trace + mean;
                let logabs_det = negative_logabs + deflated_positive_logabs + stochastic_trace;
                return Ok(SlqLogDetReport {
                    logabs_det,
                    negative_logabs,
                    deflated_positive_logabs,
                    stochastic_trace,
                    deflation_error_estimate,
                    quadrature_error_estimate: quad_err,
                    stochastic_se: se,
                    ci_halfwidth,
                    total_error_estimate: total_err,
                    confidence_level: config.confidence_level,
                    stopping_rule: config.stopping_rule,
                    samples,
                    probes,
                    matvecs,
                    accepted: total_err <= config.target_tol,
                });
            }
        }
    }

    let variance = if samples > 1 {
        m2 / ((samples - 1) as f64)
    } else {
        0.0
    };
    let se = if samples > 0 {
        (variance / (samples as f64)).sqrt()
    } else {
        f64::INFINITY
    };
    let t_mult = if samples > 1 {
        student_t_quantile(samples - 1, alpha_per_check)?
    } else {
        f64::INFINITY
    };
    let ci_halfwidth = t_mult * se;
    let quad_err = if samples > 0 {
        det_quad_err + quad_err_sum / (samples as f64)
    } else {
        det_quad_err
    };
    let total_err = deflation_error_estimate + quad_err + ci_halfwidth;
    let stochastic_trace = linear_offset + cv_projected_trace + det_q_trace + mean;
    let logabs_det = negative_logabs + deflated_positive_logabs + stochastic_trace;
    Ok(SlqLogDetReport {
        logabs_det,
        negative_logabs,
        deflated_positive_logabs,
        stochastic_trace,
        deflation_error_estimate,
        quadrature_error_estimate: quad_err,
        stochastic_se: se,
        ci_halfwidth,
        total_error_estimate: total_err,
        confidence_level: config.confidence_level,
        stopping_rule: config.stopping_rule,
        samples,
        probes,
        matvecs,
        accepted: total_err <= config.target_tol,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense_apply(a: &[f64], n: usize, x: &[f64], y: &mut [f64]) -> Result<()> {
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..n {
                s += a[i * n + j] * x[j];
            }
            y[i] = s;
        }
        Ok(())
    }

    #[test]
    fn tridiag_eigh_matches_1d_laplacian_analytic() {
        let m = 24usize;
        let diag = vec![2.0; m];
        let off = vec![-1.0; m - 1];
        let eig = tridiag_eigh(&diag, &off).expect("tridiag_eigh");
        for k in 1..=m {
            let exact = 2.0 - 2.0 * (std::f64::consts::PI * k as f64 / (m + 1) as f64).cos();
            let gap = (eig.values[k - 1] - exact).abs();
            assert!(
                gap < 1e-13,
                "k={k} got {} exact {exact} gap {gap}",
                eig.values[k - 1]
            );
        }
        // Verify orthonormality.
        for i in 0..m {
            for j in 0..m {
                let mut dot = 0.0;
                for r in 0..m {
                    dot += eig.vectors[r * m + i] * eig.vectors[r * m + j];
                }
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!((dot - expect).abs() < 1e-13);
            }
        }
    }

    #[test]
    fn sym_eigh_matches_reconstructed_matrix() {
        let m = 12usize;
        let mut a = vec![0.0; m * m];
        for i in 0..m {
            for j in 0..=i {
                let v = ((i * 7 + j * 13 + 1) as f64).sin() * 0.4;
                a[i * m + j] = v;
                a[j * m + i] = v;
            }
            a[i * m + i] += (i + 1) as f64;
        }
        let eig = sym_eigh(&a, m).expect("sym_eigh");
        for i in 0..m {
            for j in 0..m {
                let mut recon = 0.0;
                for k in 0..m {
                    recon += eig.vectors[i * m + k] * eig.values[k] * eig.vectors[j * m + k];
                }
                assert!((recon - a[i * m + j]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn gauss_and_gauss_radau_strictly_bracket_log_quadratic_form() {
        let n = 40usize;
        let mut diag = vec![0.0; n];
        for i in 0..n {
            diag[i] = 0.1 + (i as f64) * 0.25;
        }
        let v: Vec<f64> = (0..n).map(|i| ((i + 1) as f64 * 0.7).cos()).collect();
        let exact: f64 = diag.iter().zip(&v).map(|(&d, &x)| x * x * d.ln()).sum();

        let mut op = |x: &[f64], y: &mut [f64]| -> Result<()> {
            for i in 0..n {
                y[i] = diag[i] * x[i];
            }
            Ok(())
        };
        let coarse = lanczos_quadrature_log(&v, &mut op, Some(0.05), 10, 0.0, 2).expect("coarse");
        assert!(coarse.bracketed);
        assert!(
            coarse.lower <= exact + 1e-13,
            "coarse lower {} > exact {exact}",
            coarse.lower
        );
        assert!(
            coarse.upper >= exact - 1e-13,
            "coarse upper {} < exact {exact}",
            coarse.upper
        );
        assert!(coarse.error_estimate > 0.0);

        let est = lanczos_quadrature_log(&v, &mut op, Some(0.05), 38, 1e-10, 2).expect("quad");
        assert!(est.bracketed);
        assert!(
            est.lower <= exact + 1e-13,
            "lower {} > exact {exact}",
            est.lower
        );
        assert!(
            est.upper >= exact - 1e-13,
            "upper {} < exact {exact}",
            est.upper
        );
        assert!((est.value - exact).abs() <= est.error_estimate + 1e-13);
        assert!(
            est.error_estimate <= 1e-10,
            "error_estimate {} exceeded 1e-10 (steps={})",
            est.error_estimate,
            est.steps
        );

        // Adaptive Gauss mode (lower_bound = None) also converges accurately.
        let adapt = lanczos_quadrature_log(&v, &mut op, None, 38, 1e-10, 2).expect("adapt");
        assert!(!adapt.bracketed || adapt.invariant_breakdown);
        assert!(
            (adapt.value - exact).abs() < 1e-9,
            "adapt got {} exact {exact}",
            adapt.value
        );

        // Lower bound above lambda_min must return a typed refusal and never panic.
        assert!(lanczos_quadrature_log(&v, &mut op, Some(50.0), 10, 1e-8, 2).is_err());
    }

    #[test]
    fn pcg_solves_spd_with_deflation_and_refuses_indefinite() {
        let n = 20usize;
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            a[i * n + i] = 2.0 + 0.5 * (i as f64);
            if i + 1 < n {
                a[i * n + (i + 1)] = -0.3;
                a[(i + 1) * n + i] = -0.3;
            }
        }
        // Make mode 0 a deflated kernel direction.
        let mut q0 = vec![0.0; n];
        let inv = (n as f64).sqrt().recip();
        q0.fill(inv);
        let rhs: Vec<f64> = (0..n).map(|i| ((i + 1) as f64).sin()).collect();
        let mut op = |x: &[f64], y: &mut [f64]| dense_apply(&a, n, x, y);
        let mut precond = |r: &[f64], z: &mut [f64]| -> Result<()> {
            for i in 0..n {
                z[i] = r[i] / a[i * n + i];
            }
            Ok(())
        };
        let res = pcg_solve(
            &rhs,
            &mut op,
            &mut precond,
            std::slice::from_ref(&q0),
            PcgConfig {
                rel_tol: 1e-12,
                max_iter: 100,
            },
        )
        .expect("pcg");
        assert!(res.rel_residual <= 1e-12);
        assert!(dot_slice(&res.solution, &q0).abs() < 1e-12);

        // Indefinite operator must be refused.
        let mut indef = a.clone();
        indef[0] = -10.0;
        let mut op_indef = |x: &[f64], y: &mut [f64]| dense_apply(&indef, n, x, y);
        let mut id = |r: &[f64], z: &mut [f64]| -> Result<()> {
            z.copy_from_slice(r);
            Ok(())
        };
        assert!(
            pcg_solve(
                &rhs,
                &mut op_indef,
                &mut id,
                &[],
                PcgConfig {
                    rel_tol: 1e-10,
                    max_iter: 50,
                }
            )
            .is_err()
        );
    }

    #[test]
    fn deflated_slq_recovers_indefinite_logabs_det_within_error_estimate() {
        // Construct a 64x64 banded symmetric matrix with:
        // - 1 exact zero mode (constant vector projected out as kernel),
        // - 1 negative mode (-1e-4),
        // - remaining positive modes clustered near 1.0 (like preconditioned Hill).
        let n = 64usize;
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            let k = i as f64;
            a[i * n + i] = if i == 0 {
                0.0
            } else if i == 1 {
                -1e-4
            } else {
                1.0 + 0.8 / (k * k)
            };
        }
        // Add small banded off-diagonal couplings on modes >= 2.
        for i in 2..(n - 1) {
            let c = 0.08 / ((i as f64) * (i as f64));
            a[i * n + (i + 1)] = c;
            a[(i + 1) * n + i] = c;
        }
        // Dense reference log|det'| excluding mode 0.
        let eig = sym_eigh(&a, n).expect("dense ref");
        let mut exact_logdet = 0.0;
        let mut zero_dropped = false;
        for &v in &eig.values {
            if !zero_dropped && v.abs() < 1e-12 {
                zero_dropped = true;
                continue;
            }
            exact_logdet += v.abs().ln();
        }
        let mut kernel = vec![0.0; n];
        kernel[0] = 1.0;

        let mut op = |x: &[f64], y: &mut [f64]| dense_apply(&a, n, x, y);
        let extreme = lanczos_extreme_pairs(
            n,
            &mut op,
            std::slice::from_ref(&kernel),
            ExtremePairsConfig {
                neg_threshold: 0.0,
                neg_abs_tol: 1e-10,
                neg_log_tol: 1e-8,
                num_low_positive: 4,
                num_high_positive: 4,
                pos_residual_tol: 1e-8,
                subspace_dim: 28,
                max_restarts: 12,
                seed: 42,
            },
        )
        .expect("extreme");
        assert!(extreme.converged);
        assert_eq!(extreme.negative.len(), 1);
        assert!((extreme.negative[0].eigenvalue - (-1e-4)).abs() < 1e-11);

        let report = slq_logdet_deflated(
            n,
            &mut op,
            SlqDeflation {
                kernel: std::slice::from_ref(&kernel),
                negative_pairs: &extreme.negative,
                positive_deflation: &extreme.positive_deflation,
                deflation_lower_estimate: extreme.positive_lower_estimate,
                radau_lower_bound: None,
            },
            SlqConfig {
                strategy: ProbeStrategy::StridedColors {
                    stride: 4,
                    block_size: 1,
                },
                stopping_rule: SlqStoppingRule::AlphaSpending,
                min_samples: 8,
                max_samples: 32,
                max_lanczos_steps: 24,
                quad_check_every: 2,
                target_tol: 5e-4,
                confidence_level: 0.995,
                seed: 1337,
            },
        )
        .expect("slq");
        assert!(report.accepted, "report: {report:?}");
        let gap = (report.logabs_det - exact_logdet).abs();
        assert!(
            gap <= report.total_error_estimate,
            "got {} exact {exact_logdet} gap {gap:.3e} est {:.3e}",
            report.logabs_det,
            report.total_error_estimate
        );
    }

    #[test]
    fn lanczos_quadrature_fn_matches_exact_sinh_like_function() {
        let n = 24usize;
        let mut diag = vec![0.0; n];
        for i in 0..n {
            diag[i] = -0.2 + 0.15 * (i as f64);
        }
        let v: Vec<f64> = (0..n).map(|i| ((i + 1) as f64 * 0.43).sin()).collect();
        let f = |x: f64| -> Result<f64> { Ok((1.0 + 0.5 * x * x).ln()) };
        let exact: f64 = diag
            .iter()
            .zip(&v)
            .map(|(&d, &x)| x * x * (1.0 + 0.5 * d * d).ln())
            .sum();
        let mut op = |x: &[f64], y: &mut [f64]| -> Result<()> {
            for i in 0..n {
                y[i] = diag[i] * x[i];
            }
            Ok(())
        };
        let est = lanczos_quadrature_fn(&v, &mut op, f, 20, 1e-11).expect("fn quad");
        assert!((est.value - exact).abs() < 1e-10);
    }

    #[test]
    fn slq_with_control_variate_collapses_variance_on_block_diagonal_perturbation() {
        // 32x32 SPD matrix = block-diagonal 2x2 blocks + tiny off-block coupling.
        // Control variate = exact ln(block-diagonal) cancels intra-block variance.
        let n = 32usize;
        let mut base = vec![0.0; n * n];
        let mut a = vec![0.0; n * n];
        for blk in 0..(n / 2) {
            let i = 2 * blk;
            let k = (blk + 1) as f64;
            let d0 = 1.0 + 0.6 / (k * k);
            let d1 = 1.0 + 0.3 / (k * k);
            let off = 0.25 / (k * k);
            base[i * n + i] = d0;
            base[(i + 1) * n + (i + 1)] = d1;
            base[i * n + (i + 1)] = off;
            base[(i + 1) * n + i] = off;
        }
        a.copy_from_slice(&base);
        for i in 0..(n - 2) {
            let eps = 0.01 / (((i / 2) + 1) as f64).powi(2);
            a[i * n + (i + 2)] += eps;
            a[(i + 2) * n + i] += eps;
        }

        let eig_a = sym_eigh(&a, n).expect("eigh a");
        let exact: f64 = eig_a.values.iter().map(|&v| v.ln()).sum();

        // Build exact ln(base) via dense eigh for the control variate.
        let eig_b = sym_eigh(&base, n).expect("eigh base");
        let cv_trace: f64 = eig_b.values.iter().map(|&v| v.ln()).sum();
        let mut l_cv = vec![0.0; n * n];
        for r in 0..n {
            for c in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += eig_b.vectors[r * n + k] * eig_b.values[k].ln() * eig_b.vectors[c * n + k];
                }
                l_cv[r * n + c] = s;
            }
        }
        let mut op = |x: &[f64], y: &mut [f64]| dense_apply(&a, n, x, y);
        let mut cv_quad = |z: &[f64]| -> Result<f64> {
            let mut tmp = vec![0.0; n];
            dense_apply(&l_cv, n, z, &mut tmp)?;
            Ok(dot_slice(z, &tmp))
        };
        let cfg = SlqConfig {
            strategy: ProbeStrategy::Rademacher,
            stopping_rule: SlqStoppingRule::FixedSamples,
            min_samples: 8,
            max_samples: 8,
            max_lanczos_steps: 20,
            quad_check_every: 2,
            target_tol: 1e-3,
            confidence_level: 0.995,
            seed: 99,
        };
        let defl = SlqDeflation {
            kernel: &[],
            negative_pairs: &[],
            positive_deflation: &[],
            deflation_lower_estimate: 0.5,
            radau_lower_bound: Some(0.5),
        };
        let plain = slq_logdet_deflated(n, &mut op, defl, cfg).expect("plain");
        let with_cv = slq_logdet_deflated_with_cv(n, &mut op, defl, cv_trace, &mut cv_quad, cfg)
            .expect("with_cv");
        assert!(
            with_cv.stochastic_se < 0.1 * plain.stochastic_se,
            "expected >10x SE reduction: cv={:.3e} plain={:.3e}",
            with_cv.stochastic_se,
            plain.stochastic_se
        );
        assert!(
            (with_cv.logabs_det - exact).abs() <= with_cv.total_error_estimate,
            "got {} exact {exact} est {:.3e}",
            with_cv.logabs_det,
            with_cv.total_error_estimate
        );
    }

    #[test]
    fn slq_hutchpp_collapses_low_rank_perturbation_variance() {
        // 64x64 SPD matrix = I + low-rank perturbation of rank 6 + tiny 1e-4 tail.
        let n = 64usize;
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            a[i * n + i] = 1.0 + 1e-4 / ((i + 1) as f64).powi(2);
        }
        for r in 0..6 {
            let weight = 0.8 / ((r + 1) as f64);
            let mut v = vec![0.0; n];
            for i in 0..n {
                v[i] = (((i + 1) * (r + 1)) as f64 * 0.29).sin();
            }
            let nrm = norm2_slice(&v);
            for slot in &mut v {
                *slot /= nrm;
            }
            for i in 0..n {
                for j in 0..n {
                    a[i * n + j] += weight * v[i] * v[j];
                }
            }
        }
        let eig = sym_eigh(&a, n).expect("eigh");
        let exact: f64 = eig.values.iter().map(|&v| v.ln()).sum();
        let mut op = |x: &[f64], y: &mut [f64]| dense_apply(&a, n, x, y);
        let mut zero_cv = |_z: &[f64]| Ok(0.0);
        let mut id_cv = |x: &[f64], y: &mut [f64]| {
            y.copy_from_slice(x);
            Ok(())
        };
        let cfg = SlqConfig {
            strategy: ProbeStrategy::Rademacher,
            stopping_rule: SlqStoppingRule::AlphaSpending,
            min_samples: 8,
            max_samples: 16,
            max_lanczos_steps: 24,
            quad_check_every: 2,
            target_tol: 1e-5,
            confidence_level: 0.995,
            seed: 2026,
        };
        let tr_a: f64 = (0..n).map(|i| a[i * n + i]).sum();
        let tr_a2: f64 = a.iter().map(|&v| v * v).sum();
        let rep = slq_logdet_hutchpp(
            n,
            &mut op,
            SlqDeflation {
                kernel: &[],
                negative_pairs: &[],
                positive_deflation: &[],
                deflation_lower_estimate: 0.5,
                radau_lower_bound: None,
            },
            &mut zero_cv,
            &mut id_cv,
            SlqHutchPpOptions {
                moments: SlqMoments {
                    trace: Some(tr_a),
                    trace_sq: Some(tr_a2),
                },
                cv_full_trace: 0.0,
                hutchpp: HutchPpConfig {
                    sketch_dim: 10,
                    power_iters: 2,
                    sketch_seed: 7777,
                },
            },
            cfg,
        )
        .expect("hutchpp");
        assert!(rep.accepted, "rep: {rep:?}");
        assert!(
            (rep.logabs_det - exact).abs() <= rep.total_error_estimate,
            "got {} exact {exact} est {:.3e}",
            rep.logabs_det,
            rep.total_error_estimate
        );
    }

    #[test]
    fn lanczos_quadrature_moments_are_exact() {
        let n = 12usize;
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            a[i * n + i] = 1.0 + 0.25 * (i as f64);
            if i + 1 < n {
                let off = 0.15 * ((i + 1) as f64).sin();
                a[i * n + (i + 1)] = off;
                a[(i + 1) * n + i] = off;
            }
        }
        let v: Vec<f64> = (0..n).map(|i| ((i as f64 + 1.0) * 0.7).cos()).collect();
        let mut av = vec![0.0; n];
        dense_apply(&a, n, &v, &mut av).expect("av");
        let exact_m1 = dot_slice(&v, &av);
        let exact_m2 = dot_slice(&av, &av);

        let mut op = |x: &[f64], y: &mut [f64]| dense_apply(&a, n, x, y);
        let est = lanczos_quadrature_log(&v, &mut op, Some(0.2), 12, 1e-10, 2).expect("log quad");
        assert!(
            (est.first_moment - exact_m1).abs() < 1e-12,
            "first_moment: got {} expected {exact_m1}",
            est.first_moment
        );
        assert!(
            (est.second_moment - exact_m2).abs() < 1e-12,
            "second_moment: got {} expected {exact_m2}",
            est.second_moment
        );
    }

    #[test]
    fn slq_refuses_non_positive_lower_bound() {
        let n = 8usize;
        let mut op = |x: &[f64], y: &mut [f64]| -> Result<()> {
            y.copy_from_slice(x);
            Ok(())
        };
        assert!(
            slq_logdet_deflated(
                n,
                &mut op,
                SlqDeflation {
                    kernel: &[],
                    negative_pairs: &[],
                    positive_deflation: &[],
                    deflation_lower_estimate: 0.0,
                    radau_lower_bound: None,
                },
                SlqConfig {
                    strategy: ProbeStrategy::Rademacher,
                    stopping_rule: SlqStoppingRule::FixedSamples,
                    min_samples: 8,
                    max_samples: 8,
                    max_lanczos_steps: 8,
                    quad_check_every: 2,
                    target_tol: 1e-3,
                    confidence_level: 0.995,
                    seed: 1,
                },
            )
            .is_err()
        );
    }

    #[test]
    fn student_t_quantile_matches_standard_tables() {
        // Exact reference values (two-sided tail alpha = 1 - level):
        let cases: &[(usize, f64, f64)] = &[
            // 95% two-sided (alpha = 0.05)
            (1, 0.05, 12.706_204_736_432_095),
            (2, 0.05, 4.302_652_729_911_275),
            (3, 0.05, 3.182_446_305_284_263),
            (4, 0.05, 2.776_445_105_197_798_7),
            (10, 0.05, 2.228_138_851_964_938_5),
            // 99.5% two-sided (alpha = 0.005)
            (1, 0.005, 127.321_336_468_874_83),
            (2, 0.005, 14.089_047_275_548_754),
            (3, 0.005, 7.453_318_505_150_54),
            (4, 0.005, 5.597_568_368_322_182),
            (7, 0.005, 4.029_337_177_642_779),
            (10, 0.005, 3.581_406_202_090_67),
            (20, 0.005, 3.153_400_532_846_034_4),
            (30, 0.005, 3.029_798_223_646_942_3),
            // 99.73% two-sided (alpha = 0.0027)
            (4, 0.0027, 6.620_071_562_817_905),
            (10, 0.0027, 3.956_889_989_511_066_3),
            (20, 0.0027, 3.422_118_566_423_340_4),
            (1000, 0.0027, 3.007_494_111_288_657),
        ];
        for &(df, alpha, expected) in cases {
            let got = student_t_quantile(df, alpha).expect("quantile");
            let rel = (got - expected).abs() / expected;
            assert!(
                rel < 1e-8,
                "df={df} alpha={alpha}: got {got:.12} expected {expected:.12} rel={rel:.3e}"
            );
            let tail_err = (student_t_two_sided_tail(df as f64, got) - alpha).abs();
            assert!(
                tail_err <= 1e-11 * alpha,
                "df={df} alpha={alpha}: tail residual {tail_err:.3e}"
            );
        }
        assert!(student_t_quantile(0, 0.05).is_err());
        assert!(student_t_quantile(5, 0.0).is_err());
        assert!(student_t_quantile(5, 1.0).is_err());
        assert!(student_t_quantile(5, f64::NAN).is_err());
    }

    #[test]
    fn extreme_pairs_converged_honors_restart_limit() {
        // 80x80 dense SPD matrix with clustered low eigenvalues; with subspace_dim = 8
        // and max_restarts = 0, target positive eigenpairs cannot converge to 1e-12.
        let n = 80usize;
        let mut diag = vec![0.0; n];
        for i in 0..n {
            diag[i] = 0.1 + 0.01 * (i as f64);
        }
        let mut op = |x: &[f64], y: &mut [f64]| -> Result<()> {
            for i in 0..n {
                y[i] = diag[i] * x[i];
            }
            Ok(())
        };
        let rep = lanczos_extreme_pairs(
            n,
            &mut op,
            &[],
            ExtremePairsConfig {
                neg_threshold: 0.0,
                neg_abs_tol: 1e-10,
                neg_log_tol: 1e-8,
                num_low_positive: 4,
                num_high_positive: 2,
                pos_residual_tol: 1e-12,
                subspace_dim: 8,
                max_restarts: 0,
                seed: 7,
            },
        )
        .expect("extreme");
        assert!(
            !rep.converged,
            "expected converged=false when max_restarts=0 cannot meet pos_residual_tol=1e-12"
        );
    }

    #[test]
    fn slq_stopping_rules_honor_fixed_and_alpha_spending_checkpoints() {
        let n = 32usize;
        let diag: Vec<f64> = (0..n)
            .map(|i| 1.0 + 0.2 * ((i as f64 * 0.3).sin()))
            .collect();
        let mut op = |x: &[f64], y: &mut [f64]| -> Result<()> {
            for i in 0..n {
                y[i] = diag[i] * x[i];
            }
            Ok(())
        };
        let defl = SlqDeflation {
            kernel: &[],
            negative_pairs: &[],
            positive_deflation: &[],
            deflation_lower_estimate: 0.5,
            radau_lower_bound: None,
        };
        let fixed = slq_logdet_deflated(
            n,
            &mut op,
            defl,
            SlqConfig {
                strategy: ProbeStrategy::Rademacher,
                stopping_rule: SlqStoppingRule::FixedSamples,
                min_samples: 4,
                max_samples: 12,
                max_lanczos_steps: 16,
                quad_check_every: 2,
                target_tol: 1e2, // huge tol, but FixedSamples must still draw all 12
                confidence_level: 0.995,
                seed: 11,
            },
        )
        .expect("fixed");
        assert_eq!(fixed.samples, 12);
        assert_eq!(fixed.stopping_rule, SlqStoppingRule::FixedSamples);

        let spend = slq_logdet_deflated(
            n,
            &mut op,
            defl,
            SlqConfig {
                strategy: ProbeStrategy::Rademacher,
                stopping_rule: SlqStoppingRule::AlphaSpending,
                min_samples: 8,
                max_samples: 32,
                max_lanczos_steps: 16,
                quad_check_every: 2,
                target_tol: 1e2, // stops at first geometric checkpoint (8)
                confidence_level: 0.995,
                seed: 11,
            },
        )
        .expect("spend");
        assert_eq!(spend.samples, 8);
        assert_eq!(spend.stopping_rule, SlqStoppingRule::AlphaSpending);
    }
}
