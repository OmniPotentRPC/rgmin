//! Bordered cyclic block determinants without the period product.
//!
//! A periodic linear map `x_{k+1} = T_k x_k`, `k = 0..b`, has monodromy
//! `M = T_{b-1} ... T_0`. Multiple shooting writes `det(I - M)` as the
//! determinant of the cyclic block matrix
//!
//! ```text
//! row block i:   -T_i in block column i,   +I in block column (i + 1) mod b
//! ```
//!
//! optionally bordered by `p` columns that live in the rows of block
//! `b - 1` and `p` rows that live in the columns of block `0`.
//!
//! Gaussian elimination with partial pivoting on that matrix has no
//! growth bound, and multiplying the factors overflows or drops the
//! decaying multipliers. [`crate::periodic::bordered_cyclic_log_abs`] instead eliminates
//! one block column at a time, in the natural order `0, 1, ..., b - 1`,
//! with Householder reflectors on the block rows it couples. Each new
//! block row is stacked above the running one and the fill stays in
//! block column `b - 1` and in the border, so the reflectors are those of
//! a dense Householder QR in natural row and column order that skips
//! structural zeros.
//! No product of two factors is formed, every transformation is
//! orthogonal, and the arithmetic is `O(b m^3)` instead of `O((b m)^3)`.
//!
//! Rows and columns are first scaled by powers of two to unit largest
//! entry. Householder QR is backward stable column by column only, and
//! shooting blocks mix positions and momenta whose rows differ by many
//! decades. Without the scaling, or in another row or column order,
//! strongly hyperbolic periods lose digits that partial-pivoting LU
//! keeps. The scaling is exact and is undone in `log|det|`, in
//! [`crate::periodic::CyclicFactor::solve`] and in
//! [`crate::periodic::CyclicFactor::solve_transpose`].
//!
//! Orthogonal transformations keep the 2-norm of every column of the
//! stacked, equilibrated matrix `B = D_r A D_c`, so no intermediate entry
//! grows. The QR backward error is therefore relative to `B`: the computed
//! factorization is exact for `B + ΔB`, with each column of `ΔB` bounded by
//! a few rounding units times that column's 2-norm. In unscaled coordinates
//! the perturbation is `ΔA = D_r^{-1} ΔB D_c^{-1}`; it is not normalized by
//! `||A||`.
//!
//! The public norm estimates instead describe the original, unscaled
//! bordered matrix `A`. [`crate::periodic::CyclicFactor::norm1`] is
//! `||A||_1`, and [`crate::periodic::CyclicFactor::inverse_norm1`] estimates
//! `||A^{-1}||_1`. Their product estimates the normwise relative condition
//! number `||A||_1 ||A^{-1}||_1`; neither field is a backward-error estimate.
//! The inverse-norm estimate is Hager's method with Higham's safeguard
//! vector, evaluated through the stored factors.

use thiserror::Error;

/// Refusal of a bordered cyclic determinant.
#[derive(Debug, Error, Clone, PartialEq)]
#[non_exhaustive]
pub enum CyclicError {
    /// A block or border has the wrong length, or there are no blocks.
    #[error("cyclic shape: {0}")]
    Shape(&'static str),
    /// An input entry is NaN or infinite.
    #[error("a cyclic block entry is not finite")]
    NonFinite,
    /// A matrix or inverse-matrix 1-norm estimate overflowed or became NaN.
    #[error("a cyclic norm estimate is not finite")]
    NonFiniteNormEstimate,
    /// A triangular factor has an exactly zero diagonal entry.
    #[error("the bordered cyclic matrix is singular")]
    Singular,
}

/// Householder reflectors of one panel, LAPACK `dgeqr2` convention.
#[derive(Clone, Debug)]
struct Panel {
    rows: usize,
    cols: usize,
    /// Row-major `rows x cols`; column `k` holds `v_k[k+1..]`, `v_k[k] = 1`.
    v: Vec<f64>,
    tau: Vec<f64>,
}

impl Panel {
    /// `Q^T y` for a vector of length `rows`.
    fn apply_transpose(&self, y: &mut [f64]) {
        for k in 0..self.cols {
            self.reflect(k, y);
        }
    }

    /// `Q y` for a vector of length `rows`.
    fn apply(&self, y: &mut [f64]) {
        for k in (0..self.cols).rev() {
            self.reflect(k, y);
        }
    }

    fn reflect(&self, k: usize, y: &mut [f64]) {
        let tau = self.tau[k];
        if tau == 0.0 {
            return;
        }
        let mut dot = y[k];
        for row in (k + 1)..self.rows {
            dot += self.v[row * self.cols + k] * y[row];
        }
        let scale = tau * dot;
        y[k] -= scale;
        for row in (k + 1)..self.rows {
            y[row] -= scale * self.v[row * self.cols + k];
        }
    }
}

/// 2-norm of a strided column without overflow or harmful underflow.
fn column_norm(work: &[f64], width: usize, column: usize, first: usize, rows: usize) -> f64 {
    let mut scale = 0.0_f64;
    for row in first..rows {
        scale = scale.max(work[row * width + column].abs());
    }
    if scale == 0.0 {
        return 0.0;
    }
    let mut sum = 0.0;
    for row in first..rows {
        let ratio = work[row * width + column] / scale;
        sum += ratio * ratio;
    }
    scale * sum.sqrt()
}

/// In-place Householder QR of the first `cols` columns of a row-major
/// `rows x width` array, applied to every column of the array.
///
/// On return the upper triangle of the leading block is `R` and the
/// trailing columns hold `Q^T` times their input.
fn householder(work: &mut [f64], rows: usize, width: usize, cols: usize) -> Panel {
    let mut v = vec![0.0; rows * cols];
    let mut tau = vec![0.0; cols];
    for k in 0..cols {
        let alpha = work[k * width + k];
        let tail = column_norm(work, width, k, k + 1, rows);
        if tail == 0.0 {
            continue;
        }
        let beta = -alpha.signum() * alpha.hypot(tail);
        tau[k] = (beta - alpha) / beta;
        let pivot = alpha - beta;
        for row in (k + 1)..rows {
            v[row * cols + k] = work[row * width + k] / pivot;
            work[row * width + k] = 0.0;
        }
        work[k * width + k] = beta;
        for column in (k + 1)..width {
            let mut dot = work[k * width + column];
            for row in (k + 1)..rows {
                dot += v[row * cols + k] * work[row * width + column];
            }
            let scale = tau[k] * dot;
            work[k * width + column] -= scale;
            for row in (k + 1)..rows {
                work[row * width + column] -= scale * v[row * cols + k];
            }
        }
    }
    Panel { rows, cols, v, tau }
}

/// One eliminated block column `j <= b - 2`.
#[derive(Clone, Debug)]
struct Step {
    panel: Panel,
    /// Upper triangle in block column `j`.
    r: Vec<f64>,
    /// Fill in block column `j + 1`; empty when that is block `b - 1`.
    next: Vec<f64>,
    /// Fill in block column `b - 1`.
    last: Vec<f64>,
    /// Fill in the border columns.
    border: Vec<f64>,
}

/// Orthogonal factorization of a bordered cyclic block matrix.
#[derive(Clone, Debug)]
pub struct CyclicFactor {
    m: usize,
    p: usize,
    blocks: usize,
    steps: Vec<Step>,
    last: Panel,
    /// Powers of two `D_r`, `D_c` with `D_r A D_c` the factored matrix.
    row_scale: Vec<f64>,
    column_scale: Vec<f64>,
    /// Upper triangle of the final `(m + p)` square system.
    last_r: Vec<f64>,
    /// `ln|det A|`.
    pub log_abs: f64,
    /// `||A||_1` for the original, unscaled bordered matrix `A`.
    ///
    /// This is the largest unscaled column 1-norm, not a backward error.
    pub norm1: f64,
    /// Hager–Higham estimate of `||A^{-1}||_1` for the original, unscaled
    /// bordered matrix `A`.
    ///
    /// In exact arithmetic this is a lower bound that is usually within a
    /// factor of three of the value. `norm1 * inverse_norm1` estimates the
    /// normwise relative condition number; neither factor is a backward
    /// error.
    pub inverse_norm1: f64,
}

impl CyclicFactor {
    /// Order of the bordered matrix, `b m + p`.
    pub fn order(&self) -> usize {
        self.blocks * self.m + self.p
    }

    /// Solve `A x = y` through the stored factors.
    pub fn solve(&self, y: &[f64]) -> Result<Vec<f64>, CyclicError> {
        if y.len() != self.order() {
            return Err(CyclicError::Shape("right-hand side length"));
        }
        let scaled: Vec<f64> = y.iter().zip(&self.row_scale).map(|(a, s)| a * s).collect();
        let mut x = self.scaled_solve(&scaled)?;
        x.iter_mut()
            .zip(&self.column_scale)
            .for_each(|(a, s)| *a *= s);
        Ok(x)
    }

    /// Solve `A^T z = w` through the stored factors.
    pub fn solve_transpose(&self, w: &[f64]) -> Result<Vec<f64>, CyclicError> {
        if w.len() != self.order() {
            return Err(CyclicError::Shape("right-hand side length"));
        }
        let scaled: Vec<f64> = w
            .iter()
            .zip(&self.column_scale)
            .map(|(a, s)| a * s)
            .collect();
        let mut z = self.scaled_solve_transpose(&scaled)?;
        z.iter_mut().zip(&self.row_scale).for_each(|(a, s)| *a *= s);
        Ok(z)
    }

    fn scaled_solve(&self, y: &[f64]) -> Result<Vec<f64>, CyclicError> {
        let (m, p, b) = (self.m, self.p, self.blocks);
        let border = &y[b * m..];
        let mut pivots: Vec<Vec<f64>> = Vec::with_capacity(self.steps.len());
        let mut carry: Vec<f64> = Vec::new();
        for (j, step) in self.steps.iter().enumerate() {
            let mut stacked = if j == 0 {
                let mut rows = y[..m].to_vec();
                rows.extend_from_slice(&y[(b - 1) * m..b * m]);
                rows.extend_from_slice(border);
                rows
            } else {
                let mut rows = y[j * m..(j + 1) * m].to_vec();
                rows.extend_from_slice(&carry);
                rows
            };
            step.panel.apply_transpose(&mut stacked);
            pivots.push(stacked[..m].to_vec());
            carry = stacked[m..].to_vec();
        }
        let mut tail = if b == 1 {
            let mut rows = y[..m].to_vec();
            rows.extend_from_slice(border);
            rows
        } else {
            carry
        };
        self.last.apply_transpose(&mut tail);
        upper_solve(&self.last_r, m + p, &mut tail)?;
        let mut x = vec![0.0; self.order()];
        x[(b - 1) * m..].copy_from_slice(&tail);
        for (j, step) in self.steps.iter().enumerate().rev() {
            let mut rhs = pivots[j].clone();
            if !step.next.is_empty() {
                let next = x[(j + 1) * m..(j + 2) * m].to_vec();
                subtract_product(&mut rhs, &step.next, m, m, &next);
            }
            subtract_product(&mut rhs, &step.last, m, m, &x[(b - 1) * m..b * m]);
            subtract_product(&mut rhs, &step.border, m, p, &x[b * m..]);
            upper_solve(&step.r, m, &mut rhs)?;
            x[j * m..(j + 1) * m].copy_from_slice(&rhs);
        }
        Ok(x)
    }

    fn scaled_solve_transpose(&self, w: &[f64]) -> Result<Vec<f64>, CyclicError> {
        let (m, p, b) = (self.m, self.p, self.blocks);
        let mut pivots: Vec<Vec<f64>> = Vec::with_capacity(self.steps.len());
        let mut tail = w[(b - 1) * m..].to_vec();
        for (j, step) in self.steps.iter().enumerate() {
            let mut rhs = w[j * m..(j + 1) * m].to_vec();
            if j > 0 {
                transpose_subtract(&mut rhs, &self.steps[j - 1].next, m, m, &pivots[j - 1]);
            }
            lower_transpose_solve(&step.r, m, &mut rhs)?;
            let (head, border) = tail.split_at_mut(m);
            transpose_subtract(head, &step.last, m, m, &rhs);
            transpose_subtract(border, &step.border, m, p, &rhs);
            pivots.push(rhs);
        }
        lower_transpose_solve(&self.last_r, m + p, &mut tail)?;
        self.last.apply(&mut tail);
        let mut z = vec![0.0; self.order()];
        if b == 1 {
            z.copy_from_slice(&tail);
            return Ok(z);
        }
        let mut carry = tail;
        for (j, step) in self.steps.iter().enumerate().rev() {
            let mut stacked = pivots[j].clone();
            stacked.extend_from_slice(&carry);
            step.panel.apply(&mut stacked);
            if j == 0 {
                z[..m].copy_from_slice(&stacked[..m]);
                z[(b - 1) * m..b * m].copy_from_slice(&stacked[m..2 * m]);
                z[b * m..].copy_from_slice(&stacked[2 * m..]);
            } else {
                z[j * m..(j + 1) * m].copy_from_slice(&stacked[..m]);
                carry = stacked[m..].to_vec();
            }
        }
        Ok(z)
    }

    fn estimate_inverse_norm1(&self) -> Result<f64, CyclicError> {
        let n = self.order();
        let mut x = vec![1.0 / n as f64; n];
        let mut estimate = 0.0_f64;
        let mut previous: Option<usize> = None;
        for _ in 0..5 {
            let y = self.solve(&x)?;
            let norm = finite_abs_sum(&y)?;
            if norm <= estimate {
                break;
            }
            estimate = norm;
            let signs: Vec<f64> = y
                .iter()
                .map(|value| if *value >= 0.0 { 1.0 } else { -1.0 })
                .collect();
            let z = self.solve_transpose(&signs)?;
            if z.iter().any(|value| !value.is_finite()) {
                return Err(CyclicError::NonFiniteNormEstimate);
            }
            let mut best = 0usize;
            for (index, value) in z.iter().enumerate() {
                if value.abs() > z[best].abs() {
                    best = index;
                }
            }
            let projected: f64 = z.iter().zip(&x).map(|(a, b)| a * b).sum();
            if !projected.is_finite() {
                return Err(CyclicError::NonFiniteNormEstimate);
            }
            if z[best].abs() <= projected || previous == Some(best) {
                break;
            }
            previous = Some(best);
            x = vec![0.0; n];
            x[best] = 1.0;
        }
        let denominator = if n > 1 { (n - 1) as f64 } else { 1.0 };
        let alternating: Vec<f64> = (0..n)
            .map(|index| {
                let sign = if index % 2 == 0 { 1.0 } else { -1.0 };
                sign * (1.0 + index as f64 / denominator)
            })
            .collect();
        let y = self.solve(&alternating)?;
        let norm = finite_abs_sum(&y)?;
        let safeguard = 2.0 * norm / (3.0 * n as f64);
        if !safeguard.is_finite() {
            return Err(CyclicError::NonFiniteNormEstimate);
        }
        estimate = estimate.max(safeguard);
        Ok(estimate)
    }
}

fn finite_abs_sum(values: &[f64]) -> Result<f64, CyclicError> {
    let mut sum = 0.0;
    for value in values {
        sum += value.abs();
        if !sum.is_finite() {
            return Err(CyclicError::NonFiniteNormEstimate);
        }
    }
    Ok(sum)
}

fn subtract_product(rhs: &mut [f64], matrix: &[f64], rows: usize, cols: usize, x: &[f64]) {
    for row in 0..rows {
        let mut sum = 0.0;
        for column in 0..cols {
            sum += matrix[row * cols + column] * x[column];
        }
        rhs[row] -= sum;
    }
}

fn transpose_subtract(rhs: &mut [f64], matrix: &[f64], rows: usize, cols: usize, v: &[f64]) {
    for column in 0..cols {
        let mut sum = 0.0;
        for row in 0..rows {
            sum += matrix[row * cols + column] * v[row];
        }
        rhs[column] -= sum;
    }
}

fn upper_solve(r: &[f64], n: usize, rhs: &mut [f64]) -> Result<(), CyclicError> {
    for row in (0..n).rev() {
        let mut sum = rhs[row];
        for column in (row + 1)..n {
            sum -= r[row * n + column] * rhs[column];
        }
        let diagonal = r[row * n + row];
        if diagonal == 0.0 {
            return Err(CyclicError::Singular);
        }
        rhs[row] = sum / diagonal;
    }
    Ok(())
}

fn lower_transpose_solve(r: &[f64], n: usize, rhs: &mut [f64]) -> Result<(), CyclicError> {
    for row in 0..n {
        let mut sum = rhs[row];
        for inner in 0..row {
            sum -= r[inner * n + row] * rhs[inner];
        }
        let diagonal = r[row * n + row];
        if diagonal == 0.0 {
            return Err(CyclicError::Singular);
        }
        rhs[row] = sum / diagonal;
    }
    Ok(())
}

fn diagonal_log(r: &[f64], n: usize) -> Result<f64, CyclicError> {
    let mut total = 0.0;
    for index in 0..n {
        let value = r[index * n + index].abs();
        if value == 0.0 {
            return Err(CyclicError::Singular);
        }
        total += value.ln();
    }
    Ok(total)
}

fn upper_block(work: &[f64], width: usize, n: usize, column0: usize) -> Vec<f64> {
    let mut block = vec![0.0; n * n];
    for row in 0..n {
        for column in row..n {
            block[row * n + column] = work[row * width + column0 + column];
        }
    }
    block
}

fn sub_block(
    work: &[f64],
    width: usize,
    row0: usize,
    rows: usize,
    column0: usize,
    cols: usize,
) -> Vec<f64> {
    let mut block = vec![0.0; rows * cols];
    for row in 0..rows {
        for column in 0..cols {
            block[row * cols + column] = work[(row0 + row) * width + column0 + column];
        }
    }
    block
}

/// The bordered cyclic matrix after exact power-of-two scaling.
struct Equilibrated {
    transfers: Vec<Vec<f64>>,
    /// Diagonal of row block `i` in block column `(i + 1) mod b`.
    coupling: Vec<Vec<f64>>,
    column_border: Vec<Vec<f64>>,
    row_border: Vec<Vec<f64>>,
    row_scale: Vec<f64>,
    column_scale: Vec<f64>,
    /// `log2 det(D_r D_c)`.
    exponent: i64,
}

/// Power of two nearest `1 / magnitude`, or one for an empty row or column.
fn inverse_power_of_two(magnitude: f64) -> (f64, i64) {
    if magnitude == 0.0 {
        return (1.0, 0);
    }
    let exponent = (-magnitude.log2().round()).clamp(-1000.0, 1000.0) as i64;
    (2.0_f64.powi(exponent as i32), exponent)
}

/// Scale every row, then every column, to unit largest entry.
///
/// Householder QR is backward stable column by column, so a matrix
/// whose rows differ by orders of magnitude loses digits that row
/// pivoting would keep. The scales are powers of two: the scaled
/// entries and the determinant correction are exact, and the block
/// pattern is unchanged.
fn equilibrate(
    transfers: &[Vec<f64>],
    m: usize,
    column_border: &[Vec<f64>],
    row_border: &[Vec<f64>],
) -> Equilibrated {
    let (b, p) = (transfers.len(), column_border.len());
    let entry = |block: usize, row: usize, column: usize| -> f64 {
        let value = -transfers[block][row * m + column];
        if b == 1 && row == column {
            value + 1.0
        } else {
            value
        }
    };
    let coupling_entry: f64 = if b == 1 { 0.0 } else { 1.0 };
    let mut row_scale = vec![1.0; b * m + p];
    let mut column_scale = vec![1.0; b * m + p];
    let mut exponent = 0_i64;
    for block in 0..b {
        for row in 0..m {
            let mut largest = coupling_entry;
            for column in 0..m {
                largest = largest.max(entry(block, row, column).abs());
            }
            if block == b - 1 {
                for vector in column_border {
                    largest = largest.max(vector[row].abs());
                }
            }
            let (scale, power) = inverse_power_of_two(largest);
            row_scale[block * m + row] = scale;
            exponent += power;
        }
    }
    for (slot, vector) in row_border.iter().enumerate() {
        let largest = vector
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        let (scale, power) = inverse_power_of_two(largest);
        row_scale[b * m + slot] = scale;
        exponent += power;
    }
    for block in 0..b {
        let previous = (block + b - 1) % b;
        for column in 0..m {
            let mut largest = coupling_entry * row_scale[previous * m + column];
            for row in 0..m {
                largest =
                    largest.max((entry(block, row, column) * row_scale[block * m + row]).abs());
            }
            if block == 0 {
                for (slot, vector) in row_border.iter().enumerate() {
                    largest = largest.max((vector[column] * row_scale[b * m + slot]).abs());
                }
            }
            let (scale, power) = inverse_power_of_two(largest);
            column_scale[block * m + column] = scale;
            exponent += power;
        }
    }
    for (slot, vector) in column_border.iter().enumerate() {
        let mut largest = 0.0_f64;
        for row in 0..m {
            largest = largest.max((vector[row] * row_scale[(b - 1) * m + row]).abs());
        }
        let (scale, power) = inverse_power_of_two(largest);
        column_scale[b * m + slot] = scale;
        exponent += power;
    }
    let scaled_transfers = transfers
        .iter()
        .enumerate()
        .map(|(block, transfer)| {
            (0..m * m)
                .map(|index| {
                    let (row, column) = (index / m, index % m);
                    row_scale[block * m + row] * transfer[index] * column_scale[block * m + column]
                })
                .collect()
        })
        .collect();
    let coupling = (0..b)
        .map(|block| {
            let next = (block + 1) % b;
            (0..m)
                .map(|row| row_scale[block * m + row] * column_scale[next * m + row])
                .collect()
        })
        .collect();
    let scaled_column = column_border
        .iter()
        .enumerate()
        .map(|(slot, vector)| {
            (0..m)
                .map(|row| row_scale[(b - 1) * m + row] * vector[row] * column_scale[b * m + slot])
                .collect()
        })
        .collect();
    let scaled_row = row_border
        .iter()
        .enumerate()
        .map(|(slot, vector)| {
            (0..m)
                .map(|column| row_scale[b * m + slot] * vector[column] * column_scale[column])
                .collect()
        })
        .collect();
    Equilibrated {
        transfers: scaled_transfers,
        coupling,
        column_border: scaled_column,
        row_border: scaled_row,
        row_scale,
        column_scale,
        exponent,
    }
}

/// Factor the bordered cyclic block matrix and return `ln|det|`.
///
/// `transfers[i]` is the row-major `m x m` block `T_i`. Row block `i`
/// holds `-T_i` in block column `i` and the identity in block column
/// `(i + 1) mod b`; with one block the two coincide as `I - T_0`.
/// `column_border[s]` (length `m`) fills border column `s` in the rows
/// of block `b - 1`. `row_border[s]` (length `m`) fills border row `s`
/// in the columns of block `0`. The two border lists have equal length
/// `p` and the trailing `p x p` block is zero.
pub fn bordered_cyclic_log_abs(
    transfers: &[Vec<f64>],
    m: usize,
    column_border: &[Vec<f64>],
    row_border: &[Vec<f64>],
) -> Result<CyclicFactor, CyclicError> {
    let b = transfers.len();
    let p = column_border.len();
    if b == 0 || m == 0 {
        return Err(CyclicError::Shape("no blocks"));
    }
    if p > m {
        return Err(CyclicError::Shape("border count exceeds block size"));
    }
    if row_border.len() != p {
        return Err(CyclicError::Shape("border row and column counts differ"));
    }
    let block_len = m
        .checked_mul(m)
        .ok_or(CyclicError::Shape("dimensions overflow"))?;
    let matrix_order = b
        .checked_mul(m)
        .and_then(|value| value.checked_add(p))
        .ok_or(CyclicError::Shape("dimensions overflow"))?;
    if transfers.iter().any(|block| block.len() != block_len) {
        return Err(CyclicError::Shape("a block is not m x m"));
    }
    if column_border
        .iter()
        .chain(row_border)
        .any(|vector| vector.len() != m)
    {
        return Err(CyclicError::Shape("a border vector is not length m"));
    }
    if transfers
        .iter()
        .chain(column_border)
        .chain(row_border)
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err(CyclicError::NonFinite);
    }

    let mut column_norms1 = vec![0.0; matrix_order];
    for (block, transfer) in transfers.iter().enumerate() {
        for column in 0..m {
            let mut sum = 0.0;
            for row in 0..m {
                let mut entry = -transfer[row * m + column];
                if b == 1 && row == column {
                    entry += 1.0;
                }
                sum += entry.abs();
                if !sum.is_finite() {
                    return Err(CyclicError::NonFiniteNormEstimate);
                }
            }
            if b > 1 {
                sum += 1.0;
                if !sum.is_finite() {
                    return Err(CyclicError::NonFiniteNormEstimate);
                }
            }
            if block == 0 {
                for vector in row_border {
                    sum += vector[column].abs();
                    if !sum.is_finite() {
                        return Err(CyclicError::NonFiniteNormEstimate);
                    }
                }
            }
            column_norms1[block * m + column] = sum;
        }
    }
    for (slot, vector) in column_border.iter().enumerate() {
        column_norms1[b * m + slot] = finite_abs_sum(vector)?;
    }
    let norm1 = column_norms1
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(*value));

    let Equilibrated {
        transfers,
        coupling,
        column_border,
        row_border,
        row_scale,
        column_scale,
        exponent,
    } = equilibrate(transfers, m, column_border, row_border);
    let mut steps = Vec::with_capacity(b.saturating_sub(1));
    let mut log_abs = 0.0;
    // Rows of the running block row: `m + p` after every step, holding
    // block column `j` (the next to eliminate), block column `b - 1` and
    // the border.
    let mut carry: Vec<f64> = Vec::new();
    for j in 0..b.saturating_sub(1) {
        let has_next = j + 2 < b;
        let next_width = if has_next { m } else { 0 };
        let off_next = m;
        let off_last = m + next_width;
        let off_border = off_last + m;
        let width = off_border + p;
        let rows = 2 * m + p;
        let mut work = vec![0.0; rows * width];
        if j == 0 {
            for row in 0..m {
                for column in 0..m {
                    work[row * width + column] = -transfers[0][row * m + column];
                    work[(m + row) * width + off_last + column] =
                        -transfers[b - 1][row * m + column];
                }
                work[(m + row) * width + row] = coupling[b - 1][row];
                for slot in 0..p {
                    work[(m + row) * width + off_border + slot] = column_border[slot][row];
                }
            }
            for (slot, vector) in row_border.iter().enumerate() {
                for column in 0..m {
                    work[(2 * m + slot) * width + column] = vector[column];
                }
            }
        } else {
            let carried_width = m + m + p;
            for row in 0..m {
                for column in 0..m {
                    work[row * width + column] = -transfers[j][row * m + column];
                }
            }
            for row in 0..(m + p) {
                let target = (m + row) * width;
                for column in 0..m {
                    work[target + column] = carry[row * carried_width + column];
                    work[target + off_last + column] = carry[row * carried_width + m + column];
                }
                for slot in 0..p {
                    work[target + off_border + slot] = carry[row * carried_width + 2 * m + slot];
                }
            }
        }
        // Row block `j` steps into block column `j + 1`.
        let target = if has_next { off_next } else { off_last };
        for row in 0..m {
            work[row * width + target + row] += coupling[j][row];
        }
        let panel = householder(&mut work, rows, width, m);
        let r = upper_block(&work, width, m, 0);
        log_abs += diagonal_log(&r, m)?;
        let next = if has_next {
            sub_block(&work, width, 0, m, off_next, m)
        } else {
            Vec::new()
        };
        let last = sub_block(&work, width, 0, m, off_last, m);
        let border = sub_block(&work, width, 0, m, off_border, p);
        let carried_width = m + m + p;
        let mut next_carry = vec![0.0; (m + p) * carried_width];
        for row in 0..(m + p) {
            let source = (m + row) * width;
            if has_next {
                next_carry[row * carried_width..row * carried_width + m]
                    .copy_from_slice(&work[source + off_next..source + off_next + m]);
            }
            next_carry[row * carried_width + m..row * carried_width + 2 * m]
                .copy_from_slice(&work[source + off_last..source + off_last + m]);
            next_carry[row * carried_width + 2 * m..(row + 1) * carried_width]
                .copy_from_slice(&work[source + off_border..source + off_border + p]);
        }
        carry = next_carry;
        steps.push(Step {
            panel,
            r,
            next,
            last,
            border,
        });
    }
    let order = m + p;
    let mut work = vec![0.0; order * order];
    if b == 1 {
        for row in 0..m {
            for column in 0..m {
                work[row * order + column] = -transfers[0][row * m + column];
            }
            work[row * order + row] += coupling[0][row];
            for slot in 0..p {
                work[row * order + m + slot] = column_border[slot][row];
            }
        }
        for (slot, vector) in row_border.iter().enumerate() {
            for column in 0..m {
                work[(m + slot) * order + column] = vector[column];
            }
        }
    } else {
        let carried_width = m + m + p;
        for row in 0..order {
            work[row * order..(row + 1) * order]
                .copy_from_slice(&carry[row * carried_width + m..(row + 1) * carried_width]);
        }
    }
    let last = householder(&mut work, order, order, order);
    let last_r = upper_block(&work, order, order, 0);
    log_abs += diagonal_log(&last_r, order)?;
    log_abs -= exponent as f64 * std::f64::consts::LN_2;
    let mut factor = CyclicFactor {
        m,
        p,
        blocks: b,
        steps,
        last,
        row_scale,
        column_scale,
        last_r,
        log_abs,
        norm1,
        inverse_norm1: 0.0,
    };
    factor.inverse_norm1 = factor.estimate_inverse_norm1()?;
    Ok(factor)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assemble(
        transfers: &[Vec<f64>],
        m: usize,
        column: &[Vec<f64>],
        row: &[Vec<f64>],
    ) -> Vec<f64> {
        let b = transfers.len();
        let p = column.len();
        let n = b * m + p;
        let mut a = vec![0.0; n * n];
        for block in 0..b {
            let next = (block + 1) % b;
            for i in 0..m {
                a[(block * m + i) * n + next * m + i] += 1.0;
                for j in 0..m {
                    a[(block * m + i) * n + block * m + j] -= transfers[block][i * m + j];
                }
            }
        }
        for slot in 0..p {
            for i in 0..m {
                a[((b - 1) * m + i) * n + b * m + slot] = column[slot][i];
                a[(b * m + slot) * n + i] = row[slot][i];
            }
        }
        a
    }

    /// Dense partial-pivoting LU, the reference for small orders.
    fn dense_log_abs(mut a: Vec<f64>, n: usize) -> f64 {
        let mut total = 0.0;
        for k in 0..n {
            let mut pivot = k;
            for row in k..n {
                if a[row * n + k].abs() > a[pivot * n + k].abs() {
                    pivot = row;
                }
            }
            for column in 0..n {
                a.swap(k * n + column, pivot * n + column);
            }
            let diagonal = a[k * n + k];
            total += diagonal.abs().ln();
            for row in (k + 1)..n {
                let factor = a[row * n + k] / diagonal;
                for column in k..n {
                    a[row * n + column] -= factor * a[k * n + column];
                }
            }
        }
        total
    }

    fn dense_inverse_norm1(a: &[f64], n: usize) -> f64 {
        let mut best = 0.0_f64;
        for column in 0..n {
            let mut work = a.to_vec();
            let mut rhs = vec![0.0; n];
            rhs[column] = 1.0;
            for k in 0..n {
                let mut pivot = k;
                for row in k..n {
                    if work[row * n + k].abs() > work[pivot * n + k].abs() {
                        pivot = row;
                    }
                }
                for c in 0..n {
                    work.swap(k * n + c, pivot * n + c);
                }
                rhs.swap(k, pivot);
                for row in (k + 1)..n {
                    let factor = work[row * n + k] / work[k * n + k];
                    for c in k..n {
                        work[row * n + c] -= factor * work[k * n + c];
                    }
                    rhs[row] -= factor * rhs[k];
                }
            }
            for row in (0..n).rev() {
                let mut sum = rhs[row];
                for c in (row + 1)..n {
                    sum -= work[row * n + c] * rhs[c];
                }
                rhs[row] = sum / work[row * n + row];
            }
            best = best.max(rhs.iter().map(|value| value.abs()).sum());
        }
        best
    }

    /// Deterministic entries in `[-1, 1)`.
    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    type Case = (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>);

    fn random_case(b: usize, m: usize, p: usize, seed: u64) -> Case {
        let mut state = seed;
        let transfers = (0..b)
            .map(|_| (0..m * m).map(|_| 1.5 * lcg(&mut state)).collect())
            .collect();
        let column = (0..p)
            .map(|_| (0..m).map(|_| lcg(&mut state)).collect())
            .collect();
        let row = (0..p)
            .map(|_| (0..m).map(|_| lcg(&mut state)).collect())
            .collect();
        (transfers, column, row)
    }

    #[test]
    fn small_cyclic_matrices_match_dense_elimination() {
        for (b, m, p) in [
            (1, 1, 0),
            (1, 3, 1),
            (2, 2, 0),
            (2, 4, 2),
            (3, 2, 1),
            (5, 3, 2),
            (7, 4, 0),
        ] {
            let (transfers, column, row) = random_case(b, m, p, 17 + (b * 31 + m * 7 + p) as u64);
            let factor = bordered_cyclic_log_abs(&transfers, m, &column, &row).expect("factor");
            let n = b * m + p;
            let dense = assemble(&transfers, m, &column, &row);
            let reference = dense_log_abs(dense.clone(), n);
            let bar = 64.0 * (n * n) as f64 * f64::EPSILON * factor.inverse_norm1 * factor.norm1;
            assert!(
                (factor.log_abs - reference).abs() <= bar,
                "b {b} m {m} p {p}: {} vs {reference}, bar {bar}",
                factor.log_abs
            );
        }
    }

    #[test]
    fn stored_factors_solve_both_systems() {
        for (b, m, p) in [(1, 3, 1), (2, 3, 2), (3, 2, 0), (6, 3, 2)] {
            solve_both_systems(b, m, p);
        }
    }

    fn solve_both_systems(b: usize, m: usize, p: usize) {
        let (transfers, column, row) = random_case(b, m, p, 99 + b as u64);
        let factor = bordered_cyclic_log_abs(&transfers, m, &column, &row).expect("factor");
        let n = factor.order();
        let a = assemble(&transfers, m, &column, &row);
        let mut state = 5u64;
        let x: Vec<f64> = (0..n).map(|_| lcg(&mut state)).collect();
        let mut ax = vec![0.0; n];
        let mut atx = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                ax[i] += a[i * n + j] * x[j];
                atx[i] += a[j * n + i] * x[j];
            }
        }
        let solved = factor.solve(&ax).expect("solve");
        let transposed = factor.solve_transpose(&atx).expect("transpose solve");
        let bar = 64.0 * n as f64 * f64::EPSILON * factor.inverse_norm1 * factor.norm1;
        for i in 0..n {
            assert!(
                (solved[i] - x[i]).abs() <= bar,
                "solve {i}: {} vs {}",
                solved[i],
                x[i]
            );
            assert!(
                (transposed[i] - x[i]).abs() <= bar,
                "transpose {i}: {} vs {}",
                transposed[i],
                x[i]
            );
        }
    }

    #[test]
    fn the_condition_estimate_is_a_lower_bound_within_a_factor_of_three() {
        for seed in [3u64, 11, 29, 47] {
            let (transfers, column, row) = random_case(4, 3, 1, seed);
            let factor = bordered_cyclic_log_abs(&transfers, 3, &column, &row).expect("factor");
            let n = factor.order();
            let exact = dense_inverse_norm1(&assemble(&transfers, 3, &column, &row), n);
            let slack = 1.0 + 64.0 * n as f64 * f64::EPSILON * exact * factor.norm1;
            assert!(
                factor.inverse_norm1 <= exact * slack,
                "seed {seed}: {} > {exact}",
                factor.inverse_norm1
            );
            assert!(
                3.0 * factor.inverse_norm1 >= exact,
                "seed {seed}: {} << {exact}",
                factor.inverse_norm1
            );
        }
    }

    fn oscillator(h: f64, omega: f64) -> Vec<f64> {
        let (c, s) = ((h * omega).cosh(), (h * omega).sinh());
        vec![c, s / omega, omega * s, c]
    }

    /// `det(I - M) = 2 - 2 cosh(βω)` for the hyperbolic oscillator. At
    /// `βω = 2000` the monodromy entries are `e^{1000}`-sized and do not
    /// exist in binary64; the factors are `e^{2000/b}`-sized.
    #[test]
    fn an_overflowing_monodromy_keeps_its_log_determinant() {
        let omega = 0.5;
        let beta = 4000.0;
        let b = 256usize;
        let h = beta / b as f64;
        let transfers: Vec<Vec<f64>> = (0..b).map(|_| oscillator(h, omega)).collect();
        let factor = bordered_cyclic_log_abs(&transfers, 2, &[], &[]).expect("factor");
        let x = beta * omega;
        let exact = x + (-x).exp_m1().abs().ln() * 2.0;
        let gap = (factor.log_abs - exact).abs();
        assert!(
            gap <= 1.0e-11 * x,
            "log|det| {} exact {exact} gap {gap}",
            factor.log_abs
        );
        assert!(factor.inverse_norm1.is_finite() && factor.inverse_norm1 < 1.0e3);
    }

    /// An `e^{1500}` hyperbolic pair and an elliptic pair, rotated into
    /// each other by one fixed orthogonal frame. The elliptic factor
    /// `|2 - 2 cos(βγ)|` is order one and must survive beside the other.
    #[test]
    fn an_elliptic_pair_survives_beside_an_overflowing_pair() {
        let (omega, gamma, beta) = (0.75, 0.3, 2000.0);
        let b = 512usize;
        let h = beta / b as f64;
        let (c, s) = (0.6_f64, 0.8_f64);
        let frame = [c, -s, s, c];
        let hyperbolic = oscillator(h, omega);
        let (cg, sg) = ((h * gamma).cos(), (h * gamma).sin());
        let elliptic = [cg, sg / gamma, -gamma * sg, cg];
        // Sub-block `entry` of the 4 x 4 transfer is Q diag(hyp[entry], ell[entry]) Q^T.
        let mut local = vec![0.0; 16];
        for entry in 0..4 {
            let (row0, column0) = (2 * (entry / 2), 2 * (entry % 2));
            let diagonal = [hyperbolic[entry], elliptic[entry]];
            for i in 0..2 {
                for j in 0..2 {
                    let mut sum = 0.0;
                    for mode in 0..2 {
                        sum += frame[i * 2 + mode] * diagonal[mode] * frame[j * 2 + mode];
                    }
                    local[(row0 + i) * 4 + column0 + j] = sum;
                }
            }
        }
        let transfers: Vec<Vec<f64>> = (0..b).map(|_| local.clone()).collect();
        let factor = bordered_cyclic_log_abs(&transfers, 4, &[], &[]).expect("factor");
        let x = beta * omega;
        let exact =
            x + 2.0 * (-x).exp_m1().abs().ln() + (2.0 - 2.0 * (beta * gamma).cos()).abs().ln();
        let gap = (factor.log_abs - exact).abs();
        assert!(
            gap <= 1.0e-10 * x,
            "log|det| {} exact {exact} gap {gap}",
            factor.log_abs
        );
    }

    /// Two hyperbolic modes eighty-fold apart in frequency, mixed by a
    /// fixed rotation and written in coordinates with units spanning
    /// five decades. A change of units `S T S^{-1}` leaves `det(I - M)`,
    /// the product of the two scalar factors, unchanged, but leaves rows
    /// and columns that differ by up to `10^9`.
    #[test]
    fn an_anisotropic_hyperbolic_period_keeps_double_precision() {
        for (b, beta) in [(12usize, 1.5), (32, 4.5)] {
            anisotropic_period(b, beta);
        }
    }

    fn anisotropic_period(b: usize, beta: f64) {
        let (fast, slow) = (40.0, 0.5);
        let h = beta / b as f64;
        let (c, s) = (0.6_f64, 0.8_f64);
        let frame = [c, -s, s, c];
        let first = oscillator(h, fast);
        let second = oscillator(h, slow);
        let mut local = vec![0.0; 16];
        for entry in 0..4 {
            let (row0, column0) = (2 * (entry / 2), 2 * (entry % 2));
            let diagonal = [first[entry], second[entry]];
            for i in 0..2 {
                for j in 0..2 {
                    let mut sum = 0.0;
                    for mode in 0..2 {
                        sum += frame[i * 2 + mode] * diagonal[mode] * frame[j * 2 + mode];
                    }
                    local[(row0 + i) * 4 + column0 + j] = sum;
                }
            }
        }
        let units = [1.0, 3.0e3, 7.0e-2, 41.0];
        for (index, value) in local.iter_mut().enumerate() {
            *value *= units[index / 4] / units[index % 4];
        }
        let transfers: Vec<Vec<f64>> = (0..b).map(|_| local.clone()).collect();
        let factor = bordered_cyclic_log_abs(&transfers, 4, &[], &[]).expect("factor");
        let scalar = |x: f64| x + 2.0 * (-x).exp_m1().abs().ln();
        let exact = scalar(beta * fast) + (2.0 * (beta * slow).cosh() - 2.0).ln();
        let gap = (factor.log_abs - exact).abs();
        assert!(
            gap <= 16.0 * (4 * b) as f64 * f64::EPSILON * exact.abs(),
            "b {b}: log|det| {} exact {exact} gap {gap}",
            factor.log_abs
        );
    }

    /// The free particle has the Jordan block `[[1, β], [0, 1]]` at the
    /// multiplier one. Bordering the velocity row and the position column
    /// leaves `|det| = β`.
    #[test]
    fn a_bordered_jordan_block_returns_the_period() {
        let beta = 7.0;
        for b in [1usize, 2, 5, 16] {
            let h = beta / b as f64;
            let transfers: Vec<Vec<f64>> = (0..b).map(|_| vec![1.0, h, 0.0, 1.0]).collect();
            let column = [vec![0.0, 1.0]];
            let row = [vec![1.0, 0.0]];
            let factor = bordered_cyclic_log_abs(&transfers, 2, &column, &row).expect("factor");
            let dense = dense_log_abs(assemble(&transfers, 2, &column, &row), 2 * b + 1);
            assert!(
                (factor.log_abs - beta.ln()).abs() <= 64.0 * b as f64 * f64::EPSILON,
                "b {b}: {}",
                factor.log_abs
            );
            assert!(
                (dense - beta.ln()).abs() <= 64.0 * b as f64 * f64::EPSILON,
                "dense b {b}: {dense}"
            );
        }
    }

    #[test]
    fn malformed_and_singular_inputs_are_refused() {
        assert_eq!(
            bordered_cyclic_log_abs(&[], 2, &[], &[]).unwrap_err(),
            CyclicError::Shape("no blocks")
        );
        assert_eq!(
            bordered_cyclic_log_abs(&[vec![1.0; 3]], 2, &[], &[]).unwrap_err(),
            CyclicError::Shape("a block is not m x m")
        );
        assert_eq!(
            bordered_cyclic_log_abs(&[vec![1.0; 4]], 2, &[vec![1.0]], &[vec![1.0]]).unwrap_err(),
            CyclicError::Shape("a border vector is not length m")
        );
        assert_eq!(
            bordered_cyclic_log_abs(&[vec![0.0; 4]], 2, &[vec![1.0, 0.0]], &[]).unwrap_err(),
            CyclicError::Shape("border row and column counts differ")
        );
        let two_borders = [vec![1.0], vec![2.0]];
        assert_eq!(
            bordered_cyclic_log_abs(&[vec![2.0]], 1, &two_borders, &two_borders).unwrap_err(),
            CyclicError::Shape("border count exceeds block size")
        );
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                bordered_cyclic_log_abs(&[vec![bad, 0.0, 0.0, 1.0]], 2, &[], &[]).unwrap_err(),
                CyclicError::NonFinite
            );
            assert_eq!(
                bordered_cyclic_log_abs(
                    &[vec![2.0, 0.0, 0.0, 2.0]],
                    2,
                    &[vec![bad, 0.0]],
                    &[vec![1.0, 0.0]],
                )
                .unwrap_err(),
                CyclicError::NonFinite
            );
            assert_eq!(
                bordered_cyclic_log_abs(
                    &[vec![2.0, 0.0, 0.0, 2.0]],
                    2,
                    &[vec![1.0, 0.0]],
                    &[vec![0.0, bad]],
                )
                .unwrap_err(),
                CyclicError::NonFinite
            );
        }
        assert_eq!(
            bordered_cyclic_log_abs(&[vec![f64::MAX, 0.0, f64::MAX, 2.0]], 2, &[], &[],)
                .unwrap_err(),
            CyclicError::NonFiniteNormEstimate
        );
        let identity = vec![1.0, 0.0, 0.0, 1.0];
        assert_eq!(
            bordered_cyclic_log_abs(&[identity.clone(), identity], 2, &[], &[]).unwrap_err(),
            CyclicError::Singular
        );
    }

    #[test]
    fn a_nan_inverse_norm_estimate_is_not_silently_dropped() {
        let mut factor = bordered_cyclic_log_abs(&[vec![2.0]], 1, &[], &[]).expect("factor");
        assert_eq!(
            factor.solve(&[]).unwrap_err(),
            CyclicError::Shape("right-hand side length")
        );
        assert_eq!(
            factor.solve_transpose(&[1.0, 2.0]).unwrap_err(),
            CyclicError::Shape("right-hand side length")
        );
        factor.last_r[0] = f64::NAN;
        assert_eq!(
            factor.estimate_inverse_norm1().unwrap_err(),
            CyclicError::NonFiniteNormEstimate
        );
    }
}
