//! Linear-storage factorization of a periodic block-tridiagonal matrix.
//!
//! The open chain is eliminated by block Gaussian elimination.  The two
//! corner blocks are then restored by a rank-`2 * block_size` determinant
//! lemma and Woodbury solve.  This is intended for bead Hessians: the block
//! size is the number of spatial degrees of freedom and the number of blocks
//! is the bead count.

use thiserror::Error;

/// Failure from a periodic block-tridiagonal factorization or solve.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum PeriodicBlockError {
    /// At least three blocks and a nonzero block size are required.
    #[error("periodic block dimensions require at least three nonempty blocks")]
    InvalidDimensions,
    /// One of the packed block slices has the wrong length.
    #[error("{name} has length {got}, expected {expected}")]
    DimensionMismatch {
        /// Name of the rejected slice.
        name: &'static str,
        /// Required length.
        expected: usize,
        /// Supplied length.
        got: usize,
    },
    /// An input or an arithmetic result was not finite.
    #[error("periodic block arithmetic was not finite")]
    NonFinite,
    /// A diagonal Schur block or the cyclic closure was numerically singular.
    #[error("periodic block pivot {pivot} was singular at scale {scale:e}")]
    Singular {
        /// Scalar pivot within the dense factorization.
        pivot: usize,
        /// Largest absolute entry before elimination.
        scale: f64,
    },
    /// A workspace size overflowed `usize`.
    #[error("periodic block workspace size overflowed")]
    CapacityOverflow,
}

#[derive(Debug, Clone)]
struct DenseLu {
    data: Vec<f64>,
    n: usize,
    swaps: Vec<usize>,
    sign: i8,
    log_abs_det: f64,
}

impl DenseLu {
    fn factor(mut data: Vec<f64>, n: usize) -> Result<Self, PeriodicBlockError> {
        let scale = data.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
        if !scale.is_finite() {
            return Err(PeriodicBlockError::NonFinite);
        }
        let threshold = (n as f64) * f64::EPSILON * scale.max(f64::MIN_POSITIVE);
        let mut swaps = Vec::new();
        swaps
            .try_reserve_exact(n)
            .map_err(|_| PeriodicBlockError::CapacityOverflow)?;
        let mut sign = 1_i8;
        let mut log_abs_det = 0.0;
        for column in 0..n {
            let mut pivot = column;
            let mut largest = data[column * n + column].abs();
            for row in column + 1..n {
                let candidate = data[row * n + column].abs();
                if candidate > largest {
                    largest = candidate;
                    pivot = row;
                }
            }
            if !largest.is_finite() {
                return Err(PeriodicBlockError::NonFinite);
            }
            if largest <= threshold {
                return Err(PeriodicBlockError::Singular {
                    pivot: column,
                    scale,
                });
            }
            swaps.push(pivot);
            if pivot != column {
                for entry in 0..n {
                    data.swap(column * n + entry, pivot * n + entry);
                }
                sign = -sign;
            }
            let diagonal = data[column * n + column];
            if !diagonal.is_finite() {
                return Err(PeriodicBlockError::NonFinite);
            }
            if diagonal < 0.0 {
                sign = -sign;
            }
            log_abs_det += diagonal.abs().ln();
            for row in column + 1..n {
                data[row * n + column] /= diagonal;
                let multiplier = data[row * n + column];
                for entry in column + 1..n {
                    data[row * n + entry] -= multiplier * data[column * n + entry];
                }
            }
        }
        if data.iter().any(|value| !value.is_finite()) || !log_abs_det.is_finite() {
            return Err(PeriodicBlockError::NonFinite);
        }
        Ok(Self {
            data,
            n,
            swaps,
            sign,
            log_abs_det,
        })
    }

    fn solve_in_place(&self, rhs: &mut [f64], columns: usize) -> Result<(), PeriodicBlockError> {
        let expected = self
            .n
            .checked_mul(columns)
            .ok_or(PeriodicBlockError::CapacityOverflow)?;
        if rhs.len() != expected {
            return Err(PeriodicBlockError::DimensionMismatch {
                name: "dense right hand side",
                expected,
                got: rhs.len(),
            });
        }
        for row in 0..self.n {
            let pivot = self.swaps[row];
            if pivot != row {
                for column in 0..columns {
                    rhs.swap(row * columns + column, pivot * columns + column);
                }
            }
        }
        for row in 0..self.n {
            for previous in 0..row {
                let value = self.data[row * self.n + previous];
                for column in 0..columns {
                    rhs[row * columns + column] -= value * rhs[previous * columns + column];
                }
            }
        }
        for row in (0..self.n).rev() {
            for following in row + 1..self.n {
                let value = self.data[row * self.n + following];
                for column in 0..columns {
                    rhs[row * columns + column] -= value * rhs[following * columns + column];
                }
            }
            let diagonal = self.data[row * self.n + row];
            for column in 0..columns {
                rhs[row * columns + column] /= diagonal;
            }
        }
        if rhs.iter().any(|value| !value.is_finite()) {
            return Err(PeriodicBlockError::NonFinite);
        }
        Ok(())
    }
}

/// Reusable factor of a real periodic block-tridiagonal matrix.
///
/// Blocks are row-major. `diagonal` contains `blocks` square blocks.
/// `upper` contains the `blocks - 1` open-chain blocks at `(i, i + 1)`;
/// their transposes are used below the diagonal. `corner` is the block at
/// `(blocks - 1, 0)` and its transpose is used at `(0, blocks - 1)`.
#[derive(Debug, Clone)]
pub struct PeriodicBlockFactor {
    blocks: usize,
    block_size: usize,
    schur: Vec<DenseLu>,
    upper: Vec<f64>,
    corner: Vec<f64>,
    lower_multiplier: Vec<f64>,
    woodbury_basis: Vec<f64>,
    closure: DenseLu,
    sign: i8,
    log_abs_det: f64,
}

/// Determinant-only factor of a bordered periodic block-tridiagonal matrix.
///
/// Unlike [`PeriodicBlockFactor`], this path permits the unbordered periodic
/// matrix to be singular. It eliminates the nonsingular open chain and
/// factors one dense matrix of order `2 * block_size + border_rank`.
#[derive(Debug, Clone)]
pub struct PeriodicBorderedFactor {
    sign: i8,
    log_abs_det: f64,
}

impl PeriodicBorderedFactor {
    /// Factor `[A Q; Q^T 0]`, with `A` in periodic block form.
    pub fn factor(
        diagonal: &[f64],
        upper: &[f64],
        corner: &[f64],
        border: &[f64],
        blocks: usize,
        block_size: usize,
        border_rank: usize,
    ) -> Result<Self, PeriodicBlockError> {
        if blocks < 3 || block_size == 0 || border_rank == 0 {
            return Err(PeriodicBlockError::InvalidDimensions);
        }
        let block_entries = block_size
            .checked_mul(block_size)
            .ok_or(PeriodicBlockError::CapacityOverflow)?;
        check_len("diagonal", diagonal, blocks, block_entries)?;
        check_len("upper", upper, blocks - 1, block_entries)?;
        check_len("corner", corner, 1, block_entries)?;
        let full_size = blocks
            .checked_mul(block_size)
            .ok_or(PeriodicBlockError::CapacityOverflow)?;
        let border_len = full_size
            .checked_mul(border_rank)
            .ok_or(PeriodicBlockError::CapacityOverflow)?;
        if border.len() != border_len {
            return Err(PeriodicBlockError::DimensionMismatch {
                name: "border",
                expected: border_len,
                got: border.len(),
            });
        }
        if diagonal
            .iter()
            .chain(upper)
            .chain(corner)
            .chain(border)
            .any(|value| !value.is_finite())
        {
            return Err(PeriodicBlockError::NonFinite);
        }

        let mut schur: Vec<DenseLu> = Vec::new();
        schur
            .try_reserve_exact(blocks)
            .map_err(|_| PeriodicBlockError::CapacityOverflow)?;
        let multipliers_len = (blocks - 1)
            .checked_mul(block_entries)
            .ok_or(PeriodicBlockError::CapacityOverflow)?;
        let mut lower_multiplier = try_zeros(multipliers_len)?;
        let mut sign = 1_i8;
        let mut log_abs_det = 0.0;
        for block in 0..blocks {
            let mut current = diagonal[block * block_entries..(block + 1) * block_entries].to_vec();
            if block > 0 {
                let previous_upper = &upper[(block - 1) * block_entries..block * block_entries];
                let mut solved_upper = previous_upper.to_vec();
                schur[block - 1].solve_in_place(&mut solved_upper, block_size)?;
                let multiplier = transpose(&solved_upper, block_size);
                lower_multiplier[(block - 1) * block_entries..block * block_entries]
                    .copy_from_slice(&multiplier);
                subtract_product(&mut current, &multiplier, previous_upper, block_size)?;
            }
            let factor = DenseLu::factor(current, block_size)?;
            sign *= factor.sign;
            log_abs_det += factor.log_abs_det;
            schur.push(factor);
        }

        let cyclic_rank = 2 * block_size;
        let reduced_rank = cyclic_rank
            .checked_add(border_rank)
            .ok_or(PeriodicBlockError::CapacityOverflow)?;
        let mut basis = try_zeros(
            full_size
                .checked_mul(reduced_rank)
                .ok_or(PeriodicBlockError::CapacityOverflow)?,
        )?;
        for coordinate in 0..block_size {
            basis[coordinate * reduced_rank + coordinate] = 1.0;
            let row = (blocks - 1) * block_size + coordinate;
            basis[row * reduced_rank + block_size + coordinate] = 1.0;
        }
        for row in 0..full_size {
            for column in 0..border_rank {
                basis[row * reduced_rank + cyclic_rank + column] =
                    border[row * border_rank + column];
            }
        }
        solve_open(
            &schur,
            upper,
            &lower_multiplier,
            blocks,
            block_size,
            &mut basis,
            reduced_rank,
        )?;

        let mut reduced = try_zeros(
            reduced_rank
                .checked_mul(reduced_rank)
                .ok_or(PeriodicBlockError::CapacityOverflow)?,
        )?;
        for index in 0..cyclic_rank {
            reduced[index * reduced_rank + index] = 1.0;
        }
        for row in 0..block_size {
            for column in 0..reduced_rank {
                let mut top = 0.0;
                let mut bottom = 0.0;
                for inner in 0..block_size {
                    top += corner[inner * block_size + row]
                        * basis[((blocks - 1) * block_size + inner) * reduced_rank + column];
                    bottom +=
                        corner[row * block_size + inner] * basis[inner * reduced_rank + column];
                }
                reduced[row * reduced_rank + column] += top;
                reduced[(block_size + row) * reduced_rank + column] += bottom;
            }
        }
        for row in 0..border_rank {
            for column in 0..reduced_rank {
                let mut value = 0.0;
                for inner in 0..full_size {
                    value -=
                        border[inner * border_rank + row] * basis[inner * reduced_rank + column];
                }
                reduced[(cyclic_rank + row) * reduced_rank + column] = value;
            }
        }
        let reduced_factor = DenseLu::factor(reduced, reduced_rank)?;
        sign *= reduced_factor.sign;
        log_abs_det += reduced_factor.log_abs_det;
        Ok(Self { sign, log_abs_det })
    }

    /// Sign of the bordered determinant (`-1` or `1`).
    pub fn determinant_sign(&self) -> i8 {
        self.sign
    }

    /// Natural logarithm of the bordered determinant's absolute value.
    pub fn log_abs_determinant(&self) -> f64 {
        self.log_abs_det
    }
}

impl PeriodicBlockFactor {
    /// Factor a symmetric periodic block-tridiagonal matrix.
    pub fn factor(
        diagonal: &[f64],
        upper: &[f64],
        corner: &[f64],
        blocks: usize,
        block_size: usize,
    ) -> Result<Self, PeriodicBlockError> {
        if blocks < 3 || block_size == 0 {
            return Err(PeriodicBlockError::InvalidDimensions);
        }
        let block_entries = block_size
            .checked_mul(block_size)
            .ok_or(PeriodicBlockError::CapacityOverflow)?;
        check_len("diagonal", diagonal, blocks, block_entries)?;
        check_len("upper", upper, blocks - 1, block_entries)?;
        check_len("corner", corner, 1, block_entries)?;
        if diagonal
            .iter()
            .chain(upper)
            .chain(corner)
            .any(|value| !value.is_finite())
        {
            return Err(PeriodicBlockError::NonFinite);
        }

        let mut schur: Vec<DenseLu> = Vec::new();
        schur
            .try_reserve_exact(blocks)
            .map_err(|_| PeriodicBlockError::CapacityOverflow)?;
        let multipliers_len = (blocks - 1)
            .checked_mul(block_entries)
            .ok_or(PeriodicBlockError::CapacityOverflow)?;
        let mut lower_multiplier = try_zeros(multipliers_len)?;
        let mut sign = 1_i8;
        let mut log_abs_det = 0.0;
        for block in 0..blocks {
            let mut current = diagonal[block * block_entries..(block + 1) * block_entries].to_vec();
            if block > 0 {
                let previous_upper = &upper[(block - 1) * block_entries..block * block_entries];
                let mut solved_upper = previous_upper.to_vec();
                schur[block - 1].solve_in_place(&mut solved_upper, block_size)?;
                // transpose(S^{-1} B) = B^T S^{-T}; Schur blocks are symmetric,
                // while the LU solve remains valid without assuming its factors are.
                let multiplier = transpose(&solved_upper, block_size);
                lower_multiplier[(block - 1) * block_entries..block * block_entries]
                    .copy_from_slice(&multiplier);
                subtract_product(&mut current, &multiplier, previous_upper, block_size)?;
            }
            let factor = DenseLu::factor(current, block_size)?;
            sign *= factor.sign;
            log_abs_det += factor.log_abs_det;
            schur.push(factor);
        }

        let rank = 2 * block_size;
        let full_size = blocks
            .checked_mul(block_size)
            .ok_or(PeriodicBlockError::CapacityOverflow)?;
        let mut basis = try_zeros(
            full_size
                .checked_mul(rank)
                .ok_or(PeriodicBlockError::CapacityOverflow)?,
        )?;
        for coordinate in 0..block_size {
            basis[coordinate * rank + coordinate] = 1.0;
            let row = (blocks - 1) * block_size + coordinate;
            basis[row * rank + block_size + coordinate] = 1.0;
        }
        solve_open(
            &schur,
            upper,
            &lower_multiplier,
            blocks,
            block_size,
            &mut basis,
            rank,
        )?;

        let mut closure_matrix = try_zeros(
            rank.checked_mul(rank)
                .ok_or(PeriodicBlockError::CapacityOverflow)?,
        )?;
        for index in 0..rank {
            closure_matrix[index * rank + index] = 1.0;
        }
        for row in 0..block_size {
            for column in 0..rank {
                let mut top = 0.0;
                let mut bottom = 0.0;
                for inner in 0..block_size {
                    top += corner[inner * block_size + row]
                        * basis[((blocks - 1) * block_size + inner) * rank + column];
                    bottom += corner[row * block_size + inner] * basis[inner * rank + column];
                }
                closure_matrix[row * rank + column] += top;
                closure_matrix[(block_size + row) * rank + column] += bottom;
            }
        }
        let closure = DenseLu::factor(closure_matrix, rank)?;
        sign *= closure.sign;
        log_abs_det += closure.log_abs_det;
        if !log_abs_det.is_finite() {
            return Err(PeriodicBlockError::NonFinite);
        }
        Ok(Self {
            blocks,
            block_size,
            schur,
            upper: upper.to_vec(),
            corner: corner.to_vec(),
            lower_multiplier,
            woodbury_basis: basis,
            closure,
            sign,
            log_abs_det,
        })
    }

    /// Sign of the determinant (`-1` or `1`).
    pub fn determinant_sign(&self) -> i8 {
        self.sign
    }

    /// Natural logarithm of the determinant's absolute value.
    pub fn log_abs_determinant(&self) -> f64 {
        self.log_abs_det
    }

    /// Signed log-absolute determinant after bordering by one vector.
    ///
    /// This returns the determinant of `[A q; q^T 0]` without materializing
    /// the bordered matrix. The factor must be nonsingular; callers using
    /// this to remove an approximate null mode should check the result under
    /// bead refinement.
    pub fn bordered_log_abs_determinant(
        &self,
        border: &[f64],
    ) -> Result<(i8, f64), PeriodicBlockError> {
        self.bordered_log_abs_determinant_many(border, 1)
    }

    /// Signed log-absolute determinant after bordering by several vectors.
    ///
    /// `border` is row-major `(blocks * block_size, rank)`. The returned
    /// value is the determinant of `[A Q; Q^T 0]`.
    pub fn bordered_log_abs_determinant_many(
        &self,
        border: &[f64],
        rank: usize,
    ) -> Result<(i8, f64), PeriodicBlockError> {
        if rank == 0 {
            return Err(PeriodicBlockError::InvalidDimensions);
        }
        let solution = self.solve_many(border, rank)?;
        let n = self
            .blocks
            .checked_mul(self.block_size)
            .ok_or(PeriodicBlockError::CapacityOverflow)?;
        let mut reduced = try_zeros(
            rank.checked_mul(rank)
                .ok_or(PeriodicBlockError::CapacityOverflow)?,
        )?;
        for row in 0..rank {
            for column in 0..rank {
                let mut value = 0.0;
                for inner in 0..n {
                    value -= border[inner * rank + row] * solution[inner * rank + column];
                }
                reduced[row * rank + column] = value;
            }
        }
        let reduced_factor = DenseLu::factor(reduced, rank)?;
        Ok((
            self.sign * reduced_factor.sign,
            self.log_abs_det + reduced_factor.log_abs_det,
        ))
    }

    /// Solve the factored periodic system for one right hand side.
    pub fn solve(&self, rhs: &[f64]) -> Result<Vec<f64>, PeriodicBlockError> {
        self.solve_many(rhs, 1)
    }

    /// Solve the factored periodic system for row-major right-hand sides.
    pub fn solve_many(&self, rhs: &[f64], columns: usize) -> Result<Vec<f64>, PeriodicBlockError> {
        let n = self
            .blocks
            .checked_mul(self.block_size)
            .ok_or(PeriodicBlockError::CapacityOverflow)?;
        if columns == 0 {
            return Err(PeriodicBlockError::InvalidDimensions);
        }
        let expected = n
            .checked_mul(columns)
            .ok_or(PeriodicBlockError::CapacityOverflow)?;
        if rhs.len() != expected {
            return Err(PeriodicBlockError::DimensionMismatch {
                name: "right hand side",
                expected,
                got: rhs.len(),
            });
        }
        let mut solution = rhs.to_vec();
        solve_open(
            &self.schur,
            &self.upper,
            &self.lower_multiplier,
            self.blocks,
            self.block_size,
            &mut solution,
            columns,
        )?;
        let rank = 2 * self.block_size;
        let mut reduced = try_zeros(
            rank.checked_mul(columns)
                .ok_or(PeriodicBlockError::CapacityOverflow)?,
        )?;
        for row in 0..self.block_size {
            for inner in 0..self.block_size {
                for column in 0..columns {
                    reduced[row * columns + column] += self.corner_transposed(row, inner)
                        * solution
                            [((self.blocks - 1) * self.block_size + inner) * columns + column];
                    reduced[(self.block_size + row) * columns + column] +=
                        self.corner(row, inner) * solution[inner * columns + column];
                }
            }
        }
        self.closure.solve_in_place(&mut reduced, columns)?;
        for row in 0..n {
            for rhs_column in 0..columns {
                let mut correction = 0.0;
                for column in 0..rank {
                    correction += self.woodbury_basis[row * rank + column]
                        * reduced[column * columns + rhs_column];
                }
                solution[row * columns + rhs_column] -= correction;
            }
        }
        if solution.iter().any(|value| !value.is_finite()) {
            return Err(PeriodicBlockError::NonFinite);
        }
        Ok(solution)
    }

    fn corner(&self, row: usize, column: usize) -> f64 {
        self.corner[row * self.block_size + column]
    }

    fn corner_transposed(&self, row: usize, column: usize) -> f64 {
        self.corner(column, row)
    }
}

fn check_len(
    name: &'static str,
    values: &[f64],
    count: usize,
    block_entries: usize,
) -> Result<(), PeriodicBlockError> {
    let expected = count
        .checked_mul(block_entries)
        .ok_or(PeriodicBlockError::CapacityOverflow)?;
    if values.len() != expected {
        return Err(PeriodicBlockError::DimensionMismatch {
            name,
            expected,
            got: values.len(),
        });
    }
    Ok(())
}

fn try_zeros(len: usize) -> Result<Vec<f64>, PeriodicBlockError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| PeriodicBlockError::CapacityOverflow)?;
    values.resize(len, 0.0);
    Ok(values)
}

fn transpose(matrix: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n * n];
    for row in 0..n {
        for column in 0..n {
            out[row * n + column] = matrix[column * n + row];
        }
    }
    out
}

fn subtract_product(
    target: &mut [f64],
    left: &[f64],
    right: &[f64],
    n: usize,
) -> Result<(), PeriodicBlockError> {
    for row in 0..n {
        for column in 0..n {
            let mut value = 0.0;
            for inner in 0..n {
                value += left[row * n + inner] * right[inner * n + column];
            }
            target[row * n + column] -= value;
        }
    }
    if target.iter().any(|value| !value.is_finite()) {
        return Err(PeriodicBlockError::NonFinite);
    }
    Ok(())
}

fn solve_open(
    schur: &[DenseLu],
    upper: &[f64],
    lower_multiplier: &[f64],
    blocks: usize,
    block_size: usize,
    rhs: &mut [f64],
    columns: usize,
) -> Result<(), PeriodicBlockError> {
    let block_entries = block_size * block_size;
    let block_rhs = block_size * columns;
    for block in 1..blocks {
        let multiplier = &lower_multiplier[(block - 1) * block_entries..block * block_entries];
        let (before, after) = rhs.split_at_mut(block * block_rhs);
        let previous = &before[(block - 1) * block_rhs..block * block_rhs];
        let current = &mut after[..block_rhs];
        for row in 0..block_size {
            for column in 0..columns {
                for inner in 0..block_size {
                    current[row * columns + column] -=
                        multiplier[row * block_size + inner] * previous[inner * columns + column];
                }
            }
        }
    }
    let last = (blocks - 1) * block_rhs;
    schur[blocks - 1].solve_in_place(&mut rhs[last..last + block_rhs], columns)?;
    for block in (0..blocks - 1).rev() {
        let off = block * block_rhs;
        let next = (block + 1) * block_rhs;
        let upper_block = &upper[block * block_entries..(block + 1) * block_entries];
        let (left, right) = rhs.split_at_mut(next);
        let current = &mut left[off..off + block_rhs];
        let following = &right[..block_rhs];
        for row in 0..block_size {
            for column in 0..columns {
                for inner in 0..block_size {
                    current[row * columns + column] -=
                        upper_block[row * block_size + inner] * following[inner * columns + column];
                }
            }
        }
        schur[block].solve_in_place(current, columns)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense_matrix(
        diagonal: &[f64],
        upper: &[f64],
        corner: &[f64],
        blocks: usize,
        d: usize,
    ) -> Vec<f64> {
        let n = blocks * d;
        let mut out = vec![0.0; n * n];
        for block in 0..blocks {
            for row in 0..d {
                for column in 0..d {
                    out[(block * d + row) * n + block * d + column] =
                        diagonal[(block * d + row) * d + column];
                }
            }
        }
        for block in 0..blocks - 1 {
            for row in 0..d {
                for column in 0..d {
                    let value = upper[(block * d + row) * d + column];
                    out[(block * d + row) * n + (block + 1) * d + column] = value;
                    out[((block + 1) * d + column) * n + block * d + row] = value;
                }
            }
        }
        for row in 0..d {
            for column in 0..d {
                let value = corner[row * d + column];
                out[((blocks - 1) * d + row) * n + column] = value;
                out[column * n + (blocks - 1) * d + row] = value;
            }
        }
        out
    }

    fn residual(matrix: &[f64], n: usize, x: &[f64], rhs: &[f64]) -> f64 {
        let mut largest = 0.0_f64;
        for row in 0..n {
            let mut value = -rhs[row];
            for column in 0..n {
                value += matrix[row * n + column] * x[column];
            }
            largest = largest.max(value.abs());
        }
        largest
    }

    #[test]
    fn scalar_ring_matches_its_fourier_product_and_solves() {
        for blocks in [3_usize, 4, 7, 16] {
            let diagonal = vec![3.0; blocks];
            let upper = vec![-0.4; blocks - 1];
            let corner = vec![-0.4];
            let factor =
                PeriodicBlockFactor::factor(&diagonal, &upper, &corner, blocks, 1).unwrap();
            let expected: f64 = (0..blocks)
                .map(|mode| {
                    let angle = 2.0 * std::f64::consts::PI * mode as f64 / blocks as f64;
                    (3.0 - 0.8 * angle.cos()).ln()
                })
                .sum();
            assert!((factor.log_abs_determinant() - expected).abs() < 2.0e-13);
            assert_eq!(factor.determinant_sign(), 1);
            let rhs: Vec<f64> = (0..blocks).map(|index| index as f64 - 0.5).collect();
            let solution = factor.solve(&rhs).unwrap();
            let dense = dense_matrix(&diagonal, &upper, &corner, blocks, 1);
            assert!(residual(&dense, blocks, &solution, &rhs) < 2.0e-13);
        }
    }

    #[test]
    fn bordered_factor_removes_an_exact_periodic_zero_mode() {
        for blocks in [3_usize, 8, 17] {
            let diagonal = vec![2.0; blocks];
            let upper = vec![-1.0; blocks - 1];
            let corner = vec![-1.0];
            let border = vec![(blocks as f64).sqrt().recip(); blocks];
            assert!(matches!(
                PeriodicBlockFactor::factor(&diagonal, &upper, &corner, blocks, 1),
                Err(PeriodicBlockError::Singular { .. })
            ));
            let factor =
                PeriodicBorderedFactor::factor(&diagonal, &upper, &corner, &border, blocks, 1, 1)
                    .unwrap();
            let dense = dense_matrix(&diagonal, &upper, &corner, blocks, 1);
            let bordered_n = blocks + 1;
            let mut bordered = vec![0.0; bordered_n * bordered_n];
            for row in 0..blocks {
                for column in 0..blocks {
                    bordered[row * bordered_n + column] = dense[row * blocks + column];
                }
                bordered[row * bordered_n + blocks] = border[row];
                bordered[blocks * bordered_n + row] = border[row];
            }
            let dense_factor = DenseLu::factor(bordered, bordered_n).unwrap();
            assert_eq!(factor.determinant_sign(), dense_factor.sign);
            assert!((factor.log_abs_determinant() - dense_factor.log_abs_det).abs() < 2.0e-12);
        }
    }

    #[test]
    fn coupled_indefinite_blocks_match_a_dense_factor() {
        let blocks = 5;
        let d = 2;
        let mut diagonal = Vec::new();
        for block in 0..blocks {
            diagonal.extend_from_slice(&[
                if block == 2 {
                    -1.2
                } else {
                    2.4 + block as f64 * 0.1
                },
                0.17,
                0.17,
                1.8 + block as f64 * 0.05,
            ]);
        }
        let mut upper = Vec::new();
        for block in 0..blocks - 1 {
            upper.extend_from_slice(&[-0.31, 0.02 * block as f64, -0.01, -0.27]);
        }
        let corner = [-0.23, 0.04, -0.03, -0.19];
        let factor = PeriodicBlockFactor::factor(&diagonal, &upper, &corner, blocks, d).unwrap();
        let dense = dense_matrix(&diagonal, &upper, &corner, blocks, d);
        let dense_factor = DenseLu::factor(dense.clone(), blocks * d).unwrap();
        assert_eq!(factor.determinant_sign(), dense_factor.sign);
        assert!((factor.log_abs_determinant() - dense_factor.log_abs_det).abs() < 5.0e-13);
        let rhs: Vec<f64> = (0..blocks * d)
            .map(|index| (index as f64 + 0.25).sin())
            .collect();
        let solution = factor.solve(&rhs).unwrap();
        assert!(residual(&dense, blocks * d, &solution, &rhs) < 5.0e-13);

        let border: Vec<f64> = (0..blocks * d)
            .map(|index| (0.37 * index as f64 + 0.2).cos())
            .collect();
        let bordered_n = blocks * d + 1;
        let mut bordered = vec![0.0; bordered_n * bordered_n];
        for row in 0..blocks * d {
            for column in 0..blocks * d {
                bordered[row * bordered_n + column] = dense[row * blocks * d + column];
            }
            bordered[row * bordered_n + blocks * d] = border[row];
            bordered[blocks * d * bordered_n + row] = border[row];
        }
        let bordered_dense = DenseLu::factor(bordered, bordered_n).unwrap();
        let (bordered_sign, bordered_logabs) =
            factor.bordered_log_abs_determinant(&border).unwrap();
        assert_eq!(bordered_sign, bordered_dense.sign);
        assert!((bordered_logabs - bordered_dense.log_abs_det).abs() < 5.0e-13);

        let rank = 2;
        let borders: Vec<f64> = (0..blocks * d)
            .flat_map(|index| {
                [
                    (0.37 * index as f64 + 0.2).cos(),
                    (0.23 * index as f64 - 0.1).sin(),
                ]
            })
            .collect();
        let bordered_n = blocks * d + rank;
        let mut bordered = vec![0.0; bordered_n * bordered_n];
        for row in 0..blocks * d {
            for column in 0..blocks * d {
                bordered[row * bordered_n + column] = dense[row * blocks * d + column];
            }
            for column in 0..rank {
                bordered[row * bordered_n + blocks * d + column] = borders[row * rank + column];
                bordered[(blocks * d + column) * bordered_n + row] = borders[row * rank + column];
            }
        }
        let bordered_dense = DenseLu::factor(bordered, bordered_n).unwrap();
        let (bordered_sign, bordered_logabs) = factor
            .bordered_log_abs_determinant_many(&borders, rank)
            .unwrap();
        assert_eq!(bordered_sign, bordered_dense.sign);
        assert!((bordered_logabs - bordered_dense.log_abs_det).abs() < 5.0e-13);
    }

    #[test]
    fn invalid_shapes_and_singular_pivots_fail_closed() {
        assert!(matches!(
            PeriodicBlockFactor::factor(&[], &[], &[], 2, 1),
            Err(PeriodicBlockError::InvalidDimensions)
        ));
        assert!(matches!(
            PeriodicBlockFactor::factor(&[1.0; 3], &[0.0; 2], &[0.0, 1.0], 3, 1),
            Err(PeriodicBlockError::DimensionMismatch { name: "corner", .. })
        ));
        assert!(matches!(
            PeriodicBlockFactor::factor(&[0.0; 3], &[0.0; 2], &[1.0], 3, 1),
            Err(PeriodicBlockError::Singular { .. })
        ));
        assert!(matches!(
            PeriodicBlockFactor::factor(&[f64::NAN; 3], &[0.0; 2], &[1.0], 3, 1),
            Err(PeriodicBlockError::NonFinite)
        ));
    }
}
