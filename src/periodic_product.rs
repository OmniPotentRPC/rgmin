//! Extended-precision products of short real transfer matrices.
//!
//! A long fundamental matrix can have reciprocal multipliers whose moduli
//! differ by many orders of magnitude.  Multiplying the short factors in
//! binary64 destroys multipliers near one by cancellation.  This module
//! promotes the supplied binary64 factors before the first product and takes
//! the eigenvalues with a shifted QR iteration at the same precision.

use rug::{Complex, Float, ops::Pow};
use thiserror::Error;

/// Failure of an extended-precision periodic product.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum PeriodicProductError {
    /// A factor did not have `width * width` finite entries.
    #[error("periodic factor {factor} is not a finite {width} by {width} matrix")]
    InvalidFactor {
        /// Zero-based factor index.
        factor: usize,
        /// Requested matrix width.
        width: usize,
    },
    /// The matrix width was zero or odd, or the precision or derived work
    /// limit was unusable.
    #[error("periodic product configuration was not representable")]
    InvalidConfiguration,
    /// Shifted QR did not isolate every multiplier.
    #[error("periodic product QR did not converge after {iterations} iterations")]
    NoConvergence {
        /// Number of QR steps attempted.
        iterations: usize,
    },
    /// The product had no factors or produced a zero/non-finite multiplier.
    #[error("periodic product has an empty or non-finite spectrum")]
    InvalidSpectrum,
    /// The returned multipliers failed the determinant identity.
    #[error("periodic product eigenvalues failed a-posteriori verification")]
    VerificationFailed,
    /// The returned multipliers could not be matched to reciprocal partners.
    #[error("periodic product eigenvalues failed reciprocal-pair verification")]
    ReciprocityFailed,
}

/// One multiplier of a periodic matrix product.
#[derive(Clone, Debug, PartialEq)]
pub struct PeriodicMultiplier {
    /// Natural logarithm of the multiplier modulus.
    pub log_modulus: f64,
    /// Principal argument in radians.
    pub argument: f64,
}

/// Spectrum of a product `factors[last] ... factors[0]`.
#[derive(Clone, Debug, PartialEq)]
pub struct PeriodicProductSpectrum {
    /// Multipliers, in the diagonal order of the converged triangular form.
    pub multipliers: Vec<PeriodicMultiplier>,
    /// Binary working precision used for both product and eigensolve.
    pub precision_bits: u32,
    /// Number of shifted QR steps.
    pub qr_iterations: usize,
    /// Relative defect in `product(eigenvalues) = det(product matrix)`.
    pub verification_residual: f64,
    /// Largest defect `|lambda * lambda' - 1|` in reciprocal matching.
    pub reciprocity_residual: f64,
}

trait ComplexOps {
    fn add(&self, other: &Self) -> Self;
    fn sub(&self, other: &Self) -> Self;
    fn neg(&self) -> Self;
    fn mul(&self, other: &Self) -> Self;
    fn div(&self, other: &Self) -> Self;
    fn scale(&self, value: &Float) -> Self;
    fn div_real(&self, value: &Float) -> Self;
    fn conjugate(&self) -> Self;
    fn abs2(&self) -> Float;
    fn modulus(&self) -> Float;
    fn square_root(&self) -> Self;
}

impl ComplexOps for Complex {
    fn add(&self, other: &Self) -> Self {
        let mut value = self.clone();
        value += other;
        value
    }

    fn sub(&self, other: &Self) -> Self {
        let mut value = self.clone();
        value -= other;
        value
    }

    fn neg(&self) -> Self {
        -self.clone()
    }

    fn mul(&self, other: &Self) -> Self {
        let mut value = self.clone();
        value *= other;
        value
    }

    fn div(&self, other: &Self) -> Self {
        let mut value = self.clone();
        value /= other;
        value
    }

    fn scale(&self, value: &Float) -> Self {
        let mut result = self.clone();
        result *= value;
        result
    }

    fn div_real(&self, value: &Float) -> Self {
        let mut result = self.clone();
        result /= value;
        result
    }

    fn conjugate(&self) -> Self {
        let mut value = self.clone();
        value.conj_mut();
        value
    }

    fn abs2(&self) -> Float {
        let mut value = self.real().clone();
        value.square_mut();
        let mut term = self.imag().clone();
        term.square_mut();
        value += term;
        value
    }

    fn modulus(&self) -> Float {
        let mut value = self.abs2();
        value.sqrt_mut();
        value
    }

    fn square_root(&self) -> Self {
        self.clone().sqrt()
    }
}

fn complex(precision: u32, real: f64) -> Complex {
    Complex::with_val(precision, (real, 0))
}

fn identity(width: usize, precision: u32) -> Vec<Complex> {
    let mut matrix = vec![complex(precision, 0.0); width * width];
    for diagonal in 0..width {
        matrix[diagonal * width + diagonal] = complex(precision, 1.0);
    }
    matrix
}

fn multiply(left: &[Complex], right: &[Complex], width: usize) -> Vec<Complex> {
    let precision = left[0].prec().0;
    let mut product = vec![complex(precision, 0.0); width * width];
    for row in 0..width {
        for column in 0..width {
            let mut value = complex(precision, 0.0);
            for inner in 0..width {
                value = value.add(&left[row * width + inner].mul(&right[inner * width + column]));
            }
            product[row * width + column] = value;
        }
    }
    product
}

fn determinant(matrix: &[Complex], width: usize) -> Result<Complex, PeriodicProductError> {
    let precision = matrix[0].prec().0;
    let mut work = matrix.to_vec();
    let mut result = complex(precision, 1.0);
    for column in 0..width {
        let pivot = (column..width)
            .max_by(|left, right| {
                work[*left * width + column]
                    .abs2()
                    .partial_cmp(&work[*right * width + column].abs2())
                    .expect("finite multiprecision pivot")
            })
            .ok_or(PeriodicProductError::InvalidSpectrum)?;
        if work[pivot * width + column].abs2() == 0 {
            return Err(PeriodicProductError::InvalidSpectrum);
        }
        if pivot != column {
            for entry in 0..width {
                work.swap(column * width + entry, pivot * width + entry);
            }
            result = result.neg();
        }
        let diagonal = work[column * width + column].clone();
        result = result.mul(&diagonal);
        for row in column + 1..width {
            let ratio = work[row * width + column].div(&diagonal);
            for entry in column + 1..width {
                let index = row * width + entry;
                work[index] = work[index].sub(&ratio.mul(&work[column * width + entry]));
            }
        }
    }
    Ok(result)
}

fn householder_vector(values: &[Complex]) -> Option<(Vec<Complex>, Float)> {
    let precision = values[0].prec().0;
    let mut norm2 = Float::with_val(precision, 0);
    for value in values {
        norm2 += value.abs2();
    }
    if norm2 == 0 {
        return None;
    }
    let mut norm = norm2;
    norm.sqrt_mut();
    let first_abs = values[0].modulus();
    let phase = if first_abs == 0 {
        complex(precision, 1.0)
    } else {
        values[0].div_real(&first_abs)
    };
    let alpha = phase.scale(&norm).neg();
    let mut vector = values.to_vec();
    vector[0] = vector[0].sub(&alpha);
    let mut denominator = Float::with_val(precision, 0);
    for value in &vector {
        denominator += value.abs2();
    }
    if denominator == 0 {
        None
    } else {
        Some((vector, denominator))
    }
}

fn hessenberg(matrix: &mut [Complex], width: usize) {
    if width < 3 {
        return;
    }
    let precision = matrix[0].prec().0;
    for column in 0..width - 2 {
        let start = column + 1;
        let values: Vec<_> = (start..width)
            .map(|row| matrix[row * width + column].clone())
            .collect();
        let Some((vector, denominator)) = householder_vector(&values) else {
            continue;
        };
        let two = Float::with_val(precision, 2);
        for target_column in column..width {
            let mut dot = complex(precision, 0.0);
            for (offset, value) in vector.iter().enumerate() {
                dot = dot.add(
                    &value
                        .conjugate()
                        .mul(&matrix[(start + offset) * width + target_column]),
                );
            }
            let coefficient = dot.scale(&two).div_real(&denominator);
            for (offset, value) in vector.iter().enumerate() {
                let index = (start + offset) * width + target_column;
                matrix[index] = matrix[index].sub(&value.mul(&coefficient));
            }
        }
        for target_row in 0..width {
            let mut dot = complex(precision, 0.0);
            for (offset, value) in vector.iter().enumerate() {
                dot = dot.add(&matrix[target_row * width + start + offset].mul(value));
            }
            let coefficient = dot.scale(&two).div_real(&denominator);
            for (offset, value) in vector.iter().enumerate() {
                let index = target_row * width + start + offset;
                matrix[index] = matrix[index].sub(&coefficient.mul(&value.conjugate()));
            }
        }
        for row in start + 1..width {
            matrix[row * width + column] = complex(precision, 0.0);
        }
    }
}

fn qr_decompose(matrix: &[Complex], width: usize) -> (Vec<Complex>, Vec<Complex>) {
    let precision = matrix[0].prec().0;
    let mut q = identity(width, precision);
    let mut r = matrix.to_vec();
    for column in 0..width {
        let values: Vec<_> = (column..width)
            .map(|row| r[row * width + column].clone())
            .collect();
        let Some((vector, denominator)) = householder_vector(&values) else {
            continue;
        };
        let two = Float::with_val(precision, 2);
        for target_column in column..width {
            let mut dot = complex(precision, 0.0);
            for (offset, value) in vector.iter().enumerate() {
                dot = dot.add(
                    &value
                        .conjugate()
                        .mul(&r[(column + offset) * width + target_column]),
                );
            }
            let coefficient = dot.scale(&two).div_real(&denominator);
            for (offset, value) in vector.iter().enumerate() {
                let index = (column + offset) * width + target_column;
                r[index] = r[index].sub(&value.mul(&coefficient));
            }
        }
        for target_row in 0..width {
            let mut dot = complex(precision, 0.0);
            for (offset, value) in vector.iter().enumerate() {
                dot = dot.add(&q[target_row * width + column + offset].mul(value));
            }
            let coefficient = dot.scale(&two).div_real(&denominator);
            for (offset, value) in vector.iter().enumerate() {
                let index = target_row * width + column + offset;
                q[index] = q[index].sub(&coefficient.mul(&value.conjugate()));
            }
        }
    }
    (q, r)
}

fn wilkinson_shift(matrix: &[Complex], width: usize, active: usize) -> Complex {
    let precision = matrix[0].prec().0;
    if active == 1 {
        return matrix[0].clone();
    }
    let a = &matrix[(active - 2) * width + active - 2];
    let b = &matrix[(active - 2) * width + active - 1];
    let c = &matrix[(active - 1) * width + active - 2];
    let d = &matrix[(active - 1) * width + active - 1];
    let half = Float::with_val(precision, 0.5);
    let midpoint = a.add(d).scale(&half);
    let delta = a.sub(d).scale(&half);
    let root = delta.mul(&delta).add(&b.mul(c)).square_root();
    let first = midpoint.add(&root);
    let second = midpoint.sub(&root);
    if first.sub(d).abs2() <= second.sub(d).abs2() {
        first
    } else {
        second
    }
}

/// Stalled iterations between ad-hoc shifts, as in LAPACK's `zlahqr`.
const EXCEPTIONAL_SHIFT_PERIOD: usize = 10;

/// Shift that breaks a stall of the Wilkinson shift, such as the zero
/// shift of a unitary Hessenberg matrix whose trailing 2 by 2 block is zero.
///
/// The shift moves off the diagonal by three quarters of a subdiagonal
/// modulus, alternating between the top and the bottom of the block so that
/// repeated stalls do not retry the same value.
fn exceptional_shift(
    matrix: &[Complex],
    width: usize,
    low: usize,
    high: usize,
    stalled: usize,
) -> Complex {
    let precision = matrix[0].prec().0;
    let top = (stalled / EXCEPTIONAL_SHIFT_PERIOD) % 2 == 1;
    let (anchor, coupling) = if top {
        (low, matrix[(low + 1) * width + low].modulus())
    } else {
        (high - 1, matrix[(high - 1) * width + high - 2].modulus())
    };
    let mut offset = coupling;
    offset *= Float::with_val(precision, 0.75);
    let mut shift = matrix[anchor * width + anchor].clone();
    *shift.mut_real() += &offset;
    shift
}

fn iteration_limit(precision: u32, width: usize) -> Result<usize, PeriodicProductError> {
    usize::try_from(precision)
        .ok()
        .and_then(|bits| bits.checked_mul(width))
        .and_then(|work| work.checked_mul(width))
        .ok_or(PeriodicProductError::InvalidConfiguration)
}

fn eigenvalues_with_limit_unchecked(
    mut matrix: Vec<Complex>,
    width: usize,
    precision: u32,
    limit: usize,
) -> Result<(Vec<Complex>, usize), PeriodicProductError> {
    hessenberg(&mut matrix, width);
    let mut high = width;
    let mut iterations = 0usize;
    let mut block = (0usize, width);
    let mut stalled = 0usize;
    let exponent =
        i32::try_from(precision).map_err(|_| PeriodicProductError::InvalidConfiguration)?;
    let tolerance = Float::with_val(precision, 2).pow(1 - exponent);
    while high > 0 {
        let low = active_block_start(&mut matrix, width, high, &tolerance);
        if high - low == 1 {
            high -= 1;
            continue;
        }
        if iterations == limit {
            return Err(PeriodicProductError::NoConvergence { iterations });
        }
        if block != (low, high) {
            block = (low, high);
            stalled = 0;
        }
        let size = high - low;
        let shift = if stalled > 0 && stalled % EXCEPTIONAL_SHIFT_PERIOD == 0 {
            exceptional_shift(&matrix, width, low, high, stalled)
        } else {
            wilkinson_shift(&matrix, width, high)
        };
        let mut shifted = vec![complex(precision, 0.0); size * size];
        for row in 0..size {
            for column in 0..size {
                let mut value = matrix[(low + row) * width + low + column].clone();
                if row == column {
                    value = value.sub(&shift);
                }
                shifted[row * size + column] = value;
            }
        }
        let (q, r) = qr_decompose(&shifted, size);
        let mut next = multiply(&r, &q, size);
        for diagonal in 0..size {
            let index = diagonal * size + diagonal;
            next[index] = next[index].add(&shift);
        }
        for row in 0..size {
            for column in 0..size {
                matrix[(low + row) * width + low + column] = if column + 1 < row {
                    complex(precision, 0.0)
                } else {
                    next[row * size + column].clone()
                };
            }
        }
        iterations += 1;
        stalled += 1;
    }
    let values = (0..width)
        .map(|diagonal| matrix[diagonal * width + diagonal].clone())
        .collect();
    Ok((values, iterations))
}

/// Zero every negligible subdiagonal entry among the leading `high` rows and
/// return the first row of the lowest unreduced block ending at `high`.
fn active_block_start(
    matrix: &mut [Complex],
    width: usize,
    high: usize,
    tolerance: &Float,
) -> usize {
    let precision = matrix[0].prec().0;
    for row in (1..high).rev() {
        let index = row * width + row - 1;
        let subdiagonal = matrix[index].modulus();
        let mut scale = matrix[(row - 1) * width + row - 1].modulus();
        scale += matrix[row * width + row].modulus();
        if scale == 0 {
            scale = matrix[(row - 1) * width + row].modulus();
        }
        if subdiagonal == 0
            || (scale > 0 && subdiagonal <= Float::with_val(precision, tolerance * &scale))
        {
            matrix[index] = complex(precision, 0.0);
        }
    }
    (1..high)
        .rev()
        .find(|row| matrix[row * width + row - 1].abs2() == 0)
        .unwrap_or(0)
}

fn eigenvalues_with_limit(
    matrix: Vec<Complex>,
    width: usize,
    precision: u32,
    limit: usize,
) -> Result<(Vec<Complex>, usize), PeriodicProductError> {
    if limit != iteration_limit(precision, width)? {
        return Err(PeriodicProductError::InvalidConfiguration);
    }
    eigenvalues_with_limit_unchecked(matrix, width, precision, limit)
}

fn eigenvalues(
    matrix: Vec<Complex>,
    width: usize,
    precision: u32,
) -> Result<(Vec<Complex>, usize), PeriodicProductError> {
    eigenvalues_with_limit(matrix, width, precision, iteration_limit(precision, width)?)
}

fn verify_reciprocity(
    eigenvalues: &[Complex],
    bound: &Float,
) -> Result<Float, PeriodicProductError> {
    let precision = eigenvalues[0].prec().0;
    let one = complex(precision, 1.0);
    let mut unused = vec![true; eigenvalues.len()];
    let mut worst = Float::with_val(precision, 0);
    while let Some(left) = unused.iter().position(|present| *present) {
        let mut best: Option<(usize, Float)> = None;
        for (right, candidate) in eigenvalues.iter().enumerate() {
            if right == left || !unused[right] {
                continue;
            }
            let defect = eigenvalues[left].mul(candidate).sub(&one).modulus();
            if best
                .as_ref()
                .is_none_or(|(_, old_defect)| defect < *old_defect)
            {
                best = Some((right, defect));
            }
        }
        let (right, defect) = best.ok_or(PeriodicProductError::ReciprocityFailed)?;
        if !defect.is_finite() || defect > *bound {
            return Err(PeriodicProductError::ReciprocityFailed);
        }
        if defect > worst {
            worst = defect;
        }
        unused[left] = false;
        unused[right] = false;
    }
    Ok(worst)
}

/// Multiply short transfer factors and diagonalize their product.
///
/// Every factor is promoted before it participates in a product.  Factors are
/// applied in slice order, so the returned matrix spectrum is that of
/// `factors[last] ... factors[0]`.
pub fn periodic_product_spectrum(
    factors: &[Vec<f64>],
    width: usize,
    precision_bits: u32,
) -> Result<PeriodicProductSpectrum, PeriodicProductError> {
    if factors.is_empty()
        || width == 0
        || width % 2 != 0
        || precision_bits < f64::MANTISSA_DIGITS
        || i32::try_from(precision_bits).is_err()
    {
        return Err(PeriodicProductError::InvalidConfiguration);
    }
    let entries = width
        .checked_mul(width)
        .ok_or(PeriodicProductError::InvalidConfiguration)?;
    let mut product = identity(width, precision_bits);
    for (factor_index, factor) in factors.iter().enumerate() {
        if factor.len() != entries || factor.iter().any(|value| !value.is_finite()) {
            return Err(PeriodicProductError::InvalidFactor {
                factor: factor_index,
                width,
            });
        }
        let promoted: Vec<_> = factor
            .iter()
            .map(|value| complex(precision_bits, *value))
            .collect();
        product = multiply(&promoted, &product, width);
    }
    let product_determinant = determinant(&product, width)?;
    let (eigenvalues, qr_iterations) = eigenvalues(product, width, precision_bits)?;
    let mut eigenvalue_product = complex(precision_bits, 1.0);
    for value in &eigenvalues {
        eigenvalue_product = eigenvalue_product.mul(value);
    }
    let determinant_modulus = product_determinant.modulus();
    if determinant_modulus == 0 || !determinant_modulus.is_finite() {
        return Err(PeriodicProductError::InvalidSpectrum);
    }
    let verification_residual =
        eigenvalue_product.sub(&product_determinant).modulus() / determinant_modulus;
    let operations = width
        .checked_mul(width)
        .and_then(|work| work.checked_mul(width))
        .and_then(|work| work.checked_mul(qr_iterations.checked_add(width)?))
        .ok_or(PeriodicProductError::InvalidConfiguration)?;
    let mut verification_bound = Float::with_val(precision_bits, f64::EPSILON);
    verification_bound *= operations;
    if !verification_residual.is_finite() || verification_residual > verification_bound {
        return Err(PeriodicProductError::VerificationFailed);
    }
    let verification_residual = verification_residual.to_f64();
    let product_operations = factors
        .len()
        .checked_mul(width)
        .and_then(|work| work.checked_mul(width))
        .and_then(|work| work.checked_mul(width))
        .ok_or(PeriodicProductError::InvalidConfiguration)?;
    let reciprocity_bound =
        width as f64 * f64::EPSILON.sqrt() + product_operations as f64 * f64::EPSILON;
    let reciprocity_residual = verify_reciprocity(
        &eigenvalues,
        &Float::with_val(precision_bits, reciprocity_bound),
    )?
    .to_f64();
    let mut multipliers = Vec::with_capacity(width);
    for value in eigenvalues {
        let mut modulus = value.modulus();
        if modulus == 0 || !modulus.is_finite() {
            return Err(PeriodicProductError::InvalidSpectrum);
        }
        modulus.ln_mut();
        let argument = Float::with_val(precision_bits, value.imag().atan2_ref(value.real()));
        let log_modulus = modulus.to_f64();
        let argument = argument.to_f64();
        if !log_modulus.is_finite() || !argument.is_finite() {
            return Err(PeriodicProductError::InvalidSpectrum);
        }
        multipliers.push(PeriodicMultiplier {
            log_modulus,
            argument,
        });
    }
    Ok(PeriodicProductSpectrum {
        multipliers,
        precision_bits,
        qr_iterations,
        verification_residual,
        reciprocity_residual,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn multiply_f64(left: &[f64], right: &[f64], width: usize) -> Vec<f64> {
        let mut product = vec![0.0; width * width];
        for row in 0..width {
            for column in 0..width {
                for inner in 0..width {
                    product[row * width + column] +=
                        left[row * width + inner] * right[inner * width + column];
                }
            }
        }
        product
    }

    fn transpose(matrix: &[f64], width: usize) -> Vec<f64> {
        let mut result = vec![0.0; width * width];
        for row in 0..width {
            for column in 0..width {
                result[row * width + column] = matrix[column * width + row];
            }
        }
        result
    }

    fn determinant(mut matrix: Vec<f64>, width: usize) -> f64 {
        let mut result = 1.0;
        for column in 0..width {
            let pivot = (column..width)
                .max_by(|left, right| {
                    matrix[*left * width + column]
                        .abs()
                        .total_cmp(&matrix[*right * width + column].abs())
                })
                .unwrap();
            if pivot != column {
                for entry in 0..width {
                    matrix.swap(column * width + entry, pivot * width + entry);
                }
                result = -result;
            }
            let diagonal = matrix[column * width + column];
            result *= diagonal;
            if diagonal == 0.0 {
                return 0.0;
            }
            for row in column + 1..width {
                let ratio = matrix[row * width + column] / diagonal;
                for entry in column + 1..width {
                    matrix[row * width + entry] -= ratio * matrix[column * width + entry];
                }
            }
        }
        result
    }

    #[test]
    fn qr_exhaustion_is_typed_and_iteration_cap_is_derived() {
        let precision = 53;
        let width = 2;
        let expected = 212;
        assert_eq!(iteration_limit(precision, width), Ok(expected));

        let matrix = vec![
            complex(precision, 0.0),
            complex(precision, -1.0),
            complex(precision, 1.0),
            complex(precision, 0.0),
        ];
        let exhausted = eigenvalues_with_limit_unchecked(matrix, width, precision, 0);
        assert_eq!(
            exhausted,
            Err(PeriodicProductError::NoConvergence { iterations: 0 })
        );
    }

    #[test]
    fn production_eigensolver_rejects_a_non_derived_iteration_cap() {
        let precision = 53;
        let width = 2;
        let matrix = vec![
            complex(precision, 0.0),
            complex(precision, -1.0),
            complex(precision, 1.0),
            complex(precision, 0.0),
        ];
        assert_eq!(
            eigenvalues_with_limit(matrix.clone(), width, precision, usize::MAX),
            Err(PeriodicProductError::InvalidConfiguration)
        );
        assert!(eigenvalues(matrix, width, precision).is_ok());
    }

    fn block_sorted_log_moduli(factor: Vec<f64>, width: usize) -> Vec<f64> {
        let got = periodic_product_spectrum(&[factor], width, 128).expect("verified spectrum");
        let mut exponents: Vec<_> = got
            .multipliers
            .iter()
            .map(|value| value.log_modulus)
            .collect();
        exponents.sort_by(f64::total_cmp);
        exponents
    }

    #[test]
    fn interior_deflation_splits_the_active_block_in_either_order() {
        let big = 2.0_f64.powi(133);
        let small = 2.0_f64.powi(-133);
        #[rustfmt::skip]
        let scales_first = vec![
            big, 0.0, 0.0, 0.0,
            0.0, small, 0.0, 0.0,
            0.0, 0.0, 2.0, 1.0,
            0.0, 0.0, 1.0, 1.0,
        ];
        #[rustfmt::skip]
        let scales_last = vec![
            2.0, 1.0, 0.0, 0.0,
            1.0, 1.0, 0.0, 0.0,
            0.0, 0.0, big, 0.0,
            0.0, 0.0, 0.0, small,
        ];
        let golden = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let scale = 133.0 * std::f64::consts::LN_2;
        let expected = [-scale, -2.0 * golden.ln(), 2.0 * golden.ln(), scale];
        for factor in [scales_first, scales_last] {
            let exponents = block_sorted_log_moduli(factor, 4);
            for (value, wanted) in exponents.iter().zip(expected) {
                assert!((value - wanted).abs() < 1e-12, "{exponents:?}");
            }
        }
    }

    #[test]
    fn cyclic_orthogonal_matrix_with_a_zero_trailing_block_converges() {
        for width in [4usize, 6] {
            let mut cyclic = vec![0.0; width * width];
            for row in 1..width {
                cyclic[row * width + row - 1] = 1.0;
            }
            cyclic[width - 1] = -1.0;
            let got = periodic_product_spectrum(&[cyclic], width, 128).expect("cyclic spectrum");
            assert!(got.qr_iterations < 200, "{}", got.qr_iterations);
            let mut arguments: Vec<_> =
                got.multipliers.iter().map(|value| value.argument).collect();
            arguments.sort_by(f64::total_cmp);
            let step = std::f64::consts::PI / width as f64;
            for (index, value) in arguments.iter().enumerate() {
                let wanted = (2.0 * index as f64 + 1.0 - width as f64) * step;
                assert!((value - wanted).abs() < 1e-12, "{arguments:?}");
            }
            assert!(
                got.multipliers
                    .iter()
                    .all(|value| value.log_modulus.abs() < 1e-12)
            );
        }
    }

    #[test]
    fn zero_diagonal_rotation_keeps_plus_and_minus_i() {
        let got = periodic_product_spectrum(&[vec![0.0, -1.0, 1.0, 0.0]], 2, 128)
            .expect("quarter-turn spectrum");
        let mut arguments: Vec<_> = got.multipliers.iter().map(|value| value.argument).collect();
        arguments.sort_by(f64::total_cmp);
        assert_eq!(arguments.len(), 2);
        assert!((arguments[0] + std::f64::consts::FRAC_PI_2).abs() < 1e-14);
        assert!((arguments[1] - std::f64::consts::FRAC_PI_2).abs() < 1e-14);
        assert!(
            got.multipliers
                .iter()
                .all(|value| value.log_modulus.abs() < 1e-14)
        );
    }

    #[test]
    fn zero_diagonal_reflection_is_not_a_reciprocal_spectrum() {
        assert_eq!(
            periodic_product_spectrum(&[vec![0.0, 1.0, 1.0, 0.0]], 2, 128),
            Err(PeriodicProductError::ReciprocityFailed)
        );
    }

    #[test]
    fn empty_and_singular_products_fail_closed() {
        assert_eq!(
            periodic_product_spectrum(&[], 2, 128),
            Err(PeriodicProductError::InvalidConfiguration)
        );
        assert_eq!(
            periodic_product_spectrum(&[vec![0.0; 4]], 2, 128),
            Err(PeriodicProductError::InvalidSpectrum)
        );
        assert_eq!(
            periodic_product_spectrum(&[vec![2.0, 0.0, 0.0, 3.0]], 2, 128),
            Err(PeriodicProductError::ReciprocityFailed)
        );
        assert_eq!(
            periodic_product_spectrum(&[vec![1.0, 0.0, 0.0, -1.0]], 2, 128),
            Err(PeriodicProductError::ReciprocityFailed)
        );
        assert_eq!(
            periodic_product_spectrum(&[vec![1.0]], 1, 128),
            Err(PeriodicProductError::InvalidConfiguration)
        );
    }

    #[test]
    fn non_finite_factors_never_return_a_spectrum() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for position in 0..4 {
                let mut factor = vec![1.0, 0.0, 0.0, 1.0];
                factor[position] = value;
                let got = periodic_product_spectrum(&[factor], 2, 128);
                assert!(got.is_err(), "value={value} position={position}");
            }
        }
    }

    #[test]
    fn argument_is_formed_before_large_components_are_downcast() {
        let steps = 100;
        let radial = 10.0_f64.exp();
        let angle = 0.003_f64;
        let (sine, cosine) = angle.sin_cos();
        let inverse = radial.recip();
        let factor = vec![
            radial * cosine,
            -radial * sine,
            0.0,
            0.0,
            radial * sine,
            radial * cosine,
            0.0,
            0.0,
            0.0,
            0.0,
            inverse * cosine,
            -inverse * sine,
            0.0,
            0.0,
            inverse * sine,
            inverse * cosine,
        ];
        let got = periodic_product_spectrum(&vec![factor; steps], 4, 256)
            .expect("large complex spectrum");
        for value in &got.multipliers {
            assert!((value.log_modulus.abs() - 1000.0).abs() < 1e-10);
            assert!((value.argument.abs() - 0.3).abs() < 1e-12);
        }
    }

    #[test]
    fn rotated_hyperbolic_product_keeps_the_small_multiplier() {
        let factors = [
            [202.8567368308649, 0.0, 0.0, 0.004929587331545052],
            [
                176.33067407451037,
                68.39026073811857,
                68.39026073811857,
                26.530992343686098,
            ],
            [
                110.62728465324432,
                101.00811455683828,
                101.0081145568383,
                92.23438176495213,
            ],
            [
                40.11357283045651,
                80.7923728154444,
                80.7923728154444,
                162.74809358773993,
            ],
            [
                1.6726439708272915,
                18.31713966189123,
                18.317139661891225,
                201.18902244736915,
            ],
            [
                15.41152113122111,
                -53.739109363113634,
                -53.739109363113634,
                187.4501452869753,
            ],
            [
                74.14390762124799,
                -97.68642493947162,
                -97.68642493947164,
                128.71775879694846,
            ],
            [
                147.14907206493285,
                -90.5375975018818,
                -90.5375975018818,
                55.712594353263576,
            ],
            [
                196.2407227079949,
                -36.03191333663672,
                -36.03191333663672,
                6.620943710201518,
            ],
            [
                195.74083925638703,
                37.32072728193074,
                37.32072728193074,
                7.120827161809404,
            ],
            [
                145.91089218924168,
                91.15228070989153,
                91.15228070989153,
                56.95077422895477,
            ],
            [
                72.81507725635659,
                97.30545943956604,
                97.30545943956605,
                130.04658916183985,
            ],
            [
                14.687102118197398,
                52.56176406782204,
                52.56176406782204,
                188.17456429999905,
            ],
            [
                1.9315530067907496,
                -19.67503912866775,
                -19.675039128667752,
                200.93011341140573,
            ],
            [
                41.22038420874023,
                -81.62055964441306,
                -81.62055964441306,
                161.64128220945616,
            ],
            [
                112.00306642389428,
                -100.8733949579564,
                -100.87339495795639,
                90.85859999430215,
            ],
        ];
        let factors: Vec<Vec<f64>> = factors.into_iter().map(Vec::from).collect();
        assert!(matches!(
            periodic_product_spectrum(&factors, 2, 256),
            Err(PeriodicProductError::VerificationFailed)
        ));
        let got = periodic_product_spectrum(&factors, 2, 600).expect("verified spectrum");
        let mut exponents: Vec<_> = got
            .multipliers
            .iter()
            .map(|value| value.log_modulus)
            .collect();
        exponents.sort_by(f64::total_cmp);
        assert!(
            (exponents[0] + 83.651_865_655_68).abs() < 1e-9,
            "{exponents:?}"
        );
        assert!(
            (exponents[1] - 83.651_865_655_68).abs() < 1e-9,
            "{exponents:?}"
        );
    }

    #[test]
    fn short_hyperbolic_factors_keep_a_unit_pair_beside_exp_80() {
        let steps = 16;
        let argument = 80.0 / steps as f64;
        let c = argument.cosh();
        let s = argument.sinh();
        let canonical = vec![
            c,
            0.0,
            s,
            0.0, //
            0.0,
            1.0,
            0.0,
            1.0 / steps as f64, //
            s,
            0.0,
            c,
            0.0, //
            0.0,
            0.0,
            0.0,
            1.0,
        ];
        let root = 0.5;
        let basis = vec![
            root, root, root, root, //
            root, -root, root, -root, //
            root, root, -root, -root, //
            root, -root, -root, root,
        ];
        let factor = multiply_f64(
            &multiply_f64(&basis, &canonical, 4),
            &transpose(&basis, 4),
            4,
        );
        let factors = vec![factor; steps];
        let mut binary64 = vec![
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        for factor in &factors {
            binary64 = multiply_f64(factor, &binary64, 4);
        }
        for diagonal in 0..4 {
            binary64[diagonal * 4 + diagonal] -= 1.0;
        }
        let binary64_unit_determinant = determinant(binary64, 4);
        assert!(
            binary64_unit_determinant.abs() > 1e10,
            "binary64 unexpectedly retained a singular unit eigenspace"
        );

        let got = periodic_product_spectrum(&factors, 4, 256).expect("spectrum");
        let mut exponents: Vec<_> = got
            .multipliers
            .iter()
            .map(|value| value.log_modulus)
            .collect();
        exponents.sort_by(f64::total_cmp);
        eprintln!(
            "omega_beta=80 f64_det_product_minus_identity={binary64_unit_determinant:.17e} multiprecision_log_moduli={exponents:?}"
        );
        for (value, expected) in exponents.iter().zip([-80.0, 0.0, 0.0, 80.0]) {
            assert!((value - expected).abs() < 5e-11, "{exponents:?}");
        }
    }
}
