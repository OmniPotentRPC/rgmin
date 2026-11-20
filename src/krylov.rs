//! Matrix-free Krylov linear solvers.
//!
//! The operators are fallible because a matrix action may be an expensive
//! remote calculation.  GMRES uses right preconditioning: Arnoldi sees
//! `A M^-1`, while every reported residual is in the original system's norm.

/// A fallible matrix-free linear operator.
pub trait LinearOperator {
    /// Error returned by an operator application.
    type Error;

    /// Write `A input` to `output`.
    fn apply(&mut self, input: &[f64], output: &mut [f64]) -> Result<(), Self::Error>;

    /// Apply the operator to `vectors` contiguous directions.
    ///
    /// Implementations may evaluate all directions in one backend round.
    /// The default preserves serial behavior and order.
    fn apply_batch(
        &mut self,
        inputs: &[f64],
        outputs: &mut [f64],
        vectors: usize,
    ) -> Result<(), Self::Error> {
        if vectors == 0 {
            return Ok(());
        }
        let width = inputs.len() / vectors;
        for index in 0..vectors {
            let begin = index * width;
            let end = begin + width;
            self.apply(&inputs[begin..end], &mut outputs[begin..end])?;
        }
        Ok(())
    }
}

impl<E, F> LinearOperator for F
where
    F: FnMut(&[f64], &mut [f64]) -> Result<(), E>,
{
    type Error = E;

    fn apply(&mut self, input: &[f64], output: &mut [f64]) -> Result<(), Self::Error> {
        self(input, output)
    }
}

/// A fallible right preconditioner.
pub trait Preconditioner {
    /// Error returned by a preconditioner application.
    type Error;

    /// Write `M^-1 input` to `output`.
    fn apply(&mut self, input: &[f64], output: &mut [f64]) -> Result<(), Self::Error>;
}

/// Identity right preconditioner.
#[derive(Clone, Copy, Debug, Default)]
pub struct Identity;

impl Preconditioner for Identity {
    type Error = core::convert::Infallible;

    fn apply(&mut self, input: &[f64], output: &mut [f64]) -> Result<(), Self::Error> {
        output.copy_from_slice(input);
        Ok(())
    }
}

/// One GMRES iteration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KrylovIteration {
    /// One-based iteration number.
    pub iteration: usize,
    /// Absolute residual norm in the original, unpreconditioned system.
    pub absolute_residual: f64,
    /// Residual divided by the original right-hand-side norm.
    pub relative_residual: f64,
    /// Cumulative operator applications.
    pub operator_applications: usize,
    /// Cumulative right-preconditioner applications.
    pub preconditioner_applications: usize,
}

/// Streaming sink for Krylov telemetry.
pub trait KrylovObserver {
    /// Receive one completed iteration.
    fn observe(&mut self, iteration: KrylovIteration);
}

impl<F> KrylovObserver for F
where
    F: FnMut(KrylovIteration),
{
    fn observe(&mut self, iteration: KrylovIteration) {
        self(iteration);
    }
}

/// Observer that discards records.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoObserver;

impl KrylovObserver for NoObserver {
    fn observe(&mut self, _iteration: KrylovIteration) {}
}

/// Why GMRES stopped normally.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GmresStatus {
    /// The requested relative tolerance was met.
    Converged,
    /// The right-hand side was exactly zero.
    ZeroRightHandSide,
    /// The configured iteration budget was consumed.
    Exhausted,
}

/// Complete GMRES telemetry, available on success and refusal.
#[derive(Clone, Debug, PartialEq)]
pub struct KrylovRecord {
    /// Norm of the original right-hand side.
    pub right_hand_side_norm: f64,
    /// Every completed iteration.
    pub iterations: Vec<KrylovIteration>,
    /// Total operator applications.
    pub operator_applications: usize,
    /// Total right-preconditioner applications.
    pub preconditioner_applications: usize,
    /// Backend rounds used by operator applications.
    pub operator_rounds: usize,
}

impl KrylovRecord {
    fn new(right_hand_side_norm: f64) -> Self {
        Self {
            right_hand_side_norm,
            iterations: Vec::new(),
            operator_applications: 0,
            preconditioner_applications: 0,
            operator_rounds: 0,
        }
    }
}

/// A completed GMRES solve.
#[derive(Clone, Debug, PartialEq)]
pub struct GmresRun {
    /// Approximate solution.
    pub solution: Vec<f64>,
    /// Normal termination reason.
    pub status: GmresStatus,
    /// Iteration and application counts.
    pub record: KrylovRecord,
}

/// Typed numerical refusal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KrylovRefusal {
    /// The system has no entries.
    EmptySystem,
    /// A vector length does not match the system.
    Dimension,
    /// Tolerance is not finite and strictly positive.
    Tolerance,
    /// A nonzero system was given no iteration budget.
    ZeroIterationBudget,
    /// A right-hand side, operator image, or preconditioner image is non-finite.
    NonFinite,
    /// Arnoldi produced a singular projected system.
    SingularProjectedSystem,
}

/// A GMRES refusal with all telemetry collected before it.
#[derive(Debug)]
pub enum GmresFailure<OperatorError, PreconditionerError> {
    /// The operator returned an error.
    Operator {
        /// Original operator error.
        source: OperatorError,
        /// Partial solve record.
        record: KrylovRecord,
    },
    /// The preconditioner returned an error.
    Preconditioner {
        /// Original preconditioner error.
        source: PreconditionerError,
        /// Partial solve record.
        record: KrylovRecord,
    },
    /// The solver refused invalid or unsafe numerical data.
    Refused {
        /// Typed reason.
        reason: KrylovRefusal,
        /// Partial solve record.
        record: KrylovRecord,
    },
}

impl<OE, PE> GmresFailure<OE, PE> {
    /// Telemetry collected before refusal.
    pub fn record(&self) -> &KrylovRecord {
        match self {
            Self::Operator { record, .. }
            | Self::Preconditioner { record, .. }
            | Self::Refused { record, .. } => record,
        }
    }
}

/// Controls for unrestarted GMRES.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GmresOptions {
    /// Relative residual tolerance in the original system.
    pub relative_tolerance: f64,
    /// Maximum iterations. `None` uses the system dimension.
    pub max_iterations: Option<usize>,
}

impl GmresOptions {
    /// Unrestarted GMRES at `relative_tolerance`.
    pub fn unrestarted(relative_tolerance: f64) -> Self {
        Self {
            relative_tolerance,
            max_iterations: None,
        }
    }
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn norm(vector: &[f64]) -> f64 {
    dot(vector, vector).sqrt()
}

fn finite(vector: &[f64]) -> bool {
    vector.iter().all(|value| value.is_finite())
}

fn refusal<OE, PE>(reason: KrylovRefusal, record: KrylovRecord) -> GmresFailure<OE, PE> {
    GmresFailure::Refused { reason, record }
}

fn projected_coefficients<OE, PE>(
    columns: &[Vec<f64>],
    transformed_rhs: &[f64],
    record: KrylovRecord,
) -> Result<Vec<f64>, GmresFailure<OE, PE>> {
    let count = columns.len();
    let mut coefficients = vec![0.0; count];
    for row in (0..count).rev() {
        let mut value = transformed_rhs[row];
        for column in (row + 1)..count {
            value -= columns[column][row] * coefficients[column];
        }
        let diagonal = columns[row][row];
        if !diagonal.is_finite() || diagonal == 0.0 {
            return Err(refusal(KrylovRefusal::SingularProjectedSystem, record));
        }
        coefficients[row] = value / diagonal;
    }
    Ok(coefficients)
}

fn linear_combination(coefficients: &[f64], vectors: &[Vec<f64>], dimension: usize) -> Vec<f64> {
    let mut result = vec![0.0; dimension];
    for (coefficient, vector) in coefficients.iter().zip(vectors) {
        for (slot, value) in result.iter_mut().zip(vector) {
            *slot += coefficient * value;
        }
    }
    result
}

fn true_residual_norm(
    right_hand_side: &[f64],
    operator_images: &[Vec<f64>],
    coefficients: &[f64],
) -> f64 {
    let mut residual = right_hand_side.to_vec();
    for (coefficient, image) in coefficients.iter().zip(operator_images) {
        for (slot, value) in residual.iter_mut().zip(image) {
            *slot -= coefficient * value;
        }
    }
    norm(&residual)
}

/// Unrestarted, right-preconditioned GMRES.
///
/// Two-pass modified Gram--Schmidt and Givens rotations match full GMRES.
/// Every reported residual is recomputed as `‖b - A x_k‖` from the operator
/// images already produced by Arnoldi. This adds no operator applications and
/// prevents a small Givens estimate from accepting an inaccurate solution.
pub fn gmres_right<O, P, B>(
    operator: &mut O,
    preconditioner: &mut P,
    right_hand_side: &[f64],
    options: GmresOptions,
    observer: &mut B,
) -> Result<GmresRun, GmresFailure<O::Error, P::Error>>
where
    O: LinearOperator,
    P: Preconditioner,
    B: KrylovObserver,
{
    let dimension = right_hand_side.len();
    let right_hand_side_norm = norm(right_hand_side);
    let mut record = KrylovRecord::new(right_hand_side_norm);
    if dimension == 0 {
        return Err(refusal(KrylovRefusal::EmptySystem, record));
    }
    if !finite(right_hand_side) {
        return Err(refusal(KrylovRefusal::NonFinite, record));
    }
    if !(options.relative_tolerance.is_finite() && options.relative_tolerance > 0.0) {
        return Err(refusal(KrylovRefusal::Tolerance, record));
    }
    if right_hand_side_norm == 0.0 {
        return Ok(GmresRun {
            solution: vec![0.0; dimension],
            status: GmresStatus::ZeroRightHandSide,
            record,
        });
    }
    let requested = match options.max_iterations {
        Some(limit) => limit,
        None => dimension,
    };
    let budget = requested.min(dimension);
    if budget == 0 {
        return Err(refusal(KrylovRefusal::ZeroIterationBudget, record));
    }
    let mut basis = vec![
        right_hand_side
            .iter()
            .map(|value| value / right_hand_side_norm)
            .collect::<Vec<_>>(),
    ];
    let mut preconditioned_basis: Vec<Vec<f64>> = Vec::new();
    let mut operator_images: Vec<Vec<f64>> = Vec::new();
    let mut columns: Vec<Vec<f64>> = Vec::new();
    let mut cosines: Vec<f64> = Vec::new();
    let mut sines: Vec<f64> = Vec::new();
    let mut transformed_rhs = vec![right_hand_side_norm];
    for column_index in 0..budget {
        let mut preconditioned = vec![0.0; dimension];
        if let Err(source) = preconditioner.apply(&basis[column_index], &mut preconditioned) {
            return Err(GmresFailure::Preconditioner { source, record });
        }
        record.preconditioner_applications += 1;
        if !finite(&preconditioned) {
            return Err(refusal(KrylovRefusal::NonFinite, record));
        }
        let mut image = vec![0.0; dimension];
        if let Err(source) = operator.apply(&preconditioned, &mut image) {
            return Err(GmresFailure::Operator { source, record });
        }
        record.operator_applications += 1;
        record.operator_rounds += 1;
        if !finite(&image) {
            return Err(refusal(KrylovRefusal::NonFinite, record));
        }
        preconditioned_basis.push(preconditioned);
        operator_images.push(image.clone());
        let mut hessenberg = vec![0.0; column_index + 2];
        for _ in 0..2 {
            for (row, vector) in basis.iter().enumerate() {
                let coefficient = dot(vector, &image);
                hessenberg[row] += coefficient;
                for (slot, value) in image.iter_mut().zip(vector) {
                    *slot -= coefficient * value;
                }
            }
        }
        let subdiagonal = norm(&image);
        hessenberg[column_index + 1] = subdiagonal;
        for row in 0..column_index {
            let upper = hessenberg[row];
            let lower = hessenberg[row + 1];
            hessenberg[row] = cosines[row] * upper + sines[row] * lower;
            hessenberg[row + 1] = -sines[row] * upper + cosines[row] * lower;
        }
        let rotation_norm = hessenberg[column_index].hypot(hessenberg[column_index + 1]);
        if !rotation_norm.is_finite() || rotation_norm == 0.0 {
            let iteration = KrylovIteration {
                iteration: column_index + 1,
                absolute_residual: right_hand_side_norm,
                relative_residual: 1.0,
                operator_applications: record.operator_applications,
                preconditioner_applications: record.preconditioner_applications,
            };
            record.iterations.push(iteration);
            observer.observe(iteration);
            return Err(refusal(KrylovRefusal::SingularProjectedSystem, record));
        }
        let cosine = hessenberg[column_index] / rotation_norm;
        let sine = hessenberg[column_index + 1] / rotation_norm;
        hessenberg[column_index] = rotation_norm;
        hessenberg.truncate(column_index + 1);
        cosines.push(cosine);
        sines.push(sine);
        let previous = transformed_rhs[column_index];
        transformed_rhs[column_index] = cosine * previous;
        transformed_rhs.push(-sine * previous);
        columns.push(hessenberg);
        let coefficients = projected_coefficients(&columns, &transformed_rhs, record.clone())?;
        let absolute_residual =
            true_residual_norm(right_hand_side, &operator_images, &coefficients);
        if !absolute_residual.is_finite() {
            return Err(refusal(KrylovRefusal::NonFinite, record));
        }
        let iteration = KrylovIteration {
            iteration: column_index + 1,
            absolute_residual,
            relative_residual: absolute_residual / right_hand_side_norm,
            operator_applications: record.operator_applications,
            preconditioner_applications: record.preconditioner_applications,
        };
        record.iterations.push(iteration);
        observer.observe(iteration);
        let converged = absolute_residual <= options.relative_tolerance * right_hand_side_norm;
        let exhausted = column_index + 1 == budget || column_index + 1 == dimension;
        if converged || exhausted {
            let status = if converged {
                GmresStatus::Converged
            } else {
                GmresStatus::Exhausted
            };
            let answer = linear_combination(&coefficients, &preconditioned_basis, dimension);
            if !finite(&answer) {
                return Err(refusal(KrylovRefusal::NonFinite, record));
            }
            return Ok(GmresRun {
                solution: answer,
                status,
                record,
            });
        }
        if !subdiagonal.is_finite() || subdiagonal == 0.0 {
            return Err(refusal(KrylovRefusal::SingularProjectedSystem, record));
        }
        basis.push(image.iter().map(|value| value / subdiagonal).collect());
    }
    Err(refusal(KrylovRefusal::SingularProjectedSystem, record))
}

struct BlockState {
    basis: Vec<Vec<f64>>,
    preconditioned_basis: Vec<Vec<f64>>,
    columns: Vec<Vec<f64>>,
    cosines: Vec<f64>,
    sines: Vec<f64>,
    transformed_rhs: Vec<f64>,
    operator_images: Vec<Vec<f64>>,
    record: KrylovRecord,
    run: Option<GmresRun>,
}

/// Block GMRES for several independent right-hand sides.
///
/// Every active Arnoldi direction at an iteration is passed to
/// [`LinearOperator::apply_batch`] in one call. Each column otherwise follows
/// the scalar algorithm exactly, so batching does not change its trajectory.
/// `operator_rounds` counts batch calls and `operator_applications` counts
/// directions. The result ordering matches `right_hand_sides`.
pub fn gmres_right_block<O, P, B>(
    operator: &mut O,
    preconditioners: &mut [P],
    right_hand_sides: &[Vec<f64>],
    options: GmresOptions,
    observers: &mut [B],
) -> Result<Vec<GmresRun>, GmresFailure<O::Error, P::Error>>
where
    O: LinearOperator,
    P: Preconditioner,
    B: KrylovObserver,
{
    if preconditioners.len() != right_hand_sides.len() || observers.len() != right_hand_sides.len()
    {
        return Err(refusal(KrylovRefusal::Dimension, KrylovRecord::new(0.0)));
    }
    if !(options.relative_tolerance.is_finite() && options.relative_tolerance > 0.0) {
        return Err(refusal(KrylovRefusal::Tolerance, KrylovRecord::new(0.0)));
    }
    if right_hand_sides.is_empty() {
        return Ok(Vec::new());
    }
    let dimension = right_hand_sides[0].len();
    if dimension == 0 {
        return Err(refusal(KrylovRefusal::EmptySystem, KrylovRecord::new(0.0)));
    }
    let requested = match options.max_iterations {
        Some(limit) => limit,
        None => dimension,
    };
    let budget = requested.min(dimension);
    let mut states = Vec::with_capacity(right_hand_sides.len());
    for right_hand_side in right_hand_sides {
        if right_hand_side.len() != dimension {
            return Err(refusal(KrylovRefusal::Dimension, KrylovRecord::new(0.0)));
        }
        let right_hand_side_norm = norm(right_hand_side);
        let record = KrylovRecord::new(right_hand_side_norm);
        if !finite(right_hand_side) {
            return Err(refusal(KrylovRefusal::NonFinite, record));
        }
        if right_hand_side_norm == 0.0 {
            states.push(BlockState {
                basis: Vec::new(),
                preconditioned_basis: Vec::new(),
                columns: Vec::new(),
                cosines: Vec::new(),
                sines: Vec::new(),
                transformed_rhs: Vec::new(),
                operator_images: Vec::new(),
                run: Some(GmresRun {
                    solution: vec![0.0; dimension],
                    status: GmresStatus::ZeroRightHandSide,
                    record,
                }),
                record: KrylovRecord::new(right_hand_side_norm),
            });
        } else {
            states.push(BlockState {
                basis: vec![
                    right_hand_side
                        .iter()
                        .map(|value| value / right_hand_side_norm)
                        .collect(),
                ],
                preconditioned_basis: Vec::new(),
                columns: Vec::new(),
                cosines: Vec::new(),
                sines: Vec::new(),
                transformed_rhs: vec![right_hand_side_norm],
                operator_images: Vec::new(),
                record,
                run: None,
            });
        }
    }
    if budget == 0 {
        for state in &states {
            if state.run.is_none() {
                return Err(refusal(
                    KrylovRefusal::ZeroIterationBudget,
                    state.record.clone(),
                ));
            }
        }
        return Ok(states.into_iter().filter_map(|state| state.run).collect());
    }
    for column_index in 0..budget {
        let active: Vec<usize> = states
            .iter()
            .enumerate()
            .filter_map(|(index, state)| state.run.is_none().then_some(index))
            .collect();
        if active.is_empty() {
            break;
        }
        let mut directions = vec![0.0; active.len() * dimension];
        for (block, index) in active.iter().copied().enumerate() {
            let state = &mut states[index];
            let begin = block * dimension;
            let end = begin + dimension;
            if let Err(source) = preconditioners[index]
                .apply(&state.basis[column_index], &mut directions[begin..end])
            {
                return Err(GmresFailure::Preconditioner {
                    source,
                    record: state.record.clone(),
                });
            }
            state.record.preconditioner_applications += 1;
            if !finite(&directions[begin..end]) {
                return Err(refusal(KrylovRefusal::NonFinite, state.record.clone()));
            }
        }
        let mut images = vec![0.0; directions.len()];
        if let Err(source) = operator.apply_batch(&directions, &mut images, active.len()) {
            return Err(GmresFailure::Operator {
                source,
                record: states[active[0]].record.clone(),
            });
        }
        for (block, index) in active.iter().copied().enumerate() {
            let state = &mut states[index];
            state.record.operator_applications += 1;
            state.record.operator_rounds += 1;
            let begin = block * dimension;
            let end = begin + dimension;
            if !finite(&images[begin..end]) {
                return Err(refusal(KrylovRefusal::NonFinite, state.record.clone()));
            }
            state
                .preconditioned_basis
                .push(directions[begin..end].to_vec());
            let mut image = images[begin..end].to_vec();
            state.operator_images.push(image.clone());
            let mut hessenberg = vec![0.0; column_index + 2];
            for _ in 0..2 {
                for (row, vector) in state.basis.iter().enumerate() {
                    let coefficient = dot(vector, &image);
                    hessenberg[row] += coefficient;
                    for (slot, value) in image.iter_mut().zip(vector) {
                        *slot -= coefficient * value;
                    }
                }
            }
            let subdiagonal = norm(&image);
            hessenberg[column_index + 1] = subdiagonal;
            for row in 0..column_index {
                let upper = hessenberg[row];
                let lower = hessenberg[row + 1];
                hessenberg[row] = state.cosines[row] * upper + state.sines[row] * lower;
                hessenberg[row + 1] = -state.sines[row] * upper + state.cosines[row] * lower;
            }
            let rotation_norm = hessenberg[column_index].hypot(hessenberg[column_index + 1]);
            if !rotation_norm.is_finite() || rotation_norm == 0.0 {
                let iteration = KrylovIteration {
                    iteration: column_index + 1,
                    absolute_residual: state.record.right_hand_side_norm,
                    relative_residual: 1.0,
                    operator_applications: state.record.operator_applications,
                    preconditioner_applications: state.record.preconditioner_applications,
                };
                state.record.iterations.push(iteration);
                observers[index].observe(iteration);
                return Err(refusal(
                    KrylovRefusal::SingularProjectedSystem,
                    state.record.clone(),
                ));
            }
            let cosine = hessenberg[column_index] / rotation_norm;
            let sine = hessenberg[column_index + 1] / rotation_norm;
            hessenberg[column_index] = rotation_norm;
            hessenberg.truncate(column_index + 1);
            state.cosines.push(cosine);
            state.sines.push(sine);
            let previous = state.transformed_rhs[column_index];
            state.transformed_rhs[column_index] = cosine * previous;
            state.transformed_rhs.push(-sine * previous);
            state.columns.push(hessenberg);
            let coefficients = projected_coefficients(
                &state.columns,
                &state.transformed_rhs,
                state.record.clone(),
            )?;
            let absolute_residual = true_residual_norm(
                &right_hand_sides[index],
                &state.operator_images,
                &coefficients,
            );
            if !absolute_residual.is_finite() {
                return Err(refusal(KrylovRefusal::NonFinite, state.record.clone()));
            }
            let iteration = KrylovIteration {
                iteration: column_index + 1,
                absolute_residual,
                relative_residual: absolute_residual / state.record.right_hand_side_norm,
                operator_applications: state.record.operator_applications,
                preconditioner_applications: state.record.preconditioner_applications,
            };
            state.record.iterations.push(iteration);
            observers[index].observe(iteration);
            let converged =
                absolute_residual <= options.relative_tolerance * state.record.right_hand_side_norm;
            let exhausted = column_index + 1 == budget || column_index + 1 == dimension;
            if converged || exhausted {
                let answer =
                    linear_combination(&coefficients, &state.preconditioned_basis, dimension);
                if !finite(&answer) {
                    return Err(refusal(KrylovRefusal::NonFinite, state.record.clone()));
                }
                state.run = Some(GmresRun {
                    solution: answer,
                    status: if converged {
                        GmresStatus::Converged
                    } else {
                        GmresStatus::Exhausted
                    },
                    record: state.record.clone(),
                });
            } else {
                if !subdiagonal.is_finite() || subdiagonal == 0.0 {
                    return Err(refusal(
                        KrylovRefusal::SingularProjectedSystem,
                        state.record.clone(),
                    ));
                }
                state
                    .basis
                    .push(image.iter().map(|value| value / subdiagonal).collect());
            }
        }
    }
    let mut runs = Vec::with_capacity(states.len());
    for state in states {
        match state.run {
            Some(run) => runs.push(run),
            None => {
                return Err(refusal(
                    KrylovRefusal::SingularProjectedSystem,
                    state.record,
                ));
            }
        }
    }
    Ok(runs)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Diagonal {
        values: Vec<f64>,
        scalar_calls: usize,
        batch_calls: usize,
    }

    impl LinearOperator for Diagonal {
        type Error = ();

        fn apply(&mut self, input: &[f64], output: &mut [f64]) -> Result<(), Self::Error> {
            self.scalar_calls += 1;
            for ((slot, value), diagonal) in output.iter_mut().zip(input).zip(&self.values) {
                *slot = diagonal * value;
            }
            Ok(())
        }

        fn apply_batch(
            &mut self,
            inputs: &[f64],
            outputs: &mut [f64],
            vectors: usize,
        ) -> Result<(), Self::Error> {
            self.batch_calls += 1;
            let width = self.values.len();
            for vector in 0..vectors {
                for index in 0..width {
                    outputs[vector * width + index] =
                        self.values[index] * inputs[vector * width + index];
                }
            }
            Ok(())
        }
    }

    struct DiagonalInverse {
        values: Vec<f64>,
    }

    impl Preconditioner for DiagonalInverse {
        type Error = ();

        fn apply(&mut self, input: &[f64], output: &mut [f64]) -> Result<(), Self::Error> {
            for ((slot, value), diagonal) in output.iter_mut().zip(input).zip(&self.values) {
                *slot = value / diagonal;
            }
            Ok(())
        }
    }

    #[test]
    fn telemetry_streams_non_null_norms_and_preconditioning_reduces_iterations() {
        let diagonal = vec![1.0, 10.0, 100.0, 1000.0];
        let right_hand_side = vec![1.0; diagonal.len()];
        let options = GmresOptions::unrestarted(1e-12);
        let mut plain_operator = Diagonal {
            values: diagonal.clone(),
            scalar_calls: 0,
            batch_calls: 0,
        };
        let mut identity = Identity;
        let mut streamed = Vec::new();
        let plain = gmres_right(
            &mut plain_operator,
            &mut identity,
            &right_hand_side,
            options,
            &mut |iteration| streamed.push(iteration),
        );
        let plain = match plain {
            Ok(run) => run,
            Err(_) => return assert!(false, "plain diagonal solve refused"),
        };
        let mut preconditioned_operator = Diagonal {
            values: diagonal.clone(),
            scalar_calls: 0,
            batch_calls: 0,
        };
        let mut inverse = DiagonalInverse { values: diagonal };
        let preconditioned = gmres_right(
            &mut preconditioned_operator,
            &mut inverse,
            &right_hand_side,
            options,
            &mut NoObserver,
        );
        let preconditioned = match preconditioned {
            Ok(run) => run,
            Err(_) => return assert!(false, "preconditioned diagonal solve refused"),
        };
        assert_eq!(streamed, plain.record.iterations);
        assert!(
            streamed.iter().all(|row| {
                row.absolute_residual.is_finite() && row.relative_residual.is_finite()
            })
        );
        assert!(
            preconditioned.record.iterations.len() < plain.record.iterations.len(),
            "exact right preconditioning should shorten stiff diagonal GMRES"
        );
    }

    #[test]
    fn block_and_serial_runs_are_identical_and_batch_directions_share_rounds() {
        let diagonal = vec![1.0, 3.0, 9.0];
        let right_hand_sides = vec![vec![1.0, 2.0, 3.0], vec![3.0, -1.0, 2.0]];
        let options = GmresOptions::unrestarted(1e-13);
        let mut serial_operator = Diagonal {
            values: diagonal.clone(),
            scalar_calls: 0,
            batch_calls: 0,
        };
        let mut serial_runs = Vec::new();
        for right_hand_side in &right_hand_sides {
            let run = gmres_right(
                &mut serial_operator,
                &mut Identity,
                right_hand_side,
                options,
                &mut NoObserver,
            );
            match run {
                Ok(run) => serial_runs.push(run),
                Err(_) => return assert!(false, "serial solve refused"),
            }
        }
        let mut block_operator = Diagonal {
            values: diagonal,
            scalar_calls: 0,
            batch_calls: 0,
        };
        let block_runs = gmres_right_block(
            &mut block_operator,
            &mut [Identity, Identity],
            &right_hand_sides,
            options,
            &mut [NoObserver, NoObserver],
        );
        let block_runs = match block_runs {
            Ok(runs) => runs,
            Err(_) => return assert!(false, "block solve refused"),
        };
        assert_eq!(block_runs, serial_runs);
        assert_eq!(block_operator.scalar_calls, 0);
        assert!(block_operator.batch_calls < serial_operator.scalar_calls);
    }

    struct NanPreconditioner;

    impl Preconditioner for NanPreconditioner {
        type Error = ();

        fn apply(&mut self, _input: &[f64], output: &mut [f64]) -> Result<(), Self::Error> {
            output.fill(f64::NAN);
            Ok(())
        }
    }

    #[test]
    fn a_non_finite_preconditioner_is_a_typed_refusal_with_a_record() {
        let mut operator = Diagonal {
            values: vec![1.0, 2.0],
            scalar_calls: 0,
            batch_calls: 0,
        };
        let refused = gmres_right(
            &mut operator,
            &mut NanPreconditioner,
            &[1.0, 1.0],
            GmresOptions::unrestarted(1e-8),
            &mut NoObserver,
        );
        match refused {
            Err(GmresFailure::Refused { reason, record }) => {
                assert_eq!(reason, KrylovRefusal::NonFinite);
                assert_eq!(record.operator_applications, 0);
                assert_eq!(record.preconditioner_applications, 1);
            }
            _ => assert!(false, "NaN preconditioner did not produce a typed refusal"),
        }
    }

    #[test]
    fn a_small_givens_estimate_does_not_hide_a_large_true_residual() {
        let dimension = 12;
        let mut matrix = vec![0.0; dimension * dimension];
        for row in 0..dimension {
            matrix[row * dimension + row] = 0.5 + 0.15 * row as f64;
            if row + 1 < dimension {
                matrix[row * dimension + row + 1] = 8.0 + row as f64;
            }
            if row + 2 < dimension {
                matrix[row * dimension + row + 2] = -3.0;
            }
        }
        let mut operator =
            |input: &[f64], output: &mut [f64]| -> Result<(), core::convert::Infallible> {
                for row in 0..dimension {
                    output[row] = matrix[row * dimension..(row + 1) * dimension]
                        .iter()
                        .zip(input)
                        .map(|(entry, value)| entry * value)
                        .sum();
                }
                Ok(())
            };
        let run = gmres_right(
            &mut operator,
            &mut Identity,
            &vec![1.0; dimension],
            GmresOptions::unrestarted(1e-8),
            &mut NoObserver,
        );
        let run = match run {
            Ok(value) => value,
            Err(_) => return assert!(false, "non-normal solve refused"),
        };
        let residual = match run.record.iterations.last() {
            Some(iteration) => iteration.absolute_residual,
            None => return assert!(false, "non-normal solve returned no iteration"),
        };
        assert_eq!(run.status, GmresStatus::Exhausted);
        assert!(residual > 1e-5 && residual < 1e-3);
    }

    #[test]
    fn zero_budget_and_a_zero_arnoldi_column_have_distinct_records() {
        let options = GmresOptions {
            relative_tolerance: 1e-8,
            max_iterations: Some(0),
        };
        let mut identity_operator =
            |input: &[f64], output: &mut [f64]| -> Result<(), core::convert::Infallible> {
                output.copy_from_slice(input);
                Ok(())
            };
        let refused = gmres_right(
            &mut identity_operator,
            &mut Identity,
            &[1.0],
            options,
            &mut NoObserver,
        );
        match refused {
            Err(GmresFailure::Refused { reason, record }) => {
                assert_eq!(reason, KrylovRefusal::ZeroIterationBudget);
                assert!(record.iterations.is_empty());
                assert_eq!(record.operator_applications, 0);
            }
            _ => return assert!(false, "zero budget did not have its own refusal"),
        }

        let mut singular_operator =
            |input: &[f64], output: &mut [f64]| -> Result<(), core::convert::Infallible> {
                output[0] = input[0];
                output[1] = 0.0;
                Ok(())
            };
        let refused = gmres_right(
            &mut singular_operator,
            &mut Identity,
            &[0.0, 1.0],
            GmresOptions::unrestarted(1e-8),
            &mut NoObserver,
        );
        match refused {
            Err(GmresFailure::Refused { reason, record }) => {
                assert_eq!(reason, KrylovRefusal::SingularProjectedSystem);
                assert_eq!(record.iterations.len(), 1);
                assert_eq!(record.iterations[0].absolute_residual, 1.0);
                assert_eq!(record.operator_applications, 1);
            }
            _ => assert!(false, "zero Arnoldi column lost its iteration record"),
        }
    }
}
