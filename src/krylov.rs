//! Matrix-free Krylov linear solvers.
//!
//! The operators are fallible because a matrix action may be an expensive
//! remote calculation.  GMRES uses right preconditioning: Arnoldi sees
//! `A M^-1`, while every reported residual is in the original system's norm.

use std::error::Error;
use std::fmt;

/// Failure from a batched operator application.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BatchOperatorError<E> {
    /// The flat input/output shape does not match the direction count.
    Dimension {
        /// Number of input entries.
        inputs: usize,
        /// Number of output entries.
        outputs: usize,
        /// Number of contiguous vectors.
        vectors: usize,
    },
    /// A serial operator application failed after earlier directions completed.
    Operator {
        /// Original operator error.
        source: E,
        /// Number of directions completed before the failure.
        completed: usize,
    },
}

impl<E> BatchOperatorError<E> {
    /// Directions whose operator applications completed before this failure.
    pub fn completed(&self) -> usize {
        match self {
            Self::Dimension { .. } => 0,
            Self::Operator { completed, .. } => *completed,
        }
    }
}

impl<E: fmt::Display> fmt::Display for BatchOperatorError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dimension {
                inputs,
                outputs,
                vectors,
            } => write!(
                f,
                "batched operator shape mismatch: {inputs} input entries and {outputs} output \
                 entries for {vectors} vectors"
            ),
            Self::Operator { source, completed } => write!(
                f,
                "batched operator failed after {completed} completed directions: {source}"
            ),
        }
    }
}

impl<E: Error + 'static> Error for BatchOperatorError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Dimension { .. } => None,
            Self::Operator { source, .. } => Some(source),
        }
    }
}

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
    ) -> Result<(), BatchOperatorError<Self::Error>> {
        if vectors == 0 {
            return if inputs.is_empty() && outputs.is_empty() {
                Ok(())
            } else {
                Err(BatchOperatorError::Dimension {
                    inputs: inputs.len(),
                    outputs: outputs.len(),
                    vectors,
                })
            };
        }
        if !inputs.len().is_multiple_of(vectors) || outputs.len() != inputs.len() {
            return Err(BatchOperatorError::Dimension {
                inputs: inputs.len(),
                outputs: outputs.len(),
                vectors,
            });
        }
        let width = inputs.len() / vectors;
        for index in 0..vectors {
            let begin = index * width;
            let end = begin + width;
            if let Err(source) = self.apply(&inputs[begin..end], &mut outputs[begin..end]) {
                return Err(BatchOperatorError::Operator {
                    source,
                    completed: index,
                });
            }
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

/// Shape refusal from the identity preconditioner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IdentityError {
    /// Number of input entries.
    pub input: usize,
    /// Number of output entries.
    pub output: usize,
}

impl fmt::Display for IdentityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "identity preconditioner shape mismatch: {} input entries and {} output entries",
            self.input, self.output
        )
    }
}

impl Error for IdentityError {}

impl Preconditioner for Identity {
    type Error = IdentityError;

    fn apply(&mut self, input: &[f64], output: &mut [f64]) -> Result<(), Self::Error> {
        if input.len() != output.len() {
            return Err(IdentityError {
                input: input.len(),
                output: output.len(),
            });
        }
        output.copy_from_slice(input);
        Ok(())
    }
}

/// One GMRES iteration.
///
/// Built by the solver; start from `default()` and the `with_*` setters to
/// construct one by hand.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
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

with_setters!(KrylovIteration {
    with_iteration => iteration: val usize,
    with_absolute_residual => absolute_residual: val f64,
    with_relative_residual => relative_residual: val f64,
    with_operator_applications => operator_applications: val usize,
    with_preconditioner_applications => preconditioner_applications: val usize,
});

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
    /// Arnoldi reached an invariant subspace before meeting the tolerance.
    Stagnated,
}

/// Complete GMRES telemetry, available on success and refusal.
///
/// Built by the solver; start from `default()` and the `with_*` setters to
/// construct one by hand.
#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
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
            ..Self::default()
        }
    }
}

with_setters!(KrylovRecord {
    with_right_hand_side_norm => right_hand_side_norm: val f64,
    with_iterations => iterations: val Vec<KrylovIteration>,
    with_operator_applications => operator_applications: val usize,
    with_preconditioner_applications => preconditioner_applications: val usize,
    with_operator_rounds => operator_rounds: val usize,
});

/// A completed GMRES solve.
///
/// Built by the solver. The default is the empty solve of a zero right-hand
/// side, and the `with_*` setters replace its parts.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct GmresRun {
    /// Approximate solution.
    pub solution: Vec<f64>,
    /// Normal termination reason.
    pub status: GmresStatus,
    /// Iteration and application counts.
    pub record: KrylovRecord,
}

impl Default for GmresRun {
    fn default() -> Self {
        Self {
            solution: Vec::new(),
            status: GmresStatus::ZeroRightHandSide,
            record: KrylovRecord::default(),
        }
    }
}

with_setters!(GmresRun {
    with_solution => solution: val Vec<f64>,
    with_status => status: val GmresStatus,
    with_record => record: val KrylovRecord,
});

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
    /// A right-hand side, operator image, preconditioner image, Givens
    /// rotation, projected coefficient, or residual is non-finite.
    NonFinite,
    /// Arnoldi produced a singular projected system. The record ends with an
    /// iteration that carries the true residual of the unchanged iterate.
    SingularProjectedSystem,
}

impl fmt::Display for KrylovRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::EmptySystem => "the system has no entries",
            Self::Dimension => "a vector length does not match the system",
            Self::Tolerance => "the tolerance is not finite and strictly positive",
            Self::ZeroIterationBudget => "a nonzero system was given no iteration budget",
            Self::NonFinite => "a value in the solve is not finite",
            Self::SingularProjectedSystem => "the projected least-squares system is singular",
        })
    }
}

impl Error for KrylovRefusal {}

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
    /// A batched operator failed with every right-hand-side record.
    BlockOperator {
        /// Original batched operator error.
        source: OperatorError,
        /// Record for the first active right-hand side.
        record: KrylovRecord,
        /// Records for every other right-hand side, including completed ones.
        sibling_records: Vec<KrylovRecord>,
    },
    /// A block preconditioner failed with every right-hand-side record.
    BlockPreconditioner {
        /// Original preconditioner error.
        source: PreconditionerError,
        /// Record for the right-hand side whose preconditioner failed.
        record: KrylovRecord,
        /// Records for every other right-hand side.
        sibling_records: Vec<KrylovRecord>,
    },
    /// A block solve refused with every right-hand-side record.
    BlockRefused {
        /// Typed reason.
        reason: KrylovRefusal,
        /// Record for the right-hand side that caused the refusal.
        record: KrylovRecord,
        /// Records for every other right-hand side.
        sibling_records: Vec<KrylovRecord>,
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

impl<OE: fmt::Display, PE: fmt::Display> fmt::Display for GmresFailure<OE, PE> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Operator { source, .. } => write!(f, "operator application failed: {source}"),
            Self::BlockOperator { source, .. } => {
                write!(f, "batched operator application failed: {source}")
            }
            Self::Preconditioner { source, .. } => {
                write!(f, "preconditioner application failed: {source}")
            }
            Self::BlockPreconditioner { source, .. } => {
                write!(f, "block preconditioner application failed: {source}")
            }
            Self::Refused { reason, .. } => write!(f, "GMRES refused: {reason}"),
            Self::BlockRefused { reason, .. } => write!(f, "block GMRES refused: {reason}"),
        }
    }
}

impl<OE: Error + 'static, PE: Error + 'static> Error for GmresFailure<OE, PE> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Operator { source, .. } | Self::BlockOperator { source, .. } => Some(source),
            Self::Preconditioner { source, .. } | Self::BlockPreconditioner { source, .. } => {
                Some(source)
            }
            Self::Refused { reason, .. } | Self::BlockRefused { reason, .. } => Some(reason),
        }
    }
}

/// Failure returned by batched independent-right-hand-side GMRES.
pub type BlockGmresFailure<OperatorError, PreconditionerError> =
    GmresFailure<BatchOperatorError<OperatorError>, PreconditionerError>;

impl<OE, PE> GmresFailure<OE, PE> {
    /// Telemetry collected before refusal.
    pub fn record(&self) -> &KrylovRecord {
        match self {
            Self::Operator { record, .. }
            | Self::BlockOperator { record, .. }
            | Self::BlockPreconditioner { record, .. }
            | Self::BlockRefused { record, .. }
            | Self::Preconditioner { record, .. }
            | Self::Refused { record, .. } => record,
        }
    }

    /// Every other record carried by a failed block solve.
    pub fn sibling_records(&self) -> &[KrylovRecord] {
        match self {
            Self::BlockOperator {
                sibling_records, ..
            }
            | Self::BlockPreconditioner {
                sibling_records, ..
            }
            | Self::BlockRefused {
                sibling_records, ..
            } => sibling_records,
            _ => &[],
        }
    }
}

/// Controls for unrestarted GMRES.
///
/// The default asks for a relative residual of `1e-8` with the iteration
/// budget equal to the system dimension. Chain the `with_*` setters after
/// `default()` or [`Self::unrestarted`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
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

impl Default for GmresOptions {
    fn default() -> Self {
        Self::unrestarted(1e-8)
    }
}

with_setters!(GmresOptions {
    with_relative_tolerance => relative_tolerance: val f64,
    with_max_iterations => max_iterations: opt usize,
});

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn norm(vector: &[f64]) -> f64 {
    let mut largest = 0.0_f64;
    for value in vector {
        if value.is_nan() {
            return f64::NAN;
        }
        largest = largest.max(value.abs());
    }
    if largest == 0.0 {
        return 0.0;
    }
    if largest.is_infinite() {
        return f64::INFINITY;
    }
    let unscaled_upper = f64::MAX.sqrt() / (vector.len() as f64).sqrt();
    let unscaled_lower = f64::MIN_POSITIVE.sqrt();
    if largest >= unscaled_lower && largest <= unscaled_upper {
        return dot(vector, vector).sqrt();
    }
    let mut sum_squares = 0.0;
    for value in vector {
        let ratio = value / largest;
        sum_squares += ratio * ratio;
    }
    largest * sum_squares.sqrt()
}

fn finite(vector: &[f64]) -> bool {
    vector.iter().all(|value| value.is_finite())
}

fn refusal<OE, PE>(reason: KrylovRefusal, record: KrylovRecord) -> GmresFailure<OE, PE> {
    GmresFailure::Refused { reason, record }
}

fn projected_coefficients(
    columns: &[Vec<f64>],
    transformed_rhs: &[f64],
) -> Result<Vec<f64>, KrylovRefusal> {
    let count = columns.len();
    let mut coefficients = vec![0.0; count];
    for row in (0..count).rev() {
        let mut value = transformed_rhs[row];
        for column in (row + 1)..count {
            value -= columns[column][row] * coefficients[column];
        }
        let diagonal = columns[row][row];
        if !diagonal.is_finite() {
            return Err(KrylovRefusal::NonFinite);
        }
        if diagonal == 0.0 {
            return Err(KrylovRefusal::SingularProjectedSystem);
        }
        let coefficient = value / diagonal;
        if !coefficient.is_finite() {
            return Err(KrylovRefusal::NonFinite);
        }
        coefficients[row] = coefficient;
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

fn completed_iteration(
    iteration: usize,
    absolute_residual: f64,
    record: &KrylovRecord,
) -> Result<KrylovIteration, KrylovRefusal> {
    let relative_residual = absolute_residual / record.right_hand_side_norm;
    if !absolute_residual.is_finite() || !relative_residual.is_finite() {
        return Err(KrylovRefusal::NonFinite);
    }
    Ok(KrylovIteration {
        iteration,
        absolute_residual,
        relative_residual,
        operator_applications: record.operator_applications,
        preconditioner_applications: record.preconditioner_applications,
    })
}

/// True residual of the iterate that a breakdown leaves unchanged: the one
/// recorded by the previous iteration, or the right-hand side for the first.
fn breakdown_iteration(
    iteration: usize,
    record: &KrylovRecord,
) -> Result<KrylovIteration, KrylovRefusal> {
    let unchanged = match record.iterations.last() {
        Some(previous) => previous.absolute_residual,
        None => record.right_hand_side_norm,
    };
    completed_iteration(iteration, unchanged, record)
}

fn breakdown_reason(rotation_norm: f64) -> KrylovRefusal {
    if rotation_norm.is_finite() {
        KrylovRefusal::SingularProjectedSystem
    } else {
        KrylovRefusal::NonFinite
    }
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
    if !finite(right_hand_side) || !right_hand_side_norm.is_finite() {
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
        // A non-finite subdiagonal makes the rotation norm non-finite, so the
        // breakdown test below is also the finiteness test for Arnoldi.
        let rotation_norm = hessenberg[column_index].hypot(hessenberg[column_index + 1]);
        if !rotation_norm.is_finite() || rotation_norm == 0.0 {
            let reason = breakdown_reason(rotation_norm);
            let iteration = match breakdown_iteration(column_index + 1, &record) {
                Ok(iteration) => iteration,
                Err(reason) => return Err(refusal(reason, record)),
            };
            record.iterations.push(iteration);
            observer.observe(iteration);
            return Err(refusal(reason, record));
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
        let coefficients = match projected_coefficients(&columns, &transformed_rhs) {
            Ok(coefficients) => coefficients,
            Err(reason) => return Err(refusal(reason, record)),
        };
        let absolute_residual =
            true_residual_norm(right_hand_side, &operator_images, &coefficients);
        let iteration = match completed_iteration(column_index + 1, absolute_residual, &record) {
            Ok(iteration) => iteration,
            Err(reason) => return Err(refusal(reason, record)),
        };
        record.iterations.push(iteration);
        observer.observe(iteration);
        let converged = absolute_residual <= options.relative_tolerance * right_hand_side_norm;
        let exhausted = column_index + 1 == budget || column_index + 1 == dimension;
        if converged || exhausted || subdiagonal == 0.0 {
            let status = if converged {
                GmresStatus::Converged
            } else if subdiagonal == 0.0 {
                GmresStatus::Stagnated
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

fn block_records(states: &[BlockState], primary: usize) -> (KrylovRecord, Vec<KrylovRecord>) {
    let record = states[primary].record.clone();
    let sibling_records = states
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != primary)
        .map(|(_, state)| state.record.clone())
        .collect();
    (record, sibling_records)
}

fn block_refusal<OE, PE>(
    reason: KrylovRefusal,
    states: &[BlockState],
    primary: usize,
) -> BlockGmresFailure<OE, PE> {
    let (record, sibling_records) = block_records(states, primary);
    GmresFailure::BlockRefused {
        reason,
        record,
        sibling_records,
    }
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
) -> Result<Vec<GmresRun>, BlockGmresFailure<O::Error, P::Error>>
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
        if !finite(right_hand_side) || !right_hand_side_norm.is_finite() {
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
        for (index, state) in states.iter().enumerate() {
            if state.run.is_none() {
                return Err(block_refusal(
                    KrylovRefusal::ZeroIterationBudget,
                    &states,
                    index,
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
                let (record, sibling_records) = block_records(&states, index);
                return Err(GmresFailure::BlockPreconditioner {
                    source,
                    record,
                    sibling_records,
                });
            }
            state.record.preconditioner_applications += 1;
            if !finite(&directions[begin..end]) {
                return Err(block_refusal(KrylovRefusal::NonFinite, &states, index));
            }
        }
        let mut images = vec![0.0; directions.len()];
        if let Err(source) = operator.apply_batch(&directions, &mut images, active.len()) {
            for index in active.iter().take(source.completed()).copied() {
                states[index].record.operator_applications += 1;
                states[index].record.operator_rounds += 1;
            }
            let Some(first) = active.first() else {
                return Err(refusal(KrylovRefusal::Dimension, KrylovRecord::new(0.0)));
            };
            let (record, sibling_records) = block_records(&states, *first);
            return Err(GmresFailure::BlockOperator {
                source,
                record,
                sibling_records,
            });
        }
        for index in active.iter().copied() {
            states[index].record.operator_applications += 1;
            states[index].record.operator_rounds += 1;
        }
        for (block, index) in active.iter().copied().enumerate() {
            let state = &mut states[index];
            let begin = block * dimension;
            let end = begin + dimension;
            if !finite(&images[begin..end]) {
                return Err(block_refusal(KrylovRefusal::NonFinite, &states, index));
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
            // A non-finite subdiagonal makes the rotation norm non-finite, so
            // the breakdown test below is also the finiteness test for Arnoldi.
            let rotation_norm = hessenberg[column_index].hypot(hessenberg[column_index + 1]);
            if !rotation_norm.is_finite() || rotation_norm == 0.0 {
                let reason = breakdown_reason(rotation_norm);
                let iteration = match breakdown_iteration(column_index + 1, &state.record) {
                    Ok(iteration) => iteration,
                    Err(reason) => return Err(block_refusal(reason, &states, index)),
                };
                state.record.iterations.push(iteration);
                observers[index].observe(iteration);
                return Err(block_refusal(reason, &states, index));
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
            let coefficients = match projected_coefficients(&state.columns, &state.transformed_rhs)
            {
                Ok(coefficients) => coefficients,
                Err(reason) => return Err(block_refusal(reason, &states, index)),
            };
            let absolute_residual = true_residual_norm(
                &right_hand_sides[index],
                &state.operator_images,
                &coefficients,
            );
            let iteration =
                match completed_iteration(column_index + 1, absolute_residual, &state.record) {
                    Ok(iteration) => iteration,
                    Err(reason) => return Err(block_refusal(reason, &states, index)),
                };
            state.record.iterations.push(iteration);
            observers[index].observe(iteration);
            let converged =
                absolute_residual <= options.relative_tolerance * state.record.right_hand_side_norm;
            let exhausted = column_index + 1 == budget || column_index + 1 == dimension;
            let stagnated = subdiagonal == 0.0;
            if converged || exhausted || stagnated {
                let answer =
                    linear_combination(&coefficients, &state.preconditioned_basis, dimension);
                if !finite(&answer) {
                    return Err(block_refusal(KrylovRefusal::NonFinite, &states, index));
                }
                state.run = Some(GmresRun {
                    solution: answer,
                    status: if converged {
                        GmresStatus::Converged
                    } else if stagnated {
                        GmresStatus::Stagnated
                    } else {
                        GmresStatus::Exhausted
                    },
                    record: state.record.clone(),
                });
            } else {
                state
                    .basis
                    .push(image.iter().map(|value| value / subdiagonal).collect());
            }
        }
    }
    if let Some(index) = states.iter().position(|state| state.run.is_none()) {
        return Err(block_refusal(
            KrylovRefusal::SingularProjectedSystem,
            &states,
            index,
        ));
    }
    Ok(states.into_iter().filter_map(|state| state.run).collect())
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
        ) -> Result<(), BatchOperatorError<Self::Error>> {
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
    fn telemetry_streams_non_null_norms_and_preconditioning_reduces_iterations()
    -> Result<(), &'static str> {
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
            Err(_) => return Err("plain diagonal solve refused"),
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
            Err(_) => return Err("preconditioned diagonal solve refused"),
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
        Ok(())
    }

    #[test]
    fn block_and_serial_runs_are_identical_and_batch_directions_share_rounds()
    -> Result<(), &'static str> {
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
                Err(_) => return Err("serial solve refused"),
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
            Err(_) => return Err("block solve refused"),
        };
        assert_eq!(block_runs, serial_runs);
        assert_eq!(block_operator.scalar_calls, 0);
        assert!(block_operator.batch_calls < serial_operator.scalar_calls);
        Ok(())
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
    fn a_non_finite_preconditioner_is_a_typed_refusal_with_a_record() -> Result<(), &'static str> {
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
            _ => return Err("NaN preconditioner did not produce a typed refusal"),
        }
        Ok(())
    }

    #[test]
    fn a_small_givens_estimate_does_not_hide_a_large_true_residual() -> Result<(), &'static str> {
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
            Err(_) => return Err("non-normal solve refused"),
        };
        let residual = match run.record.iterations.last() {
            Some(iteration) => iteration.absolute_residual,
            None => return Err("non-normal solve returned no iteration"),
        };
        assert_eq!(run.status, GmresStatus::Exhausted);
        assert!(residual > 1e-5 && residual < 1e-3);
        Ok(())
    }

    #[test]
    fn zero_budget_and_a_zero_arnoldi_column_have_distinct_records() -> Result<(), &'static str> {
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
            _ => return Err("zero budget did not have its own refusal"),
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
            _ => return Err("zero Arnoldi column lost its iteration record"),
        }
        Ok(())
    }

    #[test]
    fn default_batch_and_identity_report_shape_errors() {
        let mut operator =
            |input: &[f64], output: &mut [f64]| -> Result<(), core::convert::Infallible> {
                output.copy_from_slice(input);
                Ok(())
            };
        let mut short_output = [0.0; 2];
        let batch = operator.apply_batch(&[1.0, 2.0, 3.0, 4.0], &mut short_output, 2);
        assert!(matches!(
            batch,
            Err(BatchOperatorError::Dimension {
                inputs: 4,
                outputs: 2,
                vectors: 2
            })
        ));

        let identity = Identity.apply(&[1.0, 2.0], &mut [0.0]);
        assert_eq!(
            identity,
            Err(IdentityError {
                input: 2,
                output: 1
            })
        );
    }

    struct FailingBatch;

    impl LinearOperator for FailingBatch {
        type Error = &'static str;

        fn apply(&mut self, _input: &[f64], _output: &mut [f64]) -> Result<(), Self::Error> {
            Err("scalar")
        }

        fn apply_batch(
            &mut self,
            _inputs: &[f64],
            _outputs: &mut [f64],
            _vectors: usize,
        ) -> Result<(), BatchOperatorError<Self::Error>> {
            Err(BatchOperatorError::Operator {
                source: "batch",
                completed: 0,
            })
        }
    }

    #[test]
    fn block_operator_failure_carries_every_active_record() -> Result<(), &'static str> {
        let failure = gmres_right_block(
            &mut FailingBatch,
            &mut [Identity, Identity],
            &[vec![3.0, 4.0], vec![1.0, 0.0]],
            GmresOptions::unrestarted(1e-8),
            &mut [NoObserver, NoObserver],
        );
        match failure {
            Err(GmresFailure::BlockOperator {
                source,
                record,
                sibling_records,
            }) => {
                assert_eq!(
                    source,
                    BatchOperatorError::Operator {
                        source: "batch",
                        completed: 0
                    }
                );
                assert_eq!(record.right_hand_side_norm, 5.0);
                assert_eq!(record.preconditioner_applications, 1);
                assert_eq!(sibling_records.len(), 1);
                assert_eq!(sibling_records[0].right_hand_side_norm, 1.0);
                assert_eq!(sibling_records[0].preconditioner_applications, 1);
            }
            _ => return Err("block failure dropped active records"),
        }
        Ok(())
    }

    struct FailsSecondSerial {
        calls: usize,
    }

    impl LinearOperator for FailsSecondSerial {
        type Error = &'static str;

        fn apply(&mut self, input: &[f64], output: &mut [f64]) -> Result<(), Self::Error> {
            self.calls += 1;
            if self.calls == 2 {
                return Err("second direction");
            }
            output.copy_from_slice(input);
            Ok(())
        }
    }

    #[test]
    fn failed_default_batch_counts_completed_operator_calls() -> Result<(), &'static str> {
        let failure = gmres_right_block(
            &mut FailsSecondSerial { calls: 0 },
            &mut [Identity, Identity],
            &[vec![1.0, 0.0], vec![0.0, 1.0]],
            GmresOptions::unrestarted(1e-12),
            &mut [NoObserver, NoObserver],
        );
        match failure {
            Err(GmresFailure::BlockOperator {
                source:
                    BatchOperatorError::Operator {
                        source: "second direction",
                        completed: 1,
                    },
                record,
                sibling_records,
            }) => {
                assert_eq!(record.operator_applications, 1);
                assert_eq!(sibling_records.len(), 1);
                assert_eq!(sibling_records[0].operator_applications, 0);
            }
            _ => return Err("failed default batch lost completed operator count"),
        }
        Ok(())
    }

    #[test]
    fn happy_breakdown_with_roundoff_is_stagnated() -> Result<(), &'static str> {
        let mut operator =
            |input: &[f64], output: &mut [f64]| -> Result<(), core::convert::Infallible> {
                for (slot, value) in output.iter_mut().zip(input) {
                    *slot = 0.3 * value;
                }
                Ok(())
            };
        let run = gmres_right(
            &mut operator,
            &mut Identity,
            &[1.0, 2.0],
            GmresOptions::unrestarted(1e-16),
            &mut NoObserver,
        )
        .map_err(|_| "happy breakdown was refused")?;
        assert_eq!(run.status, GmresStatus::Stagnated);
        assert_eq!(run.record.iterations.len(), 1);
        assert!(run.record.iterations[0].relative_residual > 1e-16);
        Ok(())
    }

    #[test]
    fn singular_later_column_keeps_the_current_residual() -> Result<(), &'static str> {
        let mut operator =
            |_input: &[f64], output: &mut [f64]| -> Result<(), core::convert::Infallible> {
                output.copy_from_slice(&[1.0, 1.0]);
                Ok(())
            };
        let failure = gmres_right(
            &mut operator,
            &mut Identity,
            &[1.0, 0.0],
            GmresOptions::unrestarted(1e-16),
            &mut NoObserver,
        );
        let record = match failure {
            Err(failure) => failure.record().clone(),
            Ok(_) => return Err("singular later column unexpectedly succeeded"),
        };
        assert_eq!(record.iterations.len(), 2);
        assert_eq!(
            record.iterations[1].absolute_residual,
            record.iterations[0].absolute_residual
        );
        assert_ne!(record.iterations[1].absolute_residual, 1.0);
        Ok(())
    }

    #[test]
    fn scaled_norm_accepts_small_and_large_finite_right_hand_sides() -> Result<(), &'static str> {
        let mut operator =
            |input: &[f64], output: &mut [f64]| -> Result<(), core::convert::Infallible> {
                output.copy_from_slice(input);
                Ok(())
            };
        let run = gmres_right(
            &mut operator,
            &mut Identity,
            &[1e160, 1e160],
            GmresOptions::unrestarted(1e-12),
            &mut NoObserver,
        )
        .map_err(|_| "large finite identity solve refused")?;
        assert!(run.record.right_hand_side_norm.is_finite());
        assert_eq!(run.status, GmresStatus::Converged);
        let small = gmres_right(
            &mut operator,
            &mut Identity,
            &[1e-170, 1e-170],
            GmresOptions::unrestarted(1e-12),
            &mut NoObserver,
        )
        .map_err(|_| "small finite identity solve refused")?;
        assert!(small.record.right_hand_side_norm > 0.0);
        assert!(small.record.right_hand_side_norm.is_finite());
        assert_eq!(small.status, GmresStatus::Converged);
        let overflow = gmres_right(
            &mut operator,
            &mut Identity,
            &[f64::MAX, f64::MAX],
            GmresOptions::unrestarted(1e-12),
            &mut NoObserver,
        );
        assert!(matches!(
            overflow,
            Err(GmresFailure::Refused {
                reason: KrylovRefusal::NonFinite,
                ..
            })
        ));
        Ok(())
    }

    struct FailOnSecondBatch {
        calls: usize,
    }

    impl LinearOperator for FailOnSecondBatch {
        type Error = &'static str;

        fn apply(&mut self, input: &[f64], output: &mut [f64]) -> Result<(), Self::Error> {
            output.copy_from_slice(input);
            Ok(())
        }

        fn apply_batch(
            &mut self,
            inputs: &[f64],
            outputs: &mut [f64],
            vectors: usize,
        ) -> Result<(), BatchOperatorError<Self::Error>> {
            self.calls += 1;
            if self.calls == 2 {
                return Err(BatchOperatorError::Operator {
                    source: "second batch",
                    completed: 0,
                });
            }
            let width = inputs.len() / vectors;
            outputs[..width].copy_from_slice(&inputs[..width]);
            for index in width..inputs.len() {
                let coordinate = index % width;
                outputs[index] = inputs[index] * (coordinate + 1) as f64;
            }
            Ok(())
        }
    }

    #[test]
    fn block_failure_keeps_an_already_converged_record() -> Result<(), &'static str> {
        let failure = gmres_right_block(
            &mut FailOnSecondBatch { calls: 0 },
            &mut [Identity, Identity],
            &[vec![1.0, 0.0], vec![1.0, 1.0]],
            GmresOptions::unrestarted(1e-12),
            &mut [NoObserver, NoObserver],
        );
        match failure {
            Err(GmresFailure::BlockOperator {
                record,
                sibling_records,
                ..
            }) => {
                assert_eq!(record.iterations.len(), 1);
                assert_eq!(sibling_records.len(), 1);
                assert_eq!(sibling_records[0].iterations.len(), 1);
                assert_eq!(sibling_records[0].iterations[0].absolute_residual, 0.0);
            }
            _ => return Err("block operator failure lost a completed record"),
        }
        Ok(())
    }

    struct OptionalFailure(bool);

    impl Preconditioner for OptionalFailure {
        type Error = &'static str;

        fn apply(&mut self, input: &[f64], output: &mut [f64]) -> Result<(), Self::Error> {
            if self.0 {
                return Err("preconditioner");
            }
            output.copy_from_slice(input);
            Ok(())
        }
    }

    #[test]
    fn block_preconditioner_failure_keeps_every_slot_record() -> Result<(), &'static str> {
        let mut operator = Diagonal {
            values: vec![1.0, 2.0],
            scalar_calls: 0,
            batch_calls: 0,
        };
        let failure = gmres_right_block(
            &mut operator,
            &mut [OptionalFailure(false), OptionalFailure(true)],
            &[vec![1.0, 1.0], vec![2.0, 1.0]],
            GmresOptions::unrestarted(1e-12),
            &mut [NoObserver, NoObserver],
        );
        match failure {
            Err(GmresFailure::BlockPreconditioner {
                record,
                sibling_records,
                ..
            }) => {
                assert_eq!(record.preconditioner_applications, 0);
                assert_eq!(sibling_records.len(), 1);
                assert_eq!(sibling_records[0].preconditioner_applications, 1);
            }
            _ => return Err("block preconditioner failure lost sibling records"),
        }
        Ok(())
    }

    struct TinyOperator;

    impl LinearOperator for TinyOperator {
        type Error = core::convert::Infallible;

        fn apply(&mut self, input: &[f64], output: &mut [f64]) -> Result<(), Self::Error> {
            for (slot, value) in output.iter_mut().zip(input) {
                *slot = 3e-310 * value;
            }
            Ok(())
        }
    }

    struct Amplify;

    impl Preconditioner for Amplify {
        type Error = core::convert::Infallible;

        fn apply(&mut self, input: &[f64], output: &mut [f64]) -> Result<(), Self::Error> {
            for (slot, value) in output.iter_mut().zip(input) {
                *slot = 1e10 * value;
            }
            Ok(())
        }
    }

    #[test]
    fn stagnation_with_a_non_finite_iterate_is_refused_by_scalar_and_block()
    -> Result<(), &'static str> {
        let options = GmresOptions::unrestarted(1e-16);
        match gmres_right(
            &mut TinyOperator,
            &mut Amplify,
            &[1.0, 1.0],
            options,
            &mut NoObserver,
        ) {
            Err(GmresFailure::Refused {
                reason: KrylovRefusal::NonFinite,
                record,
            }) => {
                assert_eq!(record.iterations.len(), 1);
                assert!(record.iterations[0].relative_residual > options.relative_tolerance);
            }
            _ => return Err("scalar path did not refuse a non-finite iterate"),
        }
        match gmres_right_block(
            &mut TinyOperator,
            &mut [Amplify, Amplify],
            &[vec![1.0, 1.0], vec![2.0, 2.0]],
            options,
            &mut [NoObserver, NoObserver],
        ) {
            Err(GmresFailure::BlockRefused {
                reason: KrylovRefusal::NonFinite,
                record,
                sibling_records,
            }) => {
                assert_eq!(record.iterations.len(), 1);
                assert!(record.iterations[0].relative_residual > options.relative_tolerance);
                assert_eq!(sibling_records.len(), 1);
            }
            Ok(runs) => {
                return if runs.iter().any(|run| !finite(&run.solution)) {
                    Err("block path returned a non-finite solution")
                } else {
                    Err("block path accepted a system the scalar path refused")
                };
            }
            Err(_) => return Err("block path refused with the wrong error"),
        }
        Ok(())
    }

    #[test]
    fn finite_block_stagnation_is_still_reported() -> Result<(), &'static str> {
        let mut operator = Diagonal {
            values: vec![0.3, 0.3],
            scalar_calls: 0,
            batch_calls: 0,
        };
        let runs = gmres_right_block(
            &mut operator,
            &mut [Identity],
            &[vec![1.0, 2.0]],
            GmresOptions::unrestarted(1e-16),
            &mut [NoObserver],
        )
        .map_err(|_| "finite stagnation was refused")?;
        assert_eq!(runs[0].status, GmresStatus::Stagnated);
        assert!(finite(&runs[0].solution));
        Ok(())
    }

    #[test]
    fn norm_propagates_nan_and_infinity() {
        assert!(norm(&[f64::NAN]).is_nan());
        assert!(norm(&[0.0, f64::NAN]).is_nan());
        assert!(norm(&[1.0, f64::NAN, 3.0]).is_nan());
        assert!(norm(&[f64::INFINITY, f64::NAN]).is_nan());
        assert_eq!(norm(&[0.0, f64::NEG_INFINITY]), f64::INFINITY);
        assert_eq!(norm(&[3.0, -4.0]), 5.0);
        assert_eq!(norm(&[]), 0.0);
    }

    #[test]
    fn projected_coefficients_refuses_non_finite_quotients() {
        assert_eq!(
            projected_coefficients(&[vec![1e-300]], &[1e300, 0.0]),
            Err(KrylovRefusal::NonFinite)
        );
        assert_eq!(
            projected_coefficients(&[vec![2.0]], &[f64::NAN, 0.0]),
            Err(KrylovRefusal::NonFinite)
        );
        assert_eq!(
            projected_coefficients(&[vec![f64::NAN]], &[1.0, 0.0]),
            Err(KrylovRefusal::NonFinite)
        );
        assert_eq!(
            projected_coefficients(&[vec![0.0]], &[1.0, 0.0]),
            Err(KrylovRefusal::SingularProjectedSystem)
        );
        assert_eq!(
            projected_coefficients(&[vec![2.0], vec![1.0, 4.0]], &[6.0, 8.0, 0.0]),
            Ok(vec![2.0, 2.0])
        );
    }

    #[test]
    fn an_overflowing_coefficient_is_refused_before_any_iteration_is_observed()
    -> Result<(), &'static str> {
        let tiny_image =
            |input: &[f64], output: &mut [f64]| -> Result<(), core::convert::Infallible> {
                for (slot, value) in output.iter_mut().zip(input) {
                    *slot = value * 1e-200;
                }
                Ok(())
            };
        let right_hand_side = vec![1e200, 0.0];
        let mut streamed = Vec::new();
        let scalar = gmres_right(
            &mut { tiny_image },
            &mut Identity,
            &right_hand_side,
            GmresOptions::unrestarted(1e-8),
            &mut |iteration| streamed.push(iteration),
        );
        match scalar {
            Err(GmresFailure::Refused { reason, record }) => {
                assert_eq!(reason, KrylovRefusal::NonFinite);
                assert!(record.iterations.is_empty());
            }
            _ => return Err("overflowing coefficient was not refused"),
        }
        assert!(streamed.is_empty());

        let block = gmres_right_block(
            &mut { tiny_image },
            &mut [Identity],
            &[right_hand_side],
            GmresOptions::unrestarted(1e-8),
            &mut [NoObserver],
        );
        match block {
            Err(GmresFailure::BlockRefused { reason, record, .. }) => {
                assert_eq!(reason, KrylovRefusal::NonFinite);
                assert!(record.iterations.is_empty());
            }
            _ => return Err("block overflow was not refused"),
        }
        Ok(())
    }

    struct Degrading {
        dimension: usize,
        perturbation: f64,
        working_calls: usize,
        calls: usize,
    }

    impl LinearOperator for Degrading {
        type Error = core::convert::Infallible;

        fn apply(&mut self, input: &[f64], output: &mut [f64]) -> Result<(), Self::Error> {
            self.calls += 1;
            for row in 0..self.dimension {
                output[row] = if self.calls > self.working_calls {
                    0.0
                } else {
                    0.3 * input[row]
                        + self.perturbation * input[(row + 1) % self.dimension] * (row as f64 + 1.0)
                };
            }
            Ok(())
        }
    }

    fn degrading(working_calls: usize) -> Degrading {
        Degrading {
            dimension: 4,
            perturbation: 1e-12,
            working_calls,
            calls: 0,
        }
    }

    #[test]
    fn breakdown_reports_the_true_residual_of_the_unchanged_iterate() -> Result<(), &'static str> {
        let right_hand_side = vec![1.0, 2.0, 3.0, 4.0];
        let options = GmresOptions::unrestarted(1e-30);
        let reference = gmres_right(
            &mut degrading(usize::MAX),
            &mut Identity,
            &right_hand_side,
            GmresOptions {
                max_iterations: Some(2),
                ..options
            },
            &mut NoObserver,
        )
        .map_err(|_| "two-iteration reference solve refused")?;
        let true_residual = match reference.record.iterations.last() {
            Some(iteration) => iteration.absolute_residual,
            None => return Err("reference solve has no iterations"),
        };
        assert!(true_residual > 1e-19);

        let mut streamed = Vec::new();
        let scalar = gmres_right(
            &mut degrading(2),
            &mut Identity,
            &right_hand_side,
            options,
            &mut |iteration| streamed.push(iteration),
        );
        let scalar_record = match scalar {
            Err(GmresFailure::Refused { reason, record }) => {
                assert_eq!(reason, KrylovRefusal::SingularProjectedSystem);
                record
            }
            _ => return Err("scalar breakdown was not a singular refusal"),
        };
        assert_eq!(scalar_record.iterations.len(), 3);
        assert_eq!(streamed, scalar_record.iterations);
        let broken = scalar_record.iterations[2];
        assert_eq!(broken.absolute_residual, true_residual);
        assert_eq!(
            broken.relative_residual,
            true_residual / scalar_record.right_hand_side_norm
        );

        let block = gmres_right_block(
            &mut degrading(2),
            &mut [Identity],
            &[right_hand_side],
            options,
            &mut [NoObserver],
        );
        match block {
            Err(GmresFailure::BlockRefused { reason, record, .. }) => {
                assert_eq!(reason, KrylovRefusal::SingularProjectedSystem);
                assert_eq!(record, scalar_record);
            }
            _ => return Err("block breakdown was not a singular refusal"),
        }
        Ok(())
    }

    #[test]
    fn a_non_finite_rotation_is_not_a_singular_system() -> Result<(), &'static str> {
        let overflow = |_: &[f64], output: &mut [f64]| -> Result<(), core::convert::Infallible> {
            output.fill(f64::MAX);
            Ok(())
        };
        let right_hand_side = vec![1.0, 0.0];
        let options = GmresOptions::unrestarted(1e-8);
        let mut streamed = Vec::new();
        let scalar = gmres_right(
            &mut { overflow },
            &mut Identity,
            &right_hand_side,
            options,
            &mut |iteration| streamed.push(iteration),
        );
        let scalar_record = match scalar {
            Err(GmresFailure::Refused { reason, record }) => {
                assert_eq!(reason, KrylovRefusal::NonFinite);
                record
            }
            _ => return Err("overflowing rotation was not refused as non-finite"),
        };
        assert_eq!(scalar_record.iterations.len(), 1);
        assert_eq!(streamed, scalar_record.iterations);
        assert_eq!(scalar_record.iterations[0].absolute_residual, 1.0);
        assert_eq!(scalar_record.iterations[0].relative_residual, 1.0);

        let block = gmres_right_block(
            &mut { overflow },
            &mut [Identity],
            &[right_hand_side],
            options,
            &mut [NoObserver],
        );
        match block {
            Err(GmresFailure::BlockRefused { reason, record, .. }) => {
                assert_eq!(reason, KrylovRefusal::NonFinite);
                assert_eq!(record, scalar_record);
            }
            _ => return Err("block overflowing rotation was not non-finite"),
        }
        Ok(())
    }

    #[test]
    fn a_residual_ratio_that_overflows_is_refused() {
        let record = KrylovRecord::new(1e-300);
        assert_eq!(
            completed_iteration(1, 1e300, &record),
            Err(KrylovRefusal::NonFinite)
        );
        assert_eq!(
            completed_iteration(1, f64::NAN, &record),
            Err(KrylovRefusal::NonFinite)
        );
        let accepted = completed_iteration(3, 2e-300, &record);
        assert_eq!(
            accepted.map(|row| (row.iteration, row.relative_residual)),
            Ok((3, 2.0))
        );
    }

    struct Scale(f64);

    impl Preconditioner for Scale {
        type Error = core::convert::Infallible;

        fn apply(&mut self, input: &[f64], output: &mut [f64]) -> Result<(), Self::Error> {
            for (slot, value) in output.iter_mut().zip(input) {
                *slot = self.0 * value;
            }
            Ok(())
        }
    }

    #[test]
    fn block_stagnation_refuses_a_non_finite_solution_like_the_scalar_solver()
    -> Result<(), &'static str> {
        let shrink = |input: &[f64], output: &mut [f64]| -> Result<(), core::convert::Infallible> {
            for (slot, value) in output.iter_mut().zip(input) {
                *slot = value / 1e300 / 2.9e11;
            }
            Ok(())
        };
        let right_hand_side = vec![1.0, 0.0, 0.0];
        let options = GmresOptions::unrestarted(f64::MIN_POSITIVE);
        let scalar = gmres_right(
            &mut { shrink },
            &mut Scale(1e308),
            &right_hand_side,
            options,
            &mut NoObserver,
        );
        let scalar_record = match scalar {
            Err(GmresFailure::Refused { reason, record }) => {
                assert_eq!(reason, KrylovRefusal::NonFinite);
                record
            }
            Ok(run) => {
                assert_eq!(run.status, GmresStatus::Stagnated);
                return Err("scalar stagnation returned a non-finite solution");
            }
            _ => return Err("scalar stagnation refused for the wrong reason"),
        };
        assert_eq!(scalar_record.iterations.len(), 1);

        let block = gmres_right_block(
            &mut { shrink },
            &mut [Scale(1e308)],
            &[right_hand_side],
            options,
            &mut [NoObserver],
        );
        match block {
            Err(GmresFailure::BlockRefused { reason, record, .. }) => {
                assert_eq!(reason, KrylovRefusal::NonFinite);
                assert_eq!(record, scalar_record);
            }
            Ok(runs) => {
                assert!(runs[0].solution.iter().any(|value| !value.is_finite()));
                return Err("block stagnation returned a non-finite solution");
            }
            _ => return Err("block stagnation refused for the wrong reason"),
        }
        Ok(())
    }

    #[test]
    fn an_overflowing_projection_is_refused_as_non_finite() -> Result<(), &'static str> {
        let huge = |_: &[f64], output: &mut [f64]| -> Result<(), core::convert::Infallible> {
            output.fill(f64::MAX);
            Ok(())
        };
        let right_hand_side = vec![1.0, 1.0];
        let options = GmresOptions::unrestarted(1e-8);
        let scalar = gmres_right(
            &mut { huge },
            &mut Identity,
            &right_hand_side,
            options,
            &mut NoObserver,
        );
        match scalar {
            Err(GmresFailure::Refused { reason, .. }) => {
                assert_eq!(reason, KrylovRefusal::NonFinite);
            }
            _ => return Err("overflowing projection was not refused as non-finite"),
        }
        let block = gmres_right_block(
            &mut { huge },
            &mut [Identity],
            &[right_hand_side],
            options,
            &mut [NoObserver],
        );
        match block {
            Err(GmresFailure::BlockRefused { reason, .. }) => {
                assert_eq!(reason, KrylovRefusal::NonFinite);
            }
            _ => return Err("block overflowing projection was not non-finite"),
        }
        Ok(())
    }

    #[derive(Debug, PartialEq)]
    struct Boom;

    impl fmt::Display for Boom {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("boom")
        }
    }

    impl Error for Boom {}

    struct BoomOperator;

    impl LinearOperator for BoomOperator {
        type Error = Boom;

        fn apply(&mut self, _input: &[f64], _output: &mut [f64]) -> Result<(), Self::Error> {
            Err(Boom)
        }
    }

    #[test]
    fn the_error_types_display_and_chain_their_sources() -> Result<(), &'static str> {
        let shape: BatchOperatorError<Boom> = BatchOperatorError::Dimension {
            inputs: 4,
            outputs: 2,
            vectors: 2,
        };
        assert_eq!(
            shape.to_string(),
            "batched operator shape mismatch: 4 input entries and 2 output entries for 2 vectors"
        );
        assert!(shape.source().is_none());
        let failed = BatchOperatorError::Operator {
            source: Boom,
            completed: 1,
        };
        assert_eq!(
            failed.to_string(),
            "batched operator failed after 1 completed directions: boom"
        );
        assert_eq!(
            failed.source().map(ToString::to_string).as_deref(),
            Some("boom")
        );

        assert_eq!(
            IdentityError {
                input: 2,
                output: 1
            }
            .to_string(),
            "identity preconditioner shape mismatch: 2 input entries and 1 output entries"
        );
        assert_eq!(
            KrylovRefusal::NonFinite.to_string(),
            "a value in the solve is not finite"
        );

        let scalar = gmres_right(
            &mut BoomOperator,
            &mut Identity,
            &[1.0],
            GmresOptions::unrestarted(1e-8),
            &mut NoObserver,
        );
        let scalar = match scalar {
            Err(failure) => failure,
            Ok(_) => return Err("failing operator succeeded"),
        };
        assert_eq!(scalar.to_string(), "operator application failed: boom");
        assert_eq!(
            scalar.source().map(ToString::to_string).as_deref(),
            Some("boom")
        );

        let block = gmres_right_block(
            &mut BoomOperator,
            &mut [Identity],
            &[vec![1.0]],
            GmresOptions::unrestarted(1e-8),
            &mut [NoObserver],
        );
        let block = match block {
            Err(failure) => failure,
            Ok(_) => return Err("failing batched operator succeeded"),
        };
        assert_eq!(
            block.to_string(),
            "batched operator application failed: batched operator failed after 0 completed \
             directions: boom"
        );
        assert_eq!(
            block.source().map(ToString::to_string).as_deref(),
            Some("batched operator failed after 0 completed directions: boom")
        );

        let refused: GmresFailure<Boom, Boom> = GmresFailure::Refused {
            reason: KrylovRefusal::Tolerance,
            record: KrylovRecord::new(0.0),
        };
        assert_eq!(
            refused.to_string(),
            "GMRES refused: the tolerance is not finite and strictly positive"
        );
        let boxed: Box<dyn Error> = Box::new(refused);
        assert!(boxed.source().is_some());
        Ok(())
    }

    #[test]
    fn options_and_results_have_defaults_and_setters() {
        let options = GmresOptions::default();
        assert_eq!(options, GmresOptions::unrestarted(1e-8));
        let tuned = options.with_relative_tolerance(1e-3).with_max_iterations(5);
        assert_eq!(tuned.relative_tolerance, 1e-3);
        assert_eq!(tuned.max_iterations, Some(5));
        assert_eq!(tuned.with_max_iterations(None).max_iterations, None);

        let iteration = KrylovIteration::default()
            .with_iteration(2)
            .with_absolute_residual(0.5)
            .with_relative_residual(0.25)
            .with_operator_applications(2)
            .with_preconditioner_applications(3);
        assert_eq!(iteration.iteration, 2);
        assert_eq!(iteration.relative_residual, 0.25);
        assert_eq!(iteration.preconditioner_applications, 3);

        let record = KrylovRecord::default()
            .with_right_hand_side_norm(2.0)
            .with_iterations(vec![iteration])
            .with_operator_applications(2)
            .with_preconditioner_applications(3)
            .with_operator_rounds(1);
        assert_eq!(record.iterations, vec![iteration]);
        assert_eq!(record.operator_rounds, 1);
        assert_eq!(KrylovRecord::new(2.0).right_hand_side_norm, 2.0);
        assert_eq!(KrylovRecord::new(2.0).iterations, Vec::new());

        let run = GmresRun::default();
        assert_eq!(run.status, GmresStatus::ZeroRightHandSide);
        assert!(run.solution.is_empty());
        let run = run
            .with_solution(vec![1.0])
            .with_status(GmresStatus::Converged)
            .with_record(record.clone());
        assert_eq!(run.solution, vec![1.0]);
        assert_eq!(run.status, GmresStatus::Converged);
        assert_eq!(run.record, record);
    }
}
