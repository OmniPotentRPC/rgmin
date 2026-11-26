//! The public GMRES surface, used only through exported names and setters.

use std::error::Error;

use rgmin::{
    GmresFailure, GmresOptions, GmresStatus, KrylovIdentity, KrylovRefusal, NoObserver,
    gmres_right, gmres_right_block,
};

fn matvec(
    matrix: &[f64],
    n: usize,
) -> impl FnMut(&[f64], &mut [f64]) -> Result<(), std::convert::Infallible> {
    let matrix = matrix.to_vec();
    move |input, output| {
        for row in 0..n {
            output[row] = matrix[row * n..(row + 1) * n]
                .iter()
                .zip(input)
                .map(|(entry, value)| entry * value)
                .sum();
        }
        Ok(())
    }
}

fn dense_solve(matrix: &[f64], n: usize, rhs: &[f64]) -> Vec<f64> {
    let mut augmented = vec![vec![0.0; n + 1]; n];
    for row in 0..n {
        for column in 0..n {
            augmented[row][column] = matrix[row * n + column];
        }
        augmented[row][n] = rhs[row];
    }
    for pivot in 0..n {
        let mut best = pivot;
        let mut best_abs = augmented[pivot][pivot].abs();
        for (row, values) in augmented.iter().enumerate().skip(pivot + 1) {
            let value = values[pivot].abs();
            if value > best_abs {
                best = row;
                best_abs = value;
            }
        }
        if best != pivot {
            augmented.swap(best, pivot);
        }
        let diagonal = augmented[pivot][pivot];
        let (upper, lower) = augmented.split_at_mut(pivot + 1);
        let pivot_row = &upper[pivot];
        for row in lower.iter_mut().take(n - pivot - 1) {
            let scale = row[pivot] / diagonal;
            for (entry, above) in row[pivot..=n].iter_mut().zip(&pivot_row[pivot..=n]) {
                *entry -= scale * above;
            }
        }
    }
    let mut solution = vec![0.0; n];
    for row in (0..n).rev() {
        let solved: f64 = augmented[row][row + 1..n]
            .iter()
            .zip(&solution[row + 1..])
            .map(|(entry, value)| entry * value)
            .sum();
        solution[row] = (augmented[row][n] - solved) / augmented[row][row];
    }
    solution
}

fn diagonal(
    values: &'static [f64],
) -> impl FnMut(&[f64], &mut [f64]) -> Result<(), std::convert::Infallible> {
    move |input, output| {
        for ((slot, value), scale) in output.iter_mut().zip(input).zip(values) {
            *slot = scale * value;
        }
        Ok(())
    }
}

#[test]
fn a_diagonal_solve_converges_through_the_public_names() -> Result<(), Box<dyn Error>> {
    let options = GmresOptions::default().with_relative_tolerance(1e-12);
    let run = gmres_right(
        &mut diagonal(&[1.0, 2.0, 4.0]),
        &mut KrylovIdentity,
        &[1.0, 1.0, 1.0],
        options,
        &mut NoObserver,
    )?;
    assert_eq!(run.status, GmresStatus::Converged);
    assert!((run.solution[2] - 0.25).abs() < 1e-12);
    assert!(run.record.iterations.len() <= 3);
    Ok(())
}

#[test]
fn a_refusal_is_a_boxed_error_with_a_cause() {
    let refused = gmres_right_block(
        &mut diagonal(&[1.0, 2.0]),
        &mut [KrylovIdentity],
        &[vec![1.0, 1.0]],
        GmresOptions::default().with_relative_tolerance(0.0),
        &mut [NoObserver],
    );
    let failure = match refused {
        Err(failure) => failure,
        Ok(_) => panic!("a zero tolerance was accepted"),
    };
    assert!(matches!(
        failure,
        GmresFailure::Refused {
            reason: KrylovRefusal::Tolerance,
            ..
        }
    ));
    let boxed: Box<dyn Error> = Box::new(failure);
    assert_eq!(
        boxed.to_string(),
        "GMRES refused: the tolerance is not finite and strictly positive"
    );
    assert!(boxed.source().is_some());
}

#[test]
fn gmres_matches_dense_references_for_spd_and_nonsymmetric_systems() -> Result<(), Box<dyn Error>> {
    let cases: [(&[f64], &[f64], usize); 2] = [
        (
            &[4.0, 1.0, 1.0, 1.0, 3.0, 0.5, 1.0, 0.5, 2.0],
            &[1.0, 2.0, 3.0],
            3,
        ),
        (
            &[3.0, -1.0, 2.0, 2.0, 4.0, 1.0, -2.0, 1.0, 5.0],
            &[2.0, -1.0, 0.5],
            3,
        ),
    ];
    let tolerance = f64::EPSILON.sqrt() * 256.0;
    for (matrix, right_hand_side, dimension) in cases {
        let mut operator = matvec(matrix, dimension);
        let run = gmres_right(
            &mut operator,
            &mut KrylovIdentity,
            right_hand_side,
            GmresOptions::default().with_relative_tolerance(tolerance),
            &mut NoObserver,
        )?;
        assert_eq!(run.status, GmresStatus::Converged);
        let reference = dense_solve(matrix, dimension, right_hand_side);
        for (found, expected) in run.solution.iter().zip(reference) {
            assert!((found - expected).abs() <= tolerance * expected.abs().max(1.0));
        }
    }
    Ok(())
}

#[test]
fn iteration_budget_controls_unrestarted_convergence_status() -> Result<(), Box<dyn Error>> {
    let matrix = [1.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 25.0];
    let right_hand_side = [1.0, 1.0, 1.0];
    let options = GmresOptions::default()
        .with_relative_tolerance(f64::EPSILON.sqrt())
        .with_max_iterations(Some(1));
    let exhausted = gmres_right(
        &mut matvec(&matrix, 3),
        &mut KrylovIdentity,
        &right_hand_side,
        options,
        &mut NoObserver,
    )?;
    assert_eq!(exhausted.status, GmresStatus::Exhausted);
    assert_eq!(exhausted.record.iterations.len(), 1);

    let converged = gmres_right(
        &mut matvec(&matrix, 3),
        &mut KrylovIdentity,
        &right_hand_side,
        GmresOptions::default().with_relative_tolerance(f64::EPSILON.sqrt()),
        &mut NoObserver,
    )?;
    assert_eq!(converged.status, GmresStatus::Converged);
    Ok(())
}
