//! The public GMRES surface, used only through exported names and setters.

use std::error::Error;

use rgmin::{
    GmresFailure, GmresOptions, GmresStatus, KrylovIdentity, KrylovRefusal, NoObserver,
    gmres_right, gmres_right_block,
};

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
