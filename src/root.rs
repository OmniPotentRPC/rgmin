//! Bracketed scalar roots.
//!
//! Bisection keeps a sign change of a continuous function inside the
//! bracket at every step, so it cannot leave the bracket or skip a root
//! the caller has isolated. It stops when the bracket has no floating-point
//! number strictly between its ends, or at an exact zero.

use thiserror::Error;

/// A bracketed root that could not be found.
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum RootError<E> {
    /// The ends are not finite and ordered.
    #[error("the bracket [{lower}, {upper}] is not a finite ordered interval")]
    InvalidBracket {
        /// Left end.
        lower: f64,
        /// Right end.
        upper: f64,
    },
    /// The function has the same strict sign at both ends.
    #[error("f({lower}) = {f_lower} and f({upper}) = {f_upper} do not bracket a root")]
    NotBracketed {
        /// Left end.
        lower: f64,
        /// Right end.
        upper: f64,
        /// Function value at the left end.
        f_lower: f64,
        /// Function value at the right end.
        f_upper: f64,
    },
    /// The function returned NaN or an infinity.
    #[error("the function is not finite at {at}")]
    NonFinite {
        /// Argument of the non-finite value.
        at: f64,
    },
    /// The caller's evaluator failed.
    #[error("the evaluator failed")]
    Evaluator(E),
}

/// A root of `f` in `[lower, upper]`, where `f` changes sign.
///
/// The returned point is an exact zero, or an end of the final bracket of
/// adjacent floating-point numbers, the one with the smaller `|f|`.
pub fn bisect<E, F>(mut f: F, lower: f64, upper: f64) -> Result<f64, RootError<E>>
where
    F: FnMut(f64) -> Result<f64, E>,
{
    if !(lower.is_finite() && upper.is_finite() && lower < upper) {
        return Err(RootError::InvalidBracket { lower, upper });
    }
    let mut eval = |x: f64| -> Result<f64, RootError<E>> {
        let value = f(x).map_err(RootError::Evaluator)?;
        if value.is_finite() {
            Ok(value)
        } else {
            Err(RootError::NonFinite { at: x })
        }
    };
    let (mut lo, mut hi) = (lower, upper);
    let (mut f_lo, f_hi) = (eval(lo)?, eval(hi)?);
    if f_lo == 0.0 {
        return Ok(lo);
    }
    if f_hi == 0.0 {
        return Ok(hi);
    }
    if f_lo.is_sign_positive() == f_hi.is_sign_positive() {
        return Err(RootError::NotBracketed {
            lower,
            upper,
            f_lower: f_lo,
            f_upper: f_hi,
        });
    }
    let mut f_hi = f_hi;
    loop {
        let mid = lo.midpoint(hi);
        if !(mid > lo && mid < hi) {
            return Ok(if f_lo.abs() <= f_hi.abs() { lo } else { hi });
        }
        let f_mid = eval(mid)?;
        if f_mid == 0.0 {
            return Ok(mid);
        }
        if f_mid.is_sign_positive() == f_lo.is_sign_positive() {
            lo = mid;
            f_lo = f_mid;
        } else {
            hi = mid;
            f_hi = f_mid;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bisection_reaches_adjacent_floats() -> Result<(), RootError<()>> {
        let root = bisect(|x: f64| Ok::<f64, ()>(x * x - 2.0), 0.0, 2.0)?;
        let next = f64::from_bits(root.to_bits() + 1);
        let previous = f64::from_bits(root.to_bits() - 1);
        let exact = std::f64::consts::SQRT_2;
        assert!(previous <= exact && exact <= next, "root {root}");
        Ok(())
    }

    #[test]
    fn exact_zero_at_an_end_is_returned() -> Result<(), RootError<()>> {
        assert_eq!(bisect(|x: f64| Ok::<f64, ()>(x), 0.0, 1.0)?, 0.0);
        assert_eq!(bisect(|x: f64| Ok::<f64, ()>(x - 1.0), 0.0, 1.0)?, 1.0);
        Ok(())
    }

    #[test]
    fn extreme_brackets_do_not_overflow() -> Result<(), RootError<()>> {
        assert_eq!(bisect(|x: f64| Ok::<f64, ()>(x), -f64::MAX, f64::MAX)?, 0.0);
        let root = bisect(|x: f64| Ok::<f64, ()>(0.5 * x - 5e307), -f64::MAX, f64::MAX)?;
        let next = f64::from_bits(root.to_bits() + 1);
        let previous = f64::from_bits(root.to_bits() - 1);
        assert!(previous <= 1e308 && 1e308 <= next, "root {root}");
        Ok(())
    }

    #[test]
    fn refusals_are_typed() {
        assert!(matches!(
            bisect(|x: f64| Ok::<f64, ()>(x * x + 1.0), -1.0, 1.0),
            Err(RootError::NotBracketed { .. })
        ));
        assert!(matches!(
            bisect(|x: f64| Ok::<f64, ()>(x), 1.0, 0.0),
            Err(RootError::InvalidBracket { .. })
        ));
        assert!(matches!(
            bisect(
                |x: f64| Ok::<f64, ()>(if x > 0.5 { f64::NAN } else { x - 0.75 }),
                0.0,
                1.0
            ),
            Err(RootError::NonFinite { .. })
        ));
        assert!(matches!(
            bisect(|_: f64| Err::<f64, u8>(7), 0.0, 1.0),
            Err(RootError::Evaluator(7))
        ));
    }
}
