//! Strict Armijo backtracking on a scalar merit.
//!
//! [`crate::LineSearch::Backtracking`] evaluates a gradient at every trial
//! and, if the budget runs out, keeps the lowest decrease it saw.
//! Gould, Ortner, and Packwood need a stricter loop: the merit is a
//! scalar, a refused trial is not a decrease, and the next iteration
//! opens at twice the step that was accepted.
//!
//! Gould, Ortner, and Packwood, *A dimer-type saddle search algorithm
//! with preconditioning and linesearch*, Math. Comp. (2016),
//! <https://doi.org/10.1090/mcom/3096>, Algorithm 3.

use super::conditions::armijo;

/// A trial that satisfied the Armijo inequality.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArmijoAccept {
    /// Accepted length, `alpha > 0`.
    pub alpha: f64,
    /// Merit at that length.
    pub phi: f64,
}

/// Armijo backtracking along a direction whose slope at zero is known.
///
/// The opening trial is `alpha_max` until a step has been accepted, and
/// `min(alpha_max, 2 * alpha)` after that. Algorithm 3 of Gould, Ortner,
/// and Packwood writes the same recurrence as `α := min(α_max, 2 α_{k-1})`
/// and then halves while
///
/// `φ(α) > φ(0) + c α φ'(0)`.
///
/// `slope` is `φ'(0)` and has to be negative: `p = -∇φ` gives
/// `slope = -||p||²`. A probe that returns `None` is refused, which is
/// how a caller applies a test other than sufficient decrease (the dimer
/// orientation check). Refused and insufficient trials shrink by
/// [`Self::shrink`]. The accepted length is stored in [`Self::alpha`].
///
/// Gould, Ortner, and Packwood, Math. Comp. (2016),
/// <https://doi.org/10.1090/mcom/3096>.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArmijoStepper {
    /// Armijo constant in `(0, 1)`. Their numerical section uses `√0.1`.
    pub c: f64,
    /// Shrink factor in `(0, 1)`. Algorithm 3 uses one half.
    pub shrink: f64,
    /// How many trials to make, counting the opening one.
    pub maxiter: usize,
    /// Last accepted length. Zero means no step has been accepted yet.
    pub alpha: f64,
    /// Largest trial length.
    pub alpha_max: f64,
}

impl Default for ArmijoStepper {
    fn default() -> Self {
        Self::gould()
    }
}

impl ArmijoStepper {
    /// Constants from the numerical section of Gould, Ortner, and Packwood.
    ///
    /// `c = √0.1`, shrink `1/2`, at most 40 trials, and `alpha_max = 1`.
    /// The first search opens at `alpha_max`.
    pub fn gould() -> Self {
        Self {
            c: 0.1_f64.sqrt(),
            shrink: 0.5,
            maxiter: 40,
            alpha: 0.0,
            alpha_max: 1.0,
        }
    }

    /// Length of the next opening trial.
    pub fn opening(&self) -> f64 {
        let doubled = if self.alpha > 0.0 {
            2.0 * self.alpha
        } else {
            self.alpha_max
        };
        doubled.min(self.alpha_max)
    }

    /// Accept the longest opening-or-shrunk trial that passes Armijo.
    ///
    /// `probe(alpha)` is `φ(α)`, or `None` when that trial is refused.
    /// Returns `None` when `slope` is not a strict descent, the controls
    /// are not in range, or no trial is accepted. The stored length
    /// changes only when a trial is accepted.
    pub fn search<F>(&mut self, phi0: f64, slope: f64, mut probe: F) -> Option<ArmijoAccept>
    where
        F: FnMut(f64) -> Option<f64>,
    {
        if !(phi0.is_finite() && slope.is_finite() && slope < 0.0) {
            return None;
        }
        if !(self.c.is_finite() && self.c > 0.0 && self.c < 1.0) {
            return None;
        }
        if !(self.shrink.is_finite() && self.shrink > 0.0 && self.shrink < 1.0) {
            return None;
        }
        let mut alpha = self.opening();
        if !(alpha.is_finite() && alpha > 0.0) {
            return None;
        }
        let trials = self.maxiter.max(1);
        for _ in 0..trials {
            if let Some(phi) = probe(alpha)
                && phi.is_finite()
                && armijo(phi, phi0, alpha, slope, self.c)
            {
                self.alpha = alpha;
                return Some(ArmijoAccept { alpha, phi });
            }
            alpha *= self.shrink;
            if !(alpha.is_finite() && alpha >= 1e-16) {
                break;
            }
        }
        None
    }
}
