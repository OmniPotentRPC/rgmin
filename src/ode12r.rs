//! ODE12r: adaptive first-order integration of the force.
//!
//! Makri, Ortner and Kermode, *A preconditioning scheme for minimum
//! energy path finding methods*, J. Chem. Phys. 150, 094109 (2019),
//! <https://doi.org/10.1063/1.5064465>, Sec. II C, relax a band by
//! integrating `dx/dt = P^{-1} F(x)` with forward Euler and the error
//! estimate of the embedded second-order (trapezoidal) solution. `F =
//! -∇E` (or a projected band force) and `P` is the metric, the identity
//! when none is installed. One trial of length `h` from `x` with `F =
//! F(x)`, `s = P^{-1} F` and the same quantities primed at `x' = x + h s`:
//!
//! ```text
//! R  = ||F||_inf                       residual, Eq. (11): ||P s||_inf = ||F||_inf
//! E  = (h / 2) ||s - s'||_inf          Euler against the trapezoidal solution
//! accept when R' <= R (1 - c1 h), or R' <= c2 R and E <= rtol
//! h_ode12 = (h / 2) sqrt(rtol / E)
//! h_ls    = h (F . (s - s')) / ((F - F') . (s - s'))
//! accepted: h <- max(h / 4,  min(extrapolate h, h_ls, h_ode12))
//! rejected: h <- max(h / 10, min(h / 4,         h_ls, h_ode12))
//! ```
//!
//! `E` is the local error of the Euler step against the trapezoidal one,
//! `x + (h/2)(s + s') - x' = (h/2)(s' - s)`, in the infinity norm, as the
//! authors' ODE12r implementation in ASE (`ase.optimize.ode`) computes it
//! with `rtol` in force units; the paper's Eq. for `E` writes the same
//! difference scaled by the coordinate magnitude. `h_ls` is the length at
//! which the linear interpolant `(1 - θ) f + θ f'` of the preconditioned
//! force is shortest in the `P`-norm, `||g||_P^2 = g . P g`, which in terms
//! of `F` and `s` is the ratio above; on a quadratic surface with `P` the
//! Hessian it is the Newton length. `extrapolate` is 4, `rtol` 0.1, and
//! the contraction and growth constants `c1 = 1e-2`, `c2 = 2`.
//!
//! The session owns the clamps: the displacement cap shortens the move,
//! and the step length that enters the error estimate is the one the
//! move actually took, `h |Δx_taken| / |Δx_proposed|`, so a capped step
//! is judged as the step it was. The next trial starts from the length
//! the controller chose, floored at an accepted capped move's length so
//! that a cap the move already sits on cannot shrink the step further.

use ndarray::Array1;

/// Session memory for one ODE12r integration.
#[derive(Clone, Debug)]
pub struct Ode12rState {
    /// Step length `h` for the next trial. Zero until the first step,
    /// which opens at `h0`.
    pub h: f64,
    /// Opening step length.
    pub h0: f64,
    /// Local error accepted with the growth bound, in force units.
    pub rtol: f64,
    /// Sufficient-decrease coefficient of the residual test.
    pub c1: f64,
    /// Residual growth bound of the error-controlled acceptance.
    pub c2: f64,
    /// Largest growth factor of `h` after an accepted step.
    pub extrapolate: f64,
    /// Floor on `h`.
    pub h_min: f64,
    /// Accepted trials.
    pub accepted: usize,
    /// Rejected trials.
    pub rejected: usize,
}

/// What one trial decided.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ode12rVerdict {
    /// The trial geometry is kept.
    pub accepted: bool,
    /// Local error `E` of the trial.
    pub error: f64,
    /// Residual `||F'||_inf` at the trial geometry.
    pub residual_new: f64,
    /// Step length for the next trial.
    pub h_next: f64,
}

impl Ode12rState {
    /// Paper parameters: `rtol = 0.1`, `c1 = 1e-2`, `c2 = 2`, growth at
    /// most fourfold. `istep` opens the integration.
    #[must_use]
    pub fn new(istep: f64) -> Self {
        let h0 = if istep.is_finite() && istep > 0.0 {
            istep
        } else {
            0.1
        };
        Self {
            h: 0.0,
            h0,
            rtol: 0.1,
            c1: 1e-2,
            c2: 2.0,
            extrapolate: 4.0,
            h_min: 1e-10,
            accepted: 0,
            rejected: 0,
        }
    }

    /// Forget the step length and the counters.
    pub fn reset(&mut self) {
        self.h = 0.0;
        self.accepted = 0;
        self.rejected = 0;
    }

    /// Step length of the next trial, opening at `h0`.
    #[must_use]
    pub fn open(&mut self) -> f64 {
        if self.h <= 0.0 || !self.h.is_finite() {
            self.h = self.h0;
        }
        self.h
    }

    /// Judge the trial `x' = x + h s` and set the next step length.
    ///
    /// `force` and `steepest` are `F` and `s = P^{-1} F` at `x`,
    /// `force_new` and `steepest_new` the same at `x'`, `h` the length
    /// the move actually took.
    pub fn judge(
        &mut self,
        force: &Array1<f64>,
        steepest: &Array1<f64>,
        force_new: &Array1<f64>,
        steepest_new: &Array1<f64>,
        h: f64,
    ) -> Ode12rVerdict {
        let residual = inf_norm(force);
        let residual_new = inf_norm(force_new);
        let y = steepest - steepest_new;
        let error = local_error(h, &y);
        let accepted = residual_new <= residual * (1.0 - self.c1 * h)
            || (residual_new <= self.c2 * residual && error <= self.rtol);
        let h_ode12 = if error > f64::EPSILON {
            0.5 * h * (self.rtol / error).sqrt()
        } else {
            f64::INFINITY
        };
        let df = force - force_new;
        let denominator = crate::vecops::dot(df.view(), y.view());
        let h_ls = if denominator > f64::EPSILON {
            h * crate::vecops::dot(force.view(), y.view()) / denominator
        } else {
            f64::INFINITY
        };
        let h_ls = if h_ls.is_finite() && h_ls > 0.0 {
            h_ls
        } else {
            f64::INFINITY
        };
        let h_next = if accepted {
            self.accepted += 1;
            (self.extrapolate * h).min(h_ls).min(h_ode12).max(0.25 * h)
        } else {
            self.rejected += 1;
            (0.25 * h).min(h_ls).min(h_ode12).max(0.1 * h)
        };
        let h_next = if h_next.is_finite() {
            h_next.max(self.h_min)
        } else {
            self.h_min
        };
        self.h = h_next;
        Ode12rVerdict {
            accepted,
            error,
            residual_new,
            h_next,
        }
    }
}

/// `||v||_inf`.
#[must_use]
pub fn inf_norm(v: &Array1<f64>) -> f64 {
    v.iter().fold(0.0_f64, |m, a| m.max(a.abs()))
}

/// Local error of the Euler step against the trapezoidal one,
/// `(h / 2) ||s - s'||_inf`, from the difference `y = s - s'`.
#[must_use]
pub fn local_error(h: f64, y: &Array1<f64>) -> f64 {
    0.5 * h * inf_norm(y)
}

/// Step length the move actually took when a cap or clip shortened the
/// proposed displacement: `h |Δx_taken| / |Δx_proposed|`.
#[must_use]
pub fn taken_step(h: f64, proposed: &Array1<f64>, taken: &Array1<f64>) -> f64 {
    let p = crate::vecops::dot(proposed.view(), proposed.view()).sqrt();
    let t = crate::vecops::dot(taken.view(), taken.view()).sqrt();
    if p > 0.0 && t.is_finite() {
        h * t / p
    } else {
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn quad_force(k: &Array1<f64>, x: &Array1<f64>) -> Array1<f64> {
        -(k * x)
    }

    #[test]
    fn a_constant_force_has_no_error_and_grows_the_step_fourfold() {
        let mut state = Ode12rState::new(0.5);
        let f = array![1.0, -2.0];
        let verdict = state.judge(&f, &f, &f, &f, 0.5);
        assert!(verdict.accepted);
        assert_eq!(verdict.error, 0.0);
        assert!((verdict.h_next - 2.0).abs() < 1e-12);
    }

    #[test]
    fn a_step_that_overshoots_a_stiff_quadratic_is_rejected_and_shrinks_h() {
        let k = array![400.0];
        let x = array![0.01];
        let f = quad_force(&k, &x);
        let mut state = Ode12rState::new(0.1);
        let h = state.open();
        let x_new = &x + &f.mapv(|v| h * v); // Euler with h k = 40: overshoot
        let f_new = quad_force(&k, &x_new);
        let verdict = state.judge(&f, &f, &f_new, &f_new, h);
        assert!(!verdict.accepted);
        assert!(verdict.h_next <= 0.25 * h + 1e-15);
        assert!(verdict.h_next >= 0.1 * h - 1e-15);
        assert_eq!(state.rejected, 1);
    }

    #[test]
    fn the_line_search_candidate_is_the_newton_length_on_a_quadratic() {
        // On E = k x^2 / 2 the interpolant of the force vanishes at
        // theta h = 1 / k whichever h was tried; with a loose rtol the
        // extrapolation bound 4 h = 0.4 wins over 1 / k = 0.5.
        let k = array![2.0, 2.0];
        let x = array![1.0, -1.0];
        let f = quad_force(&k, &x);
        let mut state = Ode12rState::new(0.1);
        state.rtol = 1e3;
        let h = state.open();
        let x_new = &x + &f.mapv(|v| h * v);
        let f_new = quad_force(&k, &x_new);
        let verdict = state.judge(&f, &f, &f_new, &f_new, h);
        assert!(verdict.accepted);
        assert!((verdict.h_next - 0.4).abs() < 1e-12, "{}", verdict.h_next);
        // A looser growth bound exposes the line-search length.
        let mut state = Ode12rState::new(0.2);
        state.rtol = 1e3;
        state.extrapolate = 10.0;
        let h = state.open();
        let x_new = &x + &f.mapv(|v| h * v);
        let f_new = quad_force(&k, &x_new);
        let verdict = state.judge(&f, &f, &f_new, &f_new, h);
        assert!((verdict.h_next - 0.5).abs() < 1e-12, "{}", verdict.h_next);
    }

    #[test]
    fn the_local_error_is_half_the_step_times_the_largest_force_change() {
        let y = array![0.2, -0.6, 0.1];
        assert!((local_error(0.5, &y) - 0.15).abs() < 1e-15);
        assert_eq!(local_error(0.5, &array![0.0, 0.0]), 0.0);
    }

    #[test]
    fn a_capped_move_is_judged_at_the_length_it_took() {
        let proposed = array![1.0, 0.0];
        let taken = array![0.25, 0.0];
        assert!((taken_step(0.8, &proposed, &taken) - 0.2).abs() < 1e-15);
        assert_eq!(taken_step(0.8, &array![0.0, 0.0], &taken), 0.8);
    }

    #[test]
    fn reset_forgets_the_step_and_the_counters() {
        let mut state = Ode12rState::new(0.1);
        let _ = state.open();
        state.accepted = 3;
        state.reset();
        assert_eq!(state.h, 0.0);
        assert_eq!(state.accepted, 0);
        assert_eq!(state.open(), 0.1);
    }
}
