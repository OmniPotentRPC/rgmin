//! ODE12r: adaptive first-order integration of the force.
//!
//! Makri, Ortner and Kermode, *A preconditioning scheme for minimum
//! energy path finding methods*, J. Chem. Phys. 150, 094109 (2019),
//! <https://doi.org/10.1063/1.5064465>, relax a band by integrating
//! `dx/dt = P^{-1} F(x)` with forward Euler and an embedded second-order
//! error estimate. `F = -∇E` (or a projected band force) and `P` is a
//! metric; the identity when none is installed. One step of length `h`
//! from `x` with `F = F(x)` and `s = P^{-1} F`:
//!
//! ```text
//! x' = x + h s
//! s' = P^{-1} F(x')
//! e  = (h / 2) ||s - s'||_P = (h / 2) sqrt((F - F') · (s - s'))
//! R  = ||F||_{P^{-1}} = sqrt(F · s),    R' = sqrt(F' · s')
//! ```
//!
//! `e` is the difference between the Euler update and the trapezoidal
//! update over the same step, the local truncation error of the
//! first-order rule. The trial is accepted when the residual falls by
//! the sufficient-decrease margin, `R' <= R (1 - c1 h)`, or when it stays
//! within the growth bound `R' <= c2 R` while the relative error `e / R`
//! is below `rtol`. An accepted step sets the next length to the smaller
//! of the error estimate `(h / 2) sqrt(rtol R / e)` and the secant
//! estimate `h (s · (s - s')) / ||s - s'||²`, kept within `[h / 4,
//! extrapolate h]`. A rejected trial keeps `x` and takes `h` to the error
//! estimate clamped to `[h / 10, h / 4]`.
//!
//! The session owns the clamps: the displacement cap shortens the move,
//! and the step length that enters the error estimate is the one the
//! move actually took, `h |Δx_taken| / |Δx_proposed|`, so a capped step
//! is judged as the step it was. The next trial starts from the length
//! the controller chose, not from the capped one.

use ndarray::Array1;

/// Session memory for one ODE12r integration.
#[derive(Clone, Debug)]
pub struct Ode12rState {
    /// Step length `h` for the next trial. Zero until the first step,
    /// which opens at `istep`.
    pub h: f64,
    /// Opening step length.
    pub h0: f64,
    /// Relative local error accepted with the growth bound.
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
    /// Relative local error `e / R` of the trial.
    pub relative_error: f64,
    /// Residual at the trial geometry.
    pub residual_new: f64,
    /// Step length for the next trial.
    pub h_next: f64,
}

impl Ode12rState {
    /// Paper parameters: `rtol = 0.1`, `c1 = 1e-2`, `c2 = 2`, growth at
    /// most threefold. `istep` opens the integration.
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
            extrapolate: 3.0,
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
    /// the move actually took, `residual` is `||F||_{P^{-1}}` at `x`.
    pub fn judge(
        &mut self,
        residual: f64,
        force: &Array1<f64>,
        steepest: &Array1<f64>,
        force_new: &Array1<f64>,
        steepest_new: &Array1<f64>,
        h: f64,
    ) -> Ode12rVerdict {
        let residual_new = metric_norm(force_new, steepest_new);
        let y = steepest - steepest_new;
        let df = force - force_new;
        let error = 0.5 * h * crate::vecops::dot(df.view(), y.view()).max(0.0).sqrt();
        let relative = if residual > 0.0 {
            error / residual
        } else {
            0.0
        };
        let accepted = residual_new <= residual * (1.0 - self.c1 * h)
            || (residual_new <= self.c2 * residual && relative <= self.rtol);
        let h_error = if relative > f64::EPSILON {
            0.5 * h * (self.rtol / relative).sqrt()
        } else {
            self.extrapolate * h
        };
        let h_next = if accepted {
            let yy = crate::vecops::dot(y.view(), y.view());
            let h_secant = if yy > f64::EPSILON {
                h * crate::vecops::dot(steepest.view(), y.view()) / yy
            } else {
                self.extrapolate * h
            };
            let bounded = h_error
                .min(h_secant)
                .min(self.extrapolate * h)
                .max(0.25 * h);
            self.accepted += 1;
            bounded
        } else {
            self.rejected += 1;
            h_error.clamp(0.1 * h, 0.25 * h)
        };
        let h_next = if h_next.is_finite() {
            h_next.max(self.h_min)
        } else {
            self.h_min
        };
        self.h = h_next;
        Ode12rVerdict {
            accepted,
            relative_error: relative,
            residual_new,
            h_next,
        }
    }
}

/// `||F||_{P^{-1}} = sqrt(F · P^{-1} F)` from the force and its
/// preconditioned image.
#[must_use]
pub fn metric_norm(force: &Array1<f64>, steepest: &Array1<f64>) -> f64 {
    crate::vecops::dot(force.view(), steepest.view())
        .max(0.0)
        .sqrt()
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
    fn an_exact_euler_step_on_a_linear_force_has_no_error_only_when_the_force_is_constant() {
        // Constant force: Euler and trapezoid agree, so the relative
        // error is zero and the step grows by the extrapolation bound.
        let mut state = Ode12rState::new(0.5);
        let f = array![1.0, -2.0];
        let residual = metric_norm(&f, &f);
        let verdict = state.judge(residual, &f, &f, &f, &f, 0.5);
        assert!(verdict.accepted);
        assert_eq!(verdict.relative_error, 0.0);
        assert!((verdict.h_next - 1.5).abs() < 1e-12);
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
        let residual = metric_norm(&f, &f);
        let verdict = state.judge(residual, &f, &f, &f_new, &f_new, h);
        assert!(!verdict.accepted);
        assert!(verdict.h_next < 0.25 * h + 1e-15);
        assert!(verdict.h_next >= 0.1 * h - 1e-15);
        assert_eq!(state.rejected, 1);
    }

    #[test]
    fn the_secant_estimate_lands_on_the_minimum_of_a_quadratic() {
        // On E = k x^2 / 2 the secant step h (s.y)/(y.y) is 1/k, the exact
        // Newton length, whichever h was tried (short of the error bound).
        let k = array![2.0, 2.0];
        let x = array![1.0, -1.0];
        let f = quad_force(&k, &x);
        let mut state = Ode12rState::new(0.1);
        state.rtol = 10.0; // leave the choice to the secant
        let h = state.open();
        let x_new = &x + &f.mapv(|v| h * v);
        let f_new = quad_force(&k, &x_new);
        let residual = metric_norm(&f, &f);
        let verdict = state.judge(residual, &f, &f, &f_new, &f_new, h);
        assert!(verdict.accepted);
        assert!(
            (verdict.h_next - 0.3).abs() < 1e-12,
            "secant 1/k = 0.5 is capped at 3 h = 0.3"
        );
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
