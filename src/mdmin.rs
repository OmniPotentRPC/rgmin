//! MDMin: velocity Verlet with the velocity reset onto the force.
//!
//! ASE `ase.optimize.mdmin.MDMin` (3.29) integrates `F = -∇E`. The first
//! call starts from a zero velocity and takes only the second half-kick.
//! Later calls half-kick, replace the velocity by its projection onto `F`
//! when that projection is uphill, half-kick again, and advance
//! `Δx = dt v`. A per-atom displacement longer than `maxstep` is scaled
//! by `maxstep / (offset + longest)` and clipped into `[0, 1]`. The
//! velocity is left at the value that produced `Δx`, including when the
//! displacement was scaled.
//!
//! The session owns the bounds clip. This stepper does not apply the
//! generic per-atom cap a second time: `maxstep` is that cap, copied
//! onto the state when the session sets it.

use ndarray::Array1;

/// ASE's default time step.
pub const ASE_DT: f64 = 0.2;
/// ASE's default per-atom displacement cap.
pub const ASE_MAXSTEP: f64 = 0.2;
/// Offset in the displacement scale, so a zero move does not divide by zero.
pub const SCALE_OFFSET: f64 = 1.0 / 1_000_000.0;

/// Velocity and step controls for one MDMin integration.
#[derive(Clone, Debug)]
pub struct MdMinState {
    /// Velocity. Zero until the first kick.
    pub vel: Array1<f64>,
    /// Time step. A non-positive or non-finite constructor argument
    /// selects [`ASE_DT`].
    pub dt: f64,
    /// Per-atom cap. `None` leaves the displacement unscaled.
    pub maxstep: Option<f64>,
    started: bool,
}

impl MdMinState {
    /// A zero velocity and ASE's time step unless `dt` is positive and finite.
    #[must_use]
    pub fn new(dim: usize, dt: f64) -> Self {
        let dt = if dt.is_finite() && dt > 0.0 {
            dt
        } else {
            ASE_DT
        };
        Self {
            vel: Array1::zeros(dim),
            dt,
            maxstep: None,
            started: false,
        }
    }

    /// Forget the velocity and the half-kick that opens the next call.
    pub fn reset(&mut self) {
        self.vel.fill(0.0);
        self.started = false;
    }
}

/// One MDMin displacement from `force`, updating `state.vel`.
///
/// A zero force after the first call clears the velocity, so the step
/// is zero. A tiny or subnormal force keeps the exact projection.
#[must_use]
pub fn mdmin_displacement(state: &mut MdMinState, force: &Array1<f64>) -> Array1<f64> {
    let dt = state.dt;
    let half = 0.5 * dt;
    if state.started {
        state.vel = &state.vel + &(force * half);
        // Project onto `F / max|F_i|`: its squared norm is at least one,
        // so a tiny or subnormal force cannot underflow `F . F` to zero.
        let largest = force.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        if largest > 0.0 && largest.is_finite() {
            let unit = force / largest;
            let vu = crate::vecops::dot(state.vel.view(), unit.view());
            if vu < 0.0 || !vu.is_finite() {
                state.vel.fill(0.0);
            } else {
                let uu = crate::vecops::dot(unit.view(), unit.view());
                state.vel = unit * (vu / uu);
            }
        } else {
            state.vel.fill(0.0);
        }
    } else {
        state.started = true;
    }
    state.vel = &state.vel + &(force * half);
    let mut dx = &state.vel * dt;
    if let Some(cap) = state.maxstep
        && cap.is_finite()
        && cap > 0.0
    {
        let longest = crate::step::max_atom_norm(dx.view());
        let scaling = cap / (SCALE_OFFSET + longest);
        if scaling < 1.0 {
            if scaling > 0.0 {
                dx *= scaling;
            } else {
                dx.fill(0.0);
            }
        }
    }
    dx
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn the_first_kick_is_half_a_step_and_the_cap_scales_a_long_atom() {
        let mut state = MdMinState::new(3, ASE_DT);
        state.maxstep = Some(ASE_MAXSTEP);
        let force = array![10.0, 0.0, 0.0];
        let dx = mdmin_displacement(&mut state, &force);
        let raw = ASE_DT * (0.5 * ASE_DT * 10.0);
        let longest = raw.abs();
        let scaling = ASE_MAXSTEP / (SCALE_OFFSET + longest);
        assert!((dx[0] - raw * scaling).abs() < 1e-12, "{dx:?}");
        assert_eq!(dx[1], 0.0);
        assert!((state.vel[0] - 0.5 * ASE_DT * 10.0).abs() < 1e-12);
    }

    #[test]
    fn an_uphill_velocity_is_cleared_before_the_second_kick() {
        let mut state = MdMinState::new(3, 0.2);
        state.maxstep = Some(1.0);
        let force = array![1.0, 0.0, 0.0];
        let _ = mdmin_displacement(&mut state, &force);
        state.vel = array![-1.0, 0.0, 0.0];
        let dx = mdmin_displacement(&mut state, &force);
        let kicked = 0.5 * 0.2 * 1.0;
        assert!((dx[0] - 0.2 * kicked).abs() < 1e-12, "{dx:?}");
    }

    fn second_call(force: &Array1<f64>) -> (Array1<f64>, MdMinState) {
        let mut state = MdMinState::new(3, ASE_DT);
        state.maxstep = Some(ASE_MAXSTEP);
        let _ = mdmin_displacement(&mut state, force);
        let dx = mdmin_displacement(&mut state, force);
        (dx, state)
    }

    #[test]
    fn a_zero_force_after_the_first_call_is_a_zero_step() {
        let (dx, state) = second_call(&array![0.0, 0.0, 0.0]);
        assert!(dx.iter().all(|&v| v == 0.0), "{dx:?}");
        assert!(state.vel.iter().all(|&v| v == 0.0), "{:?}", state.vel);
    }

    #[test]
    fn a_subnormal_force_is_a_finite_step_along_the_force() {
        let tiny = f64::MIN_POSITIVE / 4.0;
        assert!(tiny > 0.0 && !tiny.is_normal());
        let (dx, state) = second_call(&array![tiny, 0.0, 0.0]);
        assert!(dx.iter().all(|v| v.is_finite()), "{dx:?}");
        assert!(state.vel.iter().all(|v| v.is_finite()), "{:?}", state.vel);
        assert!(dx[0] >= 0.0 && dx[1] == 0.0 && dx[2] == 0.0, "{dx:?}");
    }

    #[test]
    fn a_tiny_force_keeps_the_projected_velocity() {
        // f.f underflows to zero here, while the projection is exact.
        let f = 1e-200;
        let force = array![f, 0.0, 0.0];
        let (dx, state) = second_call(&force);
        let first = 0.5 * ASE_DT * f;
        let vel = first + 2.0 * 0.5 * ASE_DT * f;
        assert!(dx.iter().all(|v| v.is_finite()), "{dx:?}");
        assert!(
            ((state.vel[0] - vel) / vel).abs() < 1e-12,
            "{:?}",
            state.vel
        );
        assert!(
            ((dx[0] - ASE_DT * vel) / (ASE_DT * vel)).abs() < 1e-12,
            "{dx:?}"
        );
        assert_eq!(dx[1], 0.0);
    }
}
