//! FIRE and FIRE 2.0 inertial first-order steps.
//!
//! Bitzek, Koskinen, Gähler, Moseler, Gumbsch, *Structural Relaxation
//! Made Simple*, <https://doi.org/10.1103/PhysRevLett.97.170201>.
//! Guénolé, Nöhring, Vaid, Houllé, Xie, Prakash, Bitzek, *Assessment
//! and optimization of the fast inertial relaxation engine (FIRE)*,
//! <https://doi.org/10.1016/j.commatsci.2020.109584>.

use ndarray::Array1;

use crate::step::l2;

/// Which FIRE integrator to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FireKind {
    /// Bitzek 2006 / eOn native: velocity-Verlet, mix after the MD step.
    V1,
    /// Guénolé 2020 FIRE 2.0: mix first, then semi-implicit Euler.
    V2,
}

/// Session memory for one FIRE or FIRE 2.0 solve.
pub struct FireState {
    /// Integrator.
    pub kind: FireKind,
    /// Velocity.
    pub vel: Array1<f64>,
    /// Time step.
    pub dt: f64,
    /// Time-step cap.
    pub dt_max: f64,
    /// Mixing parameter.
    pub alpha: f64,
    /// Reset value of [`Self::alpha`].
    pub alpha_start: f64,
    /// Consecutive downhill steps.
    pub n_pos: usize,
    /// Delay before growing `dt`.
    pub n_min: usize,
    /// `dt` growth factor.
    pub f_inc: f64,
    /// `dt` shrink factor.
    pub f_dec: f64,
    /// `alpha` decay.
    pub f_alpha: f64,
}

impl FireState {
    /// Bitzek / Guénolé defaults. `dt` is [`crate::Control::istep`].
    pub fn new(kind: FireKind, dim: usize, dt: f64) -> Self {
        let dt = if dt > 0.0 { dt } else { 0.1 };
        Self {
            kind,
            vel: Array1::zeros(dim),
            dt,
            dt_max: (dt * 2.5).max(dt),
            alpha: 0.1,
            alpha_start: 0.1,
            n_pos: 0,
            n_min: 5,
            f_inc: 1.1,
            f_dec: 0.5,
            f_alpha: 0.99,
        }
    }

    /// Drop velocity and mixing; keep the current `dt`.
    pub fn reset(&mut self) {
        self.vel.fill(0.0);
        self.alpha = self.alpha_start;
        self.n_pos = 0;
    }
}

fn mix_velocity(state: &mut FireState, force: &Array1<f64>) {
    let fnorm = l2(force);
    let vnorm = l2(&state.vel);
    if fnorm <= 0.0 {
        return;
    }
    let scale = state.alpha * vnorm / fnorm;
    for i in 0..state.vel.len() {
        state.vel[i] = (1.0 - state.alpha) * state.vel[i] + scale * force[i];
    }
}

fn adapt(state: &mut FireState, power: f64) {
    if power > 0.0 {
        state.n_pos += 1;
        if state.n_pos > state.n_min {
            state.dt = (state.dt * state.f_inc).min(state.dt_max);
            state.alpha *= state.f_alpha;
        }
    } else {
        state.dt *= state.f_dec;
        if state.dt < 1e-12 {
            state.dt = 1e-12;
        }
        state.vel.fill(0.0);
        state.alpha = state.alpha_start;
        state.n_pos = 0;
    }
}

/// Displacement `dx` from the current force `f = -g`.
///
/// FIRE 2.0 mixes and adapts on this force, then takes a semi-implicit
/// Euler step. FIRE 1.0 takes the Verlet step first; the caller mixes
/// on the force at the new point via [`fire_after_v1`].
pub fn fire_displacement(state: &mut FireState, force: &Array1<f64>) -> Array1<f64> {
    match state.kind {
        FireKind::V2 => {
            let power = crate::vecops::dot(force.view(), state.vel.view());
            if power > 0.0 {
                mix_velocity(state, force);
            }
            adapt(state, power);
            for i in 0..state.vel.len() {
                state.vel[i] += force[i] * state.dt;
            }
            &state.vel * state.dt
        }
        FireKind::V1 => {
            for i in 0..state.vel.len() {
                state.vel[i] += force[i] * state.dt;
            }
            &state.vel * state.dt
        }
    }
}

/// Keep the velocity consistent with the displacement the caller took.
///
/// `proposed` is the displacement [`fire_displacement`] returned, so
/// `vel = proposed / dt`; `taken` is what survived the maxmove clamp
/// and the bounds clip. When the clamp shortened the move, the velocity
/// shrinks by the same norm ratio, so the power `F . v` that
/// [`fire_after_v1`] and the next [`fire_displacement`] adapt on
/// measures the motion that happened. Without this the integral keeps
/// growing while the clamp fires every step, and `dt` grows with it.
pub fn fire_rescale_velocity(state: &mut FireState, proposed: &Array1<f64>, taken: &Array1<f64>) {
    let pn = l2(proposed);
    if pn <= 0.0 {
        return;
    }
    let factor = l2(taken) / pn;
    if factor < 1.0 {
        state.vel.mapv_inplace(|v| v * factor);
    }
}

/// FIRE 1.0 mix and adapt after the MD step, using the new force.
pub fn fire_after_v1(state: &mut FireState, force: &Array1<f64>) {
    if !matches!(state.kind, FireKind::V1) {
        return;
    }
    let power = crate::vecops::dot(force.view(), state.vel.view());
    mix_velocity(state, force);
    adapt(state, power);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn v1_first_step_follows_the_force() {
        let mut state = FireState::new(FireKind::V1, 2, 0.2);
        let force = array![1.0, 0.0];
        let dx = fire_displacement(&mut state, &force);
        assert!(dx[0] > 0.0);
        assert!(dx[1].abs() < 1e-15);
    }

    #[test]
    fn a_clamped_move_shrinks_the_velocity_by_the_same_ratio() {
        let mut state = FireState::new(FireKind::V2, 2, 0.1);
        let force = array![1000.0, 0.0];
        let dx = fire_displacement(&mut state, &force);
        // The clamp keeps 1% of the proposed move.
        let taken = &dx * 0.01;
        fire_rescale_velocity(&mut state, &dx, &taken);
        // FIRE 2.0 takes dx = vel * dt with the dt it adapted to, so the
        // velocity times that dt equals the clamped displacement.
        for i in 0..2 {
            assert!(
                (state.vel[i] * state.dt - taken[i]).abs() <= 1e-12 * taken[i].abs().max(1.0),
                "vel {} dt {} taken {}",
                state.vel[i],
                state.dt,
                taken[i]
            );
        }
        // An unclamped move leaves the velocity alone.
        let before = state.vel.clone();
        fire_rescale_velocity(&mut state, &taken, &taken);
        assert_eq!(state.vel, before);
    }

    #[test]
    fn uphill_power_resets_velocity() {
        let mut state = FireState::new(FireKind::V2, 2, 0.2);
        state.vel = array![1.0, 0.0];
        let force = array![-1.0, 0.0];
        let _ = fire_displacement(&mut state, &force);
        assert_eq!(state.n_pos, 0);
        assert!((state.alpha - state.alpha_start).abs() < 1e-15);
    }
}
