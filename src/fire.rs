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
    /// Mixing parameter. The mix uses it clamped to `[0, 1]` (NaN as 0),
    /// the range in which it never lengthens the velocity.
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
    // The mix is a convex combination only for alpha in [0, 1]; the
    // norm bound |1 - 2 alpha| |v| <= |v'| <= |v| needs it, so the
    // public field is clamped where it is used.
    let alpha = if state.alpha.is_nan() {
        0.0
    } else {
        state.alpha.clamp(0.0, 1.0)
    };
    let scale = alpha * vnorm / fnorm;
    for i in 0..state.vel.len() {
        state.vel[i] = (1.0 - alpha) * state.vel[i] + scale * force[i];
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

/// Which FIRE schedule a session runs.
///
/// Selected with [`crate::Solver::set_fire_variant`]; the default keeps
/// the [`FireKind`] the session was built with and its parameters.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FireVariant {
    /// [`FireKind::V1`] or [`FireKind::V2`] with [`FireState::new`]
    /// parameters.
    #[default]
    Rgmin,
    /// FIRE 2.0 exactly as Guénolé et al. 2020 algorithm 2 (semi-implicit
    /// Euler) publishes it, with its table 2 parameters: see
    /// [`fire2_displacement`].
    Guenole2020,
}

/// The FIRE 2.0 state that [`FireState`] does not carry.
///
/// Built only through [`Fire2Extras::new`], so fields can be added
/// without breaking callers.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct Fire2Extras {
    /// Floor under `dt` shrinks, `dt_min`.
    pub dt_min: f64,
    /// `N_delay`: steps before `dt` may grow, and the length of the
    /// initial delay.
    pub n_delay: usize,
    /// During the first `n_delay` steps an uphill step neither shrinks
    /// `dt` nor resets `alpha`.
    pub initial_delay: bool,
    /// Steps taken since the last reset.
    pub steps: usize,
}

impl Fire2Extras {
    /// Guénolé et al. 2020 table 2: `dt_min = 0.02 dt0`, `N_delay = 20`,
    /// initial delay on.
    #[must_use]
    pub fn new(dt0: f64) -> Self {
        let dt0 = if dt0 > 0.0 { dt0 } else { 0.1 };
        Self {
            dt_min: 0.02 * dt0,
            n_delay: 20,
            initial_delay: true,
            steps: 0,
        }
    }
}

/// A [`FireState`] and [`Fire2Extras`] with the Guénolé et al. 2020
/// table 2 parameters: `alpha_start = 0.25`, `f_alpha = 0.99`,
/// `f_inc = 1.1`, `f_dec = 0.5`, `N_delay = 20`, `dt_max = 10 dt0`,
/// `dt_min = 0.02 dt0`.
#[must_use]
pub fn guenole2020(dim: usize, dt0: f64) -> (FireState, Fire2Extras) {
    let mut state = FireState::new(FireKind::V2, dim, dt0);
    let ext = Fire2Extras::new(state.dt);
    state.alpha = 0.25;
    state.alpha_start = 0.25;
    state.n_min = ext.n_delay;
    state.dt_max = 10.0 * state.dt;
    (state, ext)
}

/// FIRE 2.0 displacement, Guénolé et al. 2020 algorithm 2 with the
/// semi-implicit Euler integrator, from the force `f = -g` at the
/// current point.
///
/// With power `P = f . v`: on `P > 0`, count it and, past `N_delay`
/// such steps, grow `dt` (to `dt_max`) and decay `alpha`. On `P <= 0`,
/// unless inside the initial delay, shrink `dt` (not below `dt_min`)
/// and reset `alpha`; then step back half a step, `-dt v / 2`, with the
/// updated `dt`, and zero `v`. Then `v += dt f`,
/// `v = (1 - alpha) v + alpha |v| f / |f|`, and move `dt v`. The
/// returned `dx` includes the half-step correction, so it equals
/// `dt v` only on a downhill step. The mix never lengthens `v` and never
/// lowers `f . v` (`validation/fire2.py`).
///
/// Guénolé, Nöhring, Vaid, Houllé, Xie, Prakash, Bitzek, *Assessment
/// and optimization of the fast inertial relaxation engine (FIRE)*,
/// <https://doi.org/10.1016/j.commatsci.2020.109584>.
pub fn fire2_displacement(
    state: &mut FireState,
    ext: &mut Fire2Extras,
    force: &Array1<f64>,
) -> Array1<f64> {
    let power = crate::vecops::dot(force.view(), state.vel.view());
    let mut dx = Array1::zeros(state.vel.len());
    if power > 0.0 {
        state.n_pos += 1;
        if state.n_pos > ext.n_delay {
            state.dt = (state.dt * state.f_inc).min(state.dt_max);
            state.alpha *= state.f_alpha;
        }
    } else {
        state.n_pos = 0;
        if !(ext.initial_delay && ext.steps < ext.n_delay) {
            if state.dt * state.f_dec >= ext.dt_min {
                state.dt *= state.f_dec;
            }
            state.alpha = state.alpha_start;
        }
        dx.scaled_add(-0.5 * state.dt, &state.vel);
        state.vel.fill(0.0);
    }
    ext.steps += 1;
    crate::vecops::axpy(state.dt, force.view(), &mut state.vel);
    mix_velocity(state, force);
    dx.scaled_add(state.dt, &state.vel);
    dx
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
    fn guenole_uphill_steps_back_half_a_step() {
        // Past the initial delay, an uphill step halves dt (0.2 -> 0.1),
        // resets alpha, moves back dt v / 2 with the new dt, zeroes v,
        // then integrates: v = dt f, the mix keeps v parallel to f, and
        // dx = -dt v_old / 2 + dt^2 f.
        let (mut state, mut ext) = guenole2020(2, 0.2);
        ext.steps = ext.n_delay;
        state.vel = array![1.0, 0.0];
        state.alpha = 0.05;
        let force = array![-1.0, 0.5];
        let dx = fire2_displacement(&mut state, &mut ext, &force);
        assert!((state.dt - 0.1).abs() < 1e-15);
        assert!((state.alpha - 0.25).abs() < 1e-15);
        let want = [-0.05 - 0.01, 0.005];
        for i in 0..2 {
            assert!(
                (dx[i] - want[i]).abs() < 1e-15,
                "{i}: {} vs {}",
                dx[i],
                want[i]
            );
            assert!((state.vel[i] - 0.1 * force[i]).abs() < 1e-15);
        }
    }

    #[test]
    fn guenole_initial_delay_and_dt_floor_hold_dt() {
        let force = array![-1.0, 0.0];
        let (mut state, mut ext) = guenole2020(2, 0.2);
        state.vel = array![1.0, 0.0];
        let _ = fire2_displacement(&mut state, &mut ext, &force);
        assert!((state.dt - 0.2).abs() < 1e-15, "the delay shrank dt");
        ext.steps = ext.n_delay;
        state.dt = 1.5 * ext.dt_min;
        state.vel = array![1.0, 0.0];
        let _ = fire2_displacement(&mut state, &mut ext, &force);
        assert!(
            (state.dt - 1.5 * ext.dt_min).abs() < 1e-15,
            "dt went under dt_min"
        );
    }

    #[test]
    fn guenole_downhill_mixes_after_the_kick() {
        // v = (1, 0), f = (0, 1), dt = 0.1, alpha = 0.25, P = 0 -> uphill
        // branch inside the delay: no dt change, v zeroed, dx = -dt v / 2
        // + dt^2 f. Then a downhill step: v = dt f + dt f = 2 dt f before
        // the mix, which keeps it (parallel to f), dt unchanged (delay).
        let (mut state, mut ext) = guenole2020(2, 0.1);
        state.vel = array![1.0, 0.0];
        let f = array![0.0, 1.0];
        let dx = fire2_displacement(&mut state, &mut ext, &f);
        assert!((dx[0] + 0.05).abs() < 1e-15 && (dx[1] - 0.01).abs() < 1e-15);
        let dx = fire2_displacement(&mut state, &mut ext, &f);
        assert!(dx[0].abs() < 1e-15 && (dx[1] - 0.02).abs() < 1e-15, "{dx}");
        assert_eq!(state.n_pos, 1);
    }

    #[test]
    fn an_out_of_range_alpha_never_lengthens_the_velocity() {
        for a in [2.0, -1.0, f64::NAN] {
            let mut state = FireState::new(FireKind::V1, 2, 0.1);
            state.vel = array![1.0, 0.0];
            state.alpha = a;
            mix_velocity(&mut state, &array![0.0, 3.0]);
            assert!(
                l2(&state.vel) <= 1.0 + 1e-15,
                "alpha {a}: {}",
                l2(&state.vel)
            );
        }
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
