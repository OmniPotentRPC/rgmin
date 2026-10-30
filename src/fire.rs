//! FIRE and FIRE 2.0 inertial first-order steps.
//!
//! Bitzek, Koskinen, Gähler, Moseler, Gumbsch, *Structural Relaxation
//! Made Simple*, <https://doi.org/10.1103/PhysRevLett.97.170201>.
//! Guénolé, Nöhring, Vaid, Houllé, Xie, Prakash, Bitzek, *Assessment
//! and optimization of the fast inertial relaxation engine (FIRE)*,
//! <https://doi.org/10.1016/j.commatsci.2020.109584>.

use ndarray::Array1;

use crate::step::l2;

/// Remove outward components at active coordinate bounds.
/// Wall reactions contribute neither velocity nor inertial power.
pub(crate) fn project_box_tangent(
    bounds: &eindir_core::Bounds<f64>,
    x: &Array1<f64>,
    vector: &mut Array1<f64>,
) {
    for k in 0..x.len() {
        if (x[k] <= bounds.low[k] && vector[k] < 0.0) || (x[k] >= bounds.high[k] && vector[k] > 0.0)
        {
            vector[k] = 0.0;
        }
    }
}

/// Which FIRE integrator to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FireKind {
    /// Bitzek 2006 / eOn native: velocity-Verlet, mix after the MD step.
    V1,
    /// Guénolé 2020 FIRE 2.0: semi-implicit Euler, then velocity mixing.
    V2,
}

/// Session memory for one FIRE or FIRE 2.0 solve.
#[derive(Clone)]
pub struct FireState {
    /// Integrator.
    pub kind: FireKind,
    /// Velocity.
    pub vel: Array1<f64>,
    /// Time step.
    pub dt: f64,
    /// Initial time step.
    dt_start: f64,
    /// Time-step floor.
    dt_min: f64,
    /// Time-step cap.
    pub dt_max: f64,
    /// Mixing parameter.
    pub alpha: f64,
    /// Reset value of [`Self::alpha`].
    pub alpha_start: f64,
    /// Consecutive downhill steps.
    pub n_pos: usize,
    /// Consecutive non-positive-power steps.
    n_neg: usize,
    /// Divergence limit for consecutive non-positive-power steps.
    n_neg_max: usize,
    /// Integration steps since the last cold start.
    steps: usize,
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
        let (dt_max, dt_min, alpha_start, n_min) = match kind {
            FireKind::V1 | FireKind::V2 => ((dt * 2.5).max(dt), 1.0e-12, 0.1, 5),
        };
        Self {
            kind,
            vel: Array1::zeros(dim),
            dt,
            dt_start: dt,
            dt_min,
            dt_max,
            alpha: alpha_start,
            alpha_start,
            n_pos: 0,
            n_neg: 0,
            n_neg_max: 2_000,
            steps: 0,
            n_min,
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
        self.n_neg = 0;
        if matches!(self.kind, FireKind::V2) {
            self.dt = self.dt_start;
            self.steps = 0;
        }
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

/// Norms the FIRE mix matches.
///
/// The Euclidean mix uses the 2-norm of the velocity and of the
/// steepest-descent direction. In the metric of a preconditioner `P`
/// those are `sqrt(v^T P v)` and `sqrt(F · P^{-1} F)`.
#[derive(Clone, Copy, Debug)]
pub struct FireMix {
    /// Norm of the velocity.
    pub vel: f64,
    /// Norm of the direction being mixed in.
    pub direction: f64,
}

fn mix_toward(state: &mut FireState, direction: &Array1<f64>, mix: Option<FireMix>) {
    let (vnorm, dnorm) = match mix {
        Some(m) => (m.vel, m.direction),
        None => (l2(&state.vel), l2(direction)),
    };
    if !(dnorm.is_finite() && dnorm > 0.0 && vnorm.is_finite() && vnorm >= 0.0) {
        return;
    }
    let alpha = if state.alpha.is_nan() {
        0.0
    } else {
        state.alpha.clamp(0.0, 1.0)
    };
    let scale = alpha * vnorm / dnorm;
    for i in 0..state.vel.len() {
        state.vel[i] = (1.0 - alpha) * state.vel[i] + scale * direction[i];
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
        if state.dt < state.dt_min {
            state.dt = state.dt_min;
        }
        state.vel.fill(0.0);
        state.alpha = state.alpha_start;
        state.n_pos = 0;
    }
}

/// FIRE 2.0 preparation before the forward integrator.
pub(crate) enum FirePreparation {
    /// Continue from the current point.
    Ready,
    /// Apply the half-step-back correction and refresh the force.
    Rollback(Array1<f64>),
    /// Consecutive uphill steps exceeded Guénolé's limit.
    Stalled,
}

/// Prepare a FIRE 2.0 step.
///
/// The caller must evaluate a fresh force after applying a nonzero
/// correction, before passing that force to [`fire_displacement`].
pub(crate) fn fire_prepare_v2(state: &mut FireState, force: &Array1<f64>) -> FirePreparation {
    if !matches!(state.kind, FireKind::V2) {
        return FirePreparation::Ready;
    }
    let power = crate::vecops::dot(force.view(), state.vel.view());
    if power > 0.0 {
        state.n_pos += 1;
        state.n_neg = 0;
        if state.n_pos > state.n_min {
            state.dt = (state.dt * state.f_inc).min(state.dt_max);
            state.alpha *= state.f_alpha;
        }
        FirePreparation::Ready
    } else {
        state.n_pos = 0;
        state.n_neg += 1;
        if state.n_neg > state.n_neg_max {
            return FirePreparation::Stalled;
        }
        if state.steps + 1 >= state.n_min {
            state.dt = (state.dt * state.f_dec).max(state.dt_min);
        }
        let correction = -0.5 * state.dt * &state.vel;
        let moved = correction.iter().any(|v| *v != 0.0);
        state.vel.fill(0.0);
        state.alpha = state.alpha_start;
        if moved {
            FirePreparation::Rollback(correction)
        } else {
            FirePreparation::Ready
        }
    }
}

/// Forward displacement `dx` from the current force `f = -g`.
///
/// FIRE 2.0 applies the semi-implicit Euler update before mixing. The
/// caller first applies [`fire_prepare_v2`]. FIRE 1.0 takes the Verlet
/// step first; the caller mixes on the force at the new point via
/// [`fire_after_v1`].
pub fn fire_displacement(state: &mut FireState, force: &Array1<f64>) -> Array1<f64> {
    match state.kind {
        FireKind::V2 => {
            for i in 0..state.vel.len() {
                state.vel[i] += force[i] * state.dt;
            }
            mix_velocity(state, force);
            state.steps += 1;
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

/// FIRE 1.0 mix and adapt after the MD step, using the new force.
pub fn fire_after_v1(state: &mut FireState, force: &Array1<f64>) {
    if !matches!(state.kind, FireKind::V1) {
        return;
    }
    let power = crate::vecops::dot(force.view(), state.vel.view());
    mix_velocity(state, force);
    adapt(state, power);
}

/// Displacement when the physical force and the steepest direction differ.
///
/// `power_force` is `F = -∇E`. FIRE 2.0 power, rollback, and the time-step
/// delay are [`fire_prepare_v2`] on that force, which the caller runs first.
/// `steepest` is the kick and the mix direction (`P^{-1} F` when a metric
/// is installed). `mix_at` is called with the post-kick velocity on FIRE 2.0
/// and is not called on FIRE 1.0. FIRE 2.0 then mixes and advances the
/// integration counter. FIRE 1.0 only takes the Verlet kick; the caller
/// mixes on the new force via [`fire_after_v1_metric`].
pub fn fire_displacement_metric<F>(
    state: &mut FireState,
    power_force: &Array1<f64>,
    steepest: &Array1<f64>,
    mut mix_at: F,
) -> Array1<f64>
where
    F: FnMut(&Array1<f64>) -> Option<FireMix>,
{
    debug_assert_eq!(power_force.len(), steepest.len());
    match state.kind {
        FireKind::V2 => {
            for i in 0..state.vel.len() {
                state.vel[i] += steepest[i] * state.dt;
            }
            let mix = mix_at(&state.vel);
            mix_toward(state, steepest, mix);
            state.steps += 1;
            &state.vel * state.dt
        }
        FireKind::V1 => {
            for i in 0..state.vel.len() {
                state.vel[i] += steepest[i] * state.dt;
            }
            &state.vel * state.dt
        }
    }
}

/// FIRE 1.0 mix and adapt in a preconditioner metric.
///
/// `power_force` is the physical force `F = -∇E`. The power is
/// `F · v`. `steepest` is the direction the velocity is mixed toward
/// (`P^{-1} F` when a metric is installed, otherwise `F`). `mix`
/// supplies the norms; `None` uses the Euclidean 2-norm.
pub fn fire_after_v1_metric(
    state: &mut FireState,
    power_force: &Array1<f64>,
    steepest: &Array1<f64>,
    mix: Option<FireMix>,
) {
    if !matches!(state.kind, FireKind::V1) {
        return;
    }
    let power = crate::vecops::dot(power_force.view(), state.vel.view());
    mix_toward(state, steepest, mix);
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
    fn v1_power_uses_the_physical_force() {
        let mut state = FireState::new(FireKind::V1, 2, 0.1);
        state.vel = array![1.0, 0.0];
        state.n_pos = 6;
        // Physical force is downhill. The metric direction is orthogonal,
        // so a power test on the metric direction would reset the step.
        let raw = array![1.0, 0.0];
        let steepest = array![0.0, 4.0];
        fire_after_v1_metric(&mut state, &raw, &steepest, None);
        assert!(state.vel[0] > 0.0, "velocity was reset: {}", state.vel);
        assert!(state.vel[1] > 0.0, "mix missed the metric direction");
    }

    #[test]
    fn uphill_power_resets_velocity() {
        let mut state = FireState::new(FireKind::V2, 2, 0.2);
        state.vel = array![1.0, 0.0];
        let force = array![-1.0, 0.0];
        let FirePreparation::Rollback(correction) = fire_prepare_v2(&mut state, &force) else {
            panic!("expected rollback");
        };
        let dx = fire_displacement(&mut state, &force);
        assert_eq!(state.n_pos, 0);
        assert!((state.alpha - state.alpha_start).abs() < 1e-15);
        assert!((state.dt - 0.2).abs() < 1e-15);
        assert!((correction[0] + 0.1).abs() < 1e-15);
        assert!((dx[0] + 0.04).abs() < 1e-15);
    }

    #[test]
    fn v2_delays_the_first_shrink() {
        let mut state = FireState::new(FireKind::V2, 1, 0.1);
        state.steps = 4;
        state.vel[0] = 1.0;
        let _ = fire_prepare_v2(&mut state, &array![-1.0]);
        assert!((state.dt - 0.05).abs() < 1e-15);
    }

    #[test]
    fn v2_metric_mixes_after_the_steepest_kick() {
        let mut state = FireState::new(FireKind::V2, 2, 0.2);
        state.vel = array![1.0, 0.0];
        let raw = array![1.0, 0.0];
        assert!(matches!(
            fire_prepare_v2(&mut state, &raw),
            FirePreparation::Ready
        ));
        let steepest = array![0.0, 2.0];
        let mut seen = array![0.0, 0.0];
        let dx = fire_displacement_metric(&mut state, &raw, &steepest, |vel| {
            seen.assign(vel);
            Some(FireMix {
                vel: 1.0,
                direction: 2.0,
            })
        });
        assert!((seen[0] - 1.0).abs() < 1e-15 && (seen[1] - 0.4).abs() < 1e-15);
        assert!((dx[0] - 0.18).abs() < 1e-15, "dx={dx:?}");
        assert!((dx[1] - 0.092).abs() < 1e-15, "dx={dx:?}");
        assert_eq!(state.steps, 1);
    }

    #[test]
    fn v2_stops_after_the_negative_power_limit() {
        let mut state = FireState::new(FireKind::V2, 1, 0.1);
        state.n_neg = state.n_neg_max;
        state.vel[0] = 1.0;
        assert!(matches!(
            fire_prepare_v2(&mut state, &array![-1.0]),
            FirePreparation::Stalled
        ));
    }
}
