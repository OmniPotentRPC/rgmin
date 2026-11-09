//! FIRE and FIRE 2.0 inertial first-order steps.
//!
//! Bitzek, Koskinen, Gähler, Moseler, Gumbsch, *Structural Relaxation
//! Made Simple*, <https://doi.org/10.1103/PhysRevLett.97.170201>.
//! Guénolé, Nöhring, Vaid, Houllé, Xie, Prakash, Bitzek, *Assessment
//! and optimization of the fast inertial relaxation engine (FIRE)*,
//! <https://doi.org/10.1016/j.commatsci.2020.109584>.
//! Echeverri Restrepo, Andric, *ABC-FIRE: Accelerated Bias-Corrected
//! Fast Inertial Relaxation Engine*,
//! <https://doi.org/10.1016/j.commatsci.2022.111978>.

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

/// Parameter set of a FIRE 2.0 session.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FirePreset {
    /// The values [`FireState::new`] gives both integrators.
    #[default]
    Default,
    /// Guénolé 2020 Table 1, the LAMMPS `min_style fire` defaults:
    /// `dt_max = 10 dt`, `dt_min = 0.02 dt`, `N_delay = 20`,
    /// `alpha_start = 0.25`, `f_inc = 1.1`, `f_dec = 0.5`,
    /// `f_alpha = 0.99`, 2,000 non-positive-power steps. `dt` is the
    /// start step. A shrink that would pass `dt_min` keeps `dt`
    /// (Algorithm 2, line 23).
    Guenole2020,
}

/// Which FIRE schedule a session runs.
///
/// Selected with [`crate::Solver::set_fire_variant`]. The default keeps
/// the [`FireKind`] the session was built with and the parameters of
/// [`FireState::new`]. [`Self::Guenole2020`] switches the integrator to
/// [`FireKind::V2`] and installs [`FirePreset::Guenole2020`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FireVariant {
    /// [`FireKind::V1`] or [`FireKind::V2`] with [`FireState::new`].
    #[default]
    Rgmin,
    /// FIRE 2.0 with the Guénolé 2020 table: `alpha_start = 0.25`,
    /// `N_delay = 20`, `dt_max = 10 dt`, `dt_min = 0.02 dt`.
    Guenole2020,
}

/// ASE `ase/optimize/fire2.py` and LAMMPS `src/min_fire.cpp` floor
/// `alpha` here so the bias correction stays finite.
const ABC_ALPHA_FLOOR: f64 = 1e-10;

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
    preset: FirePreset,
    /// FIRE 2.0 only: divide the mixed velocity by
    /// `1 - (1 - alpha)^(n_pos + 1)` (ABC-FIRE).
    pub abc: bool,
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
            preset: FirePreset::Default,
            abc: false,
        }
    }

    /// Replace the parameter set and cold-start the integrator. The
    /// start step and [`Self::abc`] are kept.
    pub fn set_preset(&mut self, preset: FirePreset) {
        let abc = self.abc;
        *self = Self::new(self.kind, self.vel.len(), self.dt_start);
        self.abc = abc;
        self.preset = preset;
        if preset == FirePreset::Guenole2020 {
            self.dt_max = 10.0 * self.dt_start;
            self.dt_min = 0.02 * self.dt_start;
            self.n_min = 20;
            self.f_inc = 1.1;
            self.f_dec = 0.5;
            self.alpha_start = 0.25;
            self.alpha = self.alpha_start;
            self.f_alpha = 0.99;
            self.n_neg_max = 2_000;
        }
    }

    /// Parameter set selected by [`Self::set_preset`].
    pub fn preset(&self) -> FirePreset {
        self.preset
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

fn abc_floor(state: &mut FireState) {
    if state.abc {
        state.alpha = state.alpha.max(ABC_ALPHA_FLOOR);
    }
}

/// ABC-FIRE bias correction of the velocity just mixed, in the form of
/// ASE `FIRE2(use_abc=True)`: `n_pos + 1` mixes since the last reset.
fn abc_correct(state: &mut FireState) {
    if !state.abc {
        return;
    }
    let mixes = (state.n_pos + 1) as f64;
    let scale = 1.0 / (1.0 - (1.0 - state.alpha).powf(mixes));
    state.vel.mapv_inplace(|v| v * scale);
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
            let shrunk = state.dt * state.f_dec;
            state.dt = match state.preset {
                FirePreset::Default => shrunk.max(state.dt_min),
                FirePreset::Guenole2020 if shrunk >= state.dt_min => shrunk,
                FirePreset::Guenole2020 => state.dt,
            };
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
            abc_floor(state);
            mix_velocity(state, force);
            abc_correct(state);
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

/// ABC-FIRE cap: clip each velocity component to `cap / dt` and return
/// the displacement `dt v`, as ASE `FIRE2` does with `use_abc`.
pub(crate) fn fire_clip_components(state: &mut FireState, cap: f64) -> Array1<f64> {
    let vmax = cap / state.dt;
    state.vel.mapv_inplace(|v| v.max(-vmax).min(vmax));
    &state.vel * state.dt
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
            abc_floor(state);
            mix_toward(state, steepest, mix);
            abc_correct(state);
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
    use ndarray::{ArrayView1, array};

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

    /// Higham, *Accuracy and Stability of Numerical Algorithms* (2002),
    /// Lemma 3.1: `m` rounded operations, `u` the unit roundoff.
    fn gamma(m: f64) -> f64 {
        let u = f64::EPSILON / 2.0;
        m * u / (1.0 - m * u)
    }

    /// Largest gap between two roundings of one FIRE 2.0 velocity whose
    /// kick has norm `kick`. 14 operations reach the mixed velocity. ABC
    /// adds `(c - 1)(k + 2) + 3` from `(1 - alpha)^k`, its cancellation,
    /// the division and the scaling.
    fn vel_tol(state: &FireState, kick: f64) -> f64 {
        let k = (state.n_pos + 1) as f64;
        let c = 1.0 / (1.0 - (1.0 - state.alpha).powf(k));
        let (c, bias) = if state.abc {
            (c, gamma((c - 1.0) * (k + 2.0) + 3.0))
        } else {
            (1.0, 0.0)
        };
        2.0 * (gamma(14.0) + bias) * c * kick
    }

    #[test]
    fn guenole_preset_carries_table_1_and_skips_a_shrink_past_dt_min() {
        let mut state = FireState::new(FireKind::V2, 1, 1.0);
        state.set_preset(FirePreset::Guenole2020);
        assert_eq!((state.dt_max, state.dt_min, state.n_min), (10.0, 0.02, 20));
        assert_eq!((state.alpha, state.f_inc, state.f_dec), (0.25, 1.1, 0.5));
        assert_eq!((state.f_alpha, state.n_neg_max), (0.99, 2_000));
        // Past the delay, 0.03 * 0.5 < 0.02: Algorithm 2 keeps 0.03 and
        // the default clamps nothing above its 1e-12 floor.
        for (preset, dt) in [
            (FirePreset::Guenole2020, 0.03),
            (FirePreset::Default, 0.015),
        ] {
            state.set_preset(preset);
            state.steps = state.n_min;
            state.dt = 0.03;
            state.vel[0] = 1.0;
            let _ = fire_prepare_v2(&mut state, &array![-1.0]);
            assert_eq!(state.dt, dt, "{preset:?}");
        }
    }

    /// `v = (1, 0)` under `F = (1, 1)` and `dt = 0.1` kicks to
    /// `(1.1, 0.1)` with `|v| / |F| = sqrt(0.61)`. The second mix since
    /// the reset is divided by `1 - 0.75^2`.
    #[test]
    fn abc_divides_the_mix_by_its_bias() {
        let mut state = FireState::new(FireKind::V2, 2, 0.1);
        state.set_preset(FirePreset::Guenole2020);
        state.abc = true;
        state.vel = array![1.0, 0.0];
        let force = array![1.0, 1.0];
        let ready = fire_prepare_v2(&mut state, &force);
        assert!(matches!(ready, FirePreparation::Ready) && state.n_pos == 1);
        let dx = fire_displacement(&mut state, &force);
        let (mix, scale) = (0.25 * 0.61_f64.sqrt(), 1.0 / (1.0 - 0.75 * 0.75));
        let want = [(0.75 * 1.1 + mix) * scale, (0.75 * 0.1 + mix) * scale];
        let tol = vel_tol(&state, 1.22_f64.sqrt());
        for k in 0..2 {
            assert!((state.vel[k] - want[k]).abs() <= tol, "{}", state.vel);
            assert!((dx[k] - 0.1 * want[k]).abs() <= 0.1 * tol + gamma(1.0) * dx[k]);
        }
        let dx = fire_clip_components(&mut state, 0.2);
        assert_eq!((state.vel[0], dx[0]), (2.0, 0.2));
    }

    fn rosenbrock(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let (mut e, mut g) = (0.0, Array1::zeros(x.len()));
        for i in 0..x.len() - 1 {
            let t = x[i + 1] - x[i] * x[i];
            let u = 1.0 - x[i];
            e += 100.0 * t * t + u * u;
            g[i] += -400.0 * x[i] * t - 2.0 * u;
            g[i + 1] += 200.0 * t;
        }
        (e, g)
    }

    /// One unbounded FIRE 2.0 step as ASE `FIRE2` takes it. `x` takes the
    /// half-step back. Returns the kick norm and the move.
    fn ase_step(state: &mut FireState, x: &mut Array1<f64>, cap: f64) -> (f64, Array1<f64>) {
        let mut force = -rosenbrock(x.view()).1;
        if let FirePreparation::Rollback(c) = fire_prepare_v2(state, &force) {
            *x += &c;
            force = -rosenbrock(x.view()).1;
        }
        let kick = l2(&(&state.vel + &(&force * state.dt)));
        let dx = fire_displacement(state, &force);
        let dx = if state.abc {
            fire_clip_components(state, cap)
        } else {
            dx
        };
        (kick, dx)
    }

    /// Each step of ASE `FIRE2` on the extended Rosenbrock function
    /// (`tests/ase_fire2_gold.py`), taken from the ASE state. `dt`,
    /// `alpha` and `n_pos` follow the same operations and must match
    /// exactly. Velocity may differ by [`vel_tol`]; the move adds two
    /// roundings to the position. A `Solver` on the same settings runs
    /// the same operations, so it must match the free-running update
    /// bit for bit until it stops.
    #[test]
    fn guenole_steps_reproduce_ase_fire2_from_its_state() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/ase_fire2_rosenbrock.txt"
        );
        let text = std::fs::read_to_string(path).unwrap();
        let mut rows = text.lines().filter(|l| !l.starts_with('#')).map(|l| {
            let row = l.split_whitespace().filter_map(|t| t.parse().ok());
            row.collect::<Vec<f64>>()
        });
        let head = rows.next().unwrap();
        let (dt, fmax, cap) = (head[0], head[1], head[2]);
        while let Some(run) = rows.next() {
            let (abc, steps) = (run[0] == 1.0, run[1] as usize);
            let mut state = FireState::new(FireKind::V2, 3, dt);
            state.set_preset(FirePreset::Guenole2020);
            state.abc = abc;
            let mut free = state.clone();
            let control = crate::Control {
                maxiter: steps,
                gtol: fmax,
                istep: dt,
                maxmove: abc.then_some(cap),
                ..Default::default()
            };
            let mut solver =
                crate::Solver::new(crate::Method::Fire { kind: FireKind::V2 }, control, 3);
            solver.set_fire_preset(FirePreset::Guenole2020).unwrap();
            solver.set_fire_abc(abc).unwrap();
            let obj = crate::Oracle::unbounded(3, rosenbrock);
            let mut x = array![-1.2, 1.0, -1.2];
            let (mut y, mut z) = (x.clone(), x.clone());
            let mut norm = f64::INFINITY;
            for step in 0..steps {
                let want = rows.next().unwrap();
                let (kick, dx) = ase_step(&mut state, &mut x, cap);
                let same = (state.dt, state.alpha, state.n_pos as f64);
                assert_eq!(same, (want[6], want[7], want[8]), "abc={abc} step {step}");
                let tol_v = vel_tol(&state, kick);
                for i in 0..3 {
                    let tol_x = state.dt * tol_v + 2.0 * gamma(2.0) * (x[i].abs() + dx[i].abs());
                    assert!(
                        (state.vel[i] - want[3 + i]).abs() <= tol_v,
                        "abc={abc} step {step}"
                    );
                    assert!(
                        (x[i] + dx[i] - want[i]).abs() <= tol_x,
                        "abc={abc} step {step}"
                    );
                    x[i] = want[i];
                    state.vel[i] = want[3 + i];
                }
                if norm >= fmax {
                    let (_, d) = ase_step(&mut free, &mut y, cap);
                    y += &d;
                    norm = solver.step(&obj, &mut z).unwrap().grad_norm;
                    assert_eq!(y, z, "abc={abc} step {step}");
                }
            }
        }
    }
}
