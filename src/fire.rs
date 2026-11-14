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
//!
//! [`fire_displacement`] is the rgmin integrator: a downhill step mixes
//! first, then adapts, then takes a semi-implicit Euler kick. FIRE 1.0
//! takes the kick here and mixes afterwards in [`fire_after_v1`].
//! [`fire2_displacement`] is Guénolé algorithm 2. [`fire_prepare_v2`]
//! followed by [`fire_displacement_into`] is that same published step
//! split so a session can roll the half-step back before the kick, and
//! it is where ABC-FIRE and [`FirePreset`] apply.

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
    /// Guénolé 2020 FIRE 2.0: semi-implicit Euler.
    ///
    /// [`fire_displacement`] mixes a downhill velocity before the kick.
    /// [`fire2_displacement`] and [`fire_prepare_v2`] follow algorithm 2:
    /// the kick, then the mix, with a half-step back on a non-positive power.
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
    /// start step. Inside the first `N_delay` steps an uphill step
    /// neither shrinks `dt` nor resets `alpha`. A later shrink that
    /// would pass `dt_min` keeps `dt` (Algorithm 2).
    Guenole2020,
}

/// ASE `ase/optimize/fire2.py` and LAMMPS `src/min_fire.cpp` floor
/// `alpha` here so the bias correction stays finite.
const ABC_ALPHA_FLOOR: f64 = 1e-10;

/// Integrator parameters for [`FireState`].
///
/// [`FireConfig::ase`] is the ASE `FIRE` constructor: `dt = 0.1`,
/// `dtmax = 1`, `Nmin = 5`, `finc = 1.1`, `fdec = 0.5`,
/// `astart = 0.1`, `fa = 0.99`. [`FireState::new`] uses a tighter
/// `dt_max` of `2.5 dt`, which is the eOn / Bitzek session default,
/// not ASE's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FireConfig {
    /// Time step.
    pub dt: f64,
    /// Time-step cap.
    pub dt_max: f64,
    /// Downhill steps before `dt` grows.
    pub n_min: usize,
    /// `dt` growth factor.
    pub f_inc: f64,
    /// `dt` shrink factor.
    pub f_dec: f64,
    /// Mixing parameter after a reset.
    pub alpha: f64,
    /// `alpha` decay.
    pub f_alpha: f64,
}

impl FireConfig {
    /// ASE `FIRE` defaults. `dt_max` is 1, ten times `dt`.
    #[must_use]
    pub fn ase() -> Self {
        Self {
            dt: 0.1,
            dt_max: 1.0,
            n_min: 5,
            f_inc: 1.1,
            f_dec: 0.5,
            alpha: 0.1,
            f_alpha: 0.99,
        }
    }
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
    /// Time-step floor used by [`fire_prepare_v2`].
    dt_min: f64,
    /// Time-step cap.
    pub dt_max: f64,
    /// Mixing parameter. The mix uses it clamped to `[0, 1]` (NaN as 0),
    /// the range in which it never lengthens the velocity.
    pub alpha: f64,
    /// Reset value of [`Self::alpha`].
    pub alpha_start: f64,
    /// Consecutive downhill steps.
    pub n_pos: usize,
    /// Consecutive non-positive-power steps of the preset integrator.
    n_neg: usize,
    /// Divergence limit for [`fire_prepare_v2`]. `usize::MAX` does not stop.
    n_neg_max: usize,
    /// Integration steps since the last cold start of the preset integrator.
    steps: usize,
    /// Delay before growing `dt`. For [`FirePreset::Guenole2020`] this is
    /// `N_delay`.
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
    /// Velocity from the start of the current step, so a refused trial
    /// restores it without allocating.
    saved_vel: Array1<f64>,
}

/// Scalar FIRE state saved with [`FireState::checkpoint`].
#[derive(Clone, Copy)]
pub(crate) struct FireCheckpoint {
    dt: f64,
    alpha: f64,
    n_pos: usize,
    n_neg: usize,
    steps: usize,
}

impl FireState {
    /// Bitzek / Guénolé defaults. `dt` is [`crate::Control::istep`].
    pub fn new(kind: FireKind, dim: usize, dt: f64) -> Self {
        let dt = if dt > 0.0 { dt } else { 0.1 };
        Self {
            kind,
            vel: Array1::zeros(dim),
            dt,
            dt_start: dt,
            dt_min: 1.0e-12,
            dt_max: (dt * 2.5).max(dt),
            alpha: 0.1,
            alpha_start: 0.1,
            n_pos: 0,
            n_neg: 0,
            n_neg_max: 2_000,
            steps: 0,
            n_min: 5,
            f_inc: 1.1,
            f_dec: 0.5,
            f_alpha: 0.99,
            preset: FirePreset::Default,
            abc: false,
            saved_vel: Array1::zeros(dim),
        }
    }

    /// Remember the velocity and the scalars a refused trial restores.
    pub(crate) fn checkpoint(&mut self) -> FireCheckpoint {
        if self.saved_vel.len() != self.vel.len() {
            self.saved_vel = self.vel.clone();
        } else {
            self.saved_vel.assign(&self.vel);
        }
        FireCheckpoint {
            dt: self.dt,
            alpha: self.alpha,
            n_pos: self.n_pos,
            n_neg: self.n_neg,
            steps: self.steps,
        }
    }

    /// Undo the integrator mutations made since [`Self::checkpoint`].
    pub(crate) fn restore(&mut self, saved: FireCheckpoint) {
        self.dt = saved.dt;
        self.alpha = saved.alpha;
        self.n_pos = saved.n_pos;
        self.n_neg = saved.n_neg;
        self.steps = saved.steps;
        if self.vel.len() == self.saved_vel.len() {
            self.vel.assign(&self.saved_vel);
        } else {
            self.vel = self.saved_vel.clone();
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
    #[must_use]
    pub fn preset(&self) -> FirePreset {
        self.preset
    }

    /// Replace the integrator parameters and cold-start the velocity.
    /// A non-positive `dt` or `dt_max` is left as it was.
    pub fn apply_config(&mut self, config: &FireConfig) {
        if config.dt.is_finite() && config.dt > 0.0 {
            self.dt = config.dt;
            self.dt_start = config.dt;
        }
        if config.dt_max.is_finite() && config.dt_max > 0.0 {
            self.dt_max = config.dt_max;
        }
        self.n_min = config.n_min;
        if config.f_inc.is_finite() && config.f_inc > 1.0 {
            self.f_inc = config.f_inc;
        }
        if config.f_dec.is_finite() && config.f_dec > 0.0 && config.f_dec < 1.0 {
            self.f_dec = config.f_dec;
        }
        if config.alpha.is_finite() && config.alpha > 0.0 {
            self.alpha_start = config.alpha;
            self.alpha = config.alpha;
        }
        if config.f_alpha.is_finite() && config.f_alpha > 0.0 && config.f_alpha < 1.0 {
            self.f_alpha = config.f_alpha;
        }
        self.vel.fill(0.0);
        self.n_pos = 0;
        self.n_neg = 0;
        self.steps = 0;
    }

    /// Drop velocity and mixing. The current `dt` is kept.
    pub fn reset(&mut self) {
        self.vel.fill(0.0);
        self.alpha = self.alpha_start;
        self.n_pos = 0;
        self.n_neg = 0;
        self.steps = 0;
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

fn clamped_alpha(alpha: f64) -> f64 {
    if alpha.is_nan() {
        0.0
    } else {
        alpha.clamp(0.0, 1.0)
    }
}

fn mix_toward(state: &mut FireState, direction: &Array1<f64>, mix: Option<FireMix>) {
    let (vnorm, dnorm) = match mix {
        Some(m) => (m.vel, m.direction),
        None => (l2(&state.vel), l2(direction)),
    };
    if !(dnorm.is_finite() && dnorm > 0.0 && vnorm.is_finite() && vnorm >= 0.0) {
        return;
    }
    // The mix is a convex combination only for alpha in [0, 1]; the
    // norm bound |1 - 2 alpha| |v| <= |v'| <= |v| needs it, so the
    // public field is clamped where it is used.
    let alpha = clamped_alpha(state.alpha);
    let scale = alpha * vnorm / dnorm;
    for i in 0..state.vel.len() {
        state.vel[i] = (1.0 - alpha) * state.vel[i] + scale * direction[i];
    }
}

fn mix_velocity(state: &mut FireState, force: &Array1<f64>) {
    mix_toward(state, force, None);
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

/// FIRE 1.0 adapt, and the adapt inside [`fire_displacement`] for FIRE 2.0.
///
/// A positive power is downhill. Zero and negative power reset the
/// velocity. The shrink floor is `1e-12`, the rgmin integrator's floor;
/// [`fire_prepare_v2`] uses [`FireState`]'s own `dt_min` instead.
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

/// FIRE 2.0 preparation before the forward integrator.
pub(crate) enum FirePreparation {
    /// Continue from the current point.
    Ready,
    /// Apply the half-step-back correction and refresh the force.
    Rollback(Array1<f64>),
    /// Consecutive non-positive steps exceeded the limit. Position,
    /// `dt`, `alpha`, and velocity are unchanged.
    Stalled,
}

/// Prepare a FIRE 2.0 step.
///
/// The caller must evaluate a fresh force after applying a nonzero
/// correction, before passing that force to [`fire_displacement_into`].
/// A step that makes the consecutive non-positive count exceed the
/// limit returns [`FirePreparation::Stalled`] before the shrink and the
/// half-step. [`FirePreset::Guenole2020`] uses the published initial
/// delay: during the first `N_delay` steps an uphill step neither
/// shrinks `dt` nor resets `alpha`.
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
        return FirePreparation::Ready;
    }
    state.n_pos = 0;
    state.n_neg = state.n_neg.saturating_add(1);
    if state.n_neg > state.n_neg_max {
        return FirePreparation::Stalled;
    }
    match state.preset {
        FirePreset::Guenole2020 => {
            if state.steps >= state.n_min {
                let shrunk = state.dt * state.f_dec;
                if shrunk >= state.dt_min {
                    state.dt = shrunk;
                }
                state.alpha = state.alpha_start;
            }
        }
        FirePreset::Default => {
            if state.steps + 1 >= state.n_min {
                state.dt = (state.dt * state.f_dec).max(state.dt_min);
            }
            state.alpha = state.alpha_start;
        }
    }
    let correction = -0.5 * state.dt * &state.vel;
    let moved = correction.iter().any(|v| *v != 0.0);
    state.vel.fill(0.0);
    if moved {
        FirePreparation::Rollback(correction)
    } else {
        FirePreparation::Ready
    }
}

/// Displacement `dx` from the current force `f = -g`.
///
/// FIRE 2.0 mixes and adapts on this force, then takes a semi-implicit
/// Euler step. This is the rgmin integrator, not Guénolé algorithm 2:
/// there is no half-step and no non-positive-power limit. FIRE 1.0 takes
/// the Verlet step first; the caller mixes on the force at the new point
/// via [`fire_after_v1`]. The published step is [`fire2_displacement`],
/// or [`fire_prepare_v2`] plus [`fire_displacement_into`].
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
            let mut dx = Array1::zeros(state.vel.len());
            fire_displacement_into(state, force, &mut dx);
            dx
        }
    }
}

/// Forward displacement after [`fire_prepare_v2`], written into `dx`.
///
/// FIRE 2.0 applies the semi-implicit Euler update, then mixes, then
/// the ABC correction when [`FireState::abc`] is set, and counts the
/// step. FIRE 1.0 only takes the Verlet kick.
pub fn fire_displacement_into(state: &mut FireState, force: &Array1<f64>, dx: &mut Array1<f64>) {
    if dx.len() != state.vel.len() {
        *dx = Array1::zeros(state.vel.len());
    }
    match state.kind {
        FireKind::V2 => {
            for i in 0..state.vel.len() {
                state.vel[i] += force[i] * state.dt;
            }
            abc_floor(state);
            mix_toward(state, force, None);
            abc_correct(state);
            state.steps += 1;
        }
        FireKind::V1 => {
            for i in 0..state.vel.len() {
                state.vel[i] += force[i] * state.dt;
            }
        }
    }
    let dt = state.dt;
    for i in 0..state.vel.len() {
        dx[i] = state.vel[i] * dt;
    }
}

/// ABC-FIRE cap: clip each velocity component to `cap / dt` and return
/// the displacement `dt v`, as ASE `FIRE2` does with `use_abc`.
#[allow(dead_code)]
pub(crate) fn fire_clip_components(state: &mut FireState, cap: f64) -> Array1<f64> {
    let vmax = cap / state.dt;
    state.vel.mapv_inplace(|v| v.max(-vmax).min(vmax));
    &state.vel * state.dt
}

/// Keep the velocity consistent with the displacement the caller took.
///
/// `proposed` is the displacement the integrator returned, so
/// `vel = proposed / dt`; `taken` is what survived the maxmove clamp
/// and the bounds clip. When the clamp shortened the move, the velocity
/// shrinks by the same norm ratio, so the power `F . v` that the next
/// step adapts on measures the motion that happened. Without this the
/// integral keeps growing while the clamp fires every step, and `dt`
/// grows with it.
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
    fire_after_v1_metric(state, force, force, None);
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
/// `F · v`, because that is `dE/dt` up to a sign. `steepest` is the
/// direction the velocity is mixed toward (`P^{-1} F` when a metric is
/// installed, otherwise `F`). `mix` supplies the norms; `None` uses the
/// Euclidean 2-norm, which is the unpreconditioned algorithm.
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

/// Which FIRE schedule a session runs.
///
/// Selected with [`crate::Solver::set_fire_variant`]; the default keeps
/// the [`FireKind`] the session was built with and its parameters.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FireVariant {
    /// [`FireKind::V1`] or [`FireKind::V2`] with [`FireState::new`]
    /// parameters. FIRE 2.0 then uses [`fire_displacement`].
    #[default]
    Rgmin,
    /// FIRE 2.0 exactly as Guénolé et al. 2020 algorithm 2 (semi-implicit
    /// Euler) publishes it, with its Table 1 parameters (Comput. Mater.
    /// Sci. 175 (2020) 109584, DOI 10.1016/j.commatsci.2020.109584): see
    /// [`fire2_displacement`].
    Guenole2020,
}

/// The FIRE 2.0 state that [`FireState`] does not carry.
///
/// Built only through [`Fire2Extras::new`], so fields can be added
/// without breaking callers. [`fire2_displacement`] reads this. The
/// preset integrator ([`fire_prepare_v2`]) keeps its own counter.
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
    /// Consecutive steps whose power `P = F · v` is not positive.
    pub n_neg: usize,
    /// Stop on the step that makes [`Self::n_neg`] exceed this limit.
    ///
    /// `usize::MAX` does not stop.
    pub n_neg_max: usize,
}

impl Fire2Extras {
    /// Table 1 of Guénolé et al., Comput. Mater. Sci. 175 (2020) 109584,
    /// DOI 10.1016/j.commatsci.2020.109584: `dt_min = 0.02 dt0`,
    /// `N_delay = 20`, initial delay on, `vdfmax = 2000`.
    #[must_use]
    pub fn new(dt0: f64) -> Self {
        let dt0 = if dt0 > 0.0 { dt0 } else { 0.1 };
        Self {
            dt_min: 0.02 * dt0,
            n_delay: 20,
            initial_delay: true,
            steps: 0,
            n_neg: 0,
            n_neg_max: 2_000,
        }
    }
}

/// FIRE 2.0 stopped because consecutive non-positive power exceeded
/// [`Fire2Extras::n_neg_max`]. That step does not change the position,
/// the time step, the mixing parameter, or the velocity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fire2Stalled;

/// A [`FireState`] and [`Fire2Extras`] with Table 1 of Guénolé et al.,
/// Comput. Mater. Sci. 175 (2020) 109584,
/// DOI 10.1016/j.commatsci.2020.109584: `alpha_start = 0.25`,
/// `f_alpha = 0.99`, `f_inc = 1.1`, `f_dec = 0.5`, `N_delay = 20`,
/// `dt_max = 10 dt0`, `dt_min = 0.02 dt0`.
#[must_use]
pub fn guenole2020(dim: usize, dt0: f64) -> (FireState, Fire2Extras) {
    let mut state = FireState::new(FireKind::V2, dim, dt0);
    state.set_preset(FirePreset::Guenole2020);
    let ext = Fire2Extras::new(state.dt);
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
/// lowers `f . v` (`validation/fire2.py`). A step that makes the
/// consecutive non-positive count exceed [`Fire2Extras::n_neg_max`]
/// returns [`Fire2Stalled`] before the shrink and the half-step.
///
/// Guénolé, Nöhring, Vaid, Houllé, Xie, Prakash, Bitzek, *Assessment
/// and optimization of the fast inertial relaxation engine (FIRE)*,
/// <https://doi.org/10.1016/j.commatsci.2020.109584>.
pub fn fire2_displacement(
    state: &mut FireState,
    ext: &mut Fire2Extras,
    force: &Array1<f64>,
) -> Result<Array1<f64>, Fire2Stalled> {
    fire2_displacement_metric(state, ext, force, force, None)
}

/// FIRE 2.0 with the physical force separated from the steepest direction.
///
/// `power_force` decides uphill versus downhill (`F · v`). `steepest` is
/// integrated and mixed in. `mix` is the same norm pair as
/// [`fire_after_v1_metric`].
pub fn fire2_displacement_metric(
    state: &mut FireState,
    ext: &mut Fire2Extras,
    power_force: &Array1<f64>,
    steepest: &Array1<f64>,
    mix: Option<FireMix>,
) -> Result<Array1<f64>, Fire2Stalled> {
    let power = crate::vecops::dot(power_force.view(), state.vel.view());
    let mut dx = Array1::zeros(state.vel.len());
    if power > 0.0 {
        ext.n_neg = 0;
        state.n_pos += 1;
        if state.n_pos > ext.n_delay {
            state.dt = (state.dt * state.f_inc).min(state.dt_max);
            state.alpha *= state.f_alpha;
        }
    } else {
        state.n_pos = 0;
        ext.n_neg = ext.n_neg.saturating_add(1);
        if ext.n_neg > ext.n_neg_max {
            return Err(Fire2Stalled);
        }
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
    crate::vecops::axpy(state.dt, steepest.view(), &mut state.vel);
    mix_toward(state, steepest, mix);
    dx.scaled_add(state.dt, &state.vel);
    Ok(dx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, ArrayView1, array};

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
        let taken = &dx * 0.01;
        fire_rescale_velocity(&mut state, &dx, &taken);
        for i in 0..2 {
            assert!(
                (state.vel[i] * state.dt - taken[i]).abs() <= 1e-12 * taken[i].abs().max(1.0),
                "vel {} dt {} taken {}",
                state.vel[i],
                state.dt,
                taken[i]
            );
        }
        let before = state.vel.clone();
        fire_rescale_velocity(&mut state, &taken, &taken);
        assert_eq!(state.vel, before);
    }

    #[test]
    fn guenole_uphill_steps_back_half_a_step() {
        let (mut state, mut ext) = guenole2020(2, 0.2);
        ext.steps = ext.n_delay;
        state.vel = array![1.0, 0.0];
        state.alpha = 0.05;
        let force = array![-1.0, 0.5];
        let dx = fire2_displacement(&mut state, &mut ext, &force).expect("within vdfmax");
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
        state.alpha = 0.05;
        let _ = fire2_displacement(&mut state, &mut ext, &force).expect("within vdfmax");
        assert!((state.dt - 0.2).abs() < 1e-15, "the delay shrank dt");
        assert!((state.alpha - 0.05).abs() < 1e-15, "the delay reset alpha");
        ext.steps = ext.n_delay;
        state.dt = 1.5 * ext.dt_min;
        state.vel = array![1.0, 0.0];
        let _ = fire2_displacement(&mut state, &mut ext, &force).expect("within vdfmax");
        assert!(
            (state.dt - 1.5 * ext.dt_min).abs() < 1e-15,
            "dt went under dt_min"
        );
    }

    #[test]
    fn guenole_downhill_mixes_after_the_kick() {
        let (mut state, mut ext) = guenole2020(2, 0.1);
        state.vel = array![1.0, 0.0];
        let f = array![0.0, 1.0];
        let dx = fire2_displacement(&mut state, &mut ext, &f).expect("within vdfmax");
        assert!((dx[0] + 0.05).abs() < 1e-15 && (dx[1] - 0.01).abs() < 1e-15);
        let dx = fire2_displacement(&mut state, &mut ext, &f).expect("within vdfmax");
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

    /// A force orthogonal to the velocity has power zero. The released
    /// adapt treats that as uphill and clears the velocity.
    #[test]
    fn v1_zero_power_resets_the_velocity() {
        let mut state = FireState::new(FireKind::V1, 2, 0.1);
        state.vel = array![0.0, 1.0];
        fire_after_v1(&mut state, &array![1.0, 0.0]);
        assert_eq!(state.n_pos, 0);
        assert_eq!(state.vel, array![0.0, 0.0]);
    }

    #[test]
    fn v1_power_uses_the_physical_force() {
        let mut state = FireState::new(FireKind::V1, 2, 0.1);
        state.vel = array![1.0, 0.0];
        state.n_pos = 6;
        let raw = array![1.0, 0.0];
        let steepest = array![0.0, 4.0];
        fire_after_v1_metric(&mut state, &raw, &steepest, None);
        assert!(state.vel[0] > 0.0, "velocity was reset: {}", state.vel);
        assert!(state.vel[1] > 0.0, "mix missed the metric direction");
        assert!(state.dt > 0.1, "downhill step did not grow dt");
    }

    #[test]
    fn the_metric_mix_matches_p_norms() {
        let mut state = FireState::new(FireKind::V1, 2, 0.1);
        state.vel = array![1.0, 0.0];
        state.alpha = 0.1;
        let steepest = array![0.0, 0.1];
        fire_after_v1_metric(
            &mut state,
            &array![1.0, 0.0],
            &steepest,
            Some(FireMix {
                vel: 1.0,
                direction: 1.0,
            }),
        );
        assert!((state.vel[0] - 0.9).abs() < 1e-12);
        assert!((state.vel[1] - 0.01).abs() < 1e-12);
    }

    #[test]
    fn table_1_vdfmax_stops_on_the_step_past_2000() {
        let (mut state, mut ext) = guenole2020(1, 0.1);
        assert_eq!(ext.n_neg_max, 2_000);
        let force = array![-1.0];
        for i in 1..=2_000 {
            state.vel[0] = 1.0;
            fire2_displacement(&mut state, &mut ext, &force).expect("within vdfmax");
            assert_eq!(ext.n_neg, i);
        }
        state.vel[0] = 1.0;
        let dt = state.dt;
        let alpha = state.alpha;
        assert_eq!(
            fire2_displacement(&mut state, &mut ext, &force),
            Err(Fire2Stalled)
        );
        assert_eq!(ext.n_neg, 2_001);
        assert_eq!(state.dt, dt);
        assert_eq!(state.alpha, alpha);
        assert_eq!(state.vel[0], 1.0);
    }

    #[test]
    fn vdfmax_stops_before_the_half_step_and_positive_power_resets() {
        let (mut state, mut ext) = guenole2020(1, 0.2);
        ext.n_neg_max = 1;
        ext.steps = ext.n_delay;
        state.vel[0] = 1.0;
        state.alpha = 0.05;
        let up = array![-1.0];
        let dx = fire2_displacement(&mut state, &mut ext, &up).expect("first uphill step");
        assert_eq!(ext.n_neg, 1);
        assert!(dx[0].is_finite());
        state.vel[0] = 1.0;
        state.alpha = 0.05;
        let dt = state.dt;
        assert_eq!(
            fire2_displacement(&mut state, &mut ext, &up),
            Err(Fire2Stalled)
        );
        assert_eq!(ext.n_neg, 2);
        assert_eq!(state.dt, dt);
        assert_eq!(state.alpha, 0.05);
        assert_eq!(state.vel[0], 1.0);
        let _ = fire2_displacement(&mut state, &mut ext, &array![1.0]).expect("downhill");
        assert_eq!(ext.n_neg, 0);
    }

    #[test]
    fn prepare_rolls_an_uphill_default_step_back_without_an_early_shrink() {
        let mut state = FireState::new(FireKind::V2, 2, 0.2);
        state.vel = array![1.0, 0.0];
        let force = array![-1.0, 0.0];
        let FirePreparation::Rollback(correction) = fire_prepare_v2(&mut state, &force) else {
            panic!("expected rollback");
        };
        let mut dx = Array1::zeros(2);
        fire_displacement_into(&mut state, &force, &mut dx);
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
    fn v2_stops_before_the_shrink_and_the_half_step() {
        let mut state = FireState::new(FireKind::V2, 1, 0.1);
        state.set_preset(FirePreset::Guenole2020);
        state.n_neg = state.n_neg_max;
        state.steps = state.n_min;
        state.vel[0] = 1.0;
        state.alpha = 0.05;
        let dt = state.dt;
        assert!(matches!(
            fire_prepare_v2(&mut state, &array![-1.0]),
            FirePreparation::Stalled
        ));
        assert_eq!(state.n_neg, state.n_neg_max + 1);
        assert_eq!(state.dt, dt);
        assert_eq!(state.alpha, 0.05);
        assert_eq!(state.vel[0], 1.0);
    }

    #[test]
    fn guenole_preset_initial_delay_does_not_shrink_or_reset_alpha() {
        let mut state = FireState::new(FireKind::V2, 1, 0.2);
        state.set_preset(FirePreset::Guenole2020);
        state.vel[0] = 1.0;
        state.alpha = 0.05;
        let FirePreparation::Rollback(correction) = fire_prepare_v2(&mut state, &array![-1.0])
        else {
            panic!("expected the half-step");
        };
        assert!((state.dt - 0.2).abs() < 1e-15);
        assert!((state.alpha - 0.05).abs() < 1e-15);
        assert_eq!(state.vel[0], 0.0);
        assert!((correction[0] + 0.1).abs() < 1e-15);
    }

    #[test]
    fn guenole_preset_carries_table_1_and_skips_a_shrink_past_dt_min() {
        let mut state = FireState::new(FireKind::V2, 1, 1.0);
        state.set_preset(FirePreset::Guenole2020);
        assert_eq!((state.dt_max, state.dt_min, state.n_min), (10.0, 0.02, 20));
        assert_eq!((state.alpha, state.f_inc, state.f_dec), (0.25, 1.1, 0.5));
        assert_eq!((state.f_alpha, state.n_neg_max), (0.99, 2_000));
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

    fn gamma(m: f64) -> f64 {
        let u = f64::EPSILON / 2.0;
        m * u / (1.0 - m * u)
    }

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
    fn abc_divides_the_mix_by_its_bias() {
        let mut state = FireState::new(FireKind::V2, 2, 0.1);
        state.set_preset(FirePreset::Guenole2020);
        state.abc = true;
        state.vel = array![1.0, 0.0];
        let force = array![1.0, 1.0];
        let ready = fire_prepare_v2(&mut state, &force);
        assert!(matches!(ready, FirePreparation::Ready) && state.n_pos == 1);
        let mut dx = Array1::zeros(2);
        fire_displacement_into(&mut state, &force, &mut dx);
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

    fn ase_step(state: &mut FireState, x: &mut Array1<f64>, cap: f64) -> (f64, Array1<f64>) {
        let mut force = -rosenbrock(x.view()).1;
        if let FirePreparation::Rollback(c) = fire_prepare_v2(state, &force) {
            *x += &c;
            force = -rosenbrock(x.view()).1;
        }
        let kick = l2(&(&state.vel + &(&force * state.dt)));
        let mut dx = Array1::zeros(x.len());
        fire_displacement_into(state, &force, &mut dx);
        let dx = if state.abc {
            fire_clip_components(state, cap)
        } else {
            dx
        };
        (kick, dx)
    }

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

    #[test]
    fn ase_config_caps_the_time_step_at_one() {
        let mut state = FireState::new(FireKind::V1, 3, 0.1);
        assert!((state.dt_max - 0.25).abs() < 1e-15);
        state.apply_config(&FireConfig::ase());
        assert!((state.dt - 0.1).abs() < 1e-15);
        assert!((state.dt_max - 1.0).abs() < 1e-15);
        assert_eq!(state.n_min, 5);
        assert!((state.f_alpha - 0.99).abs() < 1e-15);
    }
}
