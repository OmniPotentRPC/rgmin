//! Quick-min: project the velocity onto the force, then take an Euler step.
//!
//! The reference update, [`quickmin_baseline`], is the one a fixed force
//! sequence is checked against. Where `v` is the velocity, `F` the force
//! (`F = -∇f`) and `Δt` the time step:
//!
//! ```text
//! v ← (v · F / |F|²) F    if v · F ≥ 0 and |F| > 0
//! v ← 0                   otherwise
//! v ← v + Δt F
//! Δx = Δt v
//! ```
//!
//! A negative projection is the second branch, so the stored velocity
//! keeps none of the component that opposed `F`. A zero force has no
//! direction and takes the same branch.
//!
//! The session runs three departures from that reference. Each one is
//! the piece of machinery the other first-order methods already use,
//! and each one has a test on a named objective.
//!
//! 1. Displacement cap, then the FIRE velocity rescale. [`crate::step::scale_step_atom`]
//!    (or [`crate::Control::maxmove`]) shortens `Δx`. The velocity that
//!    entered the step was `Δx_proposed / Δt`. [`quickmin_rescale_velocity`]
//!    multiplies it by `|Δx_taken| / |Δx_proposed|` when the cap binds,
//!    which is the same ratio as [`crate::fire::fire_rescale_velocity`].
//!    Without it the next projection treats motion the cap removed as
//!    still present. On `E = ½(y − 0.2 x)² + 0.02 x²` that leftover
//!    keeps the capped run above the rescaled one.
//! 2. Time-step control taken from FIRE's factors (`n_min = 5`,
//!    `f_inc = 1.1`, `f_dec = 0.5`, `Δt_max = 2.5 Δt₀`). After more
//!    than `n_min` steps with `F · v > 0`, `Δt` grows up to the cap.
//!    `F · v < 0` shrinks `Δt` (floor `1e-12`); the velocity on that
//!    branch is already zero from the reference rule. `F · v = 0` does
//!    not shrink `Δt`: a resting velocity is not an uphill step, and
//!    cutting the caller's `Δt` before any motion only slows the first
//!    steps. On `E = ½ x²` from `x = 1` with `Δt₀ = 0.05` the growing
//!    step reaches `|x| < 10⁻³` in fewer iterations than the fixed step.
//! 3. A cell block. [`QuickMinState::cell_at`] splits the vector into
//!    atoms `[0, at)` and cell `[at, n)`. The reference replacement
//!    runs on each block, so a cell velocity that opposes the stress is
//!    zeroed even when the atomic block keeps the total power positive.
//!    One joint projection would spend atomic momentum on that opposing
//!    cell direction. On `E = ½(x² + c²)` with `v = (3, 1)` and
//!    `F = (1, −1)` the split step reaches a lower value.

use ndarray::Array1;

/// Session memory for one quick-min solve.
#[derive(Clone, Debug)]
pub struct QuickMinState {
    /// Velocity, one entry per coordinate.
    pub vel: Array1<f64>,
    /// Time step. The Euler update and the displacement both use it.
    pub dt: f64,
    /// Largest time step the controller may grow to.
    pub dt_max: f64,
    /// Consecutive steps with `F · v > 0`.
    pub n_pos: usize,
    /// Delay before [`Self::dt`] grows. FIRE's `N_min`.
    pub n_min: usize,
    /// Growth factor for [`Self::dt`].
    pub f_inc: f64,
    /// Shrink factor for [`Self::dt`].
    pub f_dec: f64,
    /// Start of the cell block. `None` projects the whole vector, which
    /// is the reference rule.
    pub cell_at: Option<usize>,
    /// Grow and shrink [`Self::dt`]. The reference update leaves this off.
    pub adapt_dt: bool,
}

impl QuickMinState {
    /// FIRE's time-step factors, adaptation on, one velocity for the
    /// whole vector. `dt` is [`crate::Control::istep`].
    pub fn new(dim: usize, dt: f64) -> Self {
        let dt = if dt.is_finite() && dt > 0.0 { dt } else { 0.1 };
        Self {
            vel: Array1::zeros(dim),
            dt,
            dt_max: (dt * 2.5).max(dt),
            n_pos: 0,
            n_min: 5,
            f_inc: 1.1,
            f_dec: 0.5,
            cell_at: None,
            adapt_dt: true,
        }
    }

    /// The reference update: fixed `dt`, one projection, no cell split.
    pub fn reference(dim: usize, dt: f64) -> Self {
        let mut state = Self::new(dim, dt);
        state.adapt_dt = false;
        state
    }

    /// Drop the velocity and the growth counter. The current `dt` stays,
    /// as a FIRE reset keeps the step it has already learned.
    pub fn reset(&mut self) {
        self.vel.fill(0.0);
        self.n_pos = 0;
    }
}

fn block_power(vel: &Array1<f64>, force: &Array1<f64>, start: usize, end: usize) -> f64 {
    let mut power = 0.0;
    for i in start..end {
        power += vel[i] * force[i];
    }
    power
}

/// Replace `vel[start..end]` by its projection on `force` when that
/// projection is non-negative, and by zero otherwise.
fn project_range(vel: &mut Array1<f64>, force: &Array1<f64>, start: usize, end: usize) {
    let mut power = 0.0;
    let mut fn2 = 0.0;
    for i in start..end {
        power += vel[i] * force[i];
        fn2 += force[i] * force[i];
    }
    if !matches!(
        power.partial_cmp(&0.0),
        Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
    ) || fn2.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater)
    {
        for i in start..end {
            vel[i] = 0.0;
        }
        return;
    }
    let scale = power / fn2;
    for i in start..end {
        vel[i] = scale * force[i];
    }
}

/// Reference replacement on the whole vector, before the Euler kick.
///
/// The result is parallel to `force` or zero. An opposing velocity
/// contributes nothing.
pub fn quickmin_project(velocity: &Array1<f64>, force: &Array1<f64>) -> Array1<f64> {
    if velocity.len() != force.len() {
        return Array1::zeros(velocity.len());
    }
    let mut vel = velocity.clone();
    let n = vel.len();
    project_range(&mut vel, force, 0, n);
    vel
}

/// Reference quick-min step: project, add `Δt F`, displace by `Δt v`.
///
/// The pair is `(velocity after the kick, displacement)`. No cap, no
/// time-step change, and no cell split: those are session departures.
pub fn quickmin_baseline(
    velocity: &Array1<f64>,
    force: &Array1<f64>,
    dt: f64,
) -> (Array1<f64>, Array1<f64>) {
    let mut state = QuickMinState::reference(velocity.len(), dt);
    if velocity.len() == force.len() {
        state.vel.assign(velocity);
    }
    let dx = quickmin_displacement(&mut state, force);
    (state.vel, dx)
}

/// One quick-min displacement, including the cell split and the
/// time-step controller when the state asks for them.
///
/// The cap is applied by the caller to the returned displacement.
/// [`quickmin_rescale_velocity`] then puts the velocity on that
/// shortened step.
pub fn quickmin_displacement(state: &mut QuickMinState, force: &Array1<f64>) -> Array1<f64> {
    displacement_blocks(state, force, false)
}

/// [`quickmin_displacement`], with each xyz triple of the atomic block
/// projected on its own force when `per_atom` is set.
fn displacement_blocks(
    state: &mut QuickMinState,
    force: &Array1<f64>,
    per_atom: bool,
) -> Array1<f64> {
    let n = state.vel.len();
    if force.len() != n || !state.dt.is_finite() || state.dt <= 0.0 {
        state.vel.fill(0.0);
        return Array1::zeros(n);
    }
    let ranges = match state.cell_at {
        Some(at) if at > 0 && at < n => [(0, at), (at, n)],
        _ => [(0, n), (0, 0)],
    };
    let mut power = 0.0;
    for (k, (start, end)) in ranges.into_iter().enumerate() {
        if start == end {
            continue;
        }
        power += block_power(&state.vel, force, start, end);
        if per_atom && k == 0 {
            let mut a = start;
            while a < end {
                let b = (a + 3).min(end);
                project_range(&mut state.vel, force, a, b);
                a = b;
            }
        } else {
            project_range(&mut state.vel, force, start, end);
        }
    }
    if state.adapt_dt {
        if power > 0.0 {
            state.n_pos += 1;
            if state.n_pos > state.n_min {
                state.dt = (state.dt * state.f_inc).min(state.dt_max);
            }
        } else if power < 0.0 {
            state.dt = (state.dt * state.f_dec).max(1e-12);
            state.n_pos = 0;
        }
    }
    let dt = state.dt;
    for i in 0..n {
        state.vel[i] += dt * force[i];
    }
    &state.vel * dt
}

/// Shrink the velocity to the displacement the caller kept.
///
/// `proposed` is the displacement [`quickmin_displacement`] returned
/// and `taken` is what remained after the cap and the bounds. The
/// factor is `|taken| / |proposed|` when that ratio is below one, the
/// same ratio [`crate::fire::fire_rescale_velocity`] applies.
pub fn quickmin_rescale_velocity(
    vel: &mut Array1<f64>,
    proposed: &Array1<f64>,
    taken: &Array1<f64>,
) {
    let proposed_norm = crate::step::l2(proposed);
    if proposed_norm <= 0.0 {
        return;
    }
    let factor = crate::step::l2(taken) / proposed_norm;
    if factor < 1.0 {
        vel.mapv_inplace(|v| v * factor);
    }
}

/// What a quick-min session does when the energy rose over its last step.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum QuickMinUphill {
    /// No energy test: the reference rule.
    #[default]
    Ignore,
    /// Zero the velocity and cut `dt` by `f_dec`.
    Restart,
    /// Zero the velocity, cut `dt` by `f_dec`, and step again from the
    /// previous point with its force. At the `dt` floor the step goes on
    /// from the current point.
    Backtrack,
}

/// Quick-min session options, set with [`crate::Solver::set_quickmin_options`].
///
/// Built through [`QuickMinOptions::rgmin`] or [`QuickMinOptions::eon`] so
/// fields can be added without breaking callers.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct QuickMinOptions {
    /// Grow and shrink `dt` with FIRE's factors.
    pub adapt_dt: bool,
    /// `dt_max` as a multiple of the starting `dt`.
    pub dt_max_factor: f64,
    /// Floor under an uphill cut of `dt`, as a multiple of the starting `dt`.
    pub dt_min_factor: f64,
    /// Shrink the velocity to the capped step ([`quickmin_rescale_velocity`]).
    pub rescale_on_clamp: bool,
    /// Project each xyz triple of the atomic block on its own force.
    pub per_atom: bool,
    /// With `Some(c)`, bound `dt <= c / sqrt(|y| / |s|)` from the force
    /// difference `y` over the last step `s`.
    pub curvature_cap: Option<f64>,
    /// Response to an energy rise.
    pub uphill: QuickMinUphill,
}

impl QuickMinOptions {
    /// The session default: FIRE time-step control (`dt_max = 2.5 dt0`),
    /// the capped-step velocity rescale, no energy test.
    #[must_use]
    pub fn rgmin() -> Self {
        Self {
            adapt_dt: true,
            dt_max_factor: 2.5,
            dt_min_factor: 0.02,
            rescale_on_clamp: true,
            per_atom: false,
            curvature_cap: None,
            uphill: QuickMinUphill::Ignore,
        }
    }

    /// eOn `Quickmin::step`: fixed `dt`, one projection, and a capped
    /// step that leaves the velocity alone.
    #[must_use]
    pub fn eon() -> Self {
        Self {
            adapt_dt: false,
            rescale_on_clamp: false,
            ..Self::rgmin()
        }
    }
}

impl Default for QuickMinOptions {
    fn default() -> Self {
        Self::rgmin()
    }
}

/// The point the last quick-min step started from.
#[derive(Clone, Debug)]
struct Anchor {
    x: Array1<f64>,
    value: f64,
    force: Array1<f64>,
}

/// Session-side quick-min memory beyond [`QuickMinState`].
#[derive(Clone, Debug)]
pub(crate) struct QuickMinSession {
    pub(crate) options: QuickMinOptions,
    dt0: f64,
    anchor: Option<Anchor>,
}

impl QuickMinSession {
    pub(crate) fn new(dt0: f64) -> Self {
        let dt0 = if dt0.is_finite() && dt0 > 0.0 {
            dt0
        } else {
            0.1
        };
        Self {
            options: QuickMinOptions::rgmin(),
            dt0,
            anchor: None,
        }
    }

    /// Apply `options` to the state; the velocity and `dt` restart.
    pub(crate) fn configure(&mut self, state: &mut QuickMinState, options: QuickMinOptions) {
        let factor = if options.dt_max_factor.is_finite() {
            options.dt_max_factor.max(1.0)
        } else {
            1.0
        };
        state.adapt_dt = options.adapt_dt;
        state.dt = self.dt0;
        state.dt_max = self.dt0 * factor;
        state.reset();
        self.options = options;
        self.anchor = None;
    }

    /// Forget the previous point: its energy and force may belong to an
    /// objective that changed.
    pub(crate) fn forget_anchor(&mut self) {
        self.anchor = None;
    }

    fn dt_min(&self) -> f64 {
        let f = self.options.dt_min_factor;
        let f = if f.is_nan() { 0.0 } else { f.clamp(0.0, 1.0) };
        (self.dt0 * f).max(1e-12)
    }
}

/// One session step: `(origin, dx)`. The caller caps `dx` against
/// `origin` and evaluates the new point. `slack` is the energy rise the
/// uphill test tolerates.
pub(crate) fn quickmin_session_step(
    state: &mut QuickMinState,
    session: &mut QuickMinSession,
    x: &Array1<f64>,
    value: f64,
    force: &Array1<f64>,
    slack: f64,
) -> (Array1<f64>, Array1<f64>) {
    let opts = session.options.clone();
    let anchor = session.anchor.take();
    let mut backtrack: Option<Anchor> = None;
    if let Some(prev) = anchor {
        let uphill = !value.is_finite() || value - prev.value > slack;
        if uphill && opts.uphill != QuickMinUphill::Ignore {
            state.vel.fill(0.0);
            state.n_pos = 0;
            let floor = session.dt_min();
            let cut = state.dt > floor;
            if cut {
                state.dt = (state.dt * state.f_dec).max(floor);
            }
            if opts.uphill == QuickMinUphill::Backtrack && cut {
                backtrack = Some(prev);
            }
        } else if let Some(c) = opts.curvature_cap.filter(|_| value.is_finite()) {
            let s = x - &prev.x;
            let y = &prev.force - force;
            let ss = crate::vecops::dot(s.view(), s.view());
            let yy = crate::vecops::dot(y.view(), y.view());
            if ss > 0.0 && yy > 0.0 {
                let bound = c / (yy / ss).sqrt().sqrt();
                if bound.is_finite() && bound > 0.0 {
                    state.dt = state.dt.min(bound).max(session.dt_min());
                }
            }
        }
    }
    match backtrack {
        Some(prev) => {
            // From rest at the previous point: v = dt F, dx = dt^2 F.
            let dt = state.dt;
            state.vel.assign(&prev.force);
            state.vel *= dt;
            let dx = &state.vel * dt;
            let origin = prev.x.clone();
            session.anchor = Some(prev);
            (origin, dx)
        }
        None => {
            let dx = displacement_blocks(state, force, opts.per_atom);
            session.anchor = Some(Anchor {
                x: x.clone(),
                value,
                force: force.clone(),
            });
            (x.clone(), dx)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Control, Method, Oracle, Solver};
    use approx::assert_abs_diff_eq;
    use ndarray::{ArrayView1, array};

    #[test]
    fn baseline_projects_then_kicks_and_zeros_an_opposing_velocity() {
        let dt = 0.1;
        let (vel, dx) = quickmin_baseline(&array![1.0, -1.0], &array![2.0, 0.0], dt);
        assert_abs_diff_eq!(vel[0], 1.2, epsilon = 1e-12);
        assert_abs_diff_eq!(vel[1], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(dx[0], 0.12, epsilon = 1e-12);
        assert_abs_diff_eq!(dx[1], 0.0, epsilon = 1e-12);

        let opposed = quickmin_project(&array![-2.0, 0.5], &array![1.0, 0.0]);
        assert_abs_diff_eq!(opposed[0], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(opposed[1], 0.0, epsilon = 1e-15);
        let (vel, dx) = quickmin_baseline(&array![1.2, 0.0], &array![-1.0, 3.0], dt);
        assert_abs_diff_eq!(vel[0], -0.1, epsilon = 1e-12);
        assert_abs_diff_eq!(vel[1], 0.3, epsilon = 1e-12);
        assert_abs_diff_eq!(dx[0], -0.01, epsilon = 1e-12);
        assert_abs_diff_eq!(dx[1], 0.03, epsilon = 1e-12);
    }

    #[test]
    fn the_projection_never_lengthens_and_never_opposes() {
        let mut seed = 12345_u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        for _ in 0..1000 {
            let v = Array1::from_iter((0..6).map(|_| next()));
            let f = Array1::from_iter((0..6).map(|_| next()));
            let p = quickmin_project(&v, &f);
            // validation/quickmin.sollya: B(6) - 1 is 7 eps.
            assert!(crate::step::l2(&p) <= crate::step::l2(&v) * (1.0 + 8.0 * f64::EPSILON));
            assert!(p.dot(&f) >= 0.0);
        }
    }

    #[test]
    fn a_skew_velocity_keeps_only_the_force_direction() {
        let projected = quickmin_project(&array![3.0, 4.0], &array![1.0, 0.0]);
        assert_abs_diff_eq!(projected[0], 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(projected[1], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn split_cell_velocity_drops_an_opposing_block() {
        // E = 1/2 (x^2 + c^2). F = (1, -1) at (-1, 1). The joint
        // projection spends the atomic velocity on the opposing cell
        // direction; the split does not.
        let force = array![1.0, -1.0];
        let start_v = array![3.0, 1.0];
        let dt = 0.1;
        let (joint_v, joint_dx) = quickmin_baseline(&start_v, &force, dt);
        let mut split = QuickMinState::reference(2, dt);
        split.cell_at = Some(1);
        split.vel.assign(&start_v);
        let split_dx = quickmin_displacement(&mut split, &force);
        let joint_x = array![-1.0 + joint_dx[0], 1.0 + joint_dx[1]];
        let split_x = array![-1.0 + split_dx[0], 1.0 + split_dx[1]];
        let energy = |x: &Array1<f64>| 0.5 * (x[0] * x[0] + x[1] * x[1]);
        assert!(
            energy(&split_x) < energy(&joint_x),
            "split {} vs joint {}",
            energy(&split_x),
            energy(&joint_x)
        );
        // The opposing cell velocity is gone before the Euler kick,
        // so the cell component is only dt * F_cell.
        assert_abs_diff_eq!(split.vel[1], dt * force[1], epsilon = 1e-12);
        assert!(joint_v[1] < split.vel[1]);
    }

    fn harmonic_steps(adapt: bool) -> usize {
        let mut state = QuickMinState::new(1, 0.05);
        state.adapt_dt = adapt;
        let mut x = 1.0;
        for step in 1..=200 {
            let force = array![-x];
            let dx = quickmin_displacement(&mut state, &force);
            x += dx[0];
            if x.abs() < 1e-3 {
                return step;
            }
        }
        200
    }

    #[test]
    fn growing_timestep_reaches_the_harmonic_minimum_sooner() {
        let fixed = harmonic_steps(false);
        let adapted = harmonic_steps(true);
        assert!(adapted < fixed, "adapted {adapted} steps, fixed {fixed}");
        assert!(adapted <= 40, "adapted {adapted}");
        assert!(fixed >= 55, "fixed {fixed}");
    }

    fn valley(x: f64, y: f64) -> (f64, [f64; 2]) {
        let slack = y - 0.2 * x;
        let energy = 0.5 * slack * slack + 0.02 * x * x;
        let force = [0.2 * slack - 0.04 * x, -slack];
        (energy, force)
    }

    fn valley_steps(rescale: bool) -> f64 {
        let mut state = QuickMinState::reference(2, 0.2);
        let mut x = array![2.0, 0.0];
        for _ in 0..80 {
            let (_, force) = valley(x[0], x[1]);
            let proposed = quickmin_displacement(&mut state, &array![force[0], force[1]]);
            let mut taken = proposed.clone();
            let norm = (taken[0] * taken[0] + taken[1] * taken[1]).sqrt();
            let cap = 0.08;
            if norm > cap {
                taken *= cap / norm;
                if rescale {
                    quickmin_rescale_velocity(&mut state.vel, &proposed, &taken);
                }
            }
            x += &taken;
        }
        valley(x[0], x[1]).0
    }

    #[test]
    fn rescaling_after_the_cap_lowers_the_valley() {
        let plain = valley_steps(false);
        let rescaled = valley_steps(true);
        assert!(rescaled < plain, "rescaled {rescaled} vs plain {plain}");
        assert!(rescaled < 1e-8, "rescaled {rescaled}");
        assert!(plain > 1e-7, "plain {plain}");
    }

    #[test]
    fn a_session_step_from_rest_is_the_reference_kick() {
        let obj = Oracle::unbounded(1, |x: ArrayView1<f64>| (0.5 * x[0] * x[0], array![x[0]]));
        let mut solver = Solver::new(
            Method::QuickMin,
            Control {
                maxiter: 1,
                gtol: 0.0,
                istep: 0.05,
                maxmove: None,
                ftol_rel: None,
                noise_floor: None,
            },
            1,
        );
        let mut x = array![1.0];
        solver.step(&obj, &mut x).unwrap();
        let (_, dx) = quickmin_baseline(&array![0.0], &array![-1.0], 0.05);
        assert_abs_diff_eq!(x[0], 1.0 + dx[0], epsilon = 1e-12);
        assert!(solver.set_quickmin_cell(0).is_err());
        assert!(solver.set_quickmin_cell(1).is_err());
    }
}
