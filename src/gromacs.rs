//! GROMACS steepest-descent step-size state.
//!
//! The state is deliberately independent of coordinates, units, and the
//! objective. A session supplies the projected force and applies its usual
//! retraction, constraint, and displacement-cap hooks.
//!
//! The update is the normalized-force trial and strict energy test of
//! GROMACS [`do_steep`][source]. A lower energy multiplies the maximum
//! displacement by [`GROMACS_STEEP_GROW`]. A refusal multiplies it by
//! [`GromacsSteepConfig::shrink`], whose default is the 0.5 factor in
//! GROMACS 2025.4 (`minimize.cpp` line 3104). The reference manual still
//! says 0.2; [`GromacsSteepConfig::with_shrink`] selects that factor.
//!
//! [source]: https://github.com/gromacs/gromacs/blob/v2025.4/src/gromacs/mdrun/minimize.cpp#L2861

use ndarray::{Array1, ArrayView1};

use crate::error::{Error, Result};

/// Growth factor after a lower-energy trial.
///
/// GROMACS 2025.4 `do_steep` multiplies `ustep` by this after an accepted
/// step when `count > 0` (`minimize.cpp` line 3092).
pub const GROMACS_STEEP_GROW: f64 = 1.2;

/// Shrink factor after a refused trial.
///
/// GROMACS 2025.4 `do_steep` multiplies `ustep` by this when the trial
/// energy is not strictly lower (`minimize.cpp` line 3104).
pub const GROMACS_STEEP_SHRINK: f64 = 0.5;

/// Canonical GROMACS initial `emstep` used for the static default.
pub const GROMACS_STEEP_EMSTEP_DEFAULT: f64 = 0.01;

/// Upper bound on [`gromacs_steep_max_constraint_shrinks`].
///
/// A `shrink` close enough to one needs an astronomic count of
/// multiplications to bring `step_size` down to
/// [`GROMACS_STEEP_MIN_USTEP`]; past this bound the count is capped
/// rather than returned exactly, so a caller still gets a usable budget
/// in bounded time.
pub const GROMACS_STEEP_SHRINK_BUDGET_CAP: u32 = 1_000_000;

/// GROMACS double precision floor on the steep displacement `ustep`.
///
/// `do_steep` in GROMACS 2025.4 `minimize.cpp` ends the minimization when
/// `ustep` is below this value under `GMX_DOUBLE`. The single precision
/// build uses a looser floor. This stepper is `f64`, so it follows the
/// double precision branch. Thirty four halvings of
/// [`GROMACS_STEEP_EMSTEP_DEFAULT`] land below it.
pub const GROMACS_STEEP_MIN_USTEP: f64 = 1e-12;

/// Largest shrink count from this step size and shrink factor before the
/// displacement falls below [`GROMACS_STEEP_MIN_USTEP`].
///
/// The count is the least `k` with `step_size * shrink.powi(k) <
/// GROMACS_STEEP_MIN_USTEP`, taken in closed form from logarithms rather
/// than by repeated multiplication, so a `shrink` near one returns in
/// bounded time instead of multiplying until a loop counter overflows.
/// The result saturates at [`GROMACS_STEEP_SHRINK_BUDGET_CAP`].
pub fn gromacs_steep_max_constraint_shrinks(step_size: f64, shrink: f64) -> u32 {
    if !(step_size.is_finite()
        && step_size > 0.0
        && shrink.is_finite()
        && shrink > 0.0
        && shrink < 1.0)
    {
        return 0;
    }
    if step_size < GROMACS_STEEP_MIN_USTEP {
        return 0;
    }
    let exact_count = (GROMACS_STEEP_MIN_USTEP / step_size).ln() / shrink.ln();
    if !(exact_count.is_finite() && exact_count >= 0.0) {
        return 0;
    }
    let count = exact_count.floor() + 1.0;
    if !(count.is_finite() && count > 0.0) {
        return 0;
    }
    if count < f64::from(GROMACS_STEEP_SHRINK_BUDGET_CAP) {
        count as u32
    } else {
        GROMACS_STEEP_SHRINK_BUDGET_CAP
    }
}

/// Consecutive constraint failures for a canonical `emstep = 0.01`.
pub const GROMACS_STEEP_MAX_CONSTRAINT_SHRINKS: u32 = 34;

/// Parameters of [`GromacsSteep`].
///
/// The struct is [`non_exhaustive`](https://doc.rust-lang.org/reference/attributes/type_system.html#the-non_exhaustive-attribute):
/// build it with [`Default`] and [`Self::with_shrink`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GromacsSteepConfig {
    /// Factor applied to the maximum displacement when a trial is refused.
    ///
    /// The default is [`GROMACS_STEEP_SHRINK`]. A session accepts a finite
    /// value in the open interval `(0, 1)`.
    pub shrink: f64,
    /// How many consecutive constraint failures shrink the displacement
    /// and keep the old point.
    ///
    /// A failure that leaves the displacement below
    /// [`GROMACS_STEEP_MIN_USTEP`] stops with [`crate::Status::MachinePrecision`].
    /// A failure after this budget, while the displacement is still at
    /// least that floor, is [`Error::Constraint`].
    ///
    /// The default is [`GROMACS_STEEP_MAX_CONSTRAINT_SHRINKS`]. Zero
    /// returns the constraint error on the first failure and does not
    /// shrink, unless the displacement is already below the floor.
    pub max_constraint_shrinks: u32,
    step_size: f64,
    /// Whether [`Self::max_constraint_shrinks`] was set by
    /// [`Self::with_max_constraint_shrinks`] rather than derived.
    ///
    /// A derived budget is stale once attached to a session whose step
    /// size differs from [`Self::step_size`], so [`GromacsSteep::set_config`]
    /// rederives it from the session's own step size; an explicit budget
    /// is kept as given.
    shrink_budget_fixed: bool,
}

impl Default for GromacsSteepConfig {
    fn default() -> Self {
        Self::for_step_size(GROMACS_STEEP_EMSTEP_DEFAULT)
    }
}

impl GromacsSteepConfig {
    /// Parameters with a shrink budget derived from `step_size`.
    pub fn for_step_size(step_size: f64) -> Self {
        Self {
            shrink: GROMACS_STEEP_SHRINK,
            max_constraint_shrinks: gromacs_steep_max_constraint_shrinks(
                step_size,
                GROMACS_STEEP_SHRINK,
            ),
            step_size,
            shrink_budget_fixed: false,
        }
    }

    /// Replace [`Self::shrink`] and, unless [`Self::with_max_constraint_shrinks`]
    /// already fixed [`Self::max_constraint_shrinks`], recompute it from
    /// the step size this configuration was built for.
    ///
    /// Chain it after `default()` or [`Self::for_step_size`]. A config
    /// built from `default()` and attached to a session by
    /// [`GromacsSteep::set_config`] has its budget rederived there from
    /// the session's own step size, so the recomputation here only
    /// matters for a config read back before that attachment.
    #[must_use]
    pub fn with_shrink(mut self, shrink: f64) -> Self {
        self.shrink = shrink;
        if !self.shrink_budget_fixed {
            self.max_constraint_shrinks =
                gromacs_steep_max_constraint_shrinks(self.step_size, shrink);
        }
        self
    }

    /// Replace [`Self::max_constraint_shrinks`] and fix it there.
    ///
    /// A fixed budget is kept as given by [`Self::with_shrink`] and by
    /// [`GromacsSteep::set_config`], instead of being rederived from a
    /// step size.
    #[must_use]
    pub fn with_max_constraint_shrinks(mut self, max_constraint_shrinks: u32) -> Self {
        self.max_constraint_shrinks = max_constraint_shrinks;
        self.shrink_budget_fixed = true;
        self
    }

    /// Rederive [`Self::max_constraint_shrinks`] from `step_size` unless
    /// [`Self::with_max_constraint_shrinks`] fixed it.
    ///
    /// [`GromacsSteep::set_config`] calls this with the session's own
    /// step size, so a config built from `default()` or
    /// [`Self::for_step_size`] with a different step size still gets a
    /// budget sized for the session it attaches to.
    #[must_use]
    pub(crate) fn rebased_shrink_budget(mut self, step_size: f64) -> Self {
        if !self.shrink_budget_fixed {
            self.max_constraint_shrinks =
                gromacs_steep_max_constraint_shrinks(step_size, self.shrink);
        }
        self
    }

    pub(crate) fn validate(self) -> Result<Self> {
        if self.shrink.is_finite() && self.shrink > 0.0 && self.shrink < 1.0 {
            Ok(self)
        } else {
            Err(Error::Steep {
                what: "shrink must be finite and in (0, 1)",
            })
        }
    }
}

/// Adaptive maximum-displacement state for GROMACS `integrator = steep`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GromacsSteep {
    step_size: f64,
    config: GromacsSteepConfig,
    constraint_shrinks: u32,
}

fn max_atom_norm_stable(gradient: ArrayView1<'_, f64>) -> f64 {
    let mut worst = 0.0_f64;
    let mut i = 0;
    while i < gradient.len() {
        let end = (i + 3).min(gradient.len());
        let mut scale = 0.0_f64;
        let mut sum = 1.0_f64;
        for k in i..end {
            let value = gradient[k].abs();
            if !value.is_finite() {
                return f64::INFINITY;
            }
            if value != 0.0 {
                if scale < value {
                    let ratio = scale / value;
                    sum = 1.0 + sum * ratio * ratio;
                    scale = value;
                } else {
                    let ratio = value / scale;
                    sum += ratio * ratio;
                }
            }
        }
        let norm = if scale == 0.0 {
            0.0
        } else {
            scale * sum.sqrt()
        };
        worst = worst.max(norm);
        i = end;
    }
    worst
}

impl GromacsSteep {
    /// Start with maximum per-atom displacement `step_size` and
    /// [`GromacsSteepConfig::default`].
    pub fn new(step_size: f64) -> Self {
        Self {
            step_size,
            config: GromacsSteepConfig::for_step_size(step_size),
            constraint_shrinks: 0,
        }
    }

    /// Current maximum per-atom displacement.
    pub fn step_size(self) -> f64 {
        self.step_size
    }

    /// Parameters that scale a refused trial.
    pub fn config(self) -> GromacsSteepConfig {
        self.config
    }

    /// Replace the parameters. The current displacement is kept and the
    /// consecutive-failure count restarts.
    ///
    /// A `max_constraint_shrinks` derived from `config`'s own step size,
    /// for example by `default().with_shrink(...)`, is rederived here
    /// from [`Self::step_size`] instead, so a config built for a
    /// different step size still gets a budget sized for this one.
    /// [`GromacsSteepConfig::with_max_constraint_shrinks`] fixes an
    /// explicit value against that rederivation.
    pub fn set_config(&mut self, config: GromacsSteepConfig) -> Result<()> {
        self.config = config.rebased_shrink_budget(self.step_size).validate()?;
        self.constraint_shrinks = 0;
        Ok(())
    }

    /// Reset the maximum displacement and the constraint-failure count.
    /// The shrink factor is kept.
    pub fn reset(&mut self, step_size: f64) {
        self.step_size = step_size;
        self.constraint_shrinks = 0;
    }

    /// Shrink after a trial the constraints could not meet.
    ///
    /// Returns true when this failure is past
    /// [`GromacsSteepConfig::max_constraint_shrinks`]. The displacement
    /// is multiplied by the shrink factor on each counted failure, and
    /// left unchanged when the bound is already spent or is zero.
    pub fn reject_constraint(&mut self) -> bool {
        if self.constraint_shrinks >= self.config.max_constraint_shrinks {
            return true;
        }
        self.step_size *= self.config.shrink;
        self.constraint_shrinks += 1;
        false
    }

    /// A placement met the constraints, so the next failure starts a new run.
    pub fn clear_constraint_shrinks(&mut self) {
        self.constraint_shrinks = 0;
    }

    /// Displacement `-h g / max_atom_norm(g)`, or zero at a stationary point.
    pub fn displacement(self, gradient: ArrayView1<'_, f64>) -> Array1<f64> {
        let norm = max_atom_norm_stable(gradient);
        if !(norm.is_finite() && norm > 0.0 && self.step_size.is_finite() && self.step_size > 0.0) {
            return Array1::zeros(gradient.len());
        }
        gradient.mapv(|g| -self.step_size * g / norm)
    }

    /// Accept a strict energy decrease and grow by [`GROMACS_STEEP_GROW`].
    ///
    /// Any other trial is refused and the displacement is multiplied by
    /// [`GromacsSteepConfig::shrink`].
    pub fn update(&mut self, energy: f64, trial_energy: f64) -> bool {
        let accepted = trial_energy < energy;
        self.step_size *= if accepted {
            GROMACS_STEEP_GROW
        } else {
            self.config.shrink
        };
        accepted
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn strict_decrease_grows_and_a_refusal_uses_the_gromacs_shrink() {
        let mut state = GromacsSteep::new(0.01);
        let grown = 0.01 * GROMACS_STEEP_GROW;
        let once = grown * GROMACS_STEEP_SHRINK;
        let twice = once * GROMACS_STEEP_SHRINK;
        assert!(state.update(2.0, 1.0));
        assert_eq!(state.step_size().to_bits(), grown.to_bits());
        assert!(!state.update(1.0, 1.0));
        assert_eq!(state.step_size().to_bits(), once.to_bits());
        assert!(!state.update(1.0, 2.0));
        assert_eq!(state.step_size().to_bits(), twice.to_bits());
    }

    #[test]
    fn with_shrink_restores_the_reference_manual_factor() {
        let mut state = GromacsSteep::new(0.01);
        state
            .set_config(GromacsSteepConfig::default().with_shrink(0.2))
            .unwrap();
        assert!(state.update(2.0, 1.0));
        let grown = 0.01 * GROMACS_STEEP_GROW;
        assert_eq!(state.step_size().to_bits(), grown.to_bits());
        assert!(!state.update(1.0, 1.0));
        assert_eq!(state.step_size().to_bits(), (grown * 0.2).to_bits());
        assert!(!state.update(1.0, 2.0));
        assert_eq!(state.step_size().to_bits(), (grown * 0.2 * 0.2).to_bits());
        assert!(
            state
                .set_config(GromacsSteepConfig::default().with_shrink(0.0))
                .is_err()
        );
        assert!(
            state
                .set_config(GromacsSteepConfig::default().with_shrink(1.0))
                .is_err()
        );
    }

    #[test]
    fn constraint_shrinks_stop_at_the_bound_and_a_success_restarts_the_count() {
        let mut state = GromacsSteep::new(1.0);
        state
            .set_config(GromacsSteepConfig::default().with_max_constraint_shrinks(2))
            .unwrap();
        assert!(!state.reject_constraint());
        assert_eq!(state.step_size().to_bits(), 0.5_f64.to_bits());
        assert!(!state.reject_constraint());
        assert_eq!(state.step_size().to_bits(), 0.25_f64.to_bits());
        assert!(state.reject_constraint());
        assert_eq!(state.step_size().to_bits(), 0.25_f64.to_bits());
        state.clear_constraint_shrinks();
        assert!(!state.reject_constraint());
        assert_eq!(state.step_size().to_bits(), 0.125_f64.to_bits());

        let mut immediate = GromacsSteep::new(1.0);
        immediate
            .set_config(GromacsSteepConfig::default().with_max_constraint_shrinks(0))
            .unwrap();
        assert!(immediate.reject_constraint());
        assert_eq!(immediate.step_size().to_bits(), 1.0_f64.to_bits());
    }

    #[test]
    fn displacement_uses_the_largest_atom_force_norm() {
        let state = GromacsSteep::new(0.1);
        let step = state.displacement(ndarray::array![3.0, 4.0, 0.0, 0.0, -10.0, 0.0].view());
        let want = ndarray::array![-0.03, -0.04, 0.0, 0.0, 0.1, 0.0];
        for (&got, &want) in step.iter().zip(want.iter()) {
            assert_abs_diff_eq!(got, want, epsilon = 1e-16);
        }
    }

    #[test]
    fn default_constraint_shrink_budget_tracks_the_initial_step_size() {
        let smaller = GromacsSteepConfig::for_step_size(0.001).max_constraint_shrinks;
        let larger = GromacsSteepConfig::for_step_size(0.1).max_constraint_shrinks;
        assert!(smaller < larger);
        assert_eq!(
            GromacsSteepConfig::default().max_constraint_shrinks,
            GROMACS_STEEP_MAX_CONSTRAINT_SHRINKS
        );
        let manual_smaller = GromacsSteepConfig::for_step_size(0.001)
            .with_shrink(0.2)
            .max_constraint_shrinks;
        let manual_larger = GromacsSteepConfig::for_step_size(0.1)
            .with_shrink(0.2)
            .max_constraint_shrinks;
        assert_eq!(
            manual_smaller,
            gromacs_steep_max_constraint_shrinks(0.001, 0.2)
        );
        assert_eq!(
            manual_larger,
            gromacs_steep_max_constraint_shrinks(0.1, 0.2)
        );
        assert!(manual_smaller < manual_larger);
        assert!(manual_larger < larger);
    }

    #[test]
    fn the_default_budget_matches_thirty_four_halvings_of_emstep() {
        assert_eq!(
            gromacs_steep_max_constraint_shrinks(
                GROMACS_STEEP_EMSTEP_DEFAULT,
                GROMACS_STEEP_SHRINK
            ),
            34
        );
        assert_eq!(GROMACS_STEEP_MAX_CONSTRAINT_SHRINKS, 34);
    }

    #[test]
    fn a_shrink_near_one_returns_quickly_with_the_capped_budget() {
        // `1.0.next_down()` is the GROMACS-manual "nextafter(1, 0)" case:
        // the loop this replaced needed close to u32::MAX multiplications
        // to bring any ordinary step size down to the double precision
        // floor, which the gate measured as a multi-second hang.
        let shrink = 1.0_f64.next_down();
        let budget_timeout = std::time::Duration::from_millis(200);
        let started = std::time::Instant::now();
        let budget = gromacs_steep_max_constraint_shrinks(0.01, shrink);
        assert!(
            started.elapsed() < budget_timeout,
            "took {:?}, expected well under {budget_timeout:?}",
            started.elapsed()
        );
        assert_eq!(budget, GROMACS_STEEP_SHRINK_BUDGET_CAP);
    }

    #[test]
    fn set_config_rederives_a_default_shrink_budget_from_the_session_step_size() {
        // `default()` bakes in `GROMACS_STEEP_EMSTEP_DEFAULT`, so without
        // the rederivation in `set_config` this budget would equal
        // `gromacs_steep_max_constraint_shrinks(GROMACS_STEEP_EMSTEP_DEFAULT, 0.3)`
        // instead of being sized for `state`'s own, noncanonical step size.
        let noncanonical_step_size = 2.0;
        let mut state = GromacsSteep::new(noncanonical_step_size);
        state
            .set_config(GromacsSteepConfig::default().with_shrink(0.3))
            .unwrap();
        assert_eq!(
            state.config().max_constraint_shrinks,
            gromacs_steep_max_constraint_shrinks(noncanonical_step_size, 0.3)
        );
        assert_ne!(
            state.config().max_constraint_shrinks,
            gromacs_steep_max_constraint_shrinks(GROMACS_STEEP_EMSTEP_DEFAULT, 0.3)
        );
    }

    #[test]
    fn set_config_keeps_an_explicit_shrink_budget_despite_rederivation() {
        let mut state = GromacsSteep::new(2.0);
        state
            .set_config(GromacsSteepConfig::default().with_max_constraint_shrinks(5))
            .unwrap();
        assert_eq!(state.config().max_constraint_shrinks, 5);
    }
}
