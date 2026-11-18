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
/// multiplications to bring `step_size` down to [`f64::EPSILON`]; past
/// this bound the count is capped rather than returned exactly, so a
/// caller still gets a usable budget in bounded time.
pub const GROMACS_STEEP_SHRINK_BUDGET_CAP: u32 = 1_000_000;

/// Largest shrink count from this step size and shrink factor before the
/// displacement reaches floating-point epsilon.
///
/// The count is the least `k` with `step_size * shrink.powi(k) <=
/// f64::EPSILON`, taken in closed form from logarithms rather than by
/// repeated multiplication, so a `shrink` near one returns in bounded
/// time instead of multiplying until a loop counter overflows. The
/// result saturates at [`GROMACS_STEEP_SHRINK_BUDGET_CAP`].
pub fn gromacs_steep_max_constraint_shrinks(step_size: f64, shrink: f64) -> u32 {
    if !(step_size.is_finite()
        && step_size > 0.0
        && shrink.is_finite()
        && shrink > 0.0
        && shrink < 1.0)
    {
        return 0;
    }
    if step_size <= f64::EPSILON {
        return 0;
    }
    let exact_count = (f64::EPSILON / step_size).ln() / shrink.ln();
    let count = exact_count.ceil();
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
pub const GROMACS_STEEP_MAX_CONSTRAINT_SHRINKS: u32 = 46;

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
    /// and keep the old point. The next failure is [`Error::Constraint`].
    ///
    /// The default is [`GROMACS_STEEP_MAX_CONSTRAINT_SHRINKS`]. Zero
    /// returns the constraint error on the first failure and does not
    /// shrink.
    pub max_constraint_shrinks: u32,
    step_size: f64,
}

with_setters!(GromacsSteepConfig {
    with_max_constraint_shrinks => max_constraint_shrinks: val u32,
});

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
        }
    }

    /// Replace [`Self::shrink`] and recompute [`Self::max_constraint_shrinks`]
    /// from the step size.
    ///
    /// Chain it after `default()` or [`Self::for_step_size`].
    #[must_use]
    pub fn with_shrink(mut self, shrink: f64) -> Self {
        self.shrink = shrink;
        self.max_constraint_shrinks = gromacs_steep_max_constraint_shrinks(self.step_size, shrink);
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
    pub fn set_config(&mut self, config: GromacsSteepConfig) -> Result<()> {
        self.config = config.validate()?;
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
        let norm = crate::StopNorm::MaxAtom.of(gradient);
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
    fn a_shrink_near_one_returns_quickly_with_the_capped_budget() {
        // `1.0.next_down()` is the GROMACS-manual "nextafter(1, 0)" case:
        // the loop this replaced needed close to u32::MAX multiplications
        // to bring any ordinary step size down to f64::EPSILON, which the
        // gate measured as a multi-second hang.
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
}
