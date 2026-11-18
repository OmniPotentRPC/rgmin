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
}

with_setters!(GromacsSteepConfig {
    with_shrink => shrink: val f64,
});

impl Default for GromacsSteepConfig {
    fn default() -> Self {
        Self {
            shrink: GROMACS_STEEP_SHRINK,
        }
    }
}

impl GromacsSteepConfig {
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
}

impl GromacsSteep {
    /// Start with maximum per-atom displacement `step_size` and
    /// [`GromacsSteepConfig::default`].
    pub fn new(step_size: f64) -> Self {
        Self {
            step_size,
            config: GromacsSteepConfig::default(),
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

    /// Replace the shrink factor. The current displacement is kept.
    pub fn set_config(&mut self, config: GromacsSteepConfig) -> Result<()> {
        self.config = config.validate()?;
        Ok(())
    }

    /// Reset the maximum displacement. The shrink factor is kept.
    pub fn reset(&mut self, step_size: f64) {
        self.step_size = step_size;
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
    fn displacement_uses_the_largest_atom_force_norm() {
        let state = GromacsSteep::new(0.1);
        let step = state.displacement(ndarray::array![3.0, 4.0, 0.0, 0.0, -10.0, 0.0].view());
        let want = ndarray::array![-0.03, -0.04, 0.0, 0.0, 0.1, 0.0];
        for (&got, &want) in step.iter().zip(want.iter()) {
            assert_abs_diff_eq!(got, want, epsilon = 1e-16);
        }
    }
}
