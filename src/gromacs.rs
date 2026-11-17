//! GROMACS steepest-descent step-size state.
//!
//! The state is deliberately independent of coordinates, units, and the
//! objective. A session supplies the projected force and applies its usual
//! retraction, constraint, and displacement-cap hooks.
//!
//! The update is the [GROMACS reference-manual steepest descent][manual].
//! GROMACS [`do_steep`][source] uses the same normalized-force trial,
//! strict energy test, and 1.2 growth. Its current source uses 0.5 on a
//! rejection, while the manual specifies 0.2; this state follows the
//! documented 0.2 rule.
//!
//! [manual]: https://manual.gromacs.org/current/reference-manual/algorithms/energy-minimization.html#steepest-descent
//! [source]: https://github.com/gromacs/gromacs/blob/main/src/gromacs/mdrun/minimize.cpp#L2881-L3178

use ndarray::{Array1, ArrayView1};

/// Growth factor after a lower-energy trial.
pub const GROMACS_STEEP_GROW: f64 = 1.2;
/// Shrink factor after a refused trial.
pub const GROMACS_STEEP_SHRINK: f64 = 0.2;

/// Adaptive maximum-displacement state for GROMACS `integrator = steep`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GromacsSteep {
    step_size: f64,
}

impl GromacsSteep {
    /// Start with maximum per-atom displacement `step_size`.
    pub fn new(step_size: f64) -> Self {
        Self { step_size }
    }

    /// Current maximum per-atom displacement.
    pub fn step_size(self) -> f64 {
        self.step_size
    }

    /// Reset the maximum displacement.
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

    /// Accept a strict energy decrease and grow by 1.2; otherwise shrink by 0.2.
    pub fn update(&mut self, energy: f64, trial_energy: f64) -> bool {
        let accepted = trial_energy < energy;
        self.step_size *= if accepted {
            GROMACS_STEEP_GROW
        } else {
            GROMACS_STEEP_SHRINK
        };
        accepted
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    use super::*;

    #[test]
    fn strict_decrease_grows_and_every_other_trial_shrinks() {
        let mut state = GromacsSteep::new(0.01);
        assert!(state.update(2.0, 1.0));
        assert_abs_diff_eq!(state.step_size(), 0.012, epsilon = f64::EPSILON);
        assert!(!state.update(1.0, 1.0));
        assert_abs_diff_eq!(state.step_size(), 0.0024, epsilon = f64::EPSILON);
        assert!(!state.update(1.0, 2.0));
        assert_abs_diff_eq!(state.step_size(), 0.00048, epsilon = f64::EPSILON);
    }

    #[test]
    fn displacement_uses_the_largest_atom_force_norm() {
        let state = GromacsSteep::new(0.1);
        let step = state.displacement(array![3.0, 4.0, 0.0, 0.0, -10.0, 0.0].view());
        for (&got, &want) in step
            .iter()
            .zip(array![-0.03, -0.04, 0.0, 0.0, 0.1, 0.0].iter())
        {
            assert_abs_diff_eq!(got, want, epsilon = 1e-16);
        }
    }
}
