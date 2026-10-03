//! Closed HiGHS engine tokens. These compile without the `highs` feature
//! so the C waist can refuse an unknown integer on every build.

/// HiGHS `solver` option. Integers match `rgmin_highs_solver_t`.
/// Constrained dest defaults to [`HighsSolverKind::Ipm`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HighsSolverKind {
    /// HiGHS `choose`.
    Choose = 0,
    /// Dual/primal simplex.
    Simplex = 1,
    /// Interior point: HiGHS picks IPX or HiPO.
    Ipm = 2,
    /// IPX (PCG, serial).
    Ipx = 3,
    /// HiPO (direct factor, parallel).
    Hipo = 4,
    /// PDLP first-order LP.
    Pdlp = 5,
    /// HiPDLP.
    Hipdlp = 6,
    /// Active-set QP (`qpasm`).
    Qpasm = 7,
}

impl HighsSolverKind {
    /// Closed map from the C token.
    pub fn from_ordinal(raw: i32) -> Option<Self> {
        Some(match raw {
            0 => Self::Choose,
            1 => Self::Simplex,
            2 => Self::Ipm,
            3 => Self::Ipx,
            4 => Self::Hipo,
            5 => Self::Pdlp,
            6 => Self::Hipdlp,
            7 => Self::Qpasm,
            _ => return None,
        })
    }

    /// HiGHS `solver` string, or `None` to leave the default.
    pub fn as_highs(self) -> Option<&'static str> {
        Some(match self {
            Self::Choose => return None,
            Self::Simplex => "simplex",
            Self::Ipm => "ipm",
            Self::Ipx => "ipx",
            Self::Hipo => "hipo",
            Self::Pdlp => "pdlp",
            Self::Hipdlp => "hipdlp",
            Self::Qpasm => "qpasm",
        })
    }
}

/// HiGHS `run_crossover`. Constrained dest defaults to
/// [`HighsCrossover::Off`]: the step is the interior point, not a basis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HighsCrossover {
    /// HiGHS `choose`.
    Choose = 0,
    /// Force crossover to a basic solution.
    On = 1,
    /// Skip crossover.
    Off = 2,
}

impl HighsCrossover {
    /// Closed map from the C token.
    pub fn from_ordinal(raw: i32) -> Option<Self> {
        Some(match raw {
            0 => Self::Choose,
            1 => Self::On,
            2 => Self::Off,
            _ => return None,
        })
    }

    /// HiGHS `run_crossover` string, or `None` to leave the default.
    pub fn as_highs(self) -> Option<&'static str> {
        Some(match self {
            Self::Choose => return None,
            Self::On => "on",
            Self::Off => "off",
        })
    }
}

/// HiGHS callback kind. Integers match `HighsCallbackType` /
/// `rgmin_highs_cb_kind_t`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HighsCallbackKind {
    /// General log line.
    Logging = 0,
    /// Simplex interrupt poll.
    SimplexInterrupt = 1,
    /// IPM interrupt poll.
    IpmInterrupt = 2,
}

impl HighsCallbackKind {
    /// Closed map from the C token.
    pub fn from_ordinal(raw: i32) -> Option<Self> {
        Some(match raw {
            0 => Self::Logging,
            1 => Self::SimplexInterrupt,
            2 => Self::IpmInterrupt,
            _ => return None,
        })
    }
}

/// Dest HiGHS user callback. `interrupt` nonzero stops the solve.
pub type HighsCCallback = unsafe extern "C" fn(
    kind: i32,
    message: *const std::os::raw::c_char,
    interrupt: *mut i32,
    user: *mut std::os::raw::c_void,
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solver_tokens_are_closed() {
        assert_eq!(HighsSolverKind::from_ordinal(2), Some(HighsSolverKind::Ipm));
        assert_eq!(HighsSolverKind::from_ordinal(3), Some(HighsSolverKind::Ipx));
        assert_eq!(HighsSolverKind::from_ordinal(4), Some(HighsSolverKind::Hipo));
        assert!(HighsSolverKind::from_ordinal(8).is_none());
        assert_eq!(HighsSolverKind::Ipm.as_highs(), Some("ipm"));
        assert_eq!(HighsSolverKind::Choose.as_highs(), None);
    }

    #[test]
    fn crossover_tokens_are_closed() {
        assert_eq!(HighsCrossover::from_ordinal(2), Some(HighsCrossover::Off));
        assert!(HighsCrossover::from_ordinal(3).is_none());
        assert_eq!(HighsCrossover::Off.as_highs(), Some("off"));
    }

    #[test]
    fn callback_tokens_match_highs() {
        assert_eq!(
            HighsCallbackKind::from_ordinal(2),
            Some(HighsCallbackKind::IpmInterrupt)
        );
        assert!(HighsCallbackKind::from_ordinal(3).is_none());
    }
}
