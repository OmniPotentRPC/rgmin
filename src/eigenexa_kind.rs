//! Closed EigenExa algorithm tokens. These compile without a linked
//! EigenExa so a host can name `eigen_s` versus `eigen_sx` as integers.

/// EigenExa solver kernel. Integers are the public ABI; there is no
/// string key.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EigenExaAlgo {
    /// `eigen_s`.
    S = 0,
    /// `eigen_sx`.
    Sx = 1,
}

impl EigenExaAlgo {
    /// Schema / C ABI name. Never a free-form string key.
    pub const fn name(self) -> &'static str {
        match self {
            Self::S => "eigen_s",
            Self::Sx => "eigen_sx",
        }
    }

    /// Decode a closed ordinal. Unknown integers are `None`.
    pub const fn from_ordinal(raw: u8) -> Option<Self> {
        match raw {
            0 => Some(Self::S),
            1 => Some(Self::Sx),
            _ => None,
        }
    }
}

/// Typed EigenExa extras. Ignored unless the backend is EigenExa.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EigenExaParams {
    /// `eigen_s` versus `eigen_sx`.
    pub algo: EigenExaAlgo,
}

impl Default for EigenExaParams {
    fn default() -> Self {
        Self {
            algo: EigenExaAlgo::Sx,
        }
    }
}
