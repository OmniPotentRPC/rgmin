//! Closed SLEPc ST tokens. These compile without the `slepc` feature
//! so a host can name the transform on every build.

use std::ffi::c_void;

/// Spectral transformation for the SLEPc arm. Integers are the public ABI.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SlepcStKind {
    /// Leave SLEPc's default (`STSHIFT` at 0).
    Default = 0,
    /// `STSHIFT`.
    Shift = 1,
    /// `STSINVERT`. Needs a host Pmat; a MatShell cannot be factored.
    Sinvert = 2,
    /// `STPRECOND`. Uses the host Pmat when one is supplied.
    Precond = 3,
    /// `STCAYLEY`. Needs a host Pmat.
    Cayley = 4,
}

impl SlepcStKind {
    /// Schema / C ABI name. Never a free-form string key.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Shift => "shift",
            Self::Sinvert => "sinvert",
            Self::Precond => "precond",
            Self::Cayley => "cayley",
        }
    }

    /// Decode a closed ordinal. Unknown integers are `None`.
    pub const fn from_ordinal(raw: u8) -> Option<Self> {
        match raw {
            0 => Some(Self::Default),
            1 => Some(Self::Shift),
            2 => Some(Self::Sinvert),
            3 => Some(Self::Precond),
            4 => Some(Self::Cayley),
            _ => None,
        }
    }

    /// Shift-and-invert and Cayley factor an assembled matrix.
    pub const fn needs_pmat(self) -> bool {
        matches!(self, Self::Sinvert | Self::Cayley)
    }
}

/// Host-owned PETSc `Mat` used as the ST preconditioner matrix.
///
/// The host lives in PETSc and keeps the object alive for the solve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlepcPmat {
    raw: *mut c_void,
}

impl SlepcPmat {
    /// Borrow a host `Mat`. The host keeps ownership.
    ///
    /// # Safety
    /// `mat` is a live PETSc `Mat` on the calling communicator for
    /// the duration of [`crate::lowest_mode_slepc`].
    pub const unsafe fn from_raw(mat: *mut c_void) -> Self {
        Self { raw: mat }
    }

    /// The borrowed `Mat` pointer.
    pub const fn as_raw(self) -> *mut c_void {
        self.raw
    }
}

/// Typed SLEPc extras. Ignored unless the backend is SLEPc.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlepcParams {
    /// `STSetType` token.
    pub st: SlepcStKind,
    /// Host Pmat for `STSetPreconditionerMat`. `None` is Krylov-Schur
    /// on the Hessian MatShell alone.
    pub pmat: Option<SlepcPmat>,
}

impl Default for SlepcParams {
    fn default() -> Self {
        Self {
            st: SlepcStKind::Default,
            pmat: None,
        }
    }
}
