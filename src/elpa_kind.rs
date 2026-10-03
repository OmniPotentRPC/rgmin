//! Closed ELPA stage and block-size tokens. These compile without
//! a linked ELPA. Kind 7 is 1-stage, kind 8 is 2-stage. GPU is a
//! runtime probe of the linked build, not a third kind.

use crate::lowest_mode::EigensolverKind;

/// ELPA solver stage. Integers match `ELPA_SOLVER_1STAGE` / `ELPA_SOLVER_2STAGE`.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ElpaStage {
    /// `ELPA_SOLVER_1STAGE`.
    OneStage = 1,
    /// `ELPA_SOLVER_2STAGE`.
    TwoStage = 2,
}

impl ElpaStage {
    /// Schema / C ABI name. Never an `elpa_set` key.
    pub const fn name(self) -> &'static str {
        match self {
            Self::OneStage => "oneStage",
            Self::TwoStage => "twoStage",
        }
    }

    /// Decode a closed ELPA solver integer. Unknown values are `None`.
    pub const fn from_solver(raw: i32) -> Option<Self> {
        match raw {
            1 => Some(Self::OneStage),
            2 => Some(Self::TwoStage),
            _ => None,
        }
    }

    /// Stage for [`EigensolverKind::Elpa`] / [`EigensolverKind::Elpa2`].
    pub const fn from_kind(kind: EigensolverKind) -> Option<Self> {
        match kind {
            EigensolverKind::Elpa => Some(Self::OneStage),
            EigensolverKind::Elpa2 => Some(Self::TwoStage),
            _ => None,
        }
    }

    /// Integer written to the linked ELPA handle. Not a string.
    pub const fn elpa_solver(self) -> i32 {
        self as i32
    }
}

/// Typed ELPA extras. `nblk == 0` selects 16.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ElpaParams {
    /// ScaLAPACK block size. 0 selects 16.
    pub nblk: u32,
}

impl Default for ElpaParams {
    fn default() -> Self {
        Self { nblk: 0 }
    }
}

impl ElpaParams {
    /// Block size after the 0 -> 16 default.
    pub const fn nblk_or_default(self) -> u32 {
        if self.nblk == 0 {
            16
        } else {
            self.nblk
        }
    }
}

/// Closed ELPA integers for a dense call: `(solver, nblk)`.
pub fn elpa_config(kind: EigensolverKind, elpa: &ElpaParams) -> Option<(i32, u32)> {
    let stage = ElpaStage::from_kind(kind)?;
    Some((stage.elpa_solver(), elpa.nblk_or_default()))
}

/// Partial-spectrum window for ELPA `elpa_eigenvectors` / PDSYEVR range I.
///
/// `il` is 1. `iu` is `nev`, not `n`, unless the caller asked for
/// the full spectrum (`nev == n`). The gathered `Z` is `n x iu`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ElpaSpectrum {
    /// 1-based first index. Always 1.
    pub il: u32,
    /// 1-based last index. Equals `nev` (clamped to `n`).
    pub iu: u32,
    /// Columns of `Z`. Equals `iu`, never `n` on a partial request.
    pub z_cols: u32,
    /// True only when `nev == n`.
    pub full: bool,
}

/// Window for a dense ELPA / ScaLAPACK call. `nev == 0` selects 1.
pub fn elpa_spectrum(n: usize, nev: usize) -> crate::error::Result<ElpaSpectrum> {
    if n == 0 {
        return Err(crate::error::Error::Dim { got: 0, dim: 0 });
    }
    let k = if nev == 0 { 1 } else { nev.min(n) };
    Ok(ElpaSpectrum {
        il: 1,
        iu: k as u32,
        z_cols: k as u32,
        full: k == n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elpa_kind_maps_to_solver_integers() {
        assert_eq!(ElpaStage::from_kind(EigensolverKind::Elpa), Some(ElpaStage::OneStage));
        assert_eq!(ElpaStage::from_kind(EigensolverKind::Elpa2), Some(ElpaStage::TwoStage));
        assert_eq!(ElpaStage::OneStage.elpa_solver(), 1);
        assert_eq!(ElpaStage::TwoStage.elpa_solver(), 2);
        assert_eq!(ElpaStage::from_solver(1), Some(ElpaStage::OneStage));
        assert_eq!(ElpaStage::from_solver(2), Some(ElpaStage::TwoStage));
        assert!(ElpaStage::from_solver(0).is_none());
        assert!(ElpaStage::from_kind(EigensolverKind::Slate).is_none());
        assert!(ElpaStage::from_kind(EigensolverKind::Lanczos).is_none());
        assert_eq!(elpa_config(EigensolverKind::Elpa, &ElpaParams::default()), Some((1, 16)));
        assert_eq!(
            elpa_config(EigensolverKind::Elpa2, &ElpaParams { nblk: 32 }),
            Some((2, 32))
        );
        assert_eq!(ElpaParams::default().nblk_or_default(), 16);
        assert_eq!(EigensolverKind::from_ordinal(7), Some(EigensolverKind::Elpa));
        assert_eq!(EigensolverKind::from_ordinal(8), Some(EigensolverKind::Elpa2));
        assert!(EigensolverKind::from_ordinal(16).is_none());
        assert!(!EigensolverKind::Elpa.is_linked());
        assert!(!EigensolverKind::Elpa2.is_linked());
    }

    #[test]
    fn elpa_spectrum_nev_one_is_not_heev_all() {
        let w = elpa_spectrum(512, 1).expect("n>0");
        assert_eq!(w.il, 1);
        assert_eq!(w.iu, 1);
        assert_eq!(w.z_cols, 1);
        assert!(!w.full);
        let all = elpa_spectrum(512, 512).expect("n>0");
        assert_eq!(all.iu, 512);
        assert_eq!(all.z_cols, 512);
        assert!(all.full);
        let def = elpa_spectrum(512, 0).expect("n>0");
        assert_eq!(def.iu, 1);
        assert!(!def.full);
    }

    #[test]
    fn elpa_public_sources_have_no_string_keys() {
        let schema = include_str!("../schema/eigen.capnp");
        let schema_code = schema
            .lines()
            .filter(|l| !l.trim_start().starts_with('#'))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(!schema_code.contains(": Text"), "schema must not declare a Text field");
        assert!(!schema_code.contains("elpa_set"));
        let impl_src = include_str!("elpa_kind.rs");
        let impl_only = impl_src.split("#[cfg(test)]").next().expect("impl");
        for src in [
            impl_only,
            include_str!("lowest_mode.rs"),
            include_str!("error.rs"),
            include_str!("ffi.rs"),
        ] {
            for line in src.lines() {
                let t = line.trim();
                if t.starts_with("//") || t.starts_with("///") {
                    continue;
                }
                assert!(!t.contains("elpa_set"), "elpa_set in {t}");
                assert!(!t.contains("ELPA_DEFAULT"), "ELPA_DEFAULT in {t}");
                assert!(!t.contains("Elpa2Cuda"), "Elpa2Cuda in {t}");
                assert!(!t.contains("Elpa2Hip"), "Elpa2Hip in {t}");
            }
        }
    }
}
