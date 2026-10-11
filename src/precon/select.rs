//! Which preconditioner a geometry should use.

/// The bonding pattern the caller is minimising.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeometryKind {
    /// Periodic bulk.
    Bulk,
    /// Slab or surface.
    Surface,
    /// Isolated molecule.
    Molecule,
    /// Finite cluster.
    Cluster,
}

/// The metric [`select_precon`] returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreconChoice {
    /// Packwood exponential metric.
    Exp,
    /// Lindh model Hessian.
    Lindh,
    /// Bonded force-field metric.
    Ff,
    /// Leave the solver unpreconditioned.
    Identity,
}

/// Pick a preconditioner for a geometry.
///
/// Exp (Packwood et al., J. Chem. Phys. 144, 164109 (2016),
/// <https://doi.org/10.1063/1.4947024>) for a bulk or a surface of at
/// least 100 atoms. Lindh (Lindh, Bernhardsson, Karlström, and
/// Malmqvist, Chem. Phys. Lett. 241, 423 (1995),
/// <https://doi.org/10.1016/0009-2614(95)00646-L>) for a molecule or a
/// cluster when the caller has no bonded force field. FF (Makri,
/// Ortner, and Kermode, J. Chem. Phys. 150, 094109 (2019),
/// <https://doi.org/10.1063/1.5064465>) for a molecule or a cluster
/// when `has_force_field` is set. Identity when `natoms` is 0, and
/// for a bulk or a surface below 100 atoms.
#[must_use]
pub fn select_precon(kind: GeometryKind, natoms: usize, has_force_field: bool) -> PreconChoice {
    if natoms == 0 {
        return PreconChoice::Identity;
    }
    match kind {
        GeometryKind::Bulk | GeometryKind::Surface => {
            if natoms >= 100 {
                PreconChoice::Exp
            } else {
                PreconChoice::Identity
            }
        }
        GeometryKind::Molecule | GeometryKind::Cluster => {
            if has_force_field {
                PreconChoice::Ff
            } else {
                PreconChoice::Lindh
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_helper_follows_the_bonding_pattern() {
        assert_eq!(
            select_precon(GeometryKind::Bulk, 160, false),
            PreconChoice::Exp
        );
        assert_eq!(
            select_precon(GeometryKind::Surface, 401, true),
            PreconChoice::Exp
        );
        assert_eq!(
            select_precon(GeometryKind::Bulk, 32, false),
            PreconChoice::Identity
        );
        assert_eq!(
            select_precon(GeometryKind::Surface, 99, false),
            PreconChoice::Identity
        );
        assert_eq!(
            select_precon(GeometryKind::Molecule, 3, false),
            PreconChoice::Lindh
        );
        assert_eq!(
            select_precon(GeometryKind::Cluster, 38, false),
            PreconChoice::Lindh
        );
        assert_eq!(
            select_precon(GeometryKind::Molecule, 18, true),
            PreconChoice::Ff
        );
        assert_eq!(
            select_precon(GeometryKind::Cluster, 7, true),
            PreconChoice::Ff
        );
        assert_eq!(
            select_precon(GeometryKind::Molecule, 0, false),
            PreconChoice::Identity
        );
    }
}
