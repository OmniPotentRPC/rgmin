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
    /// Bonded metric. A parameterised field, or the geometric metric
    /// when the caller supplies the constants.
    Ff,
    /// Leave the solver unpreconditioned.
    Identity,
}

/// Lindh's tables cover covalent non-metals in rows 1 to 3.
///
/// Metals and noble gases in those rows are excluded, and every
/// heavier element is excluded. The row index in the Lindh assembly
/// would otherwise reuse the third-row numbers.
fn lindh_element(z: i32) -> bool {
    matches!(z, 1 | 5 | 6 | 7 | 8 | 9 | 14 | 15 | 16 | 17)
}

/// Pick a preconditioner from the geometry and the atomic numbers.
///
/// Exp (Packwood et al., J. Chem. Phys. 144, 164109 (2016),
/// <https://doi.org/10.1063/1.4947024>) for a bulk or a surface of at
/// least 100 atoms, and for a metal or noble-gas cluster of at least
/// 100 atoms. That cluster rule is the ASE `PreconLBFGS` cutoff. The
/// copper vacancy bench (L-BFGS 16 / 11 / 16 calls for none / exp /
/// lindh) is the bulk row behind the 100-atom Exp threshold, and
/// Cu147 is the cluster of that size.
///
/// Lindh (Lindh, Bernhardsson, Karlström, and Malmqvist, Chem. Phys.
/// Lett. 241, 423 (1995),
/// <https://doi.org/10.1016/0009-2614(95)00646-L>) only when every
/// element is one of H, B, C, N, O, F, Si, P, S, Cl. Water is that
/// case: L-BFGS takes 5 calls with Lindh and 5 without. LJ38 is argon.
/// Lindh there reaches −164.88 against −167.77 and CG and FIRE do not
/// converge, so Ar38 is Identity. Pt7 is a metal below 100 atoms, so
/// Identity.
///
/// FF (Makri, Ortner, and Kermode, J. Chem. Phys. 150, 094109 (2019),
/// <https://doi.org/10.1063/1.5064465>) for a molecule or a cluster
/// when `has_force_field` is set, including water. Identity when `z`
/// is empty, and for a bulk or a surface below 100 atoms.
#[must_use]
pub fn select_precon(kind: GeometryKind, z: &[i32], has_force_field: bool) -> PreconChoice {
    let natoms = z.len();
    if natoms == 0 || z.iter().any(|&a| a <= 0) {
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
            } else if z.iter().all(|&a| lindh_element(a)) {
                PreconChoice::Lindh
            } else if natoms >= 100 {
                PreconChoice::Exp
            } else {
                PreconChoice::Identity
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_helper_follows_the_bonding_pattern() {
        let si = vec![14i32; 160];
        assert_eq!(
            select_precon(GeometryKind::Bulk, &si, false),
            PreconChoice::Exp
        );
        let surface = vec![14i32; 401];
        assert_eq!(
            select_precon(GeometryKind::Surface, &surface, true),
            PreconChoice::Exp
        );
        let small_bulk = vec![14i32; 32];
        assert_eq!(
            select_precon(GeometryKind::Bulk, &small_bulk, false),
            PreconChoice::Identity
        );
        let small_surface = vec![14i32; 99];
        assert_eq!(
            select_precon(GeometryKind::Surface, &small_surface, false),
            PreconChoice::Identity
        );
        // Water, Lindh: L-BFGS 5 calls, the same as the unpreconditioned run.
        let h2o = [8, 1, 1];
        assert_eq!(
            select_precon(GeometryKind::Molecule, &h2o, false),
            PreconChoice::Lindh
        );
        // Water with a force field.
        assert_eq!(
            select_precon(GeometryKind::Molecule, &h2o, true),
            PreconChoice::Ff
        );
        // Ar38. Lindh on lj38 did not converge for CG or FIRE and
        // L-BFGS stopped at a higher energy, so the cluster stays Identity.
        let ar38 = vec![18i32; 38];
        assert_eq!(
            select_precon(GeometryKind::Cluster, &ar38, false),
            PreconChoice::Identity
        );
        // Pt7 is a metal cluster below 100 atoms.
        let pt7 = vec![78i32; 7];
        assert_eq!(
            select_precon(GeometryKind::Cluster, &pt7, false),
            PreconChoice::Identity
        );
        // Cu147 is the ASE cutoff: a metal cluster of at least 100 atoms.
        let cu147 = vec![29i32; 147];
        assert_eq!(
            select_precon(GeometryKind::Cluster, &cu147, false),
            PreconChoice::Exp
        );
        assert_eq!(
            select_precon(GeometryKind::Molecule, &[], false),
            PreconChoice::Identity
        );
    }
}
