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

/// Which optimiser is asking for a metric.
///
/// FIRE applies `P` on every step. ASE `PreconFIRE` leaves the
/// preconditioner unset, and the bench loses with the metric on al,
/// si-wire, cu-vac, h2o, and ch4. [`OptimizerFamily::Fire`] therefore
/// returns [`PreconChoice::Identity`]. L-BFGS and CG keep the geometry
/// rule. Exp and Lindh stay available when the caller names them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimizerFamily {
    /// L-BFGS and the other quasi-Newton updates. `P^{-1}` is `H0`.
    QuasiNewton,
    /// Nonlinear CG. `P` is the metric of the search direction.
    Cg,
    /// FIRE. Auto stays at the identity.
    Fire,
}

/// Largest covalent molecule or cluster that stays on Lindh.
///
/// Above this count the choice is Exp when the system has at least
/// 100 atoms, and the identity below that. Jacobi PCG iterations on
/// bulk diamond Si (4 Å cutoff), one right-hand side, and the
/// one-thread time for eight solves (`tests/precon_scaling.rs`):
/// 160 atoms, Lindh 69 iterations and 0.031 s, Exp 21 and 0.001 s.
/// 2,000 atoms, Lindh 124 and 0.914 s, Exp 48 and 0.031 s.
/// 8,000 atoms, Lindh 136 and 3.69 s, Exp 58 and 0.161 s.
/// 32,000 atoms, Lindh 140 and 34.9 s, Exp 85 and 1.03 s.
/// The iteration count levels off near 140. Two thousand atoms is the
/// largest measured size at which eight Lindh solves still finish in
/// about a second. At 32,000 atoms Lindh takes 34.9 s against 1.03 s
/// for Exp.
pub const N_LINDH: usize = 2_000;

/// Pick a preconditioner from the geometry, the atomic numbers, and
/// the optimiser.
///
/// The 100-atom line is ASE's `PreconLBFGS` rule. The bench has no
/// bulk below 100 atoms and no Cu147. The copper vacancy (2047 atoms)
/// is the bulk row that uses Exp: L-BFGS takes 16 calls with none and
/// 11 with Exp.
///
/// Exp (Packwood et al., J. Chem. Phys. 144, 164109 (2016),
/// <https://doi.org/10.1063/1.4947024>) for a bulk or a surface of at
/// least 100 atoms, and for a metal or noble-gas cluster of at least
/// 100 atoms.
///
/// Lindh (Lindh, Bernhardsson, Karlström, and Malmqvist, Chem. Phys.
/// Lett. 241, 423 (1995),
/// <https://doi.org/10.1016/0009-2614(95)00646-L>) only when every
/// element is one of H, B, C, N, O, F, Si, P, S, Cl, and the atom
/// count is at most [`N_LINDH`]. Water is that case: L-BFGS takes 5
/// calls with Lindh and 5 without. LJ38 is argon. Lindh there reaches
/// −164.88 against −167.77, so Ar38 is Identity. Pt7 is a metal below
/// 100 atoms, so Identity.
///
/// The bonded metric (Makri, Ortner, and Kermode, J. Chem. Phys. 150,
/// 094109 (2019), <https://doi.org/10.1063/1.5064465>) only when
/// `has_force_field` is set and every element is in that covalent set.
/// Pt7 with the geometric metric takes 18, 29, and 78 calls against
/// 10, 18, and 49 without it. LJ38 with it ends at −166.65 against
/// −167.77, a different minimum. Identity when `z` is empty, and for
/// a bulk or a surface below 100 atoms.
#[must_use]
pub fn select_precon(
    kind: GeometryKind,
    z: &[i32],
    has_force_field: bool,
    family: OptimizerFamily,
) -> PreconChoice {
    if matches!(family, OptimizerFamily::Fire) {
        return PreconChoice::Identity;
    }
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
            let covalent = z.iter().all(|&a| lindh_element(a));
            if has_force_field && covalent {
                PreconChoice::Ff
            } else if covalent && natoms <= N_LINDH {
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

    fn qn(kind: GeometryKind, z: &[i32], field: bool) -> PreconChoice {
        select_precon(kind, z, field, OptimizerFamily::QuasiNewton)
    }

    #[test]
    fn the_helper_follows_the_bonding_pattern() {
        let si = vec![14i32; 160];
        assert_eq!(qn(GeometryKind::Bulk, &si, false), PreconChoice::Exp);
        let surface = vec![14i32; 401];
        assert_eq!(qn(GeometryKind::Surface, &surface, true), PreconChoice::Exp);
        // The 100-atom line is ASE's PreconLBFGS rule. No bulk below
        // 100 atoms is in the bench.
        let small_bulk = vec![14i32; 32];
        assert_eq!(
            qn(GeometryKind::Bulk, &small_bulk, false),
            PreconChoice::Identity
        );
        let small_surface = vec![14i32; 99];
        assert_eq!(
            qn(GeometryKind::Surface, &small_surface, false),
            PreconChoice::Identity
        );
        // Water, Lindh: L-BFGS 5 calls, the same as the unpreconditioned run.
        let h2o = [8, 1, 1];
        assert_eq!(qn(GeometryKind::Molecule, &h2o, false), PreconChoice::Lindh);
        assert_eq!(
            select_precon(GeometryKind::Molecule, &h2o, false, OptimizerFamily::Cg),
            PreconChoice::Lindh
        );
        // Water with a force field, covalent, so the bonded metric.
        assert_eq!(qn(GeometryKind::Molecule, &h2o, true), PreconChoice::Ff);
        // FIRE stays unpreconditioned even when the geometry would take Lindh.
        assert_eq!(
            select_precon(GeometryKind::Molecule, &h2o, true, OptimizerFamily::Fire),
            PreconChoice::Identity
        );
        assert_eq!(
            select_precon(GeometryKind::Bulk, &si, false, OptimizerFamily::Fire),
            PreconChoice::Identity
        );
        // Ar38. Lindh on lj38 stopped at a higher energy, so the cluster
        // stays Identity. A field does not change that: argon is outside
        // the covalent set, and the geometric metric ended at −166.65.
        let ar38 = vec![18i32; 38];
        assert_eq!(
            qn(GeometryKind::Cluster, &ar38, false),
            PreconChoice::Identity
        );
        assert_eq!(
            qn(GeometryKind::Cluster, &ar38, true),
            PreconChoice::Identity
        );
        // Pt7 is a metal cluster below 100 atoms. The field costs calls
        // (L-BFGS 18 against 10) and is not taken.
        let pt7 = vec![78i32; 7];
        assert_eq!(
            qn(GeometryKind::Cluster, &pt7, false),
            PreconChoice::Identity
        );
        assert_eq!(
            qn(GeometryKind::Cluster, &pt7, true),
            PreconChoice::Identity
        );
        // Cu147 is not in the bench. The 100-atom line is ASE's
        // PreconLBFGS rule, so 147 metal atoms still select Exp.
        let cu147 = vec![29i32; 147];
        assert_eq!(qn(GeometryKind::Cluster, &cu147, false), PreconChoice::Exp);
        assert_eq!(qn(GeometryKind::Cluster, &cu147, true), PreconChoice::Exp);
        assert_eq!(
            qn(GeometryKind::Molecule, &[], false),
            PreconChoice::Identity
        );
        let past = vec![6i32; N_LINDH + 1];
        let past_choice = if past.len() >= 100 {
            PreconChoice::Exp
        } else {
            PreconChoice::Identity
        };
        assert_eq!(qn(GeometryKind::Cluster, &past, false), past_choice);
        let at = vec![6i32; N_LINDH.max(1)];
        assert_eq!(qn(GeometryKind::Molecule, &at, false), PreconChoice::Lindh);
    }
}
