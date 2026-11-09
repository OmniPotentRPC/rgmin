//! The two `select_precon` checks from main's session test.

use rgmin::{GeometryKind, OptimizerFamily, PreconChoice, select_precon};

#[test]
fn molecule_choices_match_main() {
    assert_eq!(
        select_precon(
            GeometryKind::Molecule,
            &[8, 1, 1],
            false,
            OptimizerFamily::QuasiNewton
        ),
        PreconChoice::Lindh
    );
    assert_eq!(
        select_precon(
            GeometryKind::Molecule,
            &[8, 1, 1],
            true,
            OptimizerFamily::Cg
        ),
        PreconChoice::Ff
    );
}
