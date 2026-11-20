/// The weighted IRC axis selection and the C tensor validation each
/// need a `fixed` newsfragment. Towncrier publishes these notes and
/// leaves the Unreleased sections empty.
#[test]
fn weighted_irc_axis_and_c_tensor_checks_have_fragments() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("docs/newsfragments");
    let notes = [
        (
            "+irc-weighted-axis.fixed.md",
            "mass weight equal to its floor",
        ),
        (
            "+c-abi-tensor-validation.fixed.md",
            "RGMIN_INVALID_PARAMETER",
        ),
    ];
    for (name, needle) in notes {
        let path = root.join(name);
        let text = std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("{name}: {err}"));
        assert!(
            text.contains(needle),
            "{name} does not explain the change ({needle})"
        );
    }
}
