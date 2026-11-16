# Contributing to rgmin

Report a bug or ask a question on the [issue tracker](https://github.com/OmniPotentRPC/rgmin/issues).
Report a vulnerability as the [security policy](SECURITY.md) says, not in a public issue.
Everyone who takes part follows the [code of conduct](CODE_OF_CONDUCT.md), in issues and pull requests alike.

Pull requests go to the `develop` branch.
Before you open one, run the checks that continuous integration (CI) runs with Rust 1.88.0:

```bash
rustup toolchain install 1.88.0 --profile minimal -c rustfmt
python3 -m pip install numpy scipy
cargo +1.88.0 fmt --all -- --check
cargo +1.88.0 test --locked
cargo +1.88.0 test --locked --features capi
cargo +1.88.0 test --locked --features highs
cargo +1.88.0 test --locked --all-features
```

CI also runs clippy with warnings denied, on Rust 1.88.0 and on the latest stable, for the same four feature sets:

```bash
rustup component add clippy --toolchain 1.88.0
cargo +1.88.0 clippy --locked --all-targets -- -D warnings
cargo +1.88.0 clippy --locked --all-targets --features capi -- -D warnings
cargo +1.88.0 clippy --locked --all-targets --features highs -- -D warnings
cargo +1.88.0 clippy --locked --all-targets --all-features -- -D warnings
```

Run the same four commands with `+stable` for the second toolchain.

CI also builds and tests in release with Rust 1.93.0 and 1.94.1, which ship LLVM 21.
Its loop vectorizer once crashed on rgmin, and the debug builds above never run it:

```bash
rustup toolchain install 1.93.0 --profile minimal
cargo +1.93.0 build --locked --release --all-features
cargo +1.93.0 test --locked --release --all-features --lib
```

Install 1.94.1 the same way and run the two `cargo` commands with `+1.94.1`.

A change that users can see adds a fragment to `docs/newsfragments/`, and the fragment is its only changelog entry.
Name the fragment `<issue>.<type>.md`, or `+<name>.<type>.md` for a change with no issue.
The type is `added`, `changed`, `deprecated`, `removed`, `fixed` or `security`, the sections of [CHANGELOG.md](CHANGELOG.md).
`towncrier build` moves the fragments into `CHANGELOG.md` when a release is cut, with the settings in `towncrier.toml`.
Leave the Unreleased sections of `CHANGELOG.md` and `docs/orgmode/changelog.org` empty: an entry written there as well would appear twice after `towncrier build`.
The release commit copies the notes that towncrier wrote into `docs/orgmode/changelog.org`, under a heading for the version.
`towncrier build --draft --version 0.0.0` prints the notes that the fragments give and changes no file.
`python3 scripts/check_changelog.py`, which CI runs, fails when a fragment and a line under Unreleased exist together, or when towncrier would skip a fragment because `towncrier.toml` has no such type.

The [contributing page](docs/orgmode/contributing/index.org) of the docs covers the rest: the C library check, the docs build, citations and the formula numbers the code follows.
