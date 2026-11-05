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

A change that users can see adds a fragment to `docs/newsfragments/` and a line under Unreleased in `docs/orgmode/changelog.org`.
Name the fragment `<issue>.<type>.md`, or `+<name>.<type>.md` for a change with no issue.
The type is `added`, `changed`, `deprecated`, `removed`, `fixed` or `security`, the sections of [CHANGELOG.md](CHANGELOG.md).
`towncrier build` moves the fragments into `CHANGELOG.md` when a release is cut, with the settings in `towncrier.toml`.
`towncrier build --draft --version 0.0.0` prints the notes that the fragments give and changes no file.

The [contributing page](docs/orgmode/contributing/index.org) of the docs covers the rest: the C library check, the docs build, citations and the formula numbers the code follows.
