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
cargo +1.88.0 test --locked --features capi,highs
cargo +1.88.0 test --locked --features capi,par,highs,slepc,primme,chase,libkrylov
cargo +1.88.0 test --locked --features readcon
cargo +1.88.0 test --locked --features periodic-product
```

CI calls the fifth row all-features.
It names the features instead of passing `--all-features`, so the static library does not link `readcon-core` or the LGPL `gmp-mpfr-sys`, which get the last two rows.
When `capi` is also enabled the periodic-product module is omitted, so that build does not link GMP into the C static library. The periodic-product row is the one that builds the module, and it needs `m4`, `make`, `diffutils` and a C compiler.

The rows with `capi` also compile C and C++ programs against `include/` and run them.
Those tests run `cargo build --lib` for the target, profile and features of the test run and link the library it reports, so no `cargo build` has to come first.
They compile with `CC_<target>` and `CXX_<target>`, or `CC` and `CXX`, read the way the `cc` crate reads them.
Without those, they use the first of `cc`, `gcc` and `clang`, and of `c++`, `g++` and `clang++`, that builds a test program.
On Windows with MSVC they call `cl`, so run them from a Visual Studio developer prompt.
Under `--target` they default to the triple without its vendor and `-gcc` or `-g++`, such as `aarch64-linux-gnu-gcc`, and run the programs through `CARGO_TARGET_<TRIPLE>_RUNNER` when it is set.
With or without `--target`, a command in `RGMIN_C_TEST_RUNNER`, such as `valgrind --error-exitcode=99`, runs the programs in place of that runner, as in the capi row of CI.

CI also runs clippy with warnings denied, on Rust 1.88.0 and on the latest stable, for the first five feature sets:

```bash
rustup component add clippy --toolchain 1.88.0
cargo +1.88.0 clippy --locked --all-targets -- -D warnings
cargo +1.88.0 clippy --locked --all-targets --features capi -- -D warnings
cargo +1.88.0 clippy --locked --all-targets --features highs -- -D warnings
cargo +1.88.0 clippy --locked --all-targets --features capi,highs -- -D warnings
cargo +1.88.0 clippy --locked --all-targets --features capi,par,highs,slepc,primme,chase,libkrylov -- -D warnings
```

Run the same five commands with `+stable` for the second toolchain.

CI also builds and tests in release with Rust 1.93.0 and 1.94.1, which ship LLVM 21.
Its loop vectorizer once crashed on rgmin, and the debug builds above never run it:

```bash
rustup toolchain install 1.93.0 --profile minimal
cargo +1.93.0 build --locked --release
cargo +1.93.0 test --locked --release --lib
```

Install 1.94.1 the same way and run the two `cargo` commands with `+1.94.1`.

A change that users can see adds a fragment to `docs/newsfragments/` and nowhere else.
Name the fragment `<issue>.<type>.md`, or `+<name>.<type>.md` for a change with no issue.
The type is `added`, `changed`, `deprecated`, `removed`, `fixed` or `security`, the sections of [CHANGELOG.md](CHANGELOG.md).
The Unreleased sections of `CHANGELOG.md` and `docs/orgmode/changelog.org` stay empty: `towncrier build` writes the fragments into `CHANGELOG.md` and would otherwise list an entry twice.
`towncrier build` moves the fragments into `CHANGELOG.md` when a release is cut, with the settings in `towncrier.toml`.
The release commit copies those notes into `docs/orgmode/changelog.org` under a heading for the version.
`towncrier build --draft --version 0.0.0` prints the notes that the fragments give and changes no file.
`python3 scripts/check_changelog.py` fails when a fragment is paired with an Unreleased line, or when a file in `docs/newsfragments/` is not a fragment type.

## Releases

A release is a `vX.Y.Z` tag on `main`.
The tag starts `.github/workflows/publish.yml`, which uploads the crate to crates.io, and `.github/workflows/pypi.yml`, which uploads the Python package to PyPI.
The Python package takes its version from the `[package]` table of `Cargo.toml`, so the wheels and `rgmin.__version__` always carry the crate version.
`pypi.yml` builds the sdist and the wheels, tests every wheel, and uploads only when the tag is `v` followed by that version.

PyPI accepts the upload through trusted publishing: it trusts an OpenID Connect token from one workflow in one GitHub environment, so no API token is stored in the repository or its secrets.
Add the publisher once, as a pending publisher at <https://pypi.org/manage/account/publishing/>, and PyPI creates the `rgmin` project at the first upload:

| Field | Value |
| --- | --- |
| PyPI Project Name | `rgmin` |
| Owner | `OmniPotentRPC` |
| Repository name | `rgmin` |
| Workflow name | `pypi.yml` |
| Environment name | `pypi` |

Create the `pypi` environment under Settings, Environments before the first tag, with a deployment rule that admits only tags matching `v*`.
GitHub would otherwise create it at the first run with no rule.
Required reviewers on the environment hold each upload until one of them approves it.
Renaming `pypi.yml` or the `pypi` environment stops the uploads until the publisher on PyPI names the new ones.

Before tagging, run the PyPI workflow by hand on the release branch from the Actions tab.
A run that does not start from a tag builds and tests the sdist and every wheel, and skips the upload.
`publish.yml` has no such guard: run by hand, it publishes the crate from that branch.

The [contributing page](docs/orgmode/contributing/index.org) of the docs covers the rest: the C library check, the docs build, citations and the formula numbers the code follows.
