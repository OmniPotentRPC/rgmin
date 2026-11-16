`scripts/check_changelog.py` fails when `docs/newsfragments/` holds a fragment and the Unreleased section of `CHANGELOG.md` or `docs/orgmode/changelog.org` holds an entry, since that entry would appear twice after `towncrier build`.
It also fails on a file in `docs/newsfragments/` whose type is not in `towncrier.toml`, which towncrier skips without a warning.
CI runs it and then renders a towncrier draft.
