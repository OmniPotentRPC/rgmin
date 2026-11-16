#!/usr/bin/env python3
"""Fail when a changelog entry would appear twice in a release.

``towncrier build`` writes the fragments in ``docs/newsfragments/`` into
``CHANGELOG.md`` at a release and leaves the Unreleased sections of
``CHANGELOG.md`` and ``docs/orgmode/changelog.org`` as they are, so an
entry under either heading would sit beside its fragment. A fragment
whose type is not in ``towncrier.toml`` is skipped without a warning.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORG = ROOT / "docs" / "orgmode" / "changelog.org"
HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
ORG_HEADING = re.compile(r"^(\*+)\s+(.*?)\s*$")
ORG_COMMENT = re.compile(r"^\s*#(\s|$)")


def config() -> dict:
    with (ROOT / "towncrier.toml").open("rb") as handle:
        return tomllib.load(handle)["tool"]["towncrier"]


def fragment_type(name: str, types: set[str]) -> str | None:
    # towncrier's rule: the last dotted part, past the first, that names a type.
    for part in reversed(name.split(".")[1:]):
        if part in types:
            return part
    return None


def markdown_unreleased(path: Path, marker: str) -> list[tuple[int, str]]:
    lines = path.read_text().splitlines()
    if "## [Unreleased]" not in lines:
        return []
    start = lines.index("## [Unreleased]")
    end = start + 1
    while end < len(lines) and not lines[end].startswith("## ") and lines[end].strip() != marker:
        end += 1
    body = "\n".join(lines[start + 1 : end])
    body = HTML_COMMENT.sub(lambda match: "\n" * match.group().count("\n"), body)
    return [
        (start + 2 + offset, line.strip())
        for offset, line in enumerate(body.split("\n"))
        if line.strip()
    ]


def org_unreleased(path: Path) -> list[tuple[int, str]]:
    level = None
    entries: list[tuple[int, str]] = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        heading = ORG_HEADING.match(line)
        if level is None:
            if heading and heading.group(2) == "Unreleased":
                level = len(heading.group(1))
            continue
        if heading and len(heading.group(1)) <= level:
            break
        if line.strip() and not ORG_COMMENT.match(line):
            entries.append((number, line.strip()))
    return entries


def main() -> int:
    settings = config()
    directory = ROOT / settings["directory"]
    types = {kind["directory"] for kind in settings["type"]}
    failed = False
    fragments = []
    # towncrier build removes every fragment, and git keeps no empty directory.
    paths = sorted(directory.iterdir()) if directory.is_dir() else []
    for path in paths:
        if path.name.startswith("."):
            continue
        if path.is_file() and fragment_type(path.name, types):
            fragments.append(path)
            continue
        failed = True
        print(
            f"{path.relative_to(ROOT)}: towncrier skips this file; name it <issue>.<type>.md"
            f" or +<name>.<type>.md with a type from {', '.join(sorted(types))}"
        )
    changelog = ROOT / settings["filename"]
    marker = settings["start_string"].strip()
    unreleased = [(changelog, n, text) for n, text in markdown_unreleased(changelog, marker)]
    unreleased += [(ORG, n, text) for n, text in org_unreleased(ORG)]
    if fragments and unreleased:
        failed = True
        for path, number, text in unreleased:
            print(f"{path.relative_to(ROOT)}:{number}: entry under Unreleased: {text[:72]}")
        print(
            f"{settings['directory']}/ holds {len(fragments)} fragments, and towncrier build"
            " would add them beside the lines above; move each entry into a fragment"
        )
    if not failed:
        print(
            f"{len(fragments)} fragments in {settings['directory']}/,"
            f" {len(unreleased)} lines under Unreleased"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
