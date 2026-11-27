#!/usr/bin/env python3
"""Generate the public API inventory and the two llms.txt files."""

from __future__ import annotations

import argparse
import ast
import dataclasses
import difflib
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORG = ROOT / "docs" / "orgmode"
INVENTORY = ORG / "reference" / "public-api.org"
LLMS = ROOT / "llms.txt"
LLMS_FULL = ROOT / "llms-full.txt"


@dataclasses.dataclass(frozen=True, order=True)
class Item:
    surface: str
    kind: str
    name: str
    source: str
    line: int
    declaration: str
    # Crate, module, and type path in front of the bare name.
    scope: tuple[str, ...] = dataclasses.field(default=(), compare=False)


RUST_FUNCTION = re.compile(
    r"^\s*pub\s+(?!\()"
    r"(?P<qualifiers>(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?"
    r'(?:extern\s+"[^"]+"\s+)?)'
    r"fn\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
)
RUST_ITEM = re.compile(
    r"^\s*pub\s+(?!\()(?P<kind>mod|struct|enum|union|trait|type|const|static)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
)
RUST_FIELD = re.compile(r"^\s*pub\s+(?!\()(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*:")
TRAIT_MEMBER = re.compile(
    r"^\s*(?:unsafe\s+)?(?:async\s+)?(?P<kind>fn|type|const)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
)
ENUM_VARIANT = re.compile(r"^\s*(?P<name>[A-Z][A-Za-z0-9_]*)\s*(?:[,({=]|$)")
WITH_SETTER = re.compile(r"^\s*(?P<name>with_[A-Za-z0-9_]+)\s*=>")
# A trailing () stays outside the path, so ~Control::default()~ is Control::default.
MARKUP = re.compile(r"[~=]([A-Za-z_][A-Za-z0-9_:.-]*)(?:\(\))?[~=]")


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def rust_declaration(line: str) -> tuple[str, str] | None:
    function = RUST_FUNCTION.match(line)
    if function:
        qualifiers = function["qualifiers"]
        kind = "C export" if re.search(r'extern\s+"C"', qualifiers) else "function"
        return kind, function["name"]
    item = RUST_ITEM.match(line)
    if item:
        return item["kind"], item["name"]
    return None


def rust_trait_impl_member(line: str) -> tuple[str, str] | None:
    """Item in a trait impl. The trait sets the visibility, so `pub` is optional."""
    if re.match(r"\s*pub\s*\(", line):
        return None
    function = re.match(
        r"^\s*(?:pub\s+(?!\())?"
        r"(?P<qualifiers>(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?"
        r'(?:extern\s+"[^"]+"\s+)?)'
        r"fn\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)",
        line,
    )
    if function:
        qualifiers = function["qualifiers"] or ""
        kind = "C export" if re.search(r'extern\s+"C"', qualifiers) else "function"
        return kind, function["name"]
    item = re.match(
        r"^\s*(?:pub\s+(?!\())?(?P<kind>type|const|static)\s+"
        r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)",
        line,
    )
    if item:
        return item["kind"], item["name"]
    return None


def _collapse(text: str) -> str:
    return " ".join(text.strip().split())


def _advance_escape(cursor: int, size: int) -> int:
    """Step past a backslash and the escaped character, and stop at the end."""
    nxt = cursor + 1
    if nxt >= size:
        return size
    return nxt + 1


def _recorded_declaration(original: list[str], line_index: int) -> str:
    """Declaration cell. A template line is joined to the signature."""
    line = original[line_index]
    stripped = line.strip()
    if (
        stripped.startswith("template")
        and "(" not in stripped
        and "{" not in stripped
    ):
        nxt = line_index + 1
        while nxt < len(original) and not original[nxt].strip():
            nxt += 1
        if nxt < len(original):
            return _collapse(stripped + " " + original[nxt].strip())
    return _collapse(line)


def _blank_keep_newlines(text: str, spans: list[tuple[int, int]]) -> str:
    chars = list(text)
    for start, end in spans:
        for index in range(start, end):
            if chars[index] != "\n":
                chars[index] = " "
    return "".join(chars)


def _blank_rust(text: str) -> str:
    spans: list[tuple[int, int]] = []
    index = 0
    size = len(text)
    while index < size:
        if text.startswith("//", index):
            end = text.find("\n", index)
            spans.append((index, size if end < 0 else end))
            index = size if end < 0 else end
            continue
        if text.startswith("/*", index):
            depth = 1
            cursor = index + 2
            while cursor < size and depth:
                if text.startswith("/*", cursor):
                    depth += 1
                    cursor += 2
                    continue
                if text.startswith("*/", cursor):
                    depth -= 1
                    cursor += 2
                    continue
                cursor += 1
            spans.append((index, cursor))
            index = cursor
            continue
        prefix = ""
        for candidate in ("br", "cr", "b", "c", "r"):
            if text.startswith(candidate, index):
                prefix = candidate
                break
        cursor = index + len(prefix)
        if prefix and cursor < size and (text[cursor] == '"' or text[cursor] == "#"):
            hashes = 0
            while cursor < size and text[cursor] == "#":
                hashes += 1
                cursor += 1
            if cursor < size and text[cursor] == '"':
                cursor += 1
                if prefix in {"r", "br", "cr"}:
                    closer = '"' + ("#" * hashes)
                    end = text.find(closer, cursor)
                    cursor = size if end < 0 else end + len(closer)
                else:
                    while cursor < size:
                        if text[cursor] == "\\":
                            cursor = _advance_escape(cursor, size)
                            continue
                        if text[cursor] == '"':
                            cursor += 1
                            break
                        if text[cursor] == "\n":
                            break
                        cursor += 1
                spans.append((index, cursor))
                index = cursor
                continue
        if text[index] == '"':
            cursor = index + 1
            while cursor < size:
                if text[cursor] == "\\":
                    cursor = _advance_escape(cursor, size)
                    continue
                if text[cursor] == '"':
                    cursor += 1
                    break
                if text[cursor] == "\n":
                    break
                cursor += 1
            spans.append((index, cursor))
            index = cursor
            continue
        if text[index] == "'":
            if (
                index + 2 < size
                and text[index + 1] == "\\"
                and "'" in text[index + 2 : index + 6]
            ):
                end = text.find("'", index + 2)
                spans.append((index, size if end < 0 else end + 1))
                index = size if end < 0 else end + 1
                continue
            if index + 2 < size and text[index + 2] == "'" and text[index + 1] != "\\":
                spans.append((index, index + 3))
                index += 3
                continue
        index += 1
    return _blank_keep_newlines(text, spans)


def _blank_c(text: str) -> str:
    spans: list[tuple[int, int]] = []
    index = 0
    size = len(text)
    while index < size:
        if text.startswith("//", index):
            end = text.find("\n", index)
            spans.append((index, size if end < 0 else end))
            index = size if end < 0 else end
            continue
        if text.startswith("/*", index):
            end = text.find("*/", index + 2)
            cursor = size if end < 0 else end + 2
            spans.append((index, cursor))
            index = cursor
            continue
        if text[index] in "\"'":
            previous = text[index - 1] if index else ""
            following = text[index + 1] if index + 1 < size else ""
            if text[index] == "'" and previous.isalnum() and following.isalnum():
                index += 1
                continue
            quote = text[index]
            cursor = index + 1
            while cursor < size:
                if text[cursor] == "\\":
                    cursor = _advance_escape(cursor, size)
                    continue
                if text[cursor] == quote:
                    cursor += 1
                    break
                if text[cursor] == "\n":
                    break
                cursor += 1
            spans.append((index, cursor))
            index = cursor
            continue
        index += 1
    return _blank_keep_newlines(text, spans)


def _unrestricted_pub(text: str) -> bool:
    return re.match(r"\s*pub\s+(?!\()", text) is not None


def _split_top(text: str, sep: str = ",") -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for char in text:
        if char in "{[(":
            depth += 1
        elif char in "}])":
            depth = max(0, depth - 1)
        if char == sep and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(char)
    parts.append("".join(buf))
    return parts


def _brace_inner(text: str) -> tuple[str, str]:
    depth = 0
    for index, char in enumerate(text):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[1:index], text[index + 1 :]
    return text[1:], ""


def parse_use_tree(body: str) -> list[tuple[tuple[str, ...], str]]:
    """Public use paths as ``(segments, exported name)``."""
    found: list[tuple[tuple[str, ...], str]] = []

    def walk(prefix: list[str], text: str) -> None:
        for part in _split_top(text):
            piece = part.strip()
            if not piece:
                continue
            if piece == "*":
                found.append((tuple(prefix + ["*"]), "*"))
                continue
            brace = piece.find("{")
            if brace >= 0:
                head = piece[:brace].strip()
                if head.endswith("::"):
                    head = head[:-2].strip()
                segments = [item for item in head.split("::") if item]
                inner, tail = _brace_inner(piece[brace:])
                walk(prefix + segments, inner)
                if tail.strip():
                    walk(prefix, tail)
                continue
            alias = None
            path = piece
            if " as " in piece:
                path, alias = piece.rsplit(" as ", 1)
                path = path.strip()
                alias = alias.strip()
            segments = [item for item in path.split("::") if item]
            if not segments:
                continue
            found.append((tuple(prefix + segments), alias or segments[-1]))

    walk([], body.strip())
    return found


def _strip_generics(text: str) -> str:
    out: list[str] = []
    depth = 0
    for char in text:
        if char == "<":
            depth += 1
        elif char == ">":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(char)
    return "".join(out)


def _impl_owner(header: str) -> str:
    text = _strip_generics(header)
    text = re.sub(
        r"^\s*(?:pub(?:\s*\([^)]*\))?\s+)?(?:unsafe\s+)?(?:async\s+)?impl\b", "", text
    )
    text = re.split(r"\bwhere\b", text, maxsplit=1)[0]
    text = text.split("{", 1)[0]
    if " for " in text:
        text = text.rsplit(" for ", 1)[1]
    names = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)
    skip = {"mut", "const", "dyn", "for", "where", "impl", "unsafe", "async"}
    names = [name for name in names if name not in skip]
    return names[-1] if names else ""


def _impl_trait(header: str) -> str:
    """Trait name of ``impl Trait for Type``, or empty for an inherent impl."""
    text = _strip_generics(header)
    text = re.sub(
        r"^\s*(?:pub(?:\s*\([^)]*\))?\s+)?(?:unsafe\s+)?(?:async\s+)?impl\b",
        "",
        text,
    )
    text = re.split(r"\bwhere\b", text, maxsplit=1)[0]
    text = text.split("{", 1)[0]
    if " for " not in text:
        return ""
    trait = text.rsplit(" for ", 1)[0]
    names = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", trait)
    skip = {"mut", "const", "dyn", "for", "where", "impl", "unsafe", "async"}
    names = [name for name in names if name not in skip]
    return names[-1] if names else ""


def _item_identity(header: str) -> tuple[str, str] | None:
    if re.search(r"\bfn\s+[A-Za-z_]", header):
        match = re.search(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)", header)
        if match:
            return "fn", match.group(1)
    match = re.search(
        r"\b(struct|enum|union|trait|mod|type)\s+([A-Za-z_][A-Za-z0-9_]*)", header
    )
    if match:
        return match.group(1), match.group(2)
    if not re.search(r"\bfn\b", header):
        match = re.search(r"\b(const|static)\s+([A-Za-z_][A-Za-z0-9_]*)", header)
        if match:
            return match.group(1), match.group(2)
    return None


def _expand_rust(
    path: Path, root: Path, stack: set[Path]
) -> list[tuple[str, int, str]]:
    path = path.resolve()
    if path in stack:
        return []
    stack.add(path)
    relative_path = path.relative_to(root).as_posix()
    rows: list[tuple[str, int, str]] = []
    include = re.compile(r'^include!\(\s*"([^"]+)"\s*\)\s*;?\s*$')
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = include.match(line.strip())
        if match:
            included = (path.parent / match.group(1)).resolve()
            if included.is_file():
                rows.extend(_expand_rust(included, root, stack))
                continue
        rows.append((line, number, relative_path))
    stack.remove(path)
    return rows


def _child_module_file(anchor: Path, name: str) -> Path | None:
    if anchor.name in {"lib.rs", "main.rs", "mod.rs"}:
        base = anchor.parent
    else:
        base = anchor.parent / anchor.stem
    for candidate in (base / f"{name}.rs", base / name / "mod.rs"):
        if candidate.is_file():
            return candidate
    return None


@dataclasses.dataclass
class _Scope:
    kind: str
    depth: int
    module: tuple[str, ...]
    name: str = ""
    trait: str = ""


@dataclasses.dataclass
class _RustRec:
    kind: str
    name: str
    source: str
    line: int
    declaration: str
    module: tuple[str, ...]
    role: str
    owner: str
    public: bool
    trait: str = ""


@dataclasses.dataclass
class _Reexport:
    module: tuple[str, ...]
    segments: tuple[str, ...]
    alias: str
    record: int


@dataclasses.dataclass
class _GlobUse:
    module: tuple[str, ...]
    segments: tuple[str, ...]
    source: str
    line: int
    declaration: str


def _balanced_delimiters(text: str) -> tuple[int, int, int]:
    paren = bracket = brace = 0
    for char in text:
        if char == "(":
            paren += 1
        elif char == ")":
            paren -= 1
        elif char == "[":
            bracket += 1
        elif char == "]":
            bracket -= 1
        elif char == "{":
            brace += 1
        elif char == "}":
            brace -= 1
    return paren, bracket, brace


def _rust_scope(record: _RustRec) -> tuple[str, ...]:
    """Crate, module, and owning type in front of the bare name."""
    module = tuple(part for part in record.module if part)
    if "::" in record.name:
        head = tuple(part for part in record.name.split("::")[:-1] if part)
        return ("rgmin",) + module + head
    if record.owner:
        return ("rgmin",) + module + (record.owner,)
    return ("rgmin",) + module


def rust_items(root: Path | None = None) -> list[Item]:
    """Public Rust items reachable from the crate root.

    A public module chain or a ``pub use`` re-export makes an item
    reachable. Feature gates stay in the set. ``cargo doc`` JSON is
    nightly-only here, so the module walk is the check.
    """
    base = root or ROOT
    lib = base / "src" / "lib.rs"
    if not lib.is_file():
        raise SystemExit(f"{lib.relative_to(base)} is missing")
    records: list[_RustRec] = []
    reexports: list[_Reexport] = []
    globs: list[_GlobUse] = []
    children: dict[tuple[str, ...], dict[str, tuple[tuple[str, ...], bool]]] = {}
    defined: dict[tuple[str, ...], dict[str, list[int]]] = {}
    types: dict[tuple[tuple[str, ...], str], int] = {}
    local_traits: set[str] = set()
    seen_files: set[Path] = set()

    def add_record(record: _RustRec) -> int:
        records.append(record)
        return len(records) - 1

    def bind(module: tuple[str, ...], name: str, record_id: int) -> None:
        defined.setdefault(module, {}).setdefault(name, []).append(record_id)

    def scan_file(path: Path, module: tuple[str, ...]) -> None:
        resolved = path.resolve()
        if resolved in seen_files:
            return
        seen_files.add(resolved)
        children.setdefault(module, {})
        defined.setdefault(module, {})
        rows = _expand_rust(resolved, base, set())
        raw_lines = [row[0] for row in rows]
        blank_lines = _blank_rust("\n".join(raw_lines)).splitlines()
        while len(blank_lines) < len(rows):
            blank_lines.append("")
        depth = 0
        stack = [_Scope("module", 0, module)]
        index = 0
        total = len(rows)

        def pop() -> None:
            while len(stack) > 1 and depth < stack[-1].depth:
                stack.pop()

        def add_braces(text: str) -> None:
            nonlocal depth
            depth += text.count("{") - text.count("}")
            depth = max(depth, 0)
            pop()

        while index < total:
            pop()
            code = blank_lines[index]
            stripped = code.strip()
            scope = stack[-1]
            at_items = depth == scope.depth and scope.kind in {
                "module",
                "impl",
                "struct",
                "enum",
                "trait",
                "union",
            }
            if not stripped:
                add_braces(code)
                index += 1
                continue
            if stripped.startswith("#"):
                paren = bracket = 0
                while index < total:
                    code = blank_lines[index]
                    paren += code.count("(") - code.count(")")
                    bracket += code.count("[") - code.count("]")
                    add_braces(code)
                    index += 1
                    if paren <= 0 and bracket <= 0:
                        break
                continue
            if stripped.startswith("}"):
                add_braces(code)
                index += 1
                continue
            if at_items and scope.kind in {"struct", "union"}:
                field = RUST_FIELD.match(code)
                if field:
                    raw, number, source = rows[index]
                    add_record(
                        _RustRec(
                            "public field",
                            field.group("name"),
                            source,
                            number,
                            _collapse(raw),
                            scope.module,
                            "field",
                            scope.name,
                            True,
                        )
                    )
                add_braces(code)
                index += 1
                continue
            if at_items and scope.kind == "enum":
                variant = ENUM_VARIANT.match(code)
                if variant:
                    raw, number, source = rows[index]
                    add_record(
                        _RustRec(
                            "enum variant",
                            f"{scope.name}::{variant.group('name')}",
                            source,
                            number,
                            _collapse(raw),
                            scope.module,
                            "variant",
                            scope.name,
                            True,
                        )
                    )
                add_braces(code)
                index += 1
                continue
            if at_items and scope.kind == "trait":
                member = TRAIT_MEMBER.match(code)
                if member:
                    raw, number, source = rows[index]
                    add_record(
                        _RustRec(
                            f"trait {member.group('kind')}",
                            f"{scope.name}::{member.group('name')}",
                            source,
                            number,
                            _collapse(raw),
                            scope.module,
                            "trait_member",
                            scope.name,
                            True,
                        )
                    )
                add_braces(code)
                index += 1
                continue
            if scope.kind == "fn" or not at_items:
                if scope.kind != "fn" and "with_" in code and WITH_SETTER.match(code):
                    owner = ""
                    for inner in reversed(stack):
                        if inner.kind in {"struct", "enum", "union"}:
                            owner = inner.name
                            break
                    if owner:
                        raw, number, source = rows[index]
                        add_record(
                            _RustRec(
                                "generated method",
                                WITH_SETTER.match(code).group("name"),
                                source,
                                number,
                                _collapse(raw),
                                scope.module,
                                "setter",
                                owner,
                                True,
                            )
                        )
                add_braces(code)
                index += 1
                continue
            if re.match(
                r"(?:pub(?:\s*\([^)]*\))?\s+)?(?:unsafe\s+)?[A-Za-z_][A-Za-z0-9_]*!",
                stripped,
            ):
                owner_match = re.search(
                    r"with_setters!\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)", stripped
                )
                owner = owner_match.group(1) if owner_match else ""
                paren = bracket = brace = 0
                started = False
                while index < total:
                    code = blank_lines[index]
                    if owner and WITH_SETTER.match(code):
                        raw, number, source = rows[index]
                        add_record(
                            _RustRec(
                                "generated method",
                                WITH_SETTER.match(code).group("name"),
                                source,
                                number,
                                _collapse(raw),
                                scope.module,
                                "setter",
                                owner,
                                True,
                            )
                        )
                    for char in code:
                        if char == "(":
                            paren += 1
                            started = True
                        elif char == ")":
                            paren -= 1
                        elif char == "[":
                            bracket += 1
                            started = True
                        elif char == "]":
                            bracket -= 1
                        elif char == "{":
                            brace += 1
                            started = True
                            depth += 1
                        elif char == "}":
                            brace -= 1
                            depth = max(0, depth - 1)
                    index += 1
                    if started and paren <= 0 and bracket <= 0 and brace <= 0:
                        break
                pop()
                continue
            itemish = re.match(
                r"\s*(?:pub\b(?:\s*\([^)]*\))?|unsafe\b|async\b|extern\b|const\b|static\b|"
                r"struct\b|enum\b|union\b|trait\b|impl\b|mod\b|fn\b|type\b|use\b)",
                stripped,
            )
            if not itemish:
                add_braces(code)
                index += 1
                continue
            header_at = index
            paren = bracket = brace = 0
            use_stmt = stripped.startswith(("use ", "pub use "))
            opened = False
            while index < total:
                code = blank_lines[index]
                if index == header_at:
                    piece = code.lstrip()
                    use_stmt = piece.startswith(("use ", "pub use "))
                paren += code.count("(") - code.count(")")
                bracket += code.count("[") - code.count("]")
                brace += code.count("{") - code.count("}")
                if code.count("{") and not use_stmt:
                    opened = True
                index += 1
                balanced = paren <= 0 and bracket <= 0
                if use_stmt and ";" in code and brace <= 0 and balanced:
                    break
                if not use_stmt and opened and brace > 0 and balanced:
                    break
                if (
                    not use_stmt
                    and ";" in code
                    and brace <= 0
                    and balanced
                    and not opened
                ):
                    break
                if not use_stmt and opened and brace <= 0 and balanced:
                    break
            header_rows = rows[header_at:index]
            header_blank = "\n".join(blank_lines[header_at:index])
            header_raw = "\n".join(row[0] for row in header_rows)
            depth += header_blank.count("{") - header_blank.count("}")
            depth = max(depth, 0)
            body_open = opened and header_blank.count("{") > header_blank.count("}")
            first_raw, first_line, first_source = header_rows[0]
            if use_stmt and _unrestricted_pub(header_blank):
                body = header_blank.split("pub use", 1)[1]
                body = body.rsplit(";", 1)[0]
                for segments, alias in parse_use_tree(body):
                    declaration = _collapse(header_raw)
                    if segments and segments[-1] == "*":
                        globs.append(
                            _GlobUse(
                                scope.module,
                                segments,
                                first_source,
                                first_line,
                                declaration,
                            )
                        )
                        reexports.append(_Reexport(scope.module, segments, alias, -1))
                        continue
                    record_id = add_record(
                        _RustRec(
                            "reexport",
                            alias,
                            first_source,
                            first_line,
                            declaration,
                            scope.module,
                            "reexport",
                            "",
                            True,
                        )
                    )
                    reexports.append(
                        _Reexport(scope.module, segments, alias, record_id)
                    )
            elif scope.kind == "module" and re.match(
                r"\s*(?:pub(?:\s*\([^)]*\))?\s+)?mod\s+[A-Za-z_]", header_blank
            ):
                identity = _item_identity(header_blank)
                if identity and identity[0] == "mod":
                    name = identity[1]
                    public = _unrestricted_pub(header_blank)
                    child = scope.module + (name,)
                    if public:
                        record_id = add_record(
                            _RustRec(
                                "mod",
                                name,
                                first_source,
                                first_line,
                                _collapse(first_raw),
                                scope.module,
                                "item",
                                "",
                                True,
                            )
                        )
                        bind(scope.module, name, record_id)
                    children.setdefault(scope.module, {})[name] = (child, public)
                    children.setdefault(child, {})
                    defined.setdefault(child, {})
                    if body_open:
                        stack.append(_Scope("module", depth, child, name))
                    else:
                        child_path = _child_module_file(resolved, name)
                        if child_path is not None:
                            scan_file(child_path, child)
            elif re.match(
                r"\s*(?:pub(?:\s*\([^)]*\))?\s+)?(?:unsafe\s+)?(?:async\s+)?impl\b",
                header_blank,
            ):
                if body_open:
                    stack.append(
                        _Scope(
                            "impl",
                            depth,
                            scope.module,
                            _impl_owner(header_blank),
                            _impl_trait(header_blank),
                        )
                    )
            else:
                identity = _item_identity(header_blank)
                public = _unrestricted_pub(header_blank)
                if identity and scope.kind == "module":
                    kind, name = identity
                    if kind == "trait":
                        local_traits.add(name)
                    if kind == "fn":
                        parsed = rust_declaration(_collapse(header_raw))
                        kind = parsed[0] if parsed else "function"
                    if public and kind != "mod":
                        record_id = add_record(
                            _RustRec(
                                kind,
                                name,
                                first_source,
                                first_line,
                                _collapse(first_raw),
                                scope.module,
                                "item",
                                "",
                                True,
                            )
                        )
                        bind(scope.module, name, record_id)
                        if kind in {"struct", "enum", "trait", "union", "type"}:
                            types[(scope.module, name)] = record_id
                    if body_open and kind in {"struct", "enum", "trait", "union", "fn"}:
                        stack.append(_Scope(kind, depth, scope.module, name))
                    elif body_open:
                        stack.append(_Scope("fn", depth, scope.module, ""))
                elif scope.kind == "impl" and (public or scope.trait):
                    header = _collapse(header_raw)
                    parsed = (
                        rust_trait_impl_member(header)
                        if scope.trait
                        else rust_declaration(header)
                    )
                    if parsed and parsed[0] in {
                        "function",
                        "C export",
                        "const",
                        "type",
                    }:
                        kind, name = parsed
                        add_record(
                            _RustRec(
                                kind,
                                name,
                                first_source,
                                first_line,
                                header,
                                scope.module,
                                "method",
                                scope.name,
                                True,
                                scope.trait,
                            )
                        )
                        if body_open and kind in {"function", "C export"}:
                            stack.append(_Scope("fn", depth, scope.module, name))
                        elif body_open:
                            stack.append(_Scope("fn", depth, scope.module, ""))
                    elif body_open:
                        stack.append(_Scope("fn", depth, scope.module, ""))
                elif identity and scope.kind == "trait" and public:
                    add_record(
                        _RustRec(
                            f"trait {identity[0]}",
                            f"{scope.name}::{identity[1]}",
                            first_source,
                            first_line,
                            _collapse(header_raw),
                            scope.module,
                            "trait_member",
                            scope.name,
                            True,
                        )
                    )
                    if body_open and identity[0] == "fn":
                        stack.append(_Scope("fn", depth, scope.module, identity[1]))
                elif scope.kind == "struct" or scope.kind == "union":
                    field = RUST_FIELD.match(header_blank.splitlines()[0])
                    if field:
                        add_record(
                            _RustRec(
                                "public field",
                                field.group("name"),
                                first_source,
                                first_line,
                                _collapse(first_raw),
                                scope.module,
                                "field",
                                scope.name,
                                True,
                            )
                        )
                elif scope.kind == "enum":
                    variant = ENUM_VARIANT.match(header_blank.splitlines()[0])
                    if variant:
                        add_record(
                            _RustRec(
                                "enum variant",
                                f"{scope.name}::{variant.group('name')}",
                                first_source,
                                first_line,
                                _collapse(first_raw),
                                scope.module,
                                "variant",
                                scope.name,
                                True,
                            )
                        )
                    if body_open:
                        stack.append(_Scope("fn", depth, scope.module, ""))
                elif scope.kind == "trait":
                    member = TRAIT_MEMBER.match(header_blank.splitlines()[0])
                    if member:
                        add_record(
                            _RustRec(
                                f"trait {member.group('kind')}",
                                f"{scope.name}::{member.group('name')}",
                                first_source,
                                first_line,
                                _collapse(first_raw),
                                scope.module,
                                "trait_member",
                                scope.name,
                                True,
                            )
                        )
                    if body_open:
                        stack.append(_Scope("fn", depth, scope.module, ""))
                elif body_open:
                    stack.append(_Scope("fn", depth, scope.module, ""))
            pop()

    scan_file(lib, ())
    reachable_mods = {()}
    reachable: set[int] = set()

    def mark_item(record_id: int) -> bool:
        if record_id in reachable:
            return False
        reachable.add(record_id)
        return True

    def lookup_local(module: tuple[str, ...], name: str) -> tuple[str, object] | None:
        child = children.get(module, {}).get(name)
        if child is not None:
            return ("module", child[0])
        record_ids = defined.get(module, {}).get(name)
        if record_ids:
            return ("item", record_ids)
        for rex in reexports:
            if rex.module == module and rex.alias == name:
                return ("reexport", rex)
        return None

    def resolve(
        module: tuple[str, ...],
        segments: tuple[str, ...],
        trail: set[tuple[tuple[str, ...], tuple[str, ...]]],
    ) -> list[tuple[str, object]]:
        key = (module, segments)
        if key in trail or not segments:
            return [("module", module)] if not segments else []
        trail.add(key)
        name, rest = segments[0], segments[1:]
        if name == "crate":
            return resolve((), rest, trail)
        if name == "self":
            return resolve(module, rest, trail)
        if name == "super":
            return resolve(module[:-1], rest, trail)
        if name == "*":
            found: list[tuple[str, object]] = []
            for record_ids in defined.get(module, {}).values():
                for record_id in record_ids:
                    found.append(("item", record_id))
            for child_path, public in children.get(module, {}).values():
                if public:
                    found.append(("module", child_path))
            for rex in reexports:
                if rex.module == module:
                    found.extend(resolve(module, rex.segments, trail))
            return found
        hit = lookup_local(module, name)
        if (
            hit is None
            and segments == (name, *rest)
            and name not in {"crate", "self", "super"}
        ):
            cursor: tuple[str, ...] | None = module
            while cursor is not None and hit is None:
                if cursor != module:
                    hit = lookup_local(cursor, name)
                if not cursor:
                    break
                cursor = cursor[:-1]
        if hit is None:
            return []
        kind, value = hit
        if kind == "module":
            return resolve(value, rest, trail) if rest else [("module", value)]
        if kind == "item":
            return [("item", item) for item in value] if not rest else []
        rex = value
        followed = resolve(rex.module, rex.segments, trail)
        if not rest:
            return followed
        output: list[tuple[str, object]] = []
        for followed_kind, followed_value in followed:
            if followed_kind == "module":
                output.extend(resolve(followed_value, rest, trail))
        return output

    changed = True
    expanded_globs: set[int] = set()
    while changed:
        changed = False
        for module in tuple(reachable_mods):
            for child, public in children.get(module, {}).values():
                if public and child not in reachable_mods:
                    reachable_mods.add(child)
                    changed = True
            for record_ids in defined.get(module, {}).values():
                for record_id in record_ids:
                    if mark_item(record_id):
                        changed = True
            for rex in reexports:
                if rex.module != module:
                    continue
                if rex.record >= 0 and mark_item(rex.record):
                    changed = True
                for kind, value in resolve(module, rex.segments, set()):
                    if kind == "module" and value not in reachable_mods:
                        reachable_mods.add(value)
                        changed = True
                    elif kind == "item" and mark_item(value):
                        changed = True
            for index, glob in enumerate(globs):
                if glob.module not in reachable_mods or index in expanded_globs:
                    continue
                names: list[str] = []
                seen_names: set[str] = set()
                item_ids: list[int] = []
                module_paths: list[tuple[str, ...]] = []
                for kind, value in resolve(glob.module, glob.segments, set()):
                    if kind == "item":
                        item_ids.append(value)
                        name = records[value].name
                        if name != "*" and name not in seen_names:
                            seen_names.add(name)
                            names.append(name)
                    elif kind == "module":
                        module_paths.append(value)
                        name = value[-1] if value else ""
                        if name and name != "*" and name not in seen_names:
                            seen_names.add(name)
                            names.append(name)
                if not names:
                    raise SystemExit(
                        f"{glob.source}:{glob.line}: pub use glob expanded to no "
                        f"public names: {glob.declaration}"
                    )
                expanded_globs.add(index)
                for name in names:
                    mark_item(
                        add_record(
                            _RustRec(
                                "reexport",
                                name,
                                glob.source,
                                glob.line,
                                glob.declaration,
                                glob.module,
                                "reexport",
                                "",
                                True,
                            )
                        )
                    )
                    changed = True
                for record_id in item_ids:
                    if mark_item(record_id):
                        changed = True
                for path in module_paths:
                    if path not in reachable_mods:
                        reachable_mods.add(path)
                        changed = True
    reachable_traits = {
        record.name
        for record_id, record in enumerate(records)
        if record_id in reachable and record.kind == "trait"
    }
    for record_id, record in enumerate(records):
        if record.role not in {"method", "field", "setter", "variant", "trait_member"}:
            continue
        if record.trait and (
            record.trait in local_traits and record.trait not in reachable_traits
        ):
            continue
        owner_id = types.get((record.module, record.owner))
        if owner_id is None:
            matches = [
                item for (mod, name), item in types.items() if name == record.owner
            ]
            owner_id = matches[0] if len(matches) == 1 else None
        if owner_id is not None and owner_id in reachable:
            mark_item(record_id)
    items: list[Item] = []
    for record_id, record in enumerate(records):
        if record_id not in reachable:
            continue
        items.append(
            Item(
                "Rust",
                record.kind,
                record.name,
                record.source,
                record.line,
                record.declaration,
                _rust_scope(record),
            )
        )
    return items


_HEADER_NAME = re.compile(r"^(?:(?:rgmin|xts)_[A-Za-z0-9_]+|(?:RGMIN|XTS)_[A-Z0-9_]+)$")
_C_KEYWORDS = {
    "struct",
    "enum",
    "union",
    "class",
    "const",
    "volatile",
    "unsigned",
    "signed",
    "long",
    "short",
    "int",
    "char",
    "void",
    "float",
    "double",
    "bool",
    "typedef",
    "static",
    "inline",
    "extern",
    "constexpr",
    "typename",
    "template",
}


_CXX_KEYWORDS = _C_KEYWORDS | {
    "if",
    "else",
    "for",
    "while",
    "do",
    "switch",
    "case",
    "break",
    "continue",
    "return",
    "goto",
    "try",
    "catch",
    "throw",
    "new",
    "delete",
    "sizeof",
    "decltype",
    "typeid",
    "static_assert",
    "noexcept",
    "operator",
    "public",
    "private",
    "protected",
    "namespace",
    "using",
    "this",
    "true",
    "false",
    "nullptr",
    "auto",
    "concept",
    "requires",
}


def _header_name(name: str) -> bool:
    return _HEADER_NAME.match(name) is not None


def _include_guard(text: str) -> str | None:
    """Macro that guards this header, if the file has one."""
    pending: str | None = None
    for line in _blank_c(text).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if pending is None:
            match = re.match(r"#\s*ifndef\s+([A-Za-z_][A-Za-z0-9_]*)\s*$", stripped)
            if match is None:
                match = re.match(
                    r"#\s*if\s+!\s*defined\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*$",
                    stripped,
                )
            if match:
                pending = match.group(1)
                continue
            if stripped.startswith("#"):
                return None
            continue
        if re.match(rf"#\s*define\s+{re.escape(pending)}\s*$", stripped):
            return pending
        return None
    return None


def _namespace_public(stmt: str) -> bool:
    text = _strip_c_prefixes(_strip_template(stmt.strip()))
    match = re.match(r"namespace\b(.*)$", text)
    if not match:
        return False
    name = match.group(1).strip().split("{", 1)[0].strip()
    if not name or "=" in name:
        return False
    parts = [part.strip() for part in name.split("::") if part.strip()]
    return bool(parts) and all(part != "detail" for part in parts)


def _type_kind(stmt: str) -> str:
    text = _strip_c_prefixes(_strip_template(stmt.strip()))
    if re.match(r"enum\b", text):
        return "enum"
    if re.match(r"struct\b", text):
        return "struct"
    if re.match(r"union\b", text):
        return "union"
    if re.match(r"class\b", text):
        return "class"
    return "type"


def _using_bindings(stmt: str) -> list[tuple[str, str]]:
    body = stmt.strip()
    if not body.startswith("using"):
        return []
    body = body[len("using") :].strip()
    if not body or body.startswith("namespace"):
        return []
    depth = 0
    equals = -1
    for index, char in enumerate(body):
        if char in "<(":
            depth += 1
        elif char in ">)":
            depth = max(0, depth - 1)
        elif char == "=" and depth == 0:
            equals = index
            break
    if equals >= 0:
        ident = body[:equals].strip().split("::")[-1].strip()
        return [(ident, "type")] if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", ident) else []
    found: list[tuple[str, str]] = []
    for part in body.split(","):
        ident = part.strip().split("::")[-1].strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", ident):
            found.append((ident, "reexport"))
    return found


def _strip_c_prefixes(text: str) -> str:
    prefixes = (
        "typedef",
        "static",
        "inline",
        "constexpr",
        "consteval",
        "virtual",
        "explicit",
        "friend",
        "mutable",
        "volatile",
        "const",
        "typename",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if text.startswith(prefix) and (
                len(text) == len(prefix) or text[len(prefix)].isspace()
            ):
                text = text[len(prefix) :].strip()
                changed = True
    return text


def _strip_template(text: str) -> str:
    stripped = text.lstrip()
    if not stripped.startswith("template"):
        return stripped
    rest = stripped[len("template") :].lstrip()
    if not rest.startswith("<"):
        return rest
    depth = 0
    for index, char in enumerate(rest):
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
            if depth == 0:
                return rest[index + 1 :].strip()
    return rest


def _c_tag(stmt: str) -> str | None:
    stripped = _strip_c_prefixes(_strip_template(stmt.strip()))
    match = re.match(
        r"(?:enum(?:\s+class|\s+struct)?|struct|union|class)\s+([A-Za-z_][A-Za-z0-9_]*)",
        stripped,
    )
    if match:
        return match.group(1)
    return None


def _open_kind(stmt: str) -> str:
    text = _strip_template(stmt.strip())
    text = _strip_c_prefixes(text)
    if re.match(r"extern\s*$", text):
        return "link"
    if re.match(r"namespace\b", text):
        return "namespace"
    if re.match(r"enum\b", text):
        return "enum"
    if re.match(r"(?:class|struct|union)\b", text) and "(" not in stmt:
        return "class"
    if "(" in stmt:
        return "fn"
    if re.match(r"(?:class|struct|union)\b", text):
        return "class"
    return "skip"


def _assigned_before_call(stmt: str) -> bool:
    paren = 0
    for index, char in enumerate(stmt):
        if char == "(":
            paren += 1
        elif char == ")":
            paren = max(0, paren - 1)
        elif char == "=" and paren == 0:
            previous = stmt[index - 1] if index else ""
            nxt = stmt[index + 1] if index + 1 < len(stmt) else ""
            if previous not in "=!<>" and nxt != "=":
                return True
    return False


def _declarator_name(stmt: str) -> str | None:
    position = stmt.find("(")
    if position < 0:
        return None
    cursor = position - 1
    while cursor >= 0 and stmt[cursor].isspace():
        cursor -= 1
    while cursor >= 0 and stmt[cursor] in "*&":
        cursor -= 1
        while cursor >= 0 and stmt[cursor].isspace():
            cursor -= 1
    if cursor < 0 or not (stmt[cursor].isalnum() or stmt[cursor] == "_"):
        return None
    end = cursor + 1
    while cursor >= 0 and (stmt[cursor].isalnum() or stmt[cursor] == "_"):
        cursor -= 1
    if cursor >= 0 and stmt[cursor] in ".:~":
        return None
    return stmt[cursor + 1 : end]


def _typedef_entries(stmt: str) -> list[tuple[str, str]]:
    """Names a typedef introduces, and a kind for a distinct tagged type."""
    names = _typedef_names(stmt)
    rest = stmt.lstrip()
    if rest.startswith("typedef"):
        rest = rest[len("typedef") :].strip()
    head = rest.split(None, 1)[0] if rest else ""
    kinds = {
        "struct": "struct",
        "enum": "enum",
        "union": "union",
        "class": "class",
    }
    if head in kinds and len(names) >= 2 and names[0] != names[-1]:
        return [(names[0], kinds[head]), (names[-1], "symbol")]
    return [(name, "symbol") for name in dict.fromkeys(names)]


def _typedef_names(stmt: str) -> list[str]:
    pointer = re.search(r"\(\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)", stmt)
    if pointer:
        return [pointer.group(1)]
    rest = stmt.lstrip()
    if rest.startswith("typedef"):
        rest = rest[len("typedef") :].strip()
    names = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", rest)
    names = [name for name in names if name not in _C_KEYWORDS]
    if not names:
        return []
    head = rest.split(None, 1)[0] if rest else ""
    if head in {"struct", "enum", "union", "class"} and len(names) >= 2:
        return [names[0], names[-1]]
    return [names[-1]]


_DLPACK_TYPE = re.compile(r"^DL[A-Z][A-Za-z0-9]*$")
_PREPROCESSOR_IF = re.compile(r"#\s*(?:if|ifdef|ifndef)\b")
_PREPROCESSOR_ELIF = re.compile(r"#\s*elif\b")
_PREPROCESSOR_ELSE = re.compile(r"#\s*else\b")
_PREPROCESSOR_ENDIF = re.compile(r"#\s*endif\b")


def _mask_else_branches(text: str) -> str:
    """Blank ``#else`` and ``#elif`` regions, keeping line numbers.

    A header that spells the same declaration in both arms, such as the
    DLPack device enum, would otherwise be parsed twice and the second
    arm would swallow the typedefs that follow.
    """
    lines = text.splitlines()
    kept = [True]
    output: list[str] = []
    for line in lines:
        stripped = line.strip()
        if _PREPROCESSOR_IF.match(stripped):
            kept.append(kept[-1])
            output.append("")
            continue
        if _PREPROCESSOR_ELIF.match(stripped) or _PREPROCESSOR_ELSE.match(stripped):
            if len(kept) > 1:
                kept[-1] = False
            output.append("")
            continue
        if _PREPROCESSOR_ENDIF.match(stripped):
            if len(kept) > 1:
                kept.pop()
            output.append("")
            continue
        output.append(line if kept[-1] else "")
    return "\n".join(output)


@dataclasses.dataclass
class _CScope:
    kind: str
    depth: int
    typedef_tail: bool
    exposed: bool
    access: bool
    name: str = ""


def header_declared_names(
    text: str, *, cxx: bool = False
) -> list[tuple[int, str, str, str, tuple[str, ...]]]:
    """Line, name, source line, and kind of each inventoried declaration.

    Prefixed C names stay. A C++ header also records public namespace types,
    free functions, and public class and struct members. Shipped DLPack
    typedefs are kept. The macro that guards its own header is omitted.
    """
    guard = _include_guard(text)
    text = _mask_else_branches(text)
    blank = _blank_c(text)
    original = text.splitlines()
    masked = blank.splitlines()
    if len(masked) < len(original):
        masked.extend([""] * (len(original) - len(masked)))
    stream: list[tuple[str, int]] = []
    found: list[tuple[int, str, str, str, tuple[str, ...]]] = []
    seen: set[tuple[str, str]] = set()
    seen_members: set[tuple[tuple[str, ...], str]] = set()

    def add(
        name: str, line_index: int, *, type_name: bool = False, kind: str | None = None
    ) -> None:
        shipped = type_name and _DLPACK_TYPE.match(name) is not None
        resolved = kind if kind is not None else ("constant" if name.isupper() else "symbol")
        if (
            not name
            or name == guard
            or (name, resolved) in seen
            or not (_header_name(name) or shipped)
            or line_index >= len(original)
        ):
            return
        seen.add((name, resolved))
        found.append(
            (
                line_index + 1,
                name,
                _recorded_declaration(original, line_index),
                resolved,
                (),
            )
        )

    def member_path() -> tuple[str, ...]:
        return tuple(
            scope.name
            for scope in scopes
            if scope.kind in {"namespace", "class"} and scope.name
        )

    def add_cxx(name: str, line_index: int, kind: str) -> None:
        scope = scopes[-1]
        visible = scope.exposed and scope.access and scope.kind in {
            "file",
            "namespace",
            "class",
        }
        if (
            not cxx
            or not name
            or name == guard
            or name in _CXX_KEYWORDS
            or line_index >= len(original)
            or not visible
        ):
            return
        if scope.kind == "class":
            if name == scope.name:
                return
            token = (member_path(), name)
            if token in seen_members:
                return
            seen_members.add(token)
        elif any(existing == name for existing, _kind in seen):
            return
        else:
            seen.add((name, kind))
        found.append(
            (
                line_index + 1,
                name,
                _recorded_declaration(original, line_index),
                kind,
                member_path(),
            )
        )

    index = 0
    total = len(original)
    while index < total:
        line = masked[index]
        if line.lstrip().startswith("#"):
            define = re.match(r"#\s*define\s+([A-Za-z_][A-Za-z0-9_]*)", line.lstrip())
            if define:
                add(define.group(1), index)
            while index < total and masked[index].rstrip().endswith("\\"):
                index += 1
            index += 1
            continue
        for char in line:
            stream.append((char, index))
        stream.append(("\n", index))
        index += 1

    brace = 0
    paren = 0
    scopes: list[_CScope] = [_CScope("file", 0, False, True, True)]
    stmt: list[str] = []
    stmt_line = 0
    cursor = 0
    limit = len(stream)

    def top() -> str:
        return scopes[-1].kind

    def declaration_scope() -> bool:
        return top() in {"file", "link", "class", "namespace"}

    def scope_name(kind: str, opened: str) -> str:
        if kind == "namespace":
            text = _strip_c_prefixes(_strip_template(opened.strip()))
            match = re.match(r"namespace\b(.*)$", text)
            if not match:
                return ""
            name = match.group(1).strip().split("{", 1)[0].strip()
            if not name or "=" in name:
                return ""
            return name.split("::")[-1].strip()
        if kind in {"class", "enum"}:
            return _c_tag(opened) or ""
        return ""

    def push(kind: str, depth: int, typedef_tail: bool, opened: str) -> None:
        parent = scopes[-1]
        parent_public = parent.exposed and parent.access
        exposed = False
        access = False
        if kind == "namespace":
            exposed = parent_public and _namespace_public(opened)
            access = exposed
            if not exposed:
                kind = "private"
        elif kind == "link":
            exposed = parent_public
            access = exposed
        elif kind == "class":
            exposed = parent_public
            text = _strip_c_prefixes(_strip_template(opened.strip()))
            access = exposed and re.match(r"class\b", text) is None
        scopes.append(
            _CScope(
                kind,
                depth,
                typedef_tail,
                exposed,
                access,
                scope_name(kind, opened),
            )
        )

    while cursor < limit:
        char, line_index = stream[cursor]
        if top() in {"fn", "skip", "private"}:
            if char == "{":
                brace += 1
            elif char == "}":
                brace -= 1
                while len(scopes) > 1 and brace < scopes[-1].depth:
                    scopes.pop()
            elif char == "(":
                paren += 1
            elif char == ")":
                paren = max(0, paren - 1)
            cursor += 1
            continue
        if top() == "enum" and paren == 0:
            if char.isspace():
                cursor += 1
                continue
            if char == "(":
                paren += 1
                cursor += 1
                continue
            if char == "{":
                brace += 1
                scopes.append(_CScope("skip", brace, False, False, False))
                cursor += 1
                continue
            if char == "}":
                closed = scopes.pop()
                brace -= 1
                if closed.typedef_tail:
                    tail: list[str] = []
                    tail_line = line_index
                    cursor += 1
                    while cursor < limit:
                        nxt, nxt_line = stream[cursor]
                        if nxt == ";" and paren == 0:
                            names = _typedef_entries("typedef " + "".join(tail))
                            for name, entry_kind in names:
                                add(name, tail_line, type_name=True, kind=entry_kind)
                            cursor += 1
                            break
                        if not nxt.isspace() and tail_line == line_index:
                            tail_line = nxt_line
                        tail.append(nxt)
                        cursor += 1
                else:
                    cursor += 1
                continue
            if char.isalpha() or char == "_":
                name, cursor = _read_ident(stream, cursor)
                add(name, line_index)
                nested = 0
                while cursor < limit:
                    nxt, _nxt_line = stream[cursor]
                    if nxt == "(":
                        paren += 1
                    elif nxt == ")":
                        paren = max(0, paren - 1)
                    elif nxt == "{" and paren == 0:
                        nested += 1
                        brace += 1
                    elif nxt == "}" and paren == 0:
                        if nested:
                            nested -= 1
                            brace -= 1
                        else:
                            break
                    elif nxt == "," and paren == 0 and nested == 0:
                        cursor += 1
                        break
                    cursor += 1
                continue
            cursor += 1
            continue
        if not declaration_scope():
            cursor += 1
            continue
        if paren == 0 and char == "}":
            if len(scopes) > 1:
                closed = scopes.pop()
                brace -= 1
                stmt = []
                if closed.typedef_tail:
                    tail: list[str] = []
                    tail_line = line_index
                    cursor += 1
                    while cursor < limit:
                        nxt, nxt_line = stream[cursor]
                        if nxt == ";" and paren == 0:
                            for name, entry_kind in _typedef_entries(
                                "typedef " + "".join(tail)
                            ):
                                add(name, tail_line, type_name=True, kind=entry_kind)
                            cursor += 1
                            break
                        if nxt.strip() and tail_line == line_index:
                            tail_line = nxt_line
                        tail.append(nxt)
                        cursor += 1
                    continue
            cursor += 1
            continue
        if top() == "class" and paren == 0 and char == ":":
            label = "".join(stmt).strip()
            if label in {"public", "private", "protected"}:
                scopes[-1].access = scopes[-1].exposed and label == "public"
                stmt = []
                cursor += 1
                continue
        if char == "(":
            paren += 1
            stmt.append(char)
            cursor += 1
            continue
        if char == ")":
            paren = max(0, paren - 1)
            stmt.append(char)
            cursor += 1
            continue
        if paren == 0 and char in ";{":
            text = "".join(stmt).strip()
            if char == "{" and not text:
                brace += 1
                push("skip", brace, False, "")
            elif text:
                if char == ";" and text.lstrip().startswith("using"):
                    for uname, ukind in _using_bindings(text):
                        add_cxx(uname, stmt_line, ukind)
                elif text.lstrip().startswith("typedef") and char == ";":
                    for name, entry_kind in _typedef_entries(text):
                        add(name, stmt_line, type_name=True, kind=entry_kind)
                elif char == "{":
                    kind = _open_kind(text)
                    tag = _c_tag(text)
                    if tag:
                        add(tag, stmt_line, type_name=True)
                        if kind in {"enum", "class"}:
                            add_cxx(tag, stmt_line, _type_kind(text))
                    if kind == "fn" and not _assigned_before_call(text):
                        name = _declarator_name(text)
                        if name:
                            add(name, stmt_line)
                            add_cxx(name, stmt_line, "function")
                    brace += 1
                    typedef_tail = text.lstrip().startswith("typedef") and kind in {
                        "enum",
                        "class",
                    }
                    push(kind, brace, typedef_tail, text)
                elif (
                    not _assigned_before_call(text)
                    and "typedef" not in text.split()[:1]
                    and "(" in text
                ):
                    name = _declarator_name(text)
                    if name:
                        add(name, stmt_line)
                        add_cxx(name, stmt_line, "function")
            stmt = []
            cursor += 1
            continue
        if not stmt and char.isspace():
            cursor += 1
            continue
        if not stmt:
            stmt_line = line_index
        stmt.append(char)
        cursor += 1
    return found


def _read_ident(stream: list[tuple[str, int]], cursor: int) -> tuple[str, int]:
    chars: list[str] = []
    while cursor < len(stream) and (
        stream[cursor][0].isalnum() or stream[cursor][0] == "_"
    ):
        chars.append(stream[cursor][0])
        cursor += 1
    return "".join(chars), cursor


def header_items(root: Path | None = None) -> list[Item]:
    base = root or ROOT
    items: list[Item] = []
    include = base / "include"
    if not include.is_dir():
        raise SystemExit(f"{include.relative_to(base)} is missing")
    for path in sorted(include.rglob("*")):
        if path.suffix not in {".h", ".hpp"}:
            continue
        rel = path.relative_to(base).as_posix()
        items.append(Item("C and C++", "header", rel, rel, 1, f"Public header {rel}"))
        text = path.read_text(encoding="utf-8")
        declared = header_declared_names(text, cxx=path.suffix == ".hpp")
        for line, name, declaration, kind, scope in declared:
            items.append(Item("C and C++", kind, name, rel, line, declaration, scope))
    return items


def _constant_strings(node: ast.AST) -> list[str] | None:
    if not isinstance(node, ast.List):
        return None
    names: list[str] = []
    for elt in node.elts:
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
            names.append(elt.value)
        else:
            return None
    return names


def _string_lists(tree: ast.AST) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for node in tree.body:
        value: ast.AST | None
        targets: list[ast.AST]
        if isinstance(node, ast.Assign):
            value = node.value
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            value = node.value
            targets = [node.target]
        else:
            continue
        names = _constant_strings(value)
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            if names is not None:
                found[target.id] = names
            elif isinstance(value, ast.Name) and value.id in found:
                found[target.id] = found[value.id]
    return found


def _string_line(tree: ast.AST, name: str) -> int | None:
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and node.value == name
            and hasattr(node, "lineno")
        ):
            return int(node.lineno)
    return None


def _private_module(stem: str) -> bool:
    return stem.startswith("_") and not stem.startswith("__")


def _export_allows(name: str, exported: set[str] | None) -> bool:
    if exported is not None:
        return name in exported
    return not name.startswith("_") or name == "__version__"


def _import_is_private(node: ast.ImportFrom) -> bool:
    parts = [part for part in (node.module or "").split(".") if part]
    return any(_private_module(part) for part in parts)


def _append_class_members(items: list[Item], node: ast.ClassDef, path: Path, rel: str) -> None:
    for member in node.body:
        if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if member.name.startswith("_") and member.name not in {
                "__init__",
                "__len__",
            }:
                continue
            items.append(
                Item(
                    "Python",
                    "method",
                    f"{node.name}.{member.name}",
                    rel,
                    member.lineno,
                    signature(member),
                    ("rgmin", node.name),
                )
            )
        elif isinstance(member, (ast.Assign, ast.AnnAssign)):
            for assigned in assigned_names(member):
                if not assigned.startswith("_"):
                    items.append(
                        Item(
                            "Python",
                            "class attribute",
                            f"{node.name}.{assigned}",
                            rel,
                            member.lineno,
                            source_segment(path, member),
                            ("rgmin", node.name),
                        )
                    )


def _append_reexported_bindings(
    items: list[Item], package: Path, base: Path, names: set[str]
) -> None:
    """Public members of binding classes re-exported from a private module."""
    if not names:
        return
    stubs = {
        path.stem
        for path in package.glob("*.pyi")
        if _private_module(path.stem)
    }
    emitted: set[str] = set()
    for path in sorted(package.glob("*")):
        if path.suffix not in {".py", ".pyi"} or not _private_module(path.stem):
            continue
        if path.suffix == ".py" and path.stem in stubs:
            continue
        rel = path.relative_to(base).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        except SyntaxError as error:
            raise SystemExit(f"{rel}: cannot inventory Python API: {error}") from error
        for node in tree.body:
            if not isinstance(node, ast.ClassDef) or node.name not in names:
                continue
            if node.name in emitted:
                continue
            emitted.add(node.name)
            items.append(
                Item(
                    "Python",
                    "class",
                    node.name,
                    rel,
                    node.lineno,
                    f"class {node.name}",
                )
            )
            _append_class_members(items, node, path, rel)


def python_items(root: Path | None = None) -> list[Item]:
    base = root or ROOT
    package = base / "python" / "rgmin"
    items: list[Item] = []
    reexported: set[str] = set()
    if not package.is_dir():
        raise SystemExit(f"{package.relative_to(base)} is missing")
    for path in sorted(package.glob("*")):
        if path.suffix not in {".py", ".pyi"} or _private_module(path.stem):
            continue
        rel = path.relative_to(base).as_posix()
        module = path.stem
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        except SyntaxError as error:
            raise SystemExit(f"{rel}: cannot inventory Python API: {error}") from error
        lists = _string_lists(tree)
        exported = set(lists["__all__"]) if "__all__" in lists else None
        items.append(
            Item(
                "Python", "module", f"rgmin.{module}", rel, 1, f"module rgmin.{module}"
            )
        )
        defined: set[str] = set()

        def emit(
            kind: str,
            name: str,
            line: int,
            declaration: str,
            exported: set[str] | None = exported,
            defined: set[str] = defined,
            rel: str = rel,
        ) -> None:
            if not _export_allows(name.split(".", 1)[0], exported):
                return
            defined.add(name.split(".", 1)[0])
            items.append(Item("Python", kind, name, rel, line, declaration))

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                emit("function", node.name, node.lineno, signature(node))
            elif isinstance(node, ast.ClassDef):
                if not _export_allows(node.name, exported):
                    continue
                defined.add(node.name)
                items.append(
                    Item(
                        "Python",
                        "class",
                        node.name,
                        rel,
                        node.lineno,
                        f"class {node.name}",
                    )
                )
                _append_class_members(items, node, path, rel)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                for assigned in assigned_names(node):
                    emit(
                        "module attribute",
                        assigned,
                        node.lineno,
                        source_segment(path, node),
                    )
            elif isinstance(node, ast.ImportFrom):
                private_binding = _import_is_private(node)
                for alias in node.names:
                    name = alias.asname or alias.name
                    if name == "*":
                        continue
                    if private_binding and _export_allows(name, exported):
                        reexported.add(name)
                    emit("reexport", name, node.lineno, source_segment(path, node))
        if exported is not None:
            for name in exported:
                if name in defined:
                    continue
                reexported.add(name)
                line = _string_line(tree, name) or 1
                items.append(
                    Item("Python", "reexport", name, rel, line, f"public name {name}")
                )
    _append_reexported_bindings(items, package, base, reexported)
    return items


def assigned_names(node: ast.Assign | ast.AnnAssign) -> list[str]:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return [target.id for target in targets if isinstance(target, ast.Name)]


def source_segment(path: Path, node: ast.AST) -> str:
    text = ast.get_source_segment(path.read_text(encoding="utf-8"), node) or ""
    return " ".join(text.split())


def signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = [arg.arg for arg in node.args.posonlyargs + node.args.args]
    if node.args.vararg:
        args.append(f"*{node.args.vararg.arg}")
    args.extend(arg.arg for arg in node.args.kwonlyargs)
    if node.args.kwarg:
        args.append(f"**{node.args.kwarg.arg}")
    return f"{node.name}({', '.join(args)})"


def org_documents() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for path in sorted(ORG.rglob("*.org")):
        if path == INVENTORY:
            continue
        text = path.read_text(encoding="utf-8")
        rel = relative(path)
        for token in set(MARKUP.findall(text)):
            found.setdefault(token, []).append(rel)
    return found


def _token_parts(token: str) -> tuple[str, ...]:
    if "::" in token:
        return tuple(part for part in token.split("::") if part)
    if "." in token and "/" not in token:
        return tuple(part for part in token.split(".") if part)
    return (token,)


def _bare_name(item: Item) -> str:
    if item.kind == "header" or "/" in item.name:
        return item.name
    return _token_parts(item.name)[-1]


def _qualified_token(token: str) -> bool:
    return "::" in token or ("." in token and "/" not in token)


def _prefixes(item: Item) -> tuple[tuple[str, ...], ...]:
    """Type and module paths that may stand in front of the bare name."""
    prefixes: list[tuple[str, ...]] = []
    scope = item.scope
    bare = _bare_name(item)
    if scope:
        prefixes.append(scope)
        if item.name == bare and scope[-1][:1].isupper():
            prefixes.append((scope[-1],))
            if scope[0] == "rgmin" and len(scope) > 1:
                prefixes.append(("rgmin", scope[-1]))
            module = scope[:-1]
            if module:
                prefixes.append(module)
        elif item.name == bare:
            prefixes.append((scope[-1],))
    parts = _token_parts(item.name)
    if len(parts) > 1:
        prefixes.append(parts[:-1])
    unique: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for prefix in prefixes:
        if prefix and prefix not in seen:
            seen.add(prefix)
            unique.append(prefix)
    return tuple(unique)


def _prefix_matches(item: Item, prefix: tuple[str, ...]) -> bool:
    if not prefix:
        return False
    for candidate in _prefixes(item):
        if len(prefix) <= len(candidate) and candidate[-len(prefix) :] == prefix:
            return True
    return False


def _slug(text: str) -> str:
    parts = re.findall(r"[A-Za-z0-9]+", text)
    return "-".join(part.lower() for part in parts)


def _reference_pages(item: Item) -> set[str]:
    """Reference pages that name this item's type, module, or source file."""
    slugs: set[str] = set()
    scope = item.scope
    if scope:
        slugs.add(_slug(scope[-1]))
        body = scope[1:] if scope[0] == "rgmin" else scope
        if body and item.name == _bare_name(item) and body[-1][:1].isupper():
            body = body[:-1]
        if body:
            slugs.add(_slug(body[-1]))
            if len(body) > 1:
                slugs.add(_slug("-".join(body)))
    stem = Path(item.source).stem
    if stem not in {"mod", "lib", "__init__"} and not stem.startswith("_"):
        slugs.add(stem.replace("_", "-"))
    return {f"docs/orgmode/reference/{slug}.org" for slug in slugs if slug}


def _item_key(item: Item) -> tuple[str, str, str, str, int]:
    return (item.surface, item.kind, item.name, item.source, item.line)


def _own_reference_pages(item: Item, peers: list[Item]) -> set[str]:
    """Reference pages that belong to this item and not to another peer."""
    mine = _reference_pages(item)
    shared: set[str] = set()
    key = _item_key(item)
    for peer in peers:
        if _item_key(peer) == key:
            continue
        shared |= _reference_pages(peer)
    return mine - shared


def _index_documents(
    documents: dict[str, list[str]],
) -> dict[str, list[tuple[str, list[str]]]]:
    indexed: dict[str, list[tuple[str, list[str]]]] = {}
    for token, locations in documents.items():
        indexed.setdefault(_token_parts(token)[-1], []).append((token, locations))
    return indexed


def documented_at(
    item: Item,
    documents: dict[str, list[str]],
    peers: list[Item] | None = None,
    index: dict[str, list[tuple[str, list[str]]]] | None = None,
) -> list[str]:
    """Org pages that document this item.

    A bare token is enough when the name occurs once on that surface.
    A repeated name needs a type or module path that selects this item,
    or the item's own reference page.
    """
    group = peers if peers is not None else [item]
    bare = _bare_name(item)
    same = [
        peer
        for peer in group
        if peer.surface == item.surface and _bare_name(peer) == bare
    ]
    ambiguous = len(same) > 1
    own = _own_reference_pages(item, same) if ambiguous else set()
    if index is None:
        index = _index_documents(documents)
    pages: list[str] = []
    for token, locations in index.get(bare, []):
        if not _qualified_token(token):
            if not ambiguous:
                pages.extend(locations)
            else:
                pages.extend(page for page in locations if page in own)
            continue
        matched = [
            peer
            for peer in same
            if peer.name == token or _prefix_matches(peer, _token_parts(token)[:-1])
        ]
        keys = {_item_key(peer) for peer in matched}
        if len(keys) == 1 and _item_key(item) in keys:
            pages.extend(locations)
    return sorted(set(pages))


def clean_cell(value: str) -> str:
    return value.replace("|", "¦").replace("\n", " ").strip()


def render_inventory(items: list[Item], documents: dict[str, list[str]]) -> str:
    index = _index_documents(documents)
    before = sum(
        bool(documented_at(item, documents, items, index)) for item in items
    )
    lines = [
        "#+TITLE: Public API inventory",
        "#+OPTIONS: toc:nil todo:nil num:nil author:nil title:nil",
        "#+STARTUP: content",
        "",
        "* Public API inventory",
        "",
        "This file is generated by ~scripts/generate_docs_inventory.py~.",
        "It inventories source declarations even when a Cargo feature hides an item.",
        "The declaration is the reference entry for an item that had no narrative",
        "entry elsewhere. Run the script to update this file. Do not edit it by hand.",
        "",
        "* Coverage",
        "",
        f"Before this inventory, {before} of {len(items)} source items had an exact",
        "marked-up name in an Org source. After this inventory, every item has a",
        "reference entry and a documentation location.",
        "",
    ]
    for surface in ("Rust", "C and C++", "Python"):
        lines.extend(
            [
                f"* {surface}",
                "",
                "| Item | Kind | Declaration | Source | Documentation |",
                "|---+---+---+---+---|",
            ]
        )
        for item in sorted(i for i in items if i.surface == surface):
            locations = documented_at(item, documents, items, index)
            docs = ", ".join(f"~{loc}~" for loc in locations[:3])
            if not docs:
                docs = "~docs/orgmode/reference/public-api.org~"
            source = f"~{item.source}:{item.line}~"
            lines.append(
                f"| ~{clean_cell(item.name)}~ | {clean_cell(item.kind)} | "
                f"~{clean_cell(item.declaration)}~ | {source} | {docs} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _title_text(text: str, stem: str) -> str:
    match = re.search(
        r"^#\+TITLE:\s*(.+)$",
        text,
        re.MULTILINE | re.IGNORECASE,
    )
    return match.group(1).strip() if match else stem.replace("-", " ").title()


def org_title(path: Path) -> str:
    return _title_text(path.read_text(encoding="utf-8"), path.stem)


def documentation_site() -> str:
    """Published docs site from ``[project.urls] Documentation``.

    The page links are built from this URL alone. A missing entry, or a
    value that is not a plain absolute https URL, is an error.
    """
    lines = (ROOT / "pyproject.toml").read_text(encoding="utf-8").splitlines()
    in_urls = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_urls = stripped == "[project.urls]"
            continue
        if not in_urls:
            continue
        key, sep, raw = stripped.partition("=")
        if not sep or key.strip() != "Documentation":
            continue
        quoted = raw.strip()
        if len(quoted) < 2 or quoted[0] not in "\"'" or quoted[-1] != quoted[0]:
            raise RuntimeError("Documentation URL must be a quoted https string")
        url = quoted[1:-1]
        if not url.startswith("https://") or any(
            ch.isspace() or ch in "{}<>" for ch in url
        ):
            raise RuntimeError("Documentation URL must be a plain absolute https URL")
        return url if url.endswith("/") else url + "/"
    raise RuntimeError("pyproject.toml [project.urls] has no Documentation URL")


def docs_page(site: str, page: str) -> str:
    """Absolute URL of one generated Sphinx page.

    ``site`` is the Documentation URL with a trailing slash. ``page`` is
    the path under that site without a suffix, such as ``howto/stopping``.
    """
    if not site.startswith("https://") or not site.endswith("/"):
        raise RuntimeError("documentation site must be an absolute https URL")
    parts = page.split("/")
    if (
        not page
        or page.startswith("/")
        or any(part in ("", ".", "..") for part in parts)
        or any(ch in page for ch in "\\{}?# ")
    ):
        raise RuntimeError(f"documentation page is not a relative path: {page}")
    return f"{site}{page}.html"


def _text_for(path: Path, overrides: dict[Path, str] | None) -> str:
    if overrides:
        for key, value in overrides.items():
            if key == path or key.resolve() == path.resolve():
                return value
    return path.read_text(encoding="utf-8")


def render_llms(overrides: dict[Path, str] | None = None) -> str:
    site = documentation_site()
    lines = [
        "# rgmin",
        "",
        "> Local first order, quasi Newton, trust region, and Riemannian optimization",
        "> in Rust with C, C++, and Python interfaces.",
        "",
        "The Org files are the canonical documentation sources. The generated",
        "reStructuredText tree is for Sphinx and must not be edited directly.",
        "",
    ]
    groups = [
        ("Tutorials", "tutorials"),
        ("How to guides", "howto"),
        ("Explanation", "explanation"),
        ("Reference", "reference"),
        ("Development", "contributing"),
    ]
    for heading, directory in groups:
        lines.extend([f"## {heading}", ""])
        paths = list(sorted((ORG / directory).glob("*.org")))
        if (
            overrides
            and directory == "reference"
            and INVENTORY in overrides
            and all(path.resolve() != INVENTORY.resolve() for path in paths)
        ):
            paths.append(INVENTORY)
            paths.sort()
        for path in paths:
            rel = path.relative_to(ORG).with_suffix("").as_posix()
            title = _title_text(_text_for(path, overrides), path.stem)
            lines.append(f"- [{title}]({docs_page(site, rel)})")
        lines.append("")
    lines.extend(
        [
            "## Optional",
            "",
            f"- [Changelog]({docs_page(site, 'changelog')})",
            "- [Repository](https://github.com/OmniPotentRPC/rgmin)",
            "",
        ]
    )
    return "\n".join(lines)


def _org_heading(match: re.Match[str]) -> str:
    return "#" * len(match.group(1)) + " "


def _org_code(match: re.Match[str]) -> str:
    return f"`{match.group(1) or match.group(2)}`"


def _org_link(match: re.Match[str]) -> str:
    return f"[{match.group(2)}]({match.group(1)})"


def org_to_markdown(text: str) -> str:
    flags = re.MULTILINE | re.IGNORECASE
    text = re.sub(r"^#\+(?:TITLE|OPTIONS|STARTUP|AUTHOR):.*\n?", "", text, flags=flags)
    text = re.sub(r"^(\*+)\s+", _org_heading, text, flags=re.MULTILINE)
    text = re.sub(r"~([^~\n]+)~|=([^=\n]+)=", _org_code, text)
    text = re.sub(r"\[\[([^\]]+)\]\[([^\]]+)\]\]", _org_link, text)
    text = re.sub(r"^#\+begin_src\s+(\S+)", r"```\1", text, flags=flags)
    text = re.sub(r"^#\+end_src", "```", text, flags=flags)
    text = re.sub(
        r"^#\+begin_export rst.*?^#\+end_export\s*",
        "",
        text,
        flags=flags | re.DOTALL,
    )
    text = re.sub(
        r"^#\+INCLUDE:\s+\"([^\"]+)\"(?:\s+src\s+(\S+))?.*$",
        r"Source: `\1`",
        text,
        flags=flags,
    )
    return text.strip()


def render_llms_full(overrides: dict[Path, str] | None = None) -> str:
    parts = [
        "# rgmin full documentation",
        "",
        "> Generated from the canonical Org sources by `scripts/generate_docs_inventory.py`.",
    ]
    paths = list(sorted(ORG.rglob("*.org")))
    if overrides and INVENTORY in overrides and all(
        path.resolve() != INVENTORY.resolve() for path in paths
    ):
        paths.append(INVENTORY)
        paths.sort()
    for path in paths:
        parts.extend(
            [
                "",
                f"---\n\nSource: `{relative(path)}`",
                "",
                org_to_markdown(_text_for(path, overrides)),
            ]
        )
    return "\n".join(parts).rstrip() + "\n"


def generated() -> dict[Path, str]:
    items = sorted(set(rust_items() + header_items() + python_items()))
    documents = org_documents()
    inventory = render_inventory(items, documents)
    overrides = {INVENTORY: inventory}
    full = render_llms_full(overrides)
    short = render_llms(overrides)
    return {INVENTORY: inventory, LLMS: short, LLMS_FULL: full}


def _atomic_write(path: Path, text: str) -> None:
    """Replace ``path`` by writing a temporary file in its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _inventory_bytes(path: Path) -> bytes | None:
    """Read the inventory. A non-UTF-8 file is a one-line error."""
    data = path.read_bytes()
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        print(f"{relative(path)} is not valid UTF-8", file=sys.stderr)
        return None
    return data


def _diff_summary(path: Path, current: bytes, expected: str) -> str:
    rel = relative(path)
    expected_bytes = expected.encode("utf-8")
    try:
        current_text = current.decode("utf-8")
    except UnicodeDecodeError:
        return (
            f"{rel} differs ({len(current)} bytes on disk, "
            f"{len(expected_bytes)} bytes generated)"
        )
    diff = list(
        difflib.unified_diff(
            current_text.splitlines(keepends=True),
            expected.splitlines(keepends=True),
            fromfile=rel,
            tofile=f"{rel} generated",
            n=1,
        )
    )
    changed = sum(
        1
        for line in diff
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    )
    preview = diff[:24]
    lines = [f"{rel} differs ({changed} changed lines)"]
    lines.extend(line.rstrip("\n") for line in preview)
    if len(diff) > len(preview):
        lines.append(f"... {len(diff) - len(preview)} more diff lines")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="replace generated files")
    parser.add_argument(
        "--check", action="store_true", help="fail when generated files differ"
    )
    args = parser.parse_args()
    if args.write == args.check:
        parser.error("choose exactly one of --write or --check")
    outputs = generated()
    if args.write:
        for path, text in outputs.items():
            encoded = text.encode("utf-8")
            if path.exists() and path.read_bytes() == encoded:
                continue
            _atomic_write(path, text)
        print(f"wrote {len(outputs)} generated documentation files")
        return 0
    stale = False
    for path, text in outputs.items():
        rel = relative(path)
        if not path.exists():
            print(f"{rel} is stale; run scripts/generate_docs_inventory.py --write")
            stale = True
            continue
        if path == INVENTORY:
            current = _inventory_bytes(path)
            if current is None:
                return 1
        else:
            current = path.read_bytes()
        if current != text.encode("utf-8"):
            print(f"{rel} is stale; run scripts/generate_docs_inventory.py --write")
            print(_diff_summary(path, current, text))
            stale = True
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
