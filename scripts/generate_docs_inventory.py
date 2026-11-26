#!/usr/bin/env python3
"""Generate the public API inventory and the two llms.txt files."""

from __future__ import annotations

import argparse
import ast
import dataclasses
import re
import sys
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
MARKUP = re.compile(r"[~=]([A-Za-z_][A-Za-z0-9_:.-]*)[~=]")


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


def _collapse(text: str) -> str:
    return " ".join(text.strip().split())


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
                            cursor += 2
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
                    cursor += 2
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
                    cursor += 2
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


def rust_items(root: Path | None = None) -> list[Item]:
    """Public Rust items reachable from the crate root.

    A public module chain or a ``pub use`` re-export makes an item
    reachable. Feature gates stay in the set. ``cargo doc`` JSON is
    nightly-only here, so the module walk is the check.
    """
    base = root or ROOT
    lib = base / "src" / "lib.rs"
    if not lib.is_file():
        return []
    records: list[_RustRec] = []
    reexports: list[_Reexport] = []
    globs: list[_GlobUse] = []
    children: dict[tuple[str, ...], dict[str, tuple[tuple[str, ...], bool]]] = {}
    defined: dict[tuple[str, ...], dict[str, list[int]]] = {}
    types: dict[tuple[tuple[str, ...], str], int] = {}
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
                        _Scope("impl", depth, scope.module, _impl_owner(header_blank))
                    )
            else:
                identity = _item_identity(header_blank)
                public = _unrestricted_pub(header_blank)
                if identity and scope.kind == "module":
                    kind, name = identity
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
                elif scope.kind == "impl" and public:
                    parsed = rust_declaration(_collapse(header_raw))
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
                                _collapse(header_raw),
                                scope.module,
                                "method",
                                scope.name,
                                True,
                            )
                        )
                        if body_open and kind in {"function", "C export"}:
                            stack.append(_Scope("fn", depth, scope.module, name))
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
    for record in records:
        if record.role not in {"method", "field", "setter", "variant", "trait_member"}:
            continue
        owner_id = types.get((record.module, record.owner))
        if owner_id is None:
            matches = [
                item for (mod, name), item in types.items() if name == record.owner
            ]
            owner_id = matches[0] if len(matches) == 1 else None
        if owner_id is not None and owner_id in reachable:
            mark_item(records.index(record))
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


def _header_name(name: str) -> bool:
    return _HEADER_NAME.match(name) is not None


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
    match = re.search(
        r"\b(?:enum(?:\s+class|\s+struct)?|struct|union|class)\s+([A-Za-z_][A-Za-z0-9_]*)",
        stmt,
    )
    if match:
        return match.group(1)
    return None


def _open_kind(stmt: str) -> str:
    text = _strip_template(stmt.strip())
    text = _strip_c_prefixes(text)
    if re.match(r"(?:extern\s*$|namespace\b)", text):
        return "link"
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


def header_declared_names(text: str) -> list[tuple[int, str, str]]:
    """Line, name, and source line of each prototype, typedef, enumerator, or macro."""
    blank = _blank_c(text)
    original = text.splitlines()
    masked = blank.splitlines()
    if len(masked) < len(original):
        masked.extend([""] * (len(original) - len(masked)))
    stream: list[tuple[str, int]] = []
    found: list[tuple[int, str, str]] = []
    seen: set[str] = set()

    def add(name: str, line_index: int) -> None:
        if name in seen or not _header_name(name) or line_index >= len(original):
            return
        seen.add(name)
        found.append((line_index + 1, name, _collapse(original[line_index])))

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
    scopes: list[tuple[str, int, bool]] = [("file", 0, False)]
    stmt: list[str] = []
    stmt_line = 0
    cursor = 0
    limit = len(stream)

    def top() -> str:
        return scopes[-1][0]

    def declaration_scope() -> bool:
        return top() in {"file", "link", "class"}

    while cursor < limit:
        char, line_index = stream[cursor]
        if top() in {"fn", "skip"}:
            if char == "{":
                brace += 1
            elif char == "}":
                brace -= 1
                while len(scopes) > 1 and brace < scopes[-1][1]:
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
                scopes.append(("skip", brace, False))
                cursor += 1
                continue
            if char == "}":
                closed = scopes.pop()
                brace -= 1
                if closed[2]:
                    tail: list[str] = []
                    tail_line = line_index
                    cursor += 1
                    while cursor < limit:
                        nxt, nxt_line = stream[cursor]
                        if nxt == ";" and paren == 0:
                            names = _typedef_names("typedef " + "".join(tail))
                            for name in names:
                                add(name, tail_line)
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
                if closed[2]:
                    tail: list[str] = []
                    tail_line = line_index
                    cursor += 1
                    while cursor < limit:
                        nxt, nxt_line = stream[cursor]
                        if nxt == ";" and paren == 0:
                            for name in _typedef_names("typedef " + "".join(tail)):
                                add(name, tail_line)
                            cursor += 1
                            break
                        if nxt.strip() and tail_line == line_index:
                            tail_line = nxt_line
                        tail.append(nxt)
                        cursor += 1
                    continue
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
            if text:
                if text.lstrip().startswith("typedef") and char == ";":
                    for name in _typedef_names(text):
                        add(name, stmt_line)
                elif char == "{":
                    kind = _open_kind(text)
                    tag = _c_tag(text)
                    if tag:
                        add(tag, stmt_line)
                    if kind == "fn" and not _assigned_before_call(text):
                        name = _declarator_name(text)
                        if name:
                            add(name, stmt_line)
                    brace += 1
                    typedef_tail = text.lstrip().startswith("typedef") and kind in {
                        "enum",
                        "class",
                    }
                    scopes.append((kind, brace, typedef_tail))
                elif (
                    not _assigned_before_call(text)
                    and "typedef" not in text.split()[:1]
                    and "(" in text
                ):
                    name = _declarator_name(text)
                    if name:
                        add(name, stmt_line)
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
        return items
    for path in sorted(include.rglob("*")):
        if path.suffix not in {".h", ".hpp"}:
            continue
        rel = path.relative_to(base).as_posix()
        items.append(Item("C and C++", "header", rel, rel, 1, f"Public header {rel}"))
        text = path.read_text(encoding="utf-8")
        for line, name, declaration in header_declared_names(text):
            kind = "constant" if name.isupper() else "symbol"
            items.append(Item("C and C++", kind, name, rel, line, declaration))
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


def python_items(root: Path | None = None) -> list[Item]:
    base = root or ROOT
    package = base / "python" / "rgmin"
    items: list[Item] = []
    if not package.is_dir():
        return items
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
                                    )
                                )
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                for assigned in assigned_names(node):
                    emit(
                        "module attribute",
                        assigned,
                        node.lineno,
                        source_segment(path, node),
                    )
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    name = alias.asname or alias.name
                    if name == "*":
                        continue
                    emit("reexport", name, node.lineno, source_segment(path, node))
        if exported is not None:
            for name in exported:
                if name in defined:
                    continue
                line = _string_line(tree, name) or 1
                items.append(
                    Item("Python", "reexport", name, rel, line, f"public name {name}")
                )
    return items


def assigned_names(node: ast.Assign | ast.AnnAssign) -> list[str]:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return [target.id for target in targets if isinstance(target, ast.Name)]


def source_segment(path: Path, node: ast.AST) -> str:
    text = ast.get_source_segment(path.read_text(), node) or ""
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
        text = path.read_text()
        rel = relative(path)
        for token in set(MARKUP.findall(text)):
            found.setdefault(token, []).append(rel)
    return found


def lookup_names(item: Item) -> list[str]:
    names = [item.name]
    if "." in item.name:
        names.append(item.name.rsplit(".", 1)[-1])
    return names


def documented_at(item: Item, documents: dict[str, list[str]]) -> list[str]:
    locations: list[str] = []
    for name in lookup_names(item):
        locations.extend(documents.get(name, []))
    return sorted(set(locations))


def clean_cell(value: str) -> str:
    return value.replace("|", "¦").replace("\n", " ").strip()


def render_inventory(items: list[Item], documents: dict[str, list[str]]) -> str:
    before = sum(bool(documented_at(item, documents)) for item in items)
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
            locations = documented_at(item, documents)
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


def org_title(path: Path) -> str:
    match = re.search(
        r"^#\+TITLE:\s*(.+)$",
        path.read_text(),
        re.MULTILINE | re.IGNORECASE,
    )
    return match.group(1).strip() if match else path.stem.replace("-", " ").title()


def documentation_site() -> str:
    """Published docs site from ``[project.urls] Documentation``.

    The page links are built from this URL alone. A missing entry, or a
    value that is not a plain absolute https URL, is an error.
    """
    lines = (ROOT / "pyproject.toml").read_text().splitlines()
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


def render_llms() -> str:
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
        for path in sorted((ORG / directory).glob("*.org")):
            rel = path.relative_to(ORG).with_suffix("").as_posix()
            lines.append(f"- [{org_title(path)}]({docs_page(site, rel)})")
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


def render_llms_full() -> str:
    parts = [
        "# rgmin full documentation",
        "",
        "> Generated from the canonical Org sources by `scripts/generate_docs_inventory.py`.",
    ]
    for path in sorted(ORG.rglob("*.org")):
        parts.extend(
            [
                "",
                f"---\n\nSource: `{relative(path)}`",
                "",
                org_to_markdown(path.read_text()),
            ]
        )
    return "\n".join(parts).rstrip() + "\n"


def generated() -> dict[Path, str]:
    items = sorted(set(rust_items() + header_items() + python_items()))
    documents = org_documents()
    inventory = render_inventory(items, documents)
    previous = INVENTORY.read_text() if INVENTORY.exists() else None
    INVENTORY.parent.mkdir(parents=True, exist_ok=True)
    INVENTORY.write_text(inventory)
    try:
        full = render_llms_full()
        short = render_llms()
    finally:
        if previous is None:
            INVENTORY.unlink()
        else:
            INVENTORY.write_text(previous)
    return {INVENTORY: inventory, LLMS: short, LLMS_FULL: full}


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
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        print(f"wrote {len(outputs)} generated documentation files")
        return 0
    stale = []
    for path, text in outputs.items():
        if not path.exists() or path.read_text() != text:
            stale.append(relative(path))
    for name in stale:
        print(f"{name} is stale; run scripts/generate_docs_inventory.py --write")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
