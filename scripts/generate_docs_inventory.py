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
    r'^\s*pub\s+(?!\()'
    r'(?P<qualifiers>(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?'
    r'(?:extern\s+"[^"]+"\s+)?)'
    r"fn\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
)
RUST_ITEM = re.compile(
    r"^\s*pub\s+(?!\()(?P<kind>mod|struct|enum|union|trait|type|const|static)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
)
RUST_FIELD = re.compile(
    r"^\s*pub\s+(?!\()(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*:"
)
RUST_USE = re.compile(r"^\s*pub\s+use\s+(.+?);\s*$")
TRAIT_MEMBER = re.compile(
    r"^\s*(?:unsafe\s+)?(?:async\s+)?(?P<kind>fn|type|const)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
)
ENUM_VARIANT = re.compile(r"^\s*(?P<name>[A-Z][A-Za-z0-9_]*)\s*(?:[,({=]|$)")
WITH_SETTER = re.compile(r"^\s*(?P<name>with_[A-Za-z0-9_]+)\s*=>")
HEADER_TOKEN = re.compile(r"\b(?:rgmin|xts)_[A-Za-z0-9_]+\b|\b(?:RGMIN|XTS)_[A-Z0-9_]+\b")
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


def rust_items() -> list[Item]:
    items: list[Item] = []
    for path in sorted((ROOT / "src").rglob("*.rs")):
        rel = relative(path)
        lines = path.read_text().splitlines()
        depth = 0
        contexts: list[tuple[str, str, int]] = []
        for number, line in enumerate(lines, 1):
            declaration = " ".join(line.strip().split())
            while contexts and depth < contexts[-1][2]:
                contexts.pop()
            context = contexts[-1] if contexts and depth == contexts[-1][2] else None
            if context and context[0] == "trait":
                member = TRAIT_MEMBER.match(line)
                if member:
                    name = f"{context[1]}::{member['name']}"
                    items.append(
                        Item("Rust", f"trait {member['kind']}", name, rel, number, declaration)
                    )
            elif context and context[0] == "enum":
                variant = ENUM_VARIANT.match(line)
                if variant:
                    name = f"{context[1]}::{variant['name']}"
                    items.append(Item("Rust", "enum variant", name, rel, number, declaration))
            setter = WITH_SETTER.match(line)
            if setter:
                items.append(
                    Item("Rust", "generated method", setter["name"], rel, number, declaration)
                )
            parsed = rust_declaration(line)
            if parsed:
                kind, name = parsed
                items.append(Item("Rust", kind, name, rel, number, declaration))
                if kind in {"trait", "enum"} and "{" in line:
                    contexts.append((kind, name, depth + 1))
            field = RUST_FIELD.match(line)
            if field:
                items.append(Item("Rust", "public field", field["name"], rel, number, declaration))
            use = RUST_USE.match(line)
            if use:
                body = use.group(1)
                if "{" in body:
                    body = body.split("{", 1)[1].rsplit("}", 1)[0]
                    names = [part.strip().split(" as ")[-1] for part in body.split(",")]
                else:
                    names = [body.rsplit("::", 1)[-1].split(" as ")[-1]]
                for name in names:
                    name = name.strip()
                    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                        items.append(Item("Rust", "reexport", name, rel, number, declaration))
            depth += line.count("{") - line.count("}")
    return items


def header_items() -> list[Item]:
    items: list[Item] = []
    for path in sorted((ROOT / "include").rglob("*")):
        if path.suffix not in {".h", ".hpp"}:
            continue
        rel = relative(path)
        items.append(Item("C and C++", "header", rel, rel, 1, f"Public header {rel}"))
        seen: set[str] = set()
        text = path.read_text()
        code = re.sub(
            r"/\*.*?\*/",
            lambda match: re.sub(r"[^\n]", " ", match.group()),
            text,
            flags=re.S,
        )
        code = re.sub(r"//[^\n]*", "", code)
        for number, (line, code_line) in enumerate(
            zip(text.splitlines(), code.splitlines(), strict=True), 1
        ):
            for name in HEADER_TOKEN.findall(code_line):
                if name in seen:
                    continue
                seen.add(name)
                kind = "constant" if name.isupper() else "symbol"
                items.append(
                    Item("C and C++", kind, name, rel, number, " ".join(line.strip().split()))
                )
    return items


def python_items() -> list[Item]:
    items: list[Item] = []
    package = ROOT / "python" / "rgmin"
    for path in sorted(package.glob("*")):
        if path.suffix not in {".py", ".pyi"}:
            continue
        rel = relative(path)
        module = path.stem
        try:
            tree = ast.parse(path.read_text(), filename=rel)
        except SyntaxError as error:
            raise SystemExit(f"{rel}: cannot inventory Python API: {error}") from error
        items.append(Item("Python", "module", f"rgmin.{module}", rel, 1, f"module rgmin.{module}"))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
                items.append(Item("Python", "function", node.name, rel, node.lineno, signature(node)))
            elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                items.append(Item("Python", "class", node.name, rel, node.lineno, f"class {node.name}"))
                for member in node.body:
                    if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if member.name.startswith("_") and member.name not in {"__init__", "__len__"}:
                            continue
                        name = f"{node.name}.{member.name}"
                        items.append(
                            Item("Python", "method", name, rel, member.lineno, signature(member))
                        )
                    elif isinstance(member, (ast.Assign, ast.AnnAssign)):
                        names = assigned_names(member)
                        for name in names:
                            if not name.startswith("_"):
                                items.append(
                                    Item(
                                        "Python",
                                        "class attribute",
                                        f"{node.name}.{name}",
                                        rel,
                                        member.lineno,
                                        source_segment(path, member),
                                    )
                                )
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                for name in assigned_names(node):
                    if not name.startswith("_") or name == "__version__":
                        items.append(
                            Item(
                                "Python",
                                "module attribute",
                                name,
                                rel,
                                node.lineno,
                                source_segment(path, node),
                            )
                        )
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    name = alias.asname or alias.name
                    if not name.startswith("_"):
                        items.append(
                            Item("Python", "reexport", name, rel, node.lineno, source_segment(path, node))
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
        "entry elsewhere. Regenerate it instead of editing it.",
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
    match = re.search(r"^#\+TITLE:\s*(.+)$", path.read_text(), re.M | re.I)
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
        if not url.startswith("https://") or any(ch.isspace() or ch in "{}<>" for ch in url):
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


def org_to_markdown(text: str) -> str:
    text = re.sub(r"^#\+(?:TITLE|OPTIONS|STARTUP|AUTHOR):.*\n?", "", text, flags=re.M | re.I)
    text = re.sub(r"^(\*+)\s+", lambda m: "#" * len(m.group(1)) + " ", text, flags=re.M)
    text = re.sub(r"~([^~\n]+)~|=([^=\n]+)=", lambda m: f"`{m.group(1) or m.group(2)}`", text)
    text = re.sub(
        r"\[\[([^\]]+)\]\[([^\]]+)\]\]",
        lambda m: f"[{m.group(2)}]({m.group(1)})",
        text,
    )
    text = re.sub(r"^#\+begin_src\s+(\S+)", r"```\1", text, flags=re.M | re.I)
    text = re.sub(r"^#\+end_src", "```", text, flags=re.M | re.I)
    text = re.sub(r"^#\+begin_export rst.*?^#\+end_export\s*", "", text, flags=re.M | re.I | re.S)
    text = re.sub(r"^#\+INCLUDE:\s+\"([^\"]+)\"(?:\s+src\s+(\S+))?.*$", r"Source: `\1`", text, flags=re.M | re.I)
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
    parser.add_argument("--check", action="store_true", help="fail when generated files differ")
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
