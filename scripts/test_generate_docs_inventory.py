#!/usr/bin/env python3
"""Tests for the documentation inventory parser."""

from __future__ import annotations

import importlib.util
import re
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("generate_docs_inventory.py")
SPEC = importlib.util.spec_from_file_location("generate_docs_inventory", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
inventory = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = inventory
SPEC.loader.exec_module(inventory)


class RustDeclarationTests(unittest.TestCase):
    def test_function_qualifiers(self) -> None:
        declarations = {
            "pub fn plain() {}": ("function", "plain"),
            "pub const fn constant() {}": ("function", "constant"),
            "pub async fn asynchronous() {}": ("function", "asynchronous"),
            "pub unsafe fn unchecked() {}": ("function", "unchecked"),
            'pub extern "C" fn exported() {}': ("C export", "exported"),
            "pub const unsafe fn constant_unchecked() {}": (
                "function",
                "constant_unchecked",
            ),
            'pub const unsafe extern "C" fn constant_exported() {}': (
                "C export",
                "constant_exported",
            ),
            'pub async unsafe extern "C" fn asynchronous_exported() {}': (
                "C export",
                "asynchronous_exported",
            ),
        }
        for declaration, expected in declarations.items():
            with self.subTest(declaration=declaration):
                self.assertEqual(inventory.rust_declaration(declaration), expected)

    def test_const_item_is_not_a_function(self) -> None:
        self.assertEqual(
            inventory.rust_declaration("pub const LIMIT: usize = 4;"),
            ("const", "LIMIT"),
        )


class DocsUrlTests(unittest.TestCase):
    def test_site_is_the_published_documentation_url(self) -> None:
        site = inventory.documentation_site()
        self.assertTrue(site.startswith("https://"))
        self.assertTrue(site.endswith("/"))
        text = (inventory.ROOT / "pyproject.toml").read_text()
        self.assertIn(site.rstrip("/"), text)

    def test_page_url_joins_the_path_beside_the_site(self) -> None:
        site = inventory.documentation_site()
        url = inventory.docs_page(site, "howto/stopping")
        self.assertEqual(url, site + "howto/stopping.html")
        self.assertNotIn("{", url)
        self.assertNotIn("}", url)

    def test_page_url_rejects_a_path_that_would_leave_the_site(self) -> None:
        site = inventory.documentation_site()
        for page in ("", "/howto/stopping", "../stopping", "howto//stopping", "a b"):
            with self.subTest(page=page), self.assertRaises(RuntimeError):
                inventory.docs_page(site, page)
        with self.assertRaises(RuntimeError):
            inventory.docs_page("http://example.com/", "howto/stopping")

    def test_rendered_links_stay_on_the_published_site(self) -> None:
        site = inventory.documentation_site()
        text = inventory.render_llms()
        # Split the scheme so this source has no truncated address for the link check.
        urls = re.findall("https://" + r"[^)\s]+", text)
        self.assertGreater(len(urls), 1)
        for url in urls:
            if url == "https://github.com/OmniPotentRPC/rgmin":
                continue
            self.assertTrue(url.startswith(site), url)
            self.assertTrue(url.endswith(".html"), url)
            self.assertNotIn("{", url)
        self.assertNotIn("rgmin.rgoswami.me", text)
        self.assertNotIn("rgmin.rgoswami.me", SCRIPT.read_text())


class ReachableRustTests(unittest.TestCase):
    def test_private_module_items_stay_hidden_unless_reexported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src"
            source.mkdir()
            (source / "lib.rs").write_text(
                "mod hidden;\n"
                "pub mod open;\n"
                '#[cfg(feature = "capi")]\n'
                "pub mod gated;\n"
                "pub use hidden::{\n"
                "    Exported,\n"
                "};\n"
                "#[cfg(test)]\n"
                "mod tests {\n"
                "    pub fn not_user() {}\n"
                "}\n",
                encoding="utf-8",
            )
            (source / "hidden.rs").write_text(
                "pub fn secret() {}\n"
                "pub struct Exported {\n"
                "    pub field: i32,\n"
                "}\n"
                "impl Exported {\n"
                "    pub fn method(&self) {}\n"
                "}\n",
                encoding="utf-8",
            )
            (source / "open.rs").write_text(
                "pub fn open() {}\n"
                "pub(crate) fn crate_only() {}\n"
                '#[cfg(feature = "extra")]\n'
                "pub fn feature_kept() {}\n",
                encoding="utf-8",
            )
            (source / "gated.rs").write_text(
                "pub fn kept() {}\n",
                encoding="utf-8",
            )
            items = inventory.rust_items(root)
        names = {item.name for item in items}
        self.assertIn("Exported", names)
        self.assertIn("field", names)
        self.assertIn("method", names)
        self.assertIn("open", names)
        self.assertIn("kept", names)
        self.assertIn("feature_kept", names)
        self.assertNotIn("secret", names)
        self.assertNotIn("crate_only", names)
        self.assertNotIn("not_user", names)
        self.assertTrue(
            any(item.kind == "mod" and item.name == "open" for item in items)
        )
        self.assertTrue(
            any(item.kind == "mod" and item.name == "gated" for item in items)
        )
        self.assertFalse(any(item.name == "hidden" for item in items))

    def test_glob_reexport_expands_public_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src"
            nested = source / "outer"
            nested.mkdir(parents=True)
            (source / "lib.rs").write_text(
                "mod outer;\npub use outer::*;\npub use missing::*;\n",
                encoding="utf-8",
            )
            (source / "outer.rs").write_text(
                "mod inner;\npub use inner::*;\npub fn outer_fn() {}\n",
                encoding="utf-8",
            )
            (nested / "inner.rs").write_text(
                "pub fn inner_fn() {}\npub struct Inner {\n    pub value: i32,\n}\n",
                encoding="utf-8",
            )
            with self.assertRaises(SystemExit) as caught:
                inventory.rust_items(root)
            self.assertIn("pub use glob", str(caught.exception))
            (source / "lib.rs").write_text(
                "mod outer;\npub use outer::*;\n",
                encoding="utf-8",
            )
            items = inventory.rust_items(root)
        names = {item.name for item in items}
        self.assertIn("outer_fn", names)
        self.assertIn("inner_fn", names)
        self.assertIn("Inner", names)
        self.assertIn("value", names)
        self.assertNotIn("*", names)
        self.assertTrue(
            any(item.kind == "reexport" and item.name == "inner_fn" for item in items)
        )
        self.assertTrue(
            any(item.kind == "function" and item.name == "inner_fn" for item in items)
        )

    def test_associated_const_is_const(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src"
            source.mkdir()
            (source / "lib.rs").write_text(
                "pub struct Owner;\n"
                "impl Owner {\n"
                "    pub const LIMIT: i32 = 1;\n"
                "    pub const fn make() -> Self { Owner }\n"
                "    pub fn method(&self) {}\n"
                "}\n",
                encoding="utf-8",
            )
            items = inventory.rust_items(root)
        kinds = {item.name: item.kind for item in items}
        self.assertEqual(kinds["LIMIT"], "const")
        self.assertEqual(kinds["make"], "function")
        self.assertEqual(kinds["method"], "function")

    def test_public_trait_impl_methods_are_kept(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src"
            source.mkdir()
            (source / "lib.rs").write_text(
                "pub struct Control;\n"
                "impl Default for Control {\n"
                "    fn default() -> Self { Control }\n"
                "}\n"
                "impl Control {\n"
                "    fn hidden(&self) {}\n"
                "    pub fn shown(&self) {}\n"
                "}\n"
                "trait Secret {\n"
                "    fn leak(&self);\n"
                "}\n"
                "impl Secret for Control {\n"
                "    fn leak(&self) {}\n"
                "}\n",
                encoding="utf-8",
            )
            items = inventory.rust_items(root)
        names = {item.name for item in items}
        self.assertIn("default", names)
        self.assertIn("shown", names)
        self.assertNotIn("hidden", names)
        self.assertNotIn("leak", names)

    def test_crate_hides_private_modules_and_keeps_reexports(self) -> None:
        items = inventory.rust_items()
        names = {item.name for item in items}
        self.assertNotIn("bb1_alpha", names)
        self.assertNotIn("bb_direction", names)
        self.assertNotIn("DiisState", names)
        for name in ("Solver", "Control", "KrylovIteration", "minimize"):
            self.assertIn(name, names)


class HeaderDeclarationTests(unittest.TestCase):
    def test_uses_inside_bodies_and_macros_are_not_declarations(self) -> None:
        text = """
        #define xts_solver_create rgmin_solver_create
        enum class Method { PolakRibiere = RGMIN_POLAK_RIBIERE };
        void clear_etol() { rgmin_solver_clear_etol(ptr_); }
        static inline rgmin_solver_t *rgmin_solver_create(int method) {
            return rgmin_solver_create_sized(method);
        }
        typedef rgmin_status_t (*rgmin_eval_fn)(int);
        void rgmin_tensor_free(int n);
        int digits = 1'000;
        """
        found = {
            name
            for _line, name, _decl, _kind, *_scope in inventory.header_declared_names(
                text
            )
        }
        self.assertIn("xts_solver_create", found)
        self.assertIn("rgmin_solver_create", found)
        self.assertIn("rgmin_eval_fn", found)
        self.assertIn("rgmin_tensor_free", found)
        self.assertNotIn("rgmin_solver_create_sized", found)
        self.assertNotIn("rgmin_solver_clear_etol", found)
        self.assertNotIn("RGMIN_POLAK_RIBIERE", found)

    def test_installed_headers_keep_the_defining_line(self) -> None:
        items = inventory.header_items()
        created = [item for item in items if item.name == "rgmin_solver_create"]
        self.assertEqual([item.source for item in created], ["include/rgmin.h"])
        self.assertIn("static inline", created[0].declaration)
        self.assertNotIn("ptr_", created[0].declaration)
        header = (
            (inventory.ROOT / "include" / "rgmin.h")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        self.assertIn("static inline", header[created[0].line - 1])
        success = next(
            item
            for item in items
            if item.name == "RGMIN_SUCCESS" and item.source == "include/rgmin.h"
        )
        self.assertIn("RGMIN_SUCCESS", header[success.line - 1])
        leaked = [
            item
            for item in items
            if item.source.endswith("xtensor.hpp")
            and item.kind != "header"
            and (
                item.name.startswith("rgmin_")
                or item.name.startswith("xts_")
                or item.name.startswith("RGMIN_")
                or item.name.startswith("XTS_")
            )
        ]
        self.assertEqual(leaked, [])
        self.assertTrue(
            any(
                item.name == "xts_solver_create" and item.source == "include/xts.h"
                for item in items
            )
        )
        self.assertFalse(
            any(
                item.name == "rgmin_solver_create" and item.source == "include/xts.h"
                for item in items
            )
        )
        self.assertTrue(
            any(
                item.name == "RGMIN_SUCCESS" and item.source == "include/rgmin.h"
                for item in items
            )
        )

    def test_include_guards_are_not_declarations(self) -> None:
        text = """
        #ifndef RGMIN_H
        #define RGMIN_H
        #define RGMIN_H_MAJOR 0
        #define RGMIN_SUCCESS 0
        #endif
        """
        found = {
            name
            for _line, name, _decl, _kind, *_scope in inventory.header_declared_names(
                text
            )
        }
        self.assertNotIn("RGMIN_H", found)
        self.assertIn("RGMIN_H_MAJOR", found)
        self.assertIn("RGMIN_SUCCESS", found)
        items = inventory.header_items()
        installed = {(item.source, item.name) for item in items}
        self.assertNotIn(("include/rgmin.h", "RGMIN_H"), installed)
        self.assertNotIn(("include/xts.h", "XTS_OPTIMIZE_H"), installed)
        self.assertIn(("include/rgmin.h", "RGMIN_H_MAJOR"), installed)

    def test_public_cxx_namespace_types_and_functions(self) -> None:
        text = """
        namespace rgmin {
        enum class Method { Bfgs = 1 };
        struct Control { int maxiter; void run(); };
        class Solver { void step(); public: void solve(); };
        class LbfgsPairs { void push(); };
        inline const char* version() { return "v"; }
        inline void minimize_fn(int n);
        namespace detail {
            struct Hidden {};
            inline void helper() {}
        }
        using ScalarType = double;
        }
        namespace xts { namespace optimize {
        using rgmin::Method;
        using rgmin::Solver;
        using rgmin::LbfgsPairs;
        }}
        """
        found = {
            (name, kind)
            for _line, name, _decl, kind, *_scope in inventory.header_declared_names(
                text, cxx=True
            )
        }
        scoped = {
            (name, scope)
            for _line, name, _decl, kind, scope in inventory.header_declared_names(
                text, cxx=True
            )
            if kind == "function"
        }
        self.assertIn(("run", ("rgmin", "Control")), scoped)
        self.assertIn(("solve", ("rgmin", "Solver")), scoped)
        self.assertIn(("version", ("rgmin",)), scoped)
        self.assertIn(("Method", "enum"), found)
        self.assertIn(("Control", "struct"), found)
        self.assertIn(("Solver", "class"), found)
        self.assertIn(("LbfgsPairs", "class"), found)
        self.assertIn(("version", "function"), found)
        self.assertIn(("minimize_fn", "function"), found)
        self.assertIn(("ScalarType", "type"), found)
        self.assertIn(("run", "function"), found)
        self.assertIn(("solve", "function"), found)
        names = {name for name, _kind in found}
        self.assertNotIn("Hidden", names)
        self.assertNotIn("helper", names)
        self.assertNotIn("step", names)
        self.assertNotIn("push", names)
        self.assertNotIn("Bfgs", names)
        items = inventory.header_items()

        def kinds(source: str) -> dict[str, str]:
            return {
                item.name: item.kind
                for item in items
                if item.source == source and item.kind != "header"
            }

        opt = kinds("include/rgmin/optimize.hpp")
        for name, kind in (
            ("Method", "enum"),
            ("Control", "struct"),
            ("Solver", "class"),
            ("LbfgsPairs", "class"),
            ("version", "function"),
            ("ScalarType", "type"),
        ):
            self.assertEqual(opt.get(name), kind)
        self.assertEqual(opt.get("to_control"), "function")
        self.assertEqual(opt.get("push"), "function")
        for hidden in (
            "c_control",
            "canon_token",
            "one_shot_report",
            "BFGSOptimizer",
        ):
            self.assertNotIn(hidden, opt)
        alias = kinds("include/xts/optimize.hpp")
        self.assertEqual(alias.get("Method"), "reexport")
        self.assertEqual(alias.get("Solver"), "reexport")
        self.assertEqual(alias.get("LbfgsPairs"), "reexport")
        self.assertEqual(kinds("include/xts/xtensor.hpp"), {"minimize": "function"})


class PythonExportTests(unittest.TestCase):
    def test_underscore_modules_and_unlisted_names_are_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "python" / "rgmin"
            package.mkdir(parents=True)
            (package / "_core.py").write_text(
                "def hidden():\n    return 1\n", encoding="utf-8"
            )
            (package / "__init__.py").write_text(
                "_EXPORTS = ['visible_export', 'lazy_name']\n"
                "__all__ = _EXPORTS\n"
                "def visible_export():\n"
                "    return 1\n"
                "def skipped():\n"
                "    return 0\n"
                "class Kept:\n"
                "    def run(self):\n"
                "        return 1\n",
                encoding="utf-8",
            )
            items = inventory.python_items(root)
        names = {item.name for item in items}
        modules = {item.name for item in items if item.kind == "module"}
        self.assertIn("visible_export", names)
        self.assertIn("lazy_name", names)
        self.assertIn("rgmin.__init__", modules)
        self.assertNotIn("skipped", names)
        self.assertNotIn("Kept", names)
        self.assertNotIn("hidden", names)
        self.assertNotIn("rgmin._core", modules)

    def test_package_skips_private_modules(self) -> None:
        items = inventory.python_items()
        modules = {item.name for item in items if item.kind == "module"}
        self.assertNotIn("rgmin._core", modules)
        self.assertNotIn("rgmin._gromacs", modules)
        self.assertNotIn("rgmin._maturin", modules)
        sources = {item.source for item in items}
        self.assertNotIn("python/rgmin/_gromacs.py", sources)
        self.assertNotIn("python/rgmin/_maturin.py", sources)
        self.assertIn("python/rgmin/_core.pyi", sources)
        self.assertIn("python/rgmin/_maturin.pyi", sources)
        self.assertTrue(any(item.name == "minimize" for item in items))
        self.assertTrue(any(item.name == "gromacs_mdp_options" for item in items))
        self.assertTrue(any(item.name == "LbfgsPairs.push" for item in items))


class InventorySurfaceTests(unittest.TestCase):
    def test_generator_keeps_four_public_surfaces(self) -> None:
        rust = inventory.rust_items()
        headers = inventory.header_items()
        python = inventory.python_items()
        self.assertTrue(
            any(
                item.name == "default"
                and item.source == "src/control.rs"
                and item.declaration.startswith("fn default")
                for item in rust
            )
        )
        self.assertTrue(
            any(
                item.name == "push"
                and item.kind == "function"
                and item.source == "include/rgmin/optimize.hpp"
                and "push(" in item.declaration
                for item in headers
            )
        )
        self.assertTrue(
            any(
                item.name == "DLTensor"
                and item.source == "include/dlpack/dlpack.h"
                for item in headers
            )
        )
        self.assertTrue(
            any(
                item.name == "LbfgsPairs.push"
                and item.source == "python/rgmin/_maturin.pyi"
                for item in python
            )
        )


class AmbiguousNameTests(unittest.TestCase):
    def test_control_default_is_not_credited_to_lammps_page(self) -> None:
        items = (
            inventory.rust_items()
            + inventory.header_items()
            + inventory.python_items()
        )
        documents = inventory.org_documents()
        matches = [
            item
            for item in items
            if item.name == "default"
            and item.source == "src/control.rs"
            and item.line == 221
        ]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].scope, ("rgmin", "control", "Control"))
        places = inventory.documented_at(matches[0], documents, items)
        self.assertNotIn("docs/orgmode/reference/lammps-min-style.org", places)
        self.assertIn("docs/orgmode/reference/python.org", places)

    def test_bare_name_counts_only_on_the_items_reference_page(self) -> None:
        control = inventory.Item(
            "Rust",
            "function",
            "default",
            "src/control.rs",
            209,
            "fn default() -> Self {",
            ("rgmin", "control", "Control"),
        )
        fire = inventory.Item(
            "Rust",
            "function",
            "default",
            "src/fire.rs",
            140,
            "fn default() -> Self {",
            ("rgmin", "fire", "Fire"),
        )
        documents = {
            "default": [
                "docs/orgmode/reference/lammps-min-style.org",
                "docs/orgmode/reference/control.org",
            ]
        }
        places = inventory.documented_at(control, documents, [control, fire])
        self.assertEqual(places, ["docs/orgmode/reference/control.org"])
        self.assertEqual(
            inventory.documented_at(fire, documents, [control, fire]),
            [],
        )

    def test_type_or_module_path_selects_one_item(self) -> None:
        control = inventory.Item(
            "Rust",
            "function",
            "default",
            "src/control.rs",
            209,
            "fn default() -> Self {",
            ("rgmin", "control", "Control"),
        )
        other = inventory.Item(
            "Rust",
            "function",
            "default",
            "src/control.rs",
            40,
            "fn default() -> Self {",
            ("rgmin", "control", "StopNorm"),
        )
        fire = inventory.Item(
            "Rust",
            "function",
            "default",
            "src/fire.rs",
            140,
            "fn default() -> Self {",
            ("rgmin", "fire", "Fire"),
        )
        peers = [control, other, fire]
        typed = {"Control::default": ["docs/orgmode/howto/configure.org"]}
        self.assertEqual(
            inventory.documented_at(control, typed, peers),
            ["docs/orgmode/howto/configure.org"],
        )
        self.assertEqual(inventory.documented_at(other, typed, peers), [])
        shared = {"control::default": ["docs/orgmode/howto/configure.org"]}
        self.assertEqual(inventory.documented_at(control, shared, peers), [])
        self.assertEqual(inventory.documented_at(other, shared, peers), [])
        lone = {"fire::default": ["docs/orgmode/howto/fire.org"]}
        self.assertEqual(
            inventory.documented_at(fire, lone, peers),
            ["docs/orgmode/howto/fire.org"],
        )


class InventoryWordingTests(unittest.TestCase):
    def test_rendered_text_has_no_blocked_wording(self) -> None:
        text = inventory.render_inventory([], {}).lower()
        self.assertNotIn("regenerate", text)
        self.assertNotIn("landed", text)
        guide = (inventory.ROOT / "AGENTS.md").read_text().lower()
        self.assertNotIn("regenerate", guide)
        self.assertNotIn("landed", guide)


if __name__ == "__main__":
    unittest.main()
