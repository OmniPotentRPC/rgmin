#!/usr/bin/env python3
"""Tests for the documentation inventory parser."""

from __future__ import annotations

import importlib.util
import re
import sys
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
            with self.subTest(page=page):
                with self.assertRaises(RuntimeError):
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


if __name__ == "__main__":
    unittest.main()
