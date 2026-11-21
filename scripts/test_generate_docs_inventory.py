#!/usr/bin/env python3
"""Tests for the documentation inventory parser."""

from __future__ import annotations

import importlib.util
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


if __name__ == "__main__":
    unittest.main()
