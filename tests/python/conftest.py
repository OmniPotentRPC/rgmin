# Copyright (c) 2026 Rohit Goswami
# SPDX-License-Identifier: MIT
"""Pytest configuration. Must run before NumPy is imported."""

from __future__ import annotations

import os

# EMT's np.exp follows NumPy's CPU dispatch. AVX-512 and the baseline
# kernel do not agree on the Pt7 trajectory, so every run, including
# the subprocesses, uses the baseline kernel and one golden hash.
os.environ.setdefault(
    "NPY_DISABLE_CPU_FEATURES",
    "X86_V3,X86_V4,AVX512_ICL,AVX512_SPR",
)
