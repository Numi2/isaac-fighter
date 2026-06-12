# Copyright (c) 2026, Isaac Fight contributors.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from importlib import import_module
from typing import Any


def import_symbol(path: str) -> Any:
    module_name, _, symbol_name = path.rpartition(":")
    if not module_name:
        module_name, _, symbol_name = path.rpartition(".")
    if not module_name or not symbol_name:
        raise ValueError(f"Expected import path 'module:symbol' or 'module.symbol', got: {path}")
    module = import_module(module_name)
    return getattr(module, symbol_name)
