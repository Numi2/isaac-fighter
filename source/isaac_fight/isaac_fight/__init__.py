# Copyright (c) 2026, Isaac Fight contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Fight standalone Isaac Lab extension."""

from __future__ import annotations

__version__ = "0.1.0"

# Importing tasks registers Gymnasium environments through each task package.
try:  # Isaac Lab may not be installed when pure utility tests run.
    from . import tasks  # noqa: F401
except Exception:
    tasks = None  # type: ignore[assignment]
