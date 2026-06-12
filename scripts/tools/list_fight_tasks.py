#!/usr/bin/env python3
# Copyright (c) 2026, Isaac Fight contributors.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym

import isaac_fight.tasks  # noqa: F401

for task_id in sorted(k for k in gym.registry.keys() if "GhostFighter" in k or "Fight" in k):
    print(task_id)
