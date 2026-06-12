# Copyright (c) 2026, Isaac Fight contributors.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

FIGHTER_A = "fighter_a"
FIGHTER_B = "fighter_b"
FIGHTERS = (FIGHTER_A, FIGHTER_B)
OPPONENT = {FIGHTER_A: FIGHTER_B, FIGHTER_B: FIGHTER_A}


def opponent_of(agent: str) -> str:
    try:
        return OPPONENT[agent]
    except KeyError as exc:
        raise KeyError(f"Unknown fighter id '{agent}'. Expected one of: {', '.join(FIGHTERS)}") from exc
