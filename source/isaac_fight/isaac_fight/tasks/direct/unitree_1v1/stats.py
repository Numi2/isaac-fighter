# Copyright (c) 2026, Isaac Fight contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Episode and tournament statistics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MatchSummary:
    policy_version: str = "active"
    opponent_version: str = "active"
    winner: int = 0
    loser: int = 0
    draw: bool = False
    duration_s: float = 0.0
    fighter_a_reward: float = 0.0
    fighter_b_reward: float = 0.0
    fighter_a_knockdowns: int = 0
    fighter_b_knockdowns: int = 0
    fighter_a_self_falls: int = 0
    fighter_b_self_falls: int = 0
    fighter_a_oob_losses: int = 0
    fighter_b_oob_losses: int = 0
    avg_contact_force: float = 0.0
    avg_energy_use: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class TournamentAccumulator:
    matches: list[MatchSummary] = field(default_factory=list)

    def add(self, summary: MatchSummary) -> None:
        self.matches.append(summary)

    def aggregate(self) -> dict[str, Any]:
        n = len(self.matches)
        if n == 0:
            return {"matches": 0}
        wins_a = sum(1 for m in self.matches if m.winner == 1)
        wins_b = sum(1 for m in self.matches if m.winner == 2)
        draws = sum(1 for m in self.matches if m.draw)
        return {
            "matches": n,
            "fighter_a_win_rate": wins_a / n,
            "fighter_b_win_rate": wins_b / n,
            "draw_rate": draws / n,
            "average_duration_s": sum(m.duration_s for m in self.matches) / n,
            "average_contact_force": sum(m.avg_contact_force for m in self.matches) / n,
            "average_energy_use": sum(m.avg_energy_use for m in self.matches) / n,
            "fighter_a_knockdowns": sum(m.fighter_a_knockdowns for m in self.matches),
            "fighter_b_knockdowns": sum(m.fighter_b_knockdowns for m in self.matches),
            "fighter_a_self_falls": sum(m.fighter_a_self_falls for m in self.matches),
            "fighter_b_self_falls": sum(m.fighter_b_self_falls for m in self.matches),
            "fighter_a_oob_losses": sum(m.fighter_a_oob_losses for m in self.matches),
            "fighter_b_oob_losses": sum(m.fighter_b_oob_losses for m in self.matches),
        }
