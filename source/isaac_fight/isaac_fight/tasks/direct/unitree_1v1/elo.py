# Copyright (c) 2026, Isaac Fight contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Elo utilities for self-play and tournament evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EloRecord:
    rating: float = 1000.0
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def draw_rate(self) -> float:
        return self.draws / self.games if self.games else 0.0


@dataclass
class EloTable:
    default_rating: float = 1000.0
    k_factor: float = 32.0
    records: dict[str, EloRecord] = field(default_factory=dict)

    def ensure(self, policy_id: str) -> EloRecord:
        if policy_id not in self.records:
            self.records[policy_id] = EloRecord(rating=self.default_rating)
        return self.records[policy_id]

    def expected_score(self, a: str, b: str) -> float:
        ra = self.ensure(a).rating
        rb = self.ensure(b).rating
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

    def update(self, a: str, b: str, result_a: float) -> tuple[float, float]:
        """Update ratings.

        Args:
            a: First policy id.
            b: Second policy id.
            result_a: 1.0 if ``a`` wins, 0.0 if ``a`` loses, 0.5 for draw.

        Returns:
            The updated ratings ``(rating_a, rating_b)``.
        """

        if result_a not in (0.0, 0.5, 1.0):
            raise ValueError("result_a must be 0.0, 0.5 or 1.0")
        rec_a = self.ensure(a)
        rec_b = self.ensure(b)
        ea = self.expected_score(a, b)
        eb = 1.0 - ea
        result_b = 1.0 - result_a
        rec_a.rating += self.k_factor * (result_a - ea)
        rec_b.rating += self.k_factor * (result_b - eb)
        rec_a.games += 1
        rec_b.games += 1
        if result_a == 1.0:
            rec_a.wins += 1
            rec_b.losses += 1
        elif result_a == 0.0:
            rec_a.losses += 1
            rec_b.wins += 1
        else:
            rec_a.draws += 1
            rec_b.draws += 1
        return rec_a.rating, rec_b.rating

    def to_dict(self) -> dict[str, dict[str, float | int]]:
        return {
            key: {
                "rating": rec.rating,
                "games": rec.games,
                "wins": rec.wins,
                "losses": rec.losses,
                "draws": rec.draws,
                "win_rate": rec.win_rate,
                "draw_rate": rec.draw_rate,
            }
            for key, rec in sorted(self.records.items())
        }
