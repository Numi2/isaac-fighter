# Copyright (c) 2026, Isaac Fight contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Historical policy pool and opponent sampling for self-play."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class PolicyVersion:
    policy_id: str
    checkpoint_path: str
    version: int
    elo: float = 1000.0
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    created_at: float = field(default_factory=time.time)
    tags: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def loss_rate(self) -> float:
        return self.losses / self.games if self.games else 0.0

    @property
    def draw_rate(self) -> float:
        return self.draws / self.games if self.games else 0.0

    @property
    def weakness_score(self) -> float:
        # High score means useful for training the active policy: not too weak, but exposes losses.
        uncertainty = 1.0 / math.sqrt(max(self.games, 1))
        return 0.55 * self.loss_rate + 0.25 * uncertainty + 0.20 * (1.0 - abs(0.5 - self.win_rate) * 2.0)

    @property
    def league_role(self) -> str:
        role = self.metadata.get("league_role") if isinstance(self.metadata, dict) else None
        if isinstance(role, str) and role:
            return role
        for tag in self.tags:
            if tag.startswith("role:"):
                return tag.removeprefix("role:")
            if tag in {
                "main",
                "shove_exploiter",
                "body_slam_exploiter",
                "balance_breaker",
                "recovery_specialist",
                "brace_defender",
                "leg_kick_exploiter",
            }:
                return tag
        return "main"

    def to_json(self) -> dict:
        data = asdict(self)
        data["win_rate"] = self.win_rate
        data["loss_rate"] = self.loss_rate
        data["draw_rate"] = self.draw_rate
        data["weakness_score"] = self.weakness_score
        return data

    @classmethod
    def from_json(cls, data: dict) -> PolicyVersion:
        allowed = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        cleaned = {key: value for key, value in data.items() if key in allowed}
        if isinstance(cleaned.get("tags"), list):
            cleaned["tags"] = tuple(cleaned["tags"])
        return cls(**cleaned)


@dataclass
class OpponentSample:
    policy: PolicyVersion
    reason: str
    probability: float


class OpponentPool:
    """Persistent checkpoint pool with Elo/weakness-aware sampling."""

    def __init__(self, root: str | Path, rng: random.Random | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.root / "pool.json"
        self.rng = rng or random.Random()
        self._policies: dict[str, PolicyVersion] = {}
        self.load()

    @property
    def policies(self) -> tuple[PolicyVersion, ...]:
        return tuple(sorted(self._policies.values(), key=lambda p: p.version))

    def __len__(self) -> int:
        return len(self._policies)

    def load(self) -> None:
        if not self.metadata_path.exists():
            self._policies = {}
            return
        data = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self._policies = {item["policy_id"]: PolicyVersion.from_json(item) for item in data.get("policies", [])}

    def save(self) -> None:
        payload = {"schema": "isaac_fight.policy_pool.v1", "policies": [p.to_json() for p in self.policies]}
        tmp = self.metadata_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.metadata_path)

    def add_checkpoint(
        self,
        checkpoint_path: str | Path,
        version: int | None = None,
        policy_id: str | None = None,
        elo: float = 1000.0,
        tags: Iterable[str] = (),
        metadata: dict | None = None,
    ) -> PolicyVersion:
        checkpoint = Path(checkpoint_path)
        if version is None:
            version = 0 if not self._policies else max(p.version for p in self._policies.values()) + 1
        if policy_id is None:
            policy_id = f"policy_v{version:06d}"
        policy_id = self._dedupe_policy_id(policy_id, checkpoint)
        record = PolicyVersion(
            policy_id=policy_id,
            checkpoint_path=str(checkpoint),
            version=version,
            elo=elo,
            tags=tuple(tags),
            metadata=metadata or {},
        )
        self._policies[record.policy_id] = record
        self.save()
        return record

    def _dedupe_policy_id(self, policy_id: str, checkpoint: Path) -> str:
        existing = self._policies.get(policy_id)
        if existing is None:
            return policy_id
        try:
            if Path(existing.checkpoint_path).resolve() == checkpoint.resolve():
                return policy_id
        except OSError:
            if existing.checkpoint_path == str(checkpoint):
                return policy_id
        digest = hashlib.sha1(str(checkpoint.resolve()).encode("utf-8")).hexdigest()[:10]
        candidate = f"{policy_id}_{digest}"
        counter = 1
        while candidate in self._policies:
            counter += 1
            candidate = f"{policy_id}_{digest}_{counter}"
        return candidate

    def update_result(self, policy_id: str, result: float, elo: float | None = None) -> None:
        policy = self._policies[policy_id]
        policy.games += 1
        if result == 1.0:
            policy.wins += 1
        elif result == 0.0:
            policy.losses += 1
        elif result == 0.5:
            policy.draws += 1
        else:
            raise ValueError("result must be 1.0, 0.5 or 0.0")
        if elo is not None:
            policy.elo = elo
        self.save()

    def sample(
        self,
        active_elo: float,
        elo_window: float = 250.0,
        weakness_bias: float = 0.65,
        latest_bias: float = 0.15,
        league_role_weights: dict[str, float] | None = None,
        pfsp_hard_bias: float = 0.0,
        role_exploration: float = 0.05,
    ) -> OpponentSample | None:
        """Sample a checkpoint using Elo proximity and weakness score.

        The sampler first filters to policies within ``elo_window`` of the active policy. If none exist, it falls back
        to the full pool. The final probability combines Elo proximity, weakness score, and a small recency term.
        """

        policies = list(self.policies)
        if not policies:
            return None
        filtered = [p for p in policies if abs(p.elo - active_elo) <= elo_window]
        if not filtered:
            filtered = policies
        max_version = max(p.version for p in policies)
        role_totals = league_role_weights or {}
        role_exploration = max(0.0, float(role_exploration))
        weights = []
        for p in filtered:
            elo_proximity = math.exp(-abs(p.elo - active_elo) / max(elo_window, 1.0))
            recency = (p.version + 1) / (max_version + 1)
            weight = (1.0 - weakness_bias - latest_bias) * elo_proximity
            weight += weakness_bias * p.weakness_score
            weight += latest_bias * recency
            if role_totals:
                weight *= role_exploration + max(float(role_totals.get(p.league_role, 0.0)), 0.0)
            if pfsp_hard_bias > 0.0:
                hard_opponent = (1.0 - p.win_rate) ** 2 if p.games else 0.65
                weight = (1.0 - pfsp_hard_bias) * weight + pfsp_hard_bias * max(hard_opponent, 1.0e-6)
            weights.append(max(weight, 1.0e-6))
        total = sum(weights)
        r = self.rng.random() * total
        cumulative = 0.0
        for policy, weight in zip(filtered, weights, strict=True):
            cumulative += weight
            if r <= cumulative:
                return OpponentSample(policy=policy, reason="elo_weakness_recency", probability=weight / total)
        policy = filtered[-1]
        return OpponentSample(policy=policy, reason="elo_weakness_recency", probability=weights[-1] / total)
