# Copyright (c) 2026, Isaac Fight contributors.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .fighter_ids import FIGHTER_A, FIGHTER_B, FIGHTERS, OPPONENT, opponent_of


@dataclass
class StepEvents:
    """Per-environment match events computed by the rule engine."""

    fallen: dict[str, torch.Tensor]
    knockdown: dict[str, torch.Tensor]
    knockout: dict[str, torch.Tensor]
    out_of_bounds: dict[str, torch.Tensor]
    time_out: torch.Tensor
    terminal: torch.Tensor
    winner: torch.Tensor
    loser: torch.Tensor
    draw: torch.Tensor


@dataclass
class RewardBreakdown:
    """Named reward terms for logging and diagnostics."""

    total: torch.Tensor
    terms: dict[str, torch.Tensor] = field(default_factory=dict)

    def detached_mean_dict(self) -> dict[str, float]:
        return {k: float(v.detach().mean().item()) for k, v in self.terms.items()} | {
            "total": float(self.total.detach().mean().item())
        }


@dataclass
class FighterRuntimeInfo:
    """Runtime data resolved after Isaac Lab creates articulations."""

    agent_id: str
    robot_name: str
    joint_ids: list[int]
    joint_names: list[str]
    action_dim: int
    default_base_height: float
    action_scale: float


def dict_of_agents(value: Any) -> dict[str, Any]:
    return {FIGHTER_A: value, FIGHTER_B: value}


__all__ = [
    "FIGHTER_A",
    "FIGHTER_B",
    "FIGHTERS",
    "OPPONENT",
    "opponent_of",
    "StepEvents",
    "RewardBreakdown",
    "FighterRuntimeInfo",
    "dict_of_agents",
]
