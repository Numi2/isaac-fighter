# Copyright (c) 2026, Isaac Fight contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Rule helpers for 1v1 humanoid combat episodes."""

from __future__ import annotations

import torch

from isaac_fight.utils.torch_math import quat_apply


class FightRuleEngine:
    """Vectorized fight-rule logic used by the direct MARL environment."""

    def __init__(self, cfg):
        self.cfg = cfg

    def up_axis_z(self, root_quat_w: torch.Tensor) -> torch.Tensor:
        z_axis = torch.zeros(root_quat_w.shape[0], 3, device=root_quat_w.device, dtype=root_quat_w.dtype)
        z_axis[:, 2] = 1.0
        return quat_apply(root_quat_w, z_axis)[:, 2]

    def fallen(self, root_pos_w: torch.Tensor, root_quat_w: torch.Tensor, nominal_height: float) -> torch.Tensor:
        up_z = self.up_axis_z(root_quat_w)
        height_limit = min(self.cfg.fall_height, max(0.30, nominal_height * self.cfg.fall_height_ratio))
        return (root_pos_w[:, 2] < height_limit) | (up_z < self.cfg.fall_up_axis_z)

    def knockdown(self, root_pos_w: torch.Tensor, root_quat_w: torch.Tensor, nominal_height: float) -> torch.Tensor:
        up_z = self.up_axis_z(root_quat_w)
        height_limit = max(0.35, nominal_height * self.cfg.knockdown_height_ratio)
        return (root_pos_w[:, 2] < height_limit) | (up_z < self.cfg.knockdown_up_axis_z)

    def out_of_bounds(self, root_pos_w: torch.Tensor, arena_radius: float) -> torch.Tensor:
        return torch.linalg.norm(root_pos_w[:, :2], dim=-1) > arena_radius

    def assign_winner(
        self,
        terminal: torch.Tensor,
        time_out: torch.Tensor,
        knockout_a: torch.Tensor,
        knockout_b: torch.Tensor,
        oob_a: torch.Tensor,
        oob_b: torch.Tensor,
        score_a: torch.Tensor,
        score_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return winner, loser, draw tensors.

        Winner/loser encoding: 1 means fighter_a, 2 means fighter_b, 0 means none/draw.
        """

        winner = torch.zeros_like(score_a, dtype=torch.long)
        loser = torch.zeros_like(score_a, dtype=torch.long)

        a_loses = (knockout_a | oob_a) & ~(knockout_b | oob_b)
        b_loses = (knockout_b | oob_b) & ~(knockout_a | oob_a)
        winner[a_loses] = 2
        loser[a_loses] = 1
        winner[b_loses] = 1
        loser[b_loses] = 2

        # Timer decision remains a draw unless score separation is significant.
        decision = time_out & terminal
        margin = score_a - score_b
        a_decision = decision & (margin > self.cfg.timer_decision_margin)
        b_decision = decision & (margin < -self.cfg.timer_decision_margin)
        winner[a_decision] = 1
        loser[a_decision] = 2
        winner[b_decision] = 2
        loser[b_decision] = 1

        draw = terminal & (winner == 0)
        return winner, loser, draw
