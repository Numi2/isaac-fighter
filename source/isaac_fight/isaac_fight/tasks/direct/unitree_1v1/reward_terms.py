# Copyright (c) 2026, Isaac Fight contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Modular combat reward terms."""

from __future__ import annotations

import torch

from isaac_fight.utils.torch_math import heading_error_to_target

from .fight_common import RewardBreakdown
from .fighter_ids import FIGHTER_A, FIGHTER_B


class CombatRewardComputer:
    """Computes shaped combat rewards and terminal match reward for one fighter."""

    def __init__(self, cfg):
        self.cfg = cfg

    def compute(self, env, agent: str, opponent: str) -> RewardBreakdown:
        scales = self.cfg
        root_pos = env.root_pos(agent)
        opp_pos = env.root_pos(opponent)
        rel_pos = opp_pos - root_pos
        distance = torch.linalg.norm(rel_pos[:, :2], dim=-1)
        prev_distance = env._prev_distance_to_opponent[agent]

        up_z = env._up_z[agent]
        upright = torch.clamp(
            (up_z - env.cfg.rules.knockdown_up_axis_z) / (1.0 - env.cfg.rules.knockdown_up_axis_z), 0.0, 1.0
        )
        lateral_ang_vel = torch.linalg.norm(env.root_ang_vel_b(agent)[:, :2], dim=-1)
        balance_recovery = upright * torch.exp(-0.20 * torch.square(lateral_ang_vel))
        height_ratio = root_pos[:, 2] / max(env._runtime[agent].default_base_height, 1.0e-6)
        standing_height = upright * torch.exp(-16.0 * torch.square(height_ratio - 1.0))
        support_contact = env._support_quality(agent) * standing_height
        low_base_height = torch.relu(0.88 - height_ratio) * (1.0 + 2.0 * (1.0 - upright))
        waist_action = env._waist_action_magnitude(agent)

        heading_error = heading_error_to_target(env.root_quat(agent), rel_pos)
        facing_gate = torch.clamp(torch.cos(heading_error), min=0.0)
        approach_delta = torch.clamp(prev_distance - distance, -0.25, 0.25)
        controlled_approach = approach_delta * facing_gate * upright
        locomotion_drive = env._locomotion_drive[agent] * facing_gate * upright
        contact_intent = env._contact_intent[agent] * facing_gate * upright * env.proxy_reward_scale()
        attack_momentum = env._attack_momentum[agent] * facing_gate * upright

        radial = torch.linalg.norm(root_pos[:, :2], dim=-1)
        arena_control = torch.clamp(1.0 - torch.square(radial / env.cfg.arena.radius), 0.0, 1.0)
        stay_inside = torch.clamp((env.cfg.arena.radius - radial) / env.cfg.arena.radius, -1.0, 1.0)

        useful_contact = env._useful_contact[agent] * upright
        destabilizing_impact = env._destabilizing_impact[agent] * upright
        topple_pressure = env._topple_pressure[agent] * upright
        drive_pressure = env._drive_pressure[agent] * upright
        support_break_pressure = env._support_break_pressure[agent] * upright
        recent_attack = torch.clamp(env._recent_attack_pressure[agent], 0.0, 5.0)
        attack_credit = torch.maximum(torch.clamp(env._proof_impact[agent], 0.0, 5.0), recent_attack)
        attack_gate = torch.clamp(attack_credit, 0.0, 1.0)
        opp_destabilization = env._opponent_destabilization[agent] * attack_gate
        proof = (attack_credit >= env.cfg.contact.fall_credit_min_attack).float()
        self_fall = env._fallen[agent].float()
        clean_attack = proof * upright * (1.0 - self_fall)
        opponent_fall = clean_attack * (env._new_fall[opponent].float() + 0.08 * env._fallen[opponent].float())
        opponent_knockdown = clean_attack * (
            env._new_knockdown[opponent].float() + 0.15 * env._knockdown[opponent].float()
        )
        mutual_fall = self_fall * env._fallen[opponent].float()
        impact_balance = attack_credit * balance_recovery * (1.0 - self_fall)
        impact_self_destabilization = attack_credit * (
            (1.0 - upright) + 0.50 * torch.clamp(lateral_ang_vel / 4.0, 0.0, 2.0) + self_fall
        )

        energy = env._energy[agent]
        torque_penalty = env._torque_penalty[agent]
        joint_limit_penalty = env._joint_limit_penalty[agent]
        jitter_penalty = env._jitter_penalty[agent]
        inactivity_penalty = env._inactivity[agent]
        spin_penalty = env._spin_without_contact[agent]
        uncontrolled_collision = env._uncontrolled_collision[agent]
        posture_instability = env._posture_instability[agent]

        final_win, final_loss, final_draw = self._terminal_terms(env, agent)

        terms = {
            "upright_stability": scales.upright_stability * upright,
            "balance_recovery": scales.balance_recovery * balance_recovery,
            "standing_height": scales.standing_height * standing_height,
            "support_contact": scales.support_contact * support_contact,
            "low_base_height": -scales.low_base_height * low_base_height,
            "waist_action": -scales.waist_action * waist_action,
            "controlled_approach": scales.controlled_approach * controlled_approach,
            "locomotion_drive": scales.locomotion_drive * locomotion_drive,
            "contact_intent": scales.contact_intent * contact_intent,
            "attack_momentum": scales.attack_momentum * attack_momentum,
            "arena_control": scales.arena_control * arena_control,
            "useful_contact": scales.useful_contact * useful_contact,
            "destabilizing_impact": scales.destabilizing_impact * destabilizing_impact,
            "topple_pressure": scales.topple_pressure * topple_pressure,
            "drive_pressure": scales.drive_pressure * drive_pressure,
            "support_break_pressure": scales.support_break_pressure * support_break_pressure,
            "opponent_fall": scales.opponent_fall * opponent_fall,
            "opponent_destabilization": scales.opponent_destabilization * opp_destabilization,
            "opponent_knockdown": scales.opponent_knockdown * opponent_knockdown,
            "impact_balance": scales.impact_balance * impact_balance,
            "impact_self_destabilization": -scales.impact_self_destabilization * impact_self_destabilization,
            "posture_instability": -scales.posture_instability * posture_instability,
            "mutual_fall": -scales.mutual_fall * mutual_fall,
            "stay_inside": scales.stay_inside * stay_inside,
            "energy_efficiency": -scales.energy * energy,
            "self_fall": -scales.self_fall * self_fall,
            "out_of_bounds": -scales.out_of_bounds * env._out_of_bounds[agent].float(),
            "excessive_torque": -scales.excessive_torque * torque_penalty,
            "joint_limit_abuse": -scales.joint_limit_abuse * joint_limit_penalty,
            "jitter": -scales.jitter * jitter_penalty,
            "inactivity": -scales.inactivity * inactivity_penalty,
            "spin_without_contact": -scales.spin_without_contact * spin_penalty,
            "uncontrolled_collision": -scales.uncontrolled_collision * uncontrolled_collision,
            "final_win": scales.final_win * final_win,
            "final_loss": -scales.final_loss * final_loss,
            "final_draw": scales.final_draw * final_draw,
        }
        total = torch.zeros_like(distance)
        for value in terms.values():
            total = total + value
        return RewardBreakdown(total=total, terms=terms)

    @staticmethod
    def _terminal_terms(env, agent: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if agent == FIGHTER_A:
            own_id = 1
            opp_id = 2
        elif agent == FIGHTER_B:
            own_id = 2
            opp_id = 1
        else:
            raise KeyError(agent)
        terminal = env._match_terminal.float()
        final_win = terminal * (env._winner == own_id).float()
        final_loss = terminal * (env._winner == opp_id).float()
        final_draw = terminal * env._draw.float()
        return final_win, final_loss, final_draw
