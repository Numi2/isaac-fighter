# Copyright (c) 2026, Isaac Fight contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Observation builders for the Unitree 1v1 direct MARL environment."""

from __future__ import annotations

import torch

from isaac_fight.utils.torch_math import heading_error_to_target, rotate_yaw_inverse

BASE_FEATURE_DIM = 31


def observation_dim(action_dim: int) -> int:
    """Observation dimension for a fighter with ``action_dim`` controlled joints."""

    return BASE_FEATURE_DIM + 3 * int(action_dim)


class CombatObservationBuilder:
    """Construct per-agent ego-centric observations.

    The observation is intentionally ego-centric and asymmetric-safe: each agent receives its own joint state length,
    last action length, and robot-normalized rule features. The opponent enters through relative root state, event state,
    and contact statistics rather than through opponent joint vectors whose size may differ.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def build(self, env, agent: str, opponent: str) -> torch.Tensor:
        root_pos = env.root_pos(agent)
        root_quat = env.root_quat(agent)
        opp_pos = env.root_pos(opponent)
        rel_pos_w = opp_pos - root_pos

        root_lin_vel_b = env.root_lin_vel_b(agent)
        root_ang_vel_b = env.root_ang_vel_b(agent)
        projected_gravity_b = env.projected_gravity_b(agent)
        rel_pos_b = rotate_yaw_inverse(root_quat, rel_pos_w)
        rel_vel_b = rotate_yaw_inverse(root_quat, env.root_lin_vel_w(opponent) - env.root_lin_vel_w(agent))
        heading_error = heading_error_to_target(root_quat, rel_pos_w)
        heading_features = torch.stack((torch.cos(heading_error), torch.sin(heading_error)), dim=-1)

        center_vec_b = rotate_yaw_inverse(root_quat, -root_pos)
        arena_margin = env.cfg.arena.radius - torch.linalg.norm(root_pos[:, :2], dim=-1)
        time_fraction = env.episode_length_buf.float() / max(float(env.max_episode_length), 1.0)
        arena_features = torch.stack(
            (
                center_vec_b[:, 0] / env.cfg.arena.radius,
                center_vec_b[:, 1] / env.cfg.arena.radius,
                arena_margin / env.cfg.arena.radius,
                time_fraction,
            ),
            dim=-1,
        )

        rule_features = torch.stack(
            (
                env._fallen[agent].float(),
                env._fallen[opponent].float(),
                env._knockdown_clock[agent] / max(env.cfg.rules.knockout_grace_s, 1.0e-6),
                env._knockdown_clock[opponent] / max(env.cfg.rules.knockout_grace_s, 1.0e-6),
                env._out_of_bounds[agent].float(),
                env._out_of_bounds[opponent].float(),
                torch.clamp(env._contact_force[agent] / env.cfg.contact.force_normalizer, 0.0, 5.0),
                torch.clamp(env._useful_contact[agent], 0.0, 5.0),
                torch.clamp(env._energy_ema[agent] / env.cfg.rewards.energy_normalizer, 0.0, 5.0),
                torch.clamp(env._opponent_destabilization[agent], 0.0, 5.0),
            ),
            dim=-1,
        )

        joint_pos_rel = env.joint_pos_rel(agent)
        joint_vel = torch.clamp(env.joint_vel(agent) * self.cfg.joint_velocity_scale, -self.cfg.clip_joint_velocity, self.cfg.clip_joint_velocity)
        last_action = env._last_actions[agent]

        obs = torch.cat(
            (
                torch.clamp(root_lin_vel_b * self.cfg.base_linear_velocity_scale, -5.0, 5.0),
                torch.clamp(root_ang_vel_b * self.cfg.base_angular_velocity_scale, -5.0, 5.0),
                projected_gravity_b,
                torch.clamp(rel_pos_b / self.cfg.relative_position_normalizer, -5.0, 5.0),
                torch.clamp(rel_vel_b * self.cfg.relative_velocity_scale, -5.0, 5.0),
                heading_features,
                arena_features,
                rule_features,
                torch.clamp(joint_pos_rel * self.cfg.joint_position_scale, -5.0, 5.0),
                joint_vel,
                last_action,
            ),
            dim=-1,
        )
        return torch.clamp(obs, -self.cfg.observation_clip, self.cfg.observation_clip)
