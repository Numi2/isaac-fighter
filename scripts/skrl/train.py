#!/usr/bin/env python3
# ruff: noqa: E402,I001
# Copyright (c) 2026, Isaac Fight contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Train GhostFighter with skrl IPPO/MAPPO and closed-loop self-play."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Isaac Fight policies with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record training videos.")
parser.add_argument("--video_length", type=int, default=400)
parser.add_argument("--video_interval", type=int, default=5000)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default="GhostFighter-Unitree-1v1-Direct-v0")
parser.add_argument("--agent", type=str, default=None)
parser.add_argument("--algorithm", type=str, default="IPPO", choices=["IPPO", "MAPPO", "PPO"])
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument(
    "--curriculum_start_step",
    type=int,
    default=0,
    help="Initialize the environment curriculum/common step counter for playback or resumed staged runs.",
)
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax", "jax-numpy"])
parser.add_argument(
    "--self_play",
    action="store_true",
    default=True,
    help="Track policy versions and train against frozen pool opponents.",
)
parser.add_argument("--no_self_play", action="store_false", dest="self_play")
parser.add_argument(
    "--historical_opponent",
    action="store_true",
    default=True,
    help="Freeze opponent actions from sampled skrl/TorchScript pool policies.",
)
parser.add_argument("--no_historical_opponent", action="store_false", dest="historical_opponent")
parser.add_argument("--active_agent", type=str, default="fighter_a", choices=["fighter_a", "fighter_b"])
parser.add_argument("--pool_dir", type=str, default="policy_pool")
parser.add_argument("--snapshot_interval", type=int, default=50)
parser.add_argument("--pool_sync_interval_s", type=float, default=60.0)
parser.add_argument("--opponent_update_interval", type=int, default=None)
parser.add_argument("--side_swap_probability", type=float, default=None)
parser.add_argument("--live_self_play_fraction", type=float, default=None)
parser.add_argument(
    "--league_role",
    type=str,
    default="main",
    choices=[
        "main",
        "shove_exploiter",
        "body_slam_exploiter",
        "balance_breaker",
        "recovery_specialist",
        "brace_defender",
        "leg_kick_exploiter",
    ],
)
parser.add_argument("--residual_locomotion_checkpoint", type=str, default=None)
parser.add_argument("--residual_base_action_scale", type=float, default=1.0)
parser.add_argument("--residual_action_scale", type=float, default=0.08)
parser.add_argument("--residual_leg_action_scale", type=float, default=None)
parser.add_argument("--residual_waist_action_scale", type=float, default=None)
parser.add_argument("--residual_arm_action_scale", type=float, default=None)
parser.add_argument("--residual_other_action_scale", type=float, default=None)
parser.add_argument("--residual_late_leg_action_scale", type=float, default=None)
parser.add_argument("--residual_late_waist_action_scale", type=float, default=None)
parser.add_argument("--residual_late_arm_action_scale", type=float, default=None)
parser.add_argument("--residual_late_other_action_scale", type=float, default=None)
parser.add_argument("--residual_scale_ramp_start_step", type=int, default=None)
parser.add_argument("--residual_scale_ramp_end_step", type=int, default=None)
parser.add_argument("--residual_active_after_warmup", action=argparse.BooleanOptionalAction, default=None)
parser.add_argument("--motion_prior_artifact", type=str, default=None)
parser.add_argument("--motion_prior_discriminator", type=str, default=None)
parser.add_argument("--motion_prior_reward_scale", type=float, default=0.0)
parser.add_argument("--motion_prior_mimic_reward_weight", type=float, default=None)
parser.add_argument("--motion_prior_amp_reward_weight", type=float, default=None)
parser.add_argument("--motion_prior_discriminator_output_is_probability", action="store_true", default=False)
parser.add_argument("--motion_prior_min_joint_name_coverage", type=float, default=None)
parser.add_argument("--motion_prior_disallow_unnamed_dim_match", action="store_true", default=False)
parser.add_argument("--enable_pbt", action="store_true", default=False)
parser.add_argument("--pbt_mutation_seed", type=int, default=0)
parser.add_argument("--pbt_mutation_scale", type=float, default=0.15)
parser.add_argument(
    "--launch_preset",
    type=str,
    default="fast_contact_bootstrap",
    choices=["fast_contact_bootstrap", "stand_shove_bootstrap", "full_fight_self_play"],
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import skrl
import torch
from packaging import version

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaac_fight.tasks  # noqa: F401
from isaac_fight.locomotion_bootstrap import apply_locomotion_warmstart, is_locomotion_warmstart_checkpoint
from isaac_fight.tasks.direct.unitree_1v1.self_play import (
    LiveSelfPlayPoolSync,
    SelfPlayTrainingSupervisor,
    checkpoint_dir_from_log_dir,
    maybe_wrap_historical_opponent,
    maybe_wrap_residual_locomotion,
)

logger = logging.getLogger(__name__)

SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    raise RuntimeError(f"skrl>={SKRL_VERSION} is required, found {skrl.__version__}")

if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm == "ppo" else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


def _apply_launch_preset(env_cfg, agent_cfg: dict, preset: str, league_role: str = "main") -> None:  # noqa: ANN001
    if not hasattr(env_cfg, "fighter_a"):
        return
    refresh_spaces = False
    if preset in ("fast_contact_bootstrap", "stand_shove_bootstrap"):
        stand_shove_only = preset == "stand_shove_bootstrap"
        env_cfg.fighter_a.robot_name = "g1_29dof"
        env_cfg.fighter_b.robot_name = "g1_29dof"
        refresh_spaces = True
        env_cfg.decimation = 3
        env_cfg.sim.render_interval = env_cfg.decimation
        env_cfg.episode_length_s = 6.0 if stand_shove_only else 8.0
        env_cfg.arena.radius = 1.65
        env_cfg.rules.knockout_grace_s = max(float(env_cfg.rules.knockout_grace_s), 1.35)
        env_cfg.fighter_a.spawn_xy = (-0.50, -0.08)
        env_cfg.fighter_b.spawn_xy = (0.50, 0.08)
        env_cfg.fighter_a.spawn_yaw = 0.08
        env_cfg.fighter_b.spawn_yaw = math.pi + 0.08
        env_cfg.fighter_a.spawn_xy_noise = 0.05
        env_cfg.fighter_b.spawn_xy_noise = 0.05
        env_cfg.fighter_a.spawn_yaw_noise = 0.10
        env_cfg.fighter_b.spawn_yaw_noise = 0.10
        env_cfg.fighter_a.spawn_forward_speed = 0.0
        env_cfg.fighter_b.spawn_forward_speed = 0.0
        env_cfg.fighter_a.spawn_forward_speed_noise = 0.0
        env_cfg.fighter_b.spawn_forward_speed_noise = 0.0
        env_cfg.fighter_a.action_scale = 0.25
        env_cfg.fighter_b.action_scale = 0.25
        env_cfg.fighter_a.action_scale_profile = "unitree_velocity"
        env_cfg.fighter_b.action_scale_profile = "unitree_velocity"
        env_cfg.fighter_a.action_smoothing = 0.65
        env_cfg.fighter_b.action_smoothing = 0.65
        env_cfg.contact.useful_contact_distance = 1.45
        env_cfg.contact.attack_memory_s = 0.65
        env_cfg.contact.fall_credit_min_attack = 0.08
        env_cfg.curriculum.enabled = True
        env_cfg.curriculum.standing_warmup_s = max(
            float(env_cfg.curriculum.standing_warmup_s), 2.25 if stand_shove_only else 2.75
        )
        env_cfg.curriculum.action_hold_s = max(
            float(env_cfg.curriculum.action_hold_s), 1.40 if not stand_shove_only else 1.55
        )
        env_cfg.curriculum.action_ramp_s = max(
            float(env_cfg.curriculum.action_ramp_s), 1.35 if not stand_shove_only else 1.45
        )
        env_cfg.curriculum.fall_recovery_enabled = not stand_shove_only
        env_cfg.curriculum.fall_recovery_window_s = max(float(env_cfg.curriculum.fall_recovery_window_s), 1.60)
        env_cfg.curriculum.no_engagement_timeout_s = 6.0 if stand_shove_only else 5.2
        env_cfg.curriculum.no_engagement_grace_s = 4.4 if stand_shove_only else 3.2
        env_cfg.curriculum.proxy_gain_anneal_steps = min(
            int(env_cfg.curriculum.proxy_gain_anneal_steps), 36_000 if stand_shove_only else 20_000
        )
        env_cfg.curriculum.min_proxy_gain = max(
            float(env_cfg.curriculum.min_proxy_gain), 0.10 if stand_shove_only else 0.20
        )
        env_cfg.curriculum.stand_phase_steps = 10_000 if stand_shove_only else 12_000
        env_cfg.curriculum.approach_phase_steps = 24_000 if stand_shove_only else 32_000
        env_cfg.curriculum.hand_push_phase_steps = 64_000 if stand_shove_only else 72_000
        env_cfg.curriculum.body_slam_phase_steps = 1_000_000 if stand_shove_only else 120_000
        env_cfg.curriculum.full_fight_phase_steps = 1_200_000 if stand_shove_only else 180_000
        env_cfg.curriculum.phase_min_stance_quality = max(
            float(env_cfg.curriculum.phase_min_stance_quality), 0.62 if stand_shove_only else 0.68
        )
        env_cfg.curriculum.phase_min_support_quality = max(
            float(env_cfg.curriculum.phase_min_support_quality), 0.55 if stand_shove_only else 0.58
        )
        env_cfg.curriculum.adaptive_stability_governor_enabled = True
        env_cfg.curriculum.stability_governor_global_weight = max(
            float(env_cfg.curriculum.stability_governor_global_weight), 0.90
        )
        env_cfg.curriculum.stability_governor_min_stance = max(
            float(env_cfg.curriculum.stability_governor_min_stance), 0.58 if stand_shove_only else 0.62
        )
        env_cfg.curriculum.stability_governor_min_support = max(
            float(env_cfg.curriculum.stability_governor_min_support), 0.50 if stand_shove_only else 0.54
        )
        env_cfg.curriculum.stability_governor_max_fall_pressure = min(
            float(env_cfg.curriculum.stability_governor_max_fall_pressure), 0.28
        )
        if stand_shove_only:
            env_cfg.curriculum.fall_recovery_reset_probability = 0.0
        else:
            env_cfg.curriculum.fall_recovery_reset_probability = max(
                float(env_cfg.curriculum.fall_recovery_reset_probability), 0.08
            )
        env_cfg.curriculum.fall_recovery_reset_start_step = (
            12_000 if stand_shove_only else max(int(env_cfg.curriculum.fall_recovery_reset_start_step), 48_000)
        )
        env_cfg.perturbations.enabled = not stand_shove_only
        env_cfg.perturbations.probability = max(
            float(env_cfg.perturbations.probability), 0.85 if not stand_shove_only else 0.0
        )
        env_cfg.perturbations.start_step = max(
            int(getattr(env_cfg.perturbations, "start_step", 0)),
            1_000_000 if stand_shove_only else int(env_cfg.curriculum.hand_push_phase_steps),
        )
        env_cfg.perturbations.ramp_end_step = max(
            int(getattr(env_cfg.perturbations, "ramp_end_step", 0)),
            1_000_000 if stand_shove_only else int(env_cfg.curriculum.body_slam_phase_steps),
        )
        env_cfg.perturbations.min_history_stance = max(
            float(getattr(env_cfg.perturbations, "min_history_stance", 0.0)),
            0.62 if not stand_shove_only else 0.45,
        )
        env_cfg.perturbations.min_history_support = max(
            float(getattr(env_cfg.perturbations, "min_history_support", 0.0)),
            0.55 if not stand_shove_only else 0.40,
        )
        env_cfg.perturbations.time_min_s = min(float(env_cfg.perturbations.time_min_s), 0.45)
        env_cfg.perturbations.time_max_s = max(float(env_cfg.perturbations.time_max_s), 2.20)
        env_cfg.perturbations.linear_velocity_min = max(float(env_cfg.perturbations.linear_velocity_min), 0.25)
        env_cfg.perturbations.linear_velocity_max = max(float(env_cfg.perturbations.linear_velocity_max), 0.85)
        env_cfg.perturbations.angular_velocity_min = max(float(env_cfg.perturbations.angular_velocity_min), 0.25)
        env_cfg.perturbations.angular_velocity_max = max(float(env_cfg.perturbations.angular_velocity_max), 1.25)
        env_cfg.perturbations.recovery_window_s = max(float(env_cfg.perturbations.recovery_window_s), 1.50)
        env_cfg.observations_cfg.temporal_memory_s = max(float(env_cfg.observations_cfg.temporal_memory_s), 0.55)
        env_cfg.adr.enabled = not stand_shove_only
        env_cfg.adr.start_step = max(int(env_cfg.adr.start_step), 140_000 if not stand_shove_only else 1_000_000)
        env_cfg.adr.min_history_stance = max(float(env_cfg.adr.min_history_stance), 0.66)
        env_cfg.adr.min_history_support = max(float(env_cfg.adr.min_history_support), 0.58)
        env_cfg.self_play.opponent_update_interval = min(int(env_cfg.self_play.opponent_update_interval), 160)
        env_cfg.self_play.live_self_play_fraction = max(float(env_cfg.self_play.live_self_play_fraction), 0.45)
        env_cfg.self_play.league_training_enabled = not stand_shove_only
        env_cfg.self_play.league_role = league_role
        env_cfg.rewards.profile = "stand_shove_bootstrap"
        env_cfg.rewards.contact_intent = max(float(env_cfg.rewards.contact_intent), 2.8)
        env_cfg.rewards.alive = max(float(env_cfg.rewards.alive), 7.0 if not stand_shove_only else 9.0)
        env_cfg.rewards.standing_height = max(float(env_cfg.rewards.standing_height), 18.0)
        env_cfg.rewards.support_contact = max(float(env_cfg.rewards.support_contact), 9.0)
        env_cfg.rewards.low_base_height = max(float(env_cfg.rewards.low_base_height), 80.0)
        env_cfg.rewards.standing_pose = max(float(env_cfg.rewards.standing_pose), 12.0)
        env_cfg.rewards.warmup_action_restraint = max(float(env_cfg.rewards.warmup_action_restraint), 8.0)
        env_cfg.rewards.stand_still_joint_deviation = max(float(env_cfg.rewards.stand_still_joint_deviation), 10.0)
        env_cfg.rewards.arm_motion_restraint = max(float(env_cfg.rewards.arm_motion_restraint), 6.0)
        env_cfg.rewards.hip_yaw_roll_deviation = max(float(env_cfg.rewards.hip_yaw_roll_deviation), 5.0)
        env_cfg.rewards.center_of_mass_over_support = max(float(env_cfg.rewards.center_of_mass_over_support), 14.0)
        env_cfg.rewards.capture_point_support = max(float(env_cfg.rewards.capture_point_support), 9.0)
        env_cfg.rewards.both_feet_support_warmup = max(float(env_cfg.rewards.both_feet_support_warmup), 10.0)
        env_cfg.rewards.foot_support_quality = max(float(env_cfg.rewards.foot_support_quality), 10.0)
        env_cfg.rewards.foot_slip = max(float(env_cfg.rewards.foot_slip), 8.0)
        env_cfg.rewards.base_pitch_roll = max(float(env_cfg.rewards.base_pitch_roll), 28.0)
        env_cfg.rewards.angular_stumble = max(float(env_cfg.rewards.angular_stumble), 12.0)
        env_cfg.rewards.knee_collapse = max(float(env_cfg.rewards.knee_collapse), 18.0)
        env_cfg.rewards.leg_extension_posture = max(float(env_cfg.rewards.leg_extension_posture), 8.0)
        env_cfg.rewards.perturbation_recovery = max(float(env_cfg.rewards.perturbation_recovery), 14.0)
        env_cfg.rewards.perturbation_collapse = max(float(env_cfg.rewards.perturbation_collapse), 55.0)
        env_cfg.rewards.fall_recovery_getup = max(float(env_cfg.rewards.fall_recovery_getup), 18.0)
        env_cfg.rewards.fall_recovery_stand = max(float(env_cfg.rewards.fall_recovery_stand), 16.0)
        env_cfg.rewards.fall_recovery_failure = max(float(env_cfg.rewards.fall_recovery_failure), 45.0)
        env_cfg.rewards.airborne_without_attack = max(float(env_cfg.rewards.airborne_without_attack), 24.0)
        env_cfg.rewards.fall_early = max(float(env_cfg.rewards.fall_early), 90.0)
        env_cfg.rewards.recovery_reward = max(float(env_cfg.rewards.recovery_reward), 5.0)
        env_cfg.rewards.backward_motion = max(float(env_cfg.rewards.backward_motion), 18.0)
        env_cfg.rewards.backward_lean = max(float(env_cfg.rewards.backward_lean), 24.0)
        env_cfg.rewards.waist_action = max(float(env_cfg.rewards.waist_action), 4.0)
        env_cfg.rewards.velocity_command_tracking = max(float(env_cfg.rewards.velocity_command_tracking), 8.0)
        env_cfg.rewards.yaw_heading_tracking = max(float(env_cfg.rewards.yaw_heading_tracking), 5.0)
        env_cfg.rewards.locomotion_drive = max(float(env_cfg.rewards.locomotion_drive), 3.2)
        env_cfg.rewards.forward_step_progress = max(float(env_cfg.rewards.forward_step_progress), 5.0)
        env_cfg.rewards.retreat_from_opponent = max(float(env_cfg.rewards.retreat_from_opponent), 8.0)
        env_cfg.rewards.approach_with_feet_gate = max(float(env_cfg.rewards.approach_with_feet_gate), 5.0)
        env_cfg.rewards.stance_width = max(float(env_cfg.rewards.stance_width), 3.0)
        env_cfg.rewards.foot_clearance = max(float(env_cfg.rewards.foot_clearance), 1.5)
        env_cfg.rewards.feet_air_time_biped = max(float(env_cfg.rewards.feet_air_time_biped), 3.0)
        env_cfg.rewards.single_stance_balance = max(float(env_cfg.rewards.single_stance_balance), 4.0)
        env_cfg.rewards.cadence_or_alternating_support = max(float(env_cfg.rewards.cadence_or_alternating_support), 1.4)
        env_cfg.rewards.leg_drive_participation = max(float(env_cfg.rewards.leg_drive_participation), 4.0)
        env_cfg.rewards.foot_plant_during_push = max(float(env_cfg.rewards.foot_plant_during_push), 6.0)
        env_cfg.rewards.opponent_angular_destabilization = max(
            float(env_cfg.rewards.opponent_angular_destabilization), 5.0
        )
        env_cfg.rewards.torso_grounded_penalty = max(float(env_cfg.rewards.torso_grounded_penalty), 18.0)
        env_cfg.rewards.unstable_attack = max(float(env_cfg.rewards.unstable_attack), 22.0)
        env_cfg.rewards.collapse_contact_credit = max(float(env_cfg.rewards.collapse_contact_credit), 28.0)
        env_cfg.rewards.forward_collapse = max(float(env_cfg.rewards.forward_collapse), 24.0)
        env_cfg.rewards.torso_first_contact = max(float(env_cfg.rewards.torso_first_contact), 22.0)
        env_cfg.rewards.root_height_velocity_down = max(float(env_cfg.rewards.root_height_velocity_down), 25.0)
        env_cfg.rewards.base_vertical_velocity = max(float(env_cfg.rewards.base_vertical_velocity), 7.0)
        env_cfg.rewards.torso_only_motion = max(float(env_cfg.rewards.torso_only_motion), 24.0)
        env_cfg.rewards.attack_momentum = max(float(env_cfg.rewards.attack_momentum), 3.4)
        env_cfg.rewards.stable_contact_attack = max(float(env_cfg.rewards.stable_contact_attack), 5.0)
        env_cfg.rewards.limb_contact_reward = max(float(env_cfg.rewards.limb_contact_reward), 4.0)
        env_cfg.rewards.one_hand_push_setup = max(float(env_cfg.rewards.one_hand_push_setup), 5.0)
        env_cfg.rewards.one_hand_push_contact = max(float(env_cfg.rewards.one_hand_push_contact), 9.0)
        env_cfg.rewards.one_hand_push_balance = max(float(env_cfg.rewards.one_hand_push_balance), 8.0)
        env_cfg.rewards.one_hand_push_destabilize = max(float(env_cfg.rewards.one_hand_push_destabilize), 8.0)
        env_cfg.rewards.offhand_push_penalty = max(float(env_cfg.rewards.offhand_push_penalty), 4.5)
        env_cfg.rewards.torso_charge_reward = max(float(env_cfg.rewards.torso_charge_reward), 2.5)
        env_cfg.rewards.bad_contact_penalty = max(float(env_cfg.rewards.bad_contact_penalty), 8.0)
        env_cfg.rewards.drive_pressure = max(float(env_cfg.rewards.drive_pressure), 6.2)
        env_cfg.rewards.support_break_pressure = max(float(env_cfg.rewards.support_break_pressure), 7.2)
        env_cfg.rewards.opponent_tilt_delta = max(float(env_cfg.rewards.opponent_tilt_delta), 7.0)
        env_cfg.rewards.opponent_height_drop_delta = max(float(env_cfg.rewards.opponent_height_drop_delta), 5.0)
        env_cfg.rewards.opponent_support_break = max(float(env_cfg.rewards.opponent_support_break), 6.0)
        env_cfg.rewards.impulse_direction_reward = max(float(env_cfg.rewards.impulse_direction_reward), 4.0)
        env_cfg.rewards.opponent_fall = max(float(env_cfg.rewards.opponent_fall), 22.0)
        env_cfg.rewards.opponent_knockdown = max(float(env_cfg.rewards.opponent_knockdown), 36.0)
        env_cfg.rewards.clean_knockdown_bonus = max(float(env_cfg.rewards.clean_knockdown_bonus), 24.0)
        env_cfg.rewards.impact_self_destabilization = max(float(env_cfg.rewards.impact_self_destabilization), 18.0)
        env_cfg.rewards.posture_instability = max(float(env_cfg.rewards.posture_instability), 7.0)
        env_cfg.rewards.mutual_fall_hard_penalty = max(float(env_cfg.rewards.mutual_fall_hard_penalty), 35.0)
        env_cfg.rewards.self_contact_abuse = max(float(env_cfg.rewards.self_contact_abuse), 5.0)
        env_cfg.rewards.wall_boundary_escape = max(float(env_cfg.rewards.wall_boundary_escape), 7.0)
        env_cfg.rewards.self_fall = max(float(env_cfg.rewards.self_fall), 120.0)
        env_cfg.rewards.joint_limit_slam = max(float(env_cfg.rewards.joint_limit_slam), 2.5)
        env_cfg.rewards.action_rate = max(float(env_cfg.rewards.action_rate), 0.35)
        env_cfg.rewards.torque_spike = max(float(env_cfg.rewards.torque_spike), 0.35)
        env_cfg.rewards.spin_flail_penalty = max(float(env_cfg.rewards.spin_flail_penalty), 5.0)
        env_cfg.rewards.passive_survival = max(float(env_cfg.rewards.passive_survival), 4.0)
        env_cfg.rewards.contact_without_progress = max(float(env_cfg.rewards.contact_without_progress), 5.0)
        env_cfg.rewards.warmup_stand_reward = max(float(env_cfg.rewards.warmup_stand_reward), 14.0)
        env_cfg.rewards.combat_gate = max(float(env_cfg.rewards.combat_gate), 1.5)
        env_cfg.rewards.progressive_attack_gate = max(float(env_cfg.rewards.progressive_attack_gate), 2.5)
        env_cfg.rewards.fall_cause_credit = max(float(env_cfg.rewards.fall_cause_credit), 18.0)
        env_cfg.rewards.energy = min(float(env_cfg.rewards.energy), 0.010)
        env_cfg.rewards.jitter = min(float(env_cfg.rewards.jitter), 0.08)
        if stand_shove_only:
            env_cfg.rewards.contact_intent = 0.0
            env_cfg.rewards.attack_momentum = min(float(env_cfg.rewards.attack_momentum), 0.8)
            env_cfg.rewards.torso_charge_reward = 0.0
            env_cfg.rewards.destabilizing_impact = min(float(env_cfg.rewards.destabilizing_impact), 2.0)
            env_cfg.rewards.topple_pressure = min(float(env_cfg.rewards.topple_pressure), 2.0)
            env_cfg.rewards.drive_pressure = min(float(env_cfg.rewards.drive_pressure), 1.0)
            env_cfg.rewards.useful_contact = min(float(env_cfg.rewards.useful_contact), 1.0)
            env_cfg.rewards.standing_height = max(float(env_cfg.rewards.standing_height), 24.0)
            env_cfg.rewards.support_contact = max(float(env_cfg.rewards.support_contact), 18.0)
            env_cfg.rewards.center_of_mass_over_support = max(float(env_cfg.rewards.center_of_mass_over_support), 20.0)
            env_cfg.rewards.capture_point_support = max(float(env_cfg.rewards.capture_point_support), 16.0)
            env_cfg.rewards.foot_support_quality = max(float(env_cfg.rewards.foot_support_quality), 16.0)
            env_cfg.rewards.base_pitch_roll = max(float(env_cfg.rewards.base_pitch_roll), 34.0)
            env_cfg.rewards.root_height_velocity_down = max(float(env_cfg.rewards.root_height_velocity_down), 36.0)
            env_cfg.rewards.base_vertical_velocity = max(float(env_cfg.rewards.base_vertical_velocity), 9.0)
            env_cfg.rewards.forward_collapse = max(float(env_cfg.rewards.forward_collapse), 42.0)
            env_cfg.rewards.collapse_contact_credit = max(float(env_cfg.rewards.collapse_contact_credit), 44.0)
            env_cfg.rewards.torso_first_contact = max(float(env_cfg.rewards.torso_first_contact), 36.0)
            env_cfg.rewards.torso_grounded_penalty = max(float(env_cfg.rewards.torso_grounded_penalty), 32.0)
            env_cfg.rewards.low_base_height = max(float(env_cfg.rewards.low_base_height), 110.0)
            env_cfg.rewards.self_fall = max(float(env_cfg.rewards.self_fall), 160.0)
            env_cfg.rewards.fall_early = max(float(env_cfg.rewards.fall_early), 120.0)
            env_cfg.rewards.one_hand_push_setup = max(float(env_cfg.rewards.one_hand_push_setup), 8.0)
            env_cfg.rewards.one_hand_push_contact = max(float(env_cfg.rewards.one_hand_push_contact), 14.0)
            env_cfg.rewards.one_hand_push_balance = max(float(env_cfg.rewards.one_hand_push_balance), 14.0)
            env_cfg.rewards.one_hand_push_destabilize = max(float(env_cfg.rewards.one_hand_push_destabilize), 10.0)
            env_cfg.rewards.foot_plant_during_push = max(float(env_cfg.rewards.foot_plant_during_push), 12.0)
            env_cfg.rewards.offhand_push_penalty = max(float(env_cfg.rewards.offhand_push_penalty), 6.0)
            env_cfg.rewards.bad_contact_penalty = max(float(env_cfg.rewards.bad_contact_penalty), 12.0)
        _apply_league_role_reward_bias(env_cfg, league_role)
        env_cfg.diagnostics.reward_terms_interval = max(int(env_cfg.diagnostics.reward_terms_interval), 2048)
        agent_cfg["agent"]["rollouts"] = 32
        agent_cfg["agent"]["learning_epochs"] = min(int(agent_cfg["agent"]["learning_epochs"]), 2)
        agent_cfg["agent"]["mini_batches"] = min(int(agent_cfg["agent"]["mini_batches"]), 4)
        agent_cfg["agent"]["entropy_loss_scale"] = max(float(agent_cfg["agent"]["entropy_loss_scale"]), 0.008)
        agent_cfg["agent"]["experiment"]["write_interval"] = max(
            int(agent_cfg["agent"]["experiment"]["write_interval"]), 2000
        )
    elif preset == "full_fight_self_play":
        env_cfg.episode_length_s = 30.0
        env_cfg.arena.radius = 3.5
        env_cfg.fighter_a.spawn_xy = (-0.78, 0.0)
        env_cfg.fighter_b.spawn_xy = (0.78, 0.0)
        env_cfg.fighter_a.spawn_xy_noise = 0.06
        env_cfg.fighter_b.spawn_xy_noise = 0.06
        env_cfg.contact.useful_contact_distance = 1.95
        env_cfg.curriculum.enabled = False
        agent_cfg["agent"]["rollouts"] = max(int(agent_cfg["agent"]["rollouts"]), 64)
    if refresh_spaces and hasattr(env_cfg, "__post_init__"):
        env_cfg.__post_init__()


def _apply_league_role_reward_bias(env_cfg, league_role: str) -> None:  # noqa: ANN001
    if league_role == "shove_exploiter":
        env_cfg.rewards.one_hand_push_contact *= 1.35
        env_cfg.rewards.one_hand_push_destabilize *= 1.35
        env_cfg.rewards.opponent_support_break *= 1.25
        env_cfg.rewards.support_break_pressure *= 1.20
    elif league_role == "body_slam_exploiter":
        env_cfg.rewards.torso_charge_reward *= 1.55
        env_cfg.rewards.drive_pressure *= 1.35
        env_cfg.rewards.impact_balance *= 1.35
        env_cfg.rewards.mutual_fall_hard_penalty *= 1.20
    elif league_role == "balance_breaker":
        env_cfg.rewards.topple_pressure *= 1.35
        env_cfg.rewards.opponent_tilt_delta *= 1.35
        env_cfg.rewards.opponent_support_break *= 1.35
        env_cfg.rewards.perturbation_recovery *= 1.15
    elif league_role == "recovery_specialist":
        env_cfg.rewards.fall_recovery_getup *= 1.55
        env_cfg.rewards.fall_recovery_stand *= 1.45
        env_cfg.rewards.perturbation_recovery *= 1.35
        env_cfg.rewards.self_fall *= 1.20
        env_cfg.curriculum.fall_recovery_reset_probability = max(
            float(env_cfg.curriculum.fall_recovery_reset_probability), 0.25
        )
    elif league_role == "brace_defender":
        env_cfg.rewards.center_of_mass_over_support *= 1.35
        env_cfg.rewards.capture_point_support *= 1.35
        env_cfg.rewards.foot_plant_during_push *= 1.30
        env_cfg.rewards.impact_balance *= 1.35
        env_cfg.rewards.impact_self_destabilization *= 1.25
    elif league_role == "leg_kick_exploiter":
        env_cfg.rewards.leg_drive_participation *= 1.35
        env_cfg.rewards.limb_contact_reward *= 1.35
        env_cfg.rewards.foot_clearance *= 1.25
        env_cfg.rewards.opponent_support_break *= 1.25
        env_cfg.rewards.bad_contact_penalty *= 1.20


def _apply_pbt_reward_mutation(env_cfg, seed: int, mutation_scale: float) -> None:  # noqa: ANN001
    rng = random.Random(int(seed))
    names = (
        "standing_height",
        "support_contact",
        "center_of_mass_over_support",
        "capture_point_support",
        "foot_support_quality",
        "perturbation_recovery",
        "fall_recovery_getup",
        "fall_recovery_stand",
        "leg_drive_participation",
        "foot_plant_during_push",
        "one_hand_push_contact",
        "one_hand_push_balance",
        "one_hand_push_destabilize",
        "opponent_support_break",
        "impact_self_destabilization",
        "unstable_attack",
        "collapse_contact_credit",
        "forward_collapse",
        "torso_first_contact",
        "mutual_fall_hard_penalty",
    )
    scale = max(0.0, float(mutation_scale))
    for name in names:
        value = float(getattr(env_cfg.rewards, name))
        multiplier = math.exp(rng.uniform(-scale, scale))
        setattr(env_cfg.rewards, name, value * multiplier)


def _adapt_checkpoint_observation_space(path: str, env_cfg, algorithm_name: str, log_dir: str) -> str:  # noqa: ANN001
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception:
        return path
    if not isinstance(checkpoint, dict):
        return path
    changed = False
    for agent, expected_obs in getattr(env_cfg, "observation_spaces", {}).items():
        agent_checkpoint = checkpoint.get(agent)
        if not isinstance(agent_checkpoint, dict):
            continue
        agent_changed = False
        if _expand_model_input(agent_checkpoint.get("policy"), int(expected_obs)):
            agent_changed = True
        value_target = int(env_cfg.state_space) if algorithm_name == "mappo" else int(expected_obs)
        if _expand_model_input(agent_checkpoint.get("value"), value_target):
            agent_changed = True
        if _expand_preprocessor(agent_checkpoint.get("state_preprocessor"), int(expected_obs)):
            agent_changed = True
        if agent_changed:
            agent_checkpoint.pop("optimizer", None)
            changed = True
    if not changed:
        return path
    adapted = Path(log_dir) / "params" / f"{Path(path).stem}_obs_adapted.pt"
    adapted.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, adapted)
    print(f"[INFO] Adapted checkpoint observation inputs: {path} -> {adapted}")
    return str(adapted)


def _expand_model_input(state_dict, target_dim: int) -> bool:  # noqa: ANN001
    if not isinstance(state_dict, dict):
        return False
    weight_keys = sorted(
        k for k, value in state_dict.items() if k.endswith(".weight") and hasattr(value, "ndim") and value.ndim == 2
    )
    for key in weight_keys:
        weight = state_dict[key]
        current = int(weight.shape[1])
        if current == target_dim:
            return False
        if current > target_dim:
            return False
        state_dict[key] = torch.nn.functional.pad(weight, (0, target_dim - current, 0, 0))
        return True
    return False


def _expand_preprocessor(preprocessor, target_dim: int) -> bool:  # noqa: ANN001
    if not isinstance(preprocessor, dict):
        return False
    changed = False
    for key, fill in (("running_mean", 0.0), ("running_variance", 1.0)):
        value = preprocessor.get(key)
        if not hasattr(value, "shape") or value.numel() == 0:
            continue
        current = int(value.shape[-1])
        if current >= target_dim:
            continue
        pad = torch.full((*value.shape[:-1], target_dim - current), fill, dtype=value.dtype)
        preprocessor[key] = torch.cat((value, pad), dim=-1)
        changed = True
    return changed


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    _apply_launch_preset(env_cfg, agent_cfg, args_cli.launch_preset, args_cli.league_role)
    if args_cli.launch_preset == "stand_shove_bootstrap":
        args_cli.self_play = False
        args_cli.historical_opponent = False
    if args_cli.residual_locomotion_checkpoint and hasattr(env_cfg, "residual_locomotion"):
        env_cfg.residual_locomotion.enabled = True
        env_cfg.residual_locomotion.checkpoint_path = args_cli.residual_locomotion_checkpoint
        env_cfg.residual_locomotion.base_action_scale = args_cli.residual_base_action_scale
        env_cfg.residual_locomotion.residual_action_scale = args_cli.residual_action_scale
        for arg_name in (
            "residual_leg_action_scale",
            "residual_waist_action_scale",
            "residual_arm_action_scale",
            "residual_other_action_scale",
            "residual_late_leg_action_scale",
            "residual_late_waist_action_scale",
            "residual_late_arm_action_scale",
            "residual_late_other_action_scale",
        ):
            value = getattr(args_cli, arg_name)
            if value is not None:
                setattr(env_cfg.residual_locomotion, arg_name, value)
        if args_cli.residual_scale_ramp_start_step is not None:
            env_cfg.residual_locomotion.residual_scale_ramp_start_step = args_cli.residual_scale_ramp_start_step
        if args_cli.residual_scale_ramp_end_step is not None:
            env_cfg.residual_locomotion.residual_scale_ramp_end_step = args_cli.residual_scale_ramp_end_step
        if args_cli.residual_active_after_warmup is not None:
            env_cfg.residual_locomotion.active_after_warmup = args_cli.residual_active_after_warmup
    if args_cli.motion_prior_artifact and hasattr(env_cfg, "motion_prior"):
        env_cfg.motion_prior.enabled = True
        env_cfg.motion_prior.artifact_path = args_cli.motion_prior_artifact
        env_cfg.motion_prior.reward_scale = args_cli.motion_prior_reward_scale
        env_cfg.rewards.motion_prior = max(float(env_cfg.rewards.motion_prior), 1.0)
    if args_cli.motion_prior_discriminator and hasattr(env_cfg, "motion_prior"):
        env_cfg.motion_prior.enabled = True
        env_cfg.motion_prior.discriminator_path = args_cli.motion_prior_discriminator
        env_cfg.motion_prior.reward_scale = args_cli.motion_prior_reward_scale
        env_cfg.rewards.motion_prior = max(float(env_cfg.rewards.motion_prior), 1.0)
    if hasattr(env_cfg, "motion_prior"):
        if args_cli.motion_prior_mimic_reward_weight is not None:
            env_cfg.motion_prior.mimic_reward_weight = args_cli.motion_prior_mimic_reward_weight
        if args_cli.motion_prior_amp_reward_weight is not None:
            env_cfg.motion_prior.amp_reward_weight = args_cli.motion_prior_amp_reward_weight
        if args_cli.motion_prior_discriminator_output_is_probability:
            env_cfg.motion_prior.discriminator_output_is_probability = True
        if args_cli.motion_prior_min_joint_name_coverage is not None:
            env_cfg.motion_prior.min_joint_name_coverage = args_cli.motion_prior_min_joint_name_coverage
        if args_cli.motion_prior_disallow_unnamed_dim_match:
            env_cfg.motion_prior.allow_unnamed_dim_match = False
    if args_cli.enable_pbt and hasattr(env_cfg, "pbt"):
        env_cfg.pbt.enabled = True
        env_cfg.pbt.mutation_seed = int(args_cli.pbt_mutation_seed)
        env_cfg.pbt.mutation_scale = float(args_cli.pbt_mutation_scale)
        _apply_pbt_reward_mutation(env_cfg, args_cli.pbt_mutation_seed, args_cli.pbt_mutation_scale)
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    if args_cli.snapshot_interval:
        checkpoint_every = max(1, args_cli.snapshot_interval) * agent_cfg["agent"]["rollouts"]
        agent_cfg["agent"]["experiment"]["checkpoint_interval"] = checkpoint_every
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg.get("seed", 42)
    env_cfg.seed = agent_cfg["seed"]

    if hasattr(env_cfg, "self_play"):
        env_cfg.self_play.enabled = args_cli.self_play
        env_cfg.self_play.pool_dir = args_cli.pool_dir
        env_cfg.self_play.snapshot_interval = args_cli.snapshot_interval
        env_cfg.self_play.active_agent = args_cli.active_agent
        env_cfg.self_play.league_role = args_cli.league_role
        if args_cli.opponent_update_interval is not None:
            env_cfg.self_play.opponent_update_interval = args_cli.opponent_update_interval
        if args_cli.side_swap_probability is not None:
            env_cfg.self_play.side_swap_probability = args_cli.side_swap_probability
        if args_cli.live_self_play_fraction is not None:
            env_cfg.self_play.live_self_play_fraction = args_cli.live_self_play_fraction

    log_root_path = os.path.abspath(os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"]))
    log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    if agent_cfg["agent"]["experiment"].get("experiment_name"):
        log_dir_name += f"_{agent_cfg['agent']['experiment']['experiment_name']}"
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir_name
    log_dir = os.path.join(log_root_path, log_dir_name)
    env_cfg.log_dir = log_dir
    print(f"[INFO] Logging experiment in directory: {log_dir}")

    pool_sync: LiveSelfPlayPoolSync | None = None
    pool_metadata = None
    if args_cli.self_play and hasattr(env_cfg, "fighter_a"):
        pool_metadata = {
            "framework": args_cli.ml_framework,
            "algorithm": algorithm.upper(),
            "task": args_cli.task,
            "seed": args_cli.seed,
            "reward_version": "privileged_phase_amp_league_recovery_v20_stability_prior",
            "league_role": args_cli.league_role,
            "config_hash": hashlib.sha256(
                json.dumps(
                    {
                        "fighter_a": env_cfg.fighter_a.robot_name,
                        "fighter_b": env_cfg.fighter_b.robot_name,
                        "action_spaces": env_cfg.action_spaces,
                        "observation_spaces": env_cfg.observation_spaces,
                        "rewards": vars(env_cfg.rewards),
                        "contact": vars(env_cfg.contact),
                        "perturbations": vars(env_cfg.perturbations),
                        "residual_locomotion": vars(env_cfg.residual_locomotion),
                        "motion_prior": vars(env_cfg.motion_prior),
                        "adr": vars(env_cfg.adr),
                        "pbt": vars(env_cfg.pbt),
                        "observations": vars(env_cfg.observations_cfg),
                        "self_play": vars(env_cfg.self_play),
                        "curriculum": vars(env_cfg.curriculum),
                        "diagnostics": vars(env_cfg.diagnostics),
                        "launch_preset": args_cli.launch_preset,
                    },
                    sort_keys=True,
                    default=str,
                ).encode("utf-8")
            ).hexdigest()[:16],
            "agents": {
                "fighter_a": {
                    "side": "fighter_a",
                    "robot": env_cfg.fighter_a.robot_name,
                    "action_dim": env_cfg.action_spaces["fighter_a"],
                    "obs_dim": env_cfg.observation_spaces["fighter_a"],
                },
                "fighter_b": {
                    "side": "fighter_b",
                    "robot": env_cfg.fighter_b.robot_name,
                    "action_dim": env_cfg.action_spaces["fighter_b"],
                    "obs_dim": env_cfg.observation_spaces["fighter_b"],
                },
            },
        }

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.curriculum_start_step > 0 and hasattr(env.unwrapped, "common_step_counter"):
        env.unwrapped.common_step_counter = int(args_cli.curriculum_start_step)
        print(f"[INFO] Initialized curriculum step counter to {args_cli.curriculum_start_step}")
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm == "ppo":
        env = multi_agent_to_single_agent(env)

    env = maybe_wrap_residual_locomotion(env, env_cfg, args_cli)
    env = maybe_wrap_historical_opponent(env, env_cfg, log_dir, args_cli)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
    runner = Runner(env, agent_cfg)
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        resume_path = _adapt_checkpoint_observation_space(resume_path, env_cfg, algorithm, log_dir)
        print(f"[INFO] Loading checkpoint: {resume_path}")
        if is_locomotion_warmstart_checkpoint(resume_path):
            loaded = apply_locomotion_warmstart(runner.agent, resume_path)
            if loaded:
                print(f"[INFO] Loaded locomotion warm-start modules: {', '.join(loaded)}")
            else:
                print("[WARN] Locomotion warm-start module lookup failed; falling back to skrl checkpoint loader.")
                runner.agent.load(resume_path)
        else:
            runner.agent.load(resume_path)

    supervisor = None
    if args_cli.self_play:
        supervisor = SelfPlayTrainingSupervisor(
            pool_dir=args_cli.pool_dir,
            checkpoint_dir=checkpoint_dir_from_log_dir(log_dir),
            snapshot_interval=args_cli.snapshot_interval,
            metadata=pool_metadata,
            promotion_min_proof_impact=getattr(env_cfg.self_play, "promotion_min_proof_impact", 0.0),
            promotion_min_health_score=getattr(env_cfg.self_play, "promotion_min_health_score", -1.0e9),
            promotion_bootstrap_count=getattr(env_cfg.self_play, "promotion_bootstrap_count", 1),
        )
        if args_cli.pool_sync_interval_s > 0.0:
            pool_sync = LiveSelfPlayPoolSync(supervisor, interval_s=args_cli.pool_sync_interval_s)
            pool_sync.start()

    start = time.time()
    final_sync_added = 0
    try:
        runner.run()
    finally:
        if pool_sync is not None:
            final_sync_added = pool_sync.stop()
        elif supervisor is not None:
            final_sync_added = supervisor.sync_checkpoints()
    print(f"[INFO] Training time: {time.time() - start:.2f} s")

    if args_cli.self_play:
        print(
            "[INFO] Self-play pool synchronized. "
            f"Added {final_sync_added} checkpoint(s) to {Path(args_cli.pool_dir).resolve()}"
        )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
