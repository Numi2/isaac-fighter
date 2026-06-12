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
    "--launch_preset",
    type=str,
    default="fast_contact_bootstrap",
    choices=["fast_contact_bootstrap", "full_fight_self_play"],
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


def _apply_launch_preset(env_cfg, agent_cfg: dict, preset: str) -> None:  # noqa: ANN001
    if not hasattr(env_cfg, "fighter_a"):
        return
    refresh_spaces = False
    if preset == "fast_contact_bootstrap":
        env_cfg.fighter_a.robot_name = "g1_29dof"
        env_cfg.fighter_b.robot_name = "g1_29dof"
        refresh_spaces = True
        env_cfg.decimation = 3
        env_cfg.sim.render_interval = env_cfg.decimation
        env_cfg.episode_length_s = 6.5
        env_cfg.arena.radius = 1.65
        env_cfg.rules.knockout_grace_s = 0.55
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
        env_cfg.fighter_a.action_scale = 0.20
        env_cfg.fighter_b.action_scale = 0.20
        env_cfg.fighter_a.action_smoothing = 0.35
        env_cfg.fighter_b.action_smoothing = 0.35
        env_cfg.contact.useful_contact_distance = 1.45
        env_cfg.contact.attack_memory_s = 0.65
        env_cfg.contact.fall_credit_min_attack = 0.08
        env_cfg.curriculum.enabled = True
        env_cfg.curriculum.standing_warmup_s = max(float(env_cfg.curriculum.standing_warmup_s), 1.50)
        env_cfg.curriculum.no_engagement_timeout_s = 4.5
        env_cfg.curriculum.no_engagement_grace_s = 2.5
        env_cfg.curriculum.proxy_gain_anneal_steps = min(int(env_cfg.curriculum.proxy_gain_anneal_steps), 20_000)
        env_cfg.curriculum.min_proxy_gain = max(float(env_cfg.curriculum.min_proxy_gain), 0.20)
        env_cfg.self_play.opponent_update_interval = min(int(env_cfg.self_play.opponent_update_interval), 160)
        env_cfg.self_play.live_self_play_fraction = max(float(env_cfg.self_play.live_self_play_fraction), 0.45)
        env_cfg.rewards.contact_intent = max(float(env_cfg.rewards.contact_intent), 2.8)
        env_cfg.rewards.standing_height = max(float(env_cfg.rewards.standing_height), 9.0)
        env_cfg.rewards.support_contact = max(float(env_cfg.rewards.support_contact), 5.0)
        env_cfg.rewards.low_base_height = max(float(env_cfg.rewards.low_base_height), 35.0)
        env_cfg.rewards.waist_action = max(float(env_cfg.rewards.waist_action), 4.0)
        env_cfg.rewards.locomotion_drive = max(float(env_cfg.rewards.locomotion_drive), 3.2)
        env_cfg.rewards.attack_momentum = max(float(env_cfg.rewards.attack_momentum), 3.4)
        env_cfg.rewards.drive_pressure = max(float(env_cfg.rewards.drive_pressure), 6.2)
        env_cfg.rewards.support_break_pressure = max(float(env_cfg.rewards.support_break_pressure), 7.2)
        env_cfg.rewards.opponent_fall = max(float(env_cfg.rewards.opponent_fall), 22.0)
        env_cfg.rewards.opponent_knockdown = max(float(env_cfg.rewards.opponent_knockdown), 36.0)
        env_cfg.rewards.impact_self_destabilization = max(float(env_cfg.rewards.impact_self_destabilization), 18.0)
        env_cfg.rewards.posture_instability = max(float(env_cfg.rewards.posture_instability), 7.0)
        env_cfg.rewards.self_fall = max(float(env_cfg.rewards.self_fall), 45.0)
        env_cfg.rewards.energy = min(float(env_cfg.rewards.energy), 0.010)
        env_cfg.rewards.jitter = min(float(env_cfg.rewards.jitter), 0.08)
        env_cfg.diagnostics.reward_terms_interval = max(int(env_cfg.diagnostics.reward_terms_interval), 64)
        agent_cfg["agent"]["rollouts"] = 16
        agent_cfg["agent"]["learning_epochs"] = min(int(agent_cfg["agent"]["learning_epochs"]), 3)
        agent_cfg["agent"]["mini_batches"] = min(int(agent_cfg["agent"]["mini_batches"]), 4)
        agent_cfg["agent"]["entropy_loss_scale"] = max(float(agent_cfg["agent"]["entropy_loss_scale"]), 0.008)
        agent_cfg["agent"]["experiment"]["write_interval"] = max(
            int(agent_cfg["agent"]["experiment"]["write_interval"]), 200
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
    _apply_launch_preset(env_cfg, agent_cfg, args_cli.launch_preset)
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    if args_cli.self_play and args_cli.snapshot_interval:
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
            "reward_version": "stance_first_v7_fast8k",
            "config_hash": hashlib.sha256(
                json.dumps(
                    {
                        "fighter_a": env_cfg.fighter_a.robot_name,
                        "fighter_b": env_cfg.fighter_b.robot_name,
                        "action_spaces": env_cfg.action_spaces,
                        "observation_spaces": env_cfg.observation_spaces,
                        "rewards": vars(env_cfg.rewards),
                        "contact": vars(env_cfg.contact),
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
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm == "ppo":
        env = multi_agent_to_single_agent(env)

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
            f"[INFO] Self-play pool synchronized. Added {final_sync_added} checkpoint(s) to {Path(args_cli.pool_dir).resolve()}"
        )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
