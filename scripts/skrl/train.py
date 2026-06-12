#!/usr/bin/env python3
# Copyright (c) 2026, Isaac Fight contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Train GhostFighter with skrl IPPO/MAPPO and self-play pool tracking."""

from __future__ import annotations

import argparse
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
parser.add_argument("--self_play", action="store_true", default=True, help="Track policy versions in a self-play pool.")
parser.add_argument("--no_self_play", action="store_false", dest="self_play")
parser.add_argument("--historical_opponent", action="store_true", default=False, help="Freeze opponent actions from sampled TorchScript pool policies.")
parser.add_argument("--active_agent", type=str, default="fighter_a", choices=["fighter_a", "fighter_b"])
parser.add_argument("--pool_dir", type=str, default="policy_pool")
parser.add_argument("--snapshot_interval", type=int, default=50)
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
from packaging import version

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaac_fight.tasks  # noqa: F401
from isaac_fight.tasks.direct.unitree_1v1.self_play import (
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


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
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

    log_root_path = os.path.abspath(os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"]))
    log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    if agent_cfg["agent"]["experiment"].get("experiment_name"):
        log_dir_name += f"_{agent_cfg['agent']['experiment']['experiment_name']}"
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir_name
    log_dir = os.path.join(log_root_path, log_dir_name)
    env_cfg.log_dir = log_dir
    print(f"[INFO] Logging experiment in directory: {log_dir}")

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
        print(f"[INFO] Loading checkpoint: {resume_path}")
        runner.agent.load(resume_path)

    start = time.time()
    runner.run()
    print(f"[INFO] Training time: {time.time() - start:.2f} s")

    if args_cli.self_play:
        checkpoint_dir = checkpoint_dir_from_log_dir(log_dir)
        supervisor = SelfPlayTrainingSupervisor(
            pool_dir=args_cli.pool_dir,
            checkpoint_dir=checkpoint_dir,
            snapshot_interval=args_cli.snapshot_interval,
        )
        added = supervisor.sync_checkpoints()
        print(f"[INFO] Self-play pool synchronized. Added {added} checkpoint(s) to {Path(args_cli.pool_dir).resolve()}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
