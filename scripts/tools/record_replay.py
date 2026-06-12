#!/usr/bin/env python3
# Copyright (c) 2026, Isaac Fight contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Record a match replay using skrl play mode."""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record an Isaac Fight JSONL replay.")
parser.add_argument("--task", type=str, default="GhostFighter-Unitree-1v1-Direct-v0")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--algorithm", type=str, default="IPPO", choices=["IPPO", "MAPPO", "PPO"])
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax", "jax-numpy"])
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = getattr(args_cli, "enable_cameras", False)
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
else:
    from skrl.utils.runner.jax import Runner

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaac_fight.tasks  # noqa: F401

agent_cfg_entry_point = "skrl_cfg_entry_point" if args_cli.algorithm.lower() == "ppo" else f"skrl_{args_cli.algorithm.lower()}_cfg_entry_point"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg, agent_cfg):  # noqa: ANN001
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.replay.enabled = True
    env_cfg.replay.path = args_cli.output
    env_cfg.replay.env_index = 0
    env_cfg.replay.interval = 1
    env = gym.make(args_cli.task, cfg=env_cfg)
    wrapped = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
    runner = Runner(wrapped, agent_cfg)
    if args_cli.checkpoint:
        runner.agent.load(retrieve_file_path(args_cli.checkpoint))
    obs, _ = wrapped.reset()
    for _ in range(args_cli.steps):
        actions = runner.agent.act(obs, timestep=0, timesteps=args_cli.steps)[0]
        obs, _, terminated, truncated, _ = wrapped.step(actions)
        if bool((terminated["fighter_a"] | truncated["fighter_a"]).all().item()):
            break
    wrapped.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
