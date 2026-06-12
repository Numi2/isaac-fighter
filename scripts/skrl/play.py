#!/usr/bin/env python3
# Copyright (c) 2026, Isaac Fight contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Play GhostFighter with skrl checkpoints."""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play Isaac Fight policies.")
parser.add_argument("--task", type=str, default="GhostFighter-Unitree-1v1-Direct-v0")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--algorithm", type=str, default="IPPO", choices=["IPPO", "MAPPO", "PPO"])
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=5000)
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax", "jax-numpy"])
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
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
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
    runner = Runner(env, agent_cfg)
    if args_cli.checkpoint:
        runner.agent.load(retrieve_file_path(args_cli.checkpoint))
    obs, _ = env.reset()
    for _ in range(args_cli.steps):
        actions = runner.agent.act(obs, timestep=0, timesteps=args_cli.steps)[0]
        obs, _, terminated, truncated, _ = env.step(actions)
        if hasattr(env, "render"):
            env.render()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
