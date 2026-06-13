#!/usr/bin/env python3
# ruff: noqa: E402,I001
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
parser.add_argument(
    "--curriculum_start_step",
    type=int,
    default=0,
    help="Initialize the environment curriculum/common step counter before playback.",
)
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax", "jax-numpy"])
parser.add_argument("--residual_locomotion_checkpoint", type=str, default=None)
parser.add_argument("--residual_base_action_scale", type=float, default=1.0)
parser.add_argument("--residual_action_scale", type=float, default=0.35)
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
from isaac_fight.tasks.direct.unitree_1v1.self_play import maybe_wrap_residual_locomotion

agent_cfg_entry_point = (
    "skrl_cfg_entry_point"
    if args_cli.algorithm.lower() == "ppo"
    else f"skrl_{args_cli.algorithm.lower()}_cfg_entry_point"
)


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg, agent_cfg):  # noqa: ANN001
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.residual_locomotion_checkpoint and hasattr(env_cfg, "residual_locomotion"):
        env_cfg.residual_locomotion.enabled = True
        env_cfg.residual_locomotion.checkpoint_path = args_cli.residual_locomotion_checkpoint
        env_cfg.residual_locomotion.base_action_scale = args_cli.residual_base_action_scale
        env_cfg.residual_locomotion.residual_action_scale = args_cli.residual_action_scale
    env = gym.make(args_cli.task, cfg=env_cfg)
    if args_cli.curriculum_start_step > 0 and hasattr(env.unwrapped, "common_step_counter"):
        env.unwrapped.common_step_counter = int(args_cli.curriculum_start_step)
        print(f"[INFO] Initialized curriculum step counter to {args_cli.curriculum_start_step}")
    env = maybe_wrap_residual_locomotion(env, env_cfg, args_cli)
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
