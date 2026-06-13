#!/usr/bin/env python3
# ruff: noqa: E402,I001
# Copyright (c) 2026, Isaac Fight contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Export AMP discriminator features from policy rollouts in Isaac Lab."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="GhostFighter-Unitree-1v1-Direct-v0")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--algorithm", type=str, default="IPPO", choices=["IPPO", "MAPPO", "PPO"])
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=512)
parser.add_argument("--sample_interval", type=int, default=4)
parser.add_argument("--max_samples", type=int, default=200_000)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--robot", type=str, default="g1_29dof")
parser.add_argument("--agents", choices=["all", "fighter_a", "fighter_b"], default="all")
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax", "jax-numpy"])
parser.add_argument("--residual_locomotion_checkpoint", type=str, default=None)
parser.add_argument("--residual_base_action_scale", type=float, default=1.0)
parser.add_argument("--residual_action_scale", type=float, default=0.08)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
else:
    from skrl.utils.runner.jax import Runner

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaac_fight.tasks  # noqa: F401
from isaac_fight.motion_prior.amp import AMP_FEATURE_SCHEMA
from isaac_fight.tasks.direct.unitree_1v1.fighter_ids import FIGHTER_A, FIGHTER_B
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
    if hasattr(env_cfg, "fighter_a"):
        env_cfg.fighter_a.robot_name = args_cli.robot
        env_cfg.fighter_b.robot_name = args_cli.robot
        if hasattr(env_cfg, "__post_init__"):
            env_cfg.__post_init__()
    if args_cli.residual_locomotion_checkpoint and hasattr(env_cfg, "residual_locomotion"):
        env_cfg.residual_locomotion.enabled = True
        env_cfg.residual_locomotion.checkpoint_path = args_cli.residual_locomotion_checkpoint
        env_cfg.residual_locomotion.base_action_scale = args_cli.residual_base_action_scale
        env_cfg.residual_locomotion.residual_action_scale = args_cli.residual_action_scale

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = maybe_wrap_residual_locomotion(env, env_cfg, args_cli)
    raw_env = env.unwrapped
    wrapped = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
    runner = Runner(wrapped, agent_cfg)
    if args_cli.checkpoint:
        runner.agent.load(retrieve_file_path(args_cli.checkpoint))

    agents = (FIGHTER_A, FIGHTER_B) if args_cli.agents == "all" else (args_cli.agents,)
    chunks: list[torch.Tensor] = []
    obs, _ = wrapped.reset()
    for step in range(int(args_cli.steps)):
        actions = runner.agent.act(obs, timestep=step, timesteps=args_cli.steps)[0]
        obs, _, terminated, truncated, _ = wrapped.step(actions)
        if step % max(1, int(args_cli.sample_interval)) == 0:
            chunks.extend(raw_env._motion_prior_amp_features(agent).detach().cpu() for agent in agents)
        if bool((terminated[FIGHTER_A] | truncated[FIGHTER_A]).all().item()):
            break

    features = torch.cat(chunks, dim=0) if chunks else torch.empty(0, 0)
    if args_cli.max_samples > 0 and features.shape[0] > args_cli.max_samples:
        order = torch.randperm(features.shape[0])[: args_cli.max_samples]
        features = features.index_select(0, order)

    output = Path(args_cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "isaac_fight.amp_rollout_features.v1",
            "features": features,
            "feature_schema": AMP_FEATURE_SCHEMA,
            "robot": args_cli.robot,
            "task": args_cli.task,
            "checkpoint": args_cli.checkpoint,
            "residual_locomotion_checkpoint": args_cli.residual_locomotion_checkpoint,
            "num_envs": args_cli.num_envs,
            "steps": args_cli.steps,
            "sample_interval": args_cli.sample_interval,
            "agents": agents,
        },
        output,
    )
    print(f"[INFO] Wrote {features.shape[0]} AMP rollout features to {output.resolve()}")
    wrapped.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
