#!/usr/bin/env python3
# Copyright (c) 2026, Isaac Fight contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Run a TorchScript-policy tournament from an Isaac Fight opponent pool."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate Isaac Fight policy pool tournaments.")
parser.add_argument("--task", type=str, default="GhostFighter-Unitree-1v1-Direct-v0")
parser.add_argument("--pool_dir", type=str, required=True)
parser.add_argument("--rounds", type=int, default=16)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--output", type=str, default="logs/tournaments/latest.json")
parser.add_argument("--max_policies", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils.hydra import hydra_task_config

import isaac_fight.tasks  # noqa: F401
from isaac_fight.tasks.direct.unitree_1v1.elo import EloTable
from isaac_fight.tasks.direct.unitree_1v1.fighter_ids import FIGHTER_A, FIGHTER_B
from isaac_fight.tasks.direct.unitree_1v1.opponent_pool import OpponentPool
from isaac_fight.tasks.direct.unitree_1v1.self_play import TorchScriptPolicyBackend


@hydra_task_config(args_cli.task, "skrl_ippo_cfg_entry_point")
def main(env_cfg, agent_cfg):  # noqa: ANN001, ARG001
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    pool = OpponentPool(args_cli.pool_dir)
    policies = [p for p in pool.policies if "torchscript" in p.tags and Path(p.checkpoint_path).exists()]
    policies = policies[-args_cli.max_policies :]
    if len(policies) < 2:
        raise RuntimeError("Tournament needs at least two TorchScript policies in the pool.")

    env = gym.make(args_cli.task, cfg=env_cfg)
    elo = EloTable()
    results = []
    device = args_cli.device or "cuda:0"

    backends = {p.policy_id: TorchScriptPolicyBackend(p.checkpoint_path, device=device) for p in policies}
    for a, b in itertools.combinations(policies, 2):
        for round_idx in range(args_cli.rounds):
            obs, _ = env.reset()
            done = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=env.unwrapped.device)
            total_reward = {FIGHTER_A: 0.0, FIGHTER_B: 0.0}
            while not bool(done.all().item()):
                actions = {
                    FIGHTER_A: backends[a.policy_id].act(obs[FIGHTER_A]),
                    FIGHTER_B: backends[b.policy_id].act(obs[FIGHTER_B]),
                }
                obs, rewards, terminated, truncated, extras = env.step(actions)
                done = terminated[FIGHTER_A] | truncated[FIGHTER_A]
                total_reward[FIGHTER_A] += float(rewards[FIGHTER_A].mean().item())
                total_reward[FIGHTER_B] += float(rewards[FIGHTER_B].mean().item())
            winner = int(env.unwrapped._winner[0].item())
            if winner == 1:
                result_a = 1.0
            elif winner == 2:
                result_a = 0.0
            else:
                result_a = 0.5
            ra, rb = elo.update(a.policy_id, b.policy_id, result_a)
            results.append(
                {
                    "round": round_idx,
                    "policy_a": a.policy_id,
                    "policy_b": b.policy_id,
                    "winner": winner,
                    "draw": winner == 0,
                    "rating_a": ra,
                    "rating_b": rb,
                    "reward_a": total_reward[FIGHTER_A],
                    "reward_b": total_reward[FIGHTER_B],
                    "duration_s": float(env.unwrapped.episode_length_buf[0].item() * env.unwrapped.step_dt),
                    "contact_force_a": float(env.unwrapped._contact_force[FIGHTER_A][0].item()),
                    "contact_force_b": float(env.unwrapped._contact_force[FIGHTER_B][0].item()),
                    "energy_a": float(env.unwrapped._energy_ema[FIGHTER_A][0].item()),
                    "energy_b": float(env.unwrapped._energy_ema[FIGHTER_B][0].item()),
                }
            )
    payload = {"schema": "isaac_fight.tournament.v1", "elo": elo.to_dict(), "matches": results}
    output = Path(args_cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[INFO] Wrote tournament results to {output.resolve()}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
