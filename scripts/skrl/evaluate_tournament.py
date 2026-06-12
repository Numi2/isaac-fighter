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
from isaac_fight.tasks.direct.unitree_1v1.self_play import SkrlCheckpointPolicyBackend, TorchScriptPolicyBackend


@hydra_task_config(args_cli.task, "skrl_ippo_cfg_entry_point")
def main(env_cfg, agent_cfg):  # noqa: ANN001, ARG001
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    pool = OpponentPool(args_cli.pool_dir)
    policies = [p for p in pool.policies if ({"torchscript", "skrl"} & set(p.tags)) and Path(p.checkpoint_path).exists()]
    policies = policies[-args_cli.max_policies :]
    if len(policies) < 2:
        raise RuntimeError("Tournament needs at least two skrl or TorchScript policies in the pool.")

    env = gym.make(args_cli.task, cfg=env_cfg)
    elo = EloTable()
    results = []
    device = args_cli.device or "cuda:0"
    for policy in policies:
        elo.ensure(policy.policy_id).rating = policy.elo

    def backend(policy, agent_id: str):  # noqa: ANN001
        if "torchscript" in policy.tags:
            return TorchScriptPolicyBackend(policy.checkpoint_path, device=device)
        return SkrlCheckpointPolicyBackend(policy.checkpoint_path, agent_id=agent_id, device=device)

    backends = {(p.policy_id, agent_id): backend(p, agent_id) for p in policies for agent_id in (FIGHTER_A, FIGHTER_B)}
    for a, b in itertools.combinations(policies, 2):
        for round_idx in range(args_cli.rounds):
            for policy_on_a, policy_on_b in ((a, b), (b, a)):
                obs, _ = env.reset()
                done = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=env.unwrapped.device)
                total_reward = {FIGHTER_A: 0.0, FIGHTER_B: 0.0}
                while not bool(done.all().item()):
                    actions = {
                        FIGHTER_A: backends[(policy_on_a.policy_id, FIGHTER_A)].act(obs[FIGHTER_A]),
                        FIGHTER_B: backends[(policy_on_b.policy_id, FIGHTER_B)].act(obs[FIGHTER_B]),
                    }
                    obs, rewards, terminated, truncated, extras = env.step(actions)
                    done = terminated[FIGHTER_A] | truncated[FIGHTER_A]
                    total_reward[FIGHTER_A] += float(rewards[FIGHTER_A].mean().item())
                    total_reward[FIGHTER_B] += float(rewards[FIGHTER_B].mean().item())
                winners = env.unwrapped._winner.detach()
                result_values = torch.where(
                    winners == 1,
                    torch.ones_like(winners, dtype=torch.float32),
                    torch.where(winners == 2, torch.zeros_like(winners, dtype=torch.float32), torch.full_like(winners, 0.5, dtype=torch.float32)),
                )
                ra, rb = elo.ensure(policy_on_a.policy_id).rating, elo.ensure(policy_on_b.policy_id).rating
                for result in result_values.cpu().tolist():
                    ra, rb = elo.update(policy_on_a.policy_id, policy_on_b.policy_id, float(result))
                    pool.update_result(policy_on_a.policy_id, float(result), elo=ra)
                    pool.update_result(policy_on_b.policy_id, float(1.0 - result), elo=rb)
                wins_a = int((winners == 1).sum().item())
                wins_b = int((winners == 2).sum().item())
                draws = int((winners == 0).sum().item())
                results.append(
                    {
                        "round": round_idx,
                        "policy_fighter_a": policy_on_a.policy_id,
                        "policy_fighter_b": policy_on_b.policy_id,
                        "fighter_a_robot": env_cfg.fighter_a.robot_name,
                        "fighter_b_robot": env_cfg.fighter_b.robot_name,
                        "num_envs": args_cli.num_envs,
                        "wins_fighter_a": wins_a,
                        "wins_fighter_b": wins_b,
                        "draws": draws,
                        "result_policy_fighter_a_mean": float(result_values.mean().item()),
                        "rating_policy_fighter_a": ra,
                        "rating_policy_fighter_b": rb,
                        "reward_fighter_a": total_reward[FIGHTER_A],
                        "reward_fighter_b": total_reward[FIGHTER_B],
                        "duration_s_mean": float((env.unwrapped.episode_length_buf.float() * env.unwrapped.step_dt).mean().item()),
                        "real_opponent_contact_force_fighter_a": float(env.unwrapped._real_opponent_contact_force[FIGHTER_A].mean().item()),
                        "real_opponent_contact_force_fighter_b": float(env.unwrapped._real_opponent_contact_force[FIGHTER_B].mean().item()),
                        "ground_contact_force_fighter_a": float(env.unwrapped._ground_contact_force[FIGHTER_A].mean().item()),
                        "ground_contact_force_fighter_b": float(env.unwrapped._ground_contact_force[FIGHTER_B].mean().item()),
                        "proxy_engagement_fighter_a": float(env.unwrapped._proxy_engagement[FIGHTER_A].mean().item()),
                        "proxy_engagement_fighter_b": float(env.unwrapped._proxy_engagement[FIGHTER_B].mean().item()),
                        "eval_contact_force_fighter_a": float(env.unwrapped._eval_contact_force[FIGHTER_A].mean().item()),
                        "eval_contact_force_fighter_b": float(env.unwrapped._eval_contact_force[FIGHTER_B].mean().item()),
                        "proof_impact_fighter_a": float(env.unwrapped._proof_impact[FIGHTER_A].mean().item()),
                        "proof_impact_fighter_b": float(env.unwrapped._proof_impact[FIGHTER_B].mean().item()),
                        "energy_fighter_a": float(env.unwrapped._energy_ema[FIGHTER_A].mean().item()),
                        "energy_fighter_b": float(env.unwrapped._energy_ema[FIGHTER_B].mean().item()),
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
