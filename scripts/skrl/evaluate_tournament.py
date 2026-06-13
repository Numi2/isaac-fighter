#!/usr/bin/env python3
# ruff: noqa: E402,I001
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
parser.add_argument("--promote_to_league", action="store_true", default=False)
parser.add_argument("--promotion_min_games", type=int, default=8)
parser.add_argument("--promotion_min_health_score", type=float, default=-0.05)
parser.add_argument("--promotion_min_win_rate", type=float, default=0.15)
parser.add_argument("--promotion_max_self_fall_rate", type=float, default=0.65)
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
    policy_stats = {policy.policy_id: _new_policy_stats() for policy in policies}
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
                _accumulate_policy_stats(
                    policy_stats[policy_on_a.policy_id],
                    env.unwrapped,
                    FIGHTER_A,
                    result_values,
                )
                _accumulate_policy_stats(
                    policy_stats[policy_on_b.policy_id],
                    env.unwrapped,
                    FIGHTER_B,
                    1.0 - result_values,
                )
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
                        "fall_events_fighter_a": float(env.unwrapped._fall_events[FIGHTER_A].sum().item()),
                        "fall_events_fighter_b": float(env.unwrapped._fall_events[FIGHTER_B].sum().item()),
                        "knockdown_events_fighter_a": float(env.unwrapped._knockdown_events[FIGHTER_A].sum().item()),
                        "knockdown_events_fighter_b": float(env.unwrapped._knockdown_events[FIGHTER_B].sum().item()),
                        "result_policy_fighter_a_mean": float(result_values.mean().item()),
                        "rating_policy_fighter_a": ra,
                        "rating_policy_fighter_b": rb,
                        "reward_fighter_a": total_reward[FIGHTER_A],
                        "reward_fighter_b": total_reward[FIGHTER_B],
                        "duration_s_mean": float((env.unwrapped.episode_length_buf.float() * env.unwrapped.step_dt).mean().item()),
                        "candidate_body_contact_force_fighter_a": float(env.unwrapped._candidate_body_contact_force[FIGHTER_A].mean().item()),
                        "candidate_body_contact_force_fighter_b": float(env.unwrapped._candidate_body_contact_force[FIGHTER_B].mean().item()),
                        "opponent_contact_attribution_fighter_a": float(env.unwrapped._opponent_contact_attribution[FIGHTER_A].mean().item()),
                        "opponent_contact_attribution_fighter_b": float(env.unwrapped._opponent_contact_attribution[FIGHTER_B].mean().item()),
                        "real_opponent_contact_force_fighter_a": float(env.unwrapped._real_opponent_contact_force[FIGHTER_A].mean().item()),
                        "real_opponent_contact_force_fighter_b": float(env.unwrapped._real_opponent_contact_force[FIGHTER_B].mean().item()),
                        "ground_contact_force_fighter_a": float(env.unwrapped._ground_contact_force[FIGHTER_A].mean().item()),
                        "ground_contact_force_fighter_b": float(env.unwrapped._ground_contact_force[FIGHTER_B].mean().item()),
                        "proxy_engagement_fighter_a": float(env.unwrapped._proxy_engagement[FIGHTER_A].mean().item()),
                        "proxy_engagement_fighter_b": float(env.unwrapped._proxy_engagement[FIGHTER_B].mean().item()),
                        "training_contact_force_fighter_a": float(env.unwrapped._training_contact_force[FIGHTER_A].mean().item()),
                        "training_contact_force_fighter_b": float(env.unwrapped._training_contact_force[FIGHTER_B].mean().item()),
                        "eval_contact_force_fighter_a": float(env.unwrapped._eval_contact_force[FIGHTER_A].mean().item()),
                        "eval_contact_force_fighter_b": float(env.unwrapped._eval_contact_force[FIGHTER_B].mean().item()),
                        "attack_momentum_fighter_a": float(env.unwrapped._attack_momentum[FIGHTER_A].mean().item()),
                        "attack_momentum_fighter_b": float(env.unwrapped._attack_momentum[FIGHTER_B].mean().item()),
                        "strike_speed_fighter_a": float(env.unwrapped._strike_speed[FIGHTER_A].mean().item()),
                        "strike_speed_fighter_b": float(env.unwrapped._strike_speed[FIGHTER_B].mean().item()),
                        "destabilizing_impact_fighter_a": float(env.unwrapped._destabilizing_impact[FIGHTER_A].mean().item()),
                        "destabilizing_impact_fighter_b": float(env.unwrapped._destabilizing_impact[FIGHTER_B].mean().item()),
                        "topple_pressure_fighter_a": float(env.unwrapped._topple_pressure[FIGHTER_A].mean().item()),
                        "topple_pressure_fighter_b": float(env.unwrapped._topple_pressure[FIGHTER_B].mean().item()),
                        "drive_pressure_fighter_a": float(env.unwrapped._drive_pressure[FIGHTER_A].mean().item()),
                        "drive_pressure_fighter_b": float(env.unwrapped._drive_pressure[FIGHTER_B].mean().item()),
                        "support_break_pressure_fighter_a": float(env.unwrapped._support_break_pressure[FIGHTER_A].mean().item()),
                        "support_break_pressure_fighter_b": float(env.unwrapped._support_break_pressure[FIGHTER_B].mean().item()),
                        "proof_impact_fighter_a": float(env.unwrapped._proof_impact[FIGHTER_A].mean().item()),
                        "proof_impact_fighter_b": float(env.unwrapped._proof_impact[FIGHTER_B].mean().item()),
                        "energy_fighter_a": float(env.unwrapped._energy_ema[FIGHTER_A].mean().item()),
                        "energy_fighter_b": float(env.unwrapped._energy_ema[FIGHTER_B].mean().item()),
                    }
                )
    policy_summaries = {
        policy.policy_id: _summarize_policy_stats(policy_stats[policy.policy_id], policy)
        for policy in policies
    }
    if args_cli.promote_to_league:
        _apply_league_promotions(pool, policy_summaries)
    payload = {
        "schema": "isaac_fight.tournament.v2",
        "elo": elo.to_dict(),
        "policy_summaries": policy_summaries,
        "promotion_config": {
            "min_games": args_cli.promotion_min_games,
            "min_health_score": args_cli.promotion_min_health_score,
            "min_win_rate": args_cli.promotion_min_win_rate,
            "max_self_fall_rate": args_cli.promotion_max_self_fall_rate,
        },
        "matches": results,
    }
    output = Path(args_cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[INFO] Wrote tournament results to {output.resolve()}")
    env.close()


def _new_policy_stats() -> dict[str, float]:
    return {
        "games": 0.0,
        "wins": 0.0,
        "losses": 0.0,
        "draws": 0.0,
        "health_score_sum": 0.0,
        "upright_seconds_sum": 0.0,
        "feet_ground_support_sum": 0.0,
        "caused_knockdowns_sum": 0.0,
        "mutual_falls_sum": 0.0,
        "torso_first_contacts_sum": 0.0,
        "self_falls_sum": 0.0,
        "proof_impact_sum": 0.0,
        "support_break_pressure_sum": 0.0,
    }


def _accumulate_policy_stats(
    stats: dict[str, float],
    env,  # noqa: ANN001
    agent_id: str,
    result_values: torch.Tensor,
) -> None:
    games = float(result_values.numel())
    stats["games"] += games
    stats["wins"] += float((result_values == 1.0).sum().item())
    stats["losses"] += float((result_values == 0.0).sum().item())
    stats["draws"] += float((result_values == 0.5).sum().item())
    stats["health_score_sum"] += games * _episode_per_step_mean(env, agent_id, "combat_health_score")
    stats["upright_seconds_sum"] += games * _episode_per_step_mean(env, agent_id, "combat_health_upright_seconds")
    stats["feet_ground_support_sum"] += games * _episode_per_step_mean(env, agent_id, "combat_health_feet_ground_support")
    stats["caused_knockdowns_sum"] += games * _episode_sum_mean(env, agent_id, "combat_health_caused_knockdowns")
    stats["mutual_falls_sum"] += games * _episode_sum_mean(env, agent_id, "combat_health_mutual_falls")
    stats["torso_first_contacts_sum"] += games * _episode_sum_mean(
        env,
        agent_id,
        "combat_health_torso_first_contacts",
    )
    stats["self_falls_sum"] += games * _episode_sum_mean(env, agent_id, "combat_self_fall_events")
    stats["proof_impact_sum"] += games * _episode_per_step_mean(env, agent_id, "combat_proof_impact")
    stats["support_break_pressure_sum"] += games * _episode_per_step_mean(env, agent_id, "combat_support_break_pressure")


def _episode_sum_mean(env, agent_id: str, name: str) -> float:  # noqa: ANN001
    values = env._episode_sums.get(agent_id, {}).get(name)
    if values is None:
        return 0.0
    return float(values.mean().item())


def _episode_per_step_mean(env, agent_id: str, name: str) -> float:  # noqa: ANN001
    values = env._episode_sums.get(agent_id, {}).get(name)
    if values is None:
        return 0.0
    counts = torch.clamp(env._episode_counts[agent_id], min=1.0)
    return float((values / counts).mean().item())


def _summarize_policy_stats(stats: dict[str, float], policy) -> dict[str, float | str | bool]:  # noqa: ANN001
    games = max(float(stats["games"]), 1.0)
    win_rate = stats["wins"] / games
    loss_rate = stats["losses"] / games
    draw_rate = stats["draws"] / games
    caused_knockdown_rate = stats["caused_knockdowns_sum"] / games
    self_fall_rate = stats["self_falls_sum"] / games
    mutual_fall_rate = stats["mutual_falls_sum"] / games
    torso_first_rate = stats["torso_first_contacts_sum"] / games
    health_score = stats["health_score_sum"] / games
    support = stats["feet_ground_support_sum"] / games
    proof = stats["proof_impact_sum"] / games
    promotion_score = (
        4.0 * win_rate
        + 2.0 * health_score
        + 3.0 * caused_knockdown_rate
        + 0.75 * proof
        + 0.50 * support
        - 4.0 * self_fall_rate
        - 5.0 * mutual_fall_rate
        - 1.50 * torso_first_rate
    )
    eligible = (
        stats["games"] >= args_cli.promotion_min_games
        and health_score >= args_cli.promotion_min_health_score
        and win_rate >= args_cli.promotion_min_win_rate
        and self_fall_rate <= args_cli.promotion_max_self_fall_rate
    )
    broken = self_fall_rate > args_cli.promotion_max_self_fall_rate or health_score < args_cli.promotion_min_health_score
    return {
        "policy_id": policy.policy_id,
        "league_role": policy.league_role,
        "games": stats["games"],
        "win_rate": win_rate,
        "loss_rate": loss_rate,
        "draw_rate": draw_rate,
        "health_score": health_score,
        "upright_seconds": stats["upright_seconds_sum"] / games,
        "feet_ground_support": support,
        "caused_knockdown_rate": caused_knockdown_rate,
        "mutual_fall_rate": mutual_fall_rate,
        "torso_first_contact_rate": torso_first_rate,
        "self_fall_rate": self_fall_rate,
        "proof_impact": proof,
        "support_break_pressure": stats["support_break_pressure_sum"] / games,
        "promotion_score": promotion_score,
        "league_promotion_eligible": eligible,
        "league_suppressed": broken and not eligible,
    }


def _apply_league_promotions(pool: OpponentPool, policy_summaries: dict[str, dict]) -> None:
    for policy_id, summary in policy_summaries.items():
        promoted = bool(summary["league_promotion_eligible"])
        suppressed = bool(summary["league_suppressed"])
        tags = ["league_eval"]
        remove_tags = ["league_promoted", "league_suppressed"]
        if promoted:
            tags.append("league_promoted")
        elif suppressed:
            tags.append("league_suppressed")
        pool.update_policy_metadata(
            policy_id,
            metadata={
                "league_eval": summary,
                "league_status": "promoted" if promoted else "suppressed" if suppressed else "candidate",
            },
            add_tags=tags,
            remove_tags=remove_tags,
        )


if __name__ == "__main__":
    main()
    simulation_app.close()
