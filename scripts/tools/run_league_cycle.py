#!/usr/bin/env python3
# Copyright (c) 2026, Isaac Fight contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Run a main/exploiter league cycle with evaluation-driven promotion."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


DEFAULT_ROLES = (
    "main",
    "shove_exploiter",
    "body_slam_exploiter",
    "balance_breaker",
    "recovery_specialist",
    "brace_defender",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--isaaclab_sh", default="./isaaclab.sh")
    parser.add_argument("--task", default="GhostFighter-Unitree-1v1-Direct-v0")
    parser.add_argument("--algorithm", default="IPPO")
    parser.add_argument("--launch_preset", default="fast_contact_bootstrap")
    parser.add_argument("--pool_dir", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--residual_locomotion_checkpoint", default="")
    parser.add_argument("--residual_base_action_scale", type=float, default=1.0)
    parser.add_argument("--residual_action_scale", type=float, default=0.08)
    parser.add_argument("--roles", nargs="+", default=DEFAULT_ROLES)
    parser.add_argument("--main_num_envs", type=int, default=4096)
    parser.add_argument("--exploiter_num_envs", type=int, default=1024)
    parser.add_argument("--max_iterations", type=int, default=1250)
    parser.add_argument("--snapshot_interval", type=int, default=128)
    parser.add_argument("--pool_sync_interval_s", type=float, default=60.0)
    parser.add_argument("--tournament_rounds", type=int, default=8)
    parser.add_argument("--tournament_num_envs", type=int, default=64)
    parser.add_argument("--tournament_max_policies", type=int, default=24)
    parser.add_argument("--tournament_output", default="logs/tournaments/league_cycle_latest.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--execute", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    commands = [_train_command(args, repo_root, role) for role in args.roles]
    commands.append(_tournament_command(args, repo_root))
    for command in commands:
        print(_format_command(command), flush=True)
        if args.execute:
            subprocess.run(command, check=True)


def _train_command(args: argparse.Namespace, repo_root: Path, role: str) -> list[str]:
    num_envs = int(args.main_num_envs if role == "main" else args.exploiter_num_envs)
    command = [
        args.isaaclab_sh,
        "-p",
        str(repo_root / "scripts" / "skrl" / "train.py"),
        "--task",
        args.task,
        "--algorithm",
        args.algorithm,
        "--launch_preset",
        args.launch_preset,
        "--league_role",
        role,
        "--num_envs",
        str(num_envs),
        "--max_iterations",
        str(int(args.max_iterations)),
        "--pool_dir",
        args.pool_dir,
        "--snapshot_interval",
        str(int(args.snapshot_interval)),
        "--pool_sync_interval_s",
        str(float(args.pool_sync_interval_s)),
        "--device",
        args.device,
    ]
    if args.checkpoint:
        command.extend(("--checkpoint", args.checkpoint))
    if args.residual_locomotion_checkpoint:
        command.extend(
            (
                "--residual_locomotion_checkpoint",
                args.residual_locomotion_checkpoint,
                "--residual_base_action_scale",
                str(float(args.residual_base_action_scale)),
                "--residual_action_scale",
                str(float(args.residual_action_scale)),
            )
        )
    if args.headless:
        command.append("--headless")
    return command


def _tournament_command(args: argparse.Namespace, repo_root: Path) -> list[str]:
    command = [
        args.isaaclab_sh,
        "-p",
        str(repo_root / "scripts" / "skrl" / "evaluate_tournament.py"),
        "--task",
        args.task,
        "--pool_dir",
        args.pool_dir,
        "--rounds",
        str(int(args.tournament_rounds)),
        "--num_envs",
        str(int(args.tournament_num_envs)),
        "--max_policies",
        str(int(args.tournament_max_policies)),
        "--promote_to_league",
        "--output",
        args.tournament_output,
        "--device",
        args.device,
    ]
    if args.headless:
        command.append("--headless")
    return command


def _format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


if __name__ == "__main__":
    main()
