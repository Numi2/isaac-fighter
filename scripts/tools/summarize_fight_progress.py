#!/usr/bin/env python3
# Copyright (c) 2026, Isaac Fight contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Summarize objective fight-behavior progress from Isaac Fight artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ScalarTrend:
    tag: str
    first_step: int
    first_value: float
    last_step: int
    last_value: float
    delta: float
    samples: int


DEFAULT_TAGS = (
    "Reward / Total reward (mean)",
    "Reward / Total reward (max)",
    "Reward / Total reward (min)",
    "Reward / Instantaneous reward (mean)",
    "Episode / Total timesteps (mean)",
    "Info / Combat/mean_useful_contact",
    "Info / Combat/mean_contact_intent",
    "Info / Combat/mean_candidate_body_contact_force",
    "Info / Combat/mean_opponent_contact_attribution",
    "Info / Combat/mean_real_opponent_contact_force",
    "Info / Combat/mean_ground_contact_force",
    "Info / Combat/mean_proxy_engagement",
    "Info / Combat/mean_eval_contact_force",
    "Info / Combat/mean_opponent_destabilization",
    "Info / Combat/mean_proof_contact",
    "Info / Combat/mean_proof_impact",
    "Info / Combat/mean_proof_destabilization",
    "Info / Combat/mean_opponent_knockdown_events",
    "Info / Combat/mean_proof_opponent_knockdown_events",
    "Info / Combat/mean_self_knockdown_events",
    "Info / Combat/mean_inactivity",
    "Info / Combat/mean_spin_without_contact",
    "Info / Combat/mean_score",
    "Info / fighter_a/Match/win_rate",
    "Info / fighter_b/Match/win_rate",
)


def _load_scalars(log_dir: Path, tags: tuple[str, ...]) -> dict[str, ScalarTrend]:
    if not log_dir.exists():
        raise FileNotFoundError(f"missing skrl log directory: {log_dir}")
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:
        raise RuntimeError("tensorboard is required to read skrl event files") from exc

    accumulator = EventAccumulator(str(log_dir))
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", ()))
    trends: dict[str, ScalarTrend] = {}
    for tag in tags:
        if tag not in available:
            continue
        values = accumulator.Scalars(tag)
        if not values:
            continue
        first = values[0]
        last = values[-1]
        trends[tag] = ScalarTrend(
            tag=tag,
            first_step=int(first.step),
            first_value=float(first.value),
            last_step=int(last.step),
            last_value=float(last.value),
            delta=float(last.value - first.value),
            samples=len(values),
        )
    return trends


def _summarize_tournament(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing tournament JSON: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    matches = payload.get("matches", [])
    if not matches:
        return {"matches": 0}
    wins_a = sum(int(match.get("wins_fighter_a", 1 if match.get("winner") == 1 else 0)) for match in matches)
    wins_b = sum(int(match.get("wins_fighter_b", 1 if match.get("winner") == 2 else 0)) for match in matches)
    draws = sum(int(match.get("draws", 1 if match.get("draw") else 0)) for match in matches)
    env_rollouts = sum(int(match.get("num_envs", 1)) for match in matches)
    return {
        "matches": len(matches),
        "env_rollouts": env_rollouts,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "draws": draws,
        "mean_duration_s": sum(float(m.get("duration_s_mean", m.get("duration_s", 0.0))) for m in matches) / len(matches),
        "mean_eval_contact_force": sum(
            float(m.get("eval_contact_force_fighter_a", m.get("sensor_contact_force_fighter_a", m.get("sensor_contact_force_a", 0.0))))
            + float(m.get("eval_contact_force_fighter_b", m.get("sensor_contact_force_fighter_b", m.get("sensor_contact_force_b", 0.0))))
            for m in matches
        )
        / (2.0 * len(matches)),
        "mean_candidate_body_contact_force": sum(
            float(m.get("candidate_body_contact_force_fighter_a", 0.0)) + float(m.get("candidate_body_contact_force_fighter_b", 0.0))
            for m in matches
        )
        / (2.0 * len(matches)),
        "mean_opponent_contact_attribution": sum(
            float(m.get("opponent_contact_attribution_fighter_a", 0.0)) + float(m.get("opponent_contact_attribution_fighter_b", 0.0))
            for m in matches
        )
        / (2.0 * len(matches)),
        "mean_proof_impact": sum(float(m.get("proof_impact_fighter_a", 0.0)) + float(m.get("proof_impact_fighter_b", 0.0)) for m in matches)
        / (2.0 * len(matches)),
    }


def _summarize_replay(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing replay JSONL: {path}")
    steps = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("type") == "step":
            steps.append(record.get("payload", {}))
    if not steps:
        return {"steps": 0}
    contact_samples = []
    candidate_samples = []
    attribution_samples = []
    proof_samples = []
    knockdowns = 0
    for step in steps:
        for agent in ("fighter_a", "fighter_b"):
            fighter = step.get(agent, {})
            candidate_samples.append(float(fighter.get("candidate_body_contact_force", 0.0)))
            attribution_samples.append(float(fighter.get("opponent_contact_attribution", 0.0)))
            contact_samples.append(float(fighter.get("eval_contact_force", fighter.get("sensor_contact_force", fighter.get("contact_force", 0.0)))))
            proof_samples.append(float(fighter.get("proof_impact", 0.0)))
            knockdowns += int(bool(fighter.get("knockdown", False)))
    return {
        "steps": len(steps),
        "last_time_s": float(steps[-1].get("time_s", 0.0)),
        "mean_candidate_body_contact_force": sum(candidate_samples) / max(len(candidate_samples), 1),
        "mean_opponent_contact_attribution": sum(attribution_samples) / max(len(attribution_samples), 1),
        "mean_eval_contact_force": sum(contact_samples) / max(len(contact_samples), 1),
        "mean_proof_impact": sum(proof_samples) / max(len(proof_samples), 1),
        "knockdown_frames": knockdowns,
        "final_winner": int(steps[-1].get("winner", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Isaac Fight learning and fight-behavior metrics.")
    parser.add_argument("--log_dir", type=Path, help="skrl experiment directory containing TensorBoard event files.")
    parser.add_argument("--tournament", type=Path, help="Optional tournament JSON output.")
    parser.add_argument("--replay", type=Path, help="Optional replay JSONL output.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of Markdown.")
    args = parser.parse_args()

    report: dict[str, Any] = {"schema": "isaac_fight.progress_report.v1"}
    if args.log_dir:
        report["log_dir"] = str(args.log_dir)
        report["trends"] = {tag: asdict(trend) for tag, trend in _load_scalars(args.log_dir, DEFAULT_TAGS).items()}
    if args.tournament:
        report["tournament"] = _summarize_tournament(args.tournament)
    if args.replay:
        report["replay"] = _summarize_replay(args.replay)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    print("# Isaac Fight Progress")
    for tag, trend in report.get("trends", {}).items():
        print(f"- {tag}: {trend['first_value']:.4g} -> {trend['last_value']:.4g} (delta {trend['delta']:.4g})")
    if "tournament" in report:
        print(f"- tournament: {report['tournament']}")
    if "replay" in report:
        print(f"- replay: {report['replay']}")


if __name__ == "__main__":
    main()
