#!/usr/bin/env python3
# Copyright (c) 2026, Isaac Fight contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Thin operator entrypoint for Unitree Velocity warm-start artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaac_fight.locomotion_bootstrap import (
    create_fight_warmstart,
    inspect_rsl_rl_checkpoint,
    sync_locomotion_artifact,
    sync_motion_prior_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect, sync, and convert Unitree Velocity rsl_rl checkpoints.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("checkpoint", type=Path)
    inspect_parser.add_argument("--robot", choices=["g1_29dof", "h1"])
    inspect_parser.add_argument("--source_task")

    sync_parser = subparsers.add_parser("sync")
    sync_parser.add_argument("checkpoint", type=Path)
    sync_parser.add_argument("--root", type=Path, default=Path("locomotion_bootstrap"))
    sync_parser.add_argument("--robot", choices=["g1_29dof", "h1"])
    sync_parser.add_argument("--source_task")
    sync_parser.add_argument("--export", action="append", default=[], type=Path)

    warmstart_parser = subparsers.add_parser("warmstart")
    warmstart_parser.add_argument("source_checkpoint", type=Path)
    warmstart_parser.add_argument("output", type=Path)
    warmstart_parser.add_argument("--robot", choices=["g1_29dof", "h1"])
    warmstart_parser.add_argument("--source_task")

    motion_parser = subparsers.add_parser("motion-prior")
    motion_parser.add_argument("motion_file", type=Path)
    motion_parser.add_argument("--root", type=Path, default=Path("locomotion_bootstrap"))
    motion_parser.add_argument("--robot", default="g1_29dof", choices=["g1_29dof", "h1"])
    motion_parser.add_argument("--source_task", default="Unitree-G1-29dof-Mimic")
    motion_parser.add_argument("--kind", default="unitree_g1_mimic_motion")

    args = parser.parse_args()
    if args.command == "inspect":
        info = inspect_rsl_rl_checkpoint(args.checkpoint, robot=args.robot, source_task=args.source_task)
        print(json.dumps(info.to_json(), indent=2, sort_keys=True))
    elif args.command == "sync":
        record = sync_locomotion_artifact(
            args.checkpoint,
            root=args.root,
            robot=args.robot,
            source_task=args.source_task,
            exports=args.export,
        )
        print(json.dumps(record, indent=2, sort_keys=True))
    elif args.command == "warmstart":
        report = create_fight_warmstart(
            args.source_checkpoint,
            args.output,
            robot=args.robot,
            source_task=args.source_task,
        )
        report.print_summary()
    elif args.command == "motion-prior":
        record = sync_motion_prior_artifact(
            args.motion_file,
            root=args.root,
            robot=args.robot,
            source_task=args.source_task,
            kind=args.kind,
        )
        print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
