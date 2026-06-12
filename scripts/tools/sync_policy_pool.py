#!/usr/bin/env python3
# Copyright (c) 2026, Isaac Fight contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Synchronize skrl training checkpoints into the Isaac Fight policy pool."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from isaac_fight.tasks.direct.unitree_1v1.self_play import SelfPlayTrainingSupervisor, checkpoint_dir_from_log_dir


def _sync_once(log_root: Path, pool_dir: Path, active_elo: float, snapshot_interval: int) -> int:
    added = 0
    for log_dir in sorted(p for p in log_root.glob("*") if p.is_dir()):
        checkpoint_dir = checkpoint_dir_from_log_dir(log_dir)
        if not checkpoint_dir.exists():
            continue
        supervisor = SelfPlayTrainingSupervisor(
            pool_dir=pool_dir,
            checkpoint_dir=checkpoint_dir,
            snapshot_interval=snapshot_interval,
            active_elo=active_elo,
        )
        added += supervisor.sync_checkpoints()
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync skrl checkpoints into a persistent Isaac Fight policy pool.")
    parser.add_argument("--log_root", type=Path, required=True, help="Directory containing skrl experiment log folders.")
    parser.add_argument("--pool_dir", type=Path, required=True, help="Policy pool directory to update.")
    parser.add_argument("--active_elo", type=float, default=1000.0)
    parser.add_argument("--snapshot_interval", type=int, default=50)
    parser.add_argument("--interval_s", type=float, default=0.0, help="Repeat interval. Use 0 for a single sync.")
    args = parser.parse_args()

    while True:
        added = _sync_once(args.log_root, args.pool_dir, args.active_elo, args.snapshot_interval)
        print(f"[INFO] Policy pool sync complete. Added {added} checkpoint(s).", flush=True)
        if args.interval_s <= 0:
            break
        time.sleep(args.interval_s)


if __name__ == "__main__":
    main()
