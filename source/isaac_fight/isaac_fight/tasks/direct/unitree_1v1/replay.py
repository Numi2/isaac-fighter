# Copyright (c) 2026, Isaac Fight contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""JSONL match replay recording."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ReplayHeader:
    schema: str = "isaac_fight.match_replay.v1"
    task: str = "GhostFighter-Unitree-1v1-Direct-v0"
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class MatchReplayRecorder:
    """Streaming JSONL replay writer.

    The replay stores compact numeric summaries suitable for match inspection and later visualization. It is not an
    imitation-learning dataset by default.
    """

    def __init__(self, path: str | Path, header: ReplayHeader | None = None, flush_every: int = 128):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_every = max(1, int(flush_every))
        self._count = 0
        self._file = self.path.open("w", encoding="utf-8")
        self.write_record({"type": "header", "payload": asdict(header or ReplayHeader())}, flush=True)

    def write_record(self, record: dict[str, Any], flush: bool = False) -> None:
        self._file.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
        self._count += 1
        if flush or self._count % self.flush_every == 0:
            self._file.flush()

    def write_step(self, step: int, payload: dict[str, Any]) -> None:
        self.write_record({"type": "step", "step": step, "payload": payload})

    def write_match_end(self, payload: dict[str, Any]) -> None:
        self.write_record({"type": "match_end", "payload": payload}, flush=True)

    def close(self) -> None:
        if not self._file.closed:
            self._file.flush()
            self._file.close()

    def __enter__(self) -> "MatchReplayRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close()


def load_replay(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
