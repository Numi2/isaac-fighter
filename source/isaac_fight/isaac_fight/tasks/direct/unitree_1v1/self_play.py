# Copyright (c) 2026, Isaac Fight contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Self-play population utilities.

Historical opponents are learned policies loaded from a checkpoint pool. The wrapper does not contain scripted fight
logic: it only replaces an opponent action tensor with the output of a sampled policy network.
"""

from __future__ import annotations

import os
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import gymnasium as gym
import torch
from torch import nn

from .fighter_ids import FIGHTER_A, FIGHTER_B, opponent_of
from .opponent_pool import OpponentPool, OpponentSample


class PolicyBackend(Protocol):
    def act(self, observations: torch.Tensor) -> torch.Tensor: ...


class TorchScriptPolicyBackend:
    """Inference backend for exported TorchScript policies."""

    def __init__(self, path: str | Path, device: str = "cuda:0"):
        self.path = Path(path)
        self.device = torch.device(device if torch.cuda.is_available() and "cuda" in str(device) else "cpu")
        self.module = torch.jit.load(str(self.path), map_location=self.device)
        self.module.eval()

    @torch.no_grad()
    def act(self, observations: torch.Tensor) -> torch.Tensor:
        obs = observations.to(self.device)
        out = self.module(obs)
        if isinstance(out, dict):
            out = out.get("actions", next(iter(out.values())))
        if isinstance(out, tuple):
            out = out[0]
        return torch.clamp(out.to(observations.device), -1.0, 1.0)


class SkrlCheckpointPolicyBackend:
    """Deterministic inference backend for skrl IPPO/MAPPO checkpoints.

    A skrl multi-agent checkpoint stores one policy state dict per fighter. This backend loads the selected fighter side
    and emits the Gaussian mean network output directly, which is the stable frozen-opponent action.
    """

    def __init__(self, path: str | Path, agent_id: str, device: str = "cuda:0"):
        self.path = Path(path)
        self.agent_id = agent_id
        self.device = torch.device(device if torch.cuda.is_available() and "cuda" in str(device) else "cpu")
        checkpoint = torch.load(self.path, map_location=self.device)
        agent_checkpoint = checkpoint[agent_id] if agent_id in checkpoint else checkpoint
        state_dict = agent_checkpoint["policy"]
        self.obs_mean: torch.Tensor | None = None
        self.obs_var: torch.Tensor | None = None
        preprocessor = agent_checkpoint.get("state_preprocessor", {})
        if "running_mean" in preprocessor and "running_variance" in preprocessor:
            self.obs_mean = preprocessor["running_mean"].float().to(self.device)
            self.obs_var = preprocessor["running_variance"].float().to(self.device)
        self.module = self._build_policy(state_dict).to(self.device)
        self.module.load_state_dict(
            {key.removeprefix("net_container."): value for key, value in state_dict.items() if key.startswith("net_container.")}
        )
        self.module.eval()

    @staticmethod
    def _build_policy(state_dict: dict[str, torch.Tensor]) -> nn.Sequential:
        layers: list[nn.Module] = []
        linear_indices = sorted(
            int(key.split(".")[1])
            for key in state_dict
            if key.startswith("net_container.") and key.endswith(".weight")
        )
        for idx, layer_idx in enumerate(linear_indices):
            weight = state_dict[f"net_container.{layer_idx}.weight"]
            layers.append(nn.Linear(weight.shape[1], weight.shape[0]))
            if idx != len(linear_indices) - 1:
                layers.append(nn.ELU())
        return nn.Sequential(*layers)

    @torch.no_grad()
    def act(self, observations: torch.Tensor) -> torch.Tensor:
        obs = observations.to(self.device)
        if self.obs_mean is not None and self.obs_var is not None:
            obs = torch.clamp((obs - self.obs_mean) / (torch.sqrt(self.obs_var) + 1.0e-8), -5.0, 5.0)
        return torch.clamp(self.module(obs).to(observations.device), -1.0, 1.0)


@dataclass
class SelfPlayTrainingSupervisor:
    """Synchronizes skrl checkpoint folders with the persistent opponent pool."""

    pool_dir: str | Path
    checkpoint_dir: str | Path
    snapshot_interval: int = 50
    active_elo: float = 1000.0
    metadata: dict | None = None

    def __post_init__(self):
        self.pool = OpponentPool(self.pool_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def sync_checkpoints(self) -> int:
        """Add unseen checkpoints to the policy pool and return the number added."""

        known_paths = {Path(p.checkpoint_path).resolve() for p in self.pool.policies if Path(p.checkpoint_path).exists()}
        candidates = sorted(self.checkpoint_dir.rglob("*.pt"), key=lambda p: (p.stat().st_mtime, str(p)))
        added = 0
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in known_paths:
                continue
            version = _version_from_path(candidate)
            run_name = _safe_name(candidate.parent.parent.name if candidate.parent.name in {"checkpoints", "models"} else candidate.parent.name)
            pooled_name = f"{run_name}_{candidate.name}" if run_name else candidate.name
            pooled_path = Path(self.pool.root) / "checkpoints" / pooled_name
            pooled_path.parent.mkdir(parents=True, exist_ok=True)
            if resolved != pooled_path.resolve():
                shutil.copy2(candidate, pooled_path)
            tags = ("skrl",) if _looks_like_skrl_checkpoint(candidate) else ("torchscript",)
            self.pool.add_checkpoint(pooled_path, version=version, elo=self.active_elo, tags=tags, metadata=self.metadata or {})
            known_paths.add(pooled_path.resolve())
            added += 1
        return added

    def sample(self, active_elo: float | None = None) -> OpponentSample | None:
        return self.pool.sample(active_elo=active_elo if active_elo is not None else self.active_elo)


class HistoricalOpponentActionWrapper(gym.Wrapper):
    """Gymnasium wrapper that samples learned historical opponents from a policy pool.

    The wrapped environment remains multi-agent. One agent is active for the trainer; the opponent action is generated
    from a sampled checkpoint. If no compatible checkpoint is present, the wrapper leaves trainer-provided actions
    unchanged so IPPO/MAPPO can bootstrap the first policy population.
    """

    def __init__(
        self,
        env: gym.Env,
        pool: OpponentPool,
        active_agent: str = FIGHTER_A,
        device: str = "cuda:0",
        active_elo: float = 1000.0,
        elo_window: float = 250.0,
        weakness_bias: float = 0.65,
        latest_bias: float = 0.15,
        update_interval_steps: int = 1000,
        side_swap_probability: float = 0.5,
        train_active_only: bool = True,
    ):
        super().__init__(env)
        self.pool = pool
        self.base_active_agent = active_agent
        self.active_agent = active_agent
        self.opponent_agent = opponent_of(active_agent)
        self.device = device
        self.active_elo = active_elo
        self.elo_window = elo_window
        self.weakness_bias = weakness_bias
        self.latest_bias = latest_bias
        self.update_interval_steps = max(1, int(update_interval_steps))
        self.side_swap_probability = max(0.0, min(1.0, float(side_swap_probability)))
        self.train_active_only = train_active_only
        self._last_obs: dict[str, torch.Tensor] | None = None
        self._current_sample: OpponentSample | None = None
        self._backend: PolicyBackend | None = None
        self._step_count = 0

    def reset(self, **kwargs):  # noqa: ANN003
        obs, info = self.env.reset(**kwargs)
        self._select_active_side()
        self._last_obs = obs
        self._sample_backend()
        return obs, info

    def step(self, actions):  # noqa: ANN001
        self._step_count += 1
        if self._step_count % self.update_interval_steps == 0:
            self._sample_backend()
        if self._backend is not None and self._last_obs is not None and self.opponent_agent in self._last_obs:
            actions = dict(actions)
            actions[self.opponent_agent] = self._backend.act(self._last_obs[self.opponent_agent])
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        if self.train_active_only and self._backend is not None and self.opponent_agent in rewards:
            rewards = dict(rewards)
            rewards[self.opponent_agent] = torch.zeros_like(rewards[self.opponent_agent])
        self._last_obs = obs
        if hasattr(self.env.unwrapped, "extras"):
            extras = self.env.unwrapped.extras
            for agent_info in extras.values():
                if not isinstance(agent_info, dict):
                    continue
                agent_info.setdefault("self_play", {})
                if self._current_sample is not None:
                    agent_info["self_play"].update(
                        {
                            "active_agent": self.active_agent,
                            "frozen_agent": self.opponent_agent,
                            "opponent_policy_id": self._current_sample.policy.policy_id,
                            "opponent_version": self._current_sample.policy.version,
                            "opponent_elo": self._current_sample.policy.elo,
                            "opponent_sample_probability": self._current_sample.probability,
                        }
                    )
        return obs, rewards, terminated, truncated, infos

    def _sample_backend(self) -> None:
        self.pool.load()
        sample = self.pool.sample(
            active_elo=self.active_elo,
            elo_window=self.elo_window,
            weakness_bias=self.weakness_bias,
            latest_bias=self.latest_bias,
        )
        self._current_sample = sample
        self._backend = None
        if sample is None:
            return
        path = Path(sample.policy.checkpoint_path)
        if not path.exists():
            return
        if "skrl" in sample.policy.tags or _looks_like_skrl_checkpoint(path):
            self._backend = SkrlCheckpointPolicyBackend(path, agent_id=self.opponent_agent, device=self.device)
        elif "torchscript" in sample.policy.tags:
            self._backend = TorchScriptPolicyBackend(path, device=self.device)

    def _select_active_side(self) -> None:
        if random.random() < self.side_swap_probability:
            self.active_agent = opponent_of(self.base_active_agent)
        else:
            self.active_agent = self.base_active_agent
        self.opponent_agent = opponent_of(self.active_agent)


def _version_from_path(path: Path) -> int:
    match = re.search(r"(\d+)(?!.*\d)", path.stem)
    return int(match.group(1)) if match else int(path.stat().st_mtime)


def _looks_like_torchscript(path: Path) -> bool:
    # TorchScript archives are zip files, but modern torch.save checkpoints are zip files too.
    try:
        with path.open("rb") as f:
            return f.read(4) == b"PK\x03\x04" and not _looks_like_skrl_checkpoint(path)
    except OSError:
        return False


def _looks_like_skrl_checkpoint(path: Path) -> bool:
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception:
        return False
    if not isinstance(checkpoint, dict):
        return False
    if "policy" in checkpoint:
        return True
    return any(isinstance(value, dict) and "policy" in value for value in checkpoint.values())


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")


def maybe_wrap_historical_opponent(env: gym.Env, cfg, log_dir: str | Path | None, args) -> gym.Env:  # noqa: ANN001
    if not getattr(args, "self_play", True) or not getattr(args, "historical_opponent", True):
        return env
    pool_dir = getattr(args, "pool_dir", None) or getattr(cfg.self_play, "pool_dir", "policy_pool")
    pool = OpponentPool(pool_dir)
    return HistoricalOpponentActionWrapper(
        env,
        pool=pool,
        active_agent=getattr(args, "active_agent", cfg.self_play.active_agent),
        device=getattr(args, "device", "cuda:0") or "cuda:0",
        active_elo=getattr(cfg.self_play, "active_elo", 1000.0),
        elo_window=cfg.self_play.elo_window,
        weakness_bias=cfg.self_play.weakness_bias,
        latest_bias=cfg.self_play.latest_bias,
        update_interval_steps=getattr(cfg.self_play, "opponent_update_interval", 1000),
        side_swap_probability=getattr(cfg.self_play, "side_swap_probability", 0.5),
        train_active_only=True,
    )


def checkpoint_dir_from_log_dir(log_dir: str | Path) -> Path:
    log_dir = Path(log_dir)
    for name in ("checkpoints", "models"):
        candidate = log_dir / name
        if candidate.exists():
            return candidate
    return log_dir / "checkpoints"
