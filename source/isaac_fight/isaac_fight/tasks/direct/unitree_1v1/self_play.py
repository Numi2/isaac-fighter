# Copyright (c) 2026, Isaac Fight contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Self-play population utilities.

Historical opponents are learned policies loaded from a checkpoint pool. The wrapper does not contain scripted fight
logic: it only replaces an opponent action tensor with the output of a sampled policy network.
"""

from __future__ import annotations

import re
import shutil
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import gymnasium as gym
import torch
from torch import nn

from .fighter_ids import FIGHTER_A, FIGHTER_B, FIGHTERS, opponent_of
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
            policy_id = f"{run_name}_v{version:06d}" if run_name else f"policy_v{version:06d}"
            metadata = dict(self.metadata or {})
            metadata.setdefault("source_run", run_name)
            metadata.setdefault("source_checkpoint", candidate.name)
            self.pool.add_checkpoint(pooled_path, version=version, policy_id=policy_id, elo=self.active_elo, tags=tags, metadata=metadata)
            known_paths.add(pooled_path.resolve())
            added += 1
        return added

    def sample(self, active_elo: float | None = None) -> OpponentSample | None:
        return self.pool.sample(active_elo=active_elo if active_elo is not None else self.active_elo)


class LiveSelfPlayPoolSync:
    """Background checkpoint sync so long training runs do not depend on a separate sidecar process."""

    def __init__(self, supervisor: SelfPlayTrainingSupervisor, interval_s: float = 60.0):
        self.supervisor = supervisor
        self.interval_s = max(1.0, float(interval_s))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="isaac-fight-pool-sync", daemon=True)
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        return self.supervisor.sync_checkpoints()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                added = self.supervisor.sync_checkpoints()
                if added:
                    print(f"[INFO] Live self-play pool sync added {added} checkpoint(s).", flush=True)
            except Exception:  # noqa: BLE001 - keep training alive if the pool fs hiccups.
                print("[WARN] Live self-play pool sync failed:\n" + traceback.format_exc(), flush=True)
            self._stop.wait(self.interval_s)


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
        live_self_play_fraction: float = 0.25,
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
        self.live_self_play_fraction = max(0.0, min(1.0, float(live_self_play_fraction)))
        self.train_active_only = train_active_only
        self._last_obs: dict[str, torch.Tensor] | None = None
        self._current_sample: OpponentSample | None = None
        self._backends: dict[str, PolicyBackend] = {}
        self._active_masks: dict[str, torch.Tensor] = {}
        self._freeze_masks: dict[str, torch.Tensor] = {}
        self._step_count = 0

    def reset(self, **kwargs):  # noqa: ANN003
        obs, info = self.env.reset(**kwargs)
        self._initialize_vectorized_masks(obs)
        self._last_obs = obs
        self._sample_backend()
        return obs, info

    def step(self, actions):  # noqa: ANN001
        self._step_count += 1
        if self._step_count % self.update_interval_steps == 0:
            self._sample_backend()
        overwritten_masks: dict[str, torch.Tensor] = {}
        if self._backends and self._last_obs is not None:
            actions = dict(actions)
            for agent, backend in self._backends.items():
                if agent not in self._last_obs:
                    continue
                mask = self._freeze_masks.get(agent)
                if mask is None or not bool(mask.any().item()):
                    continue
                frozen_actions = backend.act(self._last_obs[agent])
                if frozen_actions.shape != actions[agent].shape:
                    print(
                        f"[WARN] Frozen opponent action shape mismatch for {agent}: "
                        f"got {tuple(frozen_actions.shape)}, expected {tuple(actions[agent].shape)}",
                        flush=True,
                    )
                    continue
                mixed_actions = actions[agent].clone()
                mixed_actions[mask] = frozen_actions[mask]
                actions[agent] = mixed_actions
                overwritten_masks[agent] = mask
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        if self.train_active_only and overwritten_masks:
            rewards = dict(rewards)
            for agent, mask in overwritten_masks.items():
                if agent in rewards:
                    rewards[agent] = torch.where(mask, torch.zeros_like(rewards[agent]), rewards[agent])
        self._last_obs = obs
        if FIGHTER_A in terminated and FIGHTER_A in truncated:
            done = terminated[FIGHTER_A] | truncated[FIGHTER_A]
            self._resample_vectorized_masks(done)
        if hasattr(self.env.unwrapped, "extras"):
            extras = self.env.unwrapped.extras
            for agent_info in extras.values():
                if not isinstance(agent_info, dict):
                    continue
                agent_info.setdefault("self_play", {})
                if self._current_sample is not None:
                    agent_info["self_play"].update(
                        {
                            "active_agent": "mixed",
                            "frozen_agent": "mixed",
                            "active_fraction_fighter_a": self._active_fraction(FIGHTER_A),
                            "active_fraction_fighter_b": self._active_fraction(FIGHTER_B),
                            "frozen_fraction_fighter_a": self._freeze_fraction(FIGHTER_A),
                            "frozen_fraction_fighter_b": self._freeze_fraction(FIGHTER_B),
                            "live_self_play_fraction": self.live_self_play_fraction,
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
        self._backends = {}
        if sample is None:
            return
        path = Path(sample.policy.checkpoint_path)
        if not path.exists():
            return
        for agent in FIGHTERS:
            try:
                if "skrl" in sample.policy.tags or _looks_like_skrl_checkpoint(path):
                    self._backends[agent] = SkrlCheckpointPolicyBackend(path, agent_id=agent, device=self.device)
                elif "torchscript" in sample.policy.tags:
                    self._backends[agent] = TorchScriptPolicyBackend(path, device=self.device)
            except Exception as exc:  # noqa: BLE001 - skip incompatible legacy pool entries.
                print(f"[WARN] Could not load frozen opponent backend for {agent} from {path}: {exc}", flush=True)

    def _initialize_vectorized_masks(self, obs: dict[str, torch.Tensor]) -> None:
        sample_obs = next(iter(obs.values()))
        n = sample_obs.shape[0]
        device = sample_obs.device
        self._active_masks = {agent: torch.zeros(n, dtype=torch.bool, device=device) for agent in FIGHTERS}
        self._freeze_masks = {agent: torch.zeros(n, dtype=torch.bool, device=device) for agent in FIGHTERS}
        self._resample_vectorized_masks(torch.ones(n, dtype=torch.bool, device=device))

    def _resample_vectorized_masks(self, env_mask: torch.Tensor) -> None:
        if not self._active_masks:
            return
        env_mask = env_mask.to(next(iter(self._active_masks.values())).device)
        if not bool(env_mask.any().item()):
            return
        n = int(env_mask.sum().item())
        base_device = env_mask.device
        swapped = torch.rand(n, device=base_device) < self.side_swap_probability
        use_frozen = torch.rand(n, device=base_device) >= self.live_self_play_fraction
        active_a = torch.full((n,), self.base_active_agent == FIGHTER_A, dtype=torch.bool, device=base_device)
        active_a = torch.where(swapped, ~active_a, active_a)
        active_b = ~active_a
        self._active_masks[FIGHTER_A][env_mask] = active_a
        self._active_masks[FIGHTER_B][env_mask] = active_b
        self._freeze_masks[FIGHTER_A][env_mask] = active_b & use_frozen
        self._freeze_masks[FIGHTER_B][env_mask] = active_a & use_frozen

    def _active_fraction(self, agent: str) -> float:
        mask = self._active_masks.get(agent)
        return float(mask.float().mean().item()) if mask is not None and mask.numel() else 0.0

    def _freeze_fraction(self, agent: str) -> float:
        mask = self._freeze_masks.get(agent)
        return float(mask.float().mean().item()) if mask is not None and mask.numel() else 0.0


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
        live_self_play_fraction=getattr(cfg.self_play, "live_self_play_fraction", 0.25),
        train_active_only=True,
    )


def checkpoint_dir_from_log_dir(log_dir: str | Path) -> Path:
    log_dir = Path(log_dir)
    for name in ("checkpoints", "models"):
        candidate = log_dir / name
        if candidate.exists():
            return candidate
    return log_dir / "checkpoints"
