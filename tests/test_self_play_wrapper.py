from __future__ import annotations

from pathlib import Path

import torch

import gymnasium as gym

from isaac_fight.tasks.direct.unitree_1v1.fighter_ids import FIGHTER_A, FIGHTER_B
from isaac_fight.tasks.direct.unitree_1v1.opponent_pool import OpponentPool
from isaac_fight.tasks.direct.unitree_1v1 import self_play
from isaac_fight.tasks.direct.unitree_1v1.self_play import HistoricalOpponentActionWrapper


class _ZeroBackend:
    @torch.no_grad()
    def act(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(observations)


class _DummyVectorEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.extras = {}
        self.last_actions = None

    def reset(self, **kwargs):  # noqa: ANN003
        return {FIGHTER_A: torch.ones(4, 2), FIGHTER_B: torch.ones(4, 3)}, {}

    def step(self, actions):  # noqa: ANN001
        self.last_actions = actions
        rewards = {FIGHTER_A: torch.ones(4), FIGHTER_B: torch.ones(4)}
        terminated = {FIGHTER_A: torch.zeros(4, dtype=torch.bool), FIGHTER_B: torch.zeros(4, dtype=torch.bool)}
        truncated = {FIGHTER_A: torch.zeros(4, dtype=torch.bool), FIGHTER_B: torch.zeros(4, dtype=torch.bool)}
        return {FIGHTER_A: torch.ones(4, 2), FIGHTER_B: torch.ones(4, 3)}, rewards, terminated, truncated, {}


def test_vectorized_historical_wrapper_freezes_only_frozen_side(tmp_path: Path):
    env = _DummyVectorEnv()
    wrapper = HistoricalOpponentActionWrapper(
        env,
        pool=OpponentPool(tmp_path / "pool"),
        active_agent=FIGHTER_A,
        side_swap_probability=0.0,
        live_self_play_fraction=0.0,
    )
    wrapper.reset()
    wrapper._backends = {FIGHTER_B: _ZeroBackend()}

    actions = {FIGHTER_A: torch.ones(4, 2), FIGHTER_B: torch.ones(4, 3)}
    _, rewards, _, _, _ = wrapper.step(actions)

    assert torch.all(env.last_actions[FIGHTER_A] == 1.0)
    assert torch.all(env.last_actions[FIGHTER_B] == 0.0)
    assert torch.all(rewards[FIGHTER_A] == 1.0)
    assert torch.all(rewards[FIGHTER_B] == 0.0)


def test_historical_wrapper_caches_backends_for_same_checkpoint(tmp_path: Path, monkeypatch):
    ckpt = tmp_path / "agent_000050.pt"
    ckpt.write_bytes(b"torchscript")
    pool = OpponentPool(tmp_path / "pool")
    pool.add_checkpoint(ckpt, version=50, policy_id="p", tags=("torchscript",))

    loads = {"count": 0}

    class _CountingBackend(_ZeroBackend):
        def __init__(self, path, device="cuda:0"):  # noqa: ANN001, ARG002
            loads["count"] += 1

    monkeypatch.setattr(self_play, "TorchScriptPolicyBackend", _CountingBackend)
    wrapper = HistoricalOpponentActionWrapper(_DummyVectorEnv(), pool=pool)
    wrapper.reset()
    assert loads["count"] == 2

    wrapper._sample_backend()
    assert loads["count"] == 2
