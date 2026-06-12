from __future__ import annotations

from pathlib import Path
import sys
import types

import torch

try:
    import gymnasium as gym
except ModuleNotFoundError:
    class _Env:
        @property
        def unwrapped(self):
            return self

    class _Wrapper(_Env):
        def __init__(self, env):
            self.env = env

        @property
        def unwrapped(self):
            return self.env.unwrapped

    gym = types.SimpleNamespace(Env=_Env, Wrapper=_Wrapper)
    sys.modules["gymnasium"] = gym

from isaac_fight.tasks.direct.unitree_1v1.fighter_ids import FIGHTER_A, FIGHTER_B
from isaac_fight.tasks.direct.unitree_1v1.opponent_pool import OpponentPool
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
