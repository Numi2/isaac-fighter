# Copyright (c) 2026, Isaac Fight contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Task registration for Unitree 1v1 combat."""

from __future__ import annotations

try:
    import gymnasium as gym
except Exception:  # pure utility tests may run without Gymnasium/Isaac Lab installed
    gym = None  # type: ignore[assignment]

_TASK_ID = "GhostFighter-Unitree-1v1-Direct-v0"

if gym is not None and _TASK_ID not in gym.registry:
    gym.register(
        id=_TASK_ID,
        entry_point="isaac_fight.tasks.direct.unitree_1v1.unitree_1v1_env:GhostFighterUnitree1v1Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "isaac_fight.tasks.direct.unitree_1v1.unitree_1v1_env_cfg:GhostFighterUnitree1v1EnvCfg",
            "skrl_cfg_entry_point": "isaac_fight.tasks.direct.unitree_1v1.agents.skrl_ppo_cfg:IPPO_CFG",
            "skrl_ippo_cfg_entry_point": "isaac_fight.tasks.direct.unitree_1v1.agents.skrl_ppo_cfg:IPPO_CFG",
            "skrl_mappo_cfg_entry_point": "isaac_fight.tasks.direct.unitree_1v1.agents.skrl_ppo_cfg:MAPPO_CFG",
            "skrl_ppo_cfg_entry_point": "isaac_fight.tasks.direct.unitree_1v1.agents.skrl_ppo_cfg:PPO_DEBUG_CFG",
        },
    )

try:
    from .unitree_1v1_env import GhostFighterUnitree1v1Env  # noqa: E402,F401
    from .unitree_1v1_env_cfg import GhostFighterUnitree1v1EnvCfg, GhostFighterUnitree1v1PlayEnvCfg  # noqa: E402,F401
except Exception:
    # Isaac Lab is not importable in pure Python tooling. Gym registration above remains available through string entry
    # points and the actual classes resolve inside Isaac Lab.
    GhostFighterUnitree1v1Env = None  # type: ignore[assignment]
    GhostFighterUnitree1v1EnvCfg = None  # type: ignore[assignment]
    GhostFighterUnitree1v1PlayEnvCfg = None  # type: ignore[assignment]

__all__ = ["GhostFighterUnitree1v1Env", "GhostFighterUnitree1v1EnvCfg", "GhostFighterUnitree1v1PlayEnvCfg"]
