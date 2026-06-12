# Copyright (c) 2026, Isaac Fight contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""skrl runner configurations for GhostFighter Unitree 1v1.

The configs follow Isaac Lab's skrl runner pattern. IPPO is the primary multi-agent algorithm. MAPPO is supplied for
centralized-critic experiments through the DirectMARLEnv state-space.
"""

from __future__ import annotations

_COMMON_MODEL = {
    "clip_actions": True,
    "clip_log_std": True,
    "min_log_std": -5.0,
    "max_log_std": 2.0,
    "initial_log_std": -1.2,
    "network": [{"name": "net", "input": "OBSERVATIONS", "layers": [512, 256, 128], "activations": "elu"}],
    "output": "ACTIONS",
}

IPPO_CFG = {
    "seed": 42,
    "models": {
        "separate": True,
        "policy": {
            "class": "GaussianMixin",
            **_COMMON_MODEL,
        },
        "value": {
            "class": "DeterministicMixin",
            "clip_actions": False,
            "network": [{"name": "net", "input": "OBSERVATIONS", "layers": [512, 256, 128], "activations": "elu"}],
            "output": "ONE",
        },
    },
    "memory": {
        "class": "RandomMemory",
        "memory_size": -1,
    },
    "agent": {
        "class": "IPPO",
        "rollouts": 32,
        "learning_epochs": 5,
        "mini_batches": 8,
        "discount_factor": 0.995,
        "lambda": 0.95,
        "learning_rate": 3.0e-4,
        "learning_rate_scheduler": "KLAdaptiveLR",
        "learning_rate_scheduler_kwargs": {"kl_threshold": 0.016},
        "random_timesteps": 0,
        "learning_starts": 0,
        "grad_norm_clip": 1.0,
        "ratio_clip": 0.2,
        "value_clip": 0.2,
        "clip_predicted_values": True,
        "entropy_loss_scale": 0.006,
        "value_loss_scale": 2.0,
        "kl_threshold": 0.0,
        "rewards_shaper_scale": 0.01,
        "time_limit_bootstrap": True,
        "state_preprocessor": "RunningStandardScaler",
        "state_preprocessor_kwargs": {"size": "auto", "device": "auto"},
        "value_preprocessor": "RunningStandardScaler",
        "value_preprocessor_kwargs": {"size": 1, "device": "auto"},
        "experiment": {
            "directory": "ghostfighter_unitree_1v1",
            "experiment_name": "ippo_self_play",
            "write_interval": 100,
            "checkpoint_interval": 2000,
            "store_separately": False,
        },
    },
    "trainer": {
        "class": "SequentialTrainer",
        "timesteps": 250_000_000,
        "environment_info": "log",
    },
}

MAPPO_CFG = {
    **IPPO_CFG,
    "models": {
        "separate": True,
        "policy": {
            "class": "GaussianMixin",
            **_COMMON_MODEL,
        },
        "value": {
            "class": "DeterministicMixin",
            "clip_actions": False,
            "network": [{"name": "net", "input": "STATES", "layers": [768, 384, 192], "activations": "elu"}],
            "output": "ONE",
        },
    },
    "agent": {
        **IPPO_CFG["agent"],
        "class": "MAPPO",
        "learning_rate": 2.0e-4,
        "experiment": {
            **IPPO_CFG["agent"]["experiment"],
            "experiment_name": "mappo_self_play",
        },
    },
}

# Kept only for single-agent debugging through Isaac Lab's multi_agent_to_single_agent path.
PPO_DEBUG_CFG = {
    **IPPO_CFG,
    "agent": {
        **IPPO_CFG["agent"],
        "class": "PPO",
        "experiment": {
            **IPPO_CFG["agent"]["experiment"],
            "experiment_name": "ppo_debug_flattened",
        },
    },
}
