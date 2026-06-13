# Copyright (c) 2026, Isaac Fight contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Adapters for Unitree Isaac Lab robot configurations.

This module deliberately imports validated Unitree robot configuration objects from the
Unitree Isaac Lab repositories at runtime. It does not recreate meshes, masses,
collisions, inertias, limits, or actuator models. The local joint-name metadata is used
for action-space sizing and static checks when Isaac Lab is not available.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

G1_29DOF_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

H1_JOINT_NAMES: tuple[str, ...] = (
    "right_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "left_hip_roll_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "torso_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_ankle_joint",
    "right_ankle_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
)

# H1-2 has appeared with different naming across Unitree asset drops. It is kept
# as a target, not a default, and the runtime importer will trust the installed
# upstream cfg when present.
H1_2_FALLBACK_JOINT_NAMES: tuple[str, ...] = H1_JOINT_NAMES


@dataclass(frozen=True)
class UnitreeRobotSpec:
    """Static metadata for a Unitree fighter type."""

    name: str
    display_name: str
    default_base_height: float
    controlled_joint_names: tuple[str, ...]
    torso_body_regex: str
    foot_body_regex: str
    upper_body_regex: str
    nominal_action_scale: float

    @property
    def action_dim(self) -> int:
        return len(self.controlled_joint_names)


_SPECS: dict[str, UnitreeRobotSpec] = {
    "g1_29dof": UnitreeRobotSpec(
        name="g1_29dof",
        display_name="Unitree G1 29DoF",
        default_base_height=0.80,
        controlled_joint_names=G1_29DOF_JOINT_NAMES,
        torso_body_regex="torso.*|pelvis.*|waist.*",
        foot_body_regex=".*ankle.*|.*foot.*",
        upper_body_regex=".*shoulder.*|.*elbow.*|.*wrist.*|torso.*|waist.*",
        nominal_action_scale=0.25,
    ),
    "h1": UnitreeRobotSpec(
        name="h1",
        display_name="Unitree H1",
        default_base_height=1.10,
        controlled_joint_names=H1_JOINT_NAMES,
        torso_body_regex="torso.*|pelvis.*|base.*",
        foot_body_regex=".*ankle.*|.*foot.*",
        upper_body_regex=".*shoulder.*|.*elbow.*|torso.*",
        nominal_action_scale=0.28,
    ),
    "h1_2": UnitreeRobotSpec(
        name="h1_2",
        display_name="Unitree H1-2",
        default_base_height=1.10,
        controlled_joint_names=H1_2_FALLBACK_JOINT_NAMES,
        torso_body_regex="torso.*|pelvis.*|base.*",
        foot_body_regex=".*ankle.*|.*foot.*",
        upper_body_regex=".*shoulder.*|.*elbow.*|.*wrist.*|torso.*",
        nominal_action_scale=0.28,
    ),
}

_CFG_CANDIDATES: dict[str, tuple[tuple[str, str], ...]] = {
    "g1_29dof": (
        ("unitree_rl_lab.assets.robots.unitree", "UNITREE_G1_29DOF_CFG"),
        ("unitree_sim_isaaclab.assets.robots.unitree", "UNITREE_G1_29DOF_CFG"),
    ),
    "h1": (
        ("unitree_rl_lab.assets.robots.unitree", "UNITREE_H1_CFG"),
        ("unitree_sim_isaaclab.assets.robots.unitree", "UNITREE_H1_CFG"),
    ),
    "h1_2": (
        ("unitree_rl_lab.assets.robots.unitree", "UNITREE_H1_2_CFG"),
        ("unitree_rl_lab.assets.robots.unitree", "UNITREE_H12_CFG"),
        ("unitree_sim_isaaclab.assets.robots.unitree", "UNITREE_H1_2_CFG"),
        ("unitree_sim_isaaclab.assets.robots.unitree", "UNITREE_H12_CFG"),
        ("unitree_sim_isaaclab.assets.robots.h1_2", "UNITREE_H1_2_CFG"),
    ),
}


def list_supported_unitree_robots() -> tuple[str, ...]:
    """Return robot names understood by the adapter."""

    return tuple(_SPECS.keys())


def get_unitree_robot_spec(name: str) -> UnitreeRobotSpec:
    """Return static metadata for a supported Unitree robot name."""

    normalized = _normalize_name(name)
    try:
        return _SPECS[normalized]
    except KeyError as exc:
        raise KeyError(
            f"Unsupported Unitree robot '{name}'. Supported names: {', '.join(list_supported_unitree_robots())}"
        ) from exc


def get_unitree_robot_cfg(name: str, prim_path: str | None = None) -> Any:
    """Resolve an upstream Isaac Lab ArticulationCfg for a Unitree robot.

    Args:
        name: Robot selector: ``g1_29dof``, ``h1`` or ``h1_2``.
        prim_path: Optional Isaac Lab prim path override.

    Returns:
        The upstream ArticulationCfg, optionally with ``prim_path`` replaced.

    Raises:
        RuntimeError: If no installed Unitree Isaac Lab package exposes the requested cfg.
    """

    normalized = _normalize_name(name)
    errors: list[str] = []
    for module_name, symbol in _CFG_CANDIDATES.get(normalized, ()):  # keep order deterministic
        try:
            module = import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - include import errors in the runtime diagnostic
            errors.append(f"{module_name}: {exc}")
            continue
        if not hasattr(module, symbol):
            errors.append(f"{module_name}.{symbol}: missing")
            continue
        cfg = getattr(module, symbol)
        if prim_path is not None:
            if not hasattr(cfg, "replace"):
                raise RuntimeError(
                    f"Resolved {module_name}.{symbol}, but it does not implement Isaac Lab cfg.replace()."
                )
            return cfg.replace(prim_path=prim_path)
        return cfg

    detail = "\n  - ".join(errors) if errors else "no candidate modules were configured"
    raise RuntimeError(
        "Could not resolve the Unitree Isaac Lab robot cfg for "
        f"'{name}'. Install unitree_rl_lab and configure UNITREE_MODEL_DIR/UNITREE_ROS_DIR as required by that "
        "repository. This extension does not ship or recreate Unitree physics assets. Tried:\n  - "
        f"{detail}"
    )


def get_controlled_joint_names_from_cfg_or_spec(name: str, cfg: Any | None = None) -> tuple[str, ...]:
    """Return controlled joint names from an upstream cfg if available, else from local metadata."""

    if cfg is not None:
        cfg_joint_names = tuple(j for j in getattr(cfg, "joint_sdk_names", ()) if isinstance(j, str) and j)
        if cfg_joint_names:
            return cfg_joint_names
    return get_unitree_robot_spec(name).controlled_joint_names


def _normalize_name(name: str) -> str:
    normalized = name.lower().replace("-", "_").replace("/", "_")
    aliases = {
        "g1": "g1_29dof",
        "g1_29": "g1_29dof",
        "g1_29_dof": "g1_29dof",
        "g1_29dof": "g1_29dof",
        "unitree_g1_29dof": "g1_29dof",
        "unitree_h1": "h1",
        "h1": "h1",
        "h1_2": "h1_2",
        "h12": "h1_2",
        "unitree_h1_2": "h1_2",
    }
    return aliases.get(normalized, normalized)
