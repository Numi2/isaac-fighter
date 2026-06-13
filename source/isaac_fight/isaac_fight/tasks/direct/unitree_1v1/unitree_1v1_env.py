# Copyright (c) 2026, Isaac Fight contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Direct multi-agent Unitree humanoid 1v1 combat environment."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import DirectMARLEnv
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from isaac_fight.assets.robots.unitree import (
    get_controlled_joint_names_from_cfg_or_spec,
    get_unitree_robot_cfg,
    get_unitree_robot_spec,
)
from isaac_fight.utils.torch_math import normalize, quat_apply_inverse, quat_from_yaw, yaw_from_quat

from .fight_common import FighterRuntimeInfo
from .fight_rules import FightRuleEngine
from .fighter_ids import FIGHTER_A, FIGHTER_B, FIGHTERS, opponent_of
from .observations import OPPONENT_KEYPOINTS, CombatObservationBuilder
from .replay import MatchReplayRecorder, ReplayHeader
from .reward_terms import CombatRewardComputer
from .unitree_1v1_env_cfg import GhostFighterUnitree1v1EnvCfg

STRIKE_BODY_TOKENS = (
    "shoulder",
    "upper_arm",
    "lower_arm",
    "elbow",
    "wrist",
    "hand",
    "hip",
    "thigh",
    "knee",
    "shin",
    "ankle",
    "foot",
    "toe",
    "sole",
    "head",
    "neck",
)
SUPPORT_BODY_TOKENS = ("foot", "ankle", "toe", "sole")
PUSH_HAND_BODY_TOKENS = ("hand", "wrist", "palm", "finger", "lower_arm", "elbow")


class GhostFighterUnitree1v1Env(DirectMARLEnv):
    """Two-humanoid combat environment using Isaac Lab DirectMARLEnv.

    The environment applies policy actions as joint-position target offsets. Fight behavior is not scripted; combat
    emerges from the multi-agent reward, termination, and self-play setup.
    """

    cfg: GhostFighterUnitree1v1EnvCfg

    def __init__(self, cfg: GhostFighterUnitree1v1EnvCfg, render_mode: str | None = None, **kwargs):
        cfg.__post_init__()
        self.robots: dict[str, Articulation] = {}
        self._robot_cfgs: dict[str, Any] = {}
        self._replay: MatchReplayRecorder | None = None
        super().__init__(cfg, render_mode, **kwargs)

        self._rule_engine = FightRuleEngine(cfg.rules)
        self._obs_builder = CombatObservationBuilder(cfg.observations_cfg)
        self._reward_computer = CombatRewardComputer(cfg.rewards)
        self._runtime: dict[str, FighterRuntimeInfo] = {}
        self._keypoint_body_ids: dict[str, list[int]] = {}
        self._keypoint_body_id_tensors: dict[str, torch.Tensor] = {}
        self._keypoint_body_names: dict[str, list[str]] = {}
        self._support_body_ids: dict[str, list[int]] = {}
        self._support_body_id_tensors: dict[str, torch.Tensor] = {}
        self._left_support_body_id_tensors: dict[str, torch.Tensor] = {}
        self._right_support_body_id_tensors: dict[str, torch.Tensor] = {}
        self._upper_contact_body_id_tensors: dict[str, torch.Tensor] = {}
        self._strike_body_id_tensors: dict[str, torch.Tensor] = {}
        self._left_push_body_id_tensors: dict[str, torch.Tensor] = {}
        self._right_push_body_id_tensors: dict[str, torch.Tensor] = {}
        self._torso_contact_body_id_tensors: dict[str, torch.Tensor] = {}
        self._waist_action_id_tensors: dict[str, torch.Tensor] = {}
        self._arm_action_id_tensors: dict[str, torch.Tensor] = {}
        self._left_arm_action_id_tensors: dict[str, torch.Tensor] = {}
        self._right_arm_action_id_tensors: dict[str, torch.Tensor] = {}
        self._leg_action_id_tensors: dict[str, torch.Tensor] = {}
        self._knee_action_id_tensors: dict[str, torch.Tensor] = {}
        self._hip_yaw_roll_action_id_tensors: dict[str, torch.Tensor] = {}
        self._posture_action_id_tensors: dict[str, torch.Tensor] = {}
        self._action_scale_tensors: dict[str, torch.Tensor] = {}
        self._resolve_controlled_joints()
        self._allocate_buffers()
        self._configure_replay()

    def _setup_scene(self):
        prim_a = "/World/envs/env_.*/FighterA"
        prim_b = "/World/envs/env_.*/FighterB"
        self._robot_cfgs[FIGHTER_A] = get_unitree_robot_cfg(self.cfg.fighter_a.robot_name, prim_path=prim_a)
        self._robot_cfgs[FIGHTER_B] = get_unitree_robot_cfg(self.cfg.fighter_b.robot_name, prim_path=prim_b)
        self.robots[FIGHTER_A] = Articulation(self._robot_cfgs[FIGHTER_A])
        self.robots[FIGHTER_B] = Articulation(self._robot_cfgs[FIGHTER_B])
        self.scene.sensors[f"contact_{FIGHTER_A}"] = ContactSensor(
            ContactSensorCfg(
                prim_path=f"{prim_a}/.*",
                update_period=0.0,
                history_length=1,
            )
        )
        self.scene.sensors[f"contact_{FIGHTER_B}"] = ContactSensor(
            ContactSensorCfg(
                prim_path=f"{prim_b}/.*",
                update_period=0.0,
                history_length=1,
            )
        )

        self._spawn_ground()
        self._spawn_arena_boundary_visuals()

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        for agent, robot in self.robots.items():
            self.scene.articulations[agent] = robot

        light_cfg = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.85, 0.85, 0.85))
        light_cfg.func("/World/Light", light_cfg)

    def _spawn_ground(self) -> None:
        try:
            physics_material = sim_utils.RigidBodyMaterialCfg(
                static_friction=self.cfg.arena.floor_static_friction,
                dynamic_friction=self.cfg.arena.floor_dynamic_friction,
                restitution=self.cfg.arena.floor_restitution,
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
            )
            spawn_ground_plane("/World/ground", GroundPlaneCfg(physics_material=physics_material))
        except TypeError:
            spawn_ground_plane("/World/ground", GroundPlaneCfg())

    def _spawn_arena_boundary_visuals(self) -> None:
        if not self.cfg.arena.visual_boundary:
            return
        try:
            import omni.usd
            from pxr import Gf, UsdGeom

            stage = omni.usd.get_context().get_stage()
            root = "/World/ArenaBoundary"
            UsdGeom.Xform.Define(stage, root)
            r = self.cfg.arena.radius
            h = self.cfg.arena.wall_height
            t = self.cfg.arena.wall_thickness
            walls = {
                "north": ((0.0, r + 0.5 * t, 0.5 * h), (2.0 * r + 2.0 * t, t, h)),
                "south": ((0.0, -r - 0.5 * t, 0.5 * h), (2.0 * r + 2.0 * t, t, h)),
                "east": ((r + 0.5 * t, 0.0, 0.5 * h), (t, 2.0 * r, h)),
                "west": ((-r - 0.5 * t, 0.0, 0.5 * h), (t, 2.0 * r, h)),
            }
            for name, (translation, scale) in walls.items():
                cube = UsdGeom.Cube.Define(stage, f"{root}/{name}")
                cube.CreateSizeAttr(1.0)
                cube.AddTranslateOp().Set(Gf.Vec3d(*translation))
                cube.AddScaleOp().Set(Gf.Vec3d(*scale))
        except Exception:
            # The logical arena boundary is enforced by the rule engine. Visual wall creation should not make headless
            # training fail if USD helper APIs differ across Isaac Sim releases.
            return

    def _resolve_controlled_joints(self) -> None:
        fighter_cfgs = {FIGHTER_A: self.cfg.fighter_a, FIGHTER_B: self.cfg.fighter_b}
        for agent in FIGHTERS:
            fighter_cfg = fighter_cfgs[agent]
            robot = self.robots[agent]
            spec = get_unitree_robot_spec(fighter_cfg.robot_name)
            cfg_joint_names = get_controlled_joint_names_from_cfg_or_spec(
                fighter_cfg.robot_name, self._robot_cfgs.get(agent)
            )
            joint_names = list(fighter_cfg.controlled_joint_names or cfg_joint_names or spec.controlled_joint_names)
            joint_ids, resolved_names = robot.find_joints(joint_names)
            if len(joint_ids) != len(joint_names) and fighter_cfg.strict_joint_names:
                missing = sorted(set(joint_names) - set(resolved_names))
                raise RuntimeError(
                    f"{agent} expected {len(joint_names)} controlled joints for {fighter_cfg.robot_name}, "
                    f"but Isaac resolved {len(joint_ids)}. Missing examples: {missing[:8]}. "
                    "Set strict_joint_names=False only if the upstream asset intentionally changed joint naming."
                )
            action_dim = self.cfg.action_spaces[agent]
            if len(joint_ids) != action_dim:
                raise RuntimeError(
                    f"{agent} action space is {action_dim}, but resolved {len(joint_ids)} controlled joints. "
                    "Update the FighterCfg.controlled_joint_names or robot spec before training."
                )
            scale = fighter_cfg.action_scale if fighter_cfg.action_scale is not None else spec.nominal_action_scale
            self._runtime[agent] = FighterRuntimeInfo(
                agent_id=agent,
                robot_name=fighter_cfg.robot_name,
                joint_ids=list(joint_ids),
                joint_names=list(resolved_names),
                action_dim=action_dim,
                default_base_height=spec.default_base_height,
                action_scale=float(scale),
            )
            self._action_scale_tensors[agent] = torch.as_tensor(
                [self._joint_action_scale_multiplier(name) for name in resolved_names],
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0) * float(scale)
            waist_action_ids = [idx for idx, name in enumerate(resolved_names) if "waist" in name.lower()]
            self._waist_action_id_tensors[agent] = torch.as_tensor(
                waist_action_ids, dtype=torch.long, device=self.device
            )
            arm_action_ids = [
                idx
                for idx, name in enumerate(resolved_names)
                if any(token in name.lower() for token in ("shoulder", "elbow", "wrist", "hand", "finger"))
            ]
            left_arm_action_ids = [
                idx
                for idx in arm_action_ids
                if "left" in resolved_names[idx].lower() or "_l_" in resolved_names[idx].lower()
            ]
            right_arm_action_ids = [
                idx
                for idx in arm_action_ids
                if "right" in resolved_names[idx].lower() or "_r_" in resolved_names[idx].lower()
            ]
            leg_action_ids = [
                idx
                for idx, name in enumerate(resolved_names)
                if any(token in name.lower() for token in ("hip", "knee", "ankle"))
            ]
            knee_action_ids = [idx for idx, name in enumerate(resolved_names) if "knee" in name.lower()]
            hip_yaw_roll_action_ids = [
                idx
                for idx, name in enumerate(resolved_names)
                if "hip_yaw" in name.lower() or "hip_roll" in name.lower()
            ]
            posture_action_ids = sorted(set(leg_action_ids + waist_action_ids))
            self._arm_action_id_tensors[agent] = torch.as_tensor(arm_action_ids, dtype=torch.long, device=self.device)
            self._left_arm_action_id_tensors[agent] = torch.as_tensor(
                left_arm_action_ids or arm_action_ids,
                dtype=torch.long,
                device=self.device,
            )
            self._right_arm_action_id_tensors[agent] = torch.as_tensor(
                right_arm_action_ids or arm_action_ids,
                dtype=torch.long,
                device=self.device,
            )
            self._leg_action_id_tensors[agent] = torch.as_tensor(leg_action_ids, dtype=torch.long, device=self.device)
            self._knee_action_id_tensors[agent] = torch.as_tensor(knee_action_ids, dtype=torch.long, device=self.device)
            self._hip_yaw_roll_action_id_tensors[agent] = torch.as_tensor(
                hip_yaw_roll_action_ids, dtype=torch.long, device=self.device
            )
            self._posture_action_id_tensors[agent] = torch.as_tensor(
                posture_action_ids or list(range(action_dim)), dtype=torch.long, device=self.device
            )
            self._resolve_keypoint_bodies(agent)
            self._resolve_support_bodies(agent)

    def _resolve_keypoint_bodies(self, agent: str) -> None:
        robot = self.robots[agent]
        ids: list[int] = []
        names: list[str] = []
        patterns = list(self.cfg.observations_cfg.opponent_keypoint_body_patterns)
        patterns = (patterns + [""] * OPPONENT_KEYPOINTS)[:OPPONENT_KEYPOINTS]
        for pattern in patterns:
            try:
                body_ids, body_names = robot.find_bodies(pattern, preserve_order=False)
            except Exception:
                body_ids, body_names = [], []
            ids.append(int(body_ids[0]) if body_ids else -1)
            names.append(str(body_names[0]) if body_names else "")
        self._keypoint_body_ids[agent] = ids
        self._keypoint_body_id_tensors[agent] = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        self._keypoint_body_names[agent] = names

    def _resolve_support_bodies(self, agent: str) -> None:
        body_names = tuple(getattr(self.robots[agent], "body_names", ()) or ())
        support_ids: list[int] = []
        left_support_ids: list[int] = []
        right_support_ids: list[int] = []
        upper_contact_ids: list[int] = []
        strike_ids: list[int] = []
        left_push_ids: list[int] = []
        right_push_ids: list[int] = []
        torso_contact_ids: list[int] = []
        for body_id, body_name in enumerate(body_names):
            lower_name = body_name.lower()
            is_support = any(token in lower_name for token in SUPPORT_BODY_TOKENS)
            if is_support:
                support_ids.append(body_id)
                if "left" in lower_name:
                    left_support_ids.append(body_id)
                if "right" in lower_name:
                    right_support_ids.append(body_id)
            else:
                upper_contact_ids.append(body_id)
            if any(token in lower_name for token in STRIKE_BODY_TOKENS):
                strike_ids.append(body_id)
            if any(token in lower_name for token in PUSH_HAND_BODY_TOKENS):
                if "left" in lower_name or "_l_" in lower_name:
                    left_push_ids.append(body_id)
                if "right" in lower_name or "_r_" in lower_name:
                    right_push_ids.append(body_id)
            if any(token in lower_name for token in ("pelvis", "base", "torso", "waist", "trunk", "chest")):
                torso_contact_ids.append(body_id)

        all_ids = list(range(len(body_names)))
        self._support_body_ids[agent] = support_ids
        self._support_body_id_tensors[agent] = torch.as_tensor(support_ids, dtype=torch.long, device=self.device)
        self._left_support_body_id_tensors[agent] = torch.as_tensor(
            left_support_ids or support_ids,
            dtype=torch.long,
            device=self.device,
        )
        self._right_support_body_id_tensors[agent] = torch.as_tensor(
            right_support_ids or support_ids,
            dtype=torch.long,
            device=self.device,
        )
        self._upper_contact_body_id_tensors[agent] = torch.as_tensor(
            upper_contact_ids or all_ids,
            dtype=torch.long,
            device=self.device,
        )
        self._strike_body_id_tensors[agent] = torch.as_tensor(
            strike_ids or all_ids, dtype=torch.long, device=self.device
        )
        self._left_push_body_id_tensors[agent] = torch.as_tensor(
            left_push_ids,
            dtype=torch.long,
            device=self.device,
        )
        self._right_push_body_id_tensors[agent] = torch.as_tensor(
            right_push_ids,
            dtype=torch.long,
            device=self.device,
        )
        self._torso_contact_body_id_tensors[agent] = torch.as_tensor(
            torso_contact_ids or upper_contact_ids or all_ids,
            dtype=torch.long,
            device=self.device,
        )

    def _allocate_buffers(self) -> None:
        device = self.device
        n = self.num_envs
        self._actions = {agent: torch.zeros(n, self._runtime[agent].action_dim, device=device) for agent in FIGHTERS}
        self._last_actions = {agent: torch.zeros_like(self._actions[agent]) for agent in FIGHTERS}
        self._joint_targets = {agent: torch.zeros_like(self._actions[agent]) for agent in FIGHTERS}

        self._fallen = {agent: torch.zeros(n, dtype=torch.bool, device=device) for agent in FIGHTERS}
        self._new_fall = {agent: torch.zeros(n, dtype=torch.bool, device=device) for agent in FIGHTERS}
        self._knockdown = {agent: torch.zeros(n, dtype=torch.bool, device=device) for agent in FIGHTERS}
        self._new_knockdown = {agent: torch.zeros(n, dtype=torch.bool, device=device) for agent in FIGHTERS}
        self._fall_events = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._knockdown_events = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._out_of_bounds = {agent: torch.zeros(n, dtype=torch.bool, device=device) for agent in FIGHTERS}
        self._up_z = {agent: torch.ones(n, device=device) for agent in FIGHTERS}
        self._knockdown_clock = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._candidate_body_contact_force = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._opponent_contact_attribution = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._real_opponent_contact_force = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._ground_contact_force = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._proxy_engagement = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._training_contact_force = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._eval_contact_force = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._useful_contact = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._contact_intent = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._locomotion_drive = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._attack_momentum = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._strike_speed = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._destabilizing_impact = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._topple_pressure = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._drive_pressure = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._support_break_pressure = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._opponent_destabilization = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._proof_contact = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._proof_impact = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._proof_destabilization = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._recent_attack_pressure = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._energy = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._energy_ema = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._torque_penalty = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._joint_limit_penalty = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._jitter_penalty = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._inactivity = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._spin_without_contact = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._uncontrolled_collision = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._posture_instability = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._score = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._prev_distance_to_opponent = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._prev_root_height = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._prev_root_lin_vel_w = {agent: torch.zeros(n, 3, device=device) for agent in FIGHTERS}
        self._prev_up_z = {agent: torch.ones(n, device=device) for agent in FIGHTERS}
        self._prev_support_bias = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._left_support_air_time = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._right_support_air_time = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._support_step_reward = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._push_hand_command = {
            agent: torch.where(
                torch.rand(n, device=device) < 0.5,
                -torch.ones(n, device=device),
                torch.ones(n, device=device),
            )
            for agent in FIGHTERS
        }
        self._no_engagement_clock = torch.zeros(n, device=device)

        self._winner = torch.zeros(n, dtype=torch.long, device=device)
        self._loser = torch.zeros(n, dtype=torch.long, device=device)
        self._draw = torch.zeros(n, dtype=torch.bool, device=device)
        self._match_terminal = torch.zeros(n, dtype=torch.bool, device=device)
        self._time_out = torch.zeros(n, dtype=torch.bool, device=device)

        self._episode_sums: dict[str, dict[str, torch.Tensor]] = {agent: {} for agent in FIGHTERS}
        self._episode_counts = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._last_reward_terms: dict[str, dict[str, torch.Tensor]] = {agent: {} for agent in FIGHTERS}
        self._refresh_combat_features(advance=False)

    def _configure_replay(self) -> None:
        if not self.cfg.replay.enabled:
            return
        path = self.cfg.replay.path
        if not path:
            root = Path(self.cfg.log_dir or "logs/isaac_fight") / "replays"
            path = str(root / f"match_{int(time.time())}.jsonl")
        header = ReplayHeader(
            metadata={
                "arena_radius": self.cfg.arena.radius,
                "fighter_a": self.cfg.fighter_a.robot_name,
                "fighter_b": self.cfg.fighter_b.robot_name,
                "num_envs": self.num_envs,
            }
        )
        self._replay = MatchReplayRecorder(path, header=header)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        fighter_cfgs = {FIGHTER_A: self.cfg.fighter_a, FIGHTER_B: self.cfg.fighter_b}
        for agent in FIGHTERS:
            raw = actions.get(agent)
            if raw is None:
                raw = torch.zeros_like(self._actions[agent])
            raw = torch.nan_to_num(raw.to(self.device), nan=0.0, posinf=1.0, neginf=-1.0)
            raw = torch.clamp(raw, -1.0, 1.0)
            if raw.shape[-1] != self._runtime[agent].action_dim:
                raise RuntimeError(
                    f"{agent} action has shape {tuple(raw.shape)}, expected last dim {self._runtime[agent].action_dim}"
                )
            raw = self._apply_standing_warmup_action_gate(raw)
            self._last_actions[agent].copy_(self._actions[agent])
            smoothing = float(fighter_cfgs[agent].action_smoothing)
            self._actions[agent].mul_(smoothing).add_(raw * (1.0 - smoothing))
            self._joint_targets[agent] = self._compute_joint_targets(agent)

    def _apply_action(self) -> None:
        for agent, robot in self.robots.items():
            robot.set_joint_position_target(self._joint_targets[agent], joint_ids=self._runtime[agent].joint_ids)

    def _compute_joint_targets(self, agent: str) -> torch.Tensor:
        robot = self.robots[agent]
        ids = self._runtime[agent].joint_ids
        default = robot.data.default_joint_pos[:, ids]
        action_scale = self._action_scale_tensors.get(agent)
        if action_scale is None:
            action_scale = torch.full_like(self._actions[agent], self._runtime[agent].action_scale)
        target = default + self._actions[agent] * action_scale
        limits = getattr(robot.data, "soft_joint_pos_limits", None)
        if limits is not None:
            lo = limits[:, ids, 0]
            hi = limits[:, ids, 1]
            target = torch.maximum(torch.minimum(target, hi), lo)
        return target

    def _apply_standing_warmup_action_gate(self, raw: torch.Tensor) -> torch.Tensor:
        if not self.cfg.curriculum.enabled:
            return raw
        return raw * self._standing_warmup_action_gate().unsqueeze(-1)

    def _standing_warmup_action_gate(self) -> torch.Tensor:
        if not self.cfg.curriculum.enabled:
            return torch.ones(self.num_envs, device=self.device)
        hold_s = max(0.0, float(getattr(self.cfg.curriculum, "action_hold_s", 0.0)))
        ramp_s = max(1.0e-6, float(getattr(self.cfg.curriculum, "action_ramp_s", 0.0)))
        episode_time = self.episode_length_buf.float() * self.step_dt
        return torch.clamp((episode_time - hold_s) / ramp_s, 0.0, 1.0)

    @staticmethod
    def _joint_action_scale_multiplier(joint_name: str) -> float:
        lower = joint_name.lower()
        if "waist" in lower or "torso" in lower:
            return 0.10
        if "wrist" in lower:
            return 0.18
        if "shoulder" in lower or "elbow" in lower:
            return 0.35
        if "ankle" in lower:
            return 0.65
        return 1.0

    def _get_observations(self) -> dict[str, torch.Tensor]:
        return {agent: self._obs_builder.build(self, agent, opponent_of(agent)) for agent in FIGHTERS}

    def _get_states(self) -> torch.Tensor:
        obs = self._get_observations()
        return torch.cat([obs[agent].reshape(self.num_envs, -1) for agent in FIGHTERS], dim=-1)

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        rewards: dict[str, torch.Tensor] = {}
        log_reward_terms = self._should_log_reward_terms()
        for agent in FIGHTERS:
            opponent = opponent_of(agent)
            breakdown = self._reward_computer.compute(self, agent, opponent)
            rewards[agent] = breakdown.total
            self._last_reward_terms[agent] = breakdown.terms
            self._accumulate_episode_terms(agent, self._training_episode_terms(agent, opponent, breakdown))
            if log_reward_terms:
                self.extras[agent]["reward_terms"] = breakdown.detached_mean_dict()
        self._write_replay_step(rewards)
        self._commit_history()
        return rewards

    def _should_log_reward_terms(self) -> bool:
        interval = int(getattr(self.cfg.diagnostics, "reward_terms_interval", 1))
        return interval <= 1 or self.common_step_counter % interval == 0

    def _training_episode_terms(self, agent: str, opponent: str, breakdown) -> dict[str, torch.Tensor]:  # noqa: ANN001
        reward_terms = breakdown.terms
        clean_attack = self._clean_attack_credit(agent, opponent)
        opponent_fall = self._new_fall[opponent].float()
        opponent_knockdown = self._new_knockdown[opponent].float()
        proof_gate = (self._proof_impact[agent] > 0.0).float()
        episode_terms = {
            "total_reward": breakdown.total,
            **reward_terms,
            "combat_useful_contact": self._useful_contact[agent],
            "combat_locomotion_drive": self._locomotion_drive[agent],
            "combat_attack_momentum": self._attack_momentum[agent],
            "combat_destabilizing_impact": self._destabilizing_impact[agent],
            "combat_topple_pressure": self._topple_pressure[agent],
            "combat_drive_pressure": self._drive_pressure[agent],
            "combat_support_break_pressure": self._support_break_pressure[agent],
            "combat_training_contact_force": self._training_contact_force[agent],
            "combat_eval_contact_force": self._eval_contact_force[agent],
            "combat_proof_contact": self._proof_contact[agent],
            "combat_proof_impact": self._proof_impact[agent],
            "combat_recent_attack_pressure": self._recent_attack_pressure[agent],
            "combat_opponent_fall_events": opponent_fall,
            "combat_proof_opponent_fall_events": opponent_fall * proof_gate,
            "combat_clean_opponent_fall_events": opponent_fall * clean_attack,
            "combat_opponent_knockdown_events": opponent_knockdown,
            "combat_proof_opponent_knockdown_events": opponent_knockdown * proof_gate,
            "combat_self_fall_events": self._new_fall[agent].float(),
            "combat_mutual_fall_events": self._new_fall[agent].float() * self._fallen[opponent].float(),
        }
        return episode_terms

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self._refresh_combat_features(advance=True)
        self._time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.curriculum.enabled and self.cfg.curriculum.no_engagement_timeout_s > 0.0:
            no_engagement_timeout = self._no_engagement_clock >= self.cfg.curriculum.no_engagement_timeout_s
            self._time_out = self._time_out | no_engagement_timeout
        knockout = {agent: self._knockdown_clock[agent] >= self.cfg.rules.knockout_grace_s for agent in FIGHTERS}
        terminal_by_loss = (
            knockout[FIGHTER_A] | knockout[FIGHTER_B] | self._out_of_bounds[FIGHTER_A] | self._out_of_bounds[FIGHTER_B]
        )
        self._match_terminal = terminal_by_loss | self._time_out
        self._winner, self._loser, self._draw = self._rule_engine.assign_winner(
            terminal=self._match_terminal,
            time_out=self._time_out,
            knockout_a=knockout[FIGHTER_A],
            knockout_b=knockout[FIGHTER_B],
            oob_a=self._out_of_bounds[FIGHTER_A],
            oob_b=self._out_of_bounds[FIGHTER_B],
            score_a=self._score[FIGHTER_A],
            score_b=self._score[FIGHTER_B],
        )
        terminated = {agent: terminal_by_loss for agent in FIGHTERS}
        time_outs = {agent: self._time_out & ~terminal_by_loss for agent in FIGHTERS}
        return terminated, time_outs

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if hasattr(self, "_episode_sums"):
            self._publish_episode_logs(env_ids)
        super()._reset_idx(env_ids)
        self.reset_agent_state(env_ids, FIGHTER_A)
        self.reset_agent_state(env_ids, FIGHTER_B)
        if hasattr(self, "_actions"):
            self._reset_buffers(env_ids)
            self._refresh_combat_features(advance=False)

    def reset_agent_state(self, env_ids: torch.Tensor, agent: str) -> None:
        """Reset one fighter articulation for the selected vectorized environments."""

        robot = self.robots[agent]
        fighter_cfg = self.cfg.fighter_a if agent == FIGHTER_A else self.cfg.fighter_b
        joint_pos = robot.data.default_joint_pos[env_ids].clone()
        joint_vel = robot.data.default_joint_vel[env_ids].clone()
        if joint_vel.numel() > 0:
            joint_vel += 0.02 * (2.0 * torch.rand_like(joint_vel) - 1.0)

        root_state = robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        xy_noise = fighter_cfg.spawn_xy_noise * (2.0 * torch.rand(len(env_ids), 2, device=self.device) - 1.0)
        root_state[:, 0] += fighter_cfg.spawn_xy[0] + xy_noise[:, 0]
        root_state[:, 1] += fighter_cfg.spawn_xy[1] + xy_noise[:, 1]
        yaw = torch.full((len(env_ids),), fighter_cfg.spawn_yaw, device=self.device)
        yaw += fighter_cfg.spawn_yaw_noise * (2.0 * torch.rand_like(yaw) - 1.0)
        root_state[:, 3:7] = quat_from_yaw(yaw)
        root_state[:, 7:] = 0.0
        forward_speed = float(fighter_cfg.spawn_forward_speed)
        if forward_speed != 0.0:
            noise = float(fighter_cfg.spawn_forward_speed_noise)
            speed = forward_speed + noise * (2.0 * torch.rand_like(yaw) - 1.0)
            root_state[:, 7] = speed * torch.cos(yaw)
            root_state[:, 8] = speed * torch.sin(yaw)

        robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _reset_buffers(self, env_ids: torch.Tensor) -> None:
        for agent in FIGHTERS:
            self._actions[agent][env_ids] = 0.0
            self._last_actions[agent][env_ids] = 0.0
            self._joint_targets[agent][env_ids] = self._compute_joint_targets(agent)[env_ids]
            self._fallen[agent][env_ids] = False
            self._new_fall[agent][env_ids] = False
            self._knockdown[agent][env_ids] = False
            self._new_knockdown[agent][env_ids] = False
            self._fall_events[agent][env_ids] = 0.0
            self._knockdown_events[agent][env_ids] = 0.0
            self._out_of_bounds[agent][env_ids] = False
            self._knockdown_clock[agent][env_ids] = 0.0
            self._candidate_body_contact_force[agent][env_ids] = 0.0
            self._opponent_contact_attribution[agent][env_ids] = 0.0
            self._real_opponent_contact_force[agent][env_ids] = 0.0
            self._ground_contact_force[agent][env_ids] = 0.0
            self._proxy_engagement[agent][env_ids] = 0.0
            self._training_contact_force[agent][env_ids] = 0.0
            self._eval_contact_force[agent][env_ids] = 0.0
            self._useful_contact[agent][env_ids] = 0.0
            self._contact_intent[agent][env_ids] = 0.0
            self._locomotion_drive[agent][env_ids] = 0.0
            self._attack_momentum[agent][env_ids] = 0.0
            self._strike_speed[agent][env_ids] = 0.0
            self._destabilizing_impact[agent][env_ids] = 0.0
            self._topple_pressure[agent][env_ids] = 0.0
            self._drive_pressure[agent][env_ids] = 0.0
            self._support_break_pressure[agent][env_ids] = 0.0
            self._opponent_destabilization[agent][env_ids] = 0.0
            self._proof_contact[agent][env_ids] = 0.0
            self._proof_impact[agent][env_ids] = 0.0
            self._proof_destabilization[agent][env_ids] = 0.0
            self._recent_attack_pressure[agent][env_ids] = 0.0
            self._posture_instability[agent][env_ids] = 0.0
            self._energy[agent][env_ids] = 0.0
            self._energy_ema[agent][env_ids] = 0.0
            self._score[agent][env_ids] = 0.0
            opponent = opponent_of(agent)
            distance = torch.linalg.norm((self.root_pos(opponent) - self.root_pos(agent))[env_ids, :2], dim=-1)
            self._prev_distance_to_opponent[agent][env_ids] = distance.detach()
            self._prev_root_height[agent][env_ids] = self.root_pos(agent)[env_ids, 2].detach()
            self._prev_root_lin_vel_w[agent][env_ids] = self.root_lin_vel_w(agent)[env_ids].detach()
            self._prev_up_z[agent][env_ids] = self._rule_engine.up_axis_z(self.root_quat(agent))[env_ids].detach()
            self._prev_support_bias[agent][env_ids] = self._support_bias(agent)[env_ids].detach()
            self._left_support_air_time[agent][env_ids] = 0.0
            self._right_support_air_time[agent][env_ids] = 0.0
            self._support_step_reward[agent][env_ids] = 0.0
            self._randomize_push_hand(env_ids, agent)
            for tensor in self._episode_sums[agent].values():
                tensor[env_ids] = 0.0
            self._episode_counts[agent][env_ids] = 0.0
        self._winner[env_ids] = 0
        self._loser[env_ids] = 0
        self._draw[env_ids] = False
        self._match_terminal[env_ids] = False
        self._time_out[env_ids] = False
        self._no_engagement_clock[env_ids] = 0.0

    def _refresh_combat_features(self, advance: bool) -> None:
        for agent in FIGHTERS:
            root_pos = self.root_pos(agent)
            root_quat = self.root_quat(agent)
            runtime = self._runtime[agent]
            previous_fallen = self._fallen[agent].clone()
            previous_knockdown = self._knockdown[agent].clone()
            self._up_z[agent] = self._rule_engine.up_axis_z(root_quat)
            self._fallen[agent] = self._rule_engine.fallen(root_pos, root_quat, runtime.default_base_height)
            self._new_fall[agent] = self._fallen[agent] & ~previous_fallen
            self._knockdown[agent] = self._rule_engine.knockdown(root_pos, root_quat, runtime.default_base_height)
            self._new_knockdown[agent] = self._knockdown[agent] & ~previous_knockdown
            self._out_of_bounds[agent] = self._rule_engine.out_of_bounds(root_pos, self.cfg.arena.radius)
            if advance:
                self._update_support_air_time(agent)

        for agent in FIGHTERS:
            opponent = opponent_of(agent)
            self._update_contact_and_effort(agent, opponent)
            if advance:
                self._knockdown_clock[agent] = torch.where(
                    self._knockdown[agent],
                    self._knockdown_clock[agent] + self.step_dt,
                    torch.zeros_like(self._knockdown_clock[agent]),
                )
                self._fall_events[agent] += self._new_fall[agent].float()
                self._knockdown_events[agent] += self._new_knockdown[agent].float()
                clean_attack = self._clean_attack_credit(agent, opponent)
                opponent_fall_score = self._new_fall[opponent].float() * clean_attack
                opponent_knockdown_score = self._new_knockdown[opponent].float() * clean_attack
                mutual_fall_score = self._new_fall[agent].float() * self._fallen[opponent].float()
                self._score[agent] += (
                    0.10 * self._attack_momentum[agent]
                    + 0.20 * self._useful_contact[agent]
                    + 0.40 * self._destabilizing_impact[agent]
                    + 0.35 * self._topple_pressure[agent]
                    + 0.25 * self._drive_pressure[agent]
                    + 0.25 * self._support_break_pressure[agent]
                    + 0.30 * self._proof_destabilization[agent]
                    + 0.20 * self._recent_attack_pressure[agent]
                    + 2.0 * opponent_fall_score
                    + 5.0 * opponent_knockdown_score
                    - 3.0 * self._new_fall[agent].float()
                    - 2.5 * mutual_fall_score
                    + 0.002
                    * torch.clamp(
                        1.0 - torch.linalg.norm(self.root_pos(agent)[:, :2], dim=-1) / self.cfg.arena.radius, 0.0, 1.0
                    )
                )
        if advance:
            self._update_engagement_clock()

        if not advance:
            self._commit_history(only_if_uninitialized=True)

    def _update_engagement_clock(self) -> None:
        if not self.cfg.curriculum.enabled:
            self._no_engagement_clock.zero_()
            return
        min_contact = self.cfg.curriculum.engagement_min_training_contact * self.cfg.contact.force_normalizer
        contact = (
            (self._training_contact_force[FIGHTER_A] > min_contact)
            | (self._training_contact_force[FIGHTER_B] > min_contact)
            | (self._eval_contact_force[FIGHTER_A] > min_contact)
            | (self._eval_contact_force[FIGHTER_B] > min_contact)
        )
        episode_time = self.episode_length_buf.float() * self.step_dt
        in_grace = episode_time < self.cfg.curriculum.no_engagement_grace_s
        self._no_engagement_clock = torch.where(
            contact | in_grace,
            torch.zeros_like(self._no_engagement_clock),
            self._no_engagement_clock + self.step_dt,
        )

    def _commit_history(self, only_if_uninitialized: bool = False) -> None:
        for agent in FIGHTERS:
            opponent = opponent_of(agent)
            distance = torch.linalg.norm((self.root_pos(opponent) - self.root_pos(agent))[:, :2], dim=-1)
            if only_if_uninitialized and not torch.all(self._prev_distance_to_opponent[agent] == 0):
                continue
            self._prev_distance_to_opponent[agent] = distance.detach()
            self._prev_root_height[agent] = self.root_pos(agent)[:, 2].detach()
            self._prev_root_lin_vel_w[agent] = self.root_lin_vel_w(agent).detach()
            self._prev_up_z[agent] = self._up_z[agent].detach()
            self._prev_support_bias[agent] = self._support_bias(agent).detach()

    def _randomize_push_hand(self, env_ids: torch.Tensor, agent: str) -> None:
        sampled_left = torch.rand(len(env_ids), device=self.device) < 0.5
        self._push_hand_command[agent][env_ids] = torch.where(
            sampled_left,
            -torch.ones(len(env_ids), device=self.device),
            torch.ones(len(env_ids), device=self.device),
        )

    def _update_support_air_time(self, agent: str) -> None:
        left, right = self._support_contact_sides(agent)
        left_contact = left > 0.5
        right_contact = right > 0.5
        left_swing = torch.clamp(self._left_support_air_time[agent], 0.0, 0.60)
        right_swing = torch.clamp(self._right_support_air_time[agent], 0.0, 0.60)
        left_first_contact = left_contact & (left_swing > 0.08)
        right_first_contact = right_contact & (right_swing > 0.08)
        step_reward = left_first_contact.float() * torch.clamp(
            (left_swing - 0.08) / 0.24, 0.0, 1.0
        ) + right_first_contact.float() * torch.clamp((right_swing - 0.08) / 0.24, 0.0, 1.0)
        self._support_step_reward[agent] = torch.clamp(step_reward, 0.0, 1.0)
        self._left_support_air_time[agent] = torch.where(
            left_contact,
            torch.zeros_like(left_swing),
            torch.clamp(left_swing + self.step_dt, 0.0, 0.80),
        )
        self._right_support_air_time[agent] = torch.where(
            right_contact,
            torch.zeros_like(right_swing),
            torch.clamp(right_swing + self.step_dt, 0.0, 0.80),
        )

    def _update_contact_and_effort(self, agent: str, opponent: str) -> None:
        root_pos = self.root_pos(agent)
        opp_pos = self.root_pos(opponent)
        rel = opp_pos - root_pos
        distance = torch.linalg.norm(rel[:, :2], dim=-1)
        rel_dir = normalize(torch.cat((rel[:, :2], torch.zeros_like(rel[:, 2:3])), dim=-1))
        closing_speed = torch.sum((self.root_lin_vel_w(agent) - self.root_lin_vel_w(opponent)) * rel_dir, dim=-1)
        own_forward_speed = torch.relu(torch.sum(self.root_lin_vel_w(agent) * rel_dir, dim=-1))
        strike_speed = self._strike_body_speed(agent, opponent, rel_dir)
        proximity = torch.exp(-torch.square(distance / self.cfg.contact.useful_contact_distance))
        contact_proxy = torch.relu(closing_speed - self.cfg.contact.useful_contact_min_closing_speed) * proximity
        self._contact_intent[agent] = torch.clamp((0.25 + torch.relu(closing_speed)) * proximity, 0.0, 2.0)
        support_force = self._support_contact_force(agent)
        support_gate = torch.clamp(support_force / self.cfg.contact.force_normalizer, 0.0, 1.0)
        locomotion_window = ((distance > 0.35) & (distance < self.cfg.contact.useful_contact_distance * 1.35)).float()
        upright_drive_gate = torch.clamp(
            (self._up_z[agent] - self.cfg.rules.knockdown_up_axis_z)
            / max(1.0 - self.cfg.rules.knockdown_up_axis_z, 1.0e-6),
            0.0,
            1.0,
        )
        self._locomotion_drive[agent] = (
            torch.clamp(own_forward_speed / self.cfg.contact.strike_speed_normalizer, 0.0, 2.0)
            * support_gate
            * locomotion_window
            * upright_drive_gate
        )

        lower_body_attack_window = distance < self.cfg.contact.useful_contact_distance
        candidate_contact_force = self._net_contact_force(agent, include_lower_body_contacts=lower_body_attack_window)
        ground_force = self._ground_or_scene_contact_force(agent)
        proxy_engagement = contact_proxy * self.cfg.contact.force_normalizer
        opp_height_drop = torch.relu(self._prev_root_height[opponent] - self.root_pos(opponent)[:, 2])
        opp_tilt_drop = torch.relu(self._prev_up_z[opponent] - self._up_z[opponent])
        destabilization_signal = (
            self.cfg.contact.destabilization_height_drop_scale * opp_height_drop
            + self.cfg.contact.destabilization_tilt_gain * opp_tilt_drop
            + 0.50 * self._new_knockdown[opponent].float()
        )
        close_to_opponent = (distance < self.cfg.contact.useful_contact_distance).float()
        self_stable_gate = (self._up_z[agent] > self.cfg.rules.fall_up_axis_z).float() * (~self._fallen[agent]).float()
        directed_gate = (
            (closing_speed > -0.10) | (destabilization_signal > 0.02) | self._new_knockdown[opponent]
        ).float()
        attribution = close_to_opponent * self_stable_gate * directed_gate
        strike_speed_term = torch.clamp(strike_speed / self.cfg.contact.strike_speed_normalizer, 0.0, 5.0)
        attack_momentum = strike_speed_term * proximity * self_stable_gate * (closing_speed > 0.05).float()
        real_opponent_force = candidate_contact_force * attribution
        self._candidate_body_contact_force[agent] = candidate_contact_force
        self._opponent_contact_attribution[agent] = attribution
        self._real_opponent_contact_force[agent] = real_opponent_force
        self._ground_contact_force[agent] = ground_force
        self._proxy_engagement[agent] = proxy_engagement
        proxy_gain = self._effective_proxy_gain()
        self._training_contact_force[agent] = torch.maximum(real_opponent_force, proxy_engagement * proxy_gain)
        self._eval_contact_force[agent] = real_opponent_force
        force_term = torch.clamp(self._training_contact_force[agent] / self.cfg.contact.force_normalizer, 0.0, 5.0)
        physical_contact_gate = (
            self._eval_contact_force[agent]
            > self.cfg.contact.proof_contact_force_fraction * self.cfg.contact.force_normalizer
        ).float()
        proxy_contact_gate = (
            (contact_proxy > self.cfg.contact.proxy_contact_min).float() * close_to_opponent * self_stable_gate
        )
        training_contact_gate = torch.maximum(physical_contact_gate, proxy_contact_gate)
        physical_force_term = torch.clamp(self._eval_contact_force[agent] / self.cfg.contact.force_normalizer, 0.0, 5.0)
        destabilizing_impact = (
            force_term
            * strike_speed_term
            * training_contact_gate
            * (1.0 + torch.clamp(destabilization_signal, 0.0, 2.0))
        )
        proof_destabilizing_impact = (
            physical_force_term
            * strike_speed_term
            * physical_contact_gate
            * (1.0 + torch.clamp(destabilization_signal, 0.0, 2.0))
        )
        opp_lateral_ang_vel = torch.linalg.norm(self.root_ang_vel_b(opponent)[:, :2], dim=-1)
        opp_tilt = torch.relu(1.0 - self._up_z[opponent])
        topple_signal = 0.50 * torch.clamp(opp_lateral_ang_vel / 4.0, 0.0, 2.0) + torch.clamp(
            opp_tilt / max(1.0 - self.cfg.rules.knockdown_up_axis_z, 1.0e-6), 0.0, 2.0
        )
        topple_pressure = force_term * training_contact_gate * topple_signal
        proof_topple_pressure = physical_force_term * physical_contact_gate * topple_signal
        opp_root_vel_xy = self.root_lin_vel_w(opponent)[:, :2]
        opp_velocity_delta_xy = opp_root_vel_xy - self._prev_root_lin_vel_w[opponent][:, :2]
        opp_drive_speed = torch.linalg.norm(opp_root_vel_xy, dim=-1)
        opp_drive_impulse = torch.linalg.norm(opp_velocity_delta_xy, dim=-1) / max(self.step_dt * 12.0, 1.0e-6)
        drive_signal = 0.25 * torch.clamp(
            opp_drive_speed / self.cfg.contact.strike_speed_normalizer, 0.0, 2.0
        ) + 0.75 * torch.clamp(
            opp_drive_impulse,
            0.0,
            3.0,
        )
        drive_pressure = force_term * strike_speed_term * training_contact_gate * drive_signal
        proof_drive_pressure = physical_force_term * strike_speed_term * physical_contact_gate * drive_signal
        support_break_pressure = self._support_break_pressure_term(opponent, force_term, training_contact_gate)
        proof_support_break_pressure = self._support_break_pressure_term(
            opponent, physical_force_term, physical_contact_gate
        )
        self._useful_contact[agent] = (
            torch.clamp(
                force_term,
                0.0,
                5.0,
            )
            * training_contact_gate
        )

        self._attack_momentum[agent] = attack_momentum
        self._strike_speed[agent] = strike_speed * attribution
        self._destabilizing_impact[agent] = destabilizing_impact
        self._topple_pressure[agent] = topple_pressure
        self._drive_pressure[agent] = drive_pressure
        self._support_break_pressure[agent] = support_break_pressure
        self._opponent_destabilization[agent] = destabilization_signal
        self._proof_contact[agent] = (
            torch.clamp(self._eval_contact_force[agent] / self.cfg.contact.force_normalizer, 0.0, 5.0)
            * physical_contact_gate
        )
        self._proof_destabilization[agent] = self._opponent_destabilization[agent] * physical_contact_gate
        self._proof_impact[agent] = (
            self._proof_contact[agent]
            + proof_destabilizing_impact
            + proof_topple_pressure
            + proof_drive_pressure
            + proof_support_break_pressure
            + self._proof_destabilization[agent]
            + self._new_knockdown[opponent].float() * physical_contact_gate
        )
        self._update_recent_attack_pressure(agent, close_to_opponent, self_stable_gate)
        self._update_effort_penalties(agent)

        root_speed = torch.linalg.norm(self.root_lin_vel_w(agent)[:, :2], dim=-1)
        yaw_rate = torch.abs(self.root_ang_vel_b(agent)[:, 2])
        self._inactivity[agent] = (
            (root_speed < 0.05) & (distance > 0.90) & (self._useful_contact[agent] < 0.05)
        ).float()
        self._spin_without_contact[agent] = torch.relu(yaw_rate - 2.0) * (self._useful_contact[agent] < 0.05).float()
        self._uncontrolled_collision[agent] = self._useful_contact[agent] * (
            1.0 - torch.clamp(self._up_z[agent], 0.0, 1.0)
        )
        lateral_ang_vel = torch.linalg.norm(self.root_ang_vel_b(agent)[:, :2], dim=-1)
        no_real_attack = (self._proof_impact[agent] < self.cfg.contact.fall_credit_min_attack).float()
        self._posture_instability[agent] = (
            2.0 * torch.relu(self.cfg.rules.knockdown_up_axis_z + 0.35 - self._up_z[agent])
            + torch.clamp(lateral_ang_vel / 5.0, 0.0, 2.0)
        ) * no_real_attack

    def _update_recent_attack_pressure(
        self,
        agent: str,
        close_to_opponent: torch.Tensor,
        self_stable_gate: torch.Tensor,
    ) -> None:
        current = torch.maximum(
            self._proof_impact[agent],
            torch.maximum(
                self._useful_contact[agent],
                torch.maximum(self._attack_momentum[agent], self._drive_pressure[agent]),
            ),
        )
        current = torch.clamp(current, 0.0, 5.0) * close_to_opponent * self_stable_gate
        memory_s = float(self.cfg.contact.attack_memory_s)
        if memory_s <= 0.0:
            self._recent_attack_pressure[agent] = current
            return
        decay = math.exp(-float(self.step_dt) / max(memory_s, 1.0e-6))
        self._recent_attack_pressure[agent] = torch.maximum(self._recent_attack_pressure[agent] * decay, current)

    def _clean_attack_credit(self, agent: str, opponent: str) -> torch.Tensor:
        attack = torch.maximum(self._proof_impact[agent], self._recent_attack_pressure[agent])
        enough_attack = (attack >= self.cfg.contact.fall_credit_min_attack).float()
        stable = (~self._fallen[agent]).float() * (self._up_z[agent] > self.cfg.rules.fall_up_axis_z).float()
        not_mutual_crash = 1.0 - (self._new_fall[agent] & self._fallen[opponent]).float()
        return enough_attack * stable * not_mutual_crash

    def _effective_proxy_gain(self) -> float:
        base = float(self.cfg.contact.robot_contact_proxy_gain)
        if not self.cfg.curriculum.enabled or self.cfg.curriculum.proxy_gain_anneal_steps <= 0:
            return base
        progress = min(float(self.common_step_counter) / float(self.cfg.curriculum.proxy_gain_anneal_steps), 1.0)
        floor = max(0.0, min(1.0, float(self.cfg.curriculum.min_proxy_gain)))
        return base * max(floor, 1.0 - progress)

    def proxy_reward_scale(self) -> float:
        base = max(float(self.cfg.contact.robot_contact_proxy_gain), 1.0e-6)
        return self._effective_proxy_gain() / base

    def _update_effort_penalties(self, agent: str) -> None:
        robot = self.robots[agent]
        ids = self._runtime[agent].joint_ids
        joint_vel = robot.data.joint_vel[:, ids]
        torque = getattr(robot.data, "applied_torque", None)
        if torque is None:
            torque = getattr(robot.data, "computed_torque", None)
        if torque is not None:
            torque = torque[:, ids]
            power = torch.abs(torque * joint_vel)
            self._energy[agent] = torch.mean(power, dim=-1)
            self._torque_penalty[agent] = torch.mean(torch.square(torque / 120.0), dim=-1)
        else:
            self._energy[agent] = (
                torch.mean(torch.square(self._actions[agent]), dim=-1) * self.cfg.rewards.energy_normalizer
            )
            self._torque_penalty[agent] = torch.mean(torch.square(self._actions[agent]), dim=-1)
        self._energy_ema[agent].mul_(0.95).add_(0.05 * self._energy[agent])
        self._jitter_penalty[agent] = torch.mean(torch.square(self._actions[agent] - self._last_actions[agent]), dim=-1)

        limits = getattr(robot.data, "soft_joint_pos_limits", None)
        if limits is None:
            self._joint_limit_penalty[agent].zero_()
            return
        pos = robot.data.joint_pos[:, ids]
        lo = limits[:, ids, 0]
        hi = limits[:, ids, 1]
        span = torch.clamp(hi - lo, min=1.0e-5)
        margin = torch.minimum(pos - lo, hi - pos) / span
        self._joint_limit_penalty[agent] = torch.mean(torch.relu(0.10 - margin) / 0.10, dim=-1)

    def _strike_body_speed(self, agent: str, opponent: str, rel_dir: torch.Tensor) -> torch.Tensor:
        body_vel = getattr(self.robots[agent].data, "body_lin_vel_w", None)
        if body_vel is None:
            return torch.relu(torch.sum((self.root_lin_vel_w(agent) - self.root_lin_vel_w(opponent)) * rel_dir, dim=-1))
        strike_ids = self._strike_body_id_tensors.get(agent)
        if strike_ids is not None and strike_ids.numel() > 0:
            body_vel = body_vel.index_select(1, strike_ids)
        rel_body_vel = body_vel - self.root_lin_vel_w(opponent).unsqueeze(1)
        return torch.relu(torch.sum(rel_body_vel * rel_dir.unsqueeze(1), dim=-1)).amax(dim=-1)

    def _net_contact_force(
        self, agent: str, include_lower_body_contacts: torch.Tensor | bool | None = None
    ) -> torch.Tensor:
        sensor_names = (f"contact_{agent}", agent, f"{agent}_contact")
        for name in sensor_names:
            sensor = self.scene.sensors.get(name) if hasattr(self.scene, "sensors") else None
            if sensor is None or not hasattr(sensor, "data"):
                continue
            force_matrix = getattr(sensor.data, "force_matrix_w", None)
            if force_matrix is not None:
                return torch.linalg.norm(force_matrix, dim=-1).amax(dim=(-1, -2))
            if hasattr(sensor.data, "net_forces_w"):
                forces = sensor.data.net_forces_w
                all_body_force = torch.linalg.norm(forces, dim=-1).amax(dim=-1)
                upper_ids = self._upper_contact_body_id_tensors.get(agent)
                if upper_ids is None or upper_ids.numel() == 0:
                    return all_body_force
                upper_force = torch.linalg.norm(forces.index_select(1, upper_ids), dim=-1).amax(dim=-1)
                if include_lower_body_contacts is None:
                    return upper_force
                if isinstance(include_lower_body_contacts, bool):
                    return all_body_force if include_lower_body_contacts else upper_force
                return torch.where(include_lower_body_contacts.bool(), all_body_force, upper_force)
        return torch.zeros(self.num_envs, device=self.device)

    def _support_break_pressure_term(
        self,
        opponent: str,
        force_term: torch.Tensor,
        physical_contact_gate: torch.Tensor,
    ) -> torch.Tensor:
        robot = self.robots[opponent]
        body_pos_w = getattr(robot.data, "body_pos_w", None)
        support_ids = self._support_body_id_tensors.get(opponent)
        if body_pos_w is None or support_ids is None or support_ids.numel() == 0:
            return torch.zeros(self.num_envs, device=self.device)

        support_pos = body_pos_w.index_select(1, support_ids)[:, :, :2]
        support_center = support_pos.mean(dim=1)
        support_radius = torch.linalg.norm(support_pos - support_center.unsqueeze(1), dim=-1).amax(dim=-1)
        root_xy = self.root_pos_w(opponent)[:, :2]
        root_from_support = root_xy - support_center
        root_support_distance = torch.linalg.norm(root_from_support, dim=-1)
        support_escape = torch.clamp((root_support_distance - support_radius) / 0.35, 0.0, 2.0)
        support_dir = root_from_support / torch.clamp(root_support_distance.unsqueeze(-1), min=1.0e-6)
        support_drive_speed = torch.relu(torch.sum(self.root_lin_vel_w(opponent)[:, :2] * support_dir, dim=-1))
        support_drive = torch.clamp(support_drive_speed / self.cfg.contact.strike_speed_normalizer, 0.0, 2.0)
        return force_term * physical_contact_gate * (support_escape + 0.50 * support_drive)

    def _ground_or_scene_contact_force(self, agent: str) -> torch.Tensor:
        body_forces = getattr(self.robots[agent].data, "body_net_forces_w", None)
        if body_forces is None:
            return torch.zeros(self.num_envs, device=self.device)
        return torch.linalg.norm(body_forces, dim=-1).amax(dim=-1)

    def _support_contact_force(self, agent: str) -> torch.Tensor:
        body_forces = getattr(self.robots[agent].data, "body_net_forces_w", None)
        support_ids = self._support_body_id_tensors.get(agent)
        if body_forces is None or support_ids is None or support_ids.numel() == 0:
            return torch.zeros(self.num_envs, device=self.device)
        return torch.linalg.norm(body_forces.index_select(1, support_ids), dim=-1).amax(dim=-1)

    def _support_quality(self, agent: str) -> torch.Tensor:
        support_force = self._support_contact_force(agent)
        return torch.clamp(support_force / self.cfg.contact.force_normalizer, 0.0, 1.0)

    def _support_body_state_w(self, agent: str) -> tuple[torch.Tensor, torch.Tensor]:
        robot = self.robots[agent]
        support_ids = self._support_body_id_tensors.get(agent)
        body_pos_w = getattr(robot.data, "body_pos_w", None)
        body_vel_w = getattr(robot.data, "body_lin_vel_w", None)
        if body_pos_w is None or support_ids is None or support_ids.numel() == 0:
            pos = self.root_pos_w(agent).unsqueeze(1)
            vel = self.root_lin_vel_w(agent).unsqueeze(1)
            return pos, vel
        pos = body_pos_w.index_select(1, support_ids)
        if body_vel_w is None:
            vel = self.root_lin_vel_w(agent).unsqueeze(1).expand_as(pos)
        else:
            vel = body_vel_w.index_select(1, support_ids)
        return pos, vel

    def _support_center_xy(self, agent: str) -> torch.Tensor:
        pos, _ = self._support_body_state_w(agent)
        return pos[:, :, :2].mean(dim=1)

    def _support_radius(self, agent: str) -> torch.Tensor:
        pos, _ = self._support_body_state_w(agent)
        xy = pos[:, :, :2]
        center = xy.mean(dim=1, keepdim=True)
        return torch.clamp(torch.linalg.norm(xy - center, dim=-1).amax(dim=-1), min=0.10)

    def _support_mean_speed(self, agent: str) -> torch.Tensor:
        _, vel = self._support_body_state_w(agent)
        return torch.linalg.norm(vel[:, :, :2], dim=-1).mean(dim=-1)

    def _support_clearance(self, agent: str) -> torch.Tensor:
        pos, _ = self._support_body_state_w(agent)
        return torch.clamp((pos[:, :, 2].amax(dim=-1) - 0.04) / 0.14, 0.0, 1.0)

    def _support_stance_width(self, agent: str) -> torch.Tensor:
        robot = self.robots[agent]
        body_pos_w = getattr(robot.data, "body_pos_w", None)
        left_ids = self._left_support_body_id_tensors.get(agent)
        right_ids = self._right_support_body_id_tensors.get(agent)
        if (
            body_pos_w is None
            or left_ids is None
            or right_ids is None
            or left_ids.numel() == 0
            or right_ids.numel() == 0
        ):
            return 2.0 * self._support_radius(agent)
        left = body_pos_w.index_select(1, left_ids)[:, :, :2].mean(dim=1)
        right = body_pos_w.index_select(1, right_ids)[:, :, :2].mean(dim=1)
        return torch.linalg.norm(left - right, dim=-1)

    def _selected_body_contact_force(self, agent: str, body_ids: torch.Tensor | None) -> torch.Tensor:
        body_forces = getattr(self.robots[agent].data, "body_net_forces_w", None)
        if body_forces is None or body_ids is None or body_ids.numel() == 0:
            return torch.zeros(self.num_envs, device=self.device)
        return torch.linalg.norm(body_forces.index_select(1, body_ids), dim=-1).amax(dim=-1)

    def _non_support_contact_force(self, agent: str) -> torch.Tensor:
        return self._selected_body_contact_force(agent, self._upper_contact_body_id_tensors.get(agent))

    def _torso_contact_force(self, agent: str) -> torch.Tensor:
        return self._selected_body_contact_force(agent, self._torso_contact_body_id_tensors.get(agent))

    def _push_hand_command_features(self, agent: str) -> tuple[torch.Tensor, torch.Tensor]:
        command = self._push_hand_command.get(agent)
        if command is None:
            zeros = torch.zeros(self.num_envs, device=self.device)
            return zeros, zeros
        return (command < 0.0).float(), (command > 0.0).float()

    def _side_push_contact_force(self, agent: str, left_side: bool) -> torch.Tensor:
        body_ids = (
            self._left_push_body_id_tensors.get(agent) if left_side else self._right_push_body_id_tensors.get(agent)
        )
        return self._selected_body_contact_force(agent, body_ids)

    def _selected_push_contact_force(self, agent: str) -> torch.Tensor:
        left_active, right_active = self._push_hand_command_features(agent)
        return left_active * self._side_push_contact_force(agent, True) + right_active * self._side_push_contact_force(
            agent, False
        )

    def _offhand_push_contact_force(self, agent: str) -> torch.Tensor:
        left_active, right_active = self._push_hand_command_features(agent)
        return right_active * self._side_push_contact_force(agent, True) + left_active * self._side_push_contact_force(
            agent, False
        )

    def _side_push_speed(
        self,
        agent: str,
        opponent: str,
        rel_dir: torch.Tensor,
        left_side: bool,
    ) -> torch.Tensor:
        body_vel = getattr(self.robots[agent].data, "body_lin_vel_w", None)
        if body_vel is None:
            return torch.relu(torch.sum((self.root_lin_vel_w(agent) - self.root_lin_vel_w(opponent)) * rel_dir, dim=-1))
        body_ids = (
            self._left_push_body_id_tensors.get(agent) if left_side else self._right_push_body_id_tensors.get(agent)
        )
        if body_ids is None or body_ids.numel() == 0:
            return torch.zeros(self.num_envs, device=self.device)
        rel_body_vel = body_vel.index_select(1, body_ids) - self.root_lin_vel_w(opponent).unsqueeze(1)
        return torch.relu(torch.sum(rel_body_vel * rel_dir.unsqueeze(1), dim=-1)).amax(dim=-1)

    def _selected_push_speed(self, agent: str, opponent: str, rel_dir: torch.Tensor) -> torch.Tensor:
        left_active, right_active = self._push_hand_command_features(agent)
        return left_active * self._side_push_speed(
            agent, opponent, rel_dir, True
        ) + right_active * self._side_push_speed(agent, opponent, rel_dir, False)

    def _offhand_push_speed(self, agent: str, opponent: str, rel_dir: torch.Tensor) -> torch.Tensor:
        left_active, right_active = self._push_hand_command_features(agent)
        return right_active * self._side_push_speed(
            agent, opponent, rel_dir, True
        ) + left_active * self._side_push_speed(agent, opponent, rel_dir, False)

    def _side_push_reach(self, agent: str, rel_dir: torch.Tensor, left_side: bool) -> torch.Tensor:
        body_pos_w = getattr(self.robots[agent].data, "body_pos_w", None)
        if body_pos_w is None:
            return torch.zeros(self.num_envs, device=self.device)
        body_ids = (
            self._left_push_body_id_tensors.get(agent) if left_side else self._right_push_body_id_tensors.get(agent)
        )
        if body_ids is None or body_ids.numel() == 0:
            return torch.zeros(self.num_envs, device=self.device)
        hand_offset = body_pos_w.index_select(1, body_ids)[:, :, :2] - self.root_pos_w(agent)[:, :2].unsqueeze(1)
        return torch.sum(hand_offset * rel_dir[:, :2].unsqueeze(1), dim=-1).amax(dim=-1)

    def _selected_push_reach(self, agent: str, rel_dir: torch.Tensor) -> torch.Tensor:
        left_active, right_active = self._push_hand_command_features(agent)
        return left_active * self._side_push_reach(agent, rel_dir, True) + right_active * self._side_push_reach(
            agent, rel_dir, False
        )

    def _side_arm_action_magnitude(self, agent: str, left_side: bool) -> torch.Tensor:
        action_ids = (
            self._left_arm_action_id_tensors.get(agent) if left_side else self._right_arm_action_id_tensors.get(agent)
        )
        if action_ids is None or action_ids.numel() == 0:
            return torch.zeros(self.num_envs, device=self.device)
        return torch.mean(torch.square(self._actions[agent].index_select(1, action_ids)), dim=-1)

    def _selected_push_arm_action_magnitude(self, agent: str) -> torch.Tensor:
        left_active, right_active = self._push_hand_command_features(agent)
        return left_active * self._side_arm_action_magnitude(
            agent, True
        ) + right_active * self._side_arm_action_magnitude(agent, False)

    def _offhand_push_arm_action_magnitude(self, agent: str) -> torch.Tensor:
        left_active, right_active = self._push_hand_command_features(agent)
        return right_active * self._side_arm_action_magnitude(
            agent, True
        ) + left_active * self._side_arm_action_magnitude(agent, False)

    def _support_bias(self, agent: str) -> torch.Tensor:
        left = self._selected_body_contact_force(agent, self._left_support_body_id_tensors.get(agent))
        right = self._selected_body_contact_force(agent, self._right_support_body_id_tensors.get(agent))
        return (left - right) / torch.clamp(left + right, min=1.0)

    def _leg_posture_quality(self, agent: str) -> torch.Tensor:
        leg_ids = self._leg_action_id_tensors.get(agent)
        if leg_ids is None or leg_ids.numel() == 0:
            return self._stance_quality(agent)
        leg_deviation = self.joint_pos_rel(agent).index_select(1, leg_ids)
        return torch.exp(-torch.mean(torch.square(leg_deviation / 0.45), dim=-1))

    def _standing_pose_quality(self, agent: str) -> torch.Tensor:
        posture_ids = self._posture_action_id_tensors.get(agent)
        if posture_ids is None or posture_ids.numel() == 0:
            return self._stance_quality(agent)
        deviation = self.joint_pos_rel(agent).index_select(1, posture_ids)
        return torch.exp(-torch.mean(torch.square(deviation / 0.35), dim=-1))

    def _posture_action_magnitude(self, agent: str) -> torch.Tensor:
        posture_ids = self._posture_action_id_tensors.get(agent)
        if posture_ids is None or posture_ids.numel() == 0:
            return torch.mean(torch.square(self._actions[agent]), dim=-1)
        return torch.mean(torch.square(self._actions[agent].index_select(1, posture_ids)), dim=-1)

    def _selected_joint_abs_deviation(self, agent: str, joint_ids: torch.Tensor | None) -> torch.Tensor:
        if joint_ids is None or joint_ids.numel() == 0:
            return torch.zeros(self.num_envs, device=self.device)
        return torch.mean(torch.abs(self.joint_pos_rel(agent).index_select(1, joint_ids)), dim=-1)

    def _stand_still_joint_deviation(self, agent: str) -> torch.Tensor:
        return self._selected_joint_abs_deviation(agent, self._posture_action_id_tensors.get(agent))

    def _arm_motion_magnitude(self, agent: str) -> torch.Tensor:
        arm_ids = self._arm_action_id_tensors.get(agent)
        if arm_ids is None or arm_ids.numel() == 0:
            return torch.zeros(self.num_envs, device=self.device)
        arm_actions = torch.mean(torch.square(self._actions[agent].index_select(1, arm_ids)), dim=-1)
        arm_deviation = self._selected_joint_abs_deviation(agent, arm_ids)
        return arm_actions + 0.50 * arm_deviation

    def _hip_yaw_roll_deviation(self, agent: str) -> torch.Tensor:
        return self._selected_joint_abs_deviation(agent, self._hip_yaw_roll_action_id_tensors.get(agent))

    def _support_contact_sides(self, agent: str) -> tuple[torch.Tensor, torch.Tensor]:
        threshold = 0.05 * self.cfg.contact.force_normalizer
        left = self._selected_body_contact_force(agent, self._left_support_body_id_tensors.get(agent)) > threshold
        right = self._selected_body_contact_force(agent, self._right_support_body_id_tensors.get(agent)) > threshold
        return left.float(), right.float()

    def _side_support_clearance(self, agent: str, body_ids: torch.Tensor | None) -> torch.Tensor:
        robot = self.robots[agent]
        body_pos_w = getattr(robot.data, "body_pos_w", None)
        if body_pos_w is None or body_ids is None or body_ids.numel() == 0:
            return torch.zeros(self.num_envs, device=self.device)
        height = body_pos_w.index_select(1, body_ids)[:, :, 2].amax(dim=-1)
        return torch.clamp((height - 0.04) / 0.14, 0.0, 1.0)

    def _both_feet_support(self, agent: str) -> torch.Tensor:
        left, right = self._support_contact_sides(agent)
        return left * right * self._stance_quality(agent)

    def _single_stance_balance(self, agent: str) -> torch.Tensor:
        left, right = self._support_contact_sides(agent)
        single = torch.abs(left - right)
        return single * self._stance_quality(agent) * self._capture_point_support_quality(agent)

    def _feet_air_time_biped(self, agent: str) -> torch.Tensor:
        left, right = self._support_contact_sides(agent)
        left_clearance = self._side_support_clearance(agent, self._left_support_body_id_tensors.get(agent))
        right_clearance = self._side_support_clearance(agent, self._right_support_body_id_tensors.get(agent))
        single = torch.abs(left - right)
        lifted_clearance = (1.0 - left) * left_clearance + (1.0 - right) * right_clearance
        return torch.maximum(self._support_step_reward[agent], single * lifted_clearance * 0.35) * self._stance_quality(
            agent
        )

    def _capture_point_support_quality(self, agent: str) -> torch.Tensor:
        root_xy = self.root_pos_w(agent)[:, :2]
        support_center = self._support_center_xy(agent)
        support_radius = self._support_radius(agent)
        omega = math.sqrt(9.81 / max(self._runtime[agent].default_base_height, 1.0e-6))
        capture_point = root_xy + self.root_lin_vel_w(agent)[:, :2] / omega
        capture_distance = torch.linalg.norm(capture_point - support_center, dim=-1)
        return torch.exp(-torch.square(capture_distance / torch.clamp(support_radius + 0.18, min=0.18)))

    def _desired_approach_speed(self, agent: str, opponent: str) -> torch.Tensor:
        distance = torch.linalg.norm((self.root_pos(opponent) - self.root_pos(agent))[:, :2], dim=-1)
        gate = self._standing_warmup_action_gate()
        speed = 0.85 * torch.clamp((distance - 0.45) / 0.95, 0.0, 1.0)
        return speed * gate

    def _phase_features(self, agent: str, opponent: str) -> torch.Tensor:
        episode_time = self.episode_length_buf.float() * self.step_dt
        warmup_s = max(float(self.cfg.curriculum.standing_warmup_s), 1.0e-6)
        warmup_progress = torch.clamp(episode_time / warmup_s, 0.0, 1.0)
        left, right = self._support_contact_sides(agent)
        push_left, push_right = self._push_hand_command_features(agent)
        rel = self.root_pos(opponent) - self.root_pos(agent)
        rel_dir = normalize(torch.cat((rel[:, :2], torch.zeros_like(rel[:, 2:3])), dim=-1))
        return torch.stack(
            (
                torch.clamp(self._stance_quality(agent), 0.0, 1.0),
                torch.clamp(self._combat_ready(agent), 0.0, 1.0),
                self._standing_warmup_action_gate(),
                torch.clamp(
                    self._desired_approach_speed(agent, opponent) / self.cfg.contact.strike_speed_normalizer, 0.0, 1.0
                ),
                torch.clamp(self._support_quality(agent), 0.0, 1.0),
                torch.abs(left - right),
                torch.clamp(self._capture_point_support_quality(agent), 0.0, 1.0),
                warmup_progress,
                push_left,
                push_right,
                torch.clamp(self._selected_push_contact_force(agent) / self.cfg.contact.force_normalizer, 0.0, 5.0),
                torch.clamp(
                    self._selected_push_speed(agent, opponent, rel_dir) / self.cfg.contact.strike_speed_normalizer,
                    0.0,
                    5.0,
                ),
                torch.clamp(self._offhand_push_contact_force(agent) / self.cfg.contact.force_normalizer, 0.0, 5.0),
            ),
            dim=-1,
        )

    def _knee_collapse(self, agent: str) -> torch.Tensor:
        knee_ids = self._knee_action_id_tensors.get(agent)
        if knee_ids is None or knee_ids.numel() == 0:
            knee_term = torch.zeros(self.num_envs, device=self.device)
        else:
            knee_deviation = self.joint_pos_rel(agent).index_select(1, knee_ids)
            knee_term = torch.mean(torch.relu(torch.abs(knee_deviation) - 0.45), dim=-1)
        height_ratio = self.root_pos(agent)[:, 2] / max(self._runtime[agent].default_base_height, 1.0e-6)
        return knee_term + torch.relu(0.90 - height_ratio)

    def _stance_quality(self, agent: str) -> torch.Tensor:
        height_ratio = self.root_pos(agent)[:, 2] / max(self._runtime[agent].default_base_height, 1.0e-6)
        height_quality = torch.exp(-16.0 * torch.square(height_ratio - 1.0))
        upright_quality = torch.clamp(
            (self._up_z[agent] - self.cfg.rules.knockdown_up_axis_z)
            / max(1.0 - self.cfg.rules.knockdown_up_axis_z, 1.0e-6),
            0.0,
            1.0,
        )
        return height_quality * upright_quality * self._support_quality(agent)

    def _combat_ready(self, agent: str) -> torch.Tensor:
        episode_time = self.episode_length_buf.float() * self.step_dt
        warmup = torch.clamp(
            (episode_time - float(self.cfg.curriculum.standing_warmup_s)) / 0.50,
            0.0,
            1.0,
        )
        stance = torch.clamp((self._stance_quality(agent) - 0.25) / 0.50, 0.0, 1.0)
        return warmup * stance

    def _waist_action_magnitude(self, agent: str) -> torch.Tensor:
        waist_ids = self._waist_action_id_tensors.get(agent)
        if waist_ids is None or waist_ids.numel() == 0:
            return torch.zeros(self.num_envs, device=self.device)
        return torch.mean(torch.square(self._actions[agent].index_select(1, waist_ids)), dim=-1)

    def _accumulate_episode_terms(self, agent: str, terms: dict[str, torch.Tensor]) -> None:
        for name, value in terms.items():
            if name not in self._episode_sums[agent]:
                self._episode_sums[agent][name] = torch.zeros(self.num_envs, device=self.device)
            self._episode_sums[agent][name] += value.detach()
        self._episode_counts[agent] += 1.0

    def _publish_episode_logs(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        duration = self.episode_length_buf[env_ids].float() * self.step_dt
        skrl_log: dict[str, torch.Tensor] = {}
        combat_totals: dict[str, float] = {
            "win_rate": 0.0,
            "loss_rate": 0.0,
            "draw_rate": 0.0,
            "duration_s": 0.0,
            "useful_contact": 0.0,
            "locomotion_drive": 0.0,
            "contact_intent": 0.0,
            "attack_momentum": 0.0,
            "strike_speed": 0.0,
            "destabilizing_impact": 0.0,
            "topple_pressure": 0.0,
            "drive_pressure": 0.0,
            "support_break_pressure": 0.0,
            "candidate_body_contact_force": 0.0,
            "opponent_contact_attribution": 0.0,
            "real_opponent_contact_force": 0.0,
            "ground_contact_force": 0.0,
            "proxy_engagement": 0.0,
            "training_contact_force": 0.0,
            "eval_contact_force": 0.0,
            "opponent_destabilization": 0.0,
            "proof_contact": 0.0,
            "proof_impact": 0.0,
            "proof_destabilization": 0.0,
            "recent_attack_pressure": 0.0,
            "opponent_fall_events": 0.0,
            "proof_opponent_fall_events": 0.0,
            "clean_opponent_fall_events": 0.0,
            "opponent_knockdown_events": 0.0,
            "proof_opponent_knockdown_events": 0.0,
            "self_fall_events": 0.0,
            "self_knockdown_events": 0.0,
            "mutual_fall_events": 0.0,
            "inactivity": 0.0,
            "spin_without_contact": 0.0,
            "uncontrolled_collision": 0.0,
            "score": 0.0,
        }
        for agent in FIGHTERS:
            own_id = 1 if agent == FIGHTER_A else 2
            opp_id = 2 if agent == FIGHTER_A else 1
            counts = torch.clamp(self._episode_counts[agent][env_ids], min=1.0)
            log: dict[str, float] = {}
            for name, values in self._episode_sums[agent].items():
                log[f"Episode/{name}"] = float(values[env_ids].mean().item())
                if name.startswith("combat_"):
                    metric_name = name.removeprefix("combat_")
                    average_value = float((values[env_ids] / counts).mean().item())
                    log[f"Combat/{metric_name}_per_step"] = average_value
                    skrl_log[f"{agent}/Combat/{metric_name}_per_step"] = torch.tensor(average_value, device=self.device)
                    combat_totals.setdefault(metric_name, 0.0)
                    combat_totals[metric_name] += average_value
                skrl_log[f"{agent}/Episode/{name}"] = torch.tensor(log[f"Episode/{name}"], device=self.device)
            win_rate = float((self._winner[env_ids] == own_id).float().mean().item())
            loss_rate = float((self._winner[env_ids] == opp_id).float().mean().item())
            draw_rate = float(self._draw[env_ids].float().mean().item())
            duration_s = float(duration.mean().item()) if duration.numel() else 0.0
            score = float(self._score[agent][env_ids].mean().item())
            log.update(
                {
                    "Match/win_rate": win_rate,
                    "Match/loss_rate": loss_rate,
                    "Match/draw_rate": draw_rate,
                    "Match/avg_duration_s": duration_s,
                    "Match/knockdowns": float(self._knockdown_events[agent][env_ids].sum().item()),
                    "Match/self_falls": float(self._fall_events[agent][env_ids].sum().item()),
                    "Match/out_of_bounds_losses": float(self._out_of_bounds[agent][env_ids].float().sum().item()),
                    "Match/avg_candidate_body_contact_force": float(
                        self._candidate_body_contact_force[agent][env_ids].mean().item()
                    ),
                    "Match/avg_opponent_contact_attribution": float(
                        self._opponent_contact_attribution[agent][env_ids].mean().item()
                    ),
                    "Match/avg_real_opponent_contact_force": float(
                        self._real_opponent_contact_force[agent][env_ids].mean().item()
                    ),
                    "Match/avg_ground_contact_force": float(self._ground_contact_force[agent][env_ids].mean().item()),
                    "Match/avg_proxy_engagement": float(self._proxy_engagement[agent][env_ids].mean().item()),
                    "Match/avg_training_contact_force": float(
                        self._training_contact_force[agent][env_ids].mean().item()
                    ),
                    "Match/avg_eval_contact_force": float(self._eval_contact_force[agent][env_ids].mean().item()),
                    "Match/avg_attack_momentum": float(self._attack_momentum[agent][env_ids].mean().item()),
                    "Match/avg_strike_speed": float(self._strike_speed[agent][env_ids].mean().item()),
                    "Match/avg_destabilizing_impact": float(self._destabilizing_impact[agent][env_ids].mean().item()),
                    "Match/avg_topple_pressure": float(self._topple_pressure[agent][env_ids].mean().item()),
                    "Match/avg_drive_pressure": float(self._drive_pressure[agent][env_ids].mean().item()),
                    "Match/avg_support_break_pressure": float(
                        self._support_break_pressure[agent][env_ids].mean().item()
                    ),
                    "Match/proof_impact": float(self._proof_impact[agent][env_ids].mean().item()),
                    "Match/recent_attack_pressure": float(self._recent_attack_pressure[agent][env_ids].mean().item()),
                    "Match/avg_energy_use": float(self._energy_ema[agent][env_ids].mean().item()),
                    "Match/score": score,
                    "Match/policy_version": 0.0,
                    "Match/opponent_version": 0.0,
                }
            )
            for key, value in log.items():
                skrl_log[f"{agent}/{key}"] = torch.tensor(value, device=self.device)
            combat_totals["win_rate"] += win_rate
            combat_totals["loss_rate"] += loss_rate
            combat_totals["draw_rate"] += draw_rate
            combat_totals["duration_s"] += duration_s
            combat_totals["score"] += score
            self.extras[agent]["episode"] = log
        for key, value in combat_totals.items():
            skrl_log[f"Combat/mean_{key}"] = torch.tensor(value / len(FIGHTERS), device=self.device)
        self.extras["log"] = skrl_log

    def _write_replay_step(self, rewards: dict[str, torch.Tensor]) -> None:
        if self._replay is None:
            return
        if self.common_step_counter % max(1, self.cfg.replay.interval) != 0:
            return
        idx = min(max(0, int(self.cfg.replay.env_index)), self.num_envs - 1)
        payload: dict[str, Any] = {
            "winner": int(self._winner[idx].item()),
            "draw": bool(self._draw[idx].item()),
            "time_s": float(self.episode_length_buf[idx].item() * self.step_dt),
        }
        for agent in FIGHTERS:
            payload[agent] = {
                "root_pos": [float(x) for x in self.root_pos(agent)[idx].detach().cpu().tolist()],
                "root_quat": [float(x) for x in self.root_quat(agent)[idx].detach().cpu().tolist()],
                "action": [float(x) for x in self._actions[agent][idx].detach().cpu().tolist()],
                "reward": float(rewards[agent][idx].detach().cpu().item()),
                "candidate_body_contact_force": float(
                    self._candidate_body_contact_force[agent][idx].detach().cpu().item()
                ),
                "opponent_contact_attribution": float(
                    self._opponent_contact_attribution[agent][idx].detach().cpu().item()
                ),
                "real_opponent_contact_force": float(
                    self._real_opponent_contact_force[agent][idx].detach().cpu().item()
                ),
                "ground_contact_force": float(self._ground_contact_force[agent][idx].detach().cpu().item()),
                "proxy_engagement": float(self._proxy_engagement[agent][idx].detach().cpu().item()),
                "training_contact_force": float(self._training_contact_force[agent][idx].detach().cpu().item()),
                "eval_contact_force": float(self._eval_contact_force[agent][idx].detach().cpu().item()),
                "attack_momentum": float(self._attack_momentum[agent][idx].detach().cpu().item()),
                "strike_speed": float(self._strike_speed[agent][idx].detach().cpu().item()),
                "destabilizing_impact": float(self._destabilizing_impact[agent][idx].detach().cpu().item()),
                "topple_pressure": float(self._topple_pressure[agent][idx].detach().cpu().item()),
                "drive_pressure": float(self._drive_pressure[agent][idx].detach().cpu().item()),
                "support_break_pressure": float(self._support_break_pressure[agent][idx].detach().cpu().item()),
                "proof_contact": float(self._proof_contact[agent][idx].detach().cpu().item()),
                "proof_impact": float(self._proof_impact[agent][idx].detach().cpu().item()),
                "recent_attack_pressure": float(self._recent_attack_pressure[agent][idx].detach().cpu().item()),
                "fall": bool(self._fallen[agent][idx].detach().cpu().item()),
                "knockdown": bool(self._knockdown[agent][idx].detach().cpu().item()),
                "out_of_bounds": bool(self._out_of_bounds[agent][idx].detach().cpu().item()),
            }
        self._replay.write_step(self.common_step_counter, payload)

    # ----- Articulation data accessors -----

    def root_pos_w(self, agent: str) -> torch.Tensor:
        data = self.robots[agent].data
        if hasattr(data, "root_pos_w"):
            return data.root_pos_w
        return data.root_state_w[:, :3]

    def root_pos(self, agent: str) -> torch.Tensor:
        return self.root_pos_w(agent) - self.scene.env_origins

    def root_quat(self, agent: str) -> torch.Tensor:
        data = self.robots[agent].data
        if hasattr(data, "root_quat_w"):
            return data.root_quat_w
        return data.root_state_w[:, 3:7]

    def root_lin_vel_w(self, agent: str) -> torch.Tensor:
        data = self.robots[agent].data
        if hasattr(data, "root_lin_vel_w"):
            return data.root_lin_vel_w
        if hasattr(data, "root_state_w"):
            return data.root_state_w[:, 7:10]
        return torch.zeros(self.num_envs, 3, device=self.device)

    def root_lin_vel_b(self, agent: str) -> torch.Tensor:
        data = self.robots[agent].data
        if hasattr(data, "root_lin_vel_b"):
            return data.root_lin_vel_b
        return quat_apply_inverse(self.root_quat(agent), self.root_lin_vel_w(agent))

    def root_ang_vel_b(self, agent: str) -> torch.Tensor:
        data = self.robots[agent].data
        if hasattr(data, "root_ang_vel_b"):
            return data.root_ang_vel_b
        if hasattr(data, "root_ang_vel_w"):
            return quat_apply_inverse(self.root_quat(agent), data.root_ang_vel_w)
        if hasattr(data, "root_state_w"):
            return quat_apply_inverse(self.root_quat(agent), data.root_state_w[:, 10:13])
        return torch.zeros(self.num_envs, 3, device=self.device)

    def projected_gravity_b(self, agent: str) -> torch.Tensor:
        data = self.robots[agent].data
        if hasattr(data, "projected_gravity_b"):
            return data.projected_gravity_b
        gravity = torch.zeros(self.num_envs, 3, device=self.device)
        gravity[:, 2] = -1.0
        return quat_apply_inverse(self.root_quat(agent), gravity)

    def joint_pos_rel(self, agent: str) -> torch.Tensor:
        robot = self.robots[agent]
        ids = self._runtime[agent].joint_ids
        return robot.data.joint_pos[:, ids] - robot.data.default_joint_pos[:, ids]

    def joint_vel(self, agent: str) -> torch.Tensor:
        robot = self.robots[agent]
        ids = self._runtime[agent].joint_ids
        return robot.data.joint_vel[:, ids]

    def opponent_keypoint_features(self, observer_agent: str, target_agent: str) -> torch.Tensor:
        num_keypoints = OPPONENT_KEYPOINTS
        if not self.cfg.observations_cfg.opponent_keypoints_enabled or num_keypoints == 0:
            return torch.zeros(self.num_envs, num_keypoints * 6, device=self.device)
        target_pos_w, target_vel_w = self._body_keypoint_state_w(target_agent)
        rel_pos_w = target_pos_w - self.root_pos_w(observer_agent).unsqueeze(1)
        rel_vel_w = target_vel_w - self.root_lin_vel_w(observer_agent).unsqueeze(1)
        observer_quat = self.root_quat(observer_agent)
        pos_b = self._rotate_yaw_inverse_keypoints(observer_quat, rel_pos_w)
        vel_b = self._rotate_yaw_inverse_keypoints(observer_quat, rel_vel_w)
        pos_b = torch.clamp(pos_b / self.cfg.observations_cfg.keypoint_position_normalizer, -5.0, 5.0)
        vel_b = torch.clamp(vel_b * self.cfg.observations_cfg.relative_velocity_scale, -5.0, 5.0)
        return torch.cat((pos_b, vel_b), dim=-1).reshape(self.num_envs, -1)

    @staticmethod
    def _rotate_yaw_inverse_keypoints(root_quat_w: torch.Tensor, vectors_w: torch.Tensor) -> torch.Tensor:
        yaw = yaw_from_quat(root_quat_w)
        cos_yaw = torch.cos(yaw).unsqueeze(-1)
        sin_yaw = torch.sin(yaw).unsqueeze(-1)
        x_w = vectors_w[..., 0]
        y_w = vectors_w[..., 1]
        vectors_b = torch.empty_like(vectors_w)
        vectors_b[..., 0] = cos_yaw * x_w + sin_yaw * y_w
        vectors_b[..., 1] = -sin_yaw * x_w + cos_yaw * y_w
        vectors_b[..., 2] = vectors_w[..., 2]
        return vectors_b

    def _body_keypoint_state_w(self, agent: str) -> tuple[torch.Tensor, torch.Tensor]:
        robot = self.robots[agent]
        count = OPPONENT_KEYPOINTS
        body_pos_w = getattr(robot.data, "body_pos_w", None)
        body_vel_w = getattr(robot.data, "body_lin_vel_w", None)
        fallback_pos = self.root_pos_w(agent)
        fallback_vel = self.root_lin_vel_w(agent)
        ids = self._keypoint_body_id_tensors.get(agent)
        if ids is None or ids.numel() == 0:
            return fallback_pos.unsqueeze(1).expand(-1, count, -1), fallback_vel.unsqueeze(1).expand(-1, count, -1)

        gather_ids = torch.clamp(ids[:count], min=0)
        fallback_pos_batched = fallback_pos.unsqueeze(1).expand(-1, count, -1)
        fallback_vel_batched = fallback_vel.unsqueeze(1).expand(-1, count, -1)

        if body_pos_w is None:
            positions = fallback_pos_batched
        else:
            positions = body_pos_w.index_select(1, gather_ids)
            valid_pos = (ids[:count] >= 0).view(1, -1, 1)
            positions = torch.where(valid_pos, positions, fallback_pos_batched)

        if body_vel_w is None:
            velocities = fallback_vel_batched
        else:
            velocities = body_vel_w.index_select(1, gather_ids)
            valid_vel = (ids[:count] >= 0).view(1, -1, 1)
            velocities = torch.where(valid_vel, velocities, fallback_vel_batched)
        return positions, velocities

    def close(self) -> None:
        if self._replay is not None:
            self._replay.close()
            self._replay = None
        super().close()
