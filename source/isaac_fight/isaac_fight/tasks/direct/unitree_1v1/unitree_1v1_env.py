# Copyright (c) 2026, Isaac Fight contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Direct multi-agent Unitree humanoid 1v1 combat environment."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectMARLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from isaac_fight.assets.robots.unitree import get_controlled_joint_names_from_cfg_or_spec, get_unitree_robot_cfg, get_unitree_robot_spec
from isaac_fight.utils.torch_math import normalize, quat_apply_inverse, quat_from_yaw, rotate_yaw_inverse

from .fight_common import FighterRuntimeInfo
from .fight_rules import FightRuleEngine
from .fighter_ids import FIGHTER_A, FIGHTER_B, FIGHTERS, opponent_of
from .observations import CombatObservationBuilder
from .replay import MatchReplayRecorder, ReplayHeader
from .reward_terms import CombatRewardComputer
from .unitree_1v1_env_cfg import GhostFighterUnitree1v1EnvCfg


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
            from pxr import Gf, UsdGeom, UsdPhysics

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
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
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
            cfg_joint_names = get_controlled_joint_names_from_cfg_or_spec(fighter_cfg.robot_name, self._robot_cfgs.get(agent))
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

    def _allocate_buffers(self) -> None:
        device = self.device
        n = self.num_envs
        self._actions = {agent: torch.zeros(n, self._runtime[agent].action_dim, device=device) for agent in FIGHTERS}
        self._last_actions = {agent: torch.zeros_like(self._actions[agent]) for agent in FIGHTERS}
        self._joint_targets = {agent: torch.zeros_like(self._actions[agent]) for agent in FIGHTERS}

        self._fallen = {agent: torch.zeros(n, dtype=torch.bool, device=device) for agent in FIGHTERS}
        self._knockdown = {agent: torch.zeros(n, dtype=torch.bool, device=device) for agent in FIGHTERS}
        self._new_knockdown = {agent: torch.zeros(n, dtype=torch.bool, device=device) for agent in FIGHTERS}
        self._out_of_bounds = {agent: torch.zeros(n, dtype=torch.bool, device=device) for agent in FIGHTERS}
        self._up_z = {agent: torch.ones(n, device=device) for agent in FIGHTERS}
        self._knockdown_clock = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._contact_force = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._useful_contact = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._opponent_destabilization = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._energy = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._energy_ema = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._torque_penalty = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._joint_limit_penalty = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._jitter_penalty = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._inactivity = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._spin_without_contact = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._uncontrolled_collision = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._score = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._prev_distance_to_opponent = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._prev_root_height = {agent: torch.zeros(n, device=device) for agent in FIGHTERS}
        self._prev_up_z = {agent: torch.ones(n, device=device) for agent in FIGHTERS}

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
                raise RuntimeError(f"{agent} action has shape {tuple(raw.shape)}, expected last dim {self._runtime[agent].action_dim}")
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
        target = default + self._actions[agent] * self._runtime[agent].action_scale
        limits = getattr(robot.data, "soft_joint_pos_limits", None)
        if limits is not None:
            lo = limits[:, ids, 0]
            hi = limits[:, ids, 1]
            target = torch.maximum(torch.minimum(target, hi), lo)
        return target

    def _get_observations(self) -> dict[str, torch.Tensor]:
        return {agent: self._obs_builder.build(self, agent, opponent_of(agent)) for agent in FIGHTERS}

    def _get_states(self) -> torch.Tensor:
        obs = self._get_observations()
        return torch.cat([obs[agent].reshape(self.num_envs, -1) for agent in FIGHTERS], dim=-1)

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        rewards: dict[str, torch.Tensor] = {}
        for agent in FIGHTERS:
            opponent = opponent_of(agent)
            breakdown = self._reward_computer.compute(self, agent, opponent)
            rewards[agent] = breakdown.total
            self._last_reward_terms[agent] = breakdown.terms
            combat_metrics = {
                "combat_useful_contact": self._useful_contact[agent],
                "combat_contact_force": self._contact_force[agent],
                "combat_opponent_destabilization": self._opponent_destabilization[agent],
                "combat_opponent_knockdown_events": self._new_knockdown[opponent].float(),
                "combat_self_knockdown_events": self._new_knockdown[agent].float(),
                "combat_inactivity": self._inactivity[agent],
                "combat_spin_without_contact": self._spin_without_contact[agent],
                "combat_uncontrolled_collision": self._uncontrolled_collision[agent],
            }
            self._accumulate_episode_terms(agent, breakdown.terms | {"total_reward": breakdown.total} | combat_metrics)
            self.extras[agent]["reward_terms"] = breakdown.detached_mean_dict()
        self._write_replay_step(rewards)
        self._commit_history()
        return rewards

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self._refresh_combat_features(advance=True)
        self._time_out = self.episode_length_buf >= self.max_episode_length - 1
        knockout = {
            agent: self._knockdown_clock[agent] >= self.cfg.rules.knockout_grace_s for agent in FIGHTERS
        }
        terminal_by_loss = knockout[FIGHTER_A] | knockout[FIGHTER_B] | self._out_of_bounds[FIGHTER_A] | self._out_of_bounds[FIGHTER_B]
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

        robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _reset_buffers(self, env_ids: torch.Tensor) -> None:
        for agent in FIGHTERS:
            self._actions[agent][env_ids] = 0.0
            self._last_actions[agent][env_ids] = 0.0
            self._joint_targets[agent][env_ids] = self._compute_joint_targets(agent)[env_ids]
            self._fallen[agent][env_ids] = False
            self._knockdown[agent][env_ids] = False
            self._new_knockdown[agent][env_ids] = False
            self._out_of_bounds[agent][env_ids] = False
            self._knockdown_clock[agent][env_ids] = 0.0
            self._contact_force[agent][env_ids] = 0.0
            self._useful_contact[agent][env_ids] = 0.0
            self._opponent_destabilization[agent][env_ids] = 0.0
            self._energy[agent][env_ids] = 0.0
            self._energy_ema[agent][env_ids] = 0.0
            self._score[agent][env_ids] = 0.0
            for tensor in self._episode_sums[agent].values():
                tensor[env_ids] = 0.0
            self._episode_counts[agent][env_ids] = 0.0
        self._winner[env_ids] = 0
        self._loser[env_ids] = 0
        self._draw[env_ids] = False
        self._match_terminal[env_ids] = False
        self._time_out[env_ids] = False

    def _refresh_combat_features(self, advance: bool) -> None:
        for agent in FIGHTERS:
            root_pos = self.root_pos(agent)
            root_quat = self.root_quat(agent)
            runtime = self._runtime[agent]
            previous_knockdown = self._knockdown[agent].clone()
            self._up_z[agent] = self._rule_engine.up_axis_z(root_quat)
            self._fallen[agent] = self._rule_engine.fallen(root_pos, root_quat, runtime.default_base_height)
            self._knockdown[agent] = self._rule_engine.knockdown(root_pos, root_quat, runtime.default_base_height)
            self._new_knockdown[agent] = self._knockdown[agent] & ~previous_knockdown
            self._out_of_bounds[agent] = self._rule_engine.out_of_bounds(root_pos, self.cfg.arena.radius)

        for agent in FIGHTERS:
            opponent = opponent_of(agent)
            self._update_contact_and_effort(agent, opponent)
            if advance:
                self._knockdown_clock[agent] = torch.where(
                    self._knockdown[agent], self._knockdown_clock[agent] + self.step_dt, torch.zeros_like(self._knockdown_clock[agent])
                )
                self._score[agent] += (
                    0.08 * self._useful_contact[agent]
                    + 0.12 * self._opponent_destabilization[agent]
                    + 3.0 * self._new_knockdown[opponent].float()
                    + 0.01 * torch.clamp(1.0 - torch.linalg.norm(self.root_pos(agent)[:, :2], dim=-1) / self.cfg.arena.radius, 0.0, 1.0)
                )

        if not advance:
            self._commit_history(only_if_uninitialized=True)

    def _commit_history(self, only_if_uninitialized: bool = False) -> None:
        for agent in FIGHTERS:
            opponent = opponent_of(agent)
            distance = torch.linalg.norm((self.root_pos(opponent) - self.root_pos(agent))[:, :2], dim=-1)
            if only_if_uninitialized and not torch.all(self._prev_distance_to_opponent[agent] == 0):
                continue
            self._prev_distance_to_opponent[agent] = distance.detach()
            self._prev_root_height[agent] = self.root_pos(agent)[:, 2].detach()
            self._prev_up_z[agent] = self._up_z[agent].detach()

    def _update_contact_and_effort(self, agent: str, opponent: str) -> None:
        root_pos = self.root_pos(agent)
        opp_pos = self.root_pos(opponent)
        rel = opp_pos - root_pos
        distance = torch.linalg.norm(rel[:, :2], dim=-1)
        rel_dir = normalize(torch.cat((rel[:, :2], torch.zeros_like(rel[:, 2:3])), dim=-1))
        closing_speed = torch.sum((self.root_lin_vel_w(agent) - self.root_lin_vel_w(opponent)) * rel_dir, dim=-1)
        proximity = torch.exp(-torch.square(distance / self.cfg.contact.useful_contact_distance))
        contact_proxy = torch.relu(closing_speed - self.cfg.contact.useful_contact_min_closing_speed) * proximity

        sensor_force = self._net_contact_force(agent)
        force_term = torch.clamp(sensor_force / self.cfg.contact.force_normalizer, 0.0, 5.0)
        self._contact_force[agent] = sensor_force + contact_proxy * self.cfg.contact.force_normalizer
        self._useful_contact[agent] = torch.clamp(
            force_term + self.cfg.contact.robot_contact_proxy_gain * contact_proxy,
            0.0,
            5.0,
        ) * (distance < self.cfg.contact.useful_contact_distance).float()

        opp_height_drop = torch.relu(self._prev_root_height[opponent] - self.root_pos(opponent)[:, 2])
        opp_tilt_drop = torch.relu(self._prev_up_z[opponent] - self._up_z[opponent])
        self._opponent_destabilization[agent] = (
            self.cfg.contact.destabilization_height_drop_scale * opp_height_drop
            + self.cfg.contact.destabilization_tilt_gain * opp_tilt_drop
            + 0.50 * self._new_knockdown[opponent].float()
        )
        self._update_effort_penalties(agent)

        root_speed = torch.linalg.norm(self.root_lin_vel_w(agent)[:, :2], dim=-1)
        yaw_rate = torch.abs(self.root_ang_vel_b(agent)[:, 2])
        self._inactivity[agent] = ((root_speed < 0.05) & (distance > 0.90) & (self._useful_contact[agent] < 0.05)).float()
        self._spin_without_contact[agent] = torch.relu(yaw_rate - 2.0) * (self._useful_contact[agent] < 0.05).float()
        self._uncontrolled_collision[agent] = self._useful_contact[agent] * (1.0 - torch.clamp(self._up_z[agent], 0.0, 1.0))

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
            self._energy[agent] = torch.mean(torch.square(self._actions[agent]), dim=-1) * self.cfg.rewards.energy_normalizer
            self._torque_penalty[agent] = torch.mean(torch.square(self._actions[agent]), dim=-1)
        self._energy_ema[agent].mul_(0.95).add_(0.05 * self._energy[agent])
        self._jitter_penalty[agent] = torch.mean(torch.square(self._actions[agent] - self._last_actions[agent]), dim=-1)

        limits = getattr(robot.data, "soft_joint_pos_limits", None)
        if limits is None:
            self._joint_limit_penalty[agent] = torch.zeros(self.num_envs, device=self.device)
            return
        pos = robot.data.joint_pos[:, ids]
        lo = limits[:, ids, 0]
        hi = limits[:, ids, 1]
        span = torch.clamp(hi - lo, min=1.0e-5)
        margin = torch.minimum(pos - lo, hi - pos) / span
        self._joint_limit_penalty[agent] = torch.mean(torch.relu(0.10 - margin) / 0.10, dim=-1)

    def _net_contact_force(self, agent: str) -> torch.Tensor:
        sensor_names = (f"contact_{agent}", agent, f"{agent}_contact")
        for name in sensor_names:
            sensor = self.scene.sensors.get(name) if hasattr(self.scene, "sensors") else None
            if sensor is not None and hasattr(sensor, "data") and hasattr(sensor.data, "net_forces_w"):
                return torch.linalg.norm(sensor.data.net_forces_w, dim=-1).amax(dim=-1)
        body_forces = getattr(self.robots[agent].data, "body_net_forces_w", None)
        if body_forces is not None:
            return torch.linalg.norm(body_forces, dim=-1).amax(dim=-1)
        return torch.zeros(self.num_envs, device=self.device)

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
            "contact_force": 0.0,
            "opponent_destabilization": 0.0,
            "opponent_knockdown_events": 0.0,
            "self_knockdown_events": 0.0,
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
                    "Match/knockdowns": float(self._new_knockdown[agent][env_ids].float().sum().item()),
                    "Match/self_falls": float(self._fallen[agent][env_ids].float().sum().item()),
                    "Match/out_of_bounds_losses": float(self._out_of_bounds[agent][env_ids].float().sum().item()),
                    "Match/avg_contact_force": float(self._contact_force[agent][env_ids].mean().item()),
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
                "contact_force": float(self._contact_force[agent][idx].detach().cpu().item()),
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

    def close(self) -> None:
        if self._replay is not None:
            self._replay.close()
            self._replay = None
        super().close()
