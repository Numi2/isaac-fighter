# Copyright (c) 2026, Isaac Fight contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for GhostFighter-Unitree-1v1-Direct-v0."""

from __future__ import annotations

import math

from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaac_fight.assets.robots.unitree import get_unitree_robot_spec

from .fighter_ids import FIGHTER_A, FIGHTER_B
from .observations import observation_dim


def action_dim_for_fighter(fighter: "FighterCfg") -> int:
    if fighter.controlled_joint_names:
        return len(fighter.controlled_joint_names)
    return get_unitree_robot_spec(fighter.robot_name).action_dim


@configclass
class FighterCfg:
    """Per-fighter robot and control settings."""

    robot_name: str = "g1_29dof"
    spawn_xy: tuple[float, float] = (-1.20, 0.0)
    spawn_yaw: float = 0.0
    spawn_xy_noise: float = 0.10
    spawn_yaw_noise: float = 0.20
    action_scale: float | None = None
    action_smoothing: float = 0.35
    controlled_joint_names: tuple[str, ...] = ()
    strict_joint_names: bool = True


@configclass
class ArenaCfg:
    """Arena settings."""

    radius: float = 3.5
    visual_boundary: bool = True
    wall_height: float = 0.45
    wall_thickness: float = 0.10
    floor_static_friction: float = 1.0
    floor_dynamic_friction: float = 0.95
    floor_restitution: float = 0.0


@configclass
class RuleCfg:
    """Fight termination and event rules."""

    fall_height: float = 0.25
    fall_height_ratio: float = 0.38
    fall_up_axis_z: float = 0.35
    knockdown_height_ratio: float = 0.55
    knockdown_up_axis_z: float = 0.50
    knockout_grace_s: float = 1.25
    timer_decision_margin: float = 2.0
    simultaneous_loss_is_draw: bool = True


@configclass
class ContactCfg:
    """Contact detection and logging settings."""

    force_normalizer: float = 600.0
    useful_contact_distance: float = 1.95
    useful_contact_min_closing_speed: float = 0.0
    robot_contact_proxy_gain: float = 0.85
    destabilization_height_drop_scale: float = 3.0
    destabilization_tilt_gain: float = 1.75


@configclass
class ObservationCfg:
    """Observation normalization settings."""

    base_linear_velocity_scale: float = 0.5
    base_angular_velocity_scale: float = 0.2
    relative_position_normalizer: float = 3.5
    relative_velocity_scale: float = 0.5
    joint_position_scale: float = 1.0
    joint_velocity_scale: float = 0.05
    clip_joint_velocity: float = 5.0
    observation_clip: float = 10.0


@configclass
class RewardScalesCfg:
    """Reward weights. All penalties are configured as positive magnitudes."""

    upright_stability: float = 0.12
    balance_recovery: float = 0.08
    controlled_approach: float = 1.60
    contact_intent: float = 1.75
    arena_control: float = 0.02
    useful_contact: float = 4.00
    opponent_destabilization: float = 3.00
    opponent_knockdown: float = 18.00
    stay_inside: float = 0.05
    energy: float = 0.015
    self_fall: float = 7.00
    out_of_bounds: float = 10.00
    excessive_torque: float = 0.025
    joint_limit_abuse: float = 0.80
    jitter: float = 0.12
    inactivity: float = 0.80
    spin_without_contact: float = 0.80
    uncontrolled_collision: float = 2.00
    final_win: float = 60.0
    final_loss: float = 35.0
    final_draw: float = -6.0
    energy_normalizer: float = 500.0
    action_effort_normalizer: float = 1.0


@configclass
class ReplayCfg:
    enabled: bool = False
    path: str = ""
    env_index: int = 0
    interval: int = 1


@configclass
class SelfPlayCfg:
    enabled: bool = True
    active_agent: str = FIGHTER_A
    pool_dir: str = "policy_pool"
    snapshot_interval: int = 50
    opponent_update_interval: int = 10
    elo_window: float = 250.0
    weakness_bias: float = 0.65
    latest_bias: float = 0.15
    side_swap_probability: float = 0.5


@configclass
class GhostFighterUnitree1v1EnvCfg(DirectMARLEnvCfg):
    """Direct multi-agent environment cfg for Unitree 1v1 fighting."""

    # env
    decimation: int = 4
    episode_length_s: float = 30.0
    possible_agents: list[str] = [FIGHTER_A, FIGHTER_B]
    state_space = -1

    # default robots: G1 main fighter, H1 larger opponent
    fighter_a: FighterCfg = FighterCfg(robot_name="g1_29dof", spawn_xy=(-0.78, 0.0), spawn_yaw=0.0, spawn_xy_noise=0.06)
    fighter_b: FighterCfg = FighterCfg(robot_name="h1", spawn_xy=(0.78, 0.0), spawn_yaw=math.pi, spawn_xy_noise=0.06)

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=decimation)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=8.0, replicate_physics=True)

    # modular task settings
    arena: ArenaCfg = ArenaCfg()
    rules: RuleCfg = RuleCfg()
    contact: ContactCfg = ContactCfg()
    observations_cfg: ObservationCfg = ObservationCfg()
    rewards: RewardScalesCfg = RewardScalesCfg()
    replay: ReplayCfg = ReplayCfg()
    self_play: SelfPlayCfg = SelfPlayCfg()

    # Spaces are refreshed in __post_init__ from the selected robot specs.
    action_spaces: dict[str, int] = {
        FIGHTER_A: get_unitree_robot_spec("g1_29dof").action_dim,
        FIGHTER_B: get_unitree_robot_spec("h1").action_dim,
    }
    observation_spaces: dict[str, int] = {
        FIGHTER_A: observation_dim(get_unitree_robot_spec("g1_29dof").action_dim),
        FIGHTER_B: observation_dim(get_unitree_robot_spec("h1").action_dim),
    }

    def __post_init__(self):
        self.possible_agents = [FIGHTER_A, FIGHTER_B]
        dim_a = action_dim_for_fighter(self.fighter_a)
        dim_b = action_dim_for_fighter(self.fighter_b)
        self.action_spaces = {FIGHTER_A: dim_a, FIGHTER_B: dim_b}
        self.observation_spaces = {FIGHTER_A: observation_dim(dim_a), FIGHTER_B: observation_dim(dim_b)}
        self.state_space = -1
        self.sim.render_interval = self.decimation
        if hasattr(self.sim, "physx") and hasattr(self.sim.physx, "gpu_max_rigid_patch_count"):
            self.sim.physx.gpu_max_rigid_patch_count = max(self.sim.physx.gpu_max_rigid_patch_count, 2**23)
        if self.scene.env_spacing < self.arena.radius * 2.1:
            self.scene.env_spacing = self.arena.radius * 2.1


@configclass
class GhostFighterUnitree1v1PlayEnvCfg(GhostFighterUnitree1v1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.replay.enabled = False
