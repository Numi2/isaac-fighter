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


def action_dim_for_fighter(fighter: FighterCfg) -> int:
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
    spawn_forward_speed: float = 0.0
    spawn_forward_speed_noise: float = 0.0
    action_scale: float | None = None
    action_smoothing: float = 0.35
    controlled_joint_names: tuple[str, ...] = ()
    strict_joint_names: bool = True


@configclass
class ArenaCfg:
    """Arena settings."""

    radius: float = 2.0
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
    knockout_grace_s: float = 0.85
    timer_decision_margin: float = 2.0
    simultaneous_loss_is_draw: bool = True


@configclass
class ContactCfg:
    """Contact detection and logging settings."""

    force_normalizer: float = 600.0
    strike_speed_normalizer: float = 2.5
    useful_contact_distance: float = 1.25
    useful_contact_min_closing_speed: float = 0.0
    robot_contact_proxy_gain: float = 0.85
    proof_contact_force_fraction: float = 0.04
    proxy_contact_min: float = 0.04
    attack_memory_s: float = 0.75
    fall_credit_min_attack: float = 0.12
    destabilization_height_drop_scale: float = 3.0
    destabilization_tilt_gain: float = 1.75


@configclass
class PerturbationCfg:
    """Adversarial balance perturbations for standing and early-contact competence."""

    enabled: bool = True
    probability: float = 0.70
    time_min_s: float = 0.45
    time_max_s: float = 2.35
    linear_velocity_min: float = 0.20
    linear_velocity_max: float = 0.75
    angular_velocity_min: float = 0.15
    angular_velocity_max: float = 1.10
    recovery_window_s: float = 1.40


@configclass
class ObservationCfg:
    """Observation normalization settings."""

    temporal_memory_s: float = 0.45
    base_linear_velocity_scale: float = 0.5
    base_angular_velocity_scale: float = 0.2
    relative_position_normalizer: float = 3.5
    relative_velocity_scale: float = 0.5
    keypoint_position_normalizer: float = 2.5
    joint_position_scale: float = 1.0
    joint_velocity_scale: float = 0.05
    clip_joint_velocity: float = 5.0
    observation_clip: float = 10.0
    opponent_keypoints_enabled: bool = True
    opponent_keypoint_body_patterns: tuple[str, ...] = (
        ".*pelvis.*|.*base.*",
        ".*torso.*|.*waist.*",
        ".*head.*|.*neck.*",
        ".*left.*wrist.*|.*left.*hand.*|.*left.*elbow.*",
        ".*right.*wrist.*|.*right.*hand.*|.*right.*elbow.*",
        ".*left.*foot.*|.*left.*ankle.*",
        ".*right.*foot.*|.*right.*ankle.*",
    )


@configclass
class CurriculumCfg:
    """Fast-contact bootstrap settings."""

    enabled: bool = True
    standing_warmup_s: float = 1.25
    action_hold_s: float = 0.80
    action_ramp_s: float = 0.80
    no_engagement_timeout_s: float = 3.0
    no_engagement_grace_s: float = 1.5
    engagement_min_training_contact: float = 0.02
    proxy_gain_anneal_steps: int = 50_000
    min_proxy_gain: float = 0.15


@configclass
class RewardScalesCfg:
    """Reward weights. All penalties are configured as positive magnitudes."""

    upright_stability: float = 0.03
    balance_recovery: float = 0.03
    standing_height: float = 2.50
    support_contact: float = 1.60
    low_base_height: float = 8.00
    standing_pose: float = 1.60
    warmup_action_restraint: float = 1.20
    stand_still_joint_deviation: float = 1.00
    arm_motion_restraint: float = 0.80
    hip_yaw_roll_deviation: float = 0.80
    center_of_mass_over_support: float = 2.00
    capture_point_support: float = 1.50
    both_feet_support_warmup: float = 1.20
    foot_support_quality: float = 1.80
    foot_slip: float = 1.60
    base_pitch_roll: float = 3.00
    angular_stumble: float = 1.80
    knee_collapse: float = 2.40
    leg_extension_posture: float = 1.40
    perturbation_recovery: float = 3.00
    perturbation_collapse: float = 8.00
    airborne_without_attack: float = 4.00
    fall_early: float = 12.00
    recovery_reward: float = 2.40
    backward_motion: float = 3.00
    backward_lean: float = 4.00
    waist_action: float = 0.80
    controlled_approach: float = 1.80
    velocity_command_tracking: float = 2.20
    yaw_heading_tracking: float = 1.50
    locomotion_drive: float = 2.40
    forward_step_progress: float = 2.20
    retreat_from_opponent: float = 2.40
    approach_with_feet_gate: float = 1.80
    stance_width: float = 1.20
    foot_clearance: float = 0.80
    feet_air_time_biped: float = 0.70
    single_stance_balance: float = 0.90
    cadence_or_alternating_support: float = 0.70
    root_height_velocity_down: float = 2.60
    torso_only_motion: float = 3.20
    contact_intent: float = 2.20
    attack_momentum: float = 2.80
    arena_control: float = 0.005
    useful_contact: float = 6.00
    stable_contact_attack: float = 3.00
    limb_contact_reward: float = 2.60
    one_hand_push_setup: float = 1.80
    one_hand_push_contact: float = 4.00
    one_hand_push_balance: float = 2.50
    one_hand_push_destabilize: float = 3.50
    offhand_push_penalty: float = 1.40
    torso_charge_reward: float = 1.50
    bad_contact_penalty: float = 3.00
    destabilizing_impact: float = 8.00
    topple_pressure: float = 7.00
    drive_pressure: float = 5.00
    support_break_pressure: float = 6.50
    opponent_tilt_delta: float = 4.00
    opponent_height_drop_delta: float = 3.20
    opponent_support_break: float = 4.00
    impulse_direction_reward: float = 2.40
    opponent_fall: float = 18.00
    opponent_destabilization: float = 5.00
    opponent_knockdown: float = 30.00
    clean_knockdown_bonus: float = 18.00
    impact_balance: float = 4.00
    impact_self_destabilization: float = 8.00
    posture_instability: float = 1.80
    mutual_fall: float = 26.00
    mutual_fall_hard_penalty: float = 20.00
    stay_inside: float = 0.01
    self_contact_abuse: float = 2.80
    wall_boundary_escape: float = 4.00
    energy: float = 0.015
    self_fall: float = 16.00
    out_of_bounds: float = 10.00
    excessive_torque: float = 0.025
    joint_limit_abuse: float = 0.80
    joint_limit_slam: float = 1.20
    jitter: float = 0.12
    action_rate: float = 0.20
    torque_spike: float = 0.20
    inactivity: float = 1.50
    spin_without_contact: float = 1.20
    spin_flail_penalty: float = 1.80
    uncontrolled_collision: float = 2.00
    passive_survival: float = 2.00
    contact_without_progress: float = 2.50
    warmup_stand_reward: float = 3.50
    combat_gate: float = 0.80
    progressive_attack_gate: float = 1.40
    fall_cause_credit: float = 14.00
    final_win: float = 120.0
    final_loss: float = 70.0
    final_draw: float = -25.0
    energy_normalizer: float = 500.0
    action_effort_normalizer: float = 1.0


@configclass
class ReplayCfg:
    enabled: bool = False
    path: str = ""
    env_index: int = 0
    interval: int = 1


@configclass
class DiagnosticsCfg:
    """Runtime diagnostics knobs for high-throughput training."""

    reward_terms_interval: int = 64


@configclass
class SelfPlayCfg:
    enabled: bool = True
    active_agent: str = FIGHTER_A
    pool_dir: str = "policy_pool"
    snapshot_interval: int = 50
    opponent_update_interval: int = 250
    elo_window: float = 350.0
    weakness_bias: float = 0.45
    latest_bias: float = 0.35
    side_swap_probability: float = 0.5
    live_self_play_fraction: float = 0.25
    promotion_min_proof_impact: float = 1.0e-6
    promotion_bootstrap_count: int = 1


@configclass
class GhostFighterUnitree1v1EnvCfg(DirectMARLEnvCfg):
    """Direct multi-agent environment cfg for Unitree 1v1 fighting."""

    # env
    decimation: int = 4
    episode_length_s: float = 10.0
    possible_agents: list[str] = [FIGHTER_A, FIGHTER_B]
    state_space: int = observation_dim(get_unitree_robot_spec("g1_29dof").action_dim) + observation_dim(
        get_unitree_robot_spec("h1").action_dim
    )

    # default robots: symmetric G1 self-play bootstraps emergent fighting fastest.
    fighter_a: FighterCfg = FighterCfg(robot_name="g1_29dof", spawn_xy=(-0.45, 0.0), spawn_yaw=0.0, spawn_xy_noise=0.08)
    fighter_b: FighterCfg = FighterCfg(
        robot_name="g1_29dof", spawn_xy=(0.45, 0.0), spawn_yaw=math.pi, spawn_xy_noise=0.08
    )

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=decimation)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=8.0, replicate_physics=True)

    # modular task settings
    arena: ArenaCfg = ArenaCfg()
    rules: RuleCfg = RuleCfg()
    contact: ContactCfg = ContactCfg()
    perturbations: PerturbationCfg = PerturbationCfg()
    observations_cfg: ObservationCfg = ObservationCfg()
    rewards: RewardScalesCfg = RewardScalesCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    replay: ReplayCfg = ReplayCfg()
    diagnostics: DiagnosticsCfg = DiagnosticsCfg()
    self_play: SelfPlayCfg = SelfPlayCfg()

    # Spaces are refreshed in __post_init__ from the selected robot specs.
    action_spaces: dict[str, int] = {
        FIGHTER_A: get_unitree_robot_spec("g1_29dof").action_dim,
        FIGHTER_B: get_unitree_robot_spec("g1_29dof").action_dim,
    }
    observation_spaces: dict[str, int] = {
        FIGHTER_A: observation_dim(get_unitree_robot_spec("g1_29dof").action_dim),
        FIGHTER_B: observation_dim(get_unitree_robot_spec("g1_29dof").action_dim),
    }

    def __post_init__(self):
        self.possible_agents = [FIGHTER_A, FIGHTER_B]
        dim_a = action_dim_for_fighter(self.fighter_a)
        dim_b = action_dim_for_fighter(self.fighter_b)
        self.action_spaces = {FIGHTER_A: dim_a, FIGHTER_B: dim_b}
        self.observation_spaces = {FIGHTER_A: observation_dim(dim_a), FIGHTER_B: observation_dim(dim_b)}
        self.state_space = self.observation_spaces[FIGHTER_A] + self.observation_spaces[FIGHTER_B]
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
