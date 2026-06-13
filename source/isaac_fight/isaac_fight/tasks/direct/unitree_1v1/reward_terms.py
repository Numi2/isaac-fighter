# Copyright (c) 2026, Isaac Fight contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Modular combat reward terms."""

from __future__ import annotations

import torch

from isaac_fight.utils.torch_math import heading_error_to_target

from .fight_common import RewardBreakdown
from .fighter_ids import FIGHTER_A, FIGHTER_B


class CombatRewardComputer:
    """Computes shaped combat rewards and terminal match reward for one fighter."""

    def __init__(self, cfg):
        self.cfg = cfg

    def compute(self, env, agent: str, opponent: str) -> RewardBreakdown:
        scales = self.cfg
        root_pos = env.root_pos(agent)
        opp_pos = env.root_pos(opponent)
        rel_pos = opp_pos - root_pos
        distance = torch.linalg.norm(rel_pos[:, :2], dim=-1)
        prev_distance = env._prev_distance_to_opponent[agent]
        rel_dir = torch.cat((rel_pos[:, :2], torch.zeros_like(rel_pos[:, 2:3])), dim=-1)
        rel_dir = rel_dir / torch.clamp(torch.linalg.norm(rel_dir, dim=-1, keepdim=True), min=1.0e-6)
        root_lin_vel_w = env.root_lin_vel_w(agent)
        opp_lin_vel_w = env.root_lin_vel_w(opponent)
        root_lin_vel_b = env.root_lin_vel_b(agent)
        root_ang_vel_b = env.root_ang_vel_b(agent)
        projected_gravity_b = env.projected_gravity_b(agent)
        root_speed = torch.linalg.norm(root_lin_vel_w[:, :2], dim=-1)
        toward_speed = torch.sum(root_lin_vel_w * rel_dir, dim=-1)

        up_z = env._up_z[agent]
        upright = torch.clamp(
            (up_z - env.cfg.rules.knockdown_up_axis_z) / (1.0 - env.cfg.rules.knockdown_up_axis_z), 0.0, 1.0
        )
        lateral_ang_vel = torch.linalg.norm(root_ang_vel_b[:, :2], dim=-1)
        balance_recovery = upright * torch.exp(-0.20 * torch.square(lateral_ang_vel))
        height_ratio = root_pos[:, 2] / max(env._runtime[agent].default_base_height, 1.0e-6)
        standing_height = upright * torch.exp(-16.0 * torch.square(height_ratio - 1.0))
        support_quality = env._support_quality(agent)
        support_contact = support_quality * standing_height
        low_base_height = torch.relu(0.88 - height_ratio) * (1.0 + 2.0 * (1.0 - upright))
        standing_pose = env._standing_pose_quality(agent) * upright
        support_center = env._support_center_xy(agent)
        support_radius = env._support_radius(agent)
        root_support_distance = torch.linalg.norm(root_pos[:, :2] - support_center, dim=-1)
        capture_point_support = env._capture_point_support_quality(agent) * support_quality * upright
        center_of_mass_over_support = (
            torch.exp(-torch.square(root_support_distance / torch.clamp(support_radius + 0.12, min=0.12)))
            * support_quality
            * upright
        )
        foot_support_quality = support_quality * upright
        foot_slip = support_quality * torch.clamp(env._support_mean_speed(agent) / 1.25, 0.0, 3.0)
        base_pitch_roll = torch.linalg.norm(projected_gravity_b[:, :2], dim=-1)
        angular_stumble = torch.clamp(lateral_ang_vel / 4.0, 0.0, 3.0)
        knee_collapse = env._knee_collapse(agent)
        leg_extension_posture = env._leg_posture_quality(agent) * foot_support_quality
        perturb_active = env._perturbation_active(agent)
        perturbation_recovery = perturb_active * balance_recovery * foot_support_quality * capture_point_support
        perturbation_collapse = perturb_active * (
            low_base_height
            + 0.75 * torch.relu(1.0 - upright)
            + 0.35 * angular_stumble
            + knee_collapse
            + 2.0 * env._new_fall[agent].float()
        )
        recovery_reward = torch.relu(up_z - env._prev_up_z[agent]) * support_quality * (~env._fallen[agent]).float()
        backward_motion = torch.relu(-root_lin_vel_b[:, 0]) * (0.35 + 0.65 * upright)
        backward_lean = torch.relu(projected_gravity_b[:, 0] - 0.08) * (0.5 + 0.5 * upright)
        waist_action = env._waist_action_magnitude(agent)
        combat_gate = env._combat_ready(agent)
        episode_time = env.episode_length_buf.float() * env.step_dt
        warmup_s = float(env.cfg.curriculum.standing_warmup_s)
        warmup_gate = (episode_time < warmup_s).float()
        after_warmup = torch.clamp((episode_time - warmup_s) / 0.75, 0.0, 1.0)
        early_fall_window = torch.clamp((2.0 - episode_time) / 2.0, 0.0, 1.0)
        action_gate = env._standing_warmup_action_gate()
        warmup_action_restraint = warmup_gate * env._posture_action_magnitude(agent) * (1.0 + 2.0 * (1.0 - upright))
        stand_still_joint_deviation = env._stand_still_joint_deviation(agent) * torch.clamp(
            warmup_gate + 0.50 * (1.0 - action_gate), 0.0, 1.0
        )
        arm_motion_restraint = env._arm_motion_magnitude(agent) * (
            1.0 - torch.clamp(combat_gate * (distance < 0.70).float(), 0.0, 1.0)
        )
        hip_yaw_roll_deviation = env._hip_yaw_roll_deviation(agent) * (0.35 + 0.65 * action_gate)
        both_feet_support_warmup = warmup_gate * env._both_feet_support(agent)

        heading_error = heading_error_to_target(env.root_quat(agent), rel_pos)
        facing_gate = torch.clamp(torch.cos(heading_error), min=0.0)
        approach_delta = torch.clamp(prev_distance - distance, -0.25, 0.25)
        desired_approach_speed = env._desired_approach_speed(agent, opponent)
        lateral_speed = torch.linalg.norm(root_lin_vel_w[:, :2] - toward_speed.unsqueeze(-1) * rel_dir[:, :2], dim=-1)
        velocity_command_tracking = (
            torch.exp(-torch.square((toward_speed - desired_approach_speed) / 0.45))
            * torch.exp(-torch.square(lateral_speed / 0.35))
            * foot_support_quality
            * facing_gate
            * action_gate
            * (desired_approach_speed > 0.05).float()
        )
        yaw_heading_tracking = torch.exp(-torch.square(heading_error / 0.55)) * upright * (0.25 + 0.75 * action_gate)
        controlled_approach = approach_delta * facing_gate * upright * combat_gate
        locomotion_drive = env._locomotion_drive[agent] * facing_gate * upright * combat_gate
        forward_step_progress = torch.relu(approach_delta) * foot_support_quality * facing_gate
        retreat_from_opponent = torch.relu(-toward_speed) * (distance > 0.45).float() * (0.3 + 0.7 * upright)
        approach_with_feet_gate = torch.relu(approach_delta) * facing_gate * foot_support_quality * combat_gate
        stance_width_value = env._support_stance_width(agent)
        stance_width = torch.exp(-torch.square((stance_width_value - 0.34) / 0.24)) * upright
        foot_clearance = env._support_clearance(agent) * torch.clamp(root_speed / 1.0, 0.0, 1.0) * upright
        locomotion_phase = (desired_approach_speed > 0.05).float()
        feet_air_time_biped = env._feet_air_time_biped(agent) * locomotion_phase * facing_gate
        single_stance_balance = env._single_stance_balance(agent) * locomotion_phase
        cadence_or_alternating_support = (
            torch.clamp(torch.abs(env._support_bias(agent) - env._prev_support_bias[agent]) / 0.75, 0.0, 1.0)
            * torch.clamp(root_speed / 1.0, 0.0, 1.0)
            * upright
        )
        root_height_velocity_down = torch.relu(env._prev_root_height[agent] - root_pos[:, 2]) / max(env.step_dt, 1.0e-6)
        contact_intent = env._contact_intent[agent] * facing_gate * upright * env.proxy_reward_scale() * combat_gate
        attack_momentum = env._attack_momentum[agent] * facing_gate * upright * combat_gate

        radial = torch.linalg.norm(root_pos[:, :2], dim=-1)
        arena_control = torch.clamp(1.0 - torch.square(radial / env.cfg.arena.radius), 0.0, 1.0)
        stay_inside = torch.clamp((env.cfg.arena.radius - radial) / env.cfg.arena.radius, -1.0, 1.0)
        radial_dir = root_pos[:, :2] / torch.clamp(radial.unsqueeze(-1), min=1.0e-6)
        outward_speed = torch.relu(torch.sum(root_lin_vel_w[:, :2] * radial_dir, dim=-1))
        wall_boundary_escape = torch.relu(radial / env.cfg.arena.radius - 0.82) * (1.0 + outward_speed)

        useful_contact = env._useful_contact[agent] * upright * combat_gate
        stable_contact_attack = useful_contact * env._stance_quality(agent)
        limb_contact_reward = (
            torch.clamp(env._strike_speed[agent] / env.cfg.contact.strike_speed_normalizer, 0.0, 2.0)
            * useful_contact
            * foot_support_quality
        )
        selected_push_contact = torch.clamp(
            env._selected_push_contact_force(agent) / env.cfg.contact.force_normalizer,
            0.0,
            5.0,
        )
        offhand_push_contact = torch.clamp(
            env._offhand_push_contact_force(agent) / env.cfg.contact.force_normalizer,
            0.0,
            5.0,
        )
        selected_push_speed = torch.clamp(
            env._selected_push_speed(agent, opponent, rel_dir) / env.cfg.contact.strike_speed_normalizer,
            0.0,
            2.0,
        )
        offhand_push_speed = torch.clamp(
            env._offhand_push_speed(agent, opponent, rel_dir) / env.cfg.contact.strike_speed_normalizer,
            0.0,
            2.0,
        )
        selected_push_reach = torch.clamp((env._selected_push_reach(agent, rel_dir) - 0.08) / 0.38, 0.0, 1.0)
        selected_push_action = env._selected_push_arm_action_magnitude(agent)
        offhand_push_action = env._offhand_push_arm_action_magnitude(agent)
        push_distance_gate = torch.clamp((1.35 - distance) / 0.90, 0.0, 1.0) * (distance > 0.25).float()
        push_setup_gate = torch.clamp((distance - 0.35) / 0.55, 0.0, 1.0) * torch.clamp(
            (1.40 - distance) / 0.55,
            0.0,
            1.0,
        )
        push_activity = torch.clamp(
            0.45 * selected_push_reach + 0.35 * selected_push_speed + 0.20 * selected_push_contact,
            0.0,
            1.50,
        )
        one_hand_push_setup = (
            selected_push_reach
            * selected_push_speed
            * facing_gate
            * foot_support_quality
            * action_gate
            * push_setup_gate
        )
        one_hand_push_contact = (
            selected_push_contact
            * selected_push_speed
            * facing_gate
            * foot_support_quality
            * combat_gate
            * push_distance_gate
        )
        one_hand_push_balance = (
            push_activity
            * foot_support_quality
            * env._capture_point_support_quality(agent)
            * upright
            * (0.25 + 0.75 * action_gate)
        )
        one_hand_push_destabilize = one_hand_push_contact * (
            1.0 + torch.clamp(env._opponent_destabilization[agent], 0.0, 2.0)
        )
        offhand_push_penalty = (
            offhand_push_contact * push_distance_gate
            + 0.25 * offhand_push_speed * push_setup_gate
            + 0.20 * torch.relu(offhand_push_action - 0.75 * selected_push_action)
        ) * combat_gate
        torso_contact = torch.clamp(env._torso_contact_force(agent) / env.cfg.contact.force_normalizer, 0.0, 5.0)
        torso_charge_reward = (
            torso_contact
            * torch.clamp(toward_speed / env.cfg.contact.strike_speed_normalizer, 0.0, 2.0)
            * foot_support_quality
            * combat_gate
        )
        destabilizing_impact = env._destabilizing_impact[agent] * upright * combat_gate
        topple_pressure = env._topple_pressure[agent] * upright * combat_gate
        drive_pressure = env._drive_pressure[agent] * upright * combat_gate
        support_break_pressure = env._support_break_pressure[agent] * upright * combat_gate
        recent_attack = torch.clamp(env._recent_attack_pressure[agent], 0.0, 5.0) * combat_gate
        attack_credit = torch.maximum(torch.clamp(env._proof_impact[agent], 0.0, 5.0), recent_attack)
        attack_gate = torch.clamp(attack_credit, 0.0, 1.0)
        opp_destabilization = env._opponent_destabilization[agent] * attack_gate
        proof = (attack_credit >= env.cfg.contact.fall_credit_min_attack).float()
        self_fall = env._fallen[agent].float()
        clean_attack = proof * upright * (1.0 - self_fall)
        opponent_fall = clean_attack * (env._new_fall[opponent].float() + 0.08 * env._fallen[opponent].float())
        opponent_knockdown = clean_attack * (
            env._new_knockdown[opponent].float() + 0.15 * env._knockdown[opponent].float()
        )
        mutual_fall = self_fall * env._fallen[opponent].float()
        airborne_without_attack = (support_quality < 0.08).float() * (attack_credit < 0.10).float() * (1.0 - self_fall)
        fall_early = self_fall * early_fall_window
        torso_only_motion = (
            torch.relu(root_speed - env._support_mean_speed(agent))
            * (support_quality < 0.35).float()
            * (attack_credit < 0.10).float()
        )
        bad_contact_penalty = torso_contact * (1.0 - foot_support_quality) * (attack_credit < 0.15).float()
        opponent_tilt_delta = torch.relu(env._prev_up_z[opponent] - env._up_z[opponent]) * attack_gate
        opponent_height_drop_delta = torch.relu(env._prev_root_height[opponent] - opp_pos[:, 2]) * attack_gate
        opponent_support_break = env._support_break_pressure[agent] * attack_gate
        opponent_drive_dir = torch.sum(opp_lin_vel_w * rel_dir, dim=-1)
        impulse_direction_reward = (
            (
                torch.clamp(opponent_drive_dir / env.cfg.contact.strike_speed_normalizer, 0.0, 2.0)
                + 0.35 * torch.clamp(-opp_lin_vel_w[:, 2], 0.0, 2.0)
            )
            * attack_gate
            * upright
        )
        clean_knockdown_bonus = clean_attack * env._new_knockdown[opponent].float()
        mutual_fall_hard_penalty = mutual_fall + env._new_fall[agent].float() * env._fallen[opponent].float()
        impact_balance = attack_credit * balance_recovery * (1.0 - self_fall)
        impact_self_destabilization = attack_credit * (
            (1.0 - upright) + 0.50 * torch.clamp(lateral_ang_vel / 4.0, 0.0, 2.0) + self_fall
        )

        energy = env._energy[agent]
        torque_penalty = env._torque_penalty[agent]
        joint_limit_penalty = env._joint_limit_penalty[agent]
        jitter_penalty = env._jitter_penalty[agent]
        inactivity_penalty = env._inactivity[agent]
        spin_penalty = env._spin_without_contact[agent]
        uncontrolled_collision = env._uncontrolled_collision[agent]
        posture_instability = env._posture_instability[agent]
        non_support_contact = torch.clamp(
            env._non_support_contact_force(agent) / env.cfg.contact.force_normalizer, 0.0, 5.0
        )
        self_contact_abuse = non_support_contact * (attack_credit < 0.08).float() * (distance > 0.35).float()
        joint_speed = torch.linalg.norm(env.joint_vel(agent), dim=-1) / max(env._runtime[agent].action_dim, 1)
        joint_limit_slam = joint_limit_penalty * torch.clamp(joint_speed / 8.0, 0.0, 2.0)
        action_rate = jitter_penalty
        torque_spike = torch.relu(torque_penalty - 0.25)
        spin_flail_penalty = spin_penalty * (1.0 + torch.clamp(jitter_penalty, 0.0, 2.0))
        passive_survival = inactivity_penalty * after_warmup * (attack_credit < 0.08).float()
        contact_without_progress = (
            useful_contact
            * (opponent_tilt_delta + opponent_height_drop_delta + env._new_knockdown[opponent].float() < 0.03).float()
        )
        warmup_stand_reward = warmup_gate * env._stance_quality(agent)
        progressive_phase = torch.clamp((episode_time - warmup_s) / 2.5, 0.0, 1.0)
        progressive_attack_gate = useful_contact * foot_support_quality * (0.35 + 0.65 * progressive_phase)
        fall_cause_credit = clean_attack * (
            env._new_fall[opponent].float() + 0.5 * env._new_knockdown[opponent].float()
        )

        final_win, final_loss, final_draw = self._terminal_terms(env, agent)

        terms = {
            "upright_stability": scales.upright_stability * upright,
            "balance_recovery": scales.balance_recovery * balance_recovery,
            "standing_height": scales.standing_height * standing_height,
            "support_contact": scales.support_contact * support_contact,
            "low_base_height": -scales.low_base_height * low_base_height,
            "standing_pose": scales.standing_pose * standing_pose,
            "warmup_action_restraint": -scales.warmup_action_restraint * warmup_action_restraint,
            "stand_still_joint_deviation": -scales.stand_still_joint_deviation * stand_still_joint_deviation,
            "arm_motion_restraint": -scales.arm_motion_restraint * arm_motion_restraint,
            "hip_yaw_roll_deviation": -scales.hip_yaw_roll_deviation * hip_yaw_roll_deviation,
            "center_of_mass_over_support": scales.center_of_mass_over_support * center_of_mass_over_support,
            "capture_point_support": scales.capture_point_support * capture_point_support,
            "both_feet_support_warmup": scales.both_feet_support_warmup * both_feet_support_warmup,
            "foot_support_quality": scales.foot_support_quality * foot_support_quality,
            "foot_slip": -scales.foot_slip * foot_slip,
            "base_pitch_roll": -scales.base_pitch_roll * base_pitch_roll,
            "angular_stumble": -scales.angular_stumble * angular_stumble,
            "knee_collapse": -scales.knee_collapse * knee_collapse,
            "leg_extension_posture": scales.leg_extension_posture * leg_extension_posture,
            "perturbation_recovery": scales.perturbation_recovery * perturbation_recovery,
            "perturbation_collapse": -scales.perturbation_collapse * perturbation_collapse,
            "airborne_without_attack": -scales.airborne_without_attack * airborne_without_attack,
            "fall_early": -scales.fall_early * fall_early,
            "recovery_reward": scales.recovery_reward * recovery_reward,
            "backward_motion": -scales.backward_motion * backward_motion,
            "backward_lean": -scales.backward_lean * backward_lean,
            "waist_action": -scales.waist_action * waist_action,
            "controlled_approach": scales.controlled_approach * controlled_approach,
            "velocity_command_tracking": scales.velocity_command_tracking * velocity_command_tracking,
            "yaw_heading_tracking": scales.yaw_heading_tracking * yaw_heading_tracking,
            "locomotion_drive": scales.locomotion_drive * locomotion_drive,
            "forward_step_progress": scales.forward_step_progress * forward_step_progress,
            "retreat_from_opponent": -scales.retreat_from_opponent * retreat_from_opponent,
            "approach_with_feet_gate": scales.approach_with_feet_gate * approach_with_feet_gate,
            "stance_width": scales.stance_width * stance_width,
            "foot_clearance": scales.foot_clearance * foot_clearance,
            "feet_air_time_biped": scales.feet_air_time_biped * feet_air_time_biped,
            "single_stance_balance": scales.single_stance_balance * single_stance_balance,
            "cadence_or_alternating_support": scales.cadence_or_alternating_support * cadence_or_alternating_support,
            "root_height_velocity_down": -scales.root_height_velocity_down * root_height_velocity_down,
            "torso_only_motion": -scales.torso_only_motion * torso_only_motion,
            "contact_intent": scales.contact_intent * contact_intent,
            "attack_momentum": scales.attack_momentum * attack_momentum,
            "arena_control": scales.arena_control * arena_control,
            "useful_contact": scales.useful_contact * useful_contact,
            "stable_contact_attack": scales.stable_contact_attack * stable_contact_attack,
            "limb_contact_reward": scales.limb_contact_reward * limb_contact_reward,
            "one_hand_push_setup": scales.one_hand_push_setup * one_hand_push_setup,
            "one_hand_push_contact": scales.one_hand_push_contact * one_hand_push_contact,
            "one_hand_push_balance": scales.one_hand_push_balance * one_hand_push_balance,
            "one_hand_push_destabilize": scales.one_hand_push_destabilize * one_hand_push_destabilize,
            "offhand_push_penalty": -scales.offhand_push_penalty * offhand_push_penalty,
            "torso_charge_reward": scales.torso_charge_reward * torso_charge_reward,
            "bad_contact_penalty": -scales.bad_contact_penalty * bad_contact_penalty,
            "destabilizing_impact": scales.destabilizing_impact * destabilizing_impact,
            "topple_pressure": scales.topple_pressure * topple_pressure,
            "drive_pressure": scales.drive_pressure * drive_pressure,
            "support_break_pressure": scales.support_break_pressure * support_break_pressure,
            "opponent_tilt_delta": scales.opponent_tilt_delta * opponent_tilt_delta,
            "opponent_height_drop_delta": scales.opponent_height_drop_delta * opponent_height_drop_delta,
            "opponent_support_break": scales.opponent_support_break * opponent_support_break,
            "impulse_direction_reward": scales.impulse_direction_reward * impulse_direction_reward,
            "opponent_fall": scales.opponent_fall * opponent_fall,
            "opponent_destabilization": scales.opponent_destabilization * opp_destabilization,
            "opponent_knockdown": scales.opponent_knockdown * opponent_knockdown,
            "clean_knockdown_bonus": scales.clean_knockdown_bonus * clean_knockdown_bonus,
            "impact_balance": scales.impact_balance * impact_balance,
            "impact_self_destabilization": -scales.impact_self_destabilization * impact_self_destabilization,
            "posture_instability": -scales.posture_instability * posture_instability,
            "mutual_fall": -scales.mutual_fall * mutual_fall,
            "mutual_fall_hard_penalty": -scales.mutual_fall_hard_penalty * mutual_fall_hard_penalty,
            "stay_inside": scales.stay_inside * stay_inside,
            "self_contact_abuse": -scales.self_contact_abuse * self_contact_abuse,
            "wall_boundary_escape": -scales.wall_boundary_escape * wall_boundary_escape,
            "energy_efficiency": -scales.energy * energy,
            "self_fall": -scales.self_fall * self_fall,
            "out_of_bounds": -scales.out_of_bounds * env._out_of_bounds[agent].float(),
            "excessive_torque": -scales.excessive_torque * torque_penalty,
            "joint_limit_abuse": -scales.joint_limit_abuse * joint_limit_penalty,
            "joint_limit_slam": -scales.joint_limit_slam * joint_limit_slam,
            "jitter": -scales.jitter * jitter_penalty,
            "action_rate": -scales.action_rate * action_rate,
            "torque_spike": -scales.torque_spike * torque_spike,
            "inactivity": -scales.inactivity * inactivity_penalty,
            "spin_without_contact": -scales.spin_without_contact * spin_penalty,
            "spin_flail_penalty": -scales.spin_flail_penalty * spin_flail_penalty,
            "uncontrolled_collision": -scales.uncontrolled_collision * uncontrolled_collision,
            "passive_survival": -scales.passive_survival * passive_survival,
            "contact_without_progress": -scales.contact_without_progress * contact_without_progress,
            "warmup_stand_reward": scales.warmup_stand_reward * warmup_stand_reward,
            "combat_gate": scales.combat_gate * combat_gate,
            "progressive_attack_gate": scales.progressive_attack_gate * progressive_attack_gate,
            "fall_cause_credit": scales.fall_cause_credit * fall_cause_credit,
            "final_win": scales.final_win * final_win,
            "final_loss": -scales.final_loss * final_loss,
            "final_draw": scales.final_draw * final_draw,
        }
        total = torch.zeros_like(distance)
        for value in terms.values():
            total = total + value
        return RewardBreakdown(total=total, terms=terms)

    @staticmethod
    def _terminal_terms(env, agent: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if agent == FIGHTER_A:
            own_id = 1
            opp_id = 2
        elif agent == FIGHTER_B:
            own_id = 2
            opp_id = 1
        else:
            raise KeyError(agent)
        terminal = env._match_terminal.float()
        final_win = terminal * (env._winner == own_id).float()
        final_loss = terminal * (env._winner == opp_id).float()
        final_draw = terminal * env._draw.float()
        return final_win, final_loss, final_draw
