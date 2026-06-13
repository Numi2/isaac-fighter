# Unitree G1 Config Notes

Source mined from `unitreerobotics/unitree_rl_lab` on June 13, 2026.

High-value G1-29DoF defaults:

- Velocity task: `Unitree-G1-29dof-Velocity`.
- Default base height: `0.8`.
- Velocity deploy step: `0.02` seconds.
- Velocity deploy action scale: `0.25` for all 29 joints.
- Velocity observations use history length `5` for base angular velocity, projected gravity, velocity command, joint
  position, joint velocity, and last action.
- Reset/domain randomization uses friction range `0.3..1.0`, torso mass add range `-1.0..3.0`, and interval pushes
  with planar velocity range `-0.5..0.5`.
- G1 default pose includes hip pitch `-0.1`, knees `0.3`, ankle pitch `-0.2`, shoulder pitch `0.3`, elbow `0.97`,
  and small shoulder/wrist roll offsets.
- G1 actuator gains: hip pitch/yaw `100/2`, hip roll `100/2`, knee `150/4`, waist yaw `200/5`, ankle/shoulder/elbow
  position stiffness `40`, wrist pitch/yaw stiffness `40`.

Isaac Fight implications:

- Use the `unitree_velocity` action-scale profile when bootstrapping from G1 velocity policies.
- Keep the existing `combat_safety` profile available for unstable scratch runs.
- Prefer residual combat learning on top of a frozen locomotion/warm-start policy before training full-body combat from
  random initialization.
- Treat Unitree mimic `.npz` motion files as AMP/motion-prior bootstrap artifacts, not opponent policies.
