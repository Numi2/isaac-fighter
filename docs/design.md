# Isaac Fight design

The task is a direct multi-agent Isaac Lab environment with two articulations and two agents. It follows Isaac Lab's DirectMARLEnv lifecycle: preprocess action dictionaries, apply actions at physics decimation, compute dones, compute rewards, reset vectorized environments, then compute observations.

The environment never emits scripted attack primitives. A policy controls normalized joint-position offsets for its own
robot. The combat objective is expressed through rewards and fight rules, while the policy decides gait, stance, contact
timing, pushing, recovery, retreat, and engagement. For staged competence, training can also run a residual-control
wrapper where a frozen locomotion policy supplies base actions and the fight policy supplies a smaller combat residual.

## Asymmetry

Asymmetry is handled as a first-class property. `fighter_a` and `fighter_b` each have a robot name, controlled-joint list, action scale, spawn, observation dimension, action dimension, reward buffers, fall state, contact state, and statistics. The default combat bootstrap is symmetric G1-29DoF vs G1-29DoF because getting both agents upright on a shared locomotion base is the shortest path to useful contact. H1 and H1-vs-H1 remain configurable targets by changing the fighter config and letting `__post_init__` refresh spaces.

## Contact

Robot contact is estimated from available Isaac Lab contact tensors when present and from a root-relative closing-speed proxy for early engagement shaping. Proof impact and checkpoint health stay separate from the proxy. Foot support uses near-ground upward vertical load on support bodies, while robot-robot contact and torso impacts are logged separately so wall/body collisions do not masquerade as stance.

## Self-play

The active training run writes skrl checkpoints into a persistent policy pool. The pool records policy id, version, Elo,
win/loss/draw rates, checkpoint path, tags, and league role. The opponent sampler combines Elo proximity, observed
weakness, recency, and role weights. A historical-opponent wrapper can replace one side's actions with a sampled learned
TorchScript or skrl checkpoint policy. When the pool is empty, the wrapper leaves actions unchanged so the first
IPPO/MAPPO population can bootstrap.

## Staged Competence

The focused bootstrap preset uses Unitree's G1 velocity action-scale profile, compact temporal memory, and a reduced
stand-shove reward profile. Perturbations, ADR, fall-recovery reset starts, body-slam pressure, and league sampling are
deferred until upright support and shove-survival metrics are visibly improving.
