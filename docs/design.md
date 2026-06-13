# Isaac Fight design

The task is a direct multi-agent Isaac Lab environment with two articulations and two agents. It follows Isaac Lab's DirectMARLEnv lifecycle: preprocess action dictionaries, apply actions at physics decimation, compute dones, compute rewards, reset vectorized environments, then compute observations.

The environment never emits scripted attack primitives. A policy controls normalized joint-position offsets for its own
robot. The combat objective is expressed through rewards and fight rules, while the policy decides gait, stance, contact
timing, pushing, recovery, retreat, and engagement. For staged competence, training can also run a residual-control
wrapper where a frozen locomotion policy supplies base actions and the fight policy supplies a smaller combat residual.

## Asymmetry

Asymmetry is handled as a first-class property. `fighter_a` and `fighter_b` each have a robot name, controlled-joint list, action scale, spawn, observation dimension, action dimension, reward buffers, fall state, contact state, and statistics. The default is G1-29DoF against H1. The task can be configured as G1-vs-G1 or H1-vs-H1 by changing the fighter config and letting `__post_init__` refresh spaces.

## Contact

Robot contact is estimated from available Isaac Lab contact tensors when present and from a root-relative closing-speed proxy when the installed asset stack does not expose net body forces. This keeps training functional across Unitree asset revisions while still logging real contact forces when contact sensors are available.

## Self-play

The active training run writes skrl checkpoints into a persistent policy pool. The pool records policy id, version, Elo,
win/loss/draw rates, checkpoint path, tags, and league role. The opponent sampler combines Elo proximity, observed
weakness, recency, and role weights. A historical-opponent wrapper can replace one side's actions with a sampled learned
TorchScript or skrl checkpoint policy. When the pool is empty, the wrapper leaves actions unchanged so the first
IPPO/MAPPO population can bootstrap.

## Staged Competence

The fast preset now uses Unitree's G1 velocity action-scale profile, adversarial perturbations, compact temporal memory,
fall-recovery shaping, and gated ADR. PBT reward mutation is available but disabled by default; it is intended only after
standing and shove-survival metrics are visibly improving.
