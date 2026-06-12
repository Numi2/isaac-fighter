# Isaac Fight

Isaac Fight is a standalone Isaac Lab extension for true 1v1 humanoid robot combat. The default task is:

`GhostFighter-Unitree-1v1-Direct-v0`

The task is built around emergent reinforcement-learning behavior, not scripted attacks. Both fighters are simulated Isaac Lab articulations. The default matchup is Unitree G1-29DoF against Unitree H1, with support for G1-vs-G1 and H1-2 as an asset target when the upstream Unitree assets are available.

The extension is intentionally isolated from Isaac Lab core. It follows the external-extension layout used by the Isaac Lab extension template: `source/<extension>`, `config/extension.toml`, Python package sources, `setup.py`, standalone scripts, and tests.

## Design goals

The environment is a direct multi-agent Isaac Lab task. It exposes one agent per robot, with independent observations, action spaces, rewards, terminations, and episode statistics. It supports asymmetric robots by keeping per-agent joint maps, action dimensions, observation dimensions, spawn state, reward terms, and rule state separate.

The environment implements a bounded arena, randomized facing spawns, round timer, fall detection, out-of-bounds detection, robot-contact proxies and contact-force logging, knockdown timers, knockout termination, timer draw termination, winner/loser assignment, replay logging, tournament evaluation, and self-play metadata with Elo.

Combat learning is reward-based. The reward stack includes upright stability, balance recovery, movement toward the opponent, arena control, useful contact, opponent destabilization, opponent knockdown, staying inside bounds, energy efficiency, and final win reward. Penalties include self-falls, leaving the arena, excessive torque/action effort, joint-limit pressure, jitter, inactivity, spin without useful contact, and uncontrolled body-collapse contact.

## Upstream dependencies

Install Isaac Lab and Isaac Sim first. Then install the Unitree repositories that provide validated Unitree Isaac Lab robot assets and configurations:

- `unitreerobotics/unitree_rl_lab` for G1-29DoF and H1 robot configs, actuators, joint names, initial poses, locomotion structure, and train/play workflow.
- `unitreerobotics/unitree_sim_isaaclab` for Unitree simulation, data collection/playback, and model-validation patterns.

This repository does not recreate Unitree robot meshes or physics assets. It imports the upstream Isaac Lab robot configurations at runtime. The fallback metadata in `isaac_fight.assets.robots.unitree` exists only for static validation, action-space sizing, and clearer error messages.

## Installation

From a shell configured for Isaac Lab:

```bash
cd /path/to/IsaacLab
./isaaclab.sh -p -m pip install -e /path/to/isaac-fight/source/isaac_fight
```

Register the task by importing the package:

```bash
./isaaclab.sh -p /path/to/isaac-fight/scripts/tools/list_fight_tasks.py
```

## Training

IPPO is the default because Isaac Lab’s skrl integration supports true multi-agent training and the task uses `DirectMARLEnv`. MAPPO is included as a second entry point for centralized-critic experiments.

```bash
cd /path/to/IsaacLab
./isaaclab.sh -p /path/to/isaac-fight/scripts/skrl/train.py \
  --task GhostFighter-Unitree-1v1-Direct-v0 \
  --algorithm IPPO \
  --num_envs 2048 \
  --self_play \
  --pool_dir /path/to/isaac-fight/policy_pool \
  --snapshot_interval 50
```

The self-play path starts from the first run. When no historical checkpoint exists, the opponent side is trained normally by IPPO. As checkpoints appear, the training supervisor records policy versions, Elo, win-rate weaknesses, and checkpoint metadata. The optional historical-opponent wrapper can freeze one side and replace its actions with a sampled learned checkpoint for active-policy training against a population.

## Evaluation tournament

```bash
./isaaclab.sh -p /path/to/isaac-fight/scripts/skrl/evaluate_tournament.py \
  --task GhostFighter-Unitree-1v1-Direct-v0 \
  --pool_dir /path/to/isaac-fight/policy_pool \
  --rounds 32 \
  --output /path/to/isaac-fight/logs/tournaments/latest.json
```

The tournament mode runs round-robin or filtered matchups across checkpoint versions, records wins/losses/draws, fight duration, knockdowns, self-falls, out-of-bounds losses, average contact force, energy use, policy version, opponent version, and Elo deltas.

## Replay recording

Set `env_cfg.replay.enabled = True` from Hydra overrides or use the helper:

```bash
./isaaclab.sh -p /path/to/isaac-fight/scripts/tools/record_replay.py \
  --task GhostFighter-Unitree-1v1-Direct-v0 \
  --checkpoint /path/to/checkpoint.pt \
  --output /path/to/replays/match.jsonl
```

Replay files are JSONL records. Each step stores root poses, actions, rewards, contact estimates, knockdown state, winner state, and timer state. They are intended for analysis and later visualization; they are not used as demonstrations for scripted fighting.

## Repository layout

```text
source/isaac_fight/
  config/extension.toml
  setup.py
  isaac_fight/
    assets/robots/unitree.py
    tasks/direct/unitree_1v1/
      unitree_1v1_env.py
      unitree_1v1_env_cfg.py
      reward_terms.py
      observations.py
      fight_rules.py
      opponent_pool.py
      elo.py
      replay.py
      stats.py
      agents/skrl_ppo_cfg.py
scripts/
  skrl/train.py
  skrl/play.py
  skrl/evaluate_tournament.py
  tools/list_fight_tasks.py
  tools/record_replay.py
tests/
```

## Configuration notes

The default arena is a 3.5 m radius bounded match space with logical out-of-bounds termination and optional USD boundary visuals. The episode length is 30 s, physics step is 0.005 s, and control decimation is 4. G1 uses a 29-dimensional joint-position target action. H1 uses the valid 19 controlled joints from the upstream H1 joint/actuator configuration.

Robot assets are selected by name in `GhostFighterUnitree1v1EnvCfg.fighters`:

- `g1_29dof`
- `h1`
- `h1_2`, if supplied by the installed Unitree packages

The environment code does not assume equal action dimensions, equal masses, equal body counts, or equal joint names.

## Safety and sim-to-real boundary

This repository is for simulation research. It should not be used to command physical robots to fight or make contact with people, animals, or property. The policies learned here can generate high-energy unstable motions and must be isolated to simulation unless a separate safety stack, contact constraints, and hardware validation process exist.
