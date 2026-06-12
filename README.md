# Isaac Fight

Standalone Isaac Lab extension for emergent 1v1 humanoid combat.

`GhostFighter-Unitree-1v1-Direct-v0` is a true `DirectMARLEnv`: two independent Unitree humanoid articulations, asymmetric action/observation spaces, per-agent rewards, per-agent terminations, round logic, replay traces, tournament evaluation, and a persistent self-play policy pool.

The default fight is Unitree G1-29DoF vs H1. Robot physics, meshes, actuators, joint names, and initial poses are imported from upstream Unitree Isaac Lab assets at runtime; this repo does not fake or vendor robot models.

## Core

- Isaac Lab external-extension layout, isolated from Isaac Lab core.
- Multi-agent IPPO by default, MAPPO entry point included for centralized critic experiments.
- Bounded arena, randomized facing spawns, round timer, falls, knockdowns, knockouts, out-of-bounds losses, timer decisions, winner/loser/draw assignment.
- Reward stack: upright control, balance recovery, approach pressure, arena control, useful contact, opponent destabilization, opponent knockdown, terminal win/loss/draw.
- Penalty stack: self-fall, boundary loss, torque/action effort, joint-limit pressure, jitter, inactivity, spin-without-contact, uncontrolled collision.
- Population tooling: checkpoint pool, Elo metadata, weakness/recency sampler, tournament script, replay JSONL.

## Roadmap

Achieved:

- External Isaac Lab extension registered as `GhostFighter-Unitree-1v1-Direct-v0`.
- True multi-agent `DirectMARLEnv` with independent G1/H1 action spaces, observations, rewards, dones, logs, and reset state.
- Runtime Unitree asset adapter using upstream `unitree_rl_lab` robot configs and USD assets.
- Fight rules: arena bounds, randomized facing spawns, fall/knockdown/knockout logic, timer decision, winner/loser/draw assignment.
- Combat shaping: approach pressure, useful contact, destabilization, knockdown reward, stability, efficiency, boundary discipline, terminal outcome terms.
- skrl IPPO/MAPPO configs aligned with Isaac Lab 2.3 runner schema.
- Long-running Isaac Launchable training started from live checkpoints with a persistent policy pool and sidecar checkpoint sync.

Next:

- Scale population training across larger vectorized runs and periodic resume-from-best cycles.
- Export deployable policy snapshots for frozen historical-opponent rollouts.
- Run tournament ladders to update Elo, weakness scores, and matchup selection.
- Record representative replay JSONL from late-stage checkpoints for behavior inspection.

## Install

Use an Isaac Lab shell with Unitree assets installed and `UNITREE_MODEL_DIR` configured in `unitree_rl_lab`.

```bash
cd /path/to/IsaacLab
./isaaclab.sh -p -m pip install -e /path/to/isaac-fight/source/isaac_fight
./isaaclab.sh -p /path/to/isaac-fight/scripts/tools/list_fight_tasks.py
```

## Train

```bash
cd /path/to/IsaacLab
./isaaclab.sh -p /path/to/isaac-fight/scripts/skrl/train.py \
  --task GhostFighter-Unitree-1v1-Direct-v0 \
  --algorithm IPPO \
  --num_envs 2048 \
  --self_play \
  --snapshot_interval 25 \
  --pool_dir /path/to/isaac-fight/policy_pool \
  --headless
```

Keep the pool synchronized while long training runs continue:

```bash
./isaaclab.sh -p /path/to/isaac-fight/scripts/tools/sync_policy_pool.py \
  --log_root /path/to/IsaacLab/logs/skrl/ghostfighter_unitree_1v1 \
  --pool_dir /path/to/isaac-fight/policy_pool \
  --interval_s 60
```

Resume harder:

```bash
./isaaclab.sh -p /path/to/isaac-fight/scripts/skrl/train.py \
  --task GhostFighter-Unitree-1v1-Direct-v0 \
  --algorithm IPPO \
  --num_envs 4096 \
  --checkpoint /path/to/agent_*.pt \
  --self_play \
  --pool_dir /path/to/isaac-fight/policy_pool \
  --headless
```

## Evaluate

```bash
./isaaclab.sh -p /path/to/isaac-fight/scripts/skrl/evaluate_tournament.py \
  --task GhostFighter-Unitree-1v1-Direct-v0 \
  --pool_dir /path/to/isaac-fight/policy_pool \
  --rounds 32 \
  --output /path/to/isaac-fight/logs/tournaments/latest.json
```

```bash
./isaaclab.sh -p /path/to/isaac-fight/scripts/tools/record_replay.py \
  --task GhostFighter-Unitree-1v1-Direct-v0 \
  --checkpoint /path/to/checkpoint.pt \
  --output /path/to/replays/match.jsonl
```

## Boundary

Simulation research only. Learned policies can produce high-energy unstable contact and are not hardware-safe without a separate safety stack, contact constraints, and robot validation process.
