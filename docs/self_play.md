# Self-play and tournament workflow

Training uses skrl IPPO by default. MAPPO is registered for centralized value experiments. The environment emits per-agent rewards and statistics, and the training script synchronizes checkpoints into `policy_pool/pool.json`.

Pool records track:

- Elo
- win rate
- loss rate
- draw rate
- policy version
- checkpoint path
- tags such as `skrl`, `torchscript`, `main`, `shove_exploiter`, `body_slam_exploiter`, or `balance_breaker`

The opponent sampler filters by Elo range and scores candidates using weakness, uncertainty, recency, and league-role
weights. Main runs therefore see a mix of current snapshots and specialist exploiters instead of only the latest mirror.
Tournament evaluation runs round-robin matches between TorchScript policies and writes a JSON result with Elo updates and
match metrics.

League roles:

- `main`: balanced fight objective.
- `shove_exploiter`: emphasizes selected-hand shove contact and support breaking.
- `body_slam_exploiter`: emphasizes torso charge and drive pressure while preserving mutual-fall penalties.
- `balance_breaker`: emphasizes tilt/topple/support-break pressure.

Residual locomotion:

- Pass `--residual_locomotion_checkpoint PATH` to compose a frozen G1 locomotion/warm-start policy with trainable fight
  residual actions.
- Use `--residual_action_scale` to keep combat residuals small while the base policy preserves standing/walking.

Bootstrap artifacts:

- Use `scripts/tools/locomotion_bootstrap.py warmstart` for Unitree velocity checkpoints.
- Use `scripts/tools/locomotion_bootstrap.py motion-prior` for Unitree mimic/AMP `.npz` motion files.
