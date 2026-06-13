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
Tournament evaluation runs round-robin matches between compatible policies and writes a JSON result with Elo updates,
fight-health summaries, and promotion scores. Pass `--promote_to_league` to write `league_promoted` or
`league_suppressed` tags plus the evaluation summary back into `policy_pool/pool.json`. The opponent sampler upweights
promoted policies and heavily downweights suppressed policies, so evaluation results directly shape future rollouts.

League roles:

- `main`: balanced fight objective.
- `shove_exploiter`: emphasizes selected-hand shove contact and support breaking.
- `body_slam_exploiter`: emphasizes torso charge and drive pressure while preserving mutual-fall penalties.
- `balance_breaker`: emphasizes tilt/topple/support-break pressure.

Automation:

- Use `scripts/tools/run_league_cycle.py` to run a main/exploiter train sequence followed by tournament promotion.
- Run without `--execute` first to print the exact Isaac Lab commands; add `--execute` inside the Isaac Lab/Brev shell
  once the commands look right.

Residual locomotion:

- Pass `--residual_locomotion_checkpoint PATH` to compose a frozen G1 locomotion/warm-start policy with trainable fight
  residual actions.
- Use `--residual_action_scale` to keep combat residuals small while the base policy preserves standing/walking.
- Residual actions are joint-aware: legs/waist are capped conservatively during bootstrap, arms get larger residual
  authority, and legs can open later after stability.

Bootstrap artifacts:

- Use `scripts/tools/locomotion_bootstrap.py warmstart` for Unitree velocity checkpoints.
- Use `scripts/tools/locomotion_bootstrap.py motion-prior` for Unitree mimic/AMP `.npz` motion files.
- Use `scripts/tools/export_amp_rollout_features.py` to collect policy-generated AMP feature negatives.
- Use `scripts/tools/train_amp_discriminator.py` to train a discriminator from reference motion positives and rollout
  negatives, then pass it to training with `--motion_prior_discriminator`.
- Motion-prior joint tensors are mapped by joint name when names are present. Keep `--motion_prior_min_joint_name_coverage`
  high for G1 runs; use unnamed dimensional fallback only for artifacts known to use the exact controlled-joint order.
