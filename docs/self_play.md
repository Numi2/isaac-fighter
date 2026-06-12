# Self-play and tournament workflow

Training uses skrl IPPO by default. MAPPO is registered for centralized value experiments. The environment emits per-agent rewards and statistics, and the training script synchronizes checkpoints into `policy_pool/pool.json`.

Pool records track:

- Elo
- win rate
- loss rate
- draw rate
- policy version
- checkpoint path
- tags such as `skrl` or `torchscript`

The opponent sampler filters by Elo range and scores candidates using weakness, uncertainty, and recency. Tournament evaluation runs round-robin matches between TorchScript policies and writes a JSON result with Elo updates and match metrics.
