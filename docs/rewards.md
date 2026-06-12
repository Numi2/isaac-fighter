# Reward stack

Positive terms:

- `upright_stability`: normalized body up-axis reward.
- `balance_recovery`: upright reward with lateral angular-velocity damping.
- `controlled_approach`: distance reduction gated by facing direction and upright state.
- `arena_control`: reward for holding a stable position away from the boundary.
- `useful_contact`: contact force or contact proxy gated by proximity and upright state.
- `opponent_destabilization`: opponent height drop, tilt increase, and new knockdown.
- `opponent_knockdown`: sparse event reward and small sustained pressure reward.
- `stay_inside`: signed boundary-margin reward.
- `final_win`: terminal win reward.

Penalty terms:

- `self_fall`
- `out_of_bounds`
- `excessive_torque`
- `joint_limit_abuse`
- `jitter`
- `inactivity`
- `spin_without_contact`
- `uncontrolled_collision`
- `final_loss`
- `final_draw`

The weights are in `RewardScalesCfg` and can be replaced without changing environment mechanics.
