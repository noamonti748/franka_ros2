# RF3 Constrained Reach Training Spec

This documents the observation, action, and reward setup used to train the `rf3` constrained reach model:

`training/ckpts_curriculum/raw_fresh/stage_03_rf3/ppo_final.zip`

The generated stage config was:

`Scripts/curriculum_runs/raw_fresh/stage_03_rf3.yaml`

`rf3` resumed from:

`training/ckpts_curriculum/raw_fresh/stage_02_rf2/ppo_final.zip`

## Training Stage

`rf3` was stage 3 of the `raw_fresh` curriculum.

PPO settings:

- `learning_rate`: `1.0e-4`
- `n_steps`: `2048`
- `batch_size`: `64`
- `n_epochs`: `10`
- `gamma`: `0.99`
- `gae_lambda`: `0.95`
- `clip_range`: `0.2`
- `ent_coef`: `0.01`
- `target_kl`: `0.03`
- `timesteps`: `3,000,000`
- `num_simulators`: `10`
- map: `/Game/FrankaRobots/ReachConstrained`

Curriculum values written for `rf3`:

- `PoseBandFractionPerJoint`: `[0.20, 0.20, 0.20, 0.40, 0.20, 0.40, 0.20]`
- `VelocityLimitFraction`: `0.40`
- `PoseBandPenaltyScale`: `0.5`
- `JointVelocityLimitPenaltyScale`: `0.1`
- `ActionSecondDifferencePenaltyScale`: `0.015`
- `RawActionRatePenaltyScale`: `0.015`
- `RawActionSecondDifferencePenaltyScale`: `0.0075`
- `JointJerkDeltaPenaltyScale`: `0.003`
- `NearGoalJointVelocityPenaltyScale`: `0.045`
- `NearGoalSmoothingRegionMeters`: `0.12`

Note: the live `curriculum_reach_constrained.json` may reflect a later stage. The values above are the `rf3` values from the curriculum schedule/state, not necessarily the current contents of that JSON file.

## Observation

The policy observation is a Schola dict observation built by `APandaConstrainedReachEnvironment`, inherited from `APandaReachEnvironment` and `AReachEnvironment`.

All policy observations are normalized and clipped to `[-3, 3]`.

Observation keys:

- `tcp_pos`: 3 floats. The URLab sensor reports TCP position in Unreal centimeters with Y negated; the environment converts it back to MuJoCo meters, subtracts the active episode target, divides by `ObsTcpScale=0.75`, and clips.
- `joint1` through `joint7`: 4 floats each: `[qpos / pi, qvel / 2.0, previous_qpos / pi, previous_qvel / 2.0]`, clipped.
- `prev_action`: 7 floats. Previous applied action values in actuator order, divided by `pi` and clipped. These are the governed/applied commands, not the raw policy outputs.

Joint names and order:

`joint1, joint2, joint3, joint4, joint5, joint6, joint7`

The reach environments force:

- `bIncludeJointAcceleration = false`
- `bIncludePreviousJointState = true`
- `bIncludePreviousActionInObservation = true`

So joint acceleration is not part of the policy observation.

Target sampling:

- Target center: `(0.32, 0.0, 0.50)` in env-local MuJoCo meters.
- Per-episode random offset: X/Y sampled symmetrically within `+/-0.06 m`; Z sampled upward within `[0, 0.06] m`.

## Action

The policy outputs a dict action with one scalar per Panda arm actuator:

`actuator1, actuator2, actuator3, actuator4, actuator5, actuator6, actuator7`

The gripper actuator is excluded.

Each scalar is a position target bounded by the MJCF actuator `ctrlrange`.

Training used the same deployment-style action channel before applying the action to MuJoCo:

- Low-pass smoothing enabled: `ActionSmoothingAlpha = 0.20`
- Hard action delta limiting enabled at `ActionRateLimitHz = 60`
- Measured velocity governor enabled: `MeasuredVelocityGovernorStartRatio = 0.75`

For `rf3`, the hard action target velocity limits were:

- Rated max joint velocities: `[2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61] rad/s`
- Action target velocity limit: `0.9 * VelocityLimitFraction * rated_max`
- With `VelocityLimitFraction=0.40`: `[0.783, 0.783, 0.783, 0.783, 0.9396, 0.9396, 0.9396] rad/s`
- Per-60 Hz step max deltas: about `[0.01305, 0.01305, 0.01305, 0.01305, 0.01566, 0.01566, 0.01566] rad`

The reward also separately tracks raw policy actions before this smoothing/limiting/governor path for raw-action anti-jitter penalties.

## Reward

The `rf3` reward is additive across the base reach reward, Panda hold/near-goal shaping, raw-action anti-jitter terms, and constrained pose/velocity penalties.

Base reach reward:

```text
reward =
  StepPenalty
  - DistanceScale * distance_m
  + close_region_bonus
  + settled_success_bonus
  + out_of_bounds_penalty_if_applicable
  - applied_action_rate_penalty
  - applied_action_second_difference_penalty
  - raw_action_rate_penalty
  - raw_action_second_difference_penalty
  - joint_velocity_penalty
  - joint_velocity_delta_penalty
  - joint_jerk_delta_penalty
  - joint_position_delta_penalty
  + Panda hold-still bonus
  - Panda settled-hold action penalties
  - Panda near-goal smoothing penalties
  - constrained pose-band overshoot penalty
  - constrained velocity-cap overshoot penalty
```

Core reach terms:

- `StepPenalty = -0.01`
- `DistanceScale = 1.0`
- `CloseRegionMeters = 0.10`
- `CloseRegionBonusScale = 0.20`
- `SuccessThreshold = 0.05 m`
- `SuccessMaxJointVelocity = 0.4 rad/s` as L2 norm over the 7 joint velocities
- `SuccessBonus = 0.4`, but only kept after `SettledDwellRequiredSteps=20` consecutive settled steps
- `OutOfBoundsDistance = 1.5 m`
- `OutOfBoundsPenalty = -5.0`

Success does not terminate the Panda hold task. Episodes terminate only on out-of-bounds or truncate at the step limit.

Applied-action smoothness terms, using governed/applied commands from `Snapshot.ActionsApplied`:

- `ActionRatePenaltyScale = 0.045`
- `ActionSecondDifferencePenaltyScale = 0.015`

Raw-action anti-jitter terms, using policy outputs before smoothing/rate limiting/governor:

- `RawActionRatePenaltyScale = 0.015`
- `RawActionSecondDifferencePenaltyScale = 0.0075`

Joint motion penalties:

- `JointVelocityPenaltyScale = 0.03`
- `JointVelocityDeltaPenaltyScale = 0.03`
- `JointJerkDeltaPenaltyScale = 0.003`
- `JointPositionDeltaPenaltyScale = 0.01`

Panda hold-still shaping:

- Active within `HoldStillRegionMeters = 0.12 m`
- `HoldStillBonusScale = 0.04`
- Velocity ramp goes to zero at `sum(qvel_i^2) >= 0.6`

Panda settled-hold penalties, active only when `IsSuccessReached` is true:

- `SettledHoldActionRatePenaltyScale = 0.036`
- `SettledHoldActionSqPenaltyScale = 0.00225`

Panda near-goal smoothing, linearly ramped from zero at `0.12 m` to full strength at the target:

- `NearGoalActionRatePenaltyScale = 0.042`
- `NearGoalJointVelocityPenaltyScale = 0.045`
- `NearGoalJointAccelerationPenaltyScale = 0.0`

Constrained reach penalties:

- Home pose: `[0.0, -0.943, 0.0, -2.514, 0.0, 1.611, 0.0] rad`
- Joint limits min: `[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973] rad`
- Joint limits max: `[2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973] rad`
- Pose band half-width per joint: `0.5 * PoseBandFractionPerJoint[i] * (max_i - min_i)`
- Pose penalty: `-0.5 * sum(max(0, abs(qpos_i - home_i) - half_width_i)^2)`
- Velocity cap per joint: `0.40 * rated_max_velocity_i`
- Velocity overshoot penalty: `-0.1 * sum(max(0, abs(qvel_i) - velocity_cap_i)^2)`

## RF3 Evaluation Snapshot

The `rf3` eval over 50 episodes reported:

- `success_rate = 1.0`
- `settled_success_rate = 1.0`
- `max_over_band = 0.8161`
- `max_over_cap = 0.0475`
- `near_action_delta_l2_p95 = 0.7094`
- `near_action_second_diff_l2_p95 = 0.9411`

This made `rf3` the best visual-testing candidate in the raw-action penalty curriculum: it retained perfect success/settled success while substantially reducing raw-action jitter compared with `rf0` through `rf2`. `rf4` was smoother numerically but lost reach reliability.
