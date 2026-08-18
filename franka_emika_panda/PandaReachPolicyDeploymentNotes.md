# Panda Reach Policy Deployment Notes

## Current Policy And Task

This handoff is for the current Panda constrained reach / verify setup: `APandaConstrainedReachEnvironment` inherits the Panda reach/hold task and adds pose-band and velocity-band constraints. The policy is not doing trajectory tracking. It should move the TCP to a sampled target and keep actively holding there.

Current target sampling is implemented as a per-episode random box offset in env-local MuJoCo meters, not a spherical radius:

- Target center: `TargetLocationMeters = (0.32, 0.0, 0.50)` m.
- Random offset: `TargetRandomOffsetMaxMeters = (0.06, 0.06, 0.06)` m.
- Sampling: `x,y ~ U[-0.06, +0.06]`; `z ~ U[0.0, +0.06]`.
- Effective local target region: `x=[0.26,0.38]`, `y=[-0.06,0.06]`, `z=[0.50,0.56]` m, before the environment actor transform.

Episode behavior:

- Reset uses the Panda `home` keyframe and samples a new target.
- Success means TCP distance `< 0.05 m` and joint velocity L2 `<= 0.4 rad/s`.
- In the Panda hold task, success does not terminate the episode. The policy should reach and hold until `EpisodeStepLimit` truncation, unless it goes out of bounds.
- `SettledDwellRequiredSteps = 20`, so the per-step success bonus is gated until the arm has stayed settled for 20 consecutive steps.

## Observation Space

Policy-facing observation dimension is 38. The logical layout is:

1. `tcp_pos` vector, 3 dims: target-relative TCP position, `(tcp_m - target_m) / ObsTcpScale`.
2. `joint1` through `joint7`, 4 dims each: `qpos / pi`, `qvel / 2.0`, previous `qpos / pi`, previous `qvel / 2.0`.
3. `prev_action`, 7 dims: previous applied/governed action in actuator order, divided by `pi`.

All normalized observation values are clipped to `[-3, +3]`. Joint acceleration is intentionally not included.

Unit notes:

- URLab `tcp_pos` framepos sensor readings arrive from `UMjSensor::GetReading()` in Unreal centimeters with Y flipped. The environment converts that sensor back to MuJoCo meters before computing target-relative observations, rewards, and termination.
- Joint positions are radians and joint velocities are rad/s before normalization.
- `prev_action` is the previous applied position-actuator target in radians before normalization, not the raw policy output.

When reproducing the input in ROS, use the same feature order and normalization as above. Be careful not to feed absolute TCP pose where the policy was trained on TCP-minus-target.

## Action Space

The action space is 7-dimensional, one scalar per Panda arm position actuator:

`actuator1, actuator2, actuator3, actuator4, actuator5, actuator6, actuator7`

These map to `joint1` through `joint7`; the gripper actuator is excluded. Actions are MuJoCo position actuator targets in radians, with bounds from the MJCF actuator `ctrlrange`.

Current constrained setup trains and evaluates through the hard action channel:

- First-order smoothing is enabled: `filtered = lerp(previous_filtered, raw_action, 0.20)`.
- Hard target-rate limiting is enabled at `ActionRateLimitHz = 60`.
- `VelocityLimitFraction = 0.38` in `curriculum_reach_constrained.json`.
- Hard action velocity limits are `0.9 * VelocityLimitFraction * rated_joint_velocity`, currently about `[0.744, 0.744, 0.744, 0.744, 0.893, 0.893, 0.893] rad/s`.
- Soft velocity caps for reward/verification are `VelocityLimitFraction * rated_joint_velocity`, about `[0.8265]*4` and `[0.9918]*3 rad/s`.
- The measured velocity governor is enabled with `MeasuredVelocityGovernorStartRatio = 0.75`; if measured joint velocity is already beyond 75% of the hard limit and the target pulls further in that direction, the command is clamped back to current joint position.

For deployment diagnostics, treat raw policy action, filtered/rate-limited/governed action, and final ROS controller command as distinct signals.

## Current Baseline Artifacts

Safer baseline checkpoint to start from:

`training/ckpts_curriculum/verify/stage_00_combo/ppo_final_lower_vib.zip`

The verify curriculum state records the promoted stage as `stage_00_combo` with success `1.0`, `max_over_band = 0.0`, and `max_over_cap = 0.0093`. The smoothness continuation also reached success `1.0`, but did not clearly improve deployment-relevant smoothness and should not be promoted unless explicitly chosen after ROS tests.

Useful current metrics:

- Success rate: `1.0`.
- Settled success rate: `1.0`.
- Verify `max_over_cap`: about `0.0093`.
- Smoothness continuation final eval had `max_over_cap = 0.0114`, `qvel_rms = 0.1644 rad/s`, and higher raw action delta metrics than the earlier smoothness eval.
- Residual jitter analysis to keep in mind: approximately 10 Hz on joints/actions 1-6, approximately 15 Hz on joint/action 7, and a qvel L2 harmonic near 20 Hz.

## Lower-Vib ONNX Export And Deployment

The lower-vibration checkpoint copy used by this ROS package is:

`franka_emika_panda/ppo_final_lower_vib.zip`

Export it deterministically with the MuJoCo-SB3 venv:

```bash
cd /home/amd/Robotics/franka_ros2/franka_emika_panda
/home/amd/Robotics/MuJoCo-SB3/.venv/bin/python export_ppo_final_lower_vib_onnx.py
```

The export writes:

`franka_emika_panda/ppo_final_lower_vib.onnx`

The ONNX interface is:

- Inputs: `joint1`..`joint7` as 4-vectors `[qpos/pi, qvel/2.0, prev_qpos/pi, prev_qvel/2.0]`, `prev_action` as the previous applied command divided by `pi`, and `tcp_pos` as `(tcp_m - target_m) / 0.75`.
- Output: `action`, a 7-value absolute joint position target vector in radians.

Run the lower-vib policy in Gazebo with the dedicated launch wrapper:

```bash
ros2 launch franka_gazebo_bringup gazebo_panda_lower_vib_onnx_policy.launch.py
```

The launch wrapper reuses `panda_reach_reduce_shake_ros2` but switches its observation packing and damping defaults for `ppo_final_lower_vib.onnx`:

- `joint_observation_mode=normalized_position_velocity_previous_position_velocity`
- `prev_action_scale=3.141592653589793`
- `smooth_actions=true`
- `action_smoothing_alpha=0.20`
- `limit_action_delta=true`
- `action_rate_limit_hz=60.0`
- `action_velocity_limits_rad_s=0.744,0.744,0.744,0.744,0.893,0.893,0.893`
- `use_measured_velocity_governor=true`
- `measured_velocity_governor_start_ratio=0.75`

Tune in this order in Gazebo/robot logs: first `action_smoothing_alpha`, then `action_velocity_limits_rad_s`, then `measured_velocity_governor_start_ratio`. Lower `action_smoothing_alpha` or velocity limits should reduce command vibration but can add lag and slow settling.

Run a logged lower-vib tuning session with:

```bash
mkdir -p /tmp/panda_tuning_logs
ros2 launch franka_gazebo_bringup gazebo_panda_lower_vib_onnx_policy.launch.py \
  telemetry_enabled:=true \
  telemetry_csv_path:=/tmp/panda_tuning_logs/lower_vib_alpha020.csv \
  telemetry_decimation:=1 \
  random_seed:=1
```

For lower-volume logs, set `telemetry_decimation:=2` or `telemetry_decimation:=4`. If `telemetry_enabled:=true` is set without `telemetry_csv_path`, the node writes a timestamped CSV under `~/.ros`.

The same logger can be enabled without launch by passing CLI defaults to the node:

```bash
python3 franka_emika_panda/panda_reach_reduce_shake_ros2.py --ros --policy \
  --model franka_emika_panda/ppo_final_lower_vib.onnx \
  --telemetry-csv-path /tmp/panda_tuning_logs/lower_vib_direct.csv
```

## Deployment Metrics To Log

The ROS runner writes one CSV row per logged control tick. Core columns are:

- Timing/state: `wall_time_s`, `ros_time_s`, `control_dt_s`, `telemetry_tick`, `episode_step_index`, `reward`, `success`, `terminated`, `truncated`.
- TCP/target: `target_x_m..target_z_m`, `tcp_x_m..tcp_z_m`, `tcp_error_x_m..tcp_error_z_m`, `tcp_distance_m`.
- Tuning knobs: `action_smoothing_alpha`, `action_rate_limit_hz`, `measured_velocity_governor_start_ratio`, and `action_velocity_limit_jointN_rad_s`.
- Per-joint measured state: `qpos_jointN_rad`, `qvel_jointN_rad_s`.
- Per-joint command path: `raw_action_jointN_rad` is the raw ONNX `action` output, `scaled_action_jointN_rad` is after `action_scale`, `target_action_jointN_rad` is after `action_output_mode` interpretation and joint-limit clipping, `smoothed_action_jointN_rad` is the low-pass-only preview, `governed_action_jointN_rad` is after the deployed rate limit and measured-velocity governor, and `final_command_jointN_rad` is the JointTrajectory position command after final clipping.

The runner has enough state to compute TCP distance whenever `tcp_source=tf` can resolve `frame_id <- tcp_frame_id`, or when `tcp_source=pose_topic` receives `tcp_pose_topic`. If TCP is unavailable, the node waits and does not publish policy commands or telemetry rows.

For post-run inspection without extra tooling:

```bash
python3 - <<'PY'
import csv, math, sys
path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/panda_tuning_logs/lower_vib_alpha020.csv"
rows = list(csv.DictReader(open(path, newline="")))
print("rows", len(rows))
print("final_tcp_distance_m", rows[-1]["tcp_distance_m"] if rows else "n/a")
for j in range(1, 8):
    vals = [abs(float(r[f"qvel_joint{j}_rad_s"])) for r in rows if r[f"qvel_joint{j}_rad_s"]]
    print(f"joint{j} max_abs_qvel_rad_s", max(vals) if vals else "n/a")
PY
```

Log these raw streams with timestamps and control `dt`:

- Raw policy action before any deployment-side processing.
- Post-normalization/scaled action, if the ROS wrapper rescales from network output.
- Smoothed, rate-limited, and governed applied command.
- Final ROS command sent to the controller.
- Controller-reported joint position, joint velocity, controller state, tracking error, saturation/clamping flags, and any rejected-command status.
- Target pose, TCP pose, TCP-minus-target vector, and TCP distance.

Derived metrics to compute per joint and near target:

- `qpos` peak-to-peak, RMS, p95, and max deviation.
- `qvel` RMS, p95, max, and over-cap fraction.
- Velocity delta, acceleration-like `dv/dt`, and jerk-like `d2v/dt2`.
- Raw action delta and second difference.
- Applied-command delta and second difference.
- Dominant frequency per joint and per action/command stream.
- Settled success rate: distance inside threshold while joint velocity L2 stays under the deployed cap.

The governed/applied command stream is the most important command-side signal. Raw policy action alone was not enough to explain the observed vibration because training used filtering, rate limiting, and a measured-velocity governor before MuJoCo saw the command.

## Lower-Vib Tuning Procedure

Keep the target seed and target region fixed while comparing runs. Start from the launch defaults above, then tune in this order:

1. Tune `action_smoothing_alpha` first. Try small sweeps around the default, for example `0.12`, `0.16`, `0.20`, and `0.25`. Prefer the lowest vibration setting that still reaches promptly and does not overshoot or lag badly near the goal.
2. Tune `action_velocity_limits_rad_s` second. Scale the whole vector conservatively before doing joint-specific edits, and compare `governed_action_jointN_rad`, `final_command_jointN_rad`, `qvel_jointN_rad_s`, and `tcp_distance_m`.
3. Tune `measured_velocity_governor_start_ratio` third. Lower ratios intervene earlier and can reduce measured velocity peaks; higher ratios preserve responsiveness but may allow vibration to reappear.

Only promote a setting if it improves measured `qvel` and command vibration while preserving reach success and steady TCP distance. Compare near-goal windows separately from the approach, because a setting that smooths hold behavior can still make approach too sluggish.

## Lower-Vib Notch Filter Sweep

The ROS deployment runner now has an optional cascaded per-joint IIR notch filter. It is disabled by default and can run at one of three stages:

- `action_notch_filter_stage:=target_action`: after ONNX action interpretation and before smoothing/rate limiting.
- `action_notch_filter_stage:=final_command`: after smoothing/rate limiting/governor and before final joint-limit clipping/publication.
- `action_notch_filter_stage:=both`: applies the same notch configuration at both stages.

The current Gazebo sweep held the best previous damping fixed:

- `action_smoothing_alpha:=0.16`
- `action_velocity_limits_rad_s:=0.5952,0.5952,0.5952,0.5952,0.7144,0.7144,0.7144`
- `measured_velocity_governor_start_ratio:=0.75`
- `random_seed:=1`

The target-action notch was not effective enough: 10 Hz on joints 1-6 with `Q=20`, `Q=12`, `Q=8`, and `Q=4` changed target-action delta RMS by only about 1% and produced only small near-goal `qvel` improvements. The useful placement was `final_command`.

Current best command-side candidates:

```bash
# Narrow/minimum-intervention candidate.
ros2 launch franka_gazebo_bringup gazebo_panda_lower_vib_onnx_policy.launch.py \
  rviz:=false \
  telemetry_enabled:=true \
  telemetry_csv_path:=/tmp/panda_tuning_logs/lower_vib_finalnotch10_q20_confirm.csv \
  telemetry_decimation:=1 \
  random_seed:=1 \
  action_smoothing_alpha:=0.16 \
  action_velocity_limits_rad_s:=0.5952,0.5952,0.5952,0.5952,0.7144,0.7144,0.7144 \
  action_notch_filter_enabled:=true \
  action_notch_filter_stage:=final_command \
  action_notch_filter_frequencies_hz:=10,10,10,10,10,10,0 \
  action_notch_filter_q:=20.0

# Stronger candidate if visible vibration remains.
ros2 launch franka_gazebo_bringup gazebo_panda_lower_vib_onnx_policy.launch.py \
  rviz:=false \
  telemetry_enabled:=true \
  telemetry_csv_path:=/tmp/panda_tuning_logs/lower_vib_finalnotch10_q8_confirm.csv \
  telemetry_decimation:=1 \
  random_seed:=1 \
  action_smoothing_alpha:=0.16 \
  action_velocity_limits_rad_s:=0.5952,0.5952,0.5952,0.5952,0.7144,0.7144,0.7144 \
  action_notch_filter_enabled:=true \
  action_notch_filter_stage:=final_command \
  action_notch_filter_frequencies_hz:=10,10,10,10,10,10,0 \
  action_notch_filter_q:=8.0
```

Measured near-goal results from the bounded fixed-seed Gazebo sweep, compared with the no-notch `alpha=0.16` / 80% velocity-limit confirmation log:

- No notch: `qvel_l2 rms/p95/max = 1.0842 / 1.3170 / 1.5400`.
- `final_command`, 10 Hz, `Q=20`: `0.9752 / 1.2401 / 1.3432`, about `10.1%` RMS improvement and `12.8%` max improvement.
- `final_command`, 10 Hz, `Q=8`: `1.0031 / 1.1910 / 1.3002`, about `9.6%` p95 improvement and `15.6%` max improvement.
- `final_command`, 25 Hz, `Q=8`: worse than the 10 Hz final-command notch, so do not target the observed 25 Hz final-command component first.

Start with `Q=20` if the goal is the least invasive filter. Move to `Q=8` if the vibration is still visible; it gave the best p95/max reduction in the sweep. Keep joint 7 unfiltered for now because its dominant motion changed across runs and the current visible high-frequency issue is primarily on joints 1-6.

If vibration is still visually obvious, use the aggressive final-command dual-notch preset:

```bash
ros2 launch franka_gazebo_bringup gazebo_panda_lower_vib_onnx_policy.launch.py \
  rviz:=false \
  telemetry_enabled:=true \
  telemetry_csv_path:=/tmp/panda_tuning_logs/lower_vib_aggressive_dual_notch.csv \
  telemetry_decimation:=1 \
  random_seed:=1 \
  action_smoothing_alpha:=0.16 \
  action_velocity_limits_rad_s:=0.5952,0.5952,0.5952,0.5952,0.7144,0.7144,0.7144 \
  action_notch_filter_enabled:=true \
  action_notch_filter_stage:=final_command \
  action_notch_filter_frequencies_hz:=10,10,10,10,10,10,10 \
  action_notch_filter_q:=2.0 \
  action_notch_filter_secondary_frequencies_hz:=0,0,0,20,0,20,20 \
  action_notch_filter_secondary_q:=4.0
```

In the aggressive fixed-seed sweep:

- `final_command`, 10 Hz all joints, `Q=2`: `qvel_l2 rms/p95/max = 0.9591 / 1.2199 / 1.4221`.
- `final_command`, 10 Hz all joints `Q=2` plus 20 Hz on joints 4, 6, 7 with `Q=4`: `0.9559 / 1.2038 / 1.4847`.

The dual-notch preset gave the best RMS and p95 reduction but not the lowest max spike. If the robot still shows visible high-frequency shake with the dual-notch preset, the next knob should probably be the controller/trajectory interface rather than adding more notches: lower `trajectory_duration_s` slightly or inspect the `joint_trajectory_controller` interpolation/gains, because the action filter is already strongly shaping the command stream.

## Near-Goal Command Hold (Deadband)

Deployment-side notch/smoothing filtering plateaued: near the goal the raw policy
action is effectively bang-bang, the action rate/velocity limiter clips 100% of
near-goal joint-ticks, so the final command pins to the slew cap and reverses
direction every tick. That reversal is the visible ~10 Hz limit cycle, worst on
joint 7 (wrist).

The ROS runner (`panda_reach_reduce_shake_ros2`) now has an optional near-goal
command hold with hysteresis. It is off by default and fully backward compatible.

Behavior:

- Tracks measured TCP distance and measured joint-velocity L2 each control tick.
- ENTER HOLD when `tcp_distance < hold_enter_tcp_m` AND measured `qvel L2 <
  hold_enter_qvel_l2` for `hold_enter_ticks` consecutive ticks. On enter it latches
  the current final command vector as a fixed setpoint.
- While in HOLD it publishes the latched setpoint every tick (it does not keep
  updating from the policy), so the controller sees a constant target and settles
  instead of limit-cycling. The rate-limiter/governor state and `prev_action` are
  pinned to the latched value so there is no command jump on exit.
- EXIT HOLD when `tcp_distance > hold_exit_tcp_m` (hysteresis, `hold_exit_tcp_m >
  hold_enter_tcp_m`); the normal command path then resumes from the latched value.
- Hold state resets on episode reset.
- A telemetry column `hold_active` (0/1) is written to the CSV (plus `hold_enabled`).

Parameters (ROS params and launch arguments on both
`gazebo_panda_reduce_shake_onnx_policy.launch.py` and the lower-vib wrapper):

- `hold_enabled` (default `false`)
- `hold_enter_tcp_m` (default `0.02`)
- `hold_exit_tcp_m` (default `0.04`)
- `hold_enter_qvel_l2` (default `0.5`)
- `hold_enter_ticks` (default `10`)

Tuning note: with the sustained hold-point limit cycle, measured near-goal `qvel L2`
stays around 1.0-1.6 rad/s and never drops below the default `0.5` gate, so the hold
never engages at the default. The `tcp_distance < hold_enter_tcp_m` gate plus the
`hold_enter_ticks` consecutive-tick requirement already exclude the fast approach,
so raise `hold_enter_qvel_l2` above the limit-cycle band (about `1.5`) so the hold
can latch once the TCP has arrived. Lower it again only if approach transients start
latching too early.

Enabled-hold Gazebo command actually run (fixed seed 1, matching the
`lower_vib_alpha016_vlim080_confirm.csv` baseline params, hold tuned to engage):

```bash
ros2 launch franka_gazebo_bringup gazebo_panda_lower_vib_onnx_policy.launch.py \
  rviz:=false \
  telemetry_enabled:=true \
  telemetry_csv_path:=/tmp/panda_tuning_logs/lower_vib_nearhold_qv15_seed1_TIMESTAMP.csv \
  telemetry_decimation:=1 \
  telemetry_flush_every_n_rows:=1 \
  random_seed:=1 \
  action_smoothing_alpha:=0.16 \
  action_velocity_limits_rad_s:=0.5952,0.5952,0.5952,0.5952,0.7144,0.7144,0.7144 \
  hold_enabled:=true \
  hold_enter_qvel_l2:=1.5
```

Measured near-goal results (`tcp_distance < 0.05`), hold vs the no-hold
`alpha=0.16` / 80%-velocity-limit confirmation baseline, seed 1. The hold engaged
~1.1 s after telemetry start and stayed active for 99.4% of near-goal ticks. The
"hold-active window" column is the steady-hold comparison (baseline has no equivalent
window because it never holds):

| Metric | Baseline (hold off) | Hold on, hold-active window | Change |
| --- | --- | --- | --- |
| `qvel_l2` rms (rad/s) | 1.1085 | 0.0333 | -97.0% |
| `qvel_l2` p95 (rad/s) | 1.3538 | ~0.0 | -100% |
| `qvel_l2` max (rad/s) | 1.8425 | 1.4165 | -23.1% (residual = engage/exit transient) |
| qpos p2p joint2 (rad) | 0.4002 | 0.0048 | -98.8% |
| qpos p2p joint4 (rad) | 0.4425 | 0.0047 | -98.9% |
| qpos p2p joint6 (rad) | 0.4807 | 0.0157 | -96.7% |
| qpos p2p joint7 (rad) | 0.5978 | 0.0148 | -97.5% |
| qpos p2p joints 1/3/5 (rad) | <0.014 | <0.008 | already tiny |
| final TCP distance (m) | 0.00519 | 0.01544 | still success (< 0.05) |
| near-goal TCP mean/max (m) | 0.00765 / 0.0441 | 0.0155 / 0.0447 | steady, within band |
| near-goal `hold_active` fraction | n/a | 0.994 | hold engages and stays |

The hold kills the hold-point limit cycle: joint 7 (the worst visible offender)
near-goal peak-to-peak drops from ~0.60 rad to ~0.015 rad, and near-goal joint
velocity L2 drops from ~1.1 rad/s rms to ~0.03 rad/s rms while holding. Goal-reaching
is preserved (final TCP distance stays well inside the 0.05 m success threshold). The
only trade-off is a slightly larger steady-state TCP offset (~0.015 m vs ~0.008 m),
because the setpoint is frozen at the arrival pose instead of continually re-centering;
this is still comfortably within success and removes the visible shake. The residual
`qvel_l2` max reflects the brief transient at engagement (and any exit/re-enter), not
a sustained oscillation.

## ROS-Side Fixes And Experiments

Start with applied-command smoothing and rate limiting in ROS so the real controller sees the same kind of command stream used in training. Match the effective 60 Hz limiter or explicitly log the deployed control period and derive per-step deltas from it.

Other checks and experiments:

- Confirm whether the 10 Hz / 15 Hz frequencies appear in the applied command, joint state, or both before adding filters.
- If the frequency is stable, try true notch/band-stop filters around 10 Hz for joints/actions 1-6 and around 15 Hz for joint/action 7. Use a conservative starting `Q` around 5-10 and verify phase/lag effects near the goal.
- Avoid a simple small-action deadband as the first fix unless applied commands are proven to be near zero. Deadbands can create stick-slip around the hold point.
- Check controller gains, command interpolation, command queueing, and whether the controller treats position targets as step commands.
- Measure timing jitter and missed/merged control ticks; jitter can turn a smooth target sequence into velocity spikes.
- Inspect hold behavior near the goal separately from approach behavior. A filter that helps hold may make the approach too sluggish.
- Tune low-pass filters cautiously; lowering jitter by adding phase lag can hurt settling or cause overshoot.
- Look at joint-specific handling for J5-J7, especially J6/J7, since their dynamics and observed vibration may differ from J1-J4.

If deployment-side filtering and controller tuning are not enough, the next training direction is to train directly against the deployed applied-command path with stronger applied-command smoothness terms or carefully weighted action second-difference shaping. The current smoothness continuation did not clearly improve the safer baseline, so do not assume more continuation with the same recipe will fix the ROS-side issue.

## Recommended ROS Test Protocol

1. Run one fixed target and one sampled-target batch with policy output logged but no extra ROS filtering beyond safety clamps.
2. Add deployed smoothing/rate limiting and repeat the exact same target seeds or target list.
3. Compare raw action, applied command, ROS command, joint state, and TCP distance on the same plots.
4. Compute near-goal windows separately from full-episode windows.
5. Only add notch/band-stop filtering after confirming a stable frequency in applied command or joint data.
6. Promote a deployment setting only if it preserves reach success, improves settled success, and reduces applied-command and joint-state vibration metrics.
