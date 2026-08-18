# Franka MuJoCo bringup

This package launches the Panda constrained-vision pick-place scene with
`mujoco_ros2_control`. It requires **MuJoCo 3.10.0 exactly**, matching the Warp
training and evaluation runtime.

## Isolated MuJoCo 3.10.0

Use the repository bootstrap from a clean shell:

```bash
cd /home/noamonti/franka_ros2
./scripts/bootstrap_humble_mujoco.sh --backend pixi --all
```

It pins `mujoco_vendor` at
`ff9e648e1af418c555c37cb6b4fcc42885057421`, builds it with
`AMENT_VENDOR_POLICY=NEVER_VENDOR`, and forwards the overlay-local vendor path
to the locked, separate MuJoCo 3.10.0 Pixi prefix. No upstream checkout is
patched. `franka_mujoco_bringup` verifies that forwarding path contains
`libmujoco.so.3.10.0`, the MuJoCo headers, and the Simulate headers.

Do not activate, install into, or otherwise modify
`/home/noamonti/warp/.venv`.

## ROS-free validation

Static XML, YAML, launch syntax, joint mapping, and camera contract checks:

```bash
cd /home/noamonti/franka_ros2/franka_mujoco_bringup
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/noamonti/franka_humble_mujoco_pixi/.pixi/envs/default/bin/pytest \
  -q -p no:cacheprovider test
```

When run with the prepared Pixi Python, the test also compiles the MJCF with
MuJoCo 3.10.0 and checks the compiled dimensions, actuator count, timestep,
camera resolution, and FOV.

## Launch

The policy node always starts after all controllers. Command publication is
disabled by default, leaving the node in monitor-only mode:

```bash
ros2 launch franka_mujoco_bringup panda_pick_place.launch.py \
  headless:=true sim_speed_factor:=1.0 policy_enabled:=false
```

The canonical executable is `panda_student_v6_ros2`; its default installed
model is
`share/franka_emika_panda/models/student_v6/student_v6_final.onnx`.
The policy camera publishes 64x64 RGB8 and depth at 60 Hz. The controller
manager updates at 240 Hz with RK4 physics and the policy receives
`control_hz=60.0`. A 240 Hz poller consumes each camera frame at most once,
with joints, gripper span, and FK TCP pose aligned to the image timestamp. The
simulator uses a forward position controller on
`/panda_arm_controller/commands` so each target is held directly between
policy ticks; the separate real-hardware launch retains JointTrajectory.

## Seeded parity evaluation

The evaluator resets one of the 16 recorded direct-JAX keyframes, briefly
gravity-compensates the arm command while reset observations propagate, and
then enables a fresh policy session:

```bash
ros2 run franka_mujoco_bringup evaluate_student_v24.py \
  --timeout 90 \
  --output /tmp/student_v24_ros_16seed.json \
  --trace-directory /tmp/student_v24_ros_traces
```

Add `--npz-trace-directory /tmp/student_v24_ros_npz` for timestamp-aligned
pixels, policy state, actions, geometry, controller targets, and measured
state. Compare one of those traces with a direct golden trace:

```bash
ros2 run franka_mujoco_bringup compare_policy_traces.py \
  --direct /path/to/direct_trace.npz \
  --ros /tmp/student_v24_ros_npz/seed_1.npz \
  --model /path/to/student_v24_urlab_candidate.onnx \
  --output /tmp/direct_ros_comparison.json
```

The validated RK4 run placed 9/16 seeds. All successful grasps completed
release and placement; all failures exhausted physical grasp retries.
Inference and controller replay are at numerical parity, but physical grasp
success remains below the promoted direct-JAX result, so this simulation score
is not hardware authorization.

## Labeled geometry corpus

When the sim-box publisher is enabled, NPZ traces include box poses matched to
the policy image timestamp, the static per-seed goal, and normalized true
geometry labels. Combine repeated pass directories with:

```bash
ros2 run franka_mujoco_bringup build_ros_geometry_corpus.py \
  --trace-root /path/to/ros_geometry_64eps \
  --validation-seed 4 --validation-seed 8 \
  --validation-seed 12 --validation-seed 16 \
  --output /path/to/ros_geometry_corpus_64eps.npz
```

The builder rejects excessive box/image skew, invalid pixels, non-finite
labels, inconsistent episode dimensions, and geometry that does not
reconstruct the recorded TCP/box/goal poses.

## OAK-1 camera emulation

CameraPlugin publishes an ideal stream under `/policy_camera/sim/...`.
`oak_camera_emulator.py` republishes the policy stream with original capture
timestamps and optional transport faults:

```bash
ros2 launch franka_mujoco_bringup panda_pick_place.launch.py \
  oak_emulator_enabled:=true \
  oak_delay_ms:=33 oak_jitter_ms:=5 \
  oak_drop_probability:=0.01
```

Diagnostics are published on
`/policy_camera/oak_emulator/telemetry`. The relay and policy both use
best-effort depth-1 image QoS; due frames are released newest-only and a
backlog is never replayed. Installed MJCF camera variants can be selected with
the `mujoco_model` launch argument.
