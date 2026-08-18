# ROS 2 Humble MuJoCo overlay for Student v6

This overlay is intentionally limited to Ubuntu 22.04 (Jammy), ROS 2 Humble,
and the Panda MuJoCo simulation. It does not convert the current
`franka_ros2` branch to Humble and does not build the checkout as a monolithic
workspace.

## Pinned source inputs

The pins are recorded in `dependency-humble-mujoco.repos`:

| Repository | URL | Commit |
|---|---|---|
| `mujoco_ros2_control` | <https://github.com/ros-controls/mujoco_ros2_control.git> | `7ee969776fe097f0aedbb3ae6edcf9f85fb31244` |
| `mujoco_vendor` | <https://github.com/pal-robotics/mujoco_vendor.git> | `ff9e648e1af418c555c37cb6b4fcc42885057421` |

The `mujoco_vendor` pin was resolved from its `master` branch on 2026-08-13.
It is the source expected by the `mujoco_ros2_control` upstream build
instructions. Its built-in download path is MuJoCo 3.4.0, which is **not**
used and must not be described as parity-compatible with Warp. Every overlay
build passes:

```text
-DAMENT_VENDOR_POLICY=NEVER_VENDOR
-Dmujoco_DIR=<pixi-prefix>/lib/cmake/mujoco
```

In this mode `mujoco_vendor` exports the externally satisfied
`mujoco::mujoco` CMake dependency instead of downloading MuJoCo 3.4.0.
The bootstrap verifies both source checkout `HEAD` values before every build
and refuses to move an existing checkout.

The forwarding vendor package is built first. Inside the generated overlay
only, `install/mujoco_vendor/opt/mujoco_vendor` points to the Pixi prefix so
the pinned control package's vendor-root discovery resolves 3.10.0 headers
and libraries. MuJoCo 3.10 moved the declarations from legacy `mjtnum.h` into
`mjtype.h`; the pinned optional lidar extension still includes the old name.
The bootstrap therefore generates a one-line forwarding header under the
build workspace's `compat/` directory. Neither upstream checkout is patched.

The required MuJoCo runtime is exactly 3.10.0. It lives in the separate,
rootless Pixi project `/home/noamonti/franka_humble_mujoco_pixi`, with
`mujoco = "3.10.0.*"` in `pixi.toml` and exact artifacts captured by
`pixi.lock`. The resolved conda-forge library artifact is
`libmujoco-3.10.0-h70afa4a_0.conda`, SHA-256
`6431bf092279ccd3fb27c86efe8a5f4b2d898251540b11a7016a699646f3829a`.

The existing Warp environment at `/home/noamonti/warp/.venv` remains
untouched. It is never activated, inspected, upgraded, or used by the
bootstrap.

The primary Pixi environment uses Python 3.12 and its locked ONNX Runtime.
The optional apt backend uses Jammy Python 3.10, for which ONNX Runtime 1.24+
has no wheels; that backend creates a workspace-local venv and pins
`onnxruntime==1.23.2`. Its CPython 3.10 x86-64 wheel SHA-256 is
`4ca88747e708e5c67337b0f65eed4b7d0dd70d22ac332038c9fc4635760018f7`.

## What the bootstrap changes

The default build workspace is `~/franka_humble_mujoco_ws`. External
repositories and all `build`, `install`, and `log` output stay there. The
separate Pixi project defaults to `/home/noamonti/franka_humble_mujoco_pixi`.
The source checkout is not checked out, cleaned, generated into, or otherwise
rewritten.

Only these local package directories are linked into `src`:

- `/home/noamonti/franka_ros2/franka_emika_panda`
- `/home/noamonti/franka_ros2/franka_mujoco_bringup`

Set `FRANKA_ROS2_SOURCE_ROOT` or pass `--source-root` if the checkout is
elsewhere. Set `FRANKA_HUMBLE_MUJOCO_WS` or pass `--workspace` to relocate the
overlay. Set `FRANKA_MUJOCO_PIXI_PROJECT` or pass `--pixi-project` to relocate
the rootless environment. Existing nonmatching source links are never
replaced.

The default `--backend pixi` is rootless. The script pins Pixi itself to
0.76.2, preserves an existing manifest/lock, runs `pixi install --locked`,
and verifies all of:

```text
lib/cmake/mujoco/mujocoConfig.cmake
lib/libmujoco.so.3.10.0
include/simulate/simulate.h
```

If no Pixi project exists, the script creates a manifest with the Humble
RoboStack/control/vision dependencies and `mujoco = "3.10.0.*"`, then creates
its initial lock. Subsequent installs require that lock to match the manifest.

The explicit `--backend apt` alternative configures the official ROS 2 apt
repository and invokes `sudo` for apt/repository and rosdep initialization.
It still creates/validates the external Pixi MuJoCo 3.10.0 prefix and still
uses `NEVER_VENDOR`; there is no apt fallback to the vendor's 3.4.0 download.
The apt backend installs these groups:

- ROS base and build tools: `ros-humble-ros-base`,
  `python3-colcon-common-extensions`, `python3-vcstool`, `python3-rosdep`,
  `ros-humble-ament-cmake-vendor-package`, `git`, `pkg-config`, `patchelf`,
  `build-essential`, `cmake`, `ninja-build`, and `libglfw3-dev`.
- Control: `ros-humble-ros2-control`, `ros-humble-ros2-controllers`,
  `ros-humble-controller-manager`, `ros-humble-joint-state-broadcaster`,
  `ros-humble-joint-trajectory-controller`, `ros-humble-gripper-controllers`,
  and `ros-humble-control-msgs`.
- Robot/model and vision: `ros-humble-robot-state-publisher`,
  `ros-humble-xacro`, `ros-humble-image-transport`, `ros-humble-cv-bridge`,
  `ros-humble-message-filters`, `python3-pykdl`, and
  `ros-humble-kdl-parser`.
- Python/test runtime: `python3-numpy`, `python3-opencv`, `python3-pytest`,
  `python3-pip`, `python3-venv`, and `python3-yaml`.

The Pixi backend performs no privileged operation. With the apt backend,
privileged actions occur only for `--install` or when rosdep needs apt
packages during `--build`.

## Exact setup workflow

Start from a clean terminal. Do not source a Jazzy installation or workspace;
the script rejects `ROS_DISTRO` or environment paths that indicate Jazzy.

```bash
cd /home/noamonti/franka_ros2

# Review actions without changing anything.
./scripts/bootstrap_humble_mujoco.sh --help

# Rootless locked Humble + MuJoCo 3.10.0 environment.
./scripts/bootstrap_humble_mujoco.sh --backend pixi --install

# Pinned imports and the two local symlinks.
./scripts/bootstrap_humble_mujoco.sh --backend pixi --import

# Selective colcon build with NEVER_VENDOR.
./scripts/bootstrap_humble_mujoco.sh --backend pixi --build
```

The final build selection is exactly:

```text
mujoco_vendor
mujoco_ros2_control_msgs
mujoco_3d_lidar
mujoco_ros2_control_plugins
mujoco_ros2_control
franka_emika_panda
franka_mujoco_bringup
```

`mujoco_vendor` is configured in its own first stage with
`AMENT_VENDOR_POLICY=NEVER_VENDOR`; the remaining six packages are then built
against that forwarding package. Upstream C++ tests are disabled for this
release build; the bringup package's ROS-independent static pytest validation
can be run separately.

All three stages may be run together:

```bash
./scripts/bootstrap_humble_mujoco.sh --backend pixi --all
```

To reproduce Pixi itself manually without changing shell startup files:

```bash
curl -fsSL https://pixi.sh/install.sh |
  PIXI_VERSION=0.76.2 PIXI_NO_PATH_UPDATE=1 sh
~/.pixi/bin/pixi install \
  --manifest-path /home/noamonti/franka_humble_mujoco_pixi/pixi.toml \
  --locked
```

Every new terminal must activate the separate Pixi prefix and overlay:

```bash
export CONDA_PREFIX=/home/noamonti/franka_humble_mujoco_pixi/.pixi/envs/default
export PATH="$CONDA_PREFIX/bin:$PATH"
export CMAKE_PREFIX_PATH="$CONDA_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
source "$CONDA_PREFIX/setup.bash"
source ~/franka_humble_mujoco_ws/install/setup.bash

test "$ROS_DISTRO" = humble
test -f "$CONDA_PREFIX/lib/libmujoco.so.3.10.0"
python -c 'import mujoco, onnxruntime; print(mujoco.__version__, onnxruntime.__version__)'
```

The MuJoCo version must print `3.10.0`.

The apt alternative is:

```bash
./scripts/bootstrap_humble_mujoco.sh --backend apt --all

source /opt/ros/humble/setup.bash
source ~/franka_humble_mujoco_ws/.venv/bin/activate
export MUJOCO_PREFIX=/home/noamonti/franka_humble_mujoco_pixi/.pixi/envs/default
export CMAKE_PREFIX_PATH="$MUJOCO_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export LD_LIBRARY_PATH="$MUJOCO_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
source ~/franka_humble_mujoco_ws/install/setup.bash
```

## Launch

Simulation starts with policy execution disabled:

```bash
ros2 launch franka_mujoco_bringup panda_pick_place.launch.py \
  headless:=false \
  sim_speed_factor:=1.0 \
  policy_enabled:=false
```

For headless CI or WSL without an available display:

```bash
ros2 launch franka_mujoco_bringup panda_pick_place.launch.py \
  headless:=true \
  sim_speed_factor:=1.0 \
  policy_enabled:=false
```

Before enabling Student v6, confirm the camera stream, joint names, controller
states, model path, and policy observation contract. The installed
`panda_student_v6_ros2` executable and promoted bundle are the defaults:

```bash
ros2 launch franka_mujoco_bringup panda_pick_place.launch.py \
  headless:=true \
  policy_enabled:=true
```

Use `ros2 launch franka_mujoco_bringup panda_pick_place.launch.py --show-args`
before enabling policy execution to verify the installed launch interface.
The policy must preserve the Student v6 NHWC `float32` 64x64 RGB `[0,255]`
input contract and its phase-selective DLS controller; the ONNX model alone
does not implement that complete behavior.

## Runtime interfaces

Expected controller state and command interfaces:

```bash
ros2 control list_controllers
ros2 topic echo /joint_states --once
ros2 action list -t
```

- Simulation arm command topic:
  `/panda_arm_controller/commands`
  (`std_msgs/msg/Float64MultiArray`). The forward position controller holds
  each 60 Hz policy target across the four 240 Hz physics steps.
- Gripper action:
  `/panda_gripper_controller/gripper_cmd`
  (`control_msgs/action/GripperCommand`)

The policy camera is `policy_camera` in MJCF. Its streaming interfaces are:

- RGB: `/policy_camera/color/image_raw`
- Depth: `/policy_camera/depth/image_raw`
- Camera info: `/policy_camera/camera_info`
- Optical frame: `policy_camera_optical_frame`

Check the trained input directly:

```bash
ros2 topic hz /policy_camera/color/image_raw
ros2 topic echo /policy_camera/camera_info --once
```

The pinned runtime creates private simulation services under its internal node
name:

```text
/mujoco_ros2_control_node/set_pause
  mujoco_ros2_control_msgs/srv/SetPause
/mujoco_ros2_control_node/reset_world
  mujoco_ros2_control_msgs/srv/ResetWorld
/mujoco_ros2_control_node/step_simulation
  mujoco_ros2_control_msgs/srv/StepSimulation
```

Examples:

```bash
ros2 service call /mujoco_ros2_control_node/set_pause \
  mujoco_ros2_control_msgs/srv/SetPause '{paused: true}'

ros2 service call /mujoco_ros2_control_node/step_simulation \
  mujoco_ros2_control_msgs/srv/StepSimulation '{steps: 1}'

ros2 service call /mujoco_ros2_control_node/reset_world \
  mujoco_ros2_control_msgs/srv/ResetWorld '{keyframe: home}'

ros2 service call /mujoco_ros2_control_node/set_pause \
  mujoco_ros2_control_msgs/srv/SetPause '{paused: false}'
```

Controller-manager services remain under `/controller_manager`, including
`list_controllers`, `load_controller`, `configure_controller`,
`switch_controller`, and `unload_controller`.

## Validation status

Validated on Ubuntu 22.04 in the isolated rootless Humble/Pixi overlay:

- Clean selective build: seven packages, linked to
  `libmujoco.so.3.10.0`.
- Package tests: 46 passed, zero failures (38 policy/runtime and 8 bringup).
- Monitor-only launch: all controllers active, 64x64 `rgb8` camera live,
  deterministic pause/step/reset services successful, and no arm command
  messages while `policy_enabled=false`.
- Kinematics: the portable Panda analytic TCP Jacobian agrees with MuJoCo
  over 50 random configurations to `1.1e-15`; optional PyKDL remains
  available when `kdl_parser_py` is installed.
- Enabled smoke: ONNX controls the first 30 steps of each approach phase,
  DLS engages at step 30, and phases advance through CloseGripper.
- Simulation gripper close commands follow the promoted 20-step per-finger
  target path. Goals are serialized so the 60 Hz loop does not preempt the
  active `GripperCommand`; only a physical stalled result latches a grasp.
- A grasp retry preserves the cube pose, retreats the TCP vertically by
  75 mm, opens to at least 90%, and waits for a fresh valid estimate before
  resuming HoverRed. No retry teleports the cube or arm.
- Each policy step consumes one fresh image and joint/gripper samples
  interpolated to that image timestamp. TCP pose is exact Panda FK from the
  synchronized joints; sampled TF remains a required health/frame check.
- The 240 Hz snapshot poller does not advance on reused frames. Normal
  `no_fresh_image` polls are silent, avoiding a telemetry-induced backlog.
- Simulation publishes the trained smoothed target directly and holds it over
  four 240 Hz RK4 actuator updates. The second command-history limiter is
  bypassed only in simulation; hardware retains all software bounds.
- Student-v24 ONNX replay matches the direct JAX trace to `2.98e-7` action and
  `2.38e-7` geometry max error. The ROS trace replays bit-exactly.
- A complete 943-step direct trace replays through the ROS-free state builder
  to `6e-8`, integrated target/smoother to `2.2e-6`, and analytic DLS to
  `2.7e-4`.
- The final clean RK4 16-seed ROS run placed 9/16:
  seeds 1, 4, 5, 6, 7, 8, 9, 12, and 14. Every successful grasp completed
  release and placement; all seven failures exhausted physical grasp retries
  in CloseGripper. Limiter interventions were zero.
- Successful no-retry runs measured about 0.12-0.13 rad/s command-velocity
  RMS, 1.8-1.9 rad/s² acceleration RMS, and 179-190 rad/s³ jerk RMS. Retry
  episodes are intentionally less smooth because they contain phase resets.

The direct-JAX promoted report remains 14/16, so ROS has not met the 13/16
parity target. The remaining gap is physical grasp acquisition under small
renderer/contact differences, not ONNX inference, command limiting,
timestamp alignment, release, or placement. Do not weaken the stall-only
grasp latch to inflate this score.

Run the deterministic matrix and capture aligned traces with:

```bash
ros2 run franka_mujoco_bringup evaluate_student_v24.py \
  --timeout 90 \
  --output /tmp/student_v24_ros_16seed.json \
  --trace-directory /tmp/student_v24_ros_traces

ros2 run franka_mujoco_bringup compare_policy_traces.py \
  --direct /path/to/direct_golden_trace.npz \
  --ros /path/to/ros_seed_1.npz \
  --model /path/to/student_v24_urlab_candidate.onnx \
  --output /tmp/direct_ros_trace_comparison.json
```

## ROS geometry-adaptation experiment

A sim-only oracle DescendGrasp check exposed two independent issues:

- Ground-truth box targeting with the original 5 mm grasp-width tolerance
  produced 0/16 physical grasps because the ROS gripper collision model stalls
  at 57–59 mm.
- Keeping the true-box alignment gate and using a simulation-only 10 mm width
  tolerance produced 14/16 physical placements. Hardware Grasp behavior was
  not changed.

The labeled corpus is
`artifacts/observable_sim2real/ros_geometry_corpus_64eps.npz`
(SHA-256 `dcc317b3e561872c5144df190e7975991223318976823f4a1b0eb671d3dde443`):
64 episodes, 53,655 synchronized frames, and 14,744 DescendGrasp frames.
Seeds 4, 8, 12, and 16 are held out.

The new phase-1-only residual freezes v24's action, backbone, and HoverRed
residual. Offline held-out cube MAE improved from 32.7 mm to 21.5 mm, and
ONNX parity remained below `3e-7`. No candidate was promoted:

- Full XYZ residual: 7/16 ROS physical placements, 10/16 direct MJX.
- Linear 25% XYZ residual: 1/16 ROS, 14/16 direct MJX.
- Full X-only residual: 0/16 ROS, 14/16 direct MJX.

The full residual still predicts cube Z about 29 mm high after its own
closed-loop trajectory changes. This is an on-policy covariate-shift failure;
a second labeled DAgger round is needed. The deployed
`student_v24_urlab_candidate.onnx` remains unchanged. Exact hashes and reports
are recorded in
`student_v26_ros_geometry_adaptation_report.json`.

### Round-2 and OAK-1 qualification

Round 2 added 16 full-residual on-policy episodes (15,332 frames, 3,905
DescendGrasp frames) and continued the phase-1 head with a 1/1/4 XYZ loss.
The full candidate reduced final-DescendGrasp mean Z bias to -4.55 mm, but
failed direct MJX at 10/16. An XZ-only candidate also failed at 11/16.
Neither was run as a promotion candidate in ROS; v24 remains the rollback and
deployment model.

The OAK transport emulator sits between the ideal CameraPlugin stream and the
policy, preserving capture timestamps while injecting delay, jitter and
drops. The policy subscription is best-effort depth 1. Seed 1 completed under
all tested profiles through 67 ms delay and at 1%/5% frame drops:

- 33 ms + 5 ms jitter: 35.0 ms mean, 41.7 ms p95.
- 50 ms + 5 ms jitter: 52.1 ms mean, 58.3 ms p95.
- 67 ms + 5 ms jitter: 69.2 ms mean, 75.0 ms p95.
- 100 ms + 5 ms jitter: 102.0 ms mean, 108.3 ms p95; policy did not proceed.

All six light camera variants (±1.6 mm X, ±0.3° pitch, ±0.3° FOV) completed
the oracle seed-1 rollout. The expected physical camera contract is recorded
in `config/oak1_policy_camera.yaml`; it is not a substitute for reading the
connected OAK EEPROM and calibrating the mounted camera.

### Temporal visual adapter experiment

A runtime-compatible parallel adapter was trained directly from pixels:
RGB+coordinates → 5×5 stride-2 Conv(8) → global average pooling → 3-D
DescendGrasp correction. Its final layer is zero-initialized and the v24
action, HoverRed geometry and ONNX interface remain exact.

Corpus schema v2 fixes phase-step ordering and preserves frame sequence and
timestamps. Training used 79,278 pre-contact temporal pairs, world-space
delta consistency, OAK-style blur/photometric/FOV augmentation, and balanced
ROS/MJX/URLab data. Held-out Z MAE reached 4.25 mm, but closed-loop direct MJX
remained the blocker:

- Full adapter: 11/16 placements.
- Linear 25% adapter: 12/16 placements.
- Required gate: 14/16.

The adapter was rejected before ROS DAgger collection. v24 remains deployed.
The complete decision and hashes are in
`student_v28_temporal_adapter_report.json`.

A follow-up domain-specific run applied teacher supervision only to ROS
frames, using MJX and URLab strictly as zero-change continuity anchors. Frozen
v24 tensors remained byte-exact, but the direct gate still failed:

- Full adapter: 12/16 in the 16-world comparison and 11/16 in the promoted
  256-world evaluation shape.
- Linear 25%: 13/16 in the promoted shape.
- Linear 12.5%: 12/16 in the promoted shape.

No domain adapter was allowed into ROS evaluation. The report is
`student_v29_domain_adapter_report.json`.

The linear 25% v29 adapter was subsequently accepted as an **experimental**
sim2real candidate with an explicit cross-engine waiver. It does not replace
v24 and is not hardware-qualified:

- Direct MJX promoted shape: 13/16.
- Live-geometry ROS physical placement: 1/16.
- Full-scale DR MJX: 8/16.
- OAK stress profiles (33 ms/1%, 33 ms/5%, 67 ms): 0/3 successes.

Its model and full-scale DR contract are recorded in
`student_v29_ros_sim2real_experimental_controller.json`. Hardware remains
disabled by default and v24 is the mandatory rollback.

## Branch and hardware safety

The overlay intentionally follows the current working tree through symlinks.
That permits testing uncommitted package work without modifying or committing
it, but it also means a later edit in either linked package is visible to the
next overlay build. The pinned external repositories are separate clones and
must remain at their recorded commits.

This launch is simulation-only. Do not remap its joint trajectory or gripper
actions to a real Panda, do not start a Franka hardware driver alongside it,
and do not treat a successful MuJoCo rollout as authorization for hardware
execution. Real hardware requires independent frame calibration, collision
and velocity limits, an emergency-stop procedure, workspace clearance, and a
gripper transition based on the real `franka_msgs/action/Grasp` result plus
measured width. Begin hardware validation with policy output disabled.
