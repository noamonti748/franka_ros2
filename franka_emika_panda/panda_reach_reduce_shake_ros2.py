#!/usr/bin/env python3
"""
Panda single-target "reach and hold" behaviour for the ppo_final_reduce_shake checkpoint.

Unlike panda_track_reach_ros2.py (multi-target "track": resamples a brand-new random
target every time the arm succeeds), this node reaches ONE target sampled at episode
start and just holds station there -- no resampling on success. Observation
preprocessing defaults match ppo_final_reduce_shake's ONNX interface, and launch
parameters can switch the same node to the lower-vibration checkpoint:

  - joint1..joint7: [qpos/pi, qvel/2.0], each dim clipped to +/-3.
  - lower-vib mode: [qpos/pi, qvel/2.0, prev_qpos/pi, prev_qvel/2.0].
  - prev_action: previous 7-joint command vector, optionally divided by a scale
    such as pi for ppo_final_lower_vib.
  - tcp_pos: (tcp_m - target_m) / 0.75, clipped to +/-3.
  - action: 7-value absolute joint position target vector in radians.

This module intentionally reuses the ROS-free plumbing and constants from
panda_track_reach_ros2.py (PANDA_ARM_JOINTS, joint limits, vec3 parsing, joint-state
mapping, action-output interpretation, etc.) via subclassing/composition rather than
duplicating them, so panda_track_reach_ros2.py's legacy ppo_track_franka.onnx behaviour
stays completely untouched.

Run as a ROS 2 node when rclpy is available:

    python panda_reach_reduce_shake_ros2.py --ros --policy

Dry-run without ROS (prints a short simulated episode):

    python panda_reach_reduce_shake_ros2.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

# Running this file directly (python panda_reach_reduce_shake_ros2.py, or via `ros2 run`
# against the installed extensionless executable) puts this directory on sys.path[0]
# already; the explicit insert below is just a defensive fallback for any other
# invocation style, so `import panda_track_reach_ros2` reliably finds the sibling
# module installed alongside it (see CMakeLists.txt).
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from panda_track_reach_ros2 import (  # noqa: E402
    PANDA_ARM_JOINTS,
    PANDA_HOME_JOINTS,
    PANDA_JOINT_LIMITS,
    ROBOT_JOINT_LIMITS,
    OnnxPolicyRunner,
    PandaTrackReachConfig,
    PandaTrackReachSession,
    _clip_joint_positions,
    _interpret_policy_actions,
    _parse_vec3_text,
    _vec3,
    joint_state_to_maps,
)

# Normalization constants for joint_observation_mode="normalized_position_velocity":
# obs = [qpos/pi, qvel/2.0], matching the reduce-shake ONNX joint inputs.
JOINT_OBS_POS_SCALE: float = math.pi
JOINT_OBS_VEL_SCALE: float = 2.0
LOWER_VIB_JOINT_OBSERVATION_MODES: frozenset[str] = frozenset(
    (
        "normalized_position_velocity_previous_position_velocity",
        "normalized_position_velocity_prev_position_prev_velocity",
    )
)


def _parse_float_list_text(text: str, expected_len: int) -> tuple[float, ...]:
    values = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
    if len(values) == 1 and " " in values[0]:
        values = [part for part in values[0].split() if part]
    if len(values) != expected_len:
        raise ValueError(f"expected {expected_len} comma-separated values, got {text!r}")
    return tuple(float(value) for value in values)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("1", "true", "yes", "on"):
            return True
        if normalized in ("0", "false", "no", "off"):
            return False
    return bool(value)


class _BiquadNotch:
    """Small dependency-free RBJ notch filter for one scalar command stream."""

    def __init__(self, sample_hz: float, frequency_hz: float, q: float) -> None:
        self.enabled = (
            sample_hz > 0.0
            and frequency_hz > 0.0
            and frequency_hz < 0.49 * sample_hz
            and q > 0.0
        )
        self._initialized = False
        self._x1 = 0.0
        self._x2 = 0.0
        self._y1 = 0.0
        self._y2 = 0.0
        if not self.enabled:
            self._b0 = 1.0
            self._b1 = 0.0
            self._b2 = 0.0
            self._a1 = 0.0
            self._a2 = 0.0
            return

        omega = 2.0 * math.pi * frequency_hz / sample_hz
        cos_omega = math.cos(omega)
        alpha = math.sin(omega) / (2.0 * q)
        a0 = 1.0 + alpha
        self._b0 = 1.0 / a0
        self._b1 = -2.0 * cos_omega / a0
        self._b2 = 1.0 / a0
        self._a1 = -2.0 * cos_omega / a0
        self._a2 = (1.0 - alpha) / a0

    def reset(self, value: float) -> None:
        self._initialized = True
        self._x1 = float(value)
        self._x2 = float(value)
        self._y1 = float(value)
        self._y2 = float(value)

    def filter(self, value: float) -> float:
        if not self.enabled:
            return float(value)
        x0 = float(value)
        if not self._initialized:
            self.reset(x0)
            return x0
        y0 = (
            self._b0 * x0
            + self._b1 * self._x1
            + self._b2 * self._x2
            - self._a1 * self._y1
            - self._a2 * self._y2
        )
        self._x2 = self._x1
        self._x1 = x0
        self._y2 = self._y1
        self._y1 = y0
        return float(y0)


class _JointActionNotchFilter:
    """Cascaded per-joint notch filters for absolute joint target commands."""

    def __init__(
        self,
        sample_hz: float,
        primary_frequencies_hz: Sequence[float],
        primary_q: float,
        secondary_frequencies_hz: Sequence[float],
        secondary_q: float,
    ) -> None:
        self._joint_filters: list[list[_BiquadNotch]] = []
        for i in range(7):
            stages: list[_BiquadNotch] = []
            primary_frequency = (
                float(primary_frequencies_hz[i])
                if i < len(primary_frequencies_hz)
                else 0.0
            )
            secondary_frequency = (
                float(secondary_frequencies_hz[i])
                if i < len(secondary_frequencies_hz)
                else 0.0
            )
            if primary_frequency > 0.0:
                stages.append(_BiquadNotch(sample_hz, primary_frequency, primary_q))
            if secondary_frequency > 0.0:
                stages.append(_BiquadNotch(sample_hz, secondary_frequency, secondary_q))
            self._joint_filters.append(stages)

    def reset(self, values: Sequence[float]) -> None:
        for i, stages in enumerate(self._joint_filters):
            value = float(values[i]) if i < len(values) else 0.0
            for stage in stages:
                stage.reset(value)

    def filter(self, values: Sequence[float]) -> list[float]:
        filtered: list[float] = []
        for i, value in enumerate(values):
            output = float(value)
            stages = self._joint_filters[i] if i < len(self._joint_filters) else []
            for stage in stages:
                output = stage.filter(output)
            filtered.append(output)
        return filtered


def _sample_target_offset_reach_hold(
    rng: np.random.Generator,
    offset_max_m: tuple[float, float, float],
    symmetric_z_offset: bool,
) -> np.ndarray:
    """Like panda_track_reach_ros2.sample_target_offset, but Z is also drawn
    U[-max, +max] when symmetric_z_offset is set. ppo_final_reduce_shake was trained
    with a symmetric ~0.06 m radius offset on all three axes (X, Y, Z), unlike the
    legacy track checkpoint's positive-only Z convention.

    ASSUMPTION: the training spec ("random radius 0.06 on X, Y, Z") was ambiguous
    about whether Z sampling stayed positive-only or became symmetric; this defaults
    to symmetric (see ReachConstrainedConfig.symmetric_z_offset).
    """

    def symmetric(max_offset: float) -> float:
        if max_offset <= 0.0:
            return 0.0
        return float(rng.uniform(-max_offset, max_offset))

    def positive(max_offset: float) -> float:
        if max_offset <= 0.0:
            return 0.0
        return float(rng.uniform(0.0, max_offset))

    z_offset = symmetric(offset_max_m[2]) if symmetric_z_offset else positive(offset_max_m[2])
    return np.array(
        [symmetric(offset_max_m[0]), symmetric(offset_max_m[1]), z_offset],
        dtype=np.float64,
    )


@dataclass
class ReachConstrainedConfig(PandaTrackReachConfig):
    """PandaTrackReachConfig specialized for the ppo_final_reduce_shake "reach and
    hold" checkpoint: training center/radius, symmetric offset sampling on all three
    axes, no target resampling on success, no inherited track subgoal timeout, and
    no action smoothing/rate-limiting (raw policy, matching training -- re-enable
    for real hardware if desired)."""

    target_location_m: tuple[float, float, float] = (0.32, 0.0, 0.5)
    target_random_offset_max_m: tuple[float, float, float] = (0.06, 0.06, 0.06)
    subgoal_step_limit: int = 0
    symmetric_z_offset: bool = True
    resample_target_on_success: bool = False
    smooth_actions: bool = False
    limit_action_delta: bool = False


class ReachConstrainedSession(PandaTrackReachSession):
    """PandaTrackReachSession specialized for "reach and hold": symmetric-Z target
    offset sampling, and no resampling to a new target after success (just keep
    holding the current one). Reuses all reward/termination/smoothing logic from the
    parent class unchanged."""

    def __init__(self, config: Optional[ReachConstrainedConfig] = None) -> None:
        super().__init__(config or ReachConstrainedConfig())

    def sample_episode_target_location(self) -> np.ndarray:
        cfg = self.config
        target = np.array(cfg.target_location_m, dtype=np.float64)
        if not cfg.randomize_target_per_episode:
            self.episode_target_location_m = target
            return target.copy()

        stream_seed = (
            (cfg.random_seed + self._target_counter)
            if cfg.random_seed != 0
            else self._target_counter
        )
        self._target_counter += 1
        local_rng = np.random.default_rng(stream_seed)
        target = target + _sample_target_offset_reach_hold(
            local_rng, cfg.target_random_offset_max_m, cfg.symmetric_z_offset
        )
        self.episode_target_location_m = target
        return target.copy()

    def advance_to_next_subgoal(self, completed_episode_step_index: int) -> np.ndarray:
        if not self.config.resample_target_on_success:
            # "Reach and hold": keep the same target, just push the subgoal anchor
            # forward so a continuous hold never trips subgoal_step_limit.
            self.subgoal_start_episode_step_index = completed_episode_step_index
            self.last_successful_subgoal_episode_step_index = completed_episode_step_index
            return self.episode_target_location_m.copy()
        return super().advance_to_next_subgoal(completed_episode_step_index)


class ReachReduceShakePolicyRunner:
    """ONNX runner for reach/hold policies with a single 7-value action output.

    Supports the original reduce-shake 2-value joint observations and the lower-vib
    4-value joint observations that include previous measured position/velocity.
    """

    def __init__(self, model_path: str) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required to run ppo_final_reduce_shake.onnx. "
                "Install it in the active Python environment, for example: "
                "python3 -m pip install --user onnxruntime"
            ) from exc

        self.model_path = str(Path(model_path).expanduser())
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        self._session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        self.input_names = [inp.name for inp in self._session.get_inputs()]
        self.output_names = [out.name for out in self._session.get_outputs()]

        missing_inputs = [
            name for name in (*PANDA_ARM_JOINTS, "prev_action", "tcp_pos")
            if name not in self.input_names
        ]
        if missing_inputs or "action" not in self.output_names:
            raise RuntimeError(
                "Unexpected reduce-shake ONNX policy interface. "
                f"missing_inputs={missing_inputs}, expected_output=action, "
                f"inputs={self.input_names}, outputs={self.output_names}"
            )

    @staticmethod
    def _joint_observation_normalized(
        position: float,
        velocity: float,
        previous_position: Optional[float] = None,
        previous_velocity: Optional[float] = None,
        mode: str = "normalized_position_velocity",
        clip: Optional[float] = None,
    ) -> np.ndarray:
        values = [
            position / JOINT_OBS_POS_SCALE,
            velocity / JOINT_OBS_VEL_SCALE,
        ]
        if mode in LOWER_VIB_JOINT_OBSERVATION_MODES:
            prev_pos = position if previous_position is None else previous_position
            prev_vel = velocity if previous_velocity is None else previous_velocity
            values.extend(
                [
                    prev_pos / JOINT_OBS_POS_SCALE,
                    prev_vel / JOINT_OBS_VEL_SCALE,
                ]
            )
        elif mode != "normalized_position_velocity":
            raise ValueError(
                "Unsupported joint_observation_mode for single-action reach policy: "
                f"{mode!r}. Use normalized_position_velocity or "
                "normalized_position_velocity_previous_position_velocity."
            )
        obs = np.array(values, dtype=np.float32)
        if clip is not None and clip > 0.0:
            obs = np.clip(obs, -clip, clip)
        return obs

    @staticmethod
    def _clip_obs(obs: np.ndarray, clip: Optional[float]) -> np.ndarray:
        if clip is not None and clip > 0.0:
            obs = np.clip(obs, -clip, clip)
        return obs.astype(np.float32)

    def compute_action(
        self,
        joint_pos: Sequence[float],
        joint_vel: Sequence[float],
        last_command: Sequence[float],
        tcp_m: Sequence[float],
        target_m: Sequence[float],
        *,
        tcp_observation_mode: str = "tcp_error_normalized",
        tcp_observation_scale: float = 0.75,
        tcp_observation_signs: Sequence[float] = (1.0, 1.0, 1.0),
        tcp_observation_unit_scale: float = 1.0,
        joint_observation_mode: str = "normalized_position_velocity",
        prev_joint_pos: Optional[Sequence[float]] = None,
        prev_joint_vel: Optional[Sequence[float]] = None,
        joint_acc: Optional[Sequence[float]] = None,
        observation_clip: Optional[float] = 3.0,
        action_scale: float = 1.0,
        prev_action_scale: float = 1.0,
    ) -> list[float]:
        _raw_action, scaled_action = self.compute_action_with_raw_output(
            joint_pos,
            joint_vel,
            last_command,
            tcp_m,
            target_m,
            tcp_observation_mode=tcp_observation_mode,
            tcp_observation_scale=tcp_observation_scale,
            tcp_observation_signs=tcp_observation_signs,
            tcp_observation_unit_scale=tcp_observation_unit_scale,
            joint_observation_mode=joint_observation_mode,
            prev_joint_pos=prev_joint_pos,
            prev_joint_vel=prev_joint_vel,
            joint_acc=joint_acc,
            observation_clip=observation_clip,
            action_scale=action_scale,
            prev_action_scale=prev_action_scale,
        )
        return scaled_action

    def compute_action_with_raw_output(
        self,
        joint_pos: Sequence[float],
        joint_vel: Sequence[float],
        last_command: Sequence[float],
        tcp_m: Sequence[float],
        target_m: Sequence[float],
        *,
        tcp_observation_mode: str = "tcp_error_normalized",
        tcp_observation_scale: float = 0.75,
        tcp_observation_signs: Sequence[float] = (1.0, 1.0, 1.0),
        tcp_observation_unit_scale: float = 1.0,
        joint_observation_mode: str = "normalized_position_velocity",
        prev_joint_pos: Optional[Sequence[float]] = None,
        prev_joint_vel: Optional[Sequence[float]] = None,
        joint_acc: Optional[Sequence[float]] = None,
        observation_clip: Optional[float] = 3.0,
        action_scale: float = 1.0,
        prev_action_scale: float = 1.0,
    ) -> tuple[list[float], list[float]]:
        if len(joint_pos) != 7 or len(joint_vel) != 7 or len(last_command) != 7:
            raise ValueError("joint_pos, joint_vel, and last_command must all have length 7")
        if prev_joint_pos is not None and len(prev_joint_pos) != 7:
            raise ValueError("prev_joint_pos must have length 7 when provided")
        if prev_joint_vel is not None and len(prev_joint_vel) != 7:
            raise ValueError("prev_joint_vel must have length 7 when provided")
        if joint_acc is not None and len(joint_acc) != 7:
            raise ValueError("joint_acc must have length 7 when provided")

        feed: dict[str, np.ndarray] = {}
        joint_dim = 4 if joint_observation_mode in LOWER_VIB_JOINT_OBSERVATION_MODES else 2
        for i, joint_name in enumerate(PANDA_ARM_JOINTS):
            obs = self._joint_observation_normalized(
                float(joint_pos[i]),
                float(joint_vel[i]),
                previous_position=(
                    float(prev_joint_pos[i]) if prev_joint_pos is not None else None
                ),
                previous_velocity=(
                    float(prev_joint_vel[i]) if prev_joint_vel is not None else None
                ),
                mode=joint_observation_mode,
                clip=observation_clip,
            )
            feed[joint_name] = obs.reshape(1, joint_dim)

        tcp_vec = self._clip_obs(
            OnnxPolicyRunner._tcp_observation(
                tcp_observation_mode,
                _vec3(tcp_m).astype(np.float32),
                _vec3(target_m).astype(np.float32),
                float(tcp_observation_scale),
                tcp_observation_signs,
                float(tcp_observation_unit_scale),
            ),
            observation_clip,
        )
        feed["tcp_pos"] = tcp_vec.reshape(1, 3)
        prev_action = np.asarray(last_command, dtype=np.float32)
        if prev_action_scale > 0.0:
            prev_action = prev_action / float(prev_action_scale)
        feed["prev_action"] = self._clip_obs(prev_action, observation_clip).reshape(1, 7)

        raw_action = self._session.run(["action"], feed)[0]
        action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        if action.size != 7:
            raise RuntimeError(f"Expected ONNX output 'action' to contain 7 values, got {action.size}")
        raw_values = [float(value) for value in action]
        scaled_values = [float(value) * action_scale for value in raw_values]
        return raw_values, scaled_values


def _default_onnx_model_path() -> str:
    try:
        from ament_index_python.packages import get_package_share_directory

        return str(Path(get_package_share_directory("franka_emika_panda")) / "ppo_final_reduce_shake.onnx")
    except Exception:
        return str(Path(__file__).resolve().parent / "ppo_final_reduce_shake.onnx")


def _dry_run(seed: int, steps: int) -> None:
    session = ReachConstrainedSession()
    session.seed(seed)
    target = session.reset()
    print(f"reset -> target (task frame m): {tuple(round(float(x), 4) for x in target)}")

    tcp = np.array(target) + np.array([0.12, 0.0, 0.05])
    for step_idx in range(steps):
        if step_idx > steps - 5:
            tcp = session.episode_target_location_m.copy()
        else:
            tcp = tcp + 0.02 * (session.episode_target_location_m - tcp)

        outcome = session.step(tuple(tcp))
        print(
            f"step {step_idx:3d}  dist={outcome.distance_m:.4f}  "
            f"reward={outcome.reward:7.3f}  success={outcome.success}  "
            f"target={tuple(round(x, 3) for x in outcome.target_m)}  "
            f"term={outcome.terminated} trunc={outcome.truncated}"
        )
        if outcome.terminated or outcome.truncated:
            break


def _run_ros_node(args: argparse.Namespace) -> None:
    try:
        import rclpy
        from builtin_interfaces.msg import Duration
        from geometry_msgs.msg import Point, PoseStamped, TransformStamped
        from rclpy.executors import ExternalShutdownException
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from rclpy.time import Time
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String
        from tf2_ros import Buffer, TransformException, TransformListener
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        from visualization_msgs.msg import Marker
    except ImportError as exc:
        print(
            "rclpy is not available. Source your ROS 2 install, then re-run with --ros.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    class PandaReachReduceShakeNode(Node):
        def __init__(self) -> None:
            super().__init__("panda_reach_reduce_shake")
            self.declare_parameter("robot_type", "panda")
            self.declare_parameter("frame_id", "")
            self.declare_parameter("tcp_source", "tf")
            self.declare_parameter("tcp_frame_id", "")
            self.declare_parameter("tcp_pose_topic", "ee_pose")
            self.declare_parameter("tcp_transform_topic", "measured_tf")
            self.declare_parameter("joint_states_topic", "joint_states")
            # NOTE: these default topic names intentionally match panda_track_reach_ros2.py's
            # defaults (reach_track/...), NOT "reach_constrained/...", because both nodes are
            # meant to share the same panda_track_reach.rviz config (see
            # gazebo_panda_example_controller.launch.py), which hardcodes a Marker display
            # subscribed to /reach_track/target_marker with Namespaces filter "panda_track_reach".
            # Using different topic names here would silently orphan the target marker from
            # that shared rviz config (the previous bug this comment is guarding against).
            self.declare_parameter("target_pose_topic", "reach_track/target")
            self.declare_parameter("target_marker_topic", "reach_track/target_marker")
            self.declare_parameter("target_marker_scale_m", 0.06)
            self.declare_parameter("info_topic", "reach_track/info")
            self.declare_parameter("target_republish_period_s", 1.0)
            self.declare_parameter("command_topic", "")
            self.declare_parameter("command_message_type", "joint_trajectory")
            # control_hz=60.0 matches the training control rate for this checkpoint family:
            # task/track_reach_session.py's PandaTrackReachConfig.action_rate_limit_hz defaults
            # to 60.0, and every related script in both repos (TrackReachEnv, eval.py,
            # panda_mujoco_onnx_rollout.py, panda_reach_constrained_mujoco_rollout.py) defaults
            # control_hz to 60.0 too, so this is NOT the source of the "too fast" deployment
            # feel -- see trajectory_duration_s below for that.
            self.declare_parameter("control_hz", 60.0)
            # trajectory_duration_s is the time_from_start given to joint_trajectory_controller
            # for each single-point JointTrajectory we publish every 1/control_hz seconds. JTC
            # builds a spline from the robot's current interpolated state to (target_position,
            # zero velocity) over this duration, but since we publish a brand-new trajectory
            # (replacing the in-flight one) every 1/control_hz seconds, any duration longer than
            # that period means each spline gets interrupted before it ever decelerates towards
            # its target -- the arm is perpetually caught in the high-velocity/ramp-up portion of
            # the curve, which is what produced the "too fast/snappy" feel that was reported.
            # 0.9 / control_hz keeps each spline just short of a full control period so JTC
            # (almost) finishes decelerating into each waypoint before the next one replaces it,
            # approximating a paced zero-order-hold position update at control_hz -- closer to
            # how MuJoCo's <position kp/kv> actuator servo consumes a fresh ctrl setpoint every
            # control tick during training. If you override control_hz, override this too (aim
            # for roughly 0.8-0.95 / control_hz).
            self.declare_parameter("trajectory_duration_s", 0.9 / 60.0)
            self.declare_parameter("random_seed", 0)
            self.declare_parameter("auto_reset_on_failure", True)
            self.declare_parameter("policy_enabled", bool(args.policy))
            self.declare_parameter("onnx_model_path", args.model or _default_onnx_model_path())
            self.declare_parameter("tcp_observation_mode", "tcp_error_normalized")
            self.declare_parameter("tcp_observation_scale", 0.75)
            self.declare_parameter("tcp_observation_signs", "1,1,1")
            self.declare_parameter("tcp_observation_unit_scale", 1.0)
            self.declare_parameter(
                "joint_observation_mode", "normalized_position_velocity"
            )
            self.declare_parameter("observation_clip", 3.0)
            self.declare_parameter("action_scale", 1.0)
            self.declare_parameter("action_output_mode", "absolute")
            self.declare_parameter("action_delta_scale", 0.05)
            self.declare_parameter("prev_action_scale", 1.0)

            _default_cfg = ReachConstrainedConfig()
            self.declare_parameter(
                "target_location_m",
                ",".join(str(v) for v in _default_cfg.target_location_m),
            )
            self.declare_parameter(
                "target_random_offset_max_m",
                ",".join(str(v) for v in _default_cfg.target_random_offset_max_m),
            )
            self.declare_parameter(
                "resample_target_on_success", _default_cfg.resample_target_on_success
            )
            self.declare_parameter("symmetric_z_offset", _default_cfg.symmetric_z_offset)
            self.declare_parameter("smooth_actions", _default_cfg.smooth_actions)
            self.declare_parameter("limit_action_delta", _default_cfg.limit_action_delta)
            self.declare_parameter(
                "action_smoothing_alpha", _default_cfg.action_smoothing_alpha
            )
            self.declare_parameter("action_rate_limit_hz", _default_cfg.action_rate_limit_hz)
            self.declare_parameter(
                "action_velocity_limits_rad_s",
                ",".join(str(v) for v in _default_cfg.action_velocity_limits_rad_s),
            )
            self.declare_parameter(
                "use_measured_velocity_governor",
                _default_cfg.use_measured_velocity_governor,
            )
            self.declare_parameter(
                "measured_velocity_governor_start_ratio",
                _default_cfg.measured_velocity_governor_start_ratio,
            )
            self.declare_parameter("action_notch_filter_enabled", False)
            self.declare_parameter(
                "action_notch_filter_frequencies_hz",
                "0,0,0,0,0,0,0",
            )
            self.declare_parameter("action_notch_filter_q", 8.0)
            self.declare_parameter(
                "action_notch_filter_secondary_frequencies_hz",
                "0,0,0,0,0,0,0",
            )
            self.declare_parameter("action_notch_filter_secondary_q", 8.0)
            self.declare_parameter("action_notch_filter_stage", "target_action")
            self.declare_parameter("hold_enabled", False)
            self.declare_parameter("hold_enter_tcp_m", 0.02)
            self.declare_parameter("hold_exit_tcp_m", 0.04)
            self.declare_parameter("hold_enter_qvel_l2", 0.5)
            self.declare_parameter("hold_enter_ticks", 10)
            self.declare_parameter("eval_enabled", False)
            self.declare_parameter("eval_target_count", 20)
            self.declare_parameter("eval_reset_home_between_targets", True)
            self.declare_parameter(
                "eval_home_joints",
                ",".join(str(v) for v in PANDA_HOME_JOINTS),
            )
            self.declare_parameter("eval_home_tolerance_rad", 0.03)
            self.declare_parameter("eval_home_qvel_l2", 0.15)
            self.declare_parameter("eval_home_hold_ticks", 12)
            self.declare_parameter("eval_max_steps_per_target", 0)
            self.declare_parameter("eval_shutdown_on_complete", False)
            self.declare_parameter("eval_summary_csv_path", "")
            self.declare_parameter("startup_home_enabled", False)
            self.declare_parameter(
                "startup_home_joints",
                ",".join(str(v) for v in PANDA_HOME_JOINTS),
            )
            self.declare_parameter("startup_home_tolerance_rad", 0.03)
            self.declare_parameter("startup_home_qvel_l2", 0.15)
            self.declare_parameter("startup_home_hold_ticks", 12)
            self.declare_parameter(
                "telemetry_enabled",
                bool(getattr(args, "telemetry_enabled", False) or getattr(args, "telemetry_csv_path", "")),
            )
            self.declare_parameter(
                "telemetry_csv_path",
                str(getattr(args, "telemetry_csv_path", "")),
            )
            self.declare_parameter(
                "telemetry_decimation",
                int(getattr(args, "telemetry_decimation", 1)),
            )
            self.declare_parameter(
                "telemetry_flush_every_n_rows",
                int(getattr(args, "telemetry_flush_every_n_rows", 60)),
            )

            arm_id = str(self.get_parameter("robot_type").value) or "panda"
            self._arm_id = arm_id
            self._arm_joint_names = tuple(f"{arm_id}_joint{i}" for i in range(1, 8))
            self._joint_limits = ROBOT_JOINT_LIMITS.get(arm_id, PANDA_JOINT_LIMITS)

            frame_id = str(self.get_parameter("frame_id").value) or f"{arm_id}_link0"
            self._frame_id = frame_id
            self._tcp_source = str(self.get_parameter("tcp_source").value).strip().lower()
            self._tcp_frame_id = (
                str(self.get_parameter("tcp_frame_id").value) or f"{arm_id}_hand_tcp"
            )

            cfg = ReachConstrainedConfig(
                random_seed=int(self.get_parameter("random_seed").value),
                target_location_m=_parse_vec3_text(
                    str(self.get_parameter("target_location_m").value)
                ),
                target_random_offset_max_m=_parse_vec3_text(
                    str(self.get_parameter("target_random_offset_max_m").value)
                ),
                resample_target_on_success=_as_bool(
                    self.get_parameter("resample_target_on_success").value
                ),
                symmetric_z_offset=_as_bool(self.get_parameter("symmetric_z_offset").value),
                smooth_actions=_as_bool(self.get_parameter("smooth_actions").value),
                limit_action_delta=_as_bool(self.get_parameter("limit_action_delta").value),
                action_smoothing_alpha=float(
                    self.get_parameter("action_smoothing_alpha").value
                ),
                action_rate_limit_hz=float(self.get_parameter("action_rate_limit_hz").value),
                action_velocity_limits_rad_s=_parse_float_list_text(
                    str(self.get_parameter("action_velocity_limits_rad_s").value),
                    7,
                ),
                use_measured_velocity_governor=_as_bool(
                    self.get_parameter("use_measured_velocity_governor").value
                ),
                measured_velocity_governor_start_ratio=float(
                    self.get_parameter("measured_velocity_governor_start_ratio").value
                ),
            )
            self._session = ReachConstrainedSession(cfg)
            self._session.seed(cfg.random_seed)
            self._have_tcp = False
            self._have_joints = False
            self._tcp_m = np.zeros(3, dtype=np.float64)
            self._joint_pos_vector = list(PANDA_HOME_JOINTS)
            self._joint_vel_vector = [0.0] * 7
            self._joint_acc_vector = [0.0] * 7
            self._prev_joint_pos_observation_vector = list(PANDA_HOME_JOINTS)
            self._prev_joint_vel_observation_vector = [0.0] * 7
            self._prev_joint_vel_vector: Optional[list[float]] = None
            self._prev_joint_state_stamp_s: Optional[float] = None
            self._last_command_vector = list(PANDA_HOME_JOINTS)
            self._joint_pos: dict[str, float] = {}
            self._joint_vel: dict[str, float] = {}
            self._policy_error_reported = False
            self._target_marker_scale_m = float(self.get_parameter("target_marker_scale_m").value)
            self._telemetry_enabled = _as_bool(self.get_parameter("telemetry_enabled").value)
            self._telemetry_csv_path = str(self.get_parameter("telemetry_csv_path").value).strip()
            self._telemetry_decimation = max(
                1,
                int(self.get_parameter("telemetry_decimation").value),
            )
            self._telemetry_flush_every_n_rows = max(
                1,
                int(self.get_parameter("telemetry_flush_every_n_rows").value),
            )
            self._telemetry_tick_index = 0
            self._telemetry_rows_written = 0
            self._telemetry_file = None
            self._telemetry_writer = None
            self._last_timer_stamp_s: Optional[float] = None
            self._last_policy_telemetry: dict[str, list[float]] = {}

            target_topic = self.get_parameter("target_pose_topic").value
            target_marker_topic = self.get_parameter("target_marker_topic").value
            info_topic = self.get_parameter("info_topic").value
            self._target_pub = self.create_publisher(PoseStamped, target_topic, 10)
            marker_qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self._target_marker_pub = self.create_publisher(
                Marker,
                target_marker_topic,
                marker_qos,
            )
            self._info_pub = self.create_publisher(String, info_topic, 10)
            command_topic = (
                str(self.get_parameter("command_topic").value)
                or f"{arm_id}_joint_trajectory_controller/joint_trajectory"
            )
            self._command_message_type = (
                str(self.get_parameter("command_message_type").value).strip().lower()
            )
            if self._command_message_type in ("joint_state", "joint_states"):
                self._command_pub = self.create_publisher(
                    JointState,
                    command_topic,
                    10,
                )
                self._command_message_type = "joint_state"
            else:
                if self._command_message_type != "joint_trajectory":
                    self.get_logger().warning(
                        f"Unsupported command_message_type={self._command_message_type!r}; "
                        "using joint_trajectory"
                    )
                self._command_pub = self.create_publisher(
                    JointTrajectory,
                    command_topic,
                    10,
                )
                self._command_message_type = "joint_trajectory"

            tcp_pose_topic = self.get_parameter("tcp_pose_topic").value
            tcp_transform_topic = self.get_parameter("tcp_transform_topic").value
            joint_topic = self.get_parameter("joint_states_topic").value
            if self._tcp_source == "pose_topic":
                self.create_subscription(PoseStamped, tcp_pose_topic, self._on_tcp_pose, 10)
            elif self._tcp_source in ("transform_topic", "tf_topic"):
                self.create_subscription(
                    TransformStamped,
                    tcp_transform_topic,
                    self._on_tcp_transform,
                    10,
                )
            elif self._tcp_source != "tf":
                self.get_logger().warning(
                    f"Unsupported tcp_source={self._tcp_source!r}; waiting for tf lookup"
                )
                self._tcp_source = "tf"
            self.create_subscription(JointState, joint_topic, self._on_joint_state, 10)
            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)

            hz = float(self.get_parameter("control_hz").value)
            self._action_notch_filter_enabled = _as_bool(
                self.get_parameter("action_notch_filter_enabled").value
            )
            self._action_notch_filter_frequencies_hz = _parse_float_list_text(
                str(self.get_parameter("action_notch_filter_frequencies_hz").value),
                7,
            )
            self._action_notch_filter_q = float(
                self.get_parameter("action_notch_filter_q").value
            )
            self._action_notch_filter_secondary_frequencies_hz = _parse_float_list_text(
                str(
                    self.get_parameter(
                        "action_notch_filter_secondary_frequencies_hz"
                    ).value
                ),
                7,
            )
            self._action_notch_filter_secondary_q = float(
                self.get_parameter("action_notch_filter_secondary_q").value
            )
            self._action_notch_filter_stage = str(
                self.get_parameter("action_notch_filter_stage").value
            ).strip().lower()
            if self._action_notch_filter_stage not in (
                "target_action",
                "final_command",
                "both",
            ):
                self.get_logger().warning(
                    f"Unsupported action_notch_filter_stage={self._action_notch_filter_stage!r}; "
                    "using target_action"
                )
                self._action_notch_filter_stage = "target_action"
            self._target_action_notch_filter = _JointActionNotchFilter(
                sample_hz=hz,
                primary_frequencies_hz=(
                    self._action_notch_filter_frequencies_hz
                    if self._action_notch_filter_enabled
                    and self._action_notch_filter_stage in ("target_action", "both")
                    else (0.0,) * 7
                ),
                primary_q=self._action_notch_filter_q,
                secondary_frequencies_hz=(
                    self._action_notch_filter_secondary_frequencies_hz
                    if self._action_notch_filter_enabled
                    and self._action_notch_filter_stage in ("target_action", "both")
                    else (0.0,) * 7
                ),
                secondary_q=self._action_notch_filter_secondary_q,
            )
            self._final_command_notch_filter = _JointActionNotchFilter(
                sample_hz=hz,
                primary_frequencies_hz=(
                    self._action_notch_filter_frequencies_hz
                    if self._action_notch_filter_enabled
                    and self._action_notch_filter_stage in ("final_command", "both")
                    else (0.0,) * 7
                ),
                primary_q=self._action_notch_filter_q,
                secondary_frequencies_hz=(
                    self._action_notch_filter_secondary_frequencies_hz
                    if self._action_notch_filter_enabled
                    and self._action_notch_filter_stage in ("final_command", "both")
                    else (0.0,) * 7
                ),
                secondary_q=self._action_notch_filter_secondary_q,
            )
            self.create_timer(1.0 / max(hz, 1.0), self._on_timer)
            self._auto_reset = bool(self.get_parameter("auto_reset_on_failure").value)
            self._trajectory_duration_s = float(self.get_parameter("trajectory_duration_s").value)
            self._policy_enabled = bool(self.get_parameter("policy_enabled").value)
            self._tcp_observation_mode = str(self.get_parameter("tcp_observation_mode").value)
            self._tcp_observation_scale = float(self.get_parameter("tcp_observation_scale").value)
            self._tcp_observation_signs = _parse_vec3_text(
                str(self.get_parameter("tcp_observation_signs").value)
            )
            self._tcp_observation_unit_scale = float(
                self.get_parameter("tcp_observation_unit_scale").value
            )
            self._joint_observation_mode = str(self.get_parameter("joint_observation_mode").value)
            observation_clip = float(self.get_parameter("observation_clip").value)
            self._observation_clip = observation_clip if observation_clip > 0.0 else None
            self._action_scale = float(self.get_parameter("action_scale").value)
            self._action_output_mode = str(self.get_parameter("action_output_mode").value)
            self._action_delta_scale = float(self.get_parameter("action_delta_scale").value)
            self._prev_action_scale = float(self.get_parameter("prev_action_scale").value)
            self._hold_enabled = _as_bool(self.get_parameter("hold_enabled").value)
            self._hold_enter_tcp_m = float(self.get_parameter("hold_enter_tcp_m").value)
            self._hold_exit_tcp_m = float(self.get_parameter("hold_exit_tcp_m").value)
            self._hold_enter_qvel_l2 = float(self.get_parameter("hold_enter_qvel_l2").value)
            self._hold_enter_ticks = max(1, int(self.get_parameter("hold_enter_ticks").value))
            self._hold_active = False
            self._hold_enter_counter = 0
            self._hold_setpoint: Optional[list[float]] = None
            self._control_hz = hz
            self._eval_enabled = _as_bool(self.get_parameter("eval_enabled").value)
            self._eval_target_count = max(
                0,
                int(self.get_parameter("eval_target_count").value),
            )
            self._eval_reset_home_between_targets = _as_bool(
                self.get_parameter("eval_reset_home_between_targets").value
            )
            self._eval_home_joints = list(
                _parse_float_list_text(
                    str(self.get_parameter("eval_home_joints").value),
                    7,
                )
            )
            self._eval_home_tolerance_rad = max(
                0.0,
                float(self.get_parameter("eval_home_tolerance_rad").value),
            )
            self._eval_home_qvel_l2 = max(
                0.0,
                float(self.get_parameter("eval_home_qvel_l2").value),
            )
            self._eval_home_hold_ticks = max(
                1,
                int(self.get_parameter("eval_home_hold_ticks").value),
            )
            self._eval_max_steps_per_target = max(
                0,
                int(self.get_parameter("eval_max_steps_per_target").value),
            )
            self._eval_shutdown_on_complete = _as_bool(
                self.get_parameter("eval_shutdown_on_complete").value
            )
            self._eval_summary_csv_path = str(
                self.get_parameter("eval_summary_csv_path").value
            ).strip()
            self._eval_summary_file = None
            self._eval_summary_writer = None
            self._startup_home_enabled = _as_bool(
                self.get_parameter("startup_home_enabled").value
            )
            self._startup_home_joints = list(
                _parse_float_list_text(
                    str(self.get_parameter("startup_home_joints").value),
                    7,
                )
            )
            self._startup_home_tolerance_rad = max(
                0.0,
                float(self.get_parameter("startup_home_tolerance_rad").value),
            )
            self._startup_home_qvel_l2 = max(
                0.0,
                float(self.get_parameter("startup_home_qvel_l2").value),
            )
            self._startup_home_hold_ticks = max(
                1,
                int(self.get_parameter("startup_home_hold_ticks").value),
            )
            self._startup_home_hold_counter = 0
            self._startup_home_complete = not self._startup_home_enabled
            self._startup_home_wait_logged = False
            self._eval_phase = "target" if self._eval_enabled else "disabled"
            self._eval_target_index = 1 if self._eval_enabled else 0
            self._eval_completed_targets = 0
            self._eval_successful_targets = 0
            self._eval_failed_targets = 0
            self._eval_homing_hold_counter = 0
            self._eval_complete = False
            self._eval_last_reason = ""
            self._eval_completion_logged = False
            self._policy: Optional[ReachReduceShakePolicyRunner] = None
            if self._policy_enabled:
                model_path = str(self.get_parameter("onnx_model_path").value)
                try:
                    self._policy = ReachReduceShakePolicyRunner(model_path)
                except Exception as exc:
                    self.get_logger().error(f"Failed to load ONNX policy: {exc}")
                    raise

            self._open_telemetry_log()
            self._open_eval_summary_log()
            self._publish_target(self._session.reset())

            # Defensive fix for a marker-visibility race: even with TRANSIENT_LOCAL/RELIABLE
            # QoS on the publisher side, a late-joining RViz (or one whose Marker display QoS
            # override doesn't end up transient-local-compatible) can miss the single startup
            # publish. Since this node's target never changes after reset()
            # (resample_target_on_success=False), periodically republishing is a cheap,
            # policy-behavior-neutral way to guarantee any subscriber connecting later still
            # gets the marker within one period. Set target_republish_period_s<=0 to disable.
            republish_period_s = float(self.get_parameter("target_republish_period_s").value)
            if republish_period_s > 0.0:
                self.create_timer(republish_period_s, self._republish_target_periodic)

            self.get_logger().info(
                f"Reach-and-hold ready (robot_type={arm_id}, frame={frame_id}, "
                f"tcp_frame={self._tcp_frame_id}, command_topic={command_topic}, "
                f"hz={hz:.1f}, policy_enabled={self._policy_enabled}, "
                f"tcp_source={self._tcp_source}, target={tuple(round(float(x), 4) for x in self._session.episode_target_location_m)}, "
                f"resample_target_on_success={cfg.resample_target_on_success}, "
                f"smooth_actions={cfg.smooth_actions}, alpha={cfg.action_smoothing_alpha:.3f}, "
                f"limit_action_delta={cfg.limit_action_delta}, "
                f"action_rate_limit_hz={cfg.action_rate_limit_hz:.1f}, "
                f"action_notch_filter_enabled={self._action_notch_filter_enabled}, "
                f"action_notch_filter_stage={self._action_notch_filter_stage}, "
                f"hold_enabled={self._hold_enabled}, "
                f"hold_enter_tcp_m={self._hold_enter_tcp_m:.3f}, "
                f"hold_exit_tcp_m={self._hold_exit_tcp_m:.3f}, "
                f"eval_enabled={self._eval_enabled}, "
                f"eval_target_count={self._eval_target_count}, "
                f"eval_reset_home_between_targets={self._eval_reset_home_between_targets}, "
                f"startup_home_enabled={self._startup_home_enabled}, "
                f"telemetry_enabled={self._telemetry_enabled})"
            )

        def _on_tcp_pose(self, msg: PoseStamped) -> None:
            self._tcp_m = np.array(
                [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
                dtype=np.float64,
            )
            self._have_tcp = True

        def _on_tcp_transform(self, msg: TransformStamped) -> None:
            t = msg.transform.translation
            self._tcp_m = np.array([t.x, t.y, t.z], dtype=np.float64)
            self._have_tcp = True

        def _on_joint_state(self, msg: JointState) -> None:
            pos_map, vel_map = joint_state_to_maps(
                msg.name, msg.position, msg.velocity, msg.effort,
                strip_prefix=f"{self._arm_id}_",
            )
            if all(joint in pos_map for joint in PANDA_ARM_JOINTS):
                self._joint_pos = {k: pos_map[k] for k in PANDA_ARM_JOINTS}
                self._joint_vel = {k: vel_map.get(k, 0.0) for k in PANDA_ARM_JOINTS}
                new_pos_vector = [self._joint_pos[joint] for joint in PANDA_ARM_JOINTS]
                new_vel_vector = [self._joint_vel[joint] for joint in PANDA_ARM_JOINTS]

                if self._have_joints:
                    self._prev_joint_pos_observation_vector = list(self._joint_pos_vector)
                    self._prev_joint_vel_observation_vector = list(self._joint_vel_vector)
                else:
                    self._prev_joint_pos_observation_vector = list(new_pos_vector)
                    self._prev_joint_vel_observation_vector = list(new_vel_vector)

                self._joint_pos_vector = new_pos_vector

                # Real joint_states has no acceleration field, so approximate qacc via
                # backward finite difference of velocity. Default to 0 on the first
                # sample / on a non-positive dt (startup transient, duplicate stamps)
                # to avoid divide-by-zero spikes feeding a noisy accel estimate into
                # the policy.
                now_s = self.get_clock().now().nanoseconds * 1e-9
                if (
                    self._prev_joint_vel_vector is not None
                    and self._prev_joint_state_stamp_s is not None
                ):
                    dt = now_s - self._prev_joint_state_stamp_s
                    if dt > 1e-6:
                        self._joint_acc_vector = [
                            (new_vel_vector[i] - self._prev_joint_vel_vector[i]) / dt
                            for i in range(7)
                        ]
                    # else: keep the previous acceleration estimate rather than divide by ~0.
                else:
                    self._joint_acc_vector = [0.0] * 7

                self._prev_joint_vel_vector = list(new_vel_vector)
                self._prev_joint_state_stamp_s = now_s
                self._joint_vel_vector = new_vel_vector

                if not self._have_joints:
                    self._last_command_vector = list(self._joint_pos_vector)
                self._have_joints = True

        def _publish_target(self, target_m: np.ndarray) -> None:
            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self._frame_id
            msg.pose.position = Point(
                x=float(target_m[0]),
                y=float(target_m[1]),
                z=float(target_m[2]),
            )
            msg.pose.orientation.w = 1.0
            self._target_pub.publish(msg)
            self._publish_target_marker(msg)

        def _republish_target_periodic(self) -> None:
            self._publish_target(self._session.episode_target_location_m)

        def _publish_target_marker(self, target_pose: PoseStamped) -> None:
            marker = Marker()
            marker.header = target_pose.header
            # Namespace must match panda_track_reach.rviz's Marker display "Namespaces" filter
            # (panda_track_reach: true) so the shared rviz config actually shows this marker.
            marker.ns = "panda_track_reach"
            marker.id = 0
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose = target_pose.pose
            marker.scale.x = self._target_marker_scale_m
            marker.scale.y = self._target_marker_scale_m
            marker.scale.z = self._target_marker_scale_m
            marker.color.r = 0.02
            marker.color.g = 0.55
            marker.color.b = 1.0
            marker.color.a = 0.9
            marker.lifetime = Duration()
            self._target_marker_pub.publish(marker)

        def _update_tcp_from_tf(self) -> None:
            if self._tcp_source != "tf":
                return
            try:
                transform = self._tf_buffer.lookup_transform(
                    self._frame_id,
                    self._tcp_frame_id,
                    Time(),
                )
            except TransformException as exc:
                if not self._policy_error_reported:
                    self.get_logger().warning(
                        f"Waiting for TCP transform {self._frame_id} <- {self._tcp_frame_id}: {exc}"
                    )
                    self._policy_error_reported = True
                return

            t = transform.transform.translation
            self._tcp_m = np.array([t.x, t.y, t.z], dtype=np.float64)
            self._have_tcp = True
            self._policy_error_reported = False

        def _publish_joint_command(self, joint_positions: Sequence[float]) -> None:
            if self._command_message_type == "joint_state":
                msg = JointState()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.name = list(self._arm_joint_names)
                msg.position = [float(value) for value in joint_positions]
                self._command_pub.publish(msg)
                return

            point = JointTrajectoryPoint()
            point.positions = [float(value) for value in joint_positions]
            seconds = max(self._trajectory_duration_s, 1e-3)
            point.time_from_start = Duration(
                sec=int(seconds),
                nanosec=int((seconds % 1.0) * 1e9),
            )

            msg = JointTrajectory()
            # A zero stamp asks joint_trajectory_controller to execute immediately.
            # Using wall time here makes Gazebo/sim-time controllers defer forever.
            msg.joint_names = list(self._arm_joint_names)
            msg.points = [point]
            self._command_pub.publish(msg)

        def _reset_command_filters(self, seed_vector: Sequence[float]) -> None:
            seed = _clip_joint_positions(seed_vector, self._joint_limits)
            self._target_action_notch_filter.reset(seed)
            self._final_command_notch_filter.reset(seed)
            self._reset_near_goal_hold()
            self._session.previous_action_vector = list(seed)
            self._session.last_smoothed_action_vector = list(seed)
            self._last_command_vector = list(seed)

        def _home_metrics(self, home_joints: Sequence[float]) -> tuple[float, float]:
            if not self._have_joints:
                return float("inf"), float("inf")
            error = np.asarray(self._joint_pos_vector, dtype=np.float64) - np.asarray(
                home_joints, dtype=np.float64
            )
            qvel_l2 = float(np.linalg.norm(np.asarray(self._joint_vel_vector, dtype=np.float64)))
            return float(np.max(np.abs(error))), qvel_l2

        def _eval_home_metrics(self) -> tuple[float, float]:
            return self._home_metrics(self._eval_home_joints)

        def _compute_and_publish_home_command(
            self,
            home_joints: Sequence[float],
        ) -> list[float]:
            current = (
                list(self._joint_pos_vector)
                if self._have_joints
                else list(self._last_command_vector)
            )
            rate_hz = self._session.config.action_rate_limit_hz
            if rate_hz <= 0.0:
                rate_hz = self._control_hz
            rate_hz = max(rate_hz, 1.0)

            command: list[float] = []
            for i, home in enumerate(home_joints):
                previous = current[i] if i < len(current) else float(home)
                limit = (
                    self._session.config.action_velocity_limits_rad_s[i]
                    if i < len(self._session.config.action_velocity_limits_rad_s)
                    else 1.0
                )
                max_delta = max(float(limit), 0.05) / rate_hz
                command.append(
                    float(previous + np.clip(float(home) - previous, -max_delta, max_delta))
                )

            command = _clip_joint_positions(command, self._joint_limits)
            self._last_policy_telemetry = {
                "raw_action": [],
                "scaled_action": [],
                "target_action": list(home_joints),
                "notched_action": list(home_joints),
                "smoothed_action": command,
                "governed_action": command,
                "post_notch_command": command,
                "final_command": command,
            }
            self._last_command_vector = list(command)
            self._publish_joint_command(command)
            return command

        def _compute_and_publish_eval_home_command(self) -> list[float]:
            return self._compute_and_publish_home_command(self._eval_home_joints)

        def _on_startup_home_timer(self) -> None:
            if not self._have_joints:
                if not self._startup_home_wait_logged:
                    self.get_logger().info(
                        "Waiting for joint_states before publishing startup home command"
                    )
                    self._startup_home_wait_logged = True
                return

            self._compute_and_publish_home_command(self._startup_home_joints)
            home_error_rad, home_qvel_l2 = self._home_metrics(self._startup_home_joints)
            if (
                home_error_rad <= self._startup_home_tolerance_rad
                and home_qvel_l2 <= self._startup_home_qvel_l2
            ):
                self._startup_home_hold_counter += 1
            else:
                self._startup_home_hold_counter = 0

            if self._startup_home_hold_counter >= self._startup_home_hold_ticks:
                self._startup_home_complete = True
                self._reset_command_filters(self._startup_home_joints)
                self._publish_joint_command(self._last_command_vector)
                self._publish_target(self._session.episode_target_location_m)
                self.get_logger().info(
                    "Startup home reached; policy inference enabled "
                    f"(max_joint_error={home_error_rad:.4f} rad, "
                    f"qvel_l2={home_qvel_l2:.4f} rad/s)"
                )

        def _eval_target_limit_reached(self) -> bool:
            return (
                self._eval_target_count > 0
                and self._eval_completed_targets >= self._eval_target_count
            )

        def _start_next_eval_target_or_complete(self) -> None:
            if self._eval_target_limit_reached():
                self._eval_phase = "complete"
                self._eval_complete = True
                self._eval_target_index = self._eval_completed_targets
                self._reset_command_filters(self._eval_home_joints)
                if not self._eval_completion_logged:
                    self.get_logger().info(
                        "Eval complete: "
                        f"{self._eval_successful_targets}/"
                        f"{self._eval_completed_targets} targets succeeded"
                    )
                    self._eval_completion_logged = True
                if self._eval_shutdown_on_complete and rclpy.ok():
                    rclpy.shutdown()
                return

            self._eval_phase = "target"
            self._eval_target_index = self._eval_completed_targets + 1
            self._eval_homing_hold_counter = 0
            self._reset_command_filters(
                self._eval_home_joints
                if self._eval_reset_home_between_targets
                else self._joint_pos_vector
            )
            target = self._session.reset()
            self._publish_target(target)
            self.get_logger().info(
                f"Eval target {self._eval_target_index}/"
                f"{self._eval_target_count or 'unlimited'} -> "
                f"{tuple(round(float(x), 4) for x in target)}"
            )

        def _begin_eval_target_reset(self, outcome, reason: str) -> None:
            if not self._eval_enabled or self._eval_phase != "target":
                return

            self._eval_completed_targets += 1
            if reason == "success":
                self._eval_successful_targets += 1
            else:
                self._eval_failed_targets += 1
            self._eval_last_reason = reason
            self._reset_near_goal_hold()
            self._write_eval_summary_row(outcome, reason)

            self.get_logger().info(
                f"Eval target {self._eval_target_index}/"
                f"{self._eval_target_count or 'unlimited'} {reason} "
                f"at step {self._session.episode_step_index} "
                f"(dist={outcome.distance_m:.4f} m); "
                f"completed={self._eval_completed_targets}, "
                f"successes={self._eval_successful_targets}"
            )

            if self._eval_reset_home_between_targets:
                self._eval_phase = "homing"
                self._eval_homing_hold_counter = 0
                self._reset_command_filters(self._joint_pos_vector)
                return

            self._start_next_eval_target_or_complete()

        def _publish_eval_status_info(self) -> None:
            home_error_rad, home_qvel_l2 = self._eval_home_metrics()
            tcp_distance = (
                float(
                    np.linalg.norm(
                        np.asarray(self._tcp_m, dtype=np.float64)
                        - np.asarray(
                            self._session.episode_target_location_m,
                            dtype=np.float64,
                        )
                    )
                )
                if self._have_tcp
                else None
            )
            payload = {
                "eval_enabled": self._eval_enabled,
                "eval_phase": self._eval_phase,
                "eval_target_index": self._eval_target_index,
                "eval_target_count": self._eval_target_count,
                "eval_completed_targets": self._eval_completed_targets,
                "eval_successful_targets": self._eval_successful_targets,
                "eval_failed_targets": self._eval_failed_targets,
                "eval_complete": self._eval_complete,
                "eval_last_reason": self._eval_last_reason,
                "eval_home_max_error_rad": home_error_rad,
                "eval_home_qvel_l2": home_qvel_l2,
                "tcp_distance_m": tcp_distance,
                "target_m": tuple(
                    float(x) for x in self._session.episode_target_location_m
                ),
            }
            msg = String()
            msg.data = json.dumps(payload)
            self._info_pub.publish(msg)

        def _on_eval_homing_timer(self) -> None:
            self._compute_and_publish_eval_home_command()
            home_error_rad, home_qvel_l2 = self._eval_home_metrics()
            if (
                home_error_rad <= self._eval_home_tolerance_rad
                and home_qvel_l2 <= self._eval_home_qvel_l2
            ):
                self._eval_homing_hold_counter += 1
            else:
                self._eval_homing_hold_counter = 0

            self._publish_eval_status_info()

            if self._eval_homing_hold_counter >= self._eval_home_hold_ticks:
                self._reset_command_filters(self._eval_home_joints)
                self._publish_joint_command(self._last_command_vector)
                self.get_logger().info(
                    "Eval home reset reached "
                    f"(max_joint_error={home_error_rad:.4f} rad, "
                    f"qvel_l2={home_qvel_l2:.4f} rad/s)"
                )
                self._start_next_eval_target_or_complete()

        def _telemetry_fieldnames(self) -> list[str]:
            fields = [
                "wall_time_s",
                "ros_time_s",
                "control_dt_s",
                "telemetry_tick",
                "episode_step_index",
                "policy_enabled",
                "have_joints",
                "have_tcp",
                "eval_enabled",
                "eval_phase",
                "eval_target_index",
                "eval_target_count",
                "eval_completed_targets",
                "eval_successful_targets",
                "eval_failed_targets",
                "eval_complete",
                "eval_last_reason",
                "target_x_m",
                "target_y_m",
                "target_z_m",
                "tcp_x_m",
                "tcp_y_m",
                "tcp_z_m",
                "tcp_error_x_m",
                "tcp_error_y_m",
                "tcp_error_z_m",
                "tcp_distance_m",
                "reward",
                "success",
                "terminated",
                "truncated",
                "action_smoothing_alpha",
                "action_rate_limit_hz",
                "measured_velocity_governor_start_ratio",
                "action_notch_filter_enabled",
                "action_notch_filter_stage",
                "action_notch_filter_q",
                "action_notch_filter_secondary_q",
                "hold_enabled",
                "hold_active",
            ]
            for joint in PANDA_ARM_JOINTS:
                fields.extend(
                    [
                        f"action_velocity_limit_{joint}_rad_s",
                        f"action_notch_filter_frequency_{joint}_hz",
                        f"action_notch_filter_secondary_frequency_{joint}_hz",
                        f"qpos_{joint}_rad",
                        f"qvel_{joint}_rad_s",
                        f"raw_action_{joint}_rad",
                        f"scaled_action_{joint}_rad",
                        f"target_action_{joint}_rad",
                        f"notched_action_{joint}_rad",
                        f"smoothed_action_{joint}_rad",
                        f"governed_action_{joint}_rad",
                        f"post_notch_command_{joint}_rad",
                        f"final_command_{joint}_rad",
                    ]
                )
            return fields

        def _open_telemetry_log(self) -> None:
            if not self._telemetry_enabled:
                return

            path_text = self._telemetry_csv_path
            if not path_text:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                path_text = str(
                    Path.home()
                    / ".ros"
                    / f"panda_reach_reduce_shake_telemetry_{timestamp}.csv"
                )
                self._telemetry_csv_path = path_text

            path = Path(path_text).expanduser()
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                self._telemetry_file = path.open("w", newline="", encoding="utf-8")
                self._telemetry_writer = csv.DictWriter(
                    self._telemetry_file,
                    fieldnames=self._telemetry_fieldnames(),
                    extrasaction="ignore",
                )
                self._telemetry_writer.writeheader()
            except OSError as exc:
                self._telemetry_enabled = False
                self.get_logger().error(
                    f"Failed to open telemetry CSV {path}: {exc}. Telemetry disabled."
                )
                return

            self.get_logger().info(
                f"Telemetry CSV enabled: {path} "
                f"(every {self._telemetry_decimation} control tick(s))"
            )

        def _close_telemetry_log(self) -> None:
            if self._telemetry_file is None:
                return
            self._telemetry_file.flush()
            self._telemetry_file.close()
            self._telemetry_file = None
            self._telemetry_writer = None

        def _open_eval_summary_log(self) -> None:
            if not self._eval_enabled or not self._eval_summary_csv_path:
                return

            path = Path(self._eval_summary_csv_path).expanduser()
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                self._eval_summary_file = path.open("w", newline="", encoding="utf-8")
                self._eval_summary_writer = csv.DictWriter(
                    self._eval_summary_file,
                    fieldnames=[
                        "wall_time_s",
                        "ros_time_s",
                        "eval_target_index",
                        "eval_target_count",
                        "reason",
                        "success",
                        "step_count",
                        "distance_m",
                        "target_x_m",
                        "target_y_m",
                        "target_z_m",
                        "completed_targets",
                        "successful_targets",
                        "failed_targets",
                    ],
                )
                self._eval_summary_writer.writeheader()
                self._eval_summary_file.flush()
            except OSError as exc:
                self._eval_summary_file = None
                self._eval_summary_writer = None
                self.get_logger().error(
                    f"Failed to open eval summary CSV {path}: {exc}. "
                    "Eval summary disabled."
                )
                return

            self.get_logger().info(f"Eval summary CSV enabled: {path}")

        def _close_eval_summary_log(self) -> None:
            if self._eval_summary_file is None:
                return
            self._eval_summary_file.flush()
            self._eval_summary_file.close()
            self._eval_summary_file = None
            self._eval_summary_writer = None

        def _write_eval_summary_row(self, outcome, reason: str) -> None:
            if self._eval_summary_writer is None:
                return
            target = np.asarray(self._session.episode_target_location_m, dtype=np.float64)
            self._eval_summary_writer.writerow(
                {
                    "wall_time_s": time.time(),
                    "ros_time_s": self.get_clock().now().nanoseconds * 1e-9,
                    "eval_target_index": self._eval_target_index,
                    "eval_target_count": self._eval_target_count,
                    "reason": reason,
                    "success": reason == "success",
                    "step_count": self._session.episode_step_index,
                    "distance_m": outcome.distance_m,
                    "target_x_m": float(target[0]),
                    "target_y_m": float(target[1]),
                    "target_z_m": float(target[2]),
                    "completed_targets": self._eval_completed_targets,
                    "successful_targets": self._eval_successful_targets,
                    "failed_targets": self._eval_failed_targets,
                }
            )
            self._eval_summary_file.flush()

        def _preview_smoothed_actions(self, target_actions: Sequence[float]) -> list[float]:
            cfg = self._session.config
            n = len(cfg.joint_names)
            if len(target_actions) != n:
                raise ValueError(f"expected {n} actions, got {len(target_actions)}")

            had_history = len(self._session.last_smoothed_action_vector) == n
            alpha = float(np.clip(cfg.action_smoothing_alpha, 0.01, 1.0))
            smoothed: list[float] = []
            for i, raw in enumerate(target_actions):
                previous = self._session.last_smoothed_action_vector[i] if had_history else raw
                if not had_history and i < len(self._joint_pos_vector):
                    previous = self._joint_pos_vector[i]
                value = (
                    previous + alpha * (raw - previous)
                    if cfg.smooth_actions and alpha < 1.0
                    else raw
                )
                smoothed.append(float(value))
            return smoothed

        def _write_telemetry_row(
            self,
            outcome,
            actions_applied: Sequence[float],
            control_dt_s: Optional[float],
        ) -> None:
            if not self._telemetry_enabled or self._telemetry_writer is None:
                return

            self._telemetry_tick_index += 1
            if (self._telemetry_tick_index - 1) % self._telemetry_decimation != 0:
                return

            ros_time_s = self.get_clock().now().nanoseconds * 1e-9
            target = np.asarray(self._session.episode_target_location_m, dtype=np.float64)
            tcp = np.asarray(self._tcp_m, dtype=np.float64)
            tcp_error = tcp - target
            cfg = self._session.config
            row: dict[str, object] = {
                "wall_time_s": time.time(),
                "ros_time_s": ros_time_s,
                "control_dt_s": control_dt_s,
                "telemetry_tick": self._telemetry_tick_index,
                "episode_step_index": self._session.episode_step_index - 1,
                "policy_enabled": self._policy_enabled,
                "have_joints": self._have_joints,
                "have_tcp": self._have_tcp,
                "eval_enabled": self._eval_enabled,
                "eval_phase": self._eval_phase,
                "eval_target_index": self._eval_target_index,
                "eval_target_count": self._eval_target_count,
                "eval_completed_targets": self._eval_completed_targets,
                "eval_successful_targets": self._eval_successful_targets,
                "eval_failed_targets": self._eval_failed_targets,
                "eval_complete": self._eval_complete,
                "eval_last_reason": self._eval_last_reason,
                "target_x_m": float(target[0]),
                "target_y_m": float(target[1]),
                "target_z_m": float(target[2]),
                "tcp_x_m": float(tcp[0]),
                "tcp_y_m": float(tcp[1]),
                "tcp_z_m": float(tcp[2]),
                "tcp_error_x_m": float(tcp_error[0]),
                "tcp_error_y_m": float(tcp_error[1]),
                "tcp_error_z_m": float(tcp_error[2]),
                "tcp_distance_m": outcome.distance_m,
                "reward": outcome.reward,
                "success": outcome.success,
                "terminated": outcome.terminated,
                "truncated": outcome.truncated,
                "action_smoothing_alpha": cfg.action_smoothing_alpha,
                "action_rate_limit_hz": cfg.action_rate_limit_hz,
                "measured_velocity_governor_start_ratio": (
                    cfg.measured_velocity_governor_start_ratio
                ),
                "action_notch_filter_enabled": self._action_notch_filter_enabled,
                "action_notch_filter_stage": self._action_notch_filter_stage,
                "action_notch_filter_q": self._action_notch_filter_q,
                "action_notch_filter_secondary_q": (
                    self._action_notch_filter_secondary_q
                ),
                "hold_enabled": self._hold_enabled,
                "hold_active": 1 if self._hold_active else 0,
            }

            telemetry = self._last_policy_telemetry
            governed_actions = telemetry.get("governed_action", [])
            post_notch_command = telemetry.get("post_notch_command", governed_actions)
            final_command = (
                list(actions_applied)
                if len(actions_applied) == 7
                else telemetry.get("final_command", [])
            )
            for i, joint in enumerate(PANDA_ARM_JOINTS):
                row[f"action_velocity_limit_{joint}_rad_s"] = (
                    cfg.action_velocity_limits_rad_s[i]
                    if i < len(cfg.action_velocity_limits_rad_s)
                    else None
                )
                row[f"action_notch_filter_frequency_{joint}_hz"] = (
                    self._action_notch_filter_frequencies_hz[i]
                    if i < len(self._action_notch_filter_frequencies_hz)
                    else None
                )
                row[f"action_notch_filter_secondary_frequency_{joint}_hz"] = (
                    self._action_notch_filter_secondary_frequencies_hz[i]
                    if i < len(self._action_notch_filter_secondary_frequencies_hz)
                    else None
                )
                row[f"qpos_{joint}_rad"] = (
                    self._joint_pos_vector[i] if self._have_joints else None
                )
                row[f"qvel_{joint}_rad_s"] = (
                    self._joint_vel_vector[i] if self._have_joints else None
                )
                for key, column_prefix in (
                    ("raw_action", "raw_action"),
                    ("scaled_action", "scaled_action"),
                    ("target_action", "target_action"),
                    ("notched_action", "notched_action"),
                    ("smoothed_action", "smoothed_action"),
                ):
                    values = telemetry.get(key, [])
                    row[f"{column_prefix}_{joint}_rad"] = (
                        values[i] if i < len(values) else None
                    )
                row[f"governed_action_{joint}_rad"] = (
                    governed_actions[i] if i < len(governed_actions) else None
                )
                row[f"post_notch_command_{joint}_rad"] = (
                    post_notch_command[i] if i < len(post_notch_command) else None
                )
                row[f"final_command_{joint}_rad"] = (
                    final_command[i] if i < len(final_command) else None
                )

            self._telemetry_writer.writerow(row)
            self._telemetry_rows_written += 1
            if self._telemetry_rows_written % self._telemetry_flush_every_n_rows == 0:
                self._telemetry_file.flush()

        def _compute_and_publish_policy_action(self) -> list[float]:
            self._last_policy_telemetry = {}
            if not self._policy_enabled or self._policy is None:
                return []
            if not self._have_joints:
                return []

            raw_actions, scaled_actions = self._policy.compute_action_with_raw_output(
                self._joint_pos_vector,
                self._joint_vel_vector,
                self._last_command_vector,
                self._tcp_m,
                self._session.episode_target_location_m,
                tcp_observation_mode=self._tcp_observation_mode,
                tcp_observation_scale=self._tcp_observation_scale,
                tcp_observation_signs=self._tcp_observation_signs,
                tcp_observation_unit_scale=self._tcp_observation_unit_scale,
                joint_observation_mode=self._joint_observation_mode,
                prev_joint_pos=self._prev_joint_pos_observation_vector,
                prev_joint_vel=self._prev_joint_vel_observation_vector,
                joint_acc=self._joint_acc_vector,
                observation_clip=self._observation_clip,
                action_scale=self._action_scale,
                prev_action_scale=self._prev_action_scale,
            )
            target_actions = _interpret_policy_actions(
                scaled_actions,
                self._joint_pos_vector,
                self._last_command_vector,
                self._action_output_mode,
                self._action_delta_scale,
                self._joint_limits,
            )
            notched_actions = (
                self._target_action_notch_filter.filter(target_actions)
                if self._action_notch_filter_enabled
                and self._action_notch_filter_stage in ("target_action", "both")
                else list(target_actions)
            )
            notched_actions = _clip_joint_positions(notched_actions, self._joint_limits)
            smoothed_actions = self._preview_smoothed_actions(notched_actions)
            governed_actions = self._session.smooth_actions(
                notched_actions,
                measured_joint_pos=self._joint_pos_vector,
                measured_joint_vel=self._joint_vel_vector,
            )
            post_notch_command = (
                self._final_command_notch_filter.filter(governed_actions)
                if self._action_notch_filter_enabled
                and self._action_notch_filter_stage in ("final_command", "both")
                else list(governed_actions)
            )
            command = _clip_joint_positions(post_notch_command, self._joint_limits)
            published = self._apply_near_goal_hold(command)
            self._last_policy_telemetry = {
                "raw_action": raw_actions,
                "scaled_action": scaled_actions,
                "target_action": target_actions,
                "notched_action": notched_actions,
                "smoothed_action": smoothed_actions,
                "governed_action": governed_actions,
                "post_notch_command": post_notch_command,
                "final_command": published,
            }
            self._last_command_vector = published
            if self._hold_active:
                self._session.last_smoothed_action_vector = list(published)
            self._publish_joint_command(published)
            return published

        def _reset_near_goal_hold(self) -> None:
            self._hold_active = False
            self._hold_enter_counter = 0
            self._hold_setpoint = None

        def _apply_near_goal_hold(self, command: Sequence[float]) -> list[float]:
            command = list(command)
            if not self._hold_enabled:
                self._hold_active = False
                return command

            tcp_distance = float(
                np.linalg.norm(
                    np.asarray(self._tcp_m, dtype=np.float64)
                    - np.asarray(self._session.episode_target_location_m, dtype=np.float64)
                )
            )
            qvel_l2 = float(np.linalg.norm(np.asarray(self._joint_vel_vector, dtype=np.float64)))

            if self._hold_active:
                if tcp_distance > self._hold_exit_tcp_m:
                    self._reset_near_goal_hold()
                    return command
                if self._hold_setpoint is not None:
                    return list(self._hold_setpoint)
                return command

            if tcp_distance < self._hold_enter_tcp_m and qvel_l2 < self._hold_enter_qvel_l2:
                self._hold_enter_counter += 1
            else:
                self._hold_enter_counter = 0

            if self._hold_enter_counter >= self._hold_enter_ticks:
                self._hold_active = True
                self._hold_setpoint = list(command)
                return list(self._hold_setpoint)
            return command

        def _on_timer(self) -> None:
            now_s = self.get_clock().now().nanoseconds * 1e-9
            control_dt_s = (
                now_s - self._last_timer_stamp_s
                if self._last_timer_stamp_s is not None
                else None
            )
            self._last_timer_stamp_s = now_s

            if not self._startup_home_complete:
                self._on_startup_home_timer()
                return

            self._update_tcp_from_tf()
            if not self._have_tcp:
                return

            if self._eval_enabled and self._eval_phase == "complete":
                if self._eval_reset_home_between_targets:
                    self._compute_and_publish_eval_home_command()
                self._publish_eval_status_info()
                return

            if self._eval_enabled and self._eval_phase == "homing":
                self._on_eval_homing_timer()
                return

            actions_applied = self._compute_and_publish_policy_action()
            outcome = self._session.step(
                tuple(self._tcp_m),
                joint_vel_rad_s=self._joint_vel,
                actions_applied=actions_applied,
                joint_acc_rad_s2=dict(zip(PANDA_ARM_JOINTS, self._joint_acc_vector)),
            )
            self._write_telemetry_row(outcome, actions_applied, control_dt_s)

            if outcome.success:
                self._publish_target(self._session.episode_target_location_m)

            info_payload = {
                **outcome.info,
                "terminated": outcome.terminated,
                "truncated": outcome.truncated,
                "reward": outcome.reward,
                "tcp_observation_mode": self._tcp_observation_mode,
                "tcp_observation_scale": self._tcp_observation_scale,
                "tcp_observation_signs": self._tcp_observation_signs,
                "tcp_observation_unit_scale": self._tcp_observation_unit_scale,
                "joint_observation_mode": self._joint_observation_mode,
                "observation_clip": self._observation_clip,
                "action_output_mode": self._action_output_mode,
                "action_delta_scale": self._action_delta_scale,
                "prev_action_scale": self._prev_action_scale,
                "smooth_actions": self._session.config.smooth_actions,
                "action_smoothing_alpha": self._session.config.action_smoothing_alpha,
                "limit_action_delta": self._session.config.limit_action_delta,
                "action_rate_limit_hz": self._session.config.action_rate_limit_hz,
                "action_velocity_limits_rad_s": self._session.config.action_velocity_limits_rad_s,
                "use_measured_velocity_governor": (
                    self._session.config.use_measured_velocity_governor
                ),
                "measured_velocity_governor_start_ratio": (
                    self._session.config.measured_velocity_governor_start_ratio
                ),
                "action_notch_filter_enabled": self._action_notch_filter_enabled,
                "action_notch_filter_stage": self._action_notch_filter_stage,
                "action_notch_filter_frequencies_hz": (
                    self._action_notch_filter_frequencies_hz
                ),
                "action_notch_filter_q": self._action_notch_filter_q,
                "action_notch_filter_secondary_frequencies_hz": (
                    self._action_notch_filter_secondary_frequencies_hz
                ),
                "action_notch_filter_secondary_q": (
                    self._action_notch_filter_secondary_q
                ),
                "hold_enabled": self._hold_enabled,
                "hold_active": self._hold_active,
                "hold_enter_tcp_m": self._hold_enter_tcp_m,
                "hold_exit_tcp_m": self._hold_exit_tcp_m,
                "hold_enter_qvel_l2": self._hold_enter_qvel_l2,
                "hold_enter_ticks": self._hold_enter_ticks,
                "eval_enabled": self._eval_enabled,
                "eval_phase": self._eval_phase,
                "eval_target_index": self._eval_target_index,
                "eval_target_count": self._eval_target_count,
                "eval_completed_targets": self._eval_completed_targets,
                "eval_successful_targets": self._eval_successful_targets,
                "eval_failed_targets": self._eval_failed_targets,
                "eval_complete": self._eval_complete,
                "eval_last_reason": self._eval_last_reason,
            }
            info_msg = String()
            info_msg.data = json.dumps(info_payload)
            self._info_pub.publish(info_msg)

            if self._eval_enabled:
                eval_reason = ""
                if outcome.success:
                    eval_reason = "success"
                elif outcome.terminated:
                    eval_reason = "terminated"
                elif outcome.truncated:
                    eval_reason = "truncated"
                elif (
                    self._eval_max_steps_per_target > 0
                    and self._session.episode_step_index
                    >= self._eval_max_steps_per_target
                ):
                    eval_reason = "max_steps"

                if eval_reason:
                    self._begin_eval_target_reset(outcome, eval_reason)
                return

            if outcome.terminated or outcome.truncated:
                reason = "terminated" if outcome.terminated else "truncated"
                self.get_logger().warning(
                    f"Episode {reason} at step {self._session.episode_step_index - 1} "
                    f"(dist={outcome.distance_m:.4f} m)"
                )
                if self._auto_reset:
                    self._target_action_notch_filter.reset(self._joint_pos_vector)
                    self._final_command_notch_filter.reset(self._joint_pos_vector)
                    self._reset_near_goal_hold()
                    self._publish_target(self._session.reset())
                    self.get_logger().info("Episode reset -> new subgoal published")

    rclpy.init(args=[sys.argv[0], *getattr(args, "ros_args", [])])
    node = PandaReachReduceShakeNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node._close_telemetry_log()
        node._close_eval_summary_log()
        node.destroy_node()
        # On Ctrl-C the signal handler may already have shut the context down;
        # guard to avoid a noisy "rcl_shutdown already called" traceback.
        if rclpy.ok():
            rclpy.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ros",
        action="store_true",
        help="Run as a ROS 2 node (requires rclpy and sourced ROS environment).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate a short episode without ROS.",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for --dry-run.")
    parser.add_argument("--steps", type=int, default=40, help="Max steps for --dry-run.")
    parser.add_argument(
        "--policy",
        action="store_true",
        help="When running with --ros, load the ONNX policy and publish Panda joint commands.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Path to the ONNX policy. Defaults to ppo_final_reduce_shake.onnx in franka_emika_panda.",
    )
    parser.add_argument(
        "--telemetry-enabled",
        action="store_true",
        help="Enable CSV telemetry when running with --ros.",
    )
    parser.add_argument(
        "--telemetry-csv-path",
        default="",
        help="CSV path for ROS telemetry. Supplying a path enables telemetry by default.",
    )
    parser.add_argument(
        "--telemetry-decimation",
        type=int,
        default=1,
        help="Write one telemetry row every N control ticks when telemetry is enabled.",
    )
    parser.add_argument(
        "--telemetry-flush-every-n-rows",
        type=int,
        default=60,
        help="Flush telemetry CSV after this many written rows.",
    )
    parser.add_argument(
        "--dump-config",
        action="store_true",
        help="Print default ReachConstrainedConfig as JSON and exit.",
    )
    args, ros_args = parser.parse_known_args()
    args.ros_args = ros_args

    if args.dump_config:
        print(json.dumps(asdict(ReachConstrainedConfig()), indent=2))
        return

    if args.ros:
        _run_ros_node(args)
        return

    if args.dry_run or len(sys.argv) == 1:
        _dry_run(seed=args.seed, steps=args.steps)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
