#!/usr/bin/env python3
"""
Panda single-target "reach and hold" behaviour for the ppo_final_constrained checkpoint.

Unlike panda_track_reach_ros2.py (multi-target "track": resamples a brand-new random
target every time the arm succeeds), this node reaches ONE target sampled at episode
start and just holds station there -- no resampling on success. Observation
preprocessing matches ppo_final_constrained's training exactly:

  - joint1..joint7: [qpos/pi, qvel/2.0, qacc/10.0], each dim clipped to +/-3.
    qacc has no real sensor on ROS joint_states, so it is estimated via backward
    finite difference of velocity (see PandaReachConstrainedNode._on_joint_state).
  - tcp_pos: (tcp_m - target_m) / 0.75, clipped to +/-3.
  - Actions are absolute joint position targets in radians (no delta/normalized output).

This module intentionally reuses the ROS-free plumbing and constants from
panda_track_reach_ros2.py (PANDA_ARM_JOINTS, joint limits, vec3 parsing, joint-state
mapping, action-output interpretation, etc.) via subclassing/composition rather than
duplicating them, so panda_track_reach_ros2.py's legacy ppo_track_franka.onnx behaviour
stays completely untouched.

Run as a ROS 2 node when rclpy is available:

    python panda_reach_constrained_ros2.py --ros --policy

Dry-run without ROS (prints a short simulated episode):

    python panda_reach_constrained_ros2.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

# Running this file directly (python panda_reach_constrained_ros2.py, or via `ros2 run`
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

# Normalization constants for joint_observation_mode="normalized_position_velocity_acceleration":
# obs = [qpos/pi, qvel/2.0, qacc/10.0], matching the trained Box(-3, 3) observation space.
JOINT_OBS_POS_SCALE: float = math.pi
JOINT_OBS_VEL_SCALE: float = 2.0
JOINT_OBS_ACC_SCALE: float = 10.0


def _sample_target_offset_reach_hold(
    rng: np.random.Generator,
    offset_max_m: tuple[float, float, float],
    symmetric_z_offset: bool,
) -> np.ndarray:
    """Like panda_track_reach_ros2.sample_target_offset, but Z is also drawn
    U[-max, +max] when symmetric_z_offset is set. ppo_final_constrained was trained
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
    """PandaTrackReachConfig specialized for the ppo_final_constrained "reach and
    hold" checkpoint: training center/radius, symmetric offset sampling on all three
    axes, no target resampling on success, and no action smoothing/rate-limiting
    (raw policy, matching training -- re-enable for real hardware if desired)."""

    target_location_m: tuple[float, float, float] = (0.32, 0.0, 0.5)
    target_random_offset_max_m: tuple[float, float, float] = (0.06, 0.06, 0.06)
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


class ReachConstrainedPolicyRunner(OnnxPolicyRunner):
    """OnnxPolicyRunner extended with the ppo_final_constrained observation scheme:
    normalized [qpos/pi, qvel/2, qacc/10] joints + tcp_error_normalized tcp, both
    optionally clipped to +/-observation_clip (matches the trained Box(-3, 3) space).
    Reuses the parent's __init__ (ONNX session load + I/O validation) unchanged.
    """

    @staticmethod
    def _joint_observation_normalized(
        position: float,
        velocity: float,
        acceleration: float,
        clip: Optional[float] = None,
    ) -> np.ndarray:
        obs = np.array(
            [
                position / JOINT_OBS_POS_SCALE,
                velocity / JOINT_OBS_VEL_SCALE,
                acceleration / JOINT_OBS_ACC_SCALE,
            ],
            dtype=np.float32,
        )
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
        joint_observation_mode: str = "normalized_position_velocity_acceleration",
        joint_acc: Optional[Sequence[float]] = None,
        observation_clip: Optional[float] = 3.0,
        action_scale: float = 1.0,
    ) -> list[float]:
        if len(joint_pos) != 7 or len(joint_vel) != 7 or len(last_command) != 7:
            raise ValueError("joint_pos, joint_vel, and last_command must all have length 7")
        if joint_acc is not None and len(joint_acc) != 7:
            raise ValueError("joint_acc must have length 7 when provided")

        feed: dict[str, np.ndarray] = {}
        for i, joint_name in enumerate(PANDA_ARM_JOINTS):
            if joint_observation_mode == "normalized_position_velocity_acceleration":
                acceleration = float(joint_acc[i]) if joint_acc is not None else 0.0
                obs = self._joint_observation_normalized(
                    float(joint_pos[i]),
                    float(joint_vel[i]),
                    acceleration,
                    clip=observation_clip,
                )
            else:
                obs = self._clip_obs(
                    OnnxPolicyRunner._joint_observation(
                        joint_observation_mode,
                        float(joint_pos[i]),
                        float(joint_vel[i]),
                        float(last_command[i]),
                    ),
                    observation_clip,
                )
            feed[joint_name] = obs.reshape(1, 3)

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

        raw_outputs = self._session.run([f"actuator{i}" for i in range(1, 8)], feed)
        actions = [float(np.asarray(output).reshape(-1)[0]) * action_scale for output in raw_outputs]
        return actions


def _default_onnx_model_path() -> str:
    try:
        from ament_index_python.packages import get_package_share_directory

        return str(Path(get_package_share_directory("franka_emika_panda")) / "ppo_final_constrained.onnx")
    except Exception:
        return str(Path(__file__).resolve().parent / "ppo_final_constrained.onnx")


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
        from geometry_msgs.msg import Point, PoseStamped
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

    class PandaReachConstrainedNode(Node):
        def __init__(self) -> None:
            super().__init__("panda_reach_constrained")
            self.declare_parameter("robot_type", "panda")
            self.declare_parameter("frame_id", "")
            self.declare_parameter("tcp_source", "tf")
            self.declare_parameter("tcp_frame_id", "")
            self.declare_parameter("tcp_pose_topic", "ee_pose")
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
                "joint_observation_mode", "normalized_position_velocity_acceleration"
            )
            self.declare_parameter("observation_clip", 3.0)
            self.declare_parameter("action_scale", 1.0)
            self.declare_parameter("action_output_mode", "absolute")
            self.declare_parameter("action_delta_scale", 0.05)

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

            arm_id = str(self.get_parameter("robot_type").value) or "panda"
            self._arm_id = arm_id
            self._arm_joint_names = tuple(f"{arm_id}_joint{i}" for i in range(1, 8))
            self._joint_limits = ROBOT_JOINT_LIMITS.get(arm_id, PANDA_JOINT_LIMITS)

            frame_id = str(self.get_parameter("frame_id").value) or f"{arm_id}_link0"
            self._frame_id = frame_id
            self._tcp_source = str(self.get_parameter("tcp_source").value)
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
                resample_target_on_success=bool(
                    self.get_parameter("resample_target_on_success").value
                ),
                symmetric_z_offset=bool(self.get_parameter("symmetric_z_offset").value),
                smooth_actions=bool(self.get_parameter("smooth_actions").value),
                limit_action_delta=bool(self.get_parameter("limit_action_delta").value),
            )
            self._session = ReachConstrainedSession(cfg)
            self._session.seed(cfg.random_seed)
            self._have_tcp = False
            self._have_joints = False
            self._tcp_m = np.zeros(3, dtype=np.float64)
            self._joint_pos_vector = list(PANDA_HOME_JOINTS)
            self._joint_vel_vector = [0.0] * 7
            self._joint_acc_vector = [0.0] * 7
            self._prev_joint_vel_vector: Optional[list[float]] = None
            self._prev_joint_state_stamp_s: Optional[float] = None
            self._last_command_vector = list(PANDA_HOME_JOINTS)
            self._joint_pos: dict[str, float] = {}
            self._joint_vel: dict[str, float] = {}
            self._policy_error_reported = False
            self._target_marker_scale_m = float(self.get_parameter("target_marker_scale_m").value)

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
            self._command_pub = self.create_publisher(
                JointTrajectory,
                command_topic,
                10,
            )

            tcp_topic = self.get_parameter("tcp_pose_topic").value
            joint_topic = self.get_parameter("joint_states_topic").value
            if self._tcp_source == "pose_topic":
                self.create_subscription(PoseStamped, tcp_topic, self._on_tcp_pose, 10)
            self.create_subscription(JointState, joint_topic, self._on_joint_state, 10)
            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)

            hz = float(self.get_parameter("control_hz").value)
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
            self._policy: Optional[ReachConstrainedPolicyRunner] = None
            if self._policy_enabled:
                model_path = str(self.get_parameter("onnx_model_path").value)
                try:
                    self._policy = ReachConstrainedPolicyRunner(model_path)
                except Exception as exc:
                    self.get_logger().error(f"Failed to load ONNX policy: {exc}")
                    raise

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
                f"smooth_actions={cfg.smooth_actions}, limit_action_delta={cfg.limit_action_delta})"
            )

        def _on_tcp_pose(self, msg: PoseStamped) -> None:
            self._tcp_m = np.array(
                [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
                dtype=np.float64,
            )
            self._have_tcp = True

        def _on_joint_state(self, msg: JointState) -> None:
            pos_map, vel_map = joint_state_to_maps(
                msg.name, msg.position, msg.velocity, msg.effort,
                strip_prefix=f"{self._arm_id}_",
            )
            if all(joint in pos_map for joint in PANDA_ARM_JOINTS):
                self._joint_pos = {k: pos_map[k] for k in PANDA_ARM_JOINTS}
                self._joint_vel = {k: vel_map.get(k, 0.0) for k in PANDA_ARM_JOINTS}
                self._joint_pos_vector = [self._joint_pos[joint] for joint in PANDA_ARM_JOINTS]
                new_vel_vector = [self._joint_vel[joint] for joint in PANDA_ARM_JOINTS]

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

        def _compute_and_publish_policy_action(self) -> list[float]:
            if not self._policy_enabled or self._policy is None:
                return []
            if not self._have_joints:
                return []

            raw_actions = self._policy.compute_action(
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
                joint_acc=self._joint_acc_vector,
                observation_clip=self._observation_clip,
                action_scale=self._action_scale,
            )
            target_actions = _interpret_policy_actions(
                raw_actions,
                self._joint_pos_vector,
                self._last_command_vector,
                self._action_output_mode,
                self._action_delta_scale,
                self._joint_limits,
            )
            smoothed_actions = self._session.smooth_actions(
                target_actions,
                measured_joint_pos=self._joint_pos_vector,
                measured_joint_vel=self._joint_vel_vector,
            )
            command = _clip_joint_positions(smoothed_actions, self._joint_limits)
            self._last_command_vector = command
            self._publish_joint_command(command)
            return command

        def _on_timer(self) -> None:
            self._update_tcp_from_tf()
            if not self._have_tcp:
                return

            actions_applied = self._compute_and_publish_policy_action()
            outcome = self._session.step(
                tuple(self._tcp_m),
                joint_vel_rad_s=self._joint_vel,
                actions_applied=actions_applied,
                joint_acc_rad_s2=dict(zip(PANDA_ARM_JOINTS, self._joint_acc_vector)),
            )

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
            }
            info_msg = String()
            info_msg.data = json.dumps(info_payload)
            self._info_pub.publish(info_msg)

            if outcome.terminated or outcome.truncated:
                reason = "terminated" if outcome.terminated else "truncated"
                self.get_logger().warning(
                    f"Episode {reason} at step {self._session.episode_step_index - 1} "
                    f"(dist={outcome.distance_m:.4f} m)"
                )
                if self._auto_reset:
                    self._publish_target(self._session.reset())
                    self.get_logger().info("Episode reset -> new subgoal published")

    rclpy.init(args=[sys.argv[0], *getattr(args, "ros_args", [])])
    node = PandaReachConstrainedNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
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
        help="Path to the ONNX policy. Defaults to ppo_final_constrained.onnx in franka_emika_panda.",
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
