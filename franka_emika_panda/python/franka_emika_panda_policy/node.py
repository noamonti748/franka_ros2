"""Thin ROS 2 Humble executable for the promoted student-v6 controller.

Run with ``python -m franka_emika_panda_policy.node`` after adding this
package's ``python`` directory to PYTHONPATH. All ROS/KDL imports are lazy.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from .config import (
    PHASE_NAMES,
    Phase,
    default_model_dir,
    load_runtime_config,
    resolve_artifact_paths,
)
from .dls import (
    DLSController,
    lock_wrist_to_home,
    normalize_dls_descend_target,
    rewrite_cube_geometry_to_world_anchor,
    update_hover_cube_anchor,
)
from .gripper import (
    FrankaGraspBackend,
    GripperBackend,
    NoopGripperBackend,
    SimGripperCommandBackend,
)
from .integration import IntegratedActionController
from .kinematics import PandaAnalyticJacobian, panda_hand_tcp_pose
from .observation import (
    CameraCalibration,
    OakPolicyCameraPreprocessor,
    ObservationBuilder,
    image_to_policy_pixels,
)
from .onnx_policy import OnnxPolicy, sha256_file
from .safety import (
    CommandLimiter,
    ConsecutiveReadiness,
    HomeReadiness,
    RecoveryToHome,
    may_publish_fault_hold,
    resolve_string_alias,
)
from .session import AutonomousSession, ObservableSession
from .synchronization import SnapshotAttempt, SynchronizedObservationBuffer
from .telemetry import build_command_telemetry


def quaternion_to_rotation(x: float, y: float, z: float, w: float) -> np.ndarray:
    quaternion = np.asarray((x, y, z, w), dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1.0e-12:
        raise ValueError("invalid TCP quaternion")
    x, y, z, w = quaternion / norm
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def extract_joint_measurements(
    names: Sequence[str],
    positions: Sequence[float],
    velocities: Sequence[float],
    arm_joint_names: tuple[str, ...],
    finger_joint_names: tuple[str, ...],
) -> tuple[np.ndarray | None, np.ndarray | None, float | None]:
    """Extract arm and optional finger state without coupling their availability."""

    position = {
        name: float(positions[index])
        for index, name in enumerate(names)
        if index < len(positions)
    }
    velocity = {
        name: float(velocities[index])
        for index, name in enumerate(names)
        if index < len(velocities)
    }
    q = qvel = None
    if all(name in position for name in arm_joint_names):
        q = np.asarray([position[name] for name in arm_joint_names])
        qvel = np.asarray([velocity.get(name, 0.0) for name in arm_joint_names])
    finger_span = (
        float(sum(position[name] for name in finger_joint_names))
        if all(name in position for name in finger_joint_names)
        else None
    )
    return q, qvel, finger_span


class PyKDLJacobian:
    """Lazy 6x7 base-frame Jacobian provider built from robot_description."""

    def __init__(self, urdf_xml: str, base_frame: str, tcp_frame: str) -> None:
        try:
            import PyKDL
            from kdl_parser_py.urdf import treeFromString
        except ImportError as exc:
            raise RuntimeError("PyKDL and kdl_parser_py are unavailable") from exc
        success, tree = treeFromString(urdf_xml)
        if not success:
            raise RuntimeError("failed to parse robot_description for PyKDL")
        chain = tree.getChain(base_frame, tcp_frame)
        if chain.getNrOfJoints() != 7:
            raise RuntimeError(
                f"KDL chain {base_frame}->{tcp_frame} has "
                f"{chain.getNrOfJoints()} joints, expected 7"
            )
        self._kdl = PyKDL
        self._chain = chain
        self._solver = PyKDL.ChainJntToJacSolver(chain)

    def __call__(self, q: np.ndarray) -> np.ndarray:
        q_array = np.asarray(q, dtype=np.float64)
        if q_array.shape != (7,) or not np.all(np.isfinite(q_array)):
            raise ValueError("KDL joint vector must be finite shape (7,)")
        joints = self._kdl.JntArray(7)
        for index, value in enumerate(q_array):
            joints[index] = float(value)
        jacobian = self._kdl.Jacobian(7)
        status = self._solver.JntToJac(joints, jacobian)
        if status < 0:
            raise RuntimeError(f"PyKDL Jacobian solver failed with code {status}")
        return np.asarray(
            [[jacobian[row, column] for column in range(7)] for row in range(6)],
            dtype=np.float64,
        )


def create_node_class() -> type[Any]:
    """Import ROS interfaces and return the concrete rclpy Node class."""

    try:
        from builtin_interfaces.msg import Duration
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from rclpy.time import Time
        from sensor_msgs.msg import CameraInfo, Image, JointState
        from std_msgs.msg import Float64MultiArray, String
        from tf2_ros import Buffer, TransformException, TransformListener
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    except ImportError as exc:
        raise RuntimeError("ROS 2 Humble Python packages are not available") from exc

    class StudentV6PolicyNode(Node):
        def __init__(self) -> None:
            super().__init__("student_v6_policy")
            model_dir = default_model_dir()
            self.declare_parameter("policy_enabled", False)
            self.declare_parameter("onnx_model_path", "")
            self.declare_parameter("model_path", "")
            self.declare_parameter("contract_path", "")
            self.declare_parameter("controller_config_path", "")
            self.declare_parameter("expected_model_sha256", "")
            self.declare_parameter("robot_id", "")
            self.declare_parameter("arm_id", "")
            self.declare_parameter("base_frame", "panda_link0")
            self.declare_parameter("tcp_frame", "panda_hand_tcp")
            self.declare_parameter("camera_frame", "")
            self.declare_parameter("joint_states_topic", "joint_states")
            self.declare_parameter("gripper_state_topic", "gripper_joint_states")
            self.declare_parameter("image_topic", "camera/image_raw")
            self.declare_parameter("image_subscription_depth", 1)
            self.declare_parameter("image_qos_reliability", "best_effort")
            self.declare_parameter("image_width", 64)
            self.declare_parameter("image_height", 64)
            self.declare_parameter("image_encoding", "rgb8")
            self.declare_parameter("reject_resized_or_wrong_encoding_image", False)
            self.declare_parameter("camera_preprocess_mode", "native_64")
            self.declare_parameter("camera_info_topic", "")
            self.declare_parameter("policy_vertical_fov_deg", 55.0)
            self.declare_parameter("camera_aspect_tolerance", 0.02)
            self.declare_parameter("command_topic", "")
            self.declare_parameter("arm_command_topic", "")
            self.declare_parameter("command_message_type", "joint_trajectory")
            self.declare_parameter("telemetry_topic", "student_v6/telemetry")
            self.declare_parameter("control_hz", 60.0)
            self.declare_parameter("action_latency_steps", 0)
            self.declare_parameter("response_time_s", 0.5)
            self.declare_parameter("snapshot_poll_hz", 240.0)
            self.declare_parameter("trajectory_duration_s", 0.9 / 60.0)
            self.declare_parameter("bypass_command_limiter", False)
            self.declare_parameter("dls_anchor_estimate", False)
            self.declare_parameter("dls_anchor_alpha", 0.1)
            self.declare_parameter("dls_anchor_clamp_radius_m", 0.025)
            self.declare_parameter("dls_descend_target", "live")
            self.declare_parameter("approach_lock_wrist_home", True)
            self.declare_parameter("box_state_topic", "/box/free_joint_state")
            self.declare_parameter(
                "box_state_topic_fallback",
                "/mujoco_ros2_control_node/free_joint_state_publisher/free_joint_states",
            )
            self.declare_parameter("close_phase_step_limit", 0)
            self.declare_parameter("stale_timeout_s", 0.25)
            self.declare_parameter("joint_state_stale_timeout_s", 0.25)
            self.declare_parameter("image_stale_timeout_s", 0.25)
            self.declare_parameter("tcp_tf_stale_timeout_s", 0.25)
            self.declare_parameter("gripper_state_stale_timeout_s", 0.25)
            self.declare_parameter("observation_buffer_size", 120)
            self.declare_parameter("max_observation_skew_s", 0.05)
            self.declare_parameter("use_pykdl", True)
            self.declare_parameter("robot_description", "")
            self.declare_parameter("robot_description_topic", "/robot_description")
            self.declare_parameter("gripper_backend", "none")
            self.declare_parameter("gripper_action_name", "/panda_gripper/gripper_action")
            self.declare_parameter("sim_grasp_width_tolerance_m", 0.01)
            self.declare_parameter("grasp_action_name", "/panda_gripper/grasp")
            self.declare_parameter("move_action_name", "/panda_gripper/move")
            self.declare_parameter("grasp_width_m", 0.05)
            self.declare_parameter("grasp_speed_m_s", 0.025)
            self.declare_parameter("grasp_force_n", 20.0)
            self.declare_parameter("grasp_epsilon_inner_m", 0.005)
            self.declare_parameter("grasp_epsilon_outer_m", 0.005)
            self.declare_parameter("gripper_open_width_m", 0.08)
            self.declare_parameter("hold_on_fault", False)
            self.declare_parameter("startup_home_enabled", True)
            self.declare_parameter("startup_home_requires_policy_enabled", True)
            self.declare_parameter("startup_home_command_enabled", False)
            self.declare_parameter("startup_home_trajectory_duration_s", 2.0)
            self.declare_parameter(
                "startup_home_joints_rad",
                [0.0, 0.3, 0.0, -1.5707963267948966, 0.0, 2.0, -0.7853981633974483],
            )
            self.declare_parameter("startup_home_tolerance_rad", 0.03)
            self.declare_parameter("startup_home_velocity_tolerance_rad_s", 0.10)
            self.declare_parameter("startup_home_hold_ticks", 30)
            self.declare_parameter("readiness_consecutive_ticks", 12)
            self.declare_parameter("block_command_publication_until_ready", True)
            self.declare_parameter("stop_command_publication_on_stale_input", True)
            self.declare_parameter("require_joint_states", True)
            self.declare_parameter("require_image", True)
            self.declare_parameter("require_tcp_tf", True)
            self.declare_parameter("require_base_to_tcp_tf", True)
            self.declare_parameter("require_camera_tf", False)
            self.declare_parameter("require_base_to_camera_tf", False)
            self.declare_parameter("camera_tf_may_be_static", True)
            self.declare_parameter("require_gripper_state", True)
            self.declare_parameter("require_grasp_action_server", False)
            self.declare_parameter("dls_enabled", True)
            self.declare_parameter("autonomous_controller", False)
            self.declare_parameter("advance_threshold", 0.55)
            self.declare_parameter("mode_min_dwell_steps", 20)
            self.declare_parameter("mode_filter_alpha", 0.35)
            self.declare_parameter("mode_filter_margin", 0.15)
            self.declare_parameter("mode_filter_consecutive_steps", 3)
            self.declare_parameter("require_workcell_calibration_confirmation", False)
            self.declare_parameter("goal_pose_calibrated", False)
            self.declare_parameter("seated_offset_calibrated", False)
            self.declare_parameter("camera_extrinsics_calibrated", False)
            self.declare_parameter("seated_offset_m", 0.0229)
            self.declare_parameter("seated_offset_min_m", 0.0176)
            self.declare_parameter("seated_offset_max_m", 0.0311)
            self.declare_parameter("velocity_limit_fraction", 0.38)
            self.declare_parameter("grasp_retry_recovery_enabled", True)
            self.declare_parameter(
                "command_joint_lower_limits_rad",
                [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
            )
            self.declare_parameter(
                "command_joint_upper_limits_rad",
                [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
            )
            self.declare_parameter(
                "command_joint_velocity_limits_rad_s",
                [0.8265, 0.8265, 0.8265, 0.8265, 0.9918, 0.9918, 0.9918],
            )
            self.declare_parameter(
                "max_command_step_rad",
                [0.013775, 0.013775, 0.013775, 0.013775, 0.01653, 0.01653, 0.01653],
            )
            self.declare_parameter(
                "joint_names",
                [f"panda_joint{index}" for index in range(1, 8)],
            )
            self.declare_parameter(
                "finger_joint_names",
                ["panda_finger_joint1", "panda_finger_joint2"],
            )

            model_file, contract_file, controller_file = resolve_artifact_paths(
                onnx_model_path=str(self.get_parameter("onnx_model_path").value),
                model_path=str(self.get_parameter("model_path").value),
                contract_path=str(self.get_parameter("contract_path").value),
                controller_config_path=str(
                    self.get_parameter("controller_config_path").value
                ),
                model_dir=model_dir,
            )
            self._model_file = model_file
            self._config = load_runtime_config(
                model_file.parent,
                contract_path=contract_file,
                controller_config_path=controller_file,
                expected_onnx_sha256=sha256_file(model_file),
            )
            requested_hz = float(self.get_parameter("control_hz").value)
            if abs(requested_hz - self._config.control_hz) > 1.0e-6:
                raise ValueError("student-v6 must run at exactly 60 Hz")
            snapshot_poll_hz = float(
                self.get_parameter("snapshot_poll_hz").value
            )
            if snapshot_poll_hz < requested_hz:
                raise ValueError(
                    "snapshot_poll_hz must be at least control_hz"
                )
            self._snapshot_poll_hz = snapshot_poll_hz
            seated_offset = float(self.get_parameter("seated_offset_m").value)
            seated_min = float(self.get_parameter("seated_offset_min_m").value)
            seated_max = float(self.get_parameter("seated_offset_max_m").value)
            if not seated_min <= seated_offset <= seated_max:
                raise ValueError("seated_offset_m is outside configured calibration bounds")
            velocity_fraction = float(
                self.get_parameter("velocity_limit_fraction").value
            )
            if not 0.0 < velocity_fraction <= self._config.velocity_limit_fraction:
                raise ValueError(
                    "velocity_limit_fraction must be positive and no greater than 0.38"
                )
            close_limit = int(self.get_parameter("close_phase_step_limit").value)
            limits = list(self._config.phase_step_limits)
            if close_limit > 0:
                if close_limit < 20:
                    raise ValueError("close_phase_step_limit must be at least 20")
                limits[int(Phase.CLOSE_GRIPPER)] = close_limit
            self._config = replace(
                self._config,
                seated_offset_m=seated_offset,
                velocity_limit_fraction=velocity_fraction,
                response_time=float(
                    self.get_parameter("response_time_s").value
                ),
                phase_step_limits=tuple(limits),
            )
            self._robot_id = resolve_string_alias(
                str(self.get_parameter("robot_id").value),
                str(self.get_parameter("arm_id").value),
                "robot_id/arm_id",
            ) or "panda"
            configured_joints = tuple(self.get_parameter("joint_names").value)
            configured_fingers = tuple(self.get_parameter("finger_joint_names").value)
            default_panda_joints = tuple(
                f"panda_joint{index}" for index in range(1, 8)
            )
            default_panda_fingers = (
                "panda_finger_joint1",
                "panda_finger_joint2",
            )
            self._arm_joint_names = (
                configured_joints
                if len(configured_joints) == 7
                and not (
                    self._robot_id != "panda"
                    and configured_joints == default_panda_joints
                )
                else tuple(f"{self._robot_id}_joint{index}" for index in range(1, 8))
            )
            self._finger_joint_names = (
                configured_fingers
                if len(configured_fingers) == 2
                and not (
                    self._robot_id != "panda"
                    and configured_fingers == default_panda_fingers
                )
                else (
                    f"{self._robot_id}_finger_joint1",
                    f"{self._robot_id}_finger_joint2",
                )
            )
            self._base_frame = str(self.get_parameter("base_frame").value)
            self._tcp_frame = str(self.get_parameter("tcp_frame").value)
            self._camera_frame = str(self.get_parameter("camera_frame").value)
            legacy_stale = float(self.get_parameter("stale_timeout_s").value)
            self._stale_timeouts = {
                "joint_states": float(
                    self.get_parameter("joint_state_stale_timeout_s").value
                ) or legacy_stale,
                "image": float(self.get_parameter("image_stale_timeout_s").value)
                or legacy_stale,
                "tcp_tf": float(self.get_parameter("tcp_tf_stale_timeout_s").value)
                or legacy_stale,
                "gripper_state": float(
                    self.get_parameter("gripper_state_stale_timeout_s").value
                ) or legacy_stale,
            }
            self._trajectory_duration_s = float(
                self.get_parameter("trajectory_duration_s").value
            )
            self._hold_on_fault = bool(self.get_parameter("hold_on_fault").value)
            self._require_camera_tf = bool(
                self.get_parameter("require_camera_tf").value
                or self.get_parameter("require_base_to_camera_tf").value
            )
            self._require_tcp_tf = bool(
                self.get_parameter("require_tcp_tf").value
                or self.get_parameter("require_base_to_tcp_tf").value
            )
            self._require_gripper_state = bool(
                self.get_parameter("require_gripper_state").value
            )
            # Hybrid Stage C: autonomous_controller=true with dls_enabled=true.
            # Thin Stage E: autonomous_controller=true with dls_enabled=false.
            self._dls_enabled = bool(self.get_parameter("dls_enabled").value)
            self._startup_home_enabled = bool(
                self.get_parameter("startup_home_enabled").value
            ) or str(self.get_parameter("gripper_backend").value).lower() == "hardware"

            command_topic = resolve_string_alias(
                str(self.get_parameter("command_topic").value),
                str(self.get_parameter("arm_command_topic").value),
                "command_topic/arm_command_topic",
            ) or f"{self._robot_id}_joint_trajectory_controller/joint_trajectory"
            self._command_message_type = str(
                self.get_parameter("command_message_type").value
            ).lower()
            if self._command_message_type == "joint_trajectory":
                self._command_pub = self.create_publisher(
                    JointTrajectory, command_topic, 10
                )
            elif self._command_message_type == "float64_multi_array":
                self._command_pub = self.create_publisher(
                    Float64MultiArray, command_topic, 10
                )
            else:
                raise ValueError(
                    "command_message_type must be joint_trajectory or "
                    "float64_multi_array"
                )
            self._telemetry_pub = self.create_publisher(
                String, str(self.get_parameter("telemetry_topic").value), 10
            )
            self.create_subscription(
                JointState,
                str(self.get_parameter("joint_states_topic").value),
                self._on_joint_state,
                10,
            )
            self.create_subscription(
                JointState,
                str(self.get_parameter("gripper_state_topic").value),
                self._on_gripper_state,
                10,
            )
            image_reliability_name = str(
                self.get_parameter("image_qos_reliability").value
            ).lower()
            image_reliability = {
                "best_effort": ReliabilityPolicy.BEST_EFFORT,
                "reliable": ReliabilityPolicy.RELIABLE,
            }.get(image_reliability_name)
            if image_reliability is None:
                raise ValueError(
                    "image_qos_reliability must be best_effort or reliable"
                )
            image_qos = QoSProfile(
                depth=int(self.get_parameter("image_subscription_depth").value),
                reliability=image_reliability,
            )
            self._camera_preprocess_mode = str(
                self.get_parameter("camera_preprocess_mode").value
            ).lower()
            if self._camera_preprocess_mode not in ("native_64", "oak_4_by_3"):
                raise ValueError(
                    "camera_preprocess_mode must be native_64 or oak_4_by_3"
                )
            self._camera_preprocessor: OakPolicyCameraPreprocessor | None = None
            if self._camera_preprocess_mode == "oak_4_by_3":
                camera_info_topic = str(
                    self.get_parameter("camera_info_topic").value
                ).strip()
                if not camera_info_topic:
                    raise ValueError(
                        "camera_info_topic is required for oak_4_by_3 preprocessing"
                    )
                self.create_subscription(
                    CameraInfo,
                    camera_info_topic,
                    self._on_camera_info,
                    image_qos,
                )
            self.create_subscription(
                Image,
                str(self.get_parameter("image_topic").value),
                self._on_image,
                image_qos,
            )
            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)
            description_qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.create_subscription(
                String,
                str(self.get_parameter("robot_description_topic").value),
                self._on_robot_description,
                description_qos,
            )
            self._dls_descend_target = normalize_dls_descend_target(
                str(self.get_parameter("dls_descend_target").value)
            )
            self._approach_lock_wrist_home = bool(
                self.get_parameter("approach_lock_wrist_home").value
            )
            self._sim_box_position_m: np.ndarray | None = None
            if self._dls_descend_target == "sim_box":
                try:
                    from mujoco_ros2_control_msgs.msg import FreeJointStateArray
                except ImportError as exc:
                    raise RuntimeError(
                        "dls_descend_target=sim_box requires mujoco_ros2_control_msgs"
                    ) from exc
                topics = []
                for name in ("box_state_topic", "box_state_topic_fallback"):
                    topic = str(self.get_parameter(name).value).strip()
                    if topic and topic not in topics:
                        topics.append(topic)
                        self.create_subscription(
                            FreeJointStateArray,
                            topic,
                            self._on_box_state,
                            50,
                        )

            self._q: np.ndarray | None = None
            self._qvel: np.ndarray | None = None
            self._finger_span_m: float | None = None
            self._pixels: np.ndarray | None = None
            self._tcp_position_m: np.ndarray | None = None
            self._tcp_rotation: np.ndarray | None = None
            self._joint_stamp_s: float | None = None
            self._gripper_stamp_s: float | None = None
            self._image_stamp_s: float | None = None
            self._tf_stamp_s: float | None = None
            self._camera_tf_valid = False
            self._policy: OnnxPolicy | None = None
            self._observation = ObservationBuilder()
            self._synchronizer = SynchronizedObservationBuffer(
                max_samples=int(self.get_parameter("observation_buffer_size").value),
                max_skew_s=float(self.get_parameter("max_observation_skew_s").value),
            )
            self._integrator = IntegratedActionController(
                self._config,
                action_latency_steps=int(
                    self.get_parameter("action_latency_steps").value
                ),
            )
            self._autonomous_controller = bool(
                self.get_parameter("autonomous_controller").value
            )
            self._session = (
                AutonomousSession(
                    self._config,
                    mode_alpha=float(
                        self.get_parameter("mode_filter_alpha").value
                    ),
                    mode_margin=float(
                        self.get_parameter("mode_filter_margin").value
                    ),
                    mode_consecutive_steps=int(
                        self.get_parameter(
                            "mode_filter_consecutive_steps"
                        ).value
                    ),
                    advance_threshold=float(
                        self.get_parameter("advance_threshold").value
                    ),
                    min_dwell_steps=int(
                        self.get_parameter("mode_min_dwell_steps").value
                    ),
                )
                if self._autonomous_controller
                else ObservableSession(self._config)
            )
            self._dls: DLSController | None = None
            self._analytic_fallback_warned = False
            self._gripper: GripperBackend = NoopGripperBackend()
            self._gripper_initialized = False
            self._runtime_ready = False
            self._robot_description = str(
                self.get_parameter("robot_description").value
            )
            self._faulted = False
            self._last_fault = ""
            self._was_enabled = False
            self._last_published_command: np.ndarray | None = None
            self._last_timer_sim_s: float | None = None
            self._last_timer_wall_s: float | None = None
            self._timer_sim_delta_s: float | None = None
            self._timer_wall_delta_s: float | None = None
            self._timer_callback_start_s = perf_counter()
            self._last_snapshot_attempt: SnapshotAttempt | None = None
            self._last_policy_image_stamp_s: float | None = None
            self._policy_step_delta_s: float | None = None
            self._dls_cube_anchor_m = np.zeros(3, dtype=np.float64)
            self._dls_anchor_valid = False
            self._startup_home_complete = not self._startup_home_enabled
            self._startup_home_command_sent = False
            self._home_readiness = HomeReadiness(
                np.asarray(self.get_parameter("startup_home_joints_rad").value),
                float(self.get_parameter("startup_home_tolerance_rad").value),
                float(
                    self.get_parameter(
                        "startup_home_velocity_tolerance_rad_s"
                    ).value
                ),
                int(self.get_parameter("startup_home_hold_ticks").value),
            )
            self._readiness = ConsecutiveReadiness(
                int(self.get_parameter("readiness_consecutive_ticks").value)
            )
            self._command_limiter = CommandLimiter(
                np.asarray(
                    self.get_parameter("command_joint_lower_limits_rad").value
                ),
                np.asarray(
                    self.get_parameter("command_joint_upper_limits_rad").value
                ),
                np.asarray(
                    self.get_parameter(
                        "command_joint_velocity_limits_rad_s"
                    ).value
                ),
                np.asarray(self.get_parameter("max_command_step_rad").value),
                self._config.control_hz,
            )
            self._recovery = RecoveryToHome(
                retreat_height_m=self._config.grasp_retry_retreat_height_m,
                position_tolerance_m=(
                    self._config.grasp_retry_position_tolerance_m
                ),
                max_joint_speed_rad_s=(
                    self._config.grasp_retry_max_joint_speed_rad_s
                ),
                settled_steps=self._config.grasp_retry_settled_steps,
                open_fraction=self._config.release_open_fraction,
                fresh_estimate_steps=(
                    self._config.grasp_retry_fresh_estimate_steps
                ),
            )
            self.create_timer(1.0 / self._snapshot_poll_hz, self._on_timer)
            self.get_logger().info(
                "student-v6 ready at 60 Hz with policy_enabled=false safety default"
            )

        def _now_s(self) -> float:
            return self.get_clock().now().nanoseconds * 1.0e-9

        def _message_stamp_s(self, message: Any) -> float:
            stamp = getattr(getattr(message, "header", None), "stamp", None)
            if stamp is not None:
                value = float(stamp.sec) + float(stamp.nanosec) * 1.0e-9
                if value > 0.0:
                    return value
            return self._now_s()

        def _on_joint_state(self, message: Any) -> None:
            q, qvel, finger_span = extract_joint_measurements(
                message.name,
                message.position,
                message.velocity,
                self._arm_joint_names,
                self._finger_joint_names,
            )
            if q is None or qvel is None:
                return
            if not (np.all(np.isfinite(q)) and np.all(np.isfinite(qvel))):
                self._faulted = True
                self._last_fault = "nonfinite_joint_state"
                return
            self._q = q
            self._qvel = qvel
            self._joint_stamp_s = self._message_stamp_s(message)
            self._synchronizer.add_joint(self._joint_stamp_s, q, qvel)
            tcp_position, tcp_rotation = panda_hand_tcp_pose(q)
            self._synchronizer.add_tcp(
                self._joint_stamp_s,
                tcp_position,
                tcp_rotation,
            )
            if finger_span is not None:
                self._set_finger_span(
                    finger_span, self._message_stamp_s(message)
                )
            if self._integrator.smoothed_target is None:
                self._integrator.reset(q)
                self._observation.reset(q, qvel)

        def _set_finger_span(
            self, finger_span: float, stamp_s: float
        ) -> None:
            if not np.isfinite(finger_span):
                self._faulted = True
                self._last_fault = "nonfinite_gripper_state"
                return
            self._finger_span_m = float(finger_span)
            self._gripper_stamp_s = stamp_s
            self._synchronizer.add_gripper(stamp_s, finger_span)

        def _on_gripper_state(self, message: Any) -> None:
            _, _, finger_span = extract_joint_measurements(
                message.name,
                message.position,
                message.velocity,
                self._arm_joint_names,
                self._finger_joint_names,
            )
            if finger_span is not None:
                self._set_finger_span(
                    finger_span, self._message_stamp_s(message)
                )

        def _on_box_state(self, message: Any) -> None:
            for entry in getattr(message, "free_joints", ()):
                if str(getattr(entry, "name", "")) != "box":
                    continue
                position = entry.pose.pose.position
                xyz = np.asarray(
                    (float(position.x), float(position.y), float(position.z)),
                    dtype=np.float64,
                )
                if not np.all(np.isfinite(xyz)):
                    return
                self._sim_box_position_m = xyz
                return

        def _update_cube_anchor(
            self,
            *,
            phase: int | Phase,
            tcp_position_m: np.ndarray,
            geometry: np.ndarray,
        ) -> None:
            if self._dls_descend_target == "sim_box":
                if self._sim_box_position_m is None:
                    return
                if int(phase) == int(Phase.HOVER_RED):
                    self._dls_cube_anchor_m = self._sim_box_position_m.copy()
                    self._dls_anchor_valid = True
                return
            if int(phase) != int(Phase.HOVER_RED):
                return
            if self._dls_descend_target != "hover_estimate":
                return
            observed_cube = (
                tcp_position_m + geometry[:3] * self._config.geometry_scale_m
            )
            self._dls_cube_anchor_m = update_hover_cube_anchor(
                self._dls_cube_anchor_m,
                valid=self._dls_anchor_valid,
                observed_cube_m=observed_cube,
                alpha=float(self.get_parameter("dls_anchor_alpha").value),
                clamp_radius_m=float(
                    self.get_parameter("dls_anchor_clamp_radius_m").value
                ),
            )
            self._dls_anchor_valid = True

        def _on_robot_description(self, message: Any) -> None:
            description = str(message.data)
            if description and description != self._robot_description:
                self._robot_description = description
                self._dls = None
                try:
                    self._try_initialize_dls()
                    if self._dls is not None:
                        self._faulted = False
                        self._last_fault = ""
                except Exception as exc:
                    self._last_fault = f"dls_initialization:{exc}"

        def _on_camera_info(self, message: Any) -> None:
            try:
                calibration = CameraCalibration.from_message(message)
                self._camera_preprocessor = OakPolicyCameraPreprocessor(
                    calibration,
                    target_size=int(self.get_parameter("image_width").value),
                    target_vertical_fov_deg=float(
                        self.get_parameter("policy_vertical_fov_deg").value
                    ),
                    aspect_tolerance=float(
                        self.get_parameter("camera_aspect_tolerance").value
                    ),
                )
            except (TypeError, ValueError) as exc:
                self._faulted = True
                self._last_fault = f"invalid_camera_info:{exc}"

        def _on_image(self, message: Any) -> None:
            callback_start_s = perf_counter()
            try:
                if (
                    self._camera_preprocess_mode == "oak_4_by_3"
                    and self._camera_preprocessor is None
                ):
                    raise ValueError("calibrated OAK CameraInfo has not been received")
                if bool(
                    self.get_parameter(
                        "reject_resized_or_wrong_encoding_image"
                    ).value
                ) and (
                    (
                        self._camera_preprocess_mode == "native_64"
                        and (
                            int(message.width)
                            != int(self.get_parameter("image_width").value)
                            or int(message.height)
                            != int(self.get_parameter("image_height").value)
                        )
                    )
                    or str(message.encoding).lower()
                    != str(self.get_parameter("image_encoding").value).lower()
                ):
                    raise ValueError("image does not match configured camera contract")
                pixels = image_to_policy_pixels(
                    message,
                    preprocessor=self._camera_preprocessor,
                )
                stamp_s = self._message_stamp_s(message)
                callback_duration_s = perf_counter() - callback_start_s
                self._pixels = pixels
                self._image_stamp_s = stamp_s
                self._synchronizer.add_image(
                    stamp_s,
                    pixels,
                    callback_duration_s=callback_duration_s,
                )
            except (TypeError, ValueError) as exc:
                self._faulted = True
                self._last_fault = f"invalid_image:{exc}"

        def _update_tf(self) -> None:
            if self._require_tcp_tf:
                try:
                    query_time = (
                        Time(nanoseconds=max(int(round(self._image_stamp_s * 1.0e9)), 0))
                        if self._image_stamp_s is not None
                        else Time()
                    )
                    transform = self._tf_buffer.lookup_transform(
                        self._base_frame, self._tcp_frame, query_time
                    )
                except TransformException:
                    transform = None
                if transform is not None:
                    translation = transform.transform.translation
                    rotation = transform.transform.rotation
                    try:
                        position = np.asarray(
                            (translation.x, translation.y, translation.z),
                            dtype=np.float64,
                        )
                        matrix = quaternion_to_rotation(
                            rotation.x, rotation.y, rotation.z, rotation.w
                        )
                    except ValueError:
                        self._faulted = True
                        self._last_fault = "invalid_tcp_transform"
                        return
                    if not np.all(np.isfinite(position)):
                        self._faulted = True
                        self._last_fault = "nonfinite_tcp_transform"
                        return
                    self._tcp_position_m = position
                    self._tcp_rotation = matrix
                    stamp = transform.header.stamp
                    transform_stamp_s = (
                        float(stamp.sec) + float(stamp.nanosec) * 1.0e-9
                    )
                    self._tf_stamp_s = (
                        transform_stamp_s
                        if transform_stamp_s > 0.0
                        else (
                            self._image_stamp_s
                            if self._image_stamp_s is not None
                            else self._now_s()
                        )
                    )

            if self._require_camera_tf and self._camera_frame:
                if not bool(
                    self.get_parameter("camera_tf_may_be_static").value
                ):
                    self._camera_tf_valid = False
                try:
                    camera_transform = self._tf_buffer.lookup_transform(
                        self._base_frame, self._camera_frame, Time()
                    )
                except TransformException:
                    return
                camera_translation = camera_transform.transform.translation
                camera_rotation = camera_transform.transform.rotation
                try:
                    camera_values = np.asarray(
                        (
                            camera_translation.x,
                            camera_translation.y,
                            camera_translation.z,
                        ),
                        dtype=np.float64,
                    )
                    quaternion_to_rotation(
                        camera_rotation.x,
                        camera_rotation.y,
                        camera_rotation.z,
                        camera_rotation.w,
                    )
                except ValueError:
                    self._camera_tf_valid = False
                    return
                self._camera_tf_valid = bool(np.all(np.isfinite(camera_values)))

        def _make_gripper(self) -> GripperBackend:
            backend = str(self.get_parameter("gripper_backend").value).lower()
            if backend == "none":
                return NoopGripperBackend()
            if backend == "sim":
                return SimGripperCommandBackend(
                    self,
                    str(self.get_parameter("gripper_action_name").value),
                    object_width_m=self._config.observable_object_width_m,
                    grasp_width_tolerance_m=float(
                        self.get_parameter(
                            "sim_grasp_width_tolerance_m"
                        ).value
                    ),
                    max_effort=100.0,
                )
            if backend == "hardware":
                return FrankaGraspBackend(
                    self,
                    str(self.get_parameter("grasp_action_name").value),
                    str(self.get_parameter("move_action_name").value),
                    object_width_m=float(
                        self.get_parameter("grasp_width_m").value
                    ),
                    epsilon_inner_m=float(
                        self.get_parameter("grasp_epsilon_inner_m").value
                    ),
                    epsilon_outer_m=float(
                        self.get_parameter("grasp_epsilon_outer_m").value
                    ),
                    speed_m_s=float(
                        self.get_parameter("grasp_speed_m_s").value
                    ),
                    force_n=float(self.get_parameter("grasp_force_n").value),
                    open_width_m=float(
                        self.get_parameter("gripper_open_width_m").value
                    ),
                )
            raise ValueError("gripper_backend must be one of: none, sim, hardware")

        def _ensure_runtime(self) -> bool:
            try:
                if self._policy is None:
                    expected = str(
                        self.get_parameter("expected_model_sha256").value
                    ).strip()
                    if expected:
                        self._policy = OnnxPolicy(
                            self._model_file, expected_sha256=expected
                        )
                    else:
                        self._policy = OnnxPolicy(
                            self._model_file,
                            expected_sha256=sha256_file(self._model_file),
                        )
                    if self._autonomous_controller != self._policy.autonomous:
                        raise ValueError(
                            "autonomous_controller must match the ONNX mode output"
                        )
                if not self._gripper_initialized:
                    self._gripper = self._make_gripper()
                    self._gripper_initialized = True
                self._try_initialize_dls()
                self._runtime_ready = bool(
                    self._policy is not None
                    and self._gripper_initialized
                    and (not self._dls_enabled or self._dls is not None)
                )
                if not self._runtime_ready:
                    self._last_fault = (
                        "waiting_for_robot_description"
                        if not self._robot_description
                        else "dls_unavailable"
                    )
                return self._runtime_ready
            except Exception as exc:  # ROS boundary: convert all startup errors to hold.
                self._faulted = True
                self._last_fault = f"runtime_initialization:{exc}"
                self.get_logger().error(self._last_fault)
                return False

        def _try_initialize_dls(self) -> None:
            if not self._dls_enabled or self._dls is not None:
                return
            canonical_panda_chain = (
                self._robot_id == "panda"
                and self._base_frame == "panda_link0"
                and self._tcp_frame == "panda_hand_tcp"
            )
            parameter_description = str(
                self.get_parameter("robot_description").value
            )
            if parameter_description:
                self._robot_description = parameter_description
            provider: Any | None = None
            if (
                bool(self.get_parameter("use_pykdl").value)
                and self._robot_description
            ):
                try:
                    provider = PyKDLJacobian(
                        self._robot_description,
                        self._base_frame,
                        self._tcp_frame,
                    )
                except RuntimeError:
                    if not canonical_panda_chain:
                        raise
            if provider is None and canonical_panda_chain:
                provider = PandaAnalyticJacobian()
                if not self._analytic_fallback_warned:
                    self.get_logger().warning(
                        "kdl_parser_py is unavailable; using the exact "
                        "panda_link0-to-panda_hand_tcp analytic Jacobian"
                    )
                    self._analytic_fallback_warned = True
            if provider is None:
                return
            self._dls = DLSController(self._config, provider)

        def _publish_command(
            self,
            command: np.ndarray,
            *,
            duration_s: float | None = None,
        ) -> None:
            if not bool(self.get_parameter("policy_enabled").value):
                raise RuntimeError("arm command publication is disabled")
            if self._command_message_type == "float64_multi_array":
                message = Float64MultiArray()
                message.data = [float(value) for value in command]
                self._command_pub.publish(message)
                self._last_published_command = np.asarray(
                    command, dtype=np.float64
                ).copy()
                return
            point = JointTrajectoryPoint()
            point.positions = [float(value) for value in command]
            seconds = max(
                self._trajectory_duration_s
                if duration_s is None
                else float(duration_s),
                0.001,
            )
            point.time_from_start = Duration(
                sec=int(seconds), nanosec=int((seconds % 1.0) * 1.0e9)
            )
            message = JointTrajectory()
            message.joint_names = list(self._arm_joint_names)
            message.points = [point]
            self._command_pub.publish(message)
            self._last_published_command = np.asarray(command, dtype=np.float64).copy()

        def _limit_command(
            self, desired: np.ndarray, reference: np.ndarray
        ) -> np.ndarray:
            """Apply the hardware safety envelope unless simulation requests exact parity."""

            if bool(self.get_parameter("bypass_command_limiter").value):
                return np.asarray(desired, dtype=np.float64).copy()
            return self._command_limiter.limit(desired, reference)

        def _publish_telemetry(self, payload: dict[str, Any]) -> None:
            message = String()
            message.data = json.dumps(payload, sort_keys=True)
            self._telemetry_pub.publish(message)

        def _hold(self, reason: str) -> None:
            enabled = bool(self.get_parameter("policy_enabled").value)
            command_published = False
            if (
                may_publish_fault_hold(enabled, self._hold_on_fault)
                and self._last_published_command is not None
            ):
                self._publish_command(self._last_published_command)
                command_published = True
            attempt = self._last_snapshot_attempt
            self._publish_telemetry(
                {
                    "policy_enabled": enabled,
                    "safe_hold": True,
                    "reason": reason,
                    "command_published": command_published,
                    "phase": int(self._session.phase),
                    "phase_name": PHASE_NAMES[int(self._session.phase)],
                    "phase_step": self._session.phase_step,
                    "sim_timer_delta_s": self._timer_sim_delta_s,
                    "wall_timer_delta_s": self._timer_wall_delta_s,
                    "control_callback_duration_s": (
                        perf_counter() - self._timer_callback_start_s
                    ),
                    "q": None if self._q is None else self._q.tolist(),
                    "qvel": None if self._qvel is None else self._qvel.tolist(),
                    "published_command": (
                        None
                        if self._last_published_command is None
                        else self._last_published_command.tolist()
                    ),
                    "tracking_error": (
                        None
                        if self._last_published_command is None or self._q is None
                        else (self._last_published_command - self._q).tolist()
                    ),
                    "reused_frame": (
                        False if attempt is None else attempt.frame_reused
                    ),
                    "missed_frame": (
                        False if attempt is None else attempt.missed_frames > 0
                    ),
                    "missed_frame_count": (
                        0 if attempt is None else attempt.missed_frames
                    ),
                    "total_missed_frames": (
                        0 if attempt is None else attempt.total_missed_frames
                    ),
                    "frame_sequence": (
                        None if attempt is None else attempt.frame_sequence
                    ),
                    "image_age_s": (
                        None if attempt is None else attempt.image_age_s
                    ),
                    "joint_age_s": (
                        None if attempt is None else attempt.joint_age_s
                    ),
                    "tcp_tf_age_s": (
                        None if attempt is None else attempt.tcp_tf_age_s
                    ),
                    "gripper_age_s": (
                        None if attempt is None else attempt.gripper_age_s
                    ),
                    "joint_image_skew_s": (
                        None if attempt is None else attempt.joint_image_skew_s
                    ),
                    "tcp_image_skew_s": (
                        None if attempt is None else attempt.tcp_image_skew_s
                    ),
                    "gripper_image_skew_s": (
                        None if attempt is None else attempt.gripper_image_skew_s
                    ),
                    "max_input_skew_s": (
                        None
                        if attempt is None
                        else max(
                            (
                                value
                                for value in (
                                    attempt.joint_image_skew_s,
                                    attempt.tcp_image_skew_s,
                                    attempt.gripper_image_skew_s,
                                )
                                if value is not None
                            ),
                            default=None,
                        )
                    ),
                }
            )

        def _health_reason(self) -> str:
            now = self._now_s()
            values = [
                (
                    "joint_states",
                    self._q,
                    self._joint_stamp_s,
                    bool(self.get_parameter("require_joint_states").value),
                ),
                (
                    "image",
                    self._pixels,
                    self._image_stamp_s,
                    bool(self.get_parameter("require_image").value),
                ),
                ("tcp_tf", self._tcp_position_m, self._tf_stamp_s, self._require_tcp_tf),
                (
                    "gripper_state",
                    (
                        None
                        if self._finger_span_m is None
                        else np.asarray((self._finger_span_m,))
                    ),
                    self._gripper_stamp_s,
                    self._require_gripper_state,
                ),
            ]
            for name, value, stamp, required in values:
                if not required:
                    continue
                if value is None or stamp is None:
                    return f"waiting_for_{name}"
                if (
                    bool(
                        self.get_parameter(
                            "stop_command_publication_on_stale_input"
                        ).value
                    )
                    and now - stamp > self._stale_timeouts[name]
                ):
                    return f"stale_{name}"
                if not np.all(np.isfinite(value)):
                    return f"nonfinite_{name}"
            if self._require_camera_tf and not self._camera_tf_valid:
                return "waiting_for_camera_tf"
            if (
                bool(self.get_parameter("require_grasp_action_server").value)
                and (
                    str(self.get_parameter("gripper_backend").value).lower()
                    != "hardware"
                    or not self._gripper.ready()
                )
            ):
                return "waiting_for_gripper_action_servers"
            if self._qvel is None or self._finger_span_m is None or self._tcp_rotation is None:
                return "incomplete_sensor_state"
            if bool(
                self.get_parameter(
                    "require_workcell_calibration_confirmation"
                ).value
            ):
                for parameter in (
                    "goal_pose_calibrated",
                    "seated_offset_calibrated",
                    "camera_extrinsics_calibrated",
                ):
                    if not bool(self.get_parameter(parameter).value):
                        return f"waiting_for_{parameter}"
            return ""

        def _begin_enabled_session(self) -> None:
            self._dls_descend_target = normalize_dls_descend_target(
                str(self.get_parameter("dls_descend_target").value)
            )
            latency = int(self.get_parameter("action_latency_steps").value)
            if not 0 <= latency <= 3:
                raise ValueError("action_latency_steps must be in [0, 3]")
            response_time = float(
                self.get_parameter("response_time_s").value
            )
            if response_time <= 0.0:
                raise ValueError("response_time_s must be positive")
            self._config = replace(
                self._config, response_time=response_time
            )
            self._integrator.config = self._config
            self._integrator.action_latency_steps = latency
            if self._dls is not None:
                self._dls.config = self._config
            self._faulted = False
            self._last_fault = ""
            self._startup_home_complete = not self._startup_home_enabled
            self._startup_home_command_sent = False
            self._home_readiness.reset()
            self._readiness.reset()
            self._session.reset()
            self._recovery.reset()
            self._last_published_command = None
            self._last_snapshot_attempt = None
            self._last_policy_image_stamp_s = None
            self._policy_step_delta_s = None
            self._dls_cube_anchor_m.fill(0.0)
            self._dls_anchor_valid = False
            self._sim_box_position_m = None
            self._synchronizer.discard_pending_images()
            if self._q is not None and self._qvel is not None:
                self._integrator.reset(self._q)
                self._observation.reset(self._q, self._qvel)

        def _recovery_step(self, snapshot: Any) -> None:
            if self._dls is None:
                self._hold("dls_unavailable_during_grasp_recovery")
                return
            openness = float(np.clip(snapshot.finger_span_m / 0.08, 0.0, 1.0))
            recovery = self._recovery.step(
                tcp_position_m=snapshot.tcp_position_m,
                joint_velocity=snapshot.qvel,
                gripper_openness=openness,
                grasp_latched=bool(self._gripper.latch_override()),
                estimate_generation=snapshot.frame_sequence,
                estimate_valid=True,
            )
            if recovery.request_open:
                self._gripper.request_open()

            if recovery.stage.value == "retreat":
                displacement = (
                    recovery.target_position_m - snapshot.tcp_position_m
                )
                recovery_action = self._dls.compute_displacement_action(
                    q=snapshot.q,
                    qvel=snapshot.qvel,
                    displacement_m=displacement,
                    maximum_speed_m_s=self._config.dls_hover_speed_m_s,
                    nullspace_gain=0.0,
                    braking_distance_m=0.05,
                )
                if self._approach_lock_wrist_home:
                    recovery_action = lock_wrist_to_home(
                        recovery_action, self._config
                    )
                integrated = self._integrator.step(
                    recovery_action, snapshot.q
                )
                reference = (
                    self._last_published_command
                    if self._last_published_command is not None
                    else snapshot.q
                )
                command = self._limit_command(integrated, reference)
            else:
                command = (
                    self._last_published_command.copy()
                    if self._last_published_command is not None
                    else snapshot.q.copy()
                )
            self._publish_command(command)
            if recovery.complete:
                self._integrator.reset(snapshot.q)
                self._observation.reset(snapshot.q, snapshot.qvel)
            self._publish_telemetry(
                {
                    "policy_enabled": True,
                    "safe_hold": False,
                    "reason": (
                        "grasp_retry_recovery_complete"
                        if recovery.complete
                        else f"grasp_retry_recovery_{recovery.stage.value}"
                    ),
                    "recovery_active": self._recovery.active,
                    "grasp_retries": self._session.grasp_retries,
                    "command": command.tolist(),
                    "recovery": recovery.telemetry,
                }
            )

        def _on_timer(self) -> None:
            wall_now_s = perf_counter()
            sim_now_s = self._now_s()
            self._timer_callback_start_s = wall_now_s
            self._timer_wall_delta_s = (
                None
                if self._last_timer_wall_s is None
                else max(wall_now_s - self._last_timer_wall_s, 0.0)
            )
            self._timer_sim_delta_s = (
                None
                if self._last_timer_sim_s is None
                else max(sim_now_s - self._last_timer_sim_s, 0.0)
            )
            self._last_timer_wall_s = wall_now_s
            self._last_timer_sim_s = sim_now_s
            self._last_snapshot_attempt = None
            self._update_tf()
            enabled = bool(self.get_parameter("policy_enabled").value)
            if not enabled:
                self._faulted = False
                self._last_fault = ""
                self._was_enabled = False
                self._readiness.reset()
                self._home_readiness.reset()
                self._hold("policy_disabled")
                return
            if not self._was_enabled:
                self._begin_enabled_session()
                self._was_enabled = True
            if self._faulted:
                self._hold(self._last_fault or "faulted")
                return
            if not self._ensure_runtime():
                self._hold(self._last_fault)
                return
            health = self._health_reason()
            if health:
                self._readiness.reset()
                self._hold(health)
                return
            assert self._policy is not None
            assert self._q is not None and self._qvel is not None
            assert self._pixels is not None and self._finger_span_m is not None
            assert self._tcp_position_m is not None and self._tcp_rotation is not None
            if not self._startup_home_complete:
                self._startup_home_complete = self._home_readiness.update(
                    policy_enabled=enabled,
                    q=self._q,
                    qvel=self._qvel,
                )
                if not self._startup_home_complete:
                    if bool(
                        self.get_parameter(
                            "startup_home_command_enabled"
                        ).value
                    ):
                        if not self._startup_home_command_sent:
                            startup_command = self._home_readiness.home.copy()
                            self._publish_command(
                                startup_command,
                                duration_s=float(
                                    self.get_parameter(
                                        "startup_home_trajectory_duration_s"
                                    ).value
                                ),
                            )
                            self._gripper.request_open()
                            self._startup_home_command_sent = True
                        self._hold(
                            "waiting_for_commanded_sim_startup_home:"
                            f"{self._home_readiness.count}/"
                            f"{self._home_readiness.required_ticks}"
                        )
                        return
                    self._hold(
                        "waiting_for_measured_startup_home:"
                        f"{self._home_readiness.count}/"
                        f"{self._home_readiness.required_ticks}"
                    )
                    return
                self._integrator.reset(self._q)
                self._observation.reset(self._q, self._qvel)
            if bool(
                self.get_parameter("block_command_publication_until_ready").value
            ) and not self._readiness.update(True):
                self._hold(
                    f"waiting_for_readiness:{self._readiness.count}/"
                    f"{self._readiness.required_ticks}"
                )
                return
            self._last_snapshot_attempt = self._synchronizer.try_snapshot(sim_now_s)
            snapshot = self._last_snapshot_attempt.snapshot
            if snapshot is None:
                if self._last_snapshot_attempt.reason == "no_fresh_image":
                    return
                self._hold(f"observation_sync:{self._last_snapshot_attempt.reason}")
                return
            self._policy_step_delta_s = (
                None
                if self._last_policy_image_stamp_s is None
                else max(
                    snapshot.image_stamp_s - self._last_policy_image_stamp_s,
                    0.0,
                )
            )
            self._last_policy_image_stamp_s = snapshot.image_stamp_s
            if self._recovery.active:
                self._recovery_step(snapshot)
                return
            active_phase = self._session.phase
            active_phase_step = self._session.phase_step
            openness = float(np.clip(snapshot.finger_span_m / 0.08, 0.0, 1.0))
            try:
                previous_command = (
                    self._integrator.smoothed_target
                    if self._integrator.smoothed_target is not None
                    else snapshot.q
                )
                state = self._observation.build_state(
                    snapshot.q,
                    snapshot.qvel,
                    previous_command,
                    snapshot.tcp_position_m,
                    active_phase,
                    active_phase_step,
                    openness,
                    # Feed the FSM phase counter as a clean one-hot (mode_probs
                    # left None). The learned advance head drives active_phase;
                    # the noisy mode head is not part of the action feedback.
                    mode_probs=None,
                )
                inference_start_s = perf_counter()
                (
                    policy_action,
                    geometry,
                    mode_probs,
                    advance_prob,
                ) = self._policy.infer_all(snapshot.pixels, state)
                inference_duration_s = perf_counter() - inference_start_s
                if self._autonomous_controller and mode_probs is None:
                    raise ValueError("autonomous ONNX model omitted mode_probs")
                geometry_for_control = np.asarray(geometry, dtype=np.float64)
                if not self._autonomous_controller:
                    self._update_cube_anchor(
                        phase=active_phase,
                        tcp_position_m=snapshot.tcp_position_m,
                        geometry=geometry_for_control,
                    )
                approach_frozen = bool(
                    not self._autonomous_controller
                    and self._dls_descend_target in ("sim_box", "hover_estimate")
                    and self._dls_anchor_valid
                    and int(active_phase) == int(Phase.DESCEND_GRASP)
                )
                if (
                    self._dls_descend_target == "sim_box"
                    and int(active_phase) == int(Phase.DESCEND_GRASP)
                    and not self._dls_anchor_valid
                ):
                    self._hold("sim_box_pose_unavailable")
                    return
                if approach_frozen:
                    geometry_for_control = (
                        rewrite_cube_geometry_to_world_anchor(
                            geometry_for_control,
                            snapshot.tcp_position_m,
                            self._dls_cube_anchor_m,
                            self._config.geometry_scale_m,
                        )
                    )
                needs_dls = int(active_phase) == int(Phase.DESCEND_GRASP) or (
                    int(active_phase) == int(Phase.HOVER_RED)
                    and active_phase_step
                    >= self._config.dls_warmup_steps_per_phase
                )
                # Thin autonomous (Stage E): dls_enabled=false.
                # Stage C learned-mode + DLS: autonomous_controller with dls_enabled.
                needs_dls = bool(needs_dls and self._dls_enabled)
                hold_arm = bool(
                    int(active_phase) == int(Phase.CLOSE_GRIPPER)
                    and self._dls_enabled
                )
                if hold_arm:
                    selected_action = policy_action
                    dls_duration_s = 0.0
                elif needs_dls:
                    if self._dls is None:
                        self._hold("dls_unavailable")
                        return
                    dls_start_s = perf_counter()
                    selected_action = self._dls.compute_action(
                        phase=active_phase,
                        phase_step=active_phase_step,
                        q=snapshot.q,
                        qvel=snapshot.qvel,
                        geometry=geometry_for_control,
                    )
                    dls_duration_s = perf_counter() - dls_start_s
                else:
                    selected_action = policy_action
                    dls_duration_s = 0.0
                if (
                    not self._autonomous_controller
                    and
                    self._approach_lock_wrist_home
                    and int(active_phase)
                    in (int(Phase.HOVER_RED), int(Phase.DESCEND_GRASP))
                ):
                    selected_action = lock_wrist_to_home(
                        selected_action, self._config
                    )
                if (
                    not self._autonomous_controller
                    and
                    int(active_phase) == int(Phase.CLOSE_GRIPPER)
                    and self._gripper.abort_close()
                ):
                    close_limit = self._config.phase_step_limits[
                        int(Phase.CLOSE_GRIPPER)
                    ]
                    self._session.phase_step = max(
                        self._session.phase_step, close_limit - 1
                    )
                alignment_cube = None
                require_hover_xy_clearance = bool(
                    self._dls_descend_target == "hover_estimate"
                    and int(active_phase) == int(Phase.HOVER_RED)
                )
                if approach_frozen or require_hover_xy_clearance:
                    if (
                        self._dls_descend_target == "sim_box"
                        and self._sim_box_position_m is not None
                    ):
                        alignment_cube = self._sim_box_position_m
                    elif self._dls_anchor_valid:
                        alignment_cube = self._dls_cube_anchor_m
                session_arguments = dict(
                    geometry=geometry_for_control,
                    tcp_position_m=snapshot.tcp_position_m,
                    tcp_rotation=snapshot.tcp_rotation,
                    joint_velocity=snapshot.qvel,
                    finger_span_m=snapshot.finger_span_m,
                    gripper_openness=openness,
                    grasp_latched_override=self._gripper.latch_override(),
                    require_grasp_alignment=approach_frozen,
                    require_hover_xy_clearance=require_hover_xy_clearance,
                    alignment_cube_m=alignment_cube,
                )
                if self._autonomous_controller:
                    session_arguments["mode_probs"] = mode_probs
                    if advance_prob is not None:
                        session_arguments["advance_prob"] = advance_prob
                result = self._session.step(**session_arguments)
                if result.grasp_retry:
                    self._dls_cube_anchor_m.fill(0.0)
                    self._dls_anchor_valid = False
                    if bool(
                        self.get_parameter("grasp_retry_recovery_enabled").value
                    ):
                        self._recovery.start(
                            snapshot.tcp_position_m,
                            estimate_generation=snapshot.frame_sequence,
                        )
                        self._recovery_step(snapshot)
                        return
                if hold_arm and self._last_published_command is not None:
                    integrated_command = self._last_published_command.copy()
                    command = integrated_command
                else:
                    integrated_command = self._integrator.step(
                        selected_action, snapshot.q
                    )
                    if (
                        not np.all(np.isfinite(integrated_command))
                        or integrated_command.shape != (7,)
                    ):
                        raise ValueError("integrator produced an invalid command")
                    reference = (
                        self._last_published_command
                        if self._last_published_command is not None
                        else snapshot.q
                    )
                    command = self._limit_command(integrated_command, reference)
                self._publish_command(command)
                self._gripper.command(
                    Phase(int(result.telemetry["active_phase"])),
                    result.gripper_half_width_target_m,
                )
                telemetry = {
                    **result.telemetry,
                    **build_command_telemetry(
                        snapshot=snapshot,
                        sim_timer_delta_s=self._timer_sim_delta_s,
                        wall_timer_delta_s=self._timer_wall_delta_s,
                        policy_step_delta_s=self._policy_step_delta_s,
                        control_callback_duration_s=(
                            perf_counter() - self._timer_callback_start_s
                        ),
                        inference_duration_s=inference_duration_s,
                        dls_duration_s=dls_duration_s,
                        raw_action=policy_action,
                        selected_action=selected_action,
                        integrated_target=self._integrator.integrated_target,
                        smoothed_target=integrated_command,
                        limited_command=command,
                        published_command=self._last_published_command,
                    ),
                    "policy_enabled": True,
                    "safe_hold": False,
                    "controller": (
                        "autonomous_onnx"
                        if self._autonomous_controller
                        else "hold"
                        if hold_arm
                        else "dls"
                        if needs_dls
                        else "onnx"
                    ),
                    "approach_lock_wrist_home": self._approach_lock_wrist_home,
                    "dls_descend_target": self._dls_descend_target,
                    "dls_anchor_enabled": self._dls_descend_target
                    in ("sim_box", "hover_estimate"),
                    "dls_anchor_valid": self._dls_anchor_valid,
                    "dls_cube_anchor_m": self._dls_cube_anchor_m.tolist(),
                    "sim_box_position_m": (
                        None
                        if self._sim_box_position_m is None
                        else self._sim_box_position_m.tolist()
                    ),
                    "approach_geometry_frozen": approach_frozen,
                    "geometry_for_control": geometry_for_control.tolist(),
                    "mode_probs": (
                        None if mode_probs is None else mode_probs.tolist()
                    ),
                    "action_latency_steps": (
                        self._integrator.action_latency_steps
                    ),
                    "response_time_s": self._config.response_time,
                    "command": command.tolist(),
                    "geometry": geometry.tolist(),
                    "policy_state": np.asarray(state).reshape(-1).tolist(),
                    "tcp_position_m": snapshot.tcp_position_m.tolist(),
                    "tcp_rotation": snapshot.tcp_rotation.tolist(),
                    "finger_span_m": snapshot.finger_span_m,
                    "gripper_half_width_target_m": (
                        result.gripper_half_width_target_m
                    ),
                    "gripper": self._gripper.telemetry(),
                }
                self._publish_telemetry(telemetry)
                if result.done:
                    self._faulted = True
                    if result.success:
                        self._last_fault = "session_success"
                    elif result.dropped:
                        self._last_fault = "session_dropped"
                    elif result.fatal_timeout:
                        self._last_fault = "session_timeout"
                    else:
                        self._last_fault = "session_terminated"
            except Exception as exc:  # Never publish a partially computed command.
                self._faulted = True
                self._last_fault = f"control_error:{exc}"
                self.get_logger().error(self._last_fault)
                self._hold(self._last_fault)

    return StudentV6PolicyNode


def main() -> None:
    try:
        import rclpy
    except ImportError as exc:
        raise SystemExit("rclpy is unavailable; source ROS 2 Humble first") from exc
    node_class = create_node_class()
    rclpy.init()
    node = node_class()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
