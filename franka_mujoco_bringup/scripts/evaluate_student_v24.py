#!/usr/bin/env python3
"""Run the ROS2 MuJoCo bridge on the direct evaluator's 16 reset keyframes."""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, String

from mujoco_ros2_control_msgs.msg import FreeJointStateArray
from mujoco_ros2_control_msgs.srv import ResetWorld


TERMINAL_REASONS = {
    "session_success",
    "session_dropped",
    "session_timeout",
    "session_terminated",
}
BOX_STATE_TOPIC = "/box/free_joint_state"
BOX_STATE_TOPIC_FALLBACK = (
    "/mujoco_ros2_control_node/free_joint_state_publisher/free_joint_states"
)
PLACEMENT_TOLERANCE_M = 0.02
SEATED_CUBE_HEIGHT_M = 0.025
PLACEMENT_HEIGHT_TOLERANCE_M = 0.02
GEOMETRY_SCALE_M = 0.75
PLACEMENT_GOAL_X_OFFSET_M = 0.004
# q + qfrc_bias / kp at the nominal keyframe.  This holds the reset pose only
# while the policy-enabled parameter service completes; the first policy frame
# replaces it with the ordinary integrated target.
PANDA_RESET_HOLD = [
    0.0,
    0.2603107783,
    0.0003729302,
    -1.5401124066,
    0.0021321548,
    2.0083561599,
    -0.7853965571,
]
DIRECT_CONTROLLER_EPISODES = {
    1: (3, 0.523421526),
    2: (3, 0.479208320),
    3: (2, 0.464596331),
    4: (1, 0.463780344),
    5: (0, 0.469860822),
    6: (0, 0.525307715),
    7: (0, 0.512966573),
    8: (3, 0.496446252),
    9: (2, 0.520569205),
    10: (1, 0.526497424),
    11: (0, 0.454095364),
    12: (3, 0.485066473),
    13: (1, 0.481360584),
    14: (2, 0.505119145),
    15: (2, 0.499364763),
    16: (0, 0.485091954),
}


def default_mjcf_path() -> Path:
    from ament_index_python.packages import get_package_share_directory

    return (
        Path(get_package_share_directory("franka_mujoco_bringup"))
        / "mjcf"
        / "panda_pick_place.xml"
    )


def parse_eval_seed_layout(xml_path: Path) -> dict[int, dict[str, tuple[float, ...]]]:
    root = ET.parse(xml_path).getroot()
    layout: dict[int, dict[str, tuple[float, ...]]] = {}
    for key in root.findall("./keyframe/key"):
        name = str(key.attrib.get("name", ""))
        if not name.startswith("eval_seed_"):
            continue
        seed = int(name.rsplit("_", 1)[1])
        qpos = tuple(float(value) for value in key.attrib["qpos"].split())
        mpos = tuple(float(value) for value in key.attrib["mpos"].split())
        layout[seed] = {
            "cube_spawn_m": qpos[9:12],
            "pad_m": mpos[:3],
        }
    if len(layout) != 16:
        raise ValueError(f"expected 16 eval_seed keyframes in {xml_path}, got {len(layout)}")
    return layout


def physical_place_success(
    box_xyz: tuple[float, ...] | list[float] | np.ndarray,
    pad_xyz: tuple[float, ...] | list[float] | np.ndarray,
    *,
    planar_tolerance_m: float = PLACEMENT_TOLERANCE_M,
    seated_height_m: float = SEATED_CUBE_HEIGHT_M,
    height_tolerance_m: float = PLACEMENT_HEIGHT_TOLERANCE_M,
) -> dict:
    box = np.asarray(box_xyz, dtype=np.float64)
    pad = np.asarray(pad_xyz, dtype=np.float64)
    if box.shape != (3,) or pad.shape != (3,):
        raise ValueError("box and pad positions must have shape (3,)")
    planar = float(np.linalg.norm(box[:2] - pad[:2]))
    height_error = float(abs(box[2] - seated_height_m))
    return {
        "box_position_m": box.tolist(),
        "pad_position_m": pad.tolist(),
        "box_pad_planar_m": planar,
        "box_height_error_m": height_error,
        "physical_success": bool(
            planar <= planar_tolerance_m and height_error <= height_tolerance_m
        ),
    }


class SeededEvaluator(Node):
    def __init__(
        self,
        policy_node: str,
        *,
        capture_pixels: bool = False,
        match_direct_controller_randomization: bool = False,
        seed_layout: dict[int, dict[str, tuple[float, ...]]] | None = None,
        box_state_topic: str = BOX_STATE_TOPIC,
        dls_descend_target: str = "sim_box",
    ) -> None:
        super().__init__("student_v24_seeded_evaluator")
        self.parameters = self.create_client(
            SetParameters, f"{policy_node}/set_parameters"
        )
        self.reset_client = self.create_client(
            ResetWorld, "/mujoco_ros2_control_node/reset_world"
        )
        self.arm_command = self.create_publisher(
            Float64MultiArray, "/panda_arm_controller/commands", 10
        )
        self.last_telemetry: dict | None = None
        self.terminal: dict | None = None
        self.trace: list[dict] = []
        self.capture_pixels = capture_pixels
        self.match_direct_controller_randomization = (
            match_direct_controller_randomization
        )
        self.dls_descend_target = dls_descend_target
        self.seed_layout = seed_layout or {}
        self.current_seed: int | None = None
        self.box_pose: tuple[float, float, float] | None = None
        self.box_samples: deque[tuple[float, np.ndarray]] = deque(maxlen=512)
        self.frames: dict[int, np.ndarray] = {}
        self.oak_diagnostics: list[dict] = []
        self.create_subscription(
            String, "/panda_student_v6/telemetry", self._on_telemetry, 100
        )
        self.create_subscription(
            String,
            "/policy_camera/oak_emulator/telemetry",
            self._on_oak_diagnostics,
            10,
        )
        self.create_subscription(
            FreeJointStateArray, box_state_topic, self._on_box_state, 50
        )
        if box_state_topic != BOX_STATE_TOPIC_FALLBACK:
            self.create_subscription(
                FreeJointStateArray,
                BOX_STATE_TOPIC_FALLBACK,
                self._on_box_state,
                50,
            )
        if capture_pixels:
            image_qos = QoSProfile(
                depth=1, reliability=ReliabilityPolicy.BEST_EFFORT
            )
            self.create_subscription(
                Image,
                "/policy_camera/color/image_raw",
                self._on_image,
                image_qos,
            )

    def _on_image(self, message: Image) -> None:
        if (
            int(message.width) != 64
            or int(message.height) != 64
            or str(message.encoding).lower() not in ("rgb8", "bgr8")
        ):
            return
        pixels = np.frombuffer(message.data, dtype=np.uint8).reshape(64, 64, 3)
        if str(message.encoding).lower() == "bgr8":
            pixels = pixels[..., ::-1]
        stamp_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(
            message.header.stamp.nanosec
        )
        self.frames[stamp_ns] = pixels.copy()
        while len(self.frames) > 256:
            self.frames.pop(next(iter(self.frames)))

    def _on_box_state(self, message: FreeJointStateArray) -> None:
        stamp_s = (
            float(message.header.stamp.sec)
            + float(message.header.stamp.nanosec) * 1.0e-9
        )
        for entry in message.free_joints:
            if entry.name != "box":
                continue
            position = entry.pose.pose.position
            self.box_pose = (float(position.x), float(position.y), float(position.z))
            if stamp_s > 0.0:
                self.box_samples.append(
                    (stamp_s, np.asarray(self.box_pose, dtype=np.float64))
                )
            return

    def _box_at(self, image_stamp_s: float) -> tuple[np.ndarray | None, float | None]:
        if not self.box_samples:
            return None, None
        stamp, position = min(
            self.box_samples,
            key=lambda item: abs(item[0] - image_stamp_s),
        )
        return position.copy(), abs(stamp - image_stamp_s)

    def _on_telemetry(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            return
        image_stamp_s = payload.get("image_stamp_s")
        if self.capture_pixels and image_stamp_s is not None:
            stamp_ns = int(round(float(image_stamp_s) * 1.0e9))
            pixels = self.frames.pop(stamp_ns, None)
            if pixels is not None:
                payload["pixels"] = pixels
        if image_stamp_s is not None:
            box, box_skew_s = self._box_at(float(image_stamp_s))
            layout = self.seed_layout.get(self.current_seed or -1, {})
            pad = layout.get("pad_m")
            if box is not None and pad is not None:
                goal = np.asarray(
                    (
                        float(pad[0]) + PLACEMENT_GOAL_X_OFFSET_M,
                        float(pad[1]),
                        SEATED_CUBE_HEIGHT_M,
                    ),
                    dtype=np.float64,
                )
                tcp = np.asarray(payload.get("tcp_position_m"), dtype=np.float64)
                if tcp.shape == (3,) and np.all(np.isfinite(tcp)):
                    teacher_geometry = np.concatenate(
                        ((box - tcp), (goal - box))
                    ) / GEOMETRY_SCALE_M
                    payload["sim_box_position_m"] = box.tolist()
                    payload["sim_goal_position_m"] = goal.tolist()
                    payload["sim_box_image_skew_s"] = float(box_skew_s)
                    payload["teacher_geometry"] = teacher_geometry.tolist()
        gripper = payload.get("gripper")
        if isinstance(gripper, dict):
            payload["gripper_last_result_stalled"] = bool(
                gripper.get("last_result_stalled", False)
            )
            payload["gripper_backend_latched"] = bool(
                gripper.get("latched", False)
            )
        self.last_telemetry = payload
        self.trace.append(payload)
        if payload.get("reason") in TERMINAL_REASONS:
            self.terminal = payload

    def _on_oak_diagnostics(self, message: String) -> None:
        try:
            self.oak_diagnostics.append(json.loads(message.data))
        except json.JSONDecodeError:
            return

    def wait_future(self, future, timeout: float = 10.0):
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if not future.done():
            raise TimeoutError("ROS service/parameter call timed out")
        result = future.result()
        if result is None:
            raise RuntimeError("ROS service/parameter call returned no result")
        return result

    def set_enabled(self, enabled: bool) -> None:
        if not self.parameters.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("student policy parameter service unavailable")
        request = SetParameters.Request()
        request.parameters = [
            Parameter(
                name="policy_enabled",
                value=ParameterValue(
                    type=ParameterType.PARAMETER_BOOL,
                    bool_value=enabled,
                ),
            )
        ]
        response = self.wait_future(
            self.parameters.call_async(request)
        )
        if not all(result.successful for result in response.results):
            raise RuntimeError(
                f"failed to set policy_enabled={enabled}: {response.results}"
            )

    def set_controller_episode(self, seed: int) -> None:
        latency, response_time = (
            DIRECT_CONTROLLER_EPISODES.get(seed, (1, 0.5))
            if self.match_direct_controller_randomization
            else (0, 0.5)
        )
        request = SetParameters.Request()
        request.parameters = [
            Parameter(
                name="action_latency_steps",
                value=ParameterValue(
                    type=ParameterType.PARAMETER_INTEGER,
                    integer_value=latency,
                ),
            ),
            Parameter(
                name="response_time_s",
                value=ParameterValue(
                    type=ParameterType.PARAMETER_DOUBLE,
                    double_value=response_time,
                ),
            ),
            Parameter(
                name="dls_descend_target",
                value=ParameterValue(
                    type=ParameterType.PARAMETER_STRING,
                    string_value=self.dls_descend_target,
                ),
            ),
        ]
        response = self.wait_future(self.parameters.call_async(request))
        if not all(result.successful for result in response.results):
            raise RuntimeError(
                f"failed to set controller episode {seed}: {response.results}"
            )

    def reset_seed(self, seed: int) -> None:
        if not self.reset_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("MuJoCo reset_world service unavailable")
        request = ResetWorld.Request()
        request.keyframe = f"eval_seed_{seed}"
        response = self.wait_future(self.reset_client.call_async(request))
        if not response.success:
            raise RuntimeError(f"reset seed {seed} failed: {response.message}")

    def spin_for(self, duration: float) -> None:
        stop = time.monotonic() + duration
        while time.monotonic() < stop:
            rclpy.spin_once(self, timeout_sec=min(0.05, stop - time.monotonic()))

    def run_seed(self, seed: int, timeout: float) -> dict:
        self.terminal = None
        self.trace = []
        self.frames.clear()
        self.oak_diagnostics = []
        self.box_samples.clear()
        self.box_pose = None
        self.current_seed = seed
        self.set_enabled(False)
        self.set_controller_episode(seed)
        self.reset_seed(seed)
        self.arm_command.publish(Float64MultiArray(data=PANDA_RESET_HOLD))
        self.spin_for(0.2)
        self.set_enabled(True)
        started = time.monotonic()
        while self.terminal is None and time.monotonic() - started < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.set_enabled(False)
        self.spin_for(0.2)
        result = dict(self.terminal or self.last_telemetry or {})
        estimator_success = result.get("reason") == "session_success"
        layout = self.seed_layout.get(seed, {})
        physical = {
            "box_position_m": None,
            "pad_position_m": list(layout["pad_m"]) if "pad_m" in layout else None,
            "box_pad_planar_m": None,
            "box_height_error_m": None,
            "box_pose_observed": self.box_pose is not None,
            "physical_success": False,
        }
        if self.box_pose is not None and "pad_m" in layout:
            physical.update(physical_place_success(self.box_pose, layout["pad_m"]))
            physical["box_pose_observed"] = True
        result.update(
            {
                "seed": seed,
                "wall_duration_s": time.monotonic() - started,
                "terminal_observed": self.terminal is not None,
                "estimator_success": estimator_success,
                "success": bool(physical["physical_success"]),
                **physical,
            }
        )
        result["smoothness"] = summarize_trace(self.trace)
        result["oak_emulator"] = (
            self.oak_diagnostics[-1] if self.oak_diagnostics else None
        )
        result["trace"] = list(self.trace)
        return result


def summarize_trace(trace: list[dict]) -> dict:
    active = [
        item
        for item in trace
        if not item.get("safe_hold")
        and isinstance(item.get("published_command"), list)
        and len(item["published_command"]) == 7
    ]
    if len(active) < 2:
        return {
            "samples": len(active),
            "limiter_interventions": 0,
            "missed_frames": 0,
        }
    command = np.asarray(
        [item["published_command"] for item in active], dtype=np.float64
    )
    dt = np.asarray(
        [
            float(item.get("sim_timer_delta_s") or (1.0 / 60.0))
            if item.get("policy_step_delta_s") is None
            else float(item["policy_step_delta_s"])
            for item in active
        ],
        dtype=np.float64,
    )
    velocity = np.diff(command, axis=0) / np.maximum(dt[1:, None], 1.0e-9)
    acceleration = np.diff(velocity, axis=0) / np.maximum(
        dt[2:, None], 1.0e-9
    )
    jerk = np.diff(acceleration, axis=0) / np.maximum(
        dt[3:, None], 1.0e-9
    )
    tracking = np.asarray(
        [item.get("tracking_error", [0.0] * 7) for item in active],
        dtype=np.float64,
    )
    return {
        "samples": len(active),
        "command_velocity_rms_rad_s": float(np.sqrt(np.mean(velocity**2))),
        "command_acceleration_rms_rad_s2": (
            float(np.sqrt(np.mean(acceleration**2)))
            if acceleration.size
            else 0.0
        ),
        "command_jerk_rms_rad_s3": (
            float(np.sqrt(np.mean(jerk**2))) if jerk.size else 0.0
        ),
        "command_jerk_max_rad_s3": (
            float(np.max(np.abs(jerk))) if jerk.size else 0.0
        ),
        "tracking_error_rms_rad": float(np.sqrt(np.mean(tracking**2))),
        "tracking_error_max_rad": float(np.max(np.abs(tracking))),
        "limiter_interventions": sum(
            int(bool(item.get("limiter_intervened"))) for item in active
        ),
        "missed_frames": sum(
            int(item.get("missed_frame_count", 0)) for item in active
        ),
        "max_input_skew_s": max(
            float(item.get("max_input_skew_s", 0.0)) for item in active
        ),
    }


def write_npz_trace(path: Path, trace: list[dict], seed: int) -> None:
    records = [
        item
        for item in trace
        if "pixels" in item and "policy_state" in item
    ]
    fields = (
        "frame_sequence",
        "image_stamp_s",
        "image_age_s",
        "missed_frame_count",
        "reused_frame",
        "pixels",
        "policy_state",
        "raw_action",
        "selected_action",
        "geometry",
        "integrated_target",
        "smoothed_target",
        "limited_command",
        "published_command",
        "q",
        "qvel",
        "tcp_position_m",
        "tcp_rotation",
        "finger_span_m",
        "active_phase",
        "active_phase_step",
        "phase",
        "phase_step",
        "controller",
        "sim_box_position_m",
        "sim_goal_position_m",
        "sim_box_image_skew_s",
        "teacher_geometry",
        "grasp_retries",
        "placement_retries",
        "grasp_latched",
        "gripper_last_result_stalled",
        "gripper_backend_latched",
    )
    arrays = {
        field: np.asarray([item[field] for item in records])
        for field in fields
        if records and all(field in item for item in records)
    }
    if records:
        aliases = {
            "student_pixels": "pixels",
            "student_state": "policy_state",
            "student_geometry": "geometry",
            "phase_index": "active_phase",
        }
        for destination, source in aliases.items():
            if source in arrays:
                arrays[destination] = arrays[source]
        arrays["phase_step_index"] = arrays.get(
            "phase_step",
            np.zeros((len(records),), dtype=np.int32),
        )
        arrays["frame_index"] = np.arange(len(records), dtype=np.int32)
        arrays["seed"] = np.full((len(records),), seed, dtype=np.int32)
    metadata = {
        "schema": "student-v24-ros2-policy-trace-v1",
        "seed": seed,
        "step_count": len(records),
        "fields": sorted(arrays),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--policy-node", default="/student_v6_policy")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-directory", type=Path)
    parser.add_argument("--npz-trace-directory", type=Path)
    parser.add_argument(
        "--match-direct-controller-randomization",
        action="store_true",
        help="Replay per-seed MJX latency/response values for diagnostics.",
    )
    parser.add_argument("--mjcf", type=Path, default=None)
    parser.add_argument("--box-state-topic", default=BOX_STATE_TOPIC)
    parser.add_argument(
        "--dls-descend-target",
        choices=("live", "hover_estimate", "sim_box"),
        default="sim_box",
    )
    args = parser.parse_args()
    seeds = args.seed or list(range(1, 17))
    layout = parse_eval_seed_layout(args.mjcf or default_mjcf_path())

    rclpy.init()
    evaluator = SeededEvaluator(
        args.policy_node,
        capture_pixels=args.npz_trace_directory is not None,
        match_direct_controller_randomization=(
            args.match_direct_controller_randomization
        ),
        seed_layout=layout,
        box_state_topic=args.box_state_topic,
        dls_descend_target=args.dls_descend_target,
    )
    try:
        episodes = []
        for seed in seeds:
            result = evaluator.run_seed(seed, args.timeout)
            trace = result.pop("trace")
            if args.npz_trace_directory is not None:
                write_npz_trace(
                    args.npz_trace_directory / f"seed_{seed}.npz",
                    trace,
                    seed,
                )
            if args.trace_directory is not None:
                args.trace_directory.mkdir(parents=True, exist_ok=True)
                trace_path = args.trace_directory / f"seed_{seed}.jsonl"
                trace_path.write_text(
                    "".join(
                        json.dumps(
                            {
                                key: value
                                for key, value in item.items()
                                if key != "pixels"
                            }
                        )
                        + "\n"
                        for item in trace
                    )
                )
            episodes.append(result)
            print(
                f"seed={seed} estimator={int(result['estimator_success'])} "
                f"physical={int(result['success'])} "
                f"planar={result.get('box_pad_planar_m')} "
                f"reason={result.get('reason', 'timeout')}"
            )
        report = {
            "schema": "student-v24-ros2-seeded-evaluation",
            "version": 2,
            "episodes": episodes,
            "estimator_successes": sum(
                int(item["estimator_success"]) for item in episodes
            ),
            "physical_successes": sum(int(item["success"]) for item in episodes),
            "successes": sum(int(item["success"]) for item in episodes),
            "episode_count": len(episodes),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        try:
            evaluator.set_enabled(False)
        except Exception:
            pass
        evaluator.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
