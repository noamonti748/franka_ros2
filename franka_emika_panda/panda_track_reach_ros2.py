#!/usr/bin/env python3
"""
Panda multi-target reach-track behaviour (Python port of APandaTrackReachEnvironment).

Mirrors the Unreal/Schola task:
  - Sample a random reach target around a spawn center each subgoal.
  - On success (TCP within threshold), resample the target in place (no homing reset).
  - If the arm does not reach the current target within SubgoalStepLimit steps, fail the episode.
  - Long episodes are truncated at EpisodeStepLimit (default 10000).

The core logic lives in PandaTrackReachSession and is ROS-free so you can unit-test it.
Run as a ROS 2 node when rclpy is available:

    ros2 run ...  # or:
    python Scripts/panda_track_reach_ros2.py --ros

Dry-run without ROS (prints a short simulated episode):

    python Scripts/panda_track_reach_ros2.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

PANDA_ARM_JOINTS: tuple[str, ...] = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
)
PANDA_ROS_JOINTS: tuple[str, ...] = tuple(f"panda_{name}" for name in PANDA_ARM_JOINTS)
PANDA_HOME_JOINTS: tuple[float, ...] = (
    0.0,
    0.0,
    0.0,
    -math.pi / 2.0,
    0.0,
    math.pi / 2.0,
    -math.pi / 4.0,
)
PANDA_JOINT_LIMITS: tuple[tuple[float, float], ...] = (
    (-2.8973, 2.8973),
    (-1.7628, 1.7628),
    (-2.8973, 2.8973),
    (-3.0718, -0.0698),
    (-2.8973, 2.8973),
    (-0.0175, 3.7525),
    (-2.8973, 2.8973),
)


@dataclass
class PandaTrackReachConfig:
    """Defaults aligned with APandaTrackReachEnvironment + APandaReachEnvironment."""

    # ----- Target (ReachEnvironment) -----
    target_location_m: tuple[float, float, float] = (0.5, 0.0, 0.4)
    randomize_target_per_episode: bool = True
    # X/Y: U[-max, +max], Z: U[0, +max]
    target_random_offset_max_m: tuple[float, float, float] = (0.1, 0.3, 0.2)

    # ----- Episode / track -----
    episode_step_limit: int = 10_000
    subgoal_step_limit: int = 500

    # ----- Success (Track overrides Panda hold velocity gate) -----
    success_threshold_m: float = 0.075
    success_max_joint_vel_rad_s: float = 0.0  # <= 0 => position only
    success_bonus: float = 50.0
    settled_dwell_required_steps: int = 0

    # ----- Reward shaping (ReachEnvironment defaults + Panda tweaks) -----
    distance_scale: float = 1.0
    step_penalty: float = -0.01
    close_region_meters: float = 0.1
    close_region_bonus_scale: float = 0.2
    out_of_bounds_distance_m: float = 1.5
    out_of_bounds_penalty: float = -5.0
    use_out_of_bounds: bool = True
    terminate_on_out_of_bounds: bool = True
    action_rate_penalty_scale: float = 0.085
    joint_velocity_penalty_scale: float = 0.03

    # ----- Panda hold / smoothing (still active under Track) -----
    hold_still_region_meters: float = 0.12
    hold_still_bonus_scale: float = 0.04
    hold_still_max_joint_vel_squared: float = 0.6
    settled_hold_action_rate_penalty_scale: float = 0.036
    settled_hold_action_sq_penalty_scale: float = 0.00225
    near_goal_smoothing_region_meters: float = 0.12
    near_goal_action_rate_penalty_scale: float = 0.06
    near_goal_joint_velocity_penalty_scale: float = 0.036
    near_goal_joint_acceleration_penalty_scale: float = 0.0

    # ----- Action smoothing (Track ctor) -----
    smooth_actions: bool = True
    action_smoothing_alpha: float = 0.05
    limit_action_delta: bool = True
    action_rate_limit_hz: float = 60.0
    action_velocity_limits_rad_s: tuple[float, ...] = (
        1.20,
        1.45,
        1.45,
        1.45,
        1.20,
        1.55,
        1.55,
    )
    use_measured_velocity_governor: bool = True
    measured_velocity_governor_start_ratio: float = 0.70

    joint_names: tuple[str, ...] = PANDA_ARM_JOINTS
    random_seed: int = 0


@dataclass
class StepOutcome:
    reward: float
    terminated: bool
    truncated: bool
    success: bool
    distance_m: float
    target_m: tuple[float, float, float]
    subgoal_elapsed: int
    info: dict[str, str] = field(default_factory=dict)


class OnnxPolicyRunner:
    """Thin ONNX Runtime adapter for the Schola/SB3 exported Panda policy."""

    def __init__(self, model_path: str) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required to run ppo_track_franka.onnx. "
                "Install it in the active Python environment, for example: "
                "python3 -m pip install --user onnxruntime"
            ) from exc

        self.model_path = str(Path(model_path).expanduser())
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        self._session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        self.input_names = [inp.name for inp in self._session.get_inputs()]
        self.output_names = [out.name for out in self._session.get_outputs()]

        missing_inputs = [name for name in (*PANDA_ARM_JOINTS, "tcp_pos") if name not in self.input_names]
        missing_outputs = [f"actuator{i}" for i in range(1, 8) if f"actuator{i}" not in self.output_names]
        if missing_inputs or missing_outputs:
            raise RuntimeError(
                "Unexpected ONNX policy interface. "
                f"missing_inputs={missing_inputs}, missing_outputs={missing_outputs}, "
                f"inputs={self.input_names}, outputs={self.output_names}"
            )

    @staticmethod
    def _tcp_observation(
        mode: str,
        tcp_m: np.ndarray,
        target_m: np.ndarray,
        scale: float,
        signs: Sequence[float] = (1.0, 1.0, 1.0),
        unit_scale: float = 1.0,
    ) -> np.ndarray:
        if scale <= 0.0:
            raise ValueError(f"tcp_observation_scale must be positive, got {scale}")
        sign_vec = _vec3(signs).astype(np.float32)
        normalized_scale = float(unit_scale) / float(scale)
        if mode in ("tcp_error_normalized", "tcp_minus_target_normalized"):
            return (tcp_m - target_m) * normalized_scale * sign_vec
        if mode in ("target_error_normalized", "target_minus_tcp_normalized"):
            return (target_m - tcp_m) * normalized_scale * sign_vec
        if mode == "tcp_normalized":
            return tcp_m * normalized_scale * sign_vec
        if mode == "target_normalized":
            return target_m * normalized_scale * sign_vec
        if mode == "tcp_minus_target":
            return (tcp_m - target_m) * sign_vec
        if mode == "target_minus_tcp":
            return (target_m - tcp_m) * sign_vec
        if mode == "target_delta":
            return (target_m - tcp_m) * sign_vec
        if mode == "tcp":
            return tcp_m * sign_vec
        if mode == "target":
            return target_m * sign_vec
        raise ValueError(
            f"Unsupported tcp_observation_mode={mode!r}. "
            "Use one of: tcp_error_normalized, target_error_normalized, "
            "tcp_minus_target, target_minus_tcp, target_delta, tcp, target."
        )

    @staticmethod
    def _joint_observation(
        mode: str,
        position: float,
        velocity: float,
        command: float,
    ) -> np.ndarray:
        if mode == "position_velocity_command":
            return np.array([position, velocity, command], dtype=np.float32)
        if mode == "position_velocity_zero":
            return np.array([position, velocity, 0.0], dtype=np.float32)
        if mode == "position_velocity_error":
            return np.array([position, velocity, command - position], dtype=np.float32)
        raise ValueError(
            f"Unsupported joint_observation_mode={mode!r}. "
            "Use one of: position_velocity_command, position_velocity_zero, "
            "position_velocity_error."
        )

    def compute_action(
        self,
        joint_pos: Sequence[float],
        joint_vel: Sequence[float],
        last_command: Sequence[float],
        tcp_m: Sequence[float],
        target_m: Sequence[float],
        *,
        tcp_observation_mode: str = "tcp_error_normalized",
        tcp_observation_scale: float = 1.0,
        tcp_observation_signs: Sequence[float] = (1.0, 1.0, 1.0),
        tcp_observation_unit_scale: float = 1.0,
        joint_observation_mode: str = "position_velocity_command",
        action_scale: float = 1.0,
    ) -> list[float]:
        if len(joint_pos) != 7 or len(joint_vel) != 7 or len(last_command) != 7:
            raise ValueError("joint_pos, joint_vel, and last_command must all have length 7")

        feed: dict[str, np.ndarray] = {}
        for i, joint_name in enumerate(PANDA_ARM_JOINTS):
            feed[joint_name] = self._joint_observation(
                joint_observation_mode,
                float(joint_pos[i]),
                float(joint_vel[i]),
                float(last_command[i]),
            ).reshape(1, 3)

        tcp_vec = self._tcp_observation(
            tcp_observation_mode,
            _vec3(tcp_m).astype(np.float32),
            _vec3(target_m).astype(np.float32),
            float(tcp_observation_scale),
            tcp_observation_signs,
            float(tcp_observation_unit_scale),
        )
        feed["tcp_pos"] = tcp_vec.astype(np.float32).reshape(1, 3)

        raw_outputs = self._session.run([f"actuator{i}" for i in range(1, 8)], feed)
        actions = [float(np.asarray(output).reshape(-1)[0]) * action_scale for output in raw_outputs]
        return actions


def _vec3(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"expected 3-vector, got shape {arr.shape}")
    return arr


def _parse_vec3_text(text: str) -> tuple[float, float, float]:
    values = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
    if len(values) == 1 and " " in values[0]:
        values = [part for part in values[0].split() if part]
    if len(values) != 3:
        raise ValueError(f"expected three comma-separated values, got {text!r}")
    return tuple(float(value) for value in values)


def sample_target_offset(
    rng: np.random.Generator,
    offset_max_m: tuple[float, float, float],
) -> np.ndarray:
    """Match AReachEnvironment::SampleEpisodeTargetLocation offset draws."""

    def symmetric(max_offset: float) -> float:
        if max_offset <= 0.0:
            return 0.0
        return float(rng.uniform(-max_offset, max_offset))

    def positive(max_offset: float) -> float:
        if max_offset <= 0.0:
            return 0.0
        return float(rng.uniform(0.0, max_offset))

    return np.array(
        [
            symmetric(offset_max_m[0]),
            symmetric(offset_max_m[1]),
            positive(offset_max_m[2]),
        ],
        dtype=np.float64,
    )


class PandaTrackReachSession:
    """
    Stateful reach-track session. Call reset() once per RL episode, then step() each control tick.

    Positions are expressed in the task frame (env-local MuJoCo meters: X forward, Y left, Z up).
    For ROS, set frame_id on published poses to your robot base (e.g. panda_link0).
    """

    def __init__(self, config: Optional[PandaTrackReachConfig] = None) -> None:
        self.config = config or PandaTrackReachConfig()
        self._rng = np.random.default_rng(self.config.random_seed or None)
        self._target_counter = 0
        self.episode_step_index = 0
        self.subgoal_start_episode_step_index = 0
        self.last_successful_subgoal_episode_step_index = -1
        self.episode_target_location_m = np.array(self.config.target_location_m, dtype=np.float64)
        self.settled_dwell_counter = 0
        self.previous_action_vector: list[float] = []
        self.last_smoothed_action_vector: list[float] = []

    def seed(self, seed: int) -> None:
        self.config.random_seed = seed
        self._rng = np.random.default_rng(seed if seed != 0 else None)
        self._target_counter = 0

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
        target = target + sample_target_offset(local_rng, cfg.target_random_offset_max_m)
        self.episode_target_location_m = target
        return target.copy()

    def reset(self) -> np.ndarray:
        self.episode_step_index = 0
        self.subgoal_start_episode_step_index = 0
        self.last_successful_subgoal_episode_step_index = -1
        self.settled_dwell_counter = 0
        self.previous_action_vector = []
        self.last_smoothed_action_vector = []
        return self.sample_episode_target_location()

    def advance_to_next_subgoal(self, completed_episode_step_index: int) -> np.ndarray:
        target = self.sample_episode_target_location()
        self.settled_dwell_counter = 0
        self.previous_action_vector = []
        self.subgoal_start_episode_step_index = completed_episode_step_index
        self.last_successful_subgoal_episode_step_index = completed_episode_step_index
        return target

    @staticmethod
    def distance_m(tcp_m: Sequence[float], target_m: Sequence[float]) -> float:
        return float(np.linalg.norm(_vec3(tcp_m) - _vec3(target_m)))

    def joint_velocity_squared_sum(self, joint_vel_rad_s: Mapping[str, float]) -> float:
        total = 0.0
        for name in self.config.joint_names:
            vel = joint_vel_rad_s.get(name)
            if vel is not None:
                total += vel * vel
        return total

    def joint_acceleration_squared_sum(
        self, joint_acc_rad_s2: Mapping[str, float]
    ) -> float:
        total = 0.0
        for name in self.config.joint_names:
            acc = joint_acc_rad_s2.get(name)
            if acc is not None:
                total += acc * acc
        return total

    def is_success_reached(
        self,
        distance_m: float,
        joint_vel_rad_s: Optional[Mapping[str, float]] = None,
    ) -> bool:
        cfg = self.config
        if distance_m >= cfg.success_threshold_m:
            return False
        if cfg.success_max_joint_vel_rad_s > 0.0 and joint_vel_rad_s is not None:
            max_vel_sq = cfg.success_max_joint_vel_rad_s ** 2
            if self.joint_velocity_squared_sum(joint_vel_rad_s) > max_vel_sq:
                return False
        return True

    def smooth_actions(
        self,
        raw_actions: Sequence[float],
        measured_joint_pos: Optional[Sequence[float]] = None,
        measured_joint_vel: Optional[Sequence[float]] = None,
    ) -> list[float]:
        """Port of ReachEnvironment::Step action filter (low-pass + rate limit + governor)."""
        cfg = self.config
        n = len(cfg.joint_names)
        if len(raw_actions) != n:
            raise ValueError(f"expected {n} actions, got {len(raw_actions)}")

        had_history = len(self.last_smoothed_action_vector) == n
        if not had_history:
            self.last_smoothed_action_vector = list(raw_actions)
            if measured_joint_pos is not None and len(measured_joint_pos) == n:
                self.last_smoothed_action_vector = list(measured_joint_pos)

        alpha = float(np.clip(cfg.action_smoothing_alpha, 0.01, 1.0))
        filtered: list[float] = []
        for i, raw in enumerate(raw_actions):
            previous = self.last_smoothed_action_vector[i] if had_history else raw
            if not had_history and measured_joint_pos is not None and i < len(measured_joint_pos):
                previous = measured_joint_pos[i]

            value = (
                previous + alpha * (raw - previous)
                if cfg.smooth_actions and alpha < 1.0
                else raw
            )

            if cfg.limit_action_delta and i < len(cfg.action_velocity_limits_rad_s):
                limit = cfg.action_velocity_limits_rad_s[i]
                if limit > 0.0 and cfg.action_rate_limit_hz > 0.0:
                    max_delta = limit / cfg.action_rate_limit_hz
                    value = float(np.clip(value, previous - max_delta, previous + max_delta))

                    if (
                        cfg.use_measured_velocity_governor
                        and measured_joint_vel is not None
                        and i < len(measured_joint_vel)
                    ):
                        qvel = measured_joint_vel[i]
                        threshold = cfg.measured_velocity_governor_start_ratio * limit
                        if abs(qvel) > threshold:
                            delta = value - previous
                            if (delta > 0.0 and qvel > 0.0) or (delta < 0.0 and qvel < 0.0):
                                value = previous

            filtered.append(value)

        self.last_smoothed_action_vector = filtered
        return filtered

    def compute_reward(
        self,
        tcp_m: Sequence[float],
        joint_vel_rad_s: Mapping[str, float],
        actions_applied: Sequence[float],
        joint_acc_rad_s2: Optional[Mapping[str, float]] = None,
    ) -> float:
        """Reward for the current step (PandaReach + Track success bonus semantics)."""
        cfg = self.config
        target = self.episode_target_location_m
        distance = self.distance_m(tcp_m, target)
        prior_actions = list(self.previous_action_vector)

        reward = cfg.step_penalty - (cfg.distance_scale * distance)

        if cfg.close_region_bonus_scale > 0.0 and cfg.close_region_meters > 0.0:
            if distance < cfg.close_region_meters:
                ramp = 1.0 - (distance / cfg.close_region_meters)
                reward += cfg.close_region_bonus_scale * ramp

        success = self.is_success_reached(distance, joint_vel_rad_s)
        if success:
            reward += cfg.success_bonus
        elif cfg.use_out_of_bounds and distance > cfg.out_of_bounds_distance_m:
            reward += cfg.out_of_bounds_penalty

        if (
            cfg.settled_dwell_required_steps > 0
            and cfg.success_bonus > 0.0
            and success
        ):
            self.settled_dwell_counter += 1
            if self.settled_dwell_counter < cfg.settled_dwell_required_steps:
                reward -= cfg.success_bonus
        elif not success:
            self.settled_dwell_counter = 0

        if cfg.action_rate_penalty_scale > 0.0 and len(actions_applied) == len(cfg.joint_names):
            if len(prior_actions) == len(actions_applied):
                squared_delta = sum(
                    (actions_applied[i] - prior_actions[i]) ** 2
                    for i in range(len(actions_applied))
                )
                reward -= cfg.action_rate_penalty_scale * squared_delta
            self.previous_action_vector = list(actions_applied)

        if cfg.joint_velocity_penalty_scale > 0.0:
            reward -= cfg.joint_velocity_penalty_scale * self.joint_velocity_squared_sum(
                joint_vel_rad_s
            )

        if (
            cfg.hold_still_region_meters > 0.0
            and cfg.hold_still_bonus_scale > 0.0
            and distance < cfg.hold_still_region_meters
        ):
            joint_vel_sq = self.joint_velocity_squared_sum(joint_vel_rad_s)
            dist_ramp = 1.0 - (distance / cfg.hold_still_region_meters)
            vel_denom = max(cfg.hold_still_max_joint_vel_squared, 1e-8)
            vel_ramp = float(np.clip(1.0 - (joint_vel_sq / vel_denom), 0.0, 1.0))
            reward += cfg.hold_still_bonus_scale * dist_ramp * vel_ramp

        if success:
            current = list(actions_applied)
            if (
                cfg.settled_hold_action_rate_penalty_scale > 0.0
                and len(prior_actions) == len(current) == len(cfg.joint_names)
            ):
                squared_delta = sum(
                    (current[i] - prior_actions[i]) ** 2 for i in range(len(current))
                )
                reward -= cfg.settled_hold_action_rate_penalty_scale * squared_delta
            if cfg.settled_hold_action_sq_penalty_scale > 0.0:
                reward -= cfg.settled_hold_action_sq_penalty_scale * sum(a * a for a in current)

        if (
            cfg.near_goal_smoothing_region_meters > 0.0
            and distance < cfg.near_goal_smoothing_region_meters
        ):
            dist_ramp = 1.0 - (distance / cfg.near_goal_smoothing_region_meters)
            weight = dist_ramp
            if (
                cfg.near_goal_action_rate_penalty_scale > 0.0
                and len(prior_actions) == len(actions_applied) == len(cfg.joint_names)
            ):
                squared_delta = sum(
                    (actions_applied[i] - prior_actions[i]) ** 2
                    for i in range(len(actions_applied))
                )
                reward -= cfg.near_goal_action_rate_penalty_scale * weight * squared_delta
            if cfg.near_goal_joint_velocity_penalty_scale > 0.0:
                reward -= (
                    cfg.near_goal_joint_velocity_penalty_scale
                    * weight
                    * self.joint_velocity_squared_sum(joint_vel_rad_s)
                )
            if (
                cfg.near_goal_joint_acceleration_penalty_scale > 0.0
                and joint_acc_rad_s2 is not None
            ):
                reward -= (
                    cfg.near_goal_joint_acceleration_penalty_scale
                    * weight
                    * self.joint_acceleration_squared_sum(joint_acc_rad_s2)
                )

        return reward

    def _build_info(
        self,
        distance_m: float,
        joint_vel_rad_s: Mapping[str, float],
        subgoal_elapsed: int,
        *,
        success: bool,
    ) -> dict[str, str]:
        cfg = self.config
        return {
            "distance_m": f"{distance_m:.5f}",
            "target_m": str(tuple(float(x) for x in self.episode_target_location_m)),
            "target_local_m": str(tuple(float(x) for x in self.episode_target_location_m)),
            "tcp_m": "(see caller)",
            "episode_step": str(self.episode_step_index),
            "success": "true" if success else "false",
            "subgoal_elapsed": str(subgoal_elapsed),
            "subgoal_limit": str(cfg.subgoal_step_limit),
            "subgoal_anchor": str(self.subgoal_start_episode_step_index),
            "subgoal_last_success_step": str(self.last_successful_subgoal_episode_step_index),
            "joint_vel_sq": f"{self.joint_velocity_squared_sum(joint_vel_rad_s):.5f}",
        }

    def step(
        self,
        tcp_m: Sequence[float],
        joint_vel_rad_s: Optional[Mapping[str, float]] = None,
        actions_applied: Optional[Sequence[float]] = None,
        joint_acc_rad_s2: Optional[Mapping[str, float]] = None,
    ) -> StepOutcome:
        """
        Advance one environment step: reward, termination, and optional subgoal handoff.

        Matches APandaTrackReachEnvironment::ComputeTermination_Implementation ordering.
        """
        cfg = self.config
        joint_vel_rad_s = joint_vel_rad_s or {}
        actions_applied = actions_applied or []

        distance = self.distance_m(tcp_m, self.episode_target_location_m)
        reward = self.compute_reward(
            tcp_m, joint_vel_rad_s, actions_applied, joint_acc_rad_s2
        )

        terminated = False
        truncated = False
        success = self.is_success_reached(distance, joint_vel_rad_s)

        if cfg.use_out_of_bounds and cfg.terminate_on_out_of_bounds:
            if distance > cfg.out_of_bounds_distance_m:
                terminated = True

        subgoal_elapsed = self.episode_step_index - self.subgoal_start_episode_step_index

        if not terminated and success:
            self.advance_to_next_subgoal(self.episode_step_index)
            info = self._build_info(distance, joint_vel_rad_s, subgoal_elapsed, success=True)
            info["tcp_m"] = str(tuple(float(x) for x in _vec3(tcp_m)))
            self.episode_step_index += 1
            return StepOutcome(
                reward=reward,
                terminated=False,
                truncated=False,
                success=True,
                distance_m=distance,
                target_m=tuple(float(x) for x in self.episode_target_location_m),
                subgoal_elapsed=subgoal_elapsed,
                info=info,
            )

        if not terminated and cfg.subgoal_step_limit > 0:
            if subgoal_elapsed > cfg.subgoal_step_limit:
                terminated = True

        if not terminated and self.episode_step_index >= cfg.episode_step_limit:
            truncated = True

        info = self._build_info(
            distance,
            joint_vel_rad_s,
            subgoal_elapsed,
            success=success,
        )
        info["tcp_m"] = str(tuple(float(x) for x in _vec3(tcp_m)))

        self.episode_step_index += 1
        return StepOutcome(
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            success=success,
            distance_m=distance,
            target_m=tuple(float(x) for x in self.episode_target_location_m),
            subgoal_elapsed=subgoal_elapsed,
            info=info,
        )


def joint_state_to_maps(
    joint_names: Iterable[str],
    positions: Sequence[float],
    velocities: Sequence[float],
    efforts: Optional[Sequence[float]] = None,
) -> tuple[dict[str, float], dict[str, float]]:
    pos_map: dict[str, float] = {}
    vel_map: dict[str, float] = {}
    for i, name in enumerate(joint_names):
        key = name
        if key.startswith("panda_"):
            key = key[len("panda_") :]
        if i < len(positions):
            pos_map[key] = float(positions[i])
        if i < len(velocities):
            vel_map[key] = float(velocities[i])
    return pos_map, vel_map


def _default_onnx_model_path() -> str:
    try:
        from ament_index_python.packages import get_package_share_directory

        return str(Path(get_package_share_directory("franka_emika_panda")) / "ppo_track_franka.onnx")
    except Exception:
        return str(Path(__file__).resolve().parent / "ppo_track_franka.onnx")


def _clip_joint_positions(positions: Sequence[float]) -> list[float]:
    return [
        float(np.clip(value, PANDA_JOINT_LIMITS[i][0], PANDA_JOINT_LIMITS[i][1]))
        for i, value in enumerate(positions)
    ]


def _normalized_to_joint_positions(actions: Sequence[float]) -> list[float]:
    positions = []
    for i, value in enumerate(actions):
        lower, upper = PANDA_JOINT_LIMITS[i]
        normalized = float(np.clip(value, -1.0, 1.0))
        positions.append(lower + 0.5 * (normalized + 1.0) * (upper - lower))
    return positions


def _interpret_policy_actions(
    actions: Sequence[float],
    joint_pos: Sequence[float],
    last_command: Sequence[float],
    mode: str,
    delta_scale: float,
) -> list[float]:
    if len(actions) != 7:
        raise ValueError(f"expected 7 policy actions, got {len(actions)}")
    if mode == "absolute":
        return _clip_joint_positions(actions)
    if mode == "normalized_absolute":
        return _normalized_to_joint_positions(actions)
    if mode == "delta_position":
        return _clip_joint_positions(
            float(position) + float(action) * delta_scale
            for position, action in zip(joint_pos, actions)
        )
    if mode == "normalized_delta_position":
        return _clip_joint_positions(
            float(position) + float(np.clip(action, -1.0, 1.0)) * delta_scale
            for position, action in zip(joint_pos, actions)
        )
    if mode == "delta_command":
        return _clip_joint_positions(
            float(command) + float(action) * delta_scale
            for command, action in zip(last_command, actions)
        )
    if mode == "normalized_delta_command":
        return _clip_joint_positions(
            float(command) + float(np.clip(action, -1.0, 1.0)) * delta_scale
            for command, action in zip(last_command, actions)
        )
    raise ValueError(
        f"Unsupported action_output_mode={mode!r}. "
        "Use one of: absolute, normalized_absolute, delta_position, "
        "normalized_delta_position, delta_command, normalized_delta_command."
    )


def _dry_run(seed: int, steps: int) -> None:
    session = PandaTrackReachSession()
    session.seed(seed)
    target = session.reset()
    print(f"reset -> target (task frame m): {tuple(round(float(x), 4) for x in target)}")

    tcp = np.array(target) + np.array([0.12, 0.0, 0.05])
    for step_idx in range(steps):
        # Move TCP toward target; snap inside success band on the last few steps.
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

    class PandaTrackReachNode(Node):
        def __init__(self) -> None:
            super().__init__("panda_track_reach")
            self.declare_parameter("frame_id", "panda_link0")
            self.declare_parameter("tcp_source", "tf")
            self.declare_parameter("tcp_frame_id", "panda_hand_tcp")
            self.declare_parameter("tcp_pose_topic", "ee_pose")
            self.declare_parameter("joint_states_topic", "joint_states")
            self.declare_parameter("target_pose_topic", "reach_track/target")
            self.declare_parameter("target_marker_topic", "reach_track/target_marker")
            self.declare_parameter("target_marker_scale_m", 0.06)
            self.declare_parameter("info_topic", "reach_track/info")
            self.declare_parameter("command_topic", "panda_joint_trajectory_controller/joint_trajectory")
            self.declare_parameter("control_hz", 60.0)
            self.declare_parameter("trajectory_duration_s", 0.05)
            self.declare_parameter("random_seed", 0)
            self.declare_parameter("auto_reset_on_failure", True)
            self.declare_parameter("policy_enabled", bool(args.policy))
            self.declare_parameter("onnx_model_path", args.model or _default_onnx_model_path())
            self.declare_parameter("tcp_observation_mode", "tcp_error_normalized")
            self.declare_parameter("tcp_observation_scale", 1.0)
            self.declare_parameter("tcp_observation_signs", "1,1,1")
            self.declare_parameter("tcp_observation_unit_scale", 1.0)
            self.declare_parameter("joint_observation_mode", "position_velocity_command")
            self.declare_parameter("action_scale", 1.0)
            self.declare_parameter("action_output_mode", "absolute")
            self.declare_parameter("action_delta_scale", 0.05)

            frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
            self._frame_id = frame_id
            self._tcp_source = str(self.get_parameter("tcp_source").value)
            self._tcp_frame_id = str(self.get_parameter("tcp_frame_id").value)

            cfg = PandaTrackReachConfig(
                random_seed=int(self.get_parameter("random_seed").value),
            )
            self._session = PandaTrackReachSession(cfg)
            self._session.seed(cfg.random_seed)
            self._have_tcp = False
            self._have_joints = False
            self._tcp_m = np.zeros(3, dtype=np.float64)
            self._joint_pos_vector = list(PANDA_HOME_JOINTS)
            self._joint_vel_vector = [0.0] * 7
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
            self._command_pub = self.create_publisher(
                JointTrajectory,
                str(self.get_parameter("command_topic").value),
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
            self._action_scale = float(self.get_parameter("action_scale").value)
            self._action_output_mode = str(self.get_parameter("action_output_mode").value)
            self._action_delta_scale = float(self.get_parameter("action_delta_scale").value)
            self._policy: Optional[OnnxPolicyRunner] = None
            if self._policy_enabled:
                model_path = str(self.get_parameter("onnx_model_path").value)
                try:
                    self._policy = OnnxPolicyRunner(model_path)
                except Exception as exc:
                    self.get_logger().error(f"Failed to load ONNX policy: {exc}")
                    raise

            self._publish_target(self._session.reset())
            self.get_logger().info(
                f"Panda track reach ready (frame={frame_id}, hz={hz:.1f}, "
                f"policy_enabled={self._policy_enabled}, tcp_source={self._tcp_source})"
            )

        def _on_tcp_pose(self, msg: PoseStamped) -> None:
            self._tcp_m = np.array(
                [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
                dtype=np.float64,
            )
            self._have_tcp = True

        def _on_joint_state(self, msg: JointState) -> None:
            pos_map, vel_map = joint_state_to_maps(
                msg.name, msg.position, msg.velocity, msg.effort
            )
            if all(joint in pos_map for joint in PANDA_ARM_JOINTS):
                self._joint_pos = {k: pos_map[k] for k in PANDA_ARM_JOINTS}
                self._joint_vel = {k: vel_map.get(k, 0.0) for k in PANDA_ARM_JOINTS}
                self._joint_pos_vector = [self._joint_pos[joint] for joint in PANDA_ARM_JOINTS]
                self._joint_vel_vector = [self._joint_vel[joint] for joint in PANDA_ARM_JOINTS]
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

        def _publish_target_marker(self, target_pose: PoseStamped) -> None:
            marker = Marker()
            marker.header = target_pose.header
            marker.ns = "panda_track_reach"
            marker.id = 0
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose = target_pose.pose
            marker.scale.x = self._target_marker_scale_m
            marker.scale.y = self._target_marker_scale_m
            marker.scale.z = self._target_marker_scale_m
            marker.color.r = 1.0
            marker.color.g = 0.08
            marker.color.b = 0.02
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
            msg.joint_names = list(PANDA_ROS_JOINTS)
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
                action_scale=self._action_scale,
            )
            target_actions = _interpret_policy_actions(
                raw_actions,
                self._joint_pos_vector,
                self._last_command_vector,
                self._action_output_mode,
                self._action_delta_scale,
            )
            smoothed_actions = self._session.smooth_actions(
                target_actions,
                measured_joint_pos=self._joint_pos_vector,
                measured_joint_vel=self._joint_vel_vector,
            )
            command = _clip_joint_positions(smoothed_actions)
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
    node = PandaTrackReachNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
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
        help="Path to the ONNX policy. Defaults to ppo_track_franka.onnx in franka_emika_panda.",
    )
    parser.add_argument(
        "--dump-config",
        action="store_true",
        help="Print default PandaTrackReachConfig as JSON and exit.",
    )
    args, ros_args = parser.parse_known_args()
    args.ros_args = ros_args

    if args.dump_config:
        print(json.dumps(asdict(PandaTrackReachConfig()), indent=2))
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
