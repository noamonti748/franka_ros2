"""ROS-free command interlocks and rate limiting."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

import numpy as np


def seven(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (7,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite seven-vector")
    return array


def resolve_string_alias(primary: str, alias: str, name: str) -> str:
    primary_value = str(primary).strip()
    alias_value = str(alias).strip()
    if primary_value and alias_value and primary_value != alias_value:
        raise ValueError(f"{name} canonical parameter and alias conflict")
    return primary_value or alias_value


def may_publish_fault_hold(policy_enabled: bool, hold_on_fault: bool) -> bool:
    """Monitor-only mode can never publish an arm hold."""

    return bool(policy_enabled and hold_on_fault)


@dataclass
class ConsecutiveReadiness:
    required_ticks: int
    count: int = 0

    def reset(self) -> None:
        self.count = 0

    def update(self, ready: bool) -> bool:
        self.count = min(self.count + 1, max(self.required_ticks, 1)) if ready else 0
        return self.count >= max(self.required_ticks, 1)


@dataclass
class HomeReadiness:
    home: np.ndarray
    position_tolerance_rad: float
    velocity_tolerance_rad_s: float
    required_ticks: int
    count: int = 0

    def __post_init__(self) -> None:
        self.home = seven(self.home, "home")
        if self.position_tolerance_rad < 0.0 or self.velocity_tolerance_rad_s < 0.0:
            raise ValueError("home tolerances must be nonnegative")

    def reset(self) -> None:
        self.count = 0

    def update(
        self,
        *,
        policy_enabled: bool,
        q: Sequence[float],
        qvel: Sequence[float],
    ) -> bool:
        position = seven(q, "q")
        velocity = seven(qvel, "qvel")
        at_home = bool(
            policy_enabled
            and np.max(np.abs(position - self.home)) <= self.position_tolerance_rad
            and np.max(np.abs(velocity)) <= self.velocity_tolerance_rad_s
        )
        self.count = min(self.count + 1, max(self.required_ticks, 1)) if at_home else 0
        return self.count >= max(self.required_ticks, 1)


@dataclass(frozen=True)
class CommandLimiter:
    lower: np.ndarray
    upper: np.ndarray
    velocity_limit_rad_s: np.ndarray
    max_step_rad: np.ndarray
    control_hz: float

    def __post_init__(self) -> None:
        lower = seven(self.lower, "lower")
        upper = seven(self.upper, "upper")
        velocity = seven(self.velocity_limit_rad_s, "velocity_limit_rad_s")
        maximum_step = seven(self.max_step_rad, "max_step_rad")
        if np.any(lower >= upper):
            raise ValueError("command lower limits must be below upper limits")
        if np.any(velocity <= 0.0) or np.any(maximum_step <= 0.0):
            raise ValueError("command velocity and step limits must be positive")
        if self.control_hz <= 0.0:
            raise ValueError("control_hz must be positive")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "velocity_limit_rad_s", velocity)
        object.__setattr__(self, "max_step_rad", maximum_step)

    def limit(
        self, desired: Sequence[float], reference: Sequence[float]
    ) -> np.ndarray:
        target = np.clip(seven(desired, "desired"), self.lower, self.upper)
        prior = seven(reference, "reference")
        allowed = np.minimum(
            self.max_step_rad, self.velocity_limit_rad_s / self.control_hz
        )
        return np.clip(target, prior - allowed, prior + allowed)


class RecoveryStage(str, Enum):
    IDLE = "idle"
    RETREAT = "retreat"
    OPEN = "open"
    WAIT_FRESH_ESTIMATE = "wait_fresh_estimate"
    COMPLETE = "complete"


@dataclass(frozen=True)
class RecoveryStep:
    stage: RecoveryStage
    target_position_m: np.ndarray
    request_open: bool
    complete: bool
    telemetry: dict[str, Any]


@dataclass
class GraspRetryRecovery:
    """Deterministic sim2real retry: vertical retreat, open, then reobserve.

    The helper only emits a TCP target and gripper intent. It never owns scene
    state, so neither the cube nor the robot can be teleported by recovery.
    """

    retreat_height_m: float = 0.075
    position_tolerance_m: float = 0.005
    max_joint_speed_rad_s: float = 0.10
    settled_steps: int = 3
    open_fraction: float = 0.9
    fresh_estimate_steps: int = 1
    active: bool = False
    stage: RecoveryStage = RecoveryStage.IDLE
    stage_step: int = 0
    settled_count: int = 0
    fresh_estimate_count: int = 0
    start_estimate_generation: int = 0
    wait_estimate_generation: int = 0
    retreat_target_m: np.ndarray | None = None

    def __post_init__(self) -> None:
        if (
            self.retreat_height_m <= 0.0
            or self.position_tolerance_m < 0.0
            or self.max_joint_speed_rad_s < 0.0
            or self.settled_steps <= 0
            or not 0.0 < self.open_fraction <= 1.0
            or self.fresh_estimate_steps <= 0
        ):
            raise ValueError("invalid grasp retry recovery settings")

    def start(
        self,
        tcp_position_m: Sequence[float],
        *,
        estimate_generation: int,
    ) -> None:
        tcp = self._position(tcp_position_m)
        if estimate_generation < 0:
            raise ValueError("estimate_generation must be nonnegative")
        self.retreat_target_m = tcp + np.asarray(
            (0.0, 0.0, self.retreat_height_m), dtype=np.float64
        )
        self.active = True
        self.stage = RecoveryStage.RETREAT
        self.stage_step = 0
        self.settled_count = 0
        self.fresh_estimate_count = 0
        self.start_estimate_generation = int(estimate_generation)
        self.wait_estimate_generation = int(estimate_generation)

    def reset(self) -> None:
        self.active = False
        self.stage = RecoveryStage.IDLE
        self.stage_step = 0
        self.settled_count = 0
        self.fresh_estimate_count = 0
        self.start_estimate_generation = 0
        self.wait_estimate_generation = 0
        self.retreat_target_m = None

    def step(
        self,
        *,
        tcp_position_m: Sequence[float],
        joint_velocity: Sequence[float],
        gripper_openness: float,
        grasp_latched: bool,
        estimate_generation: int,
        estimate_valid: bool,
    ) -> RecoveryStep:
        if not self.active or self.retreat_target_m is None:
            raise RuntimeError("grasp retry recovery is not active")
        tcp = self._position(tcp_position_m)
        velocity = seven(joint_velocity, "joint_velocity")
        if not np.isfinite(gripper_openness):
            raise ValueError("gripper_openness must be finite")
        if estimate_generation < 0:
            raise ValueError("estimate_generation must be nonnegative")

        position_error = float(np.linalg.norm(tcp - self.retreat_target_m))
        joint_speed = float(np.linalg.norm(velocity))
        settled = bool(
            position_error <= self.position_tolerance_m
            and joint_speed <= self.max_joint_speed_rad_s
        )
        self.stage_step += 1
        self.settled_count = self.settled_count + 1 if settled else 0

        if (
            self.stage == RecoveryStage.RETREAT
            and self.settled_count >= self.settled_steps
        ):
            self._enter(RecoveryStage.OPEN)

        open_ready = bool(
            gripper_openness >= self.open_fraction
            and not grasp_latched
            and joint_speed <= self.max_joint_speed_rad_s
        )
        if self.stage == RecoveryStage.OPEN and open_ready:
            self.wait_estimate_generation = int(estimate_generation)
            self._enter(RecoveryStage.WAIT_FRESH_ESTIMATE)

        estimate_fresh = bool(
            estimate_valid
            and estimate_generation > self.wait_estimate_generation
        )
        if self.stage == RecoveryStage.WAIT_FRESH_ESTIMATE:
            self.fresh_estimate_count = (
                self.fresh_estimate_count + 1 if estimate_fresh else 0
            )
            if (
                open_ready
                and settled
                and self.fresh_estimate_count >= self.fresh_estimate_steps
            ):
                self.active = False
                self.stage = RecoveryStage.COMPLETE

        request_open = self.stage in (
            RecoveryStage.OPEN,
            RecoveryStage.WAIT_FRESH_ESTIMATE,
            RecoveryStage.COMPLETE,
        )
        telemetry = self.telemetry(
            position_error_m=position_error,
            joint_speed_rad_s=joint_speed,
            gripper_openness=gripper_openness,
            grasp_latched=grasp_latched,
            estimate_generation=estimate_generation,
            estimate_valid=estimate_valid,
            estimate_fresh=estimate_fresh,
        )
        return RecoveryStep(
            self.stage,
            self.retreat_target_m.copy(),
            request_open,
            self.stage == RecoveryStage.COMPLETE,
            telemetry,
        )

    def telemetry(self, **observed: Any) -> dict[str, Any]:
        target = (
            None
            if self.retreat_target_m is None
            else self.retreat_target_m.tolist()
        )
        return {
            "active": self.active,
            "stage": self.stage.value,
            "stage_step": self.stage_step,
            "settled_count": self.settled_count,
            "fresh_estimate_count": self.fresh_estimate_count,
            "start_estimate_generation": self.start_estimate_generation,
            "wait_estimate_generation": self.wait_estimate_generation,
            "retreat_target_m": target,
            **observed,
        }

    def _enter(self, stage: RecoveryStage) -> None:
        self.stage = stage
        self.stage_step = 0
        self.settled_count = 0
        self.fresh_estimate_count = 0

    @staticmethod
    def _position(values: Sequence[float]) -> np.ndarray:
        position = np.asarray(values, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("tcp_position_m must be a finite three-vector")
        return position


# Temporary import compatibility while the ROS orchestration adopts the new
# TCP-space API. The implementation no longer has home-reset semantics.
RecoveryToHome = GraspRetryRecovery
