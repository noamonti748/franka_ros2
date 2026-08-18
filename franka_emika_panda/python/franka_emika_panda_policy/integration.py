"""URLab-compatible integrated-velocity action channel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import StudentV6Config


def _seven(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (7,):
        raise ValueError(f"{name} must have shape (7,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def denormalize_action(
    action: Sequence[float], lower: Sequence[float], upper: Sequence[float]
) -> np.ndarray:
    normalized = np.clip(_seven(action, "action"), -1.0, 1.0)
    low = _seven(lower, "lower")
    high = _seven(upper, "upper")
    return low + 0.5 * (normalized + 1.0) * (high - low)


def normalize_target(
    target: Sequence[float], lower: Sequence[float], upper: Sequence[float]
) -> np.ndarray:
    value = _seven(target, "target")
    low = _seven(lower, "lower")
    high = _seven(upper, "upper")
    return np.clip(2.0 * (value - low) / np.maximum(high - low, 1.0e-6) - 1.0, -1.0, 1.0)


def advance_integrated_velocity_target(
    policy_target: Sequence[float],
    current_position: Sequence[float],
    previous_target: Sequence[float],
    velocity_limit: Sequence[float],
    control_minimum: Sequence[float],
    control_maximum: Sequence[float],
    *,
    control_hz: float = 60.0,
    response_time: float = 0.5,
    lead_steps: float = 4.0,
) -> np.ndarray:
    """NumPy port of the training environment's integrated target update."""

    policy = _seven(policy_target, "policy_target")
    current = _seven(current_position, "current_position")
    previous = _seven(previous_target, "previous_target")
    limit = _seven(velocity_limit, "velocity_limit")
    lower = _seven(control_minimum, "control_minimum")
    upper = _seven(control_maximum, "control_maximum")
    if control_hz <= 0.0 or response_time <= 0.0:
        raise ValueError("control_hz and response_time must be positive")
    velocity = np.clip((policy - current) / response_time, -limit, limit)
    integrated = previous + velocity / control_hz
    lead = limit / control_hz * max(lead_steps, 1.0)
    integrated = np.clip(integrated, current - lead, current + lead)
    return np.clip(integrated, lower, upper)


@dataclass
class IntegratedActionController:
    config: StudentV6Config
    action_latency_steps: int = 0
    integrated_target: np.ndarray | None = None
    smoothed_target: np.ndarray | None = None
    action_buffer: list[np.ndarray] | None = None

    def reset(self, current_position: Sequence[float]) -> None:
        current = _seven(current_position, "current_position")
        if not 0 <= self.action_latency_steps <= 3:
            raise ValueError("action_latency_steps must be in [0, 3]")
        self.integrated_target = current.copy()
        self.smoothed_target = current.copy()
        initial_action = normalize_target(
            current,
            self.config.joint_lower,
            self.config.joint_upper,
        )
        self.action_buffer = [
            initial_action.copy() for _ in range(self.action_latency_steps)
        ]

    @property
    def velocity_limit(self) -> np.ndarray:
        return np.asarray(self.config.rated_joint_velocity) * self.config.velocity_limit_fraction

    def step(
        self,
        normalized_action: Sequence[float],
        current_position: Sequence[float],
    ) -> np.ndarray:
        current = _seven(current_position, "current_position")
        if (
            self.integrated_target is None
            or self.smoothed_target is None
            or self.action_buffer is None
        ):
            self.reset(current)
        lower = np.asarray(self.config.joint_lower)
        upper = np.asarray(self.config.joint_upper)
        action = np.clip(_seven(normalized_action, "action"), -1.0, 1.0)
        if self.action_latency_steps:
            delayed_action = self.action_buffer.pop(0)
            self.action_buffer.append(action.copy())
        else:
            delayed_action = action
        raw_target = denormalize_action(delayed_action, lower, upper)
        self.integrated_target = advance_integrated_velocity_target(
            raw_target,
            current,
            self.integrated_target,
            self.velocity_limit,
            lower,
            upper,
            control_hz=self.config.control_hz,
            response_time=self.config.response_time,
            lead_steps=self.config.integrated_lead_steps,
        )
        alpha = self.config.action_smoothing_alpha
        candidate = self.smoothed_target + alpha * (
            self.integrated_target - self.smoothed_target
        )
        max_delta = self.velocity_limit / self.config.control_hz
        candidate = np.clip(
            candidate,
            self.smoothed_target - max_delta,
            self.smoothed_target + max_delta,
        )
        self.smoothed_target = np.clip(candidate, lower, upper)
        return self.smoothed_target.astype(np.float64, copy=True)
