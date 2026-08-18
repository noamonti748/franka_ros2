"""Hardware-observable width latch and policy-geometry estimator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import StudentV6Config


@dataclass(frozen=True)
class LatchUpdate:
    latched: bool
    dwell: int
    age: int
    newly_latched: bool
    released: bool


def update_finger_span_latch(
    finger_span_m: float,
    latched: bool,
    dwell: int,
    age: int,
    *,
    object_width_m: float,
    width_tolerance_m: float,
    dwell_steps: int,
    timeout_steps: int,
) -> LatchUpdate:
    """Exact NumPy port of the environment's two-sided width latch."""

    if not np.isfinite(finger_span_m):
        raise ValueError("finger span is non-finite")
    in_band = abs(finger_span_m - object_width_m) <= width_tolerance_m
    open_release = finger_span_m > object_width_m + width_tolerance_m
    close_release = finger_span_m < object_width_m - width_tolerance_m
    timed_out = latched and age >= timeout_steps
    released = latched and (open_release or close_release or timed_out)
    candidate_dwell = dwell + 1 if (not latched and in_band) else 0
    newly_latched = not latched and candidate_dwell >= max(dwell_steps, 1)
    next_latched = (latched and not released) or newly_latched
    next_dwell = 0 if (newly_latched or released) else candidate_dwell
    next_age = (0 if newly_latched else age + 1) if next_latched else 0
    return LatchUpdate(next_latched, next_dwell, next_age, newly_latched, released)


def fixed_window_settled(
    position_history: np.ndarray,
    valid_count: int,
    *,
    max_speed_m_s: float,
    control_dt_s: float,
) -> bool:
    history = np.asarray(position_history, dtype=np.float64)
    if history.ndim != 2 or history.shape[1] != 3 or history.shape[0] < 2:
        raise ValueError("position_history must have shape (window, 3)")
    half = history.shape[0] // 2
    displacement = np.linalg.norm(
        np.mean(history[half:], axis=0) - np.mean(history[:half], axis=0)
    )
    allowed = max_speed_m_s * control_dt_s * max(history.shape[0] - half, 1)
    return bool(valid_count >= history.shape[0] and displacement <= allowed)


@dataclass(frozen=True)
class Estimate:
    cube_position_m: np.ndarray
    goal_position_m: np.ndarray
    valid: bool
    settled: bool
    grasp_latched: bool
    newly_latched: bool
    released: bool


class ObservableEstimator:
    def __init__(self, config: StudentV6Config) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.grasp_latched = False
        self.grasp_dwell = 0
        self.latch_age = 0
        self.valid = False
        self.cube_position_m = np.zeros(3, dtype=np.float64)
        self.goal_position_m = np.zeros(3, dtype=np.float64)
        self.held_rel_tcp = np.zeros(3, dtype=np.float64)
        window = self.config.observable_settle_window_steps
        self.settle_history = np.zeros((window, 3), dtype=np.float64)
        self.settle_valid_count = 0

    def clear_latch(self) -> None:
        self.grasp_latched = False
        self.grasp_dwell = 0
        self.latch_age = 0
        self.held_rel_tcp[:] = 0.0

    def update(
        self,
        *,
        geometry: Sequence[float],
        tcp_position_m: Sequence[float],
        tcp_rotation: np.ndarray,
        finger_span_m: float,
        grasp_latched_override: bool | None = None,
    ) -> Estimate:
        geometry_array = np.asarray(geometry, dtype=np.float64)
        tcp = np.asarray(tcp_position_m, dtype=np.float64)
        rotation = np.asarray(tcp_rotation, dtype=np.float64)
        if geometry_array.shape != (6,) or tcp.shape != (3,) or rotation.shape != (3, 3):
            raise ValueError("geometry/tcp/rotation shapes must be (6,), (3,), and (3,3)")
        geometry_valid = bool(
            np.all(np.isfinite(geometry_array))
            and np.all(np.isfinite(tcp))
            and np.all(np.isfinite(rotation))
        )
        if not geometry_valid:
            raise ValueError("estimator input contains non-finite values")

        prior_latched = self.grasp_latched
        latch = update_finger_span_latch(
            finger_span_m,
            self.grasp_latched,
            self.grasp_dwell,
            self.latch_age,
            object_width_m=self.config.observable_object_width_m,
            width_tolerance_m=self.config.observable_grasp_width_tolerance_m,
            dwell_steps=self.config.observable_grasp_dwell_steps,
            timeout_steps=self.config.observable_latch_timeout_steps,
        )
        self.grasp_latched = latch.latched
        self.grasp_dwell = latch.dwell
        self.latch_age = latch.age
        newly_latched = latch.newly_latched
        released = latch.released
        if grasp_latched_override is not None:
            self.grasp_latched = bool(grasp_latched_override)
            newly_latched = self.grasp_latched and not prior_latched
            released = prior_latched and not self.grasp_latched
            if newly_latched or released:
                self.grasp_dwell = 0
                self.latch_age = 0

        detected_cube = tcp + geometry_array[:3] * self.config.geometry_scale_m
        detected_goal = detected_cube + geometry_array[3:] * self.config.geometry_scale_m
        if newly_latched:
            self.held_rel_tcp = rotation.T @ (detected_cube - tcp)
        if self.grasp_latched:
            cube = tcp + rotation @ self.held_rel_tcp
            cube[2] = tcp[2] - self.config.seated_offset_m
            goal = detected_goal.copy()
            goal[2] = self.config.support_cube_center_height_m
        else:
            cube = detected_cube
            goal = detected_goal

        self.cube_position_m = cube
        self.goal_position_m = goal
        self.valid = True
        self.settle_history = np.concatenate(
            (self.settle_history[1:], cube[None, :]), axis=0
        )
        self.settle_valid_count = min(
            self.settle_valid_count + 1,
            self.config.observable_settle_window_steps,
        )
        settled = fixed_window_settled(
            self.settle_history,
            self.settle_valid_count,
            max_speed_m_s=self.config.observable_placement_max_speed_m_s,
            control_dt_s=1.0 / self.config.control_hz,
        )
        return Estimate(
            cube.copy(),
            goal.copy(),
            self.valid,
            settled,
            self.grasp_latched,
            newly_latched,
            released,
        )
