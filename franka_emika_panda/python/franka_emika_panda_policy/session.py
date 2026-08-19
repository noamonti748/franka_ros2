"""Observable eight-phase gates, dwell, retries, and gripper timing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .config import PHASE_NAMES, Phase, StudentV6Config
from .estimator import Estimate, ObservableEstimator
from .mode import AutonomousModeFilter


def scripted_gripper_target(
    current_half_width_m: float,
    phase: int | Phase,
    *,
    close_steps: int = 20,
    open_steps: int = 8,
) -> float:
    """Return the source environment's per-finger command for one step."""

    target = 0.04 if int(phase) <= int(Phase.DESCEND_GRASP) else current_half_width_m
    if int(phase) == int(Phase.CLOSE_GRIPPER):
        target = max(0.0, target - 0.04 / close_steps)
    if int(phase) >= int(Phase.RELEASE):
        target = min(0.04, target + 0.04 / open_steps)
    return float(target)


@dataclass(frozen=True)
class SessionResult:
    phase: Phase
    phase_step: int
    gate: bool
    transitioned: bool
    success: bool
    done: bool
    fatal_timeout: bool
    dropped: bool
    grasp_retry: bool
    placement_retry: bool
    use_dls: bool
    gripper_half_width_target_m: float
    target_position_m: np.ndarray
    estimate: Estimate
    telemetry: dict[str, Any]


class ObservableSession:
    def __init__(self, config: StudentV6Config) -> None:
        self.config = config
        self.estimator = ObservableEstimator(config)
        self.reset()

    def reset(self) -> None:
        self.phase = Phase.HOVER_RED
        self.phase_step = 0
        self.episode_step = 0
        self.dwell = 0
        self.placed_dwell = 0
        self.grasp_retries = 0
        self.placement_retries = 0
        self.placed_dwell = 0
        self.frozen_miss_dwell = 0
        self.gripper_half_width_target_m = 0.04
        self.estimator.reset()

    def _phase_target(
        self,
        estimate: Estimate,
        tcp_rotation: np.ndarray,
    ) -> np.ndarray:
        cube = estimate.cube_position_m
        goal = estimate.goal_position_m
        red_hover = cube + np.asarray((0.0, 0.0, self.config.hover_offset_m))
        red_grasp = cube + np.asarray((0.0, 0.0, self.config.grasp_tcp_offset_m))
        lift = cube + np.asarray(
            (
                0.0,
                0.0,
                self.config.required_lift_height_m
                + self.config.lift_tcp_overshoot_m,
            )
        )
        compensated = (
            -(tcp_rotation @ self.estimator.held_rel_tcp)
            if estimate.grasp_latched
            else np.zeros(3)
        )
        green_hover = goal + compensated + np.asarray(
            (0.0, 0.0, self.config.place_hover_offset_m)
        )
        green_release = goal + compensated
        targets = (
            red_hover,
            red_grasp,
            red_grasp,
            lift,
            green_hover,
            green_release,
            green_release,
            green_release,
        )
        return np.asarray(targets[int(self.phase)], dtype=np.float64)

    def _gate(
        self,
        estimate: Estimate,
        tcp_position_m: np.ndarray,
        tcp_rotation: np.ndarray,
        joint_velocity: np.ndarray,
        gripper_openness: float,
        require_grasp_alignment: bool = False,
        require_hover_xy_clearance: bool = False,
        alignment_cube_m: np.ndarray | Sequence[float] | None = None,
    ) -> tuple[bool, np.ndarray, float, float, float, float, float]:
        target = self._phase_target(estimate, tcp_rotation)
        target_distance = float(np.linalg.norm(tcp_position_m - target))
        joint_speed = float(np.linalg.norm(joint_velocity))
        planar = float(
            np.linalg.norm(estimate.cube_position_m[:2] - estimate.goal_position_m[:2])
        )
        height = float(
            abs(estimate.cube_position_m[2] - estimate.goal_position_m[2])
        )
        cube_for_grasp = np.asarray(estimate.cube_position_m, dtype=np.float64)
        if alignment_cube_m is not None:
            cube_for_grasp = np.asarray(alignment_cube_m, dtype=np.float64)
            if cube_for_grasp.shape != (3,) or not np.all(np.isfinite(cube_for_grasp)):
                raise ValueError("alignment_cube_m must be a finite (3,) vector")
        grasp_xy = float(
            np.linalg.norm(tcp_position_m[:2] - cube_for_grasp[:2])
        )
        grasp_z_error = abs(
            tcp_position_m[2]
            - (cube_for_grasp[2] + self.config.grasp_tcp_offset_m)
        )
        threshold = (
            self.config.phase_success_thresholds[int(self.phase)]
            + self.config.observable_position_gate_margins[int(self.phase)]
        )
        positional = bool(
            estimate.valid
            and target_distance < threshold
            and joint_speed < self.config.phase_max_joint_velocity[int(self.phase)]
        )
        descend_place = bool(
            estimate.valid
            and planar <= self.config.placement_tolerance_m
            and height <= self.config.placement_height_tolerance_m
        )
        if self.phase == Phase.CLOSE_GRIPPER:
            gate = estimate.grasp_latched
        elif self.phase == Phase.HOVER_RED and require_hover_xy_clearance:
            gate = bool(
                positional
                and grasp_xy <= self.config.observable_hover_xy_tolerance_m
            )
        elif self.phase == Phase.DESCEND_GRASP and require_grasp_alignment:
            gate = bool(
                positional
                and grasp_xy <= self.config.observable_grasp_xy_tolerance_m
                and grasp_z_error <= self.config.observable_grasp_z_tolerance_m
            )
        elif self.phase == Phase.LIFT:
            gate = bool(
                estimate.grasp_latched
                and estimate.cube_position_m[2]
                - self.config.support_cube_center_height_m
                >= self.config.required_lift_height_m
            )
        elif self.phase == Phase.DESCEND_PLACE:
            gate = descend_place
        elif self.phase == Phase.RELEASE:
            gate = bool(
                estimate.valid
                and gripper_openness >= self.config.release_open_fraction
                and not estimate.grasp_latched
                and planar <= self.config.placement_tolerance_m
            )
        elif self.phase == Phase.PLACED:
            gate = bool(
                descend_place
                and estimate.settled
                and gripper_openness >= self.config.release_open_fraction
                and not estimate.grasp_latched
            )
        else:
            gate = positional
        return gate, target, target_distance, planar, height, grasp_xy, grasp_z_error

    def step(
        self,
        *,
        geometry: Sequence[float],
        tcp_position_m: Sequence[float],
        tcp_rotation: np.ndarray,
        joint_velocity: Sequence[float],
        finger_span_m: float,
        gripper_openness: float,
        grasp_latched_override: bool | None = None,
        require_grasp_alignment: bool = False,
        require_hover_xy_clearance: bool = False,
        alignment_cube_m: np.ndarray | Sequence[float] | None = None,
    ) -> SessionResult:
        tcp = np.asarray(tcp_position_m, dtype=np.float64)
        rotation = np.asarray(tcp_rotation, dtype=np.float64)
        qvel = np.asarray(joint_velocity, dtype=np.float64)
        if tcp.shape != (3,) or rotation.shape != (3, 3) or qvel.shape != (7,):
            raise ValueError("tcp, rotation, and joint_velocity have invalid shapes")
        if not (
            np.all(np.isfinite(tcp))
            and np.all(np.isfinite(rotation))
            and np.all(np.isfinite(qvel))
            and np.isfinite(gripper_openness)
        ):
            raise ValueError("session input contains non-finite values")

        active_phase = self.phase
        active_phase_step = self.phase_step
        estimate = self.estimator.update(
            geometry=geometry,
            tcp_position_m=tcp,
            tcp_rotation=rotation,
            finger_span_m=finger_span_m,
            grasp_latched_override=grasp_latched_override,
        )
        gate, target, target_distance, planar, height, grasp_xy, grasp_z_error = self._gate(
            estimate,
            tcp,
            rotation,
            qvel,
            gripper_openness,
            require_grasp_alignment=require_grasp_alignment,
            require_hover_xy_clearance=require_hover_xy_clearance,
            alignment_cube_m=alignment_cube_m,
        )
        self.dwell = self.dwell + 1 if gate else 0
        required_dwell = max(
            self.config.phase_settled_dwell[int(active_phase)]
            + self.config.observable_phase_extra_dwell[int(active_phase)],
            1,
        )
        transitioned = bool(gate and self.dwell >= required_dwell)
        next_phase = Phase(min(int(active_phase) + int(transitioned), int(Phase.PLACED)))
        if active_phase == Phase.PLACED and gate:
            self.placed_dwell += 1
        else:
            self.placed_dwell = 0
        success = bool(
            active_phase == Phase.PLACED
            and self.placed_dwell >= self.config.success_dwell_steps
        )

        next_phase_step = 0 if transitioned else active_phase_step + 1
        timed_out = (
            next_phase_step >= self.config.phase_step_limits[int(active_phase)]
        )
        threshold = (
            self.config.phase_success_thresholds[int(active_phase)]
            + self.config.observable_position_gate_margins[int(active_phase)]
        )
        joint_speed = float(np.linalg.norm(qvel))
        missed_live_cube = bool(
            require_grasp_alignment
            and active_phase == Phase.DESCEND_GRASP
            and target_distance < threshold
            and joint_speed < self.config.phase_max_joint_velocity[int(active_phase)]
            and grasp_xy > self.config.observable_grasp_xy_tolerance_m
        )
        self.frozen_miss_dwell = self.frozen_miss_dwell + 1 if missed_live_cube else 0
        grasp_retry = bool(
            self.grasp_retries < self.config.max_grasp_retries
            and (
                (
                    timed_out
                    and (
                        active_phase == Phase.CLOSE_GRIPPER
                        or (
                            active_phase == Phase.DESCEND_GRASP
                            and require_grasp_alignment
                        )
                    )
                )
                or (
                    missed_live_cube
                    and self.frozen_miss_dwell >= 45
                    and active_phase == Phase.DESCEND_GRASP
                )
            )
        )
        descend_recovery_due = bool(
            active_phase == Phase.DESCEND_PLACE
            and not gate
            and next_phase_step >= self.config.observable_descend_recovery_steps
        )
        placement_retry = bool(
            (timed_out or descend_recovery_due)
            and active_phase == Phase.DESCEND_PLACE
            and estimate.grasp_latched
            and self.placement_retries < self.config.max_placement_retries
        )
        if grasp_retry:
            next_phase = Phase.HOVER_RED
            next_phase_step = 0
            self.grasp_retries += 1
            self.frozen_miss_dwell = 0
            self.estimator.clear_latch()
        if placement_retry:
            next_phase = Phase.HOVER_GREEN
            next_phase_step = 0
            self.placement_retries += 1
            self.settle_reset(estimate.cube_position_m)
        fatal_timeout = bool(timed_out and not grasp_retry and not placement_retry)
        dropped = bool(
            int(Phase.LIFT) <= int(active_phase) <= int(Phase.DESCEND_PLACE)
            and estimate.released
        )

        self.gripper_half_width_target_m = scripted_gripper_target(
            self.gripper_half_width_target_m,
            active_phase,
            close_steps=self.config.scripted_gripper_close_steps,
            open_steps=self.config.scripted_gripper_open_steps,
        )
        self.phase = next_phase
        self.phase_step = next_phase_step
        self.episode_step += 1
        if transitioned:
            self.dwell = 0
        done = bool(success or fatal_timeout or dropped)
        use_dls = bool(
            int(active_phase) in (int(Phase.HOVER_RED), int(Phase.DESCEND_GRASP))
            and active_phase_step >= self.config.dls_warmup_steps_per_phase
        )
        telemetry = {
            "phase": int(self.phase),
            "phase_name": PHASE_NAMES[int(self.phase)],
            "active_phase": int(active_phase),
            "active_phase_name": PHASE_NAMES[int(active_phase)],
            "phase_step": self.phase_step,
            "episode_step": self.episode_step,
            "gate": gate,
            "dwell": self.dwell,
            "required_dwell": required_dwell,
            "placed_dwell": self.placed_dwell,
            "success_dwell_steps": self.config.success_dwell_steps,
            "target_distance_m": target_distance,
            "grasp_xy_m": grasp_xy,
            "grasp_z_error_m": grasp_z_error,
            "require_grasp_alignment": require_grasp_alignment,
            "require_hover_xy_clearance": require_hover_xy_clearance,
            "hover_xy_tolerance_m": self.config.observable_hover_xy_tolerance_m,
            "cube_goal_planar_m": planar,
            "cube_goal_height_m": height,
            "gripper_half_width_target_m": self.gripper_half_width_target_m,
            "release_open_fraction": self.config.release_open_fraction,
            "grasp_latched": estimate.grasp_latched,
            "grasp_retries": self.grasp_retries,
            "placement_retries": self.placement_retries,
            "success": success,
            "done": done,
            "fatal_timeout": fatal_timeout,
            "dropped": dropped,
            "use_dls": use_dls,
        }
        return SessionResult(
            self.phase,
            self.phase_step,
            gate,
            transitioned,
            success,
            done,
            fatal_timeout,
            dropped,
            grasp_retry,
            placement_retry,
            use_dls,
            self.gripper_half_width_target_m,
            target,
            estimate,
            telemetry,
        )

    def settle_reset(self, position_m: Sequence[float]) -> None:
        position = np.asarray(position_m, dtype=np.float64)
        self.estimator.settle_history[:] = position
        self.estimator.settle_valid_count = 0


class AutonomousSession:
    """Learned-mode session retaining only physical gripper/drop interlocks."""

    def __init__(
        self,
        config: StudentV6Config,
        *,
        mode_alpha: float = 0.35,
        mode_margin: float = 0.15,
        mode_consecutive_steps: int = 3,
        advance_threshold: float = 0.55,
        min_dwell_steps: int = 20,
    ) -> None:
        self.config = config
        self.estimator = ObservableEstimator(config)
        self.mode_filter = AutonomousModeFilter(
            alpha=mode_alpha,
            margin=mode_margin,
            consecutive_steps=mode_consecutive_steps,
            advance_threshold=advance_threshold,
            min_dwell_steps=min_dwell_steps,
        )
        self.reset()

    def reset(self) -> None:
        self.mode_filter.reset()
        self.phase = Phase.HOVER_RED
        self.phase_step = 0
        self.episode_step = 0
        self.grasp_retries = 0
        self.placement_retries = 0
        self.gripper_half_width_target_m = 0.04
        self.estimator.reset()

    def step(
        self,
        *,
        mode_probs: Sequence[float],
        geometry: Sequence[float],
        tcp_position_m: Sequence[float],
        tcp_rotation: np.ndarray,
        joint_velocity: Sequence[float],
        finger_span_m: float,
        gripper_openness: float,
        grasp_latched_override: bool | None = None,
        advance_prob: float | None = None,
        **_: Any,
    ) -> SessionResult:
        tcp = np.asarray(tcp_position_m, dtype=np.float64)
        rotation = np.asarray(tcp_rotation, dtype=np.float64)
        qvel = np.asarray(joint_velocity, dtype=np.float64)
        estimate = self.estimator.update(
            geometry=geometry,
            tcp_position_m=tcp,
            tcp_rotation=rotation,
            finger_span_m=finger_span_m,
            grasp_latched_override=grasp_latched_override,
        )
        probabilities = np.asarray(mode_probs, dtype=np.float32).copy()
        # These are measured hardware facts, not learned geometric gates.
        if self.phase == Phase.CLOSE_GRIPPER and not estimate.grasp_latched:
            probabilities[int(Phase.LIFT)] = 0.0
        if self.phase == Phase.RELEASE and (
            estimate.grasp_latched
            or gripper_openness < self.config.release_open_fraction
        ):
            probabilities[int(Phase.PLACED)] = 0.0
        if float(np.sum(np.maximum(probabilities, 0.0))) <= 0.0:
            probabilities = np.zeros(8, dtype=np.float32)
            probabilities[int(self.phase)] = 1.0
        # Advance-head path ignores mode probs; still forbid leaving
        # CloseGripper/Lift without a width latch (otherwise ROS "grasps" are
        # empty phase advances with the cube on the table).
        effective_advance = advance_prob
        if advance_prob is not None and not estimate.grasp_latched:
            if self.phase in (Phase.CLOSE_GRIPPER, Phase.LIFT):
                effective_advance = 0.0
        decision = self.mode_filter.update(
            probabilities, advance_prob=effective_advance
        )
        active_phase = decision.phase
        self.phase = decision.phase
        self.phase_step = self.mode_filter.phase_step
        self.episode_step += 1

        self.gripper_half_width_target_m = scripted_gripper_target(
            self.gripper_half_width_target_m,
            active_phase,
            close_steps=self.config.scripted_gripper_close_steps,
            open_steps=self.config.scripted_gripper_open_steps,
        )
        timed_out = (
            self.phase_step
            >= self.config.phase_step_limits[int(active_phase)]
        )
        grasp_retry = bool(
            timed_out
            and active_phase == Phase.CLOSE_GRIPPER
            and not estimate.grasp_latched
            and self.grasp_retries < self.config.max_grasp_retries
        )
        if grasp_retry:
            self.grasp_retries += 1
            self.estimator.clear_latch()
            self.mode_filter.force(Phase.HOVER_RED)
            self.phase = Phase.HOVER_RED
            self.phase_step = 0
        dropped = bool(
            int(Phase.LIFT) <= int(active_phase) <= int(Phase.DESCEND_PLACE)
            and estimate.released
        )
        physically_released = bool(
            gripper_openness >= self.config.release_open_fraction
            and not estimate.grasp_latched
        )
        self.placed_dwell = (
            self.placed_dwell + 1
            if active_phase == Phase.PLACED and physically_released
            else 0
        )
        success = bool(
            self.placed_dwell >= self.config.success_dwell_steps
        )
        fatal_timeout = bool(timed_out and not grasp_retry)
        done = bool(success or dropped or fatal_timeout)
        target = (
            estimate.goal_position_m
            if int(active_phase) >= int(Phase.HOVER_GREEN)
            else estimate.cube_position_m
        )
        telemetry = {
            "phase": int(self.phase),
            "phase_name": PHASE_NAMES[int(self.phase)],
            "active_phase": int(active_phase),
            "active_phase_name": PHASE_NAMES[int(active_phase)],
            "phase_step": self.phase_step,
            "episode_step": self.episode_step,
            "mode_confidence": decision.confidence,
            "mode_changed": decision.changed,
            "mode_probs": decision.probabilities.tolist(),
            "gate": None,
            "grasp_latched": estimate.grasp_latched,
            "grasp_retries": self.grasp_retries,
            "placement_retries": self.placement_retries,
            "placed_dwell": self.placed_dwell,
            "success": success,
            "done": done,
            "fatal_timeout": fatal_timeout,
            "dropped": dropped,
            "use_dls": False,
            "gripper_half_width_target_m": self.gripper_half_width_target_m,
        }
        return SessionResult(
            phase=self.phase,
            phase_step=self.phase_step,
            gate=False,
            transitioned=decision.changed,
            success=success,
            done=done,
            fatal_timeout=fatal_timeout,
            dropped=dropped,
            grasp_retry=grasp_retry,
            placement_retry=False,
            use_dls=False,
            gripper_half_width_target_m=self.gripper_half_width_target_m,
            target_position_m=np.asarray(target, dtype=np.float64),
            estimate=estimate,
            telemetry=telemetry,
        )

    def settle_reset(self, position_m: Sequence[float]) -> None:
        position = np.asarray(position_m, dtype=np.float64)
        self.estimator.settle_history[:] = position
        self.estimator.settle_valid_count = 0
