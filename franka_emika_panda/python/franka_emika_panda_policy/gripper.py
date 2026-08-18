"""Gripper backend abstractions with lazy ROS action imports."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
import math
from typing import Any

from .config import Phase


class GripperBackend(ABC):
    """Nonblocking backend called from the 60 Hz control loop."""

    @abstractmethod
    def command(self, phase: int | Phase, half_width_target_m: float) -> None:
        raise NotImplementedError

    def latch_override(self) -> bool | None:
        """Return authoritative hardware latch state, or None for width latch."""

        return None

    def abort_close(self) -> bool:
        """Return True when a close should be abandoned and retried."""

        return False

    def ready(self) -> bool:
        return True

    def request_open(self) -> None:
        """Request recovery opening when supported."""

    def telemetry(self) -> dict[str, Any]:
        return {}


class NoopGripperBackend(GripperBackend):
    def command(self, phase: int | Phase, half_width_target_m: float) -> None:
        del phase, half_width_target_m


class SimGripperCommandBackend(GripperBackend):
    """Serialized target-following goals with a physical stall grasp latch."""

    _MAX_OUTSIDE_BAND_STALLS = 3

    def __init__(
        self,
        node: Any,
        action_name: str,
        *,
        max_effort: float = 100.0,
        open_half_width_m: float = 0.04,
        object_width_m: float = 0.05,
        grasp_width_tolerance_m: float = 0.005,
    ) -> None:
        try:
            from control_msgs.action import GripperCommand
            from rclpy.action import ActionClient
        except ImportError as exc:
            raise RuntimeError(
                "control_msgs and rclpy are required for the simulation gripper backend"
            ) from exc
        self._action_type = GripperCommand
        self._client = ActionClient(node, GripperCommand, action_name)
        self._max_effort = float(max_effort)
        self._open_half_width_m = float(open_half_width_m)
        self._object_width_m = float(object_width_m)
        self._grasp_width_tolerance_m = float(grasp_width_tolerance_m)
        if (
            not math.isfinite(self._object_width_m)
            or self._object_width_m <= 0.0
            or not math.isfinite(self._grasp_width_tolerance_m)
            or self._grasp_width_tolerance_m < 0.0
        ):
            raise ValueError("simulated grasp width band is invalid")
        self._latched = False
        self._pending = False
        self._desired_mode = ""
        self._active_mode = ""
        self._active_target_m: float | None = None
        self._queued_close_targets_m: deque[float] = deque()
        self._completed_mode = ""
        self._completed_target_m: float | None = None
        self._goals_sent = 0
        self._goals_completed = 0
        self._last_result_stalled = False
        self._last_result_reached_goal = False
        self._last_result_span_m: float | None = None
        self._last_error = ""
        self._outside_band_stalls = 0

    def command(self, phase: int | Phase, half_width_target_m: float) -> None:
        target = float(half_width_target_m)
        if not math.isfinite(target) or not 0.0 <= target <= self._open_half_width_m:
            raise ValueError("half_width_target_m is outside the simulated range")
        phase_value = int(phase)
        mode = (
            "close"
            if int(Phase.CLOSE_GRIPPER)
            <= phase_value
            < int(Phase.RELEASE)
            else "open"
        )
        self._desired_mode = mode
        if mode == "open":
            self._queued_close_targets_m.clear()
            if (
                not self._pending
                and self._completed_mode == "open"
                and self._last_error
            ):
                self._completed_mode = ""
                self._completed_target_m = None
        elif not self._latched:
            if self.abort_close():
                self._queued_close_targets_m.clear()
            else:
                # One full close, like hardware Grasp. Serialized 2 mm
                # action goals never finish in the close window and stall
                # on intermediate widths before the cube faces.
                self._queue_close_target(0.0)
        else:
            self._queued_close_targets_m.clear()
        self._dispatch_next()

    def _queue_close_target(self, target_m: float) -> None:
        if (
            self._active_mode == "close"
            and self._active_target_m is not None
            and abs(target_m - self._active_target_m) <= 1.0e-9
        ):
            return
        if (
            self._queued_close_targets_m
            and abs(target_m - self._queued_close_targets_m[-1]) <= 1.0e-9
        ):
            return
        if (
            not self._pending
            and self._completed_mode == "close"
            and self._completed_target_m is not None
            and abs(target_m - self._completed_target_m) <= 1.0e-9
        ):
            return
        self._queued_close_targets_m.append(target_m)

    def _finger_span_m(self, result: Any, target_m: float) -> float:
        position = getattr(result, "position", None)
        if position is None:
            return 2.0 * float(target_m)
        return 2.0 * float(position)

    def _span_in_grasp_band(self, span_m: float) -> bool:
        return abs(span_m - self._object_width_m) <= self._grasp_width_tolerance_m

    def _dispatch_next(self) -> None:
        if self._pending:
            return
        if not self._client.server_is_ready():
            self._last_error = "action_server_unavailable"
            return
        if self._desired_mode == "open":
            target = self._open_half_width_m
            if (
                self._completed_mode == "open"
                and self._completed_target_m is not None
                and abs(target - self._completed_target_m) <= 1.0e-9
            ):
                return
            mode = "open"
        elif self._desired_mode == "close":
            if self._latched or not self._queued_close_targets_m:
                return
            mode = "close"
            target = self._queued_close_targets_m.popleft()
        else:
            return
        goal = self._action_type.Goal()
        # The controller owns one finger joint, so this is per-finger position.
        # Each scripted close target completes before the next is dispatched;
        # no active action goal is preempted by the 60 Hz command stream.
        goal.command.position = target
        goal.command.max_effort = self._max_effort
        self._pending = True
        self._active_mode = mode
        self._active_target_m = target
        self._goals_sent += 1
        future = self._client.send_goal_async(goal)

        def goal_done(goal_future: Any) -> None:
            handle = goal_future.result()
            if handle is None or not handle.accepted:
                self._pending = False
                self._last_error = "goal_rejected"
                self._active_mode = ""
                self._active_target_m = None
                return
            result_future = handle.get_result_async()

            def result_done(done_future: Any) -> None:
                response = done_future.result()
                result = getattr(response, "result", None)
                stalled = bool(getattr(result, "stalled", False))
                reached_goal = bool(getattr(result, "reached_goal", False))
                span_m = self._finger_span_m(result, target)
                self._last_result_stalled = stalled
                self._last_result_reached_goal = reached_goal
                self._last_result_span_m = span_m
                self._completed_mode = mode
                self._completed_target_m = target
                self._goals_completed += 1
                if mode == "close":
                    in_band = self._span_in_grasp_band(span_m)
                    if stalled and in_band:
                        self._latched = True
                        self._queued_close_targets_m.clear()
                        self._last_error = ""
                        self._outside_band_stalls = 0
                    elif stalled:
                        self._latched = False
                        self._last_error = "stall_outside_width_band"
                        self._outside_band_stalls += 1
                        if (
                            self._outside_band_stalls
                            >= self._MAX_OUTSIDE_BAND_STALLS
                        ):
                            self._queued_close_targets_m.clear()
                        else:
                            # A centered direct-MJX grasp first contacts near
                            # 58 mm, then settles into the 50±5 mm band under
                            # continued closing force. Reissue the serialized
                            # full-close goal before classifying an edge hit.
                            self._completed_mode = ""
                            self._completed_target_m = None
                            self._queued_close_targets_m.appendleft(0.0)
                    elif reached_goal and target > 1.0e-9:
                        self._last_error = ""
                        self._outside_band_stalls = 0
                    elif reached_goal:
                        self._last_error = "close_without_stall"
                        self._outside_band_stalls = 0
                    else:
                        self._last_error = "close_failed"
                else:
                    if reached_goal:
                        self._latched = False
                        self._last_error = ""
                        self._outside_band_stalls = 0
                    else:
                        self._last_error = "open_stalled" if stalled else "open_failed"
                self._pending = False
                self._active_mode = ""
                self._active_target_m = None
                self._dispatch_next()

            result_future.add_done_callback(result_done)

        future.add_done_callback(goal_done)

    def ready(self) -> bool:
        return bool(self._client.server_is_ready())

    def request_open(self) -> None:
        self.command(Phase.RELEASE, self._open_half_width_m)

    def abort_close(self) -> bool:
        return self._outside_band_stalls >= self._MAX_OUTSIDE_BAND_STALLS

    def latch_override(self) -> bool | None:
        return self._latched

    def telemetry(self) -> dict[str, Any]:
        return {
            "backend": "sim_gripper_command",
            "ready": self.ready(),
            "pending": self._pending,
            "requested_mode": self._desired_mode,
            "active_mode": self._active_mode,
            "active_target_m": self._active_target_m,
            "queued_close_targets_m": list(self._queued_close_targets_m),
            "goals_sent": self._goals_sent,
            "goals_completed": self._goals_completed,
            "last_result_stalled": self._last_result_stalled,
            "last_result_reached_goal": self._last_result_reached_goal,
            "last_result_span_m": self._last_result_span_m,
            "latched": self._latched,
            "outside_band_stalls": self._outside_band_stalls,
            "abort_close": self.abort_close(),
            "last_error": self._last_error,
        }


class FrankaGraspBackend(GripperBackend):
    """Hardware Grasp-close and Move-open backend."""

    def __init__(
        self,
        node: Any,
        grasp_action_name: str,
        move_action_name: str,
        *,
        object_width_m: float = 0.05,
        epsilon_inner_m: float = 0.005,
        epsilon_outer_m: float = 0.005,
        speed_m_s: float = 0.05,
        force_n: float = 40.0,
        open_width_m: float = 0.08,
    ) -> None:
        try:
            from franka_msgs.action import Grasp, Move
            from rclpy.action import ActionClient
        except ImportError as exc:
            raise RuntimeError(
                "franka_msgs and rclpy are required for the hardware grasp backend"
            ) from exc
        self._grasp_type = Grasp
        self._move_type = Move
        self._grasp_client = ActionClient(node, Grasp, grasp_action_name)
        self._move_client = ActionClient(node, Move, move_action_name)
        self._object_width_m = float(object_width_m)
        self._epsilon_inner_m = float(epsilon_inner_m)
        self._epsilon_outer_m = float(epsilon_outer_m)
        self._speed_m_s = float(speed_m_s)
        self._force_n = float(force_n)
        self._open_width_m = float(open_width_m)
        self._latched = False
        self._pending = False
        self._requested_mode = ""
        self._last_error = ""

    def command(self, phase: int | Phase, half_width_target_m: float) -> None:
        del half_width_target_m
        phase_value = int(phase)
        mode = "grasp" if phase_value == int(Phase.CLOSE_GRIPPER) else (
            "move_open" if phase_value >= int(Phase.RELEASE) else ""
        )
        if not mode or self._pending or mode == self._requested_mode:
            return
        client = self._grasp_client if mode == "grasp" else self._move_client
        if not client.server_is_ready():
            self._last_error = "action_server_unavailable"
            return
        if mode == "grasp":
            goal = self._grasp_type.Goal()
            goal.width = self._object_width_m
            goal.epsilon.inner = self._epsilon_inner_m
            goal.epsilon.outer = self._epsilon_outer_m
            goal.speed = self._speed_m_s
            goal.force = self._force_n
        else:
            goal = self._move_type.Goal()
            goal.width = self._open_width_m
            goal.speed = self._speed_m_s
        self._pending = True
        self._requested_mode = mode
        future = client.send_goal_async(goal)

        def goal_done(goal_future: Any) -> None:
            handle = goal_future.result()
            if handle is None or not handle.accepted:
                self._pending = False
                self._last_error = "goal_rejected"
                self._requested_mode = ""
                return
            result_future = handle.get_result_async()

            def result_done(done_future: Any) -> None:
                response = done_future.result()
                result = getattr(response, "result", None)
                succeeded = bool(getattr(result, "success", False))
                if mode == "grasp":
                    self._latched = succeeded
                elif succeeded:
                    self._latched = False
                self._pending = False
                self._last_error = "" if succeeded else str(getattr(result, "error", "failed"))
                if not succeeded:
                    self._requested_mode = ""

            result_future.add_done_callback(result_done)

        future.add_done_callback(goal_done)

    def latch_override(self) -> bool | None:
        return self._latched

    def ready(self) -> bool:
        return bool(
            self._grasp_client.server_is_ready()
            and self._move_client.server_is_ready()
        )

    def request_open(self) -> None:
        self.command(Phase.RELEASE, 0.04)

    def telemetry(self) -> dict[str, Any]:
        return {
            "backend": "franka_grasp_move",
            "ready": self.ready(),
            "pending": self._pending,
            "latched": self._latched,
            "last_error": self._last_error,
        }
