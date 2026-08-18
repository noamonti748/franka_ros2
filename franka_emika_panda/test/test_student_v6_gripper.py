from pathlib import Path
import sys
import types

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from franka_emika_panda_policy.config import Phase
from franka_emika_panda_policy.gripper import (
    FrankaGraspBackend,
    SimGripperCommandBackend,
)
from franka_emika_panda_policy.session import scripted_gripper_target


def test_hardware_uses_grasp_to_close_and_move_to_open(monkeypatch):
    class Grasp:
        class Goal:
            def __init__(self):
                self.width = 0.0
                self.speed = 0.0
                self.force = 0.0
                self.epsilon = types.SimpleNamespace(inner=0.0, outer=0.0)

    class Move:
        class Goal:
            def __init__(self):
                self.width = 0.0
                self.speed = 0.0

    class ImmediateFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

        def add_done_callback(self, callback):
            callback(self)

    class Handle:
        accepted = True

        def get_result_async(self):
            result = types.SimpleNamespace(success=True, error="")
            return ImmediateFuture(types.SimpleNamespace(result=result))

    clients = []

    class ActionClient:
        def __init__(self, node, action_type, name):
            del node
            self.action_type = action_type
            self.name = name
            self.goals = []
            clients.append(self)

        def server_is_ready(self):
            return True

        def send_goal_async(self, goal):
            self.goals.append(goal)
            return ImmediateFuture(Handle())

    franka_action = types.ModuleType("franka_msgs.action")
    franka_action.Grasp = Grasp
    franka_action.Move = Move
    franka = types.ModuleType("franka_msgs")
    franka.action = franka_action
    rclpy_action = types.ModuleType("rclpy.action")
    rclpy_action.ActionClient = ActionClient
    rclpy = types.ModuleType("rclpy")
    rclpy.action = rclpy_action
    monkeypatch.setitem(sys.modules, "franka_msgs", franka)
    monkeypatch.setitem(sys.modules, "franka_msgs.action", franka_action)
    monkeypatch.setitem(sys.modules, "rclpy", rclpy)
    monkeypatch.setitem(sys.modules, "rclpy.action", rclpy_action)

    backend = FrankaGraspBackend(
        object(),
        "/grasp",
        "/move",
        object_width_m=0.05,
        open_width_m=0.08,
    )
    assert backend.ready()
    assert backend.latch_override() is False

    backend.command(Phase.CLOSE_GRIPPER, 0.0)
    grasp_client = next(client for client in clients if client.action_type is Grasp)
    move_client = next(client for client in clients if client.action_type is Move)
    assert len(grasp_client.goals) == 1
    assert len(move_client.goals) == 0
    assert backend.latch_override() is True

    backend.command(Phase.RELEASE, 0.04)
    assert len(grasp_client.goals) == 1
    assert len(move_client.goals) == 1
    assert move_client.goals[0].width == 0.08
    assert backend.latch_override() is False


def _install_sim_actions(monkeypatch, result_for_goal, *, auto_complete):
    class GripperCommand:
        class Goal:
            def __init__(self):
                self.command = types.SimpleNamespace(position=0.0, max_effort=0.0)

    class Future:
        def __init__(self, value=None, *, done=False):
            self._value = value
            self._done = done
            self._callbacks = []

        def result(self):
            return self._value

        def add_done_callback(self, callback):
            if self._done:
                callback(self)
            else:
                self._callbacks.append(callback)

        def resolve(self, value):
            self._value = value
            self._done = True
            callbacks, self._callbacks = self._callbacks, []
            for callback in callbacks:
                callback(self)

    class Handle:
        accepted = True

        def __init__(self, result_future):
            self.result_future = result_future

        def get_result_async(self):
            return self.result_future

    goals = []
    result_futures = []

    class ActionClient:
        def __init__(self, node, action_type, name):
            del node, action_type, name

        def server_is_ready(self):
            return True

        def send_goal_async(self, goal):
            goals.append(goal)
            result = result_for_goal(goal)
            result_future = Future(
                types.SimpleNamespace(result=result),
                done=auto_complete,
            )
            result_futures.append(result_future)
            return Future(Handle(result_future), done=True)

    control_action = types.ModuleType("control_msgs.action")
    control_action.GripperCommand = GripperCommand
    control = types.ModuleType("control_msgs")
    control.action = control_action
    rclpy_action = types.ModuleType("rclpy.action")
    rclpy_action.ActionClient = ActionClient
    rclpy = types.ModuleType("rclpy")
    rclpy.action = rclpy_action
    monkeypatch.setitem(sys.modules, "control_msgs", control)
    monkeypatch.setitem(sys.modules, "control_msgs.action", control_action)
    monkeypatch.setitem(sys.modules, "rclpy", rclpy)
    monkeypatch.setitem(sys.modules, "rclpy.action", rclpy_action)
    return goals, result_futures


def test_sim_controller_sends_one_full_close_goal(monkeypatch):
    goals, _ = _install_sim_actions(
        monkeypatch,
        lambda goal: types.SimpleNamespace(
            stalled=False,
            reached_goal=True,
            position=goal.command.position,
        ),
        auto_complete=True,
    )

    backend = SimGripperCommandBackend(object(), "/gripper")
    target = 0.04
    for _ in range(20):
        target = scripted_gripper_target(target, Phase.CLOSE_GRIPPER)
        backend.command(Phase.CLOSE_GRIPPER, target)

    positions = [goal.command.position for goal in goals]
    assert positions == [pytest.approx(0.0)]
    assert backend.latch_override() is False
    assert backend.telemetry()["last_error"] == "close_without_stall"

    backend.command(Phase.RELEASE, 0.04)
    assert len(goals) == 2
    assert goals[-1].command.position == 0.04
    assert backend.latch_override() is False


def test_sim_controller_does_not_queue_scripted_close_steps(monkeypatch):
    goals, result_futures = _install_sim_actions(
        monkeypatch,
        lambda goal: types.SimpleNamespace(stalled=False, reached_goal=True),
        auto_complete=False,
    )
    backend = SimGripperCommandBackend(object(), "/gripper")

    backend.command(Phase.CLOSE_GRIPPER, 0.038)
    backend.command(Phase.CLOSE_GRIPPER, 0.036)
    backend.command(Phase.CLOSE_GRIPPER, 0.034)

    assert [goal.command.position for goal in goals] == [0.0]
    assert backend.telemetry()["queued_close_targets_m"] == []
    assert backend.telemetry()["pending"]


def test_centered_simulated_grasp_latches_only_from_stall_result(monkeypatch):
    def centered_result(goal):
        del goal
        return types.SimpleNamespace(
            stalled=True,
            reached_goal=False,
            position=0.025,
        )

    _, _ = _install_sim_actions(
        monkeypatch,
        centered_result,
        auto_complete=True,
    )
    backend = SimGripperCommandBackend(object(), "/gripper")
    backend.command(Phase.CLOSE_GRIPPER, 0.038)

    assert backend.latch_override() is True
    telemetry = backend.telemetry()
    assert telemetry["last_result_stalled"] is True
    assert telemetry["last_result_reached_goal"] is False
    assert telemetry["last_result_span_m"] == pytest.approx(0.05)
    assert telemetry["last_error"] == ""


def test_sim_controller_rejects_wide_stall(monkeypatch):
    goals, _ = _install_sim_actions(
        monkeypatch,
        lambda goal: types.SimpleNamespace(
            stalled=True,
            reached_goal=False,
            position=0.036,
        ),
        auto_complete=True,
    )
    backend = SimGripperCommandBackend(object(), "/gripper")
    backend.command(Phase.CLOSE_GRIPPER, 0.038)
    assert backend.latch_override() is False
    telemetry = backend.telemetry()
    assert telemetry["last_error"] == "stall_outside_width_band"
    assert telemetry["last_result_span_m"] == pytest.approx(0.072)
    assert backend.abort_close() is True
    assert len(goals) == 3


def test_sim_controller_can_settle_from_wide_contact_into_grasp_band(
    monkeypatch,
):
    result_count = 0

    def settling_result(goal):
        nonlocal result_count
        result_count += 1
        if result_count == 1:
            return types.SimpleNamespace(
                stalled=True,
                reached_goal=False,
                position=0.029,
            )
        return types.SimpleNamespace(
            stalled=True,
            reached_goal=False,
            position=0.027,
        )

    goals, _ = _install_sim_actions(
        monkeypatch,
        settling_result,
        auto_complete=True,
    )
    backend = SimGripperCommandBackend(object(), "/gripper")
    backend.command(Phase.CLOSE_GRIPPER, 0.038)
    assert backend.latch_override() is True
    assert backend.telemetry()["last_result_span_m"] == pytest.approx(0.054)
    assert len(goals) == 2


def test_sim_controller_aborts_after_three_wide_stalls(monkeypatch):
    goals, _ = _install_sim_actions(
        monkeypatch,
        lambda goal: types.SimpleNamespace(
            stalled=True,
            reached_goal=False,
            position=0.036,
        ),
        auto_complete=True,
    )
    backend = SimGripperCommandBackend(object(), "/gripper")
    backend.command(Phase.CLOSE_GRIPPER, 0.038)
    assert backend.abort_close() is True
    assert backend.latch_override() is False
    assert len(goals) == 3
    sent_before = len(goals)
    backend.command(Phase.CLOSE_GRIPPER, 0.030)
    assert len(goals) == sent_before
