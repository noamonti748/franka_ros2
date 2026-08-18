from dataclasses import fields
from pathlib import Path
import inspect
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from franka_emika_panda_policy.config import PANDA_HOME
from franka_emika_panda_policy.node import (
    create_node_class,
    extract_joint_measurements,
)
from franka_emika_panda_policy.safety import (
    CommandLimiter,
    ConsecutiveReadiness,
    GraspRetryRecovery,
    HomeReadiness,
    RecoveryStage,
    may_publish_fault_hold,
)


def test_monitor_only_never_publishes_fault_hold():
    assert not may_publish_fault_hold(False, False)
    assert not may_publish_fault_hold(False, True)
    assert not may_publish_fault_hold(True, False)
    assert may_publish_fault_hold(True, True)


def test_measured_home_requires_consecutive_enabled_ticks():
    gate = HomeReadiness(np.asarray(PANDA_HOME), 0.03, 0.10, 3)
    assert not gate.update(
        policy_enabled=False, q=PANDA_HOME, qvel=np.zeros(7)
    )
    assert not gate.update(
        policy_enabled=True, q=PANDA_HOME, qvel=np.zeros(7)
    )
    assert not gate.update(
        policy_enabled=True, q=PANDA_HOME, qvel=np.zeros(7)
    )
    assert gate.update(
        policy_enabled=True, q=PANDA_HOME, qvel=np.zeros(7)
    )
    away = np.asarray(PANDA_HOME).copy()
    away[1] += 0.04
    assert not gate.update(policy_enabled=True, q=away, qvel=np.zeros(7))


def test_readiness_and_command_bounds_are_enforced():
    readiness = ConsecutiveReadiness(2)
    assert not readiness.update(True)
    assert readiness.update(True)
    assert not readiness.update(False)

    limiter = CommandLimiter(
        lower=-np.ones(7),
        upper=np.ones(7),
        velocity_limit_rad_s=np.full(7, 0.6),
        max_step_rad=np.full(7, 0.02),
        control_hz=60.0,
    )
    command = limiter.limit(np.full(7, 3.0), np.zeros(7))
    np.testing.assert_allclose(command, np.full(7, 0.01))
    assert np.all(command <= 1.0)


def test_arm_only_joint_state_is_accepted_and_gripper_is_optional():
    arm_names = tuple(f"panda_joint{index}" for index in range(1, 8))
    finger_names = ("panda_finger_joint1", "panda_finger_joint2")
    q, qvel, span = extract_joint_measurements(
        arm_names,
        tuple(range(7)),
        (),
        arm_names,
        finger_names,
    )
    np.testing.assert_allclose(q, np.arange(7))
    np.testing.assert_allclose(qvel, np.zeros(7))
    assert span is None

    _, _, span = extract_joint_measurements(
        finger_names,
        (0.021, 0.022),
        (),
        arm_names,
        finger_names,
    )
    assert span == 0.043


def test_grasp_retry_recovery_retreats_vertically_then_opens_and_reobserves():
    recovery = GraspRetryRecovery(
        retreat_height_m=0.075,
        position_tolerance_m=0.005,
        max_joint_speed_rad_s=0.10,
        settled_steps=2,
        open_fraction=0.9,
        fresh_estimate_steps=1,
    )
    start = np.asarray((0.42, -0.15, 0.12))
    recovery.start(start, estimate_generation=7)

    moving = recovery.step(
        tcp_position_m=start,
        joint_velocity=np.full(7, 0.02),
        gripper_openness=0.2,
        grasp_latched=False,
        estimate_generation=8,
        estimate_valid=True,
    )
    np.testing.assert_allclose(
        moving.target_position_m, start + np.asarray((0.0, 0.0, 0.075))
    )
    np.testing.assert_allclose(moving.target_position_m[:2], start[:2])
    assert moving.stage == RecoveryStage.RETREAT
    assert not moving.request_open

    target = moving.target_position_m
    recovery.step(
        tcp_position_m=target,
        joint_velocity=np.zeros(7),
        gripper_openness=0.2,
        grasp_latched=False,
        estimate_generation=9,
        estimate_valid=True,
    )
    opened = recovery.step(
        tcp_position_m=target,
        joint_velocity=np.zeros(7),
        gripper_openness=0.2,
        grasp_latched=False,
        estimate_generation=10,
        estimate_valid=True,
    )
    assert opened.stage == RecoveryStage.OPEN
    assert opened.request_open

    waiting = recovery.step(
        tcp_position_m=target,
        joint_velocity=np.zeros(7),
        gripper_openness=0.95,
        grasp_latched=False,
        estimate_generation=10,
        estimate_valid=True,
    )
    assert waiting.stage == RecoveryStage.WAIT_FRESH_ESTIMATE
    assert not waiting.complete

    complete = recovery.step(
        tcp_position_m=target,
        joint_velocity=np.zeros(7),
        gripper_openness=0.95,
        grasp_latched=False,
        estimate_generation=11,
        estimate_valid=True,
    )
    assert complete.complete
    assert complete.stage == RecoveryStage.COMPLETE
    assert not recovery.active
    assert complete.telemetry["wait_estimate_generation"] == 10
    assert complete.telemetry["estimate_fresh"] is True


def test_grasp_retry_recovery_never_owns_cube_state():
    names = {field.name for field in fields(GraspRetryRecovery)}
    assert not any("cube" in name or "home" in name for name in names)


def test_canonical_integration_parameters_are_declared():
    source = inspect.getsource(create_node_class)
    required = (
        "policy_enabled",
        "onnx_model_path",
        "model_path",
        "robot_id",
        "arm_id",
        "base_frame",
        "tcp_frame",
        "camera_frame",
        "joint_states_topic",
        "gripper_state_topic",
        "image_topic",
        "image_subscription_depth",
        "image_qos_reliability",
        "command_topic",
        "arm_command_topic",
        "command_message_type",
        "telemetry_topic",
        "control_hz",
        "action_latency_steps",
        "response_time_s",
        "snapshot_poll_hz",
        "trajectory_duration_s",
        "bypass_command_limiter",
        "dls_anchor_estimate",
        "dls_anchor_alpha",
        "dls_anchor_clamp_radius_m",
        "dls_descend_target",
        "box_state_topic",
        "joint_state_stale_timeout_s",
        "image_stale_timeout_s",
        "tcp_tf_stale_timeout_s",
        "gripper_state_stale_timeout_s",
        "use_pykdl",
        "robot_description",
        "robot_description_topic",
        "gripper_backend",
        "gripper_action_name",
        "sim_grasp_width_tolerance_m",
        "grasp_action_name",
        "move_action_name",
        "startup_home_enabled",
        "startup_home_command_enabled",
        "startup_home_trajectory_duration_s",
        "startup_home_tolerance_rad",
        "startup_home_velocity_tolerance_rad_s",
        "startup_home_hold_ticks",
        "seated_offset_m",
        "velocity_limit_fraction",
        "command_joint_lower_limits_rad",
        "command_joint_upper_limits_rad",
        "readiness_consecutive_ticks",
    )
    for name in required:
        assert f'"{name}"' in source
