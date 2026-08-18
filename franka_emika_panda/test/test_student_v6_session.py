from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from franka_emika_panda_policy.config import Phase, StudentV6Config
from franka_emika_panda_policy.estimator import Estimate, update_finger_span_latch
from franka_emika_panda_policy.session import (
    ObservableSession,
    scripted_gripper_target,
)


def _step(
    session,
    geometry,
    *,
    tcp_position_m=None,
    finger_span=0.08,
    gripper_openness=None,
    require_grasp_alignment=False,
):
    return session.step(
        geometry=geometry,
        tcp_position_m=np.zeros(3) if tcp_position_m is None else tcp_position_m,
        tcp_rotation=np.eye(3),
        joint_velocity=np.zeros(7),
        finger_span_m=finger_span,
        gripper_openness=(
            finger_span / 0.08
            if gripper_openness is None
            else gripper_openness
        ),
        require_grasp_alignment=require_grasp_alignment,
    )


def test_descend_grasp_uses_promoted_10mm_observable_margin():
    geometry = np.asarray((0.02 / 0.75, 0.0, -0.01 / 0.75, 0.0, 0.0, 0.0))
    promoted = ObservableSession(StudentV6Config())
    promoted.phase = Phase.DESCEND_GRASP
    promoted_result = _step(promoted, geometry)
    assert promoted_result.gate

    no_margin_config = replace(
        StudentV6Config(), observable_position_gate_margins=(0.0,) * 8
    )
    no_margin = ObservableSession(no_margin_config)
    no_margin.phase = Phase.DESCEND_GRASP
    assert not _step(no_margin, geometry).gate


def test_frozen_descend_requires_20mm_xy_and_5mm_z_alignment():
    session = ObservableSession(StudentV6Config())
    session.phase = Phase.DESCEND_GRASP
    aligned = np.asarray((0.019 / 0.75, 0.0, -0.01 / 0.75, 0.0, 0.0, 0.0))
    assert _step(session, aligned, require_grasp_alignment=True).gate

    session.phase = Phase.DESCEND_GRASP
    session.dwell = 0
    wide = np.asarray((0.021 / 0.75, 0.0, -0.01 / 0.75, 0.0, 0.0, 0.0))
    blocked_xy = _step(session, wide, require_grasp_alignment=True)
    assert not blocked_xy.gate
    assert blocked_xy.telemetry["grasp_xy_m"] == pytest.approx(0.021)

    session.phase = Phase.DESCEND_GRASP
    session.dwell = 0
    high = np.asarray((0.006 / 0.75, 0.0, 0.01 / 0.75, 0.0, 0.0, 0.0))
    blocked_z = _step(session, high, require_grasp_alignment=True)
    assert not blocked_z.gate
    assert blocked_z.telemetry["grasp_z_error_m"] == pytest.approx(0.02)


def test_alignment_cube_overrides_estimated_grasp_xy():
    session = ObservableSession(StudentV6Config())
    session.phase = Phase.DESCEND_GRASP
    geometry = np.asarray((0.0, 0.0, -0.01 / 0.75, 0.0, 0.0, 0.0))
    tcp = np.asarray((0.50, 0.15, 0.035))
    missed = session.step(
        geometry=geometry,
        tcp_position_m=tcp,
        tcp_rotation=np.eye(3),
        joint_velocity=np.zeros(7),
        finger_span_m=0.08,
        gripper_openness=1.0,
        require_grasp_alignment=True,
        alignment_cube_m=(0.53, 0.15, 0.025),
    )
    assert not missed.gate
    assert missed.telemetry["grasp_xy_m"] == pytest.approx(0.03)

    session.phase = Phase.DESCEND_GRASP
    session.dwell = 0
    aligned = session.step(
        geometry=geometry,
        tcp_position_m=tcp,
        tcp_rotation=np.eye(3),
        joint_velocity=np.zeros(7),
        finger_span_m=0.08,
        gripper_openness=1.0,
        require_grasp_alignment=True,
        alignment_cube_m=(0.50, 0.15, 0.025),
    )
    assert aligned.gate
    assert aligned.telemetry["grasp_xy_m"] == pytest.approx(0.0)


def test_hover_clearance_blocks_until_open_fingers_clear_the_cube():
    session = ObservableSession(StudentV6Config())
    geometry = np.asarray((0.0, 0.0, -0.075 / 0.75, 0.0, 0.0, 0.0))
    tcp = np.asarray((0.50, 0.15, 0.100))
    blocked = session.step(
        geometry=geometry,
        tcp_position_m=tcp,
        tcp_rotation=np.eye(3),
        joint_velocity=np.zeros(7),
        finger_span_m=0.08,
        gripper_openness=1.0,
        require_hover_xy_clearance=True,
        alignment_cube_m=(0.52, 0.15, 0.025),
    )
    assert not blocked.gate
    assert blocked.telemetry["grasp_xy_m"] == pytest.approx(0.02)

    session.dwell = 0
    cleared = session.step(
        geometry=geometry,
        tcp_position_m=tcp,
        tcp_rotation=np.eye(3),
        joint_velocity=np.zeros(7),
        finger_span_m=0.08,
        gripper_openness=1.0,
        require_hover_xy_clearance=True,
        alignment_cube_m=(0.51, 0.15, 0.025),
    )
    assert cleared.gate
    assert cleared.telemetry["grasp_xy_m"] == pytest.approx(0.01)


def test_settled_miss_of_alignment_cube_retries_after_45_steps():
    session = ObservableSession(StudentV6Config())
    session.phase = Phase.DESCEND_GRASP
    geometry = np.asarray((0.0, 0.0, -0.01 / 0.75, 0.0, 0.0, 0.0))
    tcp = np.asarray((0.50, 0.15, 0.035))
    result = None
    for _ in range(45):
        result = session.step(
            geometry=geometry,
            tcp_position_m=tcp,
            tcp_rotation=np.eye(3),
            joint_velocity=np.zeros(7),
            finger_span_m=0.08,
            gripper_openness=1.0,
            require_grasp_alignment=True,
            alignment_cube_m=(0.53, 0.15, 0.025),
        )
        if result.grasp_retry:
            break
        session.phase = Phase.DESCEND_GRASP
    assert result is not None
    assert result.grasp_retry
    assert result.phase == Phase.HOVER_RED
    assert session.grasp_retries == 1
    session = ObservableSession(StudentV6Config())
    session.phase = Phase.DESCEND_GRASP
    session.phase_step = session.config.phase_step_limits[int(Phase.DESCEND_GRASP)] - 1
    geometry = np.asarray((0.04 / 0.75, 0.0, -0.01 / 0.75, 0.0, 0.0, 0.0))
    result = _step(session, geometry, require_grasp_alignment=True)
    assert result.grasp_retry
    assert result.phase == Phase.HOVER_RED
    assert session.grasp_retries == 1


def test_grasp_timeout_retries_from_hover_red_and_clears_latch():
    session = ObservableSession(StudentV6Config())
    session.phase = Phase.CLOSE_GRIPPER
    session.phase_step = session.config.phase_step_limits[int(Phase.CLOSE_GRIPPER)] - 1

    result = _step(session, np.zeros(6))

    assert result.grasp_retry
    assert result.phase == Phase.HOVER_RED
    assert result.phase_step == 0
    assert session.grasp_retries == 1
    assert not session.estimator.grasp_latched


def test_descend_place_recovery_returns_to_hover_green():
    session = ObservableSession(StudentV6Config())
    session.phase = Phase.DESCEND_PLACE
    session.phase_step = session.config.observable_descend_recovery_steps - 1
    session.estimator.grasp_latched = True
    geometry = np.asarray((0.0, 0.0, 0.0, 0.2, 0.0, 0.0))

    result = _step(session, geometry, finger_span=0.05)

    assert result.placement_retry
    assert result.phase == Phase.HOVER_GREEN
    assert session.placement_retries == 1


def test_width_latch_dwell_release_and_scripted_gripper_timing():
    latched = False
    dwell = age = 0
    for index in range(5):
        update = update_finger_span_latch(
            0.05,
            latched,
            dwell,
            age,
            object_width_m=0.05,
            width_tolerance_m=0.005,
            dwell_steps=5,
            timeout_steps=900,
        )
        latched, dwell, age = update.latched, update.dwell, update.age
        assert update.newly_latched == (index == 4)
    released = update_finger_span_latch(
        0.04,
        latched,
        dwell,
        age,
        object_width_m=0.05,
        width_tolerance_m=0.005,
        dwell_steps=5,
        timeout_steps=900,
    )
    assert released.released and not released.latched

    width = 0.04
    for _ in range(20):
        width = scripted_gripper_target(width, Phase.CLOSE_GRIPPER)
    assert width == 0.0
    for _ in range(8):
        width = scripted_gripper_target(width, Phase.RELEASE)
    assert width == 0.04


def test_release_requires_ninety_percent_open_and_placed_dwells_exactly():
    session = ObservableSession(
        replace(StudentV6Config(), success_dwell_steps=3)
    )
    session.phase = Phase.RELEASE
    estimate = Estimate(
        cube_position_m=np.zeros(3),
        goal_position_m=np.zeros(3),
        valid=True,
        settled=True,
        grasp_latched=False,
        newly_latched=False,
        released=True,
    )
    session.estimator.update = lambda **kwargs: estimate

    blocked = _step(session, np.zeros(6), gripper_openness=0.89)
    assert not blocked.gate
    assert blocked.phase == Phase.RELEASE

    released = _step(session, np.zeros(6), gripper_openness=0.9)
    assert released.gate and released.transitioned
    assert released.phase == Phase.PLACED
    assert released.telemetry["release_open_fraction"] == 0.9

    first = _step(session, np.zeros(6), gripper_openness=1.0)
    second = _step(session, np.zeros(6), gripper_openness=1.0)
    third = _step(session, np.zeros(6), gripper_openness=1.0)
    assert first.telemetry["placed_dwell"] == 1 and not first.success
    assert second.telemetry["placed_dwell"] == 2 and not second.success
    assert third.telemetry["placed_dwell"] == 3
    assert third.success and third.done
