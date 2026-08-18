from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from franka_emika_panda_policy.config import Phase, StudentV6Config
from franka_emika_panda_policy.dls import (
    DLSController,
    lock_wrist_to_home,
    normalize_dls_descend_target,
    rewrite_cube_geometry_to_world_anchor,
    update_hover_cube_anchor,
)
from franka_emika_panda_policy.integration import denormalize_action, normalize_target


def test_dls_engages_hover_after_warmup_and_descend_immediately():
    config = StudentV6Config()
    controller = DLSController(config, np.zeros((6, 7)))
    assert not controller.should_engage(Phase.HOVER_RED, 29)
    assert controller.should_engage(Phase.HOVER_RED, 30)
    assert controller.should_engage(Phase.DESCEND_GRASP, 0)
    assert controller.should_engage(Phase.DESCEND_GRASP, 30)
    assert not controller.should_engage(Phase.CLOSE_GRIPPER, 0)
    assert not controller.should_engage(Phase.LIFT, 100)
    with pytest.raises(ValueError):
        controller.compute_action(
            phase=Phase.HOVER_RED,
            phase_step=29,
            q=config.home_joint_position,
            qvel=np.zeros(7),
            geometry=np.zeros(6),
        )


def test_dls_matches_promoted_translational_numerics():
    config = StudentV6Config()
    jacobian = np.zeros((6, 7))
    jacobian[:3, :3] = np.eye(3)
    controller = DLSController(config, jacobian)
    q = np.asarray(config.home_joint_position)
    geometry = np.asarray((0.01 / 0.75, 0.0, -0.01 / 0.75, 0.0, 0.0, 0.0))

    action = controller.compute_action(
        phase=Phase.DESCEND_GRASP,
        phase_step=30,
        q=q,
        qvel=np.zeros(7),
        geometry=geometry,
    )
    raw_target = denormalize_action(action, config.joint_lower, config.joint_upper)
    expected_qdot_x = (config.dls_cartesian_gain * 0.01) / (
        1.0 + config.dls_damping**2
    )
    assert np.isfinite(action).all()
    assert (raw_target[0] - q[0]) / config.response_time == pytest.approx(
        expected_qdot_x, rel=2.0e-5
    )
    np.testing.assert_allclose(raw_target[1:], q[1:], atol=4.0e-7)


def test_singular_dls_is_finite_and_callback_receives_q():
    config = StudentV6Config()
    calls = []

    def jacobian(q):
        calls.append(q.copy())
        return np.zeros((6, 7))

    controller = DLSController(config, jacobian)
    action = controller.compute_action(
        phase=Phase.HOVER_RED,
        phase_step=30,
        q=config.home_joint_position,
        qvel=np.zeros(7),
        geometry=np.zeros(6),
    )
    assert np.isfinite(action).all()
    assert len(calls) == 1


def test_hover_cube_anchor_blends_then_clamps_near_spawn():
    first = update_hover_cube_anchor(
        None,
        valid=False,
        observed_cube_m=(0.51, 0.16, 0.026),
        alpha=0.1,
        clamp_radius_m=0.04,
    )
    np.testing.assert_allclose(first, (0.51, 0.16, 0.026))
    blended = update_hover_cube_anchor(
        first,
        valid=True,
        observed_cube_m=(0.50, 0.15, 0.025),
        alpha=0.1,
        clamp_radius_m=0.04,
    )
    np.testing.assert_allclose(blended, (0.509, 0.159, 0.0259))
    clamped = update_hover_cube_anchor(
        None,
        valid=False,
        observed_cube_m=(0.8, -0.4, 0.2),
        alpha=0.1,
        clamp_radius_m=0.04,
    )
    np.testing.assert_allclose(clamped, (0.54, 0.11, 0.04))


def test_rewritten_geometry_makes_dls_aim_at_frozen_world_cube():
    config = StudentV6Config()
    tcp = np.asarray((0.46, 0.15, 0.035))
    live = np.asarray((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    frozen = rewrite_cube_geometry_to_world_anchor(
        live, tcp, (0.50, 0.15, 0.025), config.geometry_scale_m
    )
    np.testing.assert_allclose(tcp + frozen[:3] * config.geometry_scale_m, (0.50, 0.15, 0.025))
    jacobian = np.zeros((6, 7))
    jacobian[:3, :3] = np.eye(3)
    controller = DLSController(config, jacobian)
    action = controller.compute_action(
        phase=Phase.DESCEND_GRASP,
        phase_step=30,
        q=config.home_joint_position,
        qvel=np.zeros(7),
        geometry=frozen,
    )
    raw_target = denormalize_action(action, config.joint_lower, config.joint_upper)
    assert raw_target[0] > config.home_joint_position[0]


def test_descend_target_names_are_canonical():
    assert normalize_dls_descend_target("SIM_BOX") == "sim_box"
    assert normalize_dls_descend_target("hover_estimate") == "hover_estimate"
    assert normalize_dls_descend_target("live") == "live"
    with pytest.raises(ValueError, match="dls_descend_target"):
        normalize_dls_descend_target("freeze_through_close")


def test_lock_wrist_to_home_replaces_joint_seven():
    config = StudentV6Config()
    action = np.zeros(7, dtype=np.float32)
    locked = lock_wrist_to_home(action, config)
    expected = normalize_target(
        config.home_joint_position, config.joint_lower, config.joint_upper
    )
    assert locked[6] == pytest.approx(expected[6])
    np.testing.assert_allclose(locked[:6], 0.0)
