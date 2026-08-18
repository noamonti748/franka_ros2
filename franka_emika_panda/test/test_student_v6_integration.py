from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from franka_emika_panda_policy.config import StudentV6Config
from franka_emika_panda_policy.integration import (
    IntegratedActionController,
    advance_integrated_velocity_target,
)


def test_integrator_respects_joint_bounds_lead_and_rate():
    config = StudentV6Config()
    controller = IntegratedActionController(config)
    q = np.asarray(config.home_joint_position)
    controller.reset(q)

    command = controller.step(np.ones(7), q)

    velocity_limit = np.asarray(config.rated_joint_velocity) * config.velocity_limit_fraction
    assert np.all(command >= np.asarray(config.joint_lower))
    assert np.all(command <= np.asarray(config.joint_upper))
    assert np.all(np.abs(command - q) <= velocity_limit / config.control_hz + 1.0e-12)
    lead = velocity_limit / config.control_hz * config.integrated_lead_steps
    assert np.all(np.abs(controller.integrated_target - q) <= lead + 1.0e-12)


def test_integrated_velocity_port_clips_to_hardware_limits():
    config = StudentV6Config()
    upper = np.asarray(config.joint_upper)
    result = advance_integrated_velocity_target(
        upper + 10.0,
        upper,
        upper,
        np.ones(7),
        config.joint_lower,
        config.joint_upper,
        control_hz=60.0,
        response_time=0.5,
        lead_steps=4.0,
    )
    np.testing.assert_allclose(result, upper)


def test_nonfinite_action_is_rejected():
    config = StudentV6Config()
    controller = IntegratedActionController(config)
    with np.testing.assert_raises(ValueError):
        controller.step([np.nan] * 7, config.home_joint_position)
