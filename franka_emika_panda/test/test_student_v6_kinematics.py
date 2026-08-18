from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from franka_emika_panda_policy.config import PANDA_HOME
from franka_emika_panda_policy.kinematics import (
    PandaAnalyticJacobian,
    panda_hand_tcp_kinematics,
    panda_hand_tcp_pose,
)


def test_analytic_translation_jacobian_matches_finite_difference():
    q = np.asarray(PANDA_HOME, dtype=np.float64) + np.asarray(
        (0.1, -0.1, 0.08, -0.05, 0.07, -0.1, 0.06)
    )
    _, jacobian = panda_hand_tcp_kinematics(q)
    finite_difference = np.zeros((3, 7))
    epsilon = 1.0e-7
    for index in range(7):
        offset = np.zeros(7)
        offset[index] = epsilon
        plus, _ = panda_hand_tcp_kinematics(q + offset)
        minus, _ = panda_hand_tcp_kinematics(q - offset)
        finite_difference[:, index] = (plus - minus) / (2.0 * epsilon)

    assert jacobian.shape == (6, 7)
    np.testing.assert_allclose(
        jacobian[:3], finite_difference, rtol=1.0e-7, atol=1.0e-8
    )
    np.testing.assert_allclose(PandaAnalyticJacobian()(q), jacobian)
    pose_position, rotation = panda_hand_tcp_pose(q)
    np.testing.assert_allclose(pose_position, panda_hand_tcp_kinematics(q)[0])
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-12)
    assert np.isclose(np.linalg.det(rotation), 1.0)


def test_analytic_kinematics_rejects_invalid_joint_vectors():
    for invalid in (np.zeros(6), np.full(7, np.nan)):
        try:
            panda_hand_tcp_kinematics(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid Panda joints were accepted")
