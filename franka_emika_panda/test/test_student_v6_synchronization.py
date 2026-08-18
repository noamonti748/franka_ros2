from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))

from franka_emika_panda_policy.synchronization import SynchronizedObservationBuffer


def _pixels(value: float = 0.0) -> np.ndarray:
    return np.full((1, 64, 64, 3), value, dtype=np.float32)


def _add_bracketing_samples(
    synchronizer: SynchronizedObservationBuffer,
    left_stamp: float,
    right_stamp: float,
) -> None:
    synchronizer.add_joint(left_stamp, np.zeros(7), np.ones(7))
    synchronizer.add_joint(right_stamp, np.full(7, 2.0), np.full(7, 3.0))
    synchronizer.add_tcp(left_stamp, np.zeros(3), np.eye(3))
    synchronizer.add_tcp(
        right_stamp,
        np.full(3, 2.0),
        np.diag((-1.0, -1.0, 1.0)),
    )
    synchronizer.add_gripper(left_stamp, 0.04)
    synchronizer.add_gripper(right_stamp, 0.06)


def test_image_driven_snapshot_interpolates_all_timestamped_inputs():
    synchronizer = SynchronizedObservationBuffer(max_samples=8, max_skew_s=0.06)
    _add_bracketing_samples(synchronizer, 1.0, 1.1)
    synchronizer.add_image(1.05, _pixels(7.0), callback_duration_s=0.002)

    attempt = synchronizer.try_snapshot(1.08)

    assert attempt.reason == ''
    assert attempt.snapshot is not None
    snapshot = attempt.snapshot
    np.testing.assert_allclose(snapshot.q, np.ones(7))
    np.testing.assert_allclose(snapshot.qvel, np.full(7, 2.0))
    np.testing.assert_allclose(snapshot.tcp_position_m, np.ones(3))
    np.testing.assert_allclose(
        snapshot.tcp_rotation,
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        atol=1.0e-12,
    )
    assert np.isclose(snapshot.finger_span_m, 0.05)
    assert snapshot.joint_interpolated
    assert snapshot.tcp_interpolated
    assert snapshot.gripper_interpolated
    assert np.isclose(snapshot.max_input_skew_s, 0.05)
    assert np.isclose(snapshot.image_age_s, 0.03)
    assert np.isclose(snapshot.image_callback_duration_s, 0.002)


def test_excessive_skew_holds_frame_until_a_nearby_sample_arrives():
    synchronizer = SynchronizedObservationBuffer(max_samples=8, max_skew_s=0.02)
    synchronizer.add_joint(1.9, np.zeros(7), np.zeros(7))
    synchronizer.add_tcp(2.0, np.zeros(3), np.eye(3))
    synchronizer.add_gripper(2.0, 0.05)
    synchronizer.add_image(2.0, _pixels())

    held = synchronizer.try_snapshot(2.01)
    assert held.snapshot is None
    assert held.reason == 'excessive_joint_skew'
    assert not held.frame_reused
    assert np.isclose(held.joint_image_skew_s, 0.1)
    assert np.isclose(held.image_age_s, 0.01)

    synchronizer.add_joint(2.0, np.ones(7), np.full(7, 0.5))
    accepted = synchronizer.try_snapshot(2.02)
    assert accepted.snapshot is not None
    np.testing.assert_allclose(accepted.snapshot.q, np.ones(7))

    reused = synchronizer.try_snapshot(2.03)
    assert reused.snapshot is None
    assert reused.reason == 'no_fresh_image'
    assert reused.frame_reused


def test_latest_pending_image_executes_once_and_reports_missed_frames():
    synchronizer = SynchronizedObservationBuffer(max_samples=8, max_skew_s=0.02)
    for stamp in (3.0, 3.01, 3.02):
        synchronizer.add_joint(stamp, np.full(7, stamp), np.zeros(7))
        synchronizer.add_tcp(stamp, np.full(3, stamp), np.eye(3))
        synchronizer.add_gripper(stamp, 0.05)
        synchronizer.add_image(stamp, _pixels(stamp))

    attempt = synchronizer.try_snapshot(3.03)

    assert attempt.snapshot is not None
    assert attempt.snapshot.image_stamp_s == 3.02
    assert attempt.snapshot.missed_frames == 2
    assert attempt.snapshot.total_missed_frames == 2
    assert synchronizer.try_snapshot(3.04).frame_reused
