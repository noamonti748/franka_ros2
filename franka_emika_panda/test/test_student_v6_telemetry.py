from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))

from franka_emika_panda_policy.synchronization import ObservationSnapshot
from franka_emika_panda_policy.telemetry import (
    COMMAND_TELEMETRY_FIELDS,
    build_command_telemetry,
)


def _snapshot() -> ObservationSnapshot:
    return ObservationSnapshot(
        frame_sequence=9,
        image_stamp_s=4.0,
        pixels=np.zeros((1, 64, 64, 3), dtype=np.float32),
        q=np.arange(7, dtype=np.float64),
        qvel=np.arange(7, dtype=np.float64) * 0.1,
        tcp_position_m=np.zeros(3),
        tcp_rotation=np.eye(3),
        finger_span_m=0.05,
        image_callback_duration_s=0.001,
        image_age_s=0.02,
        joint_age_s=0.02,
        tcp_tf_age_s=0.02,
        gripper_age_s=0.02,
        joint_image_skew_s=0.005,
        tcp_image_skew_s=0.004,
        gripper_image_skew_s=0.01,
        max_input_skew_s=0.01,
        joint_interpolated=True,
        tcp_interpolated=True,
        gripper_interpolated=False,
        missed_frames=2,
        total_missed_frames=3,
    )


def test_command_telemetry_contains_timing_inputs_actions_and_limiter_details():
    snapshot = _snapshot()
    smoothed = np.full(7, 0.5)
    limited = smoothed.copy()
    limited[0] = 0.4
    payload = build_command_telemetry(
        snapshot=snapshot,
        sim_timer_delta_s=1.0 / 60.0,
        wall_timer_delta_s=0.017,
        policy_step_delta_s=1.0 / 60.0,
        control_callback_duration_s=0.006,
        inference_duration_s=0.003,
        dls_duration_s=0.001,
        raw_action=np.linspace(-1.0, 1.0, 7),
        selected_action=np.linspace(-0.5, 0.5, 7),
        integrated_target=np.full(7, 0.6),
        smoothed_target=smoothed,
        limited_command=limited,
        published_command=limited,
    )

    assert COMMAND_TELEMETRY_FIELDS <= payload.keys()
    assert payload['frame_sequence'] == 9
    assert payload['missed_frame']
    assert payload['missed_frame_count'] == 2
    assert not payload['reused_frame']
    assert payload['limiter_intervened']
    assert np.isclose(payload['limiter_max_abs_delta'], 0.1)
    np.testing.assert_allclose(
        payload['tracking_error'],
        limited - snapshot.q,
    )
    assert payload['raw_action'] == np.linspace(-1.0, 1.0, 7).tolist()
    assert payload['integrated_target'] == [0.6] * 7
    assert payload['smoothed_target'] == [0.5] * 7
    assert payload['limited_command'] == limited.tolist()
    assert payload['published_command'] == limited.tolist()
