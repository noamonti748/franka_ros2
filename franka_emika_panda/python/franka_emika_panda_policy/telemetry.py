"""ROS-free construction of detailed policy command telemetry."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .synchronization import ObservationSnapshot


COMMAND_TELEMETRY_FIELDS = frozenset(
    {
        'sim_timer_delta_s',
        'wall_timer_delta_s',
        'policy_step_delta_s',
        'image_callback_duration_s',
        'control_callback_duration_s',
        'inference_duration_s',
        'dls_duration_s',
        'image_age_s',
        'joint_age_s',
        'tcp_tf_age_s',
        'gripper_age_s',
        'joint_image_skew_s',
        'tcp_image_skew_s',
        'gripper_image_skew_s',
        'max_input_skew_s',
        'raw_action',
        'selected_action',
        'integrated_target',
        'smoothed_target',
        'limited_command',
        'published_command',
        'q',
        'qvel',
        'tracking_error',
        'limiter_intervened',
        'limiter_delta',
        'limiter_max_abs_delta',
        'frame_sequence',
        'image_stamp_s',
        'missed_frame',
        'missed_frame_count',
        'total_missed_frames',
        'reused_frame',
    }
)


def _vector(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (7,) or not np.all(np.isfinite(array)):
        raise ValueError(f'{name} must be a finite seven-vector')
    return array


def _duration(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    number = float(value)
    if number < 0.0 or not np.isfinite(number):
        raise ValueError(f'{name} must be finite and nonnegative')
    return number


def build_command_telemetry(
    *,
    snapshot: ObservationSnapshot,
    sim_timer_delta_s: float | None,
    wall_timer_delta_s: float | None,
    policy_step_delta_s: float | None,
    control_callback_duration_s: float,
    inference_duration_s: float,
    dls_duration_s: float,
    raw_action: Sequence[float],
    selected_action: Sequence[float],
    integrated_target: Sequence[float],
    smoothed_target: Sequence[float],
    limited_command: Sequence[float],
    published_command: Sequence[float],
) -> dict[str, Any]:
    """Return the stable command-telemetry schema used by the ROS adapter."""

    raw = _vector(raw_action, 'raw_action')
    selected = _vector(selected_action, 'selected_action')
    integrated = _vector(integrated_target, 'integrated_target')
    smoothed = _vector(smoothed_target, 'smoothed_target')
    limited = _vector(limited_command, 'limited_command')
    published = _vector(published_command, 'published_command')
    q = _vector(snapshot.q, 'q')
    qvel = _vector(snapshot.qvel, 'qvel')
    limiter_delta = limited - smoothed
    payload: dict[str, Any] = {
        'sim_timer_delta_s': _duration(sim_timer_delta_s, 'sim_timer_delta_s'),
        'wall_timer_delta_s': _duration(wall_timer_delta_s, 'wall_timer_delta_s'),
        'policy_step_delta_s': _duration(
            policy_step_delta_s, 'policy_step_delta_s'
        ),
        'image_callback_duration_s': _duration(
            snapshot.image_callback_duration_s,
            'image_callback_duration_s',
        ),
        'control_callback_duration_s': _duration(
            control_callback_duration_s,
            'control_callback_duration_s',
        ),
        'inference_duration_s': _duration(inference_duration_s, 'inference_duration_s'),
        'dls_duration_s': _duration(dls_duration_s, 'dls_duration_s'),
        'image_age_s': float(snapshot.image_age_s),
        'joint_age_s': float(snapshot.joint_age_s),
        'tcp_tf_age_s': float(snapshot.tcp_tf_age_s),
        'gripper_age_s': float(snapshot.gripper_age_s),
        'joint_image_skew_s': float(snapshot.joint_image_skew_s),
        'tcp_image_skew_s': float(snapshot.tcp_image_skew_s),
        'gripper_image_skew_s': float(snapshot.gripper_image_skew_s),
        'max_input_skew_s': float(snapshot.max_input_skew_s),
        'joint_interpolated': snapshot.joint_interpolated,
        'tcp_interpolated': snapshot.tcp_interpolated,
        'gripper_interpolated': snapshot.gripper_interpolated,
        'raw_action': raw.tolist(),
        'selected_action': selected.tolist(),
        'integrated_target': integrated.tolist(),
        'smoothed_target': smoothed.tolist(),
        'limited_command': limited.tolist(),
        'published_command': published.tolist(),
        'q': q.tolist(),
        'qvel': qvel.tolist(),
        'tracking_error': (published - q).tolist(),
        'limiter_intervened': bool(np.any(np.abs(limiter_delta) > 1.0e-12)),
        'limiter_delta': limiter_delta.tolist(),
        'limiter_max_abs_delta': float(np.max(np.abs(limiter_delta))),
        'frame_sequence': snapshot.frame_sequence,
        'image_stamp_s': snapshot.image_stamp_s,
        'missed_frame': snapshot.missed_frames > 0,
        'missed_frame_count': snapshot.missed_frames,
        'total_missed_frames': snapshot.total_missed_frames,
        'reused_frame': False,
    }
    if not COMMAND_TELEMETRY_FIELDS.issubset(payload):
        raise AssertionError('internal telemetry schema error')
    return payload
