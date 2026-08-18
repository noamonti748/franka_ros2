"""ROS-free timestamped observation synchronization.

Camera frames drive snapshot creation. Joint, TCP, and gripper samples are kept
in bounded, timestamp-ordered buffers and interpolated to the image timestamp.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import deque
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


def _finite_array(
    values: Sequence[float] | np.ndarray,
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f'{name} must be finite with shape {shape}')
    return array


def _finite_stamp(stamp_s: float) -> float:
    value = float(stamp_s)
    if not np.isfinite(value):
        raise ValueError('sample timestamp must be finite')
    return value


def _rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    matrix = _finite_array(rotation, (3, 3), 'rotation')
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            (
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            )
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.asarray(
                (
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                )
            )
        elif axis == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.asarray(
                (
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                )
            )
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.asarray(
                (
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                )
            )
    norm = float(np.linalg.norm(quaternion))
    if norm < 1.0e-12:
        raise ValueError('rotation produced an invalid quaternion')
    return quaternion / norm


def _quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion / np.linalg.norm(quaternion)
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _interpolate_transform(left: np.ndarray, right: np.ndarray, alpha: float) -> np.ndarray:
    position = left[:3] + alpha * (right[:3] - left[:3])
    first = left[3:].copy()
    second = right[3:].copy()
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        quaternion = first + alpha * (second - first)
        quaternion /= np.linalg.norm(quaternion)
    else:
        angle = float(np.arccos(dot))
        sine = float(np.sin(angle))
        quaternion = (
            np.sin((1.0 - alpha) * angle) / sine * first
            + np.sin(alpha * angle) / sine * second
        )
    return np.concatenate((position, quaternion))


@dataclass(frozen=True)
class _StampedValue:
    stamp_s: float
    value: np.ndarray


@dataclass(frozen=True)
class _Selection:
    value: np.ndarray
    stamp_s: float
    skew_s: float
    interpolated: bool


class _BoundedSeries:
    def __init__(self, max_samples: int) -> None:
        if max_samples < 2:
            raise ValueError('max_samples must be at least two')
        self._max_samples = int(max_samples)
        self._samples: list[_StampedValue] = []

    def add(self, stamp_s: float, value: np.ndarray) -> None:
        stamp = _finite_stamp(stamp_s)
        index = bisect_right([sample.stamp_s for sample in self._samples], stamp)
        self._samples.insert(index, _StampedValue(stamp, value.copy()))
        if len(self._samples) > self._max_samples:
            del self._samples[: len(self._samples) - self._max_samples]

    def select(
        self,
        stamp_s: float,
        interpolate: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
    ) -> _Selection | None:
        if not self._samples:
            return None
        stamps = [sample.stamp_s for sample in self._samples]
        right_index = bisect_right(stamps, stamp_s)
        if right_index == 0:
            sample = self._samples[0]
            return _Selection(
                sample.value.copy(),
                sample.stamp_s,
                abs(sample.stamp_s - stamp_s),
                False,
            )
        if right_index == len(self._samples):
            sample = self._samples[-1]
            return _Selection(
                sample.value.copy(),
                sample.stamp_s,
                abs(sample.stamp_s - stamp_s),
                False,
            )
        left = self._samples[right_index - 1]
        right = self._samples[right_index]
        if abs(left.stamp_s - stamp_s) <= 1.0e-12:
            return _Selection(left.value.copy(), left.stamp_s, 0.0, False)
        interval = right.stamp_s - left.stamp_s
        if interval <= 0.0:
            return _Selection(
                left.value.copy(),
                left.stamp_s,
                abs(left.stamp_s - stamp_s),
                False,
            )
        alpha = (stamp_s - left.stamp_s) / interval
        value = interpolate(left.value, right.value, alpha)
        support_skew = max(stamp_s - left.stamp_s, right.stamp_s - stamp_s)
        return _Selection(value, stamp_s, support_skew, True)


@dataclass(frozen=True)
class _ImageFrame:
    sequence: int
    stamp_s: float
    pixels: np.ndarray
    callback_duration_s: float


@dataclass(frozen=True)
class ObservationSnapshot:
    """A coherent set of policy inputs aligned to one camera timestamp."""

    frame_sequence: int
    image_stamp_s: float
    pixels: np.ndarray
    q: np.ndarray
    qvel: np.ndarray
    tcp_position_m: np.ndarray
    tcp_rotation: np.ndarray
    finger_span_m: float
    image_callback_duration_s: float
    image_age_s: float
    joint_age_s: float
    tcp_tf_age_s: float
    gripper_age_s: float
    joint_image_skew_s: float
    tcp_image_skew_s: float
    gripper_image_skew_s: float
    max_input_skew_s: float
    joint_interpolated: bool
    tcp_interpolated: bool
    gripper_interpolated: bool
    missed_frames: int
    total_missed_frames: int


@dataclass(frozen=True)
class SnapshotAttempt:
    snapshot: ObservationSnapshot | None
    reason: str
    frame_reused: bool
    missed_frames: int
    total_missed_frames: int
    frame_sequence: int | None
    image_age_s: float | None = None
    joint_age_s: float | None = None
    tcp_tf_age_s: float | None = None
    gripper_age_s: float | None = None
    joint_image_skew_s: float | None = None
    tcp_image_skew_s: float | None = None
    gripper_image_skew_s: float | None = None


class SynchronizedObservationBuffer:
    """Build at most one snapshot for each received image frame."""

    def __init__(self, *, max_samples: int = 120, max_skew_s: float = 0.05) -> None:
        if max_skew_s < 0.0 or not np.isfinite(max_skew_s):
            raise ValueError('max_skew_s must be finite and nonnegative')
        self.max_skew_s = float(max_skew_s)
        self._max_samples = int(max_samples)
        self._images: deque[_ImageFrame] = deque(maxlen=max_samples)
        self._joints = _BoundedSeries(max_samples)
        self._transforms = _BoundedSeries(max_samples)
        self._gripper = _BoundedSeries(max_samples)
        self._next_sequence = 0
        self._last_consumed_sequence = -1
        self._total_missed_frames = 0
        self._pending_overflow_missed = 0

    def discard_pending_images(self) -> None:
        """Prevent pre-session frames from driving a later policy step."""

        if self._images:
            self._last_consumed_sequence = self._images[-1].sequence
        self._images.clear()
        self._pending_overflow_missed = 0

    def add_image(
        self,
        stamp_s: float,
        pixels: np.ndarray,
        *,
        callback_duration_s: float = 0.0,
    ) -> int:
        stamp = _finite_stamp(stamp_s)
        array = np.asarray(pixels)
        if array.shape != (1, 64, 64, 3) or not np.all(np.isfinite(array)):
            raise ValueError('pixels must be finite with shape (1, 64, 64, 3)')
        duration = float(callback_duration_s)
        if duration < 0.0 or not np.isfinite(duration):
            raise ValueError('callback_duration_s must be finite and nonnegative')
        sequence = self._next_sequence
        self._next_sequence += 1
        if (
            len(self._images) == self._max_samples
            and self._images[0].sequence > self._last_consumed_sequence
        ):
            self._total_missed_frames += 1
            self._pending_overflow_missed += 1
        self._images.append(_ImageFrame(sequence, stamp, array.copy(), duration))
        return sequence

    def add_joint(
        self,
        stamp_s: float,
        q: Sequence[float],
        qvel: Sequence[float],
    ) -> None:
        value = np.concatenate(
            (
                _finite_array(q, (7,), 'q'),
                _finite_array(qvel, (7,), 'qvel'),
            )
        )
        self._joints.add(stamp_s, value)

    def add_tcp(
        self,
        stamp_s: float,
        position_m: Sequence[float],
        rotation: np.ndarray,
    ) -> None:
        position = _finite_array(position_m, (3,), 'tcp_position_m')
        quaternion = _rotation_to_quaternion(rotation)
        self._transforms.add(stamp_s, np.concatenate((position, quaternion)))

    def add_gripper(self, stamp_s: float, finger_span_m: float) -> None:
        value = _finite_array((finger_span_m,), (1,), 'finger_span_m')
        self._gripper.add(stamp_s, value)

    def _latest_pending_image(self) -> tuple[_ImageFrame | None, int]:
        pending = [
            frame for frame in self._images if frame.sequence > self._last_consumed_sequence
        ]
        if not pending:
            return None, 0
        skipped_in_buffer = max(len(pending) - 1, 0)
        missed = self._pending_overflow_missed + skipped_in_buffer
        self._pending_overflow_missed = 0
        if skipped_in_buffer:
            self._total_missed_frames += skipped_in_buffer
            for _ in range(skipped_in_buffer):
                self._images.popleft()
        return pending[-1], missed

    def try_snapshot(self, now_s: float) -> SnapshotAttempt:
        now = _finite_stamp(now_s)
        image, missed = self._latest_pending_image()
        if image is None:
            return SnapshotAttempt(
                None,
                'no_fresh_image',
                self._last_consumed_sequence >= 0,
                0,
                self._total_missed_frames,
                None,
            )

        linear = lambda left, right, alpha: left + alpha * (right - left)
        joint = self._joints.select(image.stamp_s, linear)
        tcp = self._transforms.select(image.stamp_s, _interpolate_transform)
        gripper = self._gripper.select(image.stamp_s, linear)
        for name, selection in (
            ('joint', joint),
            ('tcp_tf', tcp),
            ('gripper', gripper),
        ):
            if selection is None:
                return SnapshotAttempt(
                    None,
                    f'missing_{name}_sample',
                    False,
                    missed,
                    self._total_missed_frames,
                    image.sequence,
                    image_age_s=now - image.stamp_s,
                    joint_age_s=None if joint is None else now - joint.stamp_s,
                    tcp_tf_age_s=None if tcp is None else now - tcp.stamp_s,
                    gripper_age_s=None if gripper is None else now - gripper.stamp_s,
                    joint_image_skew_s=None if joint is None else joint.skew_s,
                    tcp_image_skew_s=None if tcp is None else tcp.skew_s,
                    gripper_image_skew_s=None if gripper is None else gripper.skew_s,
                )
            if selection.skew_s > self.max_skew_s:
                return SnapshotAttempt(
                    None,
                    f'excessive_{name}_skew',
                    False,
                    missed,
                    self._total_missed_frames,
                    image.sequence,
                    image_age_s=now - image.stamp_s,
                    joint_age_s=None if joint is None else now - joint.stamp_s,
                    tcp_tf_age_s=None if tcp is None else now - tcp.stamp_s,
                    gripper_age_s=None if gripper is None else now - gripper.stamp_s,
                    joint_image_skew_s=None if joint is None else joint.skew_s,
                    tcp_image_skew_s=None if tcp is None else tcp.skew_s,
                    gripper_image_skew_s=None if gripper is None else gripper.skew_s,
                )

        assert joint is not None and tcp is not None and gripper is not None
        self._last_consumed_sequence = image.sequence
        while self._images and self._images[0].sequence <= image.sequence:
            self._images.popleft()
        rotation = _quaternion_to_rotation(tcp.value[3:])
        max_skew = max(joint.skew_s, tcp.skew_s, gripper.skew_s)
        snapshot = ObservationSnapshot(
            frame_sequence=image.sequence,
            image_stamp_s=image.stamp_s,
            pixels=image.pixels.copy(),
            q=joint.value[:7].copy(),
            qvel=joint.value[7:].copy(),
            tcp_position_m=tcp.value[:3].copy(),
            tcp_rotation=rotation,
            finger_span_m=float(gripper.value[0]),
            image_callback_duration_s=image.callback_duration_s,
            image_age_s=now - image.stamp_s,
            joint_age_s=now - joint.stamp_s,
            tcp_tf_age_s=now - tcp.stamp_s,
            gripper_age_s=now - gripper.stamp_s,
            joint_image_skew_s=joint.skew_s,
            tcp_image_skew_s=tcp.skew_s,
            gripper_image_skew_s=gripper.skew_s,
            max_input_skew_s=max_skew,
            joint_interpolated=joint.interpolated,
            tcp_interpolated=tcp.interpolated,
            gripper_interpolated=gripper.interpolated,
            missed_frames=missed,
            total_missed_frames=self._total_missed_frames,
        )
        return SnapshotAttempt(
            snapshot,
            '',
            False,
            missed,
            self._total_missed_frames,
            image.sequence,
            image_age_s=snapshot.image_age_s,
            joint_age_s=snapshot.joint_age_s,
            tcp_tf_age_s=snapshot.tcp_tf_age_s,
            gripper_age_s=snapshot.gripper_age_s,
            joint_image_skew_s=snapshot.joint_image_skew_s,
            tcp_image_skew_s=snapshot.tcp_image_skew_s,
            gripper_image_skew_s=snapshot.gripper_image_skew_s,
        )
