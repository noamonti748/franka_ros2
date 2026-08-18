"""Exact 47-D actor state and calibrated policy-camera conversion."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan, degrees, radians, tan
from typing import Any, Sequence

import numpy as np

from .config import Phase


def _resize_nearest(image: np.ndarray, height: int = 64, width: int = 64) -> np.ndarray:
    if image.shape[:2] == (height, width):
        return image
    rows = np.floor(np.arange(height) * image.shape[0] / height).astype(np.int64)
    cols = np.floor(np.arange(width) * image.shape[1] / width).astype(np.int64)
    return image[rows[:, None], cols[None, :]]


def _area_weights(source_size: int, target_size: int) -> np.ndarray:
    """Return normalized box-filter weights for one resize axis."""
    if source_size <= 0 or target_size <= 0:
        raise ValueError("resize dimensions must be positive")
    scale = source_size / target_size
    weights = np.zeros((target_size, source_size), dtype=np.float32)
    for target in range(target_size):
        left = target * scale
        right = (target + 1) * scale
        first = int(np.floor(left))
        last = int(np.ceil(right))
        for source in range(first, last):
            overlap = max(0.0, min(right, source + 1.0) - max(left, source))
            if overlap:
                weights[target, source] = overlap / scale
    return weights


def _resize_area(image: np.ndarray, height: int = 64, width: int = 64) -> np.ndarray:
    """Antialiased area resize without an OpenCV runtime dependency."""
    if image.shape[:2] == (height, width):
        return image
    row_weights = _area_weights(image.shape[0], height)
    column_weights = _area_weights(image.shape[1], width)
    vertically_filtered = np.einsum("oh,hwc->owc", row_weights, image, optimize=True)
    return np.einsum(
        "owc,xw->oxc", vertically_filtered, column_weights, optimize=True
    )


def _bilinear_remap(image: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """Sample ``image`` at floating-point source coordinates."""
    height, width = image.shape[:2]
    valid = (
        (map_x >= 0.0)
        & (map_x <= width - 1.0)
        & (map_y >= 0.0)
        & (map_y <= height - 1.0)
    )
    x0 = np.floor(np.clip(map_x, 0.0, width - 1.0)).astype(np.int64)
    y0 = np.floor(np.clip(map_y, 0.0, height - 1.0)).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    dx = (map_x - x0)[..., None]
    dy = (map_y - y0)[..., None]
    sampled = (
        image[y0, x0] * (1.0 - dx) * (1.0 - dy)
        + image[y0, x1] * dx * (1.0 - dy)
        + image[y1, x0] * (1.0 - dx) * dy
        + image[y1, x1] * dx * dy
    )
    return np.where(valid[..., None], sampled, 0.0)


@dataclass(frozen=True)
class CameraCalibration:
    """Pinhole and plumb-bob calibration read from ``sensor_msgs/CameraInfo``."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, ...] = ()
    distortion_model: str = "plumb_bob"

    @classmethod
    def from_message(cls, message: Any) -> "CameraCalibration":
        matrix = np.asarray(message.k, dtype=np.float64)
        if matrix.shape != (9,):
            raise ValueError("camera-info K matrix must contain 9 values")
        calibration = cls(
            width=int(message.width),
            height=int(message.height),
            fx=float(matrix[0]),
            fy=float(matrix[4]),
            cx=float(matrix[2]),
            cy=float(matrix[5]),
            distortion=tuple(float(value) for value in message.d),
            distortion_model=str(message.distortion_model or "plumb_bob").lower(),
        )
        calibration.validate()
        return calibration

    @property
    def vertical_fov_deg(self) -> float:
        return degrees(2.0 * atan(self.height / (2.0 * self.fy)))

    def validate(self) -> None:
        values = np.asarray(
            (self.width, self.height, self.fx, self.fy, self.cx, self.cy),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)) or self.width <= 0 or self.height <= 0:
            raise ValueError("camera calibration contains invalid dimensions")
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("camera focal lengths must be positive")
        if self.distortion_model not in ("plumb_bob", "rational_polynomial"):
            raise ValueError(
                "camera distortion model must be plumb_bob or rational_polynomial"
            )


class OakPolicyCameraPreprocessor:
    """Rectify a calibrated 4:3 OAK frame to Unreal's square 55-degree view."""

    def __init__(
        self,
        calibration: CameraCalibration,
        *,
        target_size: int = 64,
        target_vertical_fov_deg: float = 55.0,
        aspect_tolerance: float = 0.02,
    ) -> None:
        calibration.validate()
        if target_size <= 0 or not 0.0 < target_vertical_fov_deg < 180.0:
            raise ValueError("invalid policy-camera target contract")
        aspect = calibration.width / calibration.height
        if abs(aspect - 4.0 / 3.0) > aspect_tolerance:
            raise ValueError(
                f"OAK source must be 4:3, got {calibration.width}x{calibration.height}"
            )
        self.calibration = calibration
        self.target_size = int(target_size)
        self.target_vertical_fov_deg = float(target_vertical_fov_deg)
        self._map_x, self._map_y = self._undistortion_map()
        self._crop = self._square_crop()

    def _undistortion_map(self) -> tuple[np.ndarray, np.ndarray]:
        calibration = self.calibration
        rows, columns = np.indices(
            (calibration.height, calibration.width), dtype=np.float64
        )
        x = (columns - calibration.cx) / calibration.fx
        y = (rows - calibration.cy) / calibration.fy
        coefficients = list(calibration.distortion) + [0.0] * 8
        k1, k2, p1, p2, k3, k4, k5, k6 = coefficients[:8]
        radius2 = x * x + y * y
        numerator = 1.0 + k1 * radius2 + k2 * radius2**2 + k3 * radius2**3
        denominator = 1.0 + k4 * radius2 + k5 * radius2**2 + k6 * radius2**3
        radial = numerator / np.maximum(denominator, 1.0e-12)
        distorted_x = (
            x * radial + 2.0 * p1 * x * y + p2 * (radius2 + 2.0 * x * x)
        )
        distorted_y = (
            y * radial + p1 * (radius2 + 2.0 * y * y) + 2.0 * p2 * x * y
        )
        return (
            calibration.fx * distorted_x + calibration.cx,
            calibration.fy * distorted_y + calibration.cy,
        )

    def _square_crop(self) -> tuple[slice, slice]:
        calibration = self.calibration
        side = int(
            round(
                2.0
                * calibration.fy
                * tan(radians(self.target_vertical_fov_deg) / 2.0)
            )
        )
        side = min(side, calibration.height)
        if side < self.target_size or side > calibration.width:
            raise ValueError(
                "calibrated camera cannot provide the requested square 55-degree view"
            )
        # Camera principal points are expressed in pixel-centre coordinates.
        left = int(round(calibration.cx - (side - 1.0) / 2.0))
        top = int(round(calibration.cy - (side - 1.0) / 2.0))
        left = min(max(left, 0), calibration.width - side)
        top = min(max(top, 0), calibration.height - side)
        return slice(top, top + side), slice(left, left + side)

    def convert(self, image: np.ndarray) -> np.ndarray:
        if image.shape[:2] != (
            self.calibration.height,
            self.calibration.width,
        ):
            raise ValueError(
                "image dimensions do not match the calibrated OAK camera"
            )
        rectified = _bilinear_remap(image, self._map_x, self._map_y)
        square = rectified[self._crop]
        return _resize_area(square, self.target_size, self.target_size)


def _ros_image_array(message: Any) -> tuple[np.ndarray, str]:
    height = int(message.height)
    width = int(message.width)
    encoding = str(message.encoding).lower()
    channels = 4 if encoding in ("rgba8", "bgra8") else 3
    if encoding in ("mono8", "8uc1"):
        channels = 1
    if encoding not in ("rgb8", "bgr8", "rgba8", "bgra8", "mono8", "8uc1"):
        raise ValueError(f"unsupported image encoding {message.encoding!r}")
    step = int(message.step)
    minimum_step = width * channels
    if height <= 0 or width <= 0 or step < minimum_step:
        raise ValueError("invalid ROS image dimensions or row stride")
    raw = np.frombuffer(message.data, dtype=np.uint8)
    if raw.size < height * step:
        raise ValueError("ROS image data is shorter than height * step")
    rows = raw[: height * step].reshape(height, step)
    return rows[:, :minimum_step].reshape(height, width, channels), encoding


def image_to_policy_pixels(
    image: Any,
    *,
    encoding: str = "rgb8",
    preprocessor: OakPolicyCameraPreprocessor | None = None,
) -> np.ndarray:
    """Convert an ndarray or sensor_msgs/Image-like object to [1,64,64,3].

    NumPy inputs may be HWC, CHW, NHWC, or NCHW. Float images in [0, 1] are
    scaled by 255, matching the URLab conversion used for the promoted model.
    """

    if isinstance(image, np.ndarray):
        array = image
        source_encoding = encoding.lower()
    elif all(hasattr(image, name) for name in ("height", "width", "encoding", "step", "data")):
        array, source_encoding = _ros_image_array(image)
    else:
        raise TypeError("image must be a NumPy array or sensor_msgs/Image-like object")

    array = np.asarray(array)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError(f"expected one image, got shape {array.shape}")
        array = array[0]
    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3 or array.shape[-1] not in (1, 3, 4):
        raise ValueError(f"expected HWC/CHW image with 1, 3, or 4 channels, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("image contains non-finite values")

    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif source_encoding in ("bgr8", "bgra8"):
        array = array[..., [2, 1, 0]]
    else:
        array = array[..., :3]
    array = array.astype(np.float32, copy=False)
    if array.size and float(np.min(array)) >= 0.0 and float(np.max(array)) <= 1.0:
        array = array * np.float32(255.0)
    array = np.clip(array, 0.0, 255.0)
    array = (
        preprocessor.convert(array)
        if preprocessor is not None
        else _resize_nearest(array)
    )
    return np.ascontiguousarray(array[None, ...], dtype=np.float32)


def _vector(values: Sequence[float], size: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


@dataclass
class ObservationBuilder:
    """Build observations while retaining the source environment's prior state."""

    previous_q: np.ndarray | None = None
    previous_dq: np.ndarray | None = None

    def reset(self, q: Sequence[float], dq: Sequence[float] | None = None) -> None:
        self.previous_q = _vector(q, 7, "q").copy()
        self.previous_dq = (
            np.zeros(7, dtype=np.float32)
            if dq is None
            else _vector(dq, 7, "dq").copy()
        )

    def build_state(
        self,
        q: Sequence[float],
        dq: Sequence[float],
        previous_command: Sequence[float],
        tcp_position_m: Sequence[float],
        phase: int | Phase,
        phase_step: int,
        gripper_openness: float,
        mode_probs: Sequence[float] | None = None,
    ) -> np.ndarray:
        q_array = _vector(q, 7, "q")
        dq_array = _vector(dq, 7, "dq")
        command = _vector(previous_command, 7, "previous_command")
        tcp = _vector(tcp_position_m, 3, "tcp_position_m")
        if self.previous_q is None or self.previous_dq is None:
            self.reset(q_array, dq_array)
        phase_value = int(phase)
        if phase_value < int(Phase.HOVER_RED) or phase_value > int(Phase.PLACED):
            raise ValueError(f"phase must be in [0, 7], got {phase_value}")
        if phase_step < 0 or not np.isfinite(gripper_openness):
            raise ValueError("phase_step and gripper_openness must be finite and nonnegative")

        joint_obs = np.stack(
            (
                q_array / np.pi,
                dq_array / 2.0,
                self.previous_q / np.pi,
                self.previous_dq / 2.0,
            ),
            axis=1,
        ).reshape(-1)
        transport_adapter = (
            float(np.clip((phase_step + 1) / 30.0, 0.0, 1.0))
            if phase_value == int(Phase.HOVER_GREEN)
            else 0.0
        )
        phase_flags = np.asarray(
            (
                phase_value == int(Phase.HOVER_RED),
                phase_value == int(Phase.DESCEND_GRASP),
                phase_value == int(Phase.CLOSE_GRIPPER),
                phase_value == int(Phase.LIFT),
                transport_adapter,
                phase_value == int(Phase.DESCEND_PLACE),
                phase_value == int(Phase.RELEASE),
            ),
            dtype=np.float32,
        )
        active_target_stage = float(phase_value >= int(Phase.HOVER_GREEN))
        if mode_probs is not None:
            probabilities = _vector(mode_probs, 8, "mode_probs")
            if np.any(probabilities < 0.0) or float(np.sum(probabilities)) <= 0.0:
                raise ValueError("mode_probs must be nonnegative with positive mass")
            probabilities = probabilities / np.sum(probabilities)
            active_target_stage = float(np.sum(probabilities[4:]))
            phase_flags = probabilities[:7]
        state = np.concatenate(
            (
                joint_obs,
                np.clip(command / np.pi, -3.0, 3.0),
                np.clip(tcp / 0.75, -3.0, 3.0),
                np.asarray(
                    (
                        active_target_stage,
                        float(np.clip(gripper_openness, 0.0, 1.0)),
                    ),
                    dtype=np.float32,
                ),
                phase_flags,
            )
        ).astype(np.float32)
        if state.shape != (47,):
            raise AssertionError(f"internal state layout error: {state.shape}")
        self.previous_q = q_array.copy()
        self.previous_dq = dq_array.copy()
        return state[None, :]
