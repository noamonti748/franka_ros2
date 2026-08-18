from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from franka_emika_panda_policy.config import Phase
from franka_emika_panda_policy.observation import (
    CameraCalibration,
    OakPolicyCameraPreprocessor,
    ObservationBuilder,
    image_to_policy_pixels,
)


def test_47d_state_layout_matches_training_order():
    previous_q = np.arange(1.0, 8.0)
    previous_dq = np.arange(11.0, 18.0)
    q = np.arange(21.0, 28.0)
    dq = np.arange(31.0, 38.0)
    command = np.arange(41.0, 48.0)
    builder = ObservationBuilder()
    builder.reset(previous_q, previous_dq)

    state = builder.build_state(
        q,
        dq,
        command,
        (0.75, -0.75, 1.5),
        Phase.HOVER_GREEN,
        14,
        0.25,
    )[0]

    assert state.shape == (47,)
    np.testing.assert_allclose(
        state[:28].reshape(7, 4),
        np.stack((q / np.pi, dq / 2.0, previous_q / np.pi, previous_dq / 2.0), axis=1),
    )
    np.testing.assert_allclose(state[28:35], np.clip(command / np.pi, -3.0, 3.0))
    np.testing.assert_allclose(state[35:38], (1.0, -1.0, 2.0))
    np.testing.assert_allclose(state[38:40], (1.0, 0.25))
    np.testing.assert_allclose(state[40:47], (0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0))


def test_47d_autonomous_state_uses_previous_mode_feedback():
    builder = ObservationBuilder()
    builder.reset(np.zeros(7), np.zeros(7))
    mode = np.zeros(8, dtype=np.float32)
    mode[int(Phase.DESCEND_PLACE)] = 1.0
    state = builder.build_state(
        np.zeros(7),
        np.zeros(7),
        np.zeros(7),
        np.zeros(3),
        Phase.HOVER_RED,
        0,
        0.5,
        mode_probs=mode,
    )[0]
    assert state[38] == 1.0
    assert state[39] == 0.5
    np.testing.assert_array_equal(state[40:47], mode[:7])


def test_nchw_unit_image_converts_to_float32_nhwc_255():
    image = np.zeros((1, 3, 2, 2), dtype=np.float32)
    image[:, 0] = 1.0
    pixels = image_to_policy_pixels(image)
    assert pixels.shape == (1, 64, 64, 3)
    assert pixels.dtype == np.float32
    np.testing.assert_allclose(pixels[0, 0, 0], (255.0, 0.0, 0.0))


def test_bgr_ros_image_with_padding_converts_to_rgb():
    class Message:
        height = 1
        width = 1
        encoding = "bgr8"
        step = 4
        data = bytes((1, 2, 3, 99))

    pixels = image_to_policy_pixels(Message())
    np.testing.assert_allclose(pixels[0, 0, 0], (3.0, 2.0, 1.0))


def test_oak_preprocessor_crops_4_by_3_view_before_downsampling():
    focal = 6.0 / (2.0 * np.tan(np.deg2rad(55.0) / 2.0))
    calibration = CameraCalibration(
        width=8,
        height=6,
        fx=focal,
        fy=focal,
        cx=3.5,
        cy=2.5,
    )
    preprocessor = OakPolicyCameraPreprocessor(
        calibration,
        target_size=3,
        target_vertical_fov_deg=55.0,
    )
    image = np.zeros((6, 8, 3), dtype=np.float32)
    image[:, 0, 2] = 255.0
    image[:, -1, 2] = 255.0
    image[:, 1:7, 0] = 120.0

    pixels = image_to_policy_pixels(image, preprocessor=preprocessor)

    assert pixels.shape == (1, 3, 3, 3)
    np.testing.assert_allclose(pixels[0, ..., 0], 120.0, atol=1.0e-5)
    np.testing.assert_allclose(pixels[0, ..., 2], 0.0, atol=1.0e-5)


def test_camera_calibration_reads_ros_camera_info_contract():
    class CameraInfo:
        width = 640
        height = 480
        k = (
            500.0,
            0.0,
            319.5,
            0.0,
            500.0,
            239.5,
            0.0,
            0.0,
            1.0,
        )
        d = (0.1, -0.2, 0.0, 0.0, 0.03)
        distortion_model = "plumb_bob"

    calibration = CameraCalibration.from_message(CameraInfo())

    assert calibration.width == 640
    assert calibration.height == 480
    assert calibration.distortion == CameraInfo.d
    assert calibration.vertical_fov_deg > 0.0
