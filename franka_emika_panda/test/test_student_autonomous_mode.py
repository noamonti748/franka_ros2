from pathlib import Path
import json
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from franka_emika_panda_policy.config import Phase, StudentV6Config
from franka_emika_panda_policy.mode import AutonomousModeFilter
from franka_emika_panda_policy.onnx_policy import OnnxPolicy, sha256_file
from franka_emika_panda_policy.session import AutonomousSession


def _one_hot(phase: Phase) -> np.ndarray:
    result = np.zeros(8, dtype=np.float32)
    result[int(phase)] = 1.0
    return result


def test_mode_filter_debounces_and_forbids_skips():
    controller = AutonomousModeFilter(
        alpha=1.0, margin=0.0, consecutive_steps=2
    )
    skipped = controller.update(_one_hot(Phase.LIFT))
    assert skipped.phase == Phase.HOVER_RED

    first = controller.update(_one_hot(Phase.DESCEND_GRASP))
    second = controller.update(_one_hot(Phase.DESCEND_GRASP))
    assert first.phase == Phase.HOVER_RED
    assert second.phase == Phase.DESCEND_GRASP


def test_autonomous_session_requires_measured_grasp_before_lift():
    session = AutonomousSession(
        StudentV6Config(),
        mode_alpha=1.0,
        mode_margin=0.0,
        mode_consecutive_steps=1,
    )
    session.mode_filter.force(Phase.CLOSE_GRIPPER)
    session.phase = Phase.CLOSE_GRIPPER
    common = {
        "geometry": np.zeros(6),
        "tcp_position_m": np.zeros(3),
        "tcp_rotation": np.eye(3),
        "joint_velocity": np.zeros(7),
        "finger_span_m": 0.08,
        "gripper_openness": 1.0,
    }

    blocked = session.step(
        mode_probs=_one_hot(Phase.LIFT),
        grasp_latched_override=False,
        **common,
    )
    assert blocked.phase == Phase.CLOSE_GRIPPER

    advanced = session.step(
        mode_probs=_one_hot(Phase.LIFT),
        grasp_latched_override=True,
        **common,
    )
    assert advanced.phase == Phase.LIFT
    assert not advanced.use_dls


def test_onnx_adapter_accepts_autonomous_three_output_contract(tmp_path):
    class ValueInfo:
        def __init__(self, name, shape):
            self.name = name
            self.shape = shape

    class Session:
        def get_inputs(self):
            return [
                ValueInfo("pixels", ["batch", 64, 64, 3]),
                ValueInfo("state", ["batch", 47]),
            ]

        def get_outputs(self):
            return [
                ValueInfo("action", ["batch", 7]),
                ValueInfo("geometry", ["batch", 6]),
                ValueInfo("mode_probs", ["batch", 8]),
            ]

        def run(self, names, inputs):
            del inputs
            assert names == ["action", "geometry", "mode_probs"]
            return (
                np.zeros((1, 7), dtype=np.float32),
                np.zeros((1, 6), dtype=np.float32),
                _one_hot(Phase.HOVER_RED)[None],
            )

    model = tmp_path / "autonomous.onnx"
    model.write_bytes(b"autonomous-test")
    policy = OnnxPolicy(
        model,
        expected_sha256=sha256_file(model),
        session=Session(),
    )
    action, geometry, mode, advance = policy.infer_all(
        np.zeros((1, 64, 64, 3), dtype=np.float32),
        np.zeros((1, 47), dtype=np.float32),
    )
    assert policy.autonomous
    assert action.shape == (7,)
    assert geometry.shape == (6,)
    assert advance is None
    np.testing.assert_array_equal(mode, _one_hot(Phase.HOVER_RED))


def test_autonomous_bundle_is_versioned_and_keeps_v24_rollback():
    bundle = (
        Path(__file__).resolve().parents[1]
        / "models"
        / "student_autonomous_v1"
    )
    manifest = json.loads(
        (bundle / "manifest.json").read_text(encoding="utf-8")
    )
    model = bundle / manifest["model"]
    assert sha256_file(model) == manifest["onnx_sha256"]
    assert manifest["rollback"] == "student_v24_urlab_candidate.onnx"
    contract = json.loads(
        (bundle / "student_v6_final_observable_contract.json").read_text(
            encoding="utf-8"
        )
    )
    assert contract["outputs"]["mode_probs"]["shape"] == ["batch", 8]
    assert not contract["controller_ownership"]["dls"]
