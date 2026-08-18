import importlib
import json
import math
from pathlib import Path
import sys
import types

import numpy as np
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "python"))

from franka_emika_panda_policy.config import (
    EXPECTED_ONNX_SHA256,
    PANDA_HOME,
    default_model_dir,
    load_runtime_config,
    resolve_artifact_paths,
)
from franka_emika_panda_policy.onnx_policy import sha256_file


MODEL_DIR = PACKAGE_ROOT / "models" / "student_v6"


def test_copied_contract_config_and_sha_are_promoted_values():
    required = (
        "student_v6_final.onnx",
        "student_v6_final_observable_contract.json",
        "student_v6_final_controller.json",
        "student_v6_final_onnx_report.json",
        "student_v6_final_unreal_parity.json",
    )
    assert all((MODEL_DIR / name).is_file() for name in required)
    assert sha256_file(MODEL_DIR / "student_v6_final.onnx") == EXPECTED_ONNX_SHA256

    config = load_runtime_config(MODEL_DIR)
    assert config.control_hz == 60.0
    assert config.geometry_scale_m == 0.75
    assert config.observable_position_gate_margins[1] == 0.01
    assert config.dls_warmup_steps_per_phase == 30
    assert config.dls_joint_velocity_limit_fraction == 0.72
    np.testing.assert_allclose(
        PANDA_HOME,
        (0.0, 0.3, 0.0, -math.pi / 2.0, 0.0, 2.0, -math.pi / 4.0),
    )
    np.testing.assert_allclose(config.home_joint_position, PANDA_HOME)


def test_contract_declares_exact_io_and_node_imports_without_ros():
    contract = json.loads(
        (MODEL_DIR / "student_v6_final_observable_contract.json").read_text(
            encoding="utf-8"
        )
    )
    assert contract["inputs"]["pixels"]["shape"] == ["batch", 64, 64, 3]
    assert contract["inputs"]["pixels"]["dtype"] == "float32"
    assert contract["inputs"]["pixels"]["range"] == [0.0, 255.0]
    assert contract["inputs"]["state"]["shape"] == ["batch", 47]
    assert contract["outputs"]["action"]["shape"] == ["batch", 7]
    assert contract["outputs"]["geometry"]["shape"] == ["batch", 6]

    module = importlib.import_module("franka_emika_panda_policy.node")
    assert callable(module.main)


def test_default_model_dir_prefers_installed_ament_share(monkeypatch, tmp_path):
    packages = types.ModuleType("ament_index_python.packages")
    packages.get_package_share_directory = lambda _: str(tmp_path / "share")
    package = types.ModuleType("ament_index_python")
    package.packages = packages
    monkeypatch.setitem(sys.modules, "ament_index_python", package)
    monkeypatch.setitem(sys.modules, "ament_index_python.packages", packages)

    assert default_model_dir() == tmp_path / "share" / "models" / "student_v6"


def test_model_path_alias_is_explicit_and_conflicts_fail(tmp_path):
    model = tmp_path / "student_v6_final.onnx"
    resolved = resolve_artifact_paths(model_path=model)
    assert resolved[0] == model.resolve()
    assert resolved[1].parent == model.parent.resolve()
    with pytest.raises(ValueError):
        resolve_artifact_paths(
            onnx_model_path=tmp_path / "one.onnx",
            model_path=tmp_path / "two.onnx",
        )
