"""Promoted student-v6 contract and controller configuration."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import IntEnum
import json
from pathlib import Path
from typing import Any


EXPECTED_ONNX_SHA256 = (
    "526824bcf868dff9738c09a62cd190a7b950db649d228e21a30ff55c2bcdb95f"
)


class Phase(IntEnum):
    HOVER_RED = 0
    DESCEND_GRASP = 1
    CLOSE_GRIPPER = 2
    LIFT = 3
    HOVER_GREEN = 4
    DESCEND_PLACE = 5
    RELEASE = 6
    PLACED = 7


PHASE_NAMES = (
    "hover_red",
    "descend_grasp",
    "close_gripper",
    "lift",
    "hover_green",
    "descend_place",
    "release",
    "placed",
)

PANDA_LOWER = (-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973)
PANDA_UPPER = (2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973)
PANDA_RATED_VELOCITY = (2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61)
PANDA_HOME = (
    0.0,
    0.3,
    0.0,
    -1.5707963267948966,
    0.0,
    2.0,
    -0.7853981633974483,
)


@dataclass(frozen=True)
class StudentV6Config:
    """Runtime values promoted with ``student_v6_final``."""

    control_hz: float = 60.0
    response_time: float = 0.5
    integrated_lead_steps: float = 4.0
    action_smoothing_alpha: float = 0.1
    velocity_limit_fraction: float = 0.38
    geometry_scale_m: float = 0.75

    hover_offset_m: float = 0.075
    grasp_tcp_offset_m: float = 0.01
    lift_tcp_overshoot_m: float = 0.01
    required_lift_height_m: float = 0.04
    place_hover_offset_m: float = 0.085
    placement_tolerance_m: float = 0.02
    placement_height_tolerance_m: float = 0.02

    phase_success_thresholds: tuple[float, ...] = (
        0.05, 0.015, 0.05, 0.01, 0.05, 0.02, 0.05, 0.1
    )
    phase_max_joint_velocity: tuple[float, ...] = (
        0.5, 0.5, 0.5, 0.5, 0.4, 0.3, 0.5, 0.4
    )
    phase_settled_dwell: tuple[int, ...] = (6, 4, 0, 0, 10, 8, 0, 0)
    phase_step_limits: tuple[int, ...] = (800, 800, 120, 350, 800, 900, 90, 120)
    observable_position_gate_margins: tuple[float, ...] = (
        0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    )
    observable_phase_extra_dwell: tuple[int, ...] = (0, 0, 0, 0, 0, 0, 0, 0)

    observable_object_width_m: float = 0.05
    observable_grasp_width_tolerance_m: float = 0.005
    observable_grasp_xy_tolerance_m: float = 0.02
    observable_grasp_z_tolerance_m: float = 0.005
    observable_hover_xy_tolerance_m: float = 0.015
    observable_grasp_dwell_steps: int = 5
    observable_latch_timeout_steps: int = 900
    observable_placement_max_speed_m_s: float = 0.02
    observable_settle_window_steps: int = 12
    observable_descend_recovery_steps: int = 300
    release_open_fraction: float = 0.9
    success_dwell_steps: int = 10
    max_grasp_retries: int = 4
    max_placement_retries: int = 2
    scripted_gripper_close_steps: int = 20
    scripted_gripper_open_steps: int = 8
    grasp_retry_retreat_height_m: float = 0.075
    grasp_retry_position_tolerance_m: float = 0.005
    grasp_retry_max_joint_speed_rad_s: float = 0.10
    grasp_retry_settled_steps: int = 3
    grasp_retry_fresh_estimate_steps: int = 1
    seated_offset_m: float = 0.0229
    support_cube_center_height_m: float = 0.025

    dls_warmup_steps_per_phase: int = 30
    dls_damping: float = 0.05
    dls_cartesian_gain: float = 1.4
    dls_hover_speed_m_s: float = 0.18
    dls_descend_speed_m_s: float = 0.06
    dls_joint_velocity_limit_fraction: float = 0.72

    joint_lower: tuple[float, ...] = PANDA_LOWER
    joint_upper: tuple[float, ...] = PANDA_UPPER
    rated_joint_velocity: tuple[float, ...] = PANDA_RATED_VELOCITY
    home_joint_position: tuple[float, ...] = PANDA_HOME

    def __post_init__(self) -> None:
        for name in (
            "phase_success_thresholds",
            "phase_max_joint_velocity",
            "phase_settled_dwell",
            "phase_step_limits",
            "observable_position_gate_margins",
            "observable_phase_extra_dwell",
        ):
            if len(getattr(self, name)) != 8:
                raise ValueError(f"{name} must contain eight values")
        for name in (
            "joint_lower",
            "joint_upper",
            "rated_joint_velocity",
            "home_joint_position",
        ):
            if len(getattr(self, name)) != 7:
                raise ValueError(f"{name} must contain seven values")
        window = self.observable_settle_window_steps
        if window < 2 or window % 2:
            raise ValueError("observable_settle_window_steps must be even and >= 2")
        if not 0.0 < self.release_open_fraction <= 1.0:
            raise ValueError("release_open_fraction must be in (0, 1]")
        if (
            self.scripted_gripper_close_steps <= 0
            or self.scripted_gripper_open_steps <= 0
        ):
            raise ValueError("scripted gripper step counts must be positive")
        if (
            self.grasp_retry_retreat_height_m <= 0.0
            or self.grasp_retry_position_tolerance_m < 0.0
            or self.grasp_retry_max_joint_speed_rad_s < 0.0
            or self.grasp_retry_settled_steps <= 0
            or self.grasp_retry_fresh_estimate_steps <= 0
        ):
            raise ValueError("grasp retry recovery settings are invalid")


def default_model_dir() -> Path:
    """Return the installed share model directory, or the source-tree fallback."""

    try:
        from ament_index_python.packages import get_package_share_directory

        return (
            Path(get_package_share_directory("franka_emika_panda"))
            / "models"
            / "student_v6"
        )
    except (ImportError, LookupError, OSError):
        pass
    return Path(__file__).resolve().parents[2] / "models" / "student_v6"


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def resolve_artifact_paths(
    *,
    onnx_model_path: str | Path | None = None,
    model_path: str | Path | None = None,
    contract_path: str | Path | None = None,
    controller_config_path: str | Path | None = None,
    model_dir: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    """Resolve canonical paths and launch aliases without silent precedence."""

    canonical = str(onnx_model_path or "").strip()
    alias = str(model_path or "").strip()
    if canonical and alias and Path(canonical).expanduser() != Path(alias).expanduser():
        raise ValueError("onnx_model_path and model_path specify different files")
    directory = Path(model_dir) if model_dir is not None else default_model_dir()
    model = Path(canonical or alias) if (canonical or alias) else (
        directory / "student_v6_final.onnx"
    )
    contract = (
        Path(contract_path)
        if str(contract_path or "").strip()
        else model.parent / "student_v6_final_observable_contract.json"
    )
    controller = (
        Path(controller_config_path)
        if str(controller_config_path or "").strip()
        else model.parent / "student_v6_final_controller.json"
    )
    return tuple(path.expanduser().resolve() for path in (model, contract, controller))


def load_runtime_config(
    model_dir: str | Path | None = None,
    *,
    contract_path: str | Path | None = None,
    controller_config_path: str | Path | None = None,
    expected_onnx_sha256: str = EXPECTED_ONNX_SHA256,
) -> StudentV6Config:
    """Load and validate the copied promoted contract/controller artifacts."""

    directory = Path(model_dir) if model_dir is not None else default_model_dir()
    contract_file = (
        Path(contract_path)
        if contract_path is not None
        else directory / "student_v6_final_observable_contract.json"
    )
    controller_file = (
        Path(controller_config_path)
        if controller_config_path is not None
        else directory / "student_v6_final_controller.json"
    )
    contract = load_json(contract_file)
    controller = load_json(controller_file)
    contract_sha = contract.get("onnx", {}).get("sha256")
    controller_sha = controller.get("onnx_sha256")
    if contract_sha != expected_onnx_sha256 or controller_sha != expected_onnx_sha256:
        raise ValueError("artifact SHA references do not match the selected model")
    if contract.get("inputs", {}).get("state", {}).get("shape") != ["batch", 47]:
        raise ValueError("student-v6 state contract must be [batch, 47]")
    if contract.get("inputs", {}).get("pixels", {}).get("shape") != [
        "batch", 64, 64, 3
    ]:
        raise ValueError("student-v6 pixel contract must be NHWC [batch, 64, 64, 3]")

    observable = contract["estimator_and_gates"]
    approach = controller["approach"]
    overrides: dict[str, Any] = {
        "control_hz": float(contract["controller_hz"]),
        "response_time": float(observable["response_time"]),
        "geometry_scale_m": float(observable["geometry_scale_m"]),
        "phase_success_thresholds": tuple(observable["phase_success_thresholds"]),
        "phase_max_joint_velocity": tuple(observable["phase_max_joint_velocity"]),
        "phase_settled_dwell": tuple(observable["phase_settled_dwell"]),
        "phase_step_limits": tuple(observable["phase_step_limits"]),
        "observable_position_gate_margins": tuple(
            observable["observable_position_gate_margins"]
        ),
        "observable_phase_extra_dwell": tuple(
            observable["observable_phase_extra_dwell"]
        ),
        "observable_object_width_m": float(observable["observable_object_width"]),
        "observable_grasp_width_tolerance_m": float(
            observable["observable_grasp_width_tolerance"]
        ),
        "observable_grasp_dwell_steps": int(
            observable["observable_grasp_dwell_steps"]
        ),
        "observable_latch_timeout_steps": int(
            observable["observable_latch_timeout_steps"]
        ),
        "observable_placement_max_speed_m_s": float(
            observable["observable_placement_max_speed"]
        ),
        "observable_settle_window_steps": int(
            observable["observable_settle_window_steps"]
        ),
        "observable_descend_recovery_steps": int(
            observable["observable_descend_recovery_steps"]
        ),
        "success_dwell_steps": int(observable["success_dwell_steps"]),
        "dls_warmup_steps_per_phase": int(approach["policy_warmup_steps_per_phase"]),
        "dls_damping": float(approach["damping"]),
        "dls_cartesian_gain": float(approach["cartesian_gain"]),
        "dls_hover_speed_m_s": float(approach["hover_max_speed_m_s"]),
        "dls_descend_speed_m_s": float(approach["descend_max_speed_m_s"]),
        "dls_joint_velocity_limit_fraction": float(
            approach["joint_velocity_limit_fraction"]
        ),
        "hover_offset_m": float(approach["hover_offset_m"]),
        "grasp_tcp_offset_m": float(approach["grasp_tcp_offset_m"]),
    }
    allowed = {item.name for item in fields(StudentV6Config)}
    return StudentV6Config(**{key: value for key, value in overrides.items() if key in allowed})
