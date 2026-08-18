#!/usr/bin/env python3
"""Compare direct-JAX and ROS2 student policy traces without running ROS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def _error(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    delta = np.asarray(actual, dtype=np.float64) - np.asarray(
        reference, dtype=np.float64
    )
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "rmse": _rmse(delta),
    }


def _infer(
    session: ort.InferenceSession,
    pixels: np.ndarray,
    state: np.ndarray,
    *,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    actions: list[np.ndarray] = []
    geometry: list[np.ndarray] = []
    for start in range(0, len(pixels), batch_size):
        action, geometric = session.run(
            None,
            {
                "pixels": pixels[start : start + batch_size].astype(np.float32),
                "state": state[start : start + batch_size].astype(np.float32),
            },
        )
        actions.append(action)
        geometry.append(geometric)
    return np.concatenate(actions), np.concatenate(geometry)


def _transitions(pre: np.ndarray, post: np.ndarray) -> list[dict[str, int]]:
    return [
        {"step": index, "from": int(left), "to": int(right)}
        for index, (left, right) in enumerate(zip(pre, post))
        if int(left) != int(right)
    ]


def compare(direct_path: Path, ros_path: Path, model_path: Path) -> dict:
    with np.load(direct_path, allow_pickle=False) as direct_file:
        direct = {name: direct_file[name] for name in direct_file.files}
    with np.load(ros_path, allow_pickle=False) as ros_file:
        ros = {name: ros_file[name] for name in ros_file.files}

    session = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    direct_action, direct_geometry = _infer(
        session, direct["pixels"], direct["state"]
    )
    ros_action, ros_geometry = _infer(
        session, ros["pixels"], ros["policy_state"]
    )

    common = min(len(direct["pixels"]), len(ros["pixels"]), 32)
    report = {
        "schema": "student-v24-direct-ros-trace-comparison-v1",
        "direct_trace": str(direct_path),
        "ros_trace": str(ros_path),
        "model": str(model_path),
        "steps": {
            "direct": int(len(direct["pixels"])),
            "ros": int(len(ros["pixels"])),
            "common_prefix_compared": common,
        },
        "onnx_replay": {
            "direct_action": _error(
                direct["policy_action"], direct_action
            ),
            "direct_geometry": _error(
                direct["policy_geometry"], direct_geometry
            ),
            "ros_action": _error(ros["raw_action"], ros_action),
            "ros_geometry": _error(ros["geometry"], ros_geometry),
        },
        "initial_snapshot": {
            "joint_position": _error(
                direct["pre_measured_q"][0], ros["q"][0]
            ),
            "tcp_position_m": _error(
                direct["pre_measured_tcp_position"][0],
                ros["tcp_position_m"][0],
            ),
            "policy_state": _error(
                direct["state"][0], ros["policy_state"][0]
            ),
            "pixels": _error(
                direct["pixels"][0], ros["pixels"][0]
            ),
            "policy_action": _error(
                direct["policy_action"][0], ros["raw_action"][0]
            ),
            "policy_geometry": _error(
                direct["policy_geometry"][0], ros["geometry"][0]
            ),
        },
        "common_step_prefix": {
            "pixels": _error(
                direct["pixels"][:common], ros["pixels"][:common]
            ),
            "policy_state": _error(
                direct["state"][:common], ros["policy_state"][:common]
            ),
            "policy_action": _error(
                direct["policy_action"][:common],
                ros["raw_action"][:common],
            ),
            "policy_geometry": _error(
                direct["policy_geometry"][:common],
                ros["geometry"][:common],
            ),
        },
        "phase_transitions": {
            "direct": _transitions(
                direct["phase"], direct["post_phase"]
            ),
            "ros": _transitions(ros["active_phase"], ros["phase"]),
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--ros", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.direct, args.ros, args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
