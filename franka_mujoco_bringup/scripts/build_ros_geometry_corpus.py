#!/usr/bin/env python3
"""Combine labeled ROS2 policy traces into a geometry-training corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


REQUIRED = (
    "student_pixels",
    "student_state",
    "student_geometry",
    "teacher_geometry",
    "phase_index",
    "sim_box_position_m",
    "sim_goal_position_m",
    "tcp_position_m",
    "sim_box_image_skew_s",
    "seed",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-seed", type=int, action="append")
    parser.add_argument("--max-box-skew-s", type=float, default=0.02)
    args = parser.parse_args()
    validation_seeds = set(args.validation_seed or (4, 8, 12, 16))
    paths = sorted(
        {
            path.resolve()
            for root in args.trace_root
            for path in root.rglob("seed_*.npz")
        }
    )
    if not paths:
        raise ValueError("no seed_*.npz traces found")

    episodes: list[dict[str, np.ndarray]] = []
    sources = []
    for episode_index, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as archive:
            missing = [key for key in REQUIRED if key not in archive]
            if missing:
                raise ValueError(f"{path} is missing {missing}")
            item = {key: np.asarray(archive[key]) for key in REQUIRED}
            for optional in (
                "phase_step",
                "phase_step_index",
                "frame_index",
                "frame_sequence",
                "image_stamp_s",
                "missed_frame_count",
                "reused_frame",
                "image_age_s",
                "grasp_retries",
                "placement_retries",
                "grasp_latched",
                "gripper_last_result_stalled",
                "gripper_backend_latched",
            ):
                if optional in archive:
                    item[optional] = np.asarray(archive[optional])
        count = len(item["student_state"])
        if "phase_step" in item:
            item["phase_step_index"] = item["phase_step"].astype(np.int32)
        item.setdefault("frame_index", np.arange(count, dtype=np.int32))
        if any(len(value) != count for value in item.values()):
            raise ValueError(f"{path} has inconsistent leading dimensions")
        valid = (
            np.all(np.isfinite(item["student_state"]), axis=-1)
            & np.all(np.isfinite(item["student_geometry"]), axis=-1)
            & np.all(np.isfinite(item["teacher_geometry"]), axis=-1)
            & (item["sim_box_image_skew_s"] <= args.max_box_skew_s)
        )
        item = {key: value[valid] for key, value in item.items()}
        count = int(valid.sum())
        if not count:
            raise ValueError(f"{path} has no valid synchronized samples")
        item["episode_index"] = np.full(count, episode_index, np.int32)
        seed = int(item["seed"][0])
        item["episode_split"] = np.full(
            count, int(seed in validation_seeds), np.int8
        )
        item["episode_weight"] = np.full(count, 1.0 / count, np.float32)
        episodes.append(item)
        sources.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "seed": seed,
                "samples": count,
            }
        )

    common = set.intersection(*(set(item) for item in episodes))
    arrays = {
        key: np.concatenate([item[key] for item in episodes], axis=0)
        for key in sorted(common)
    }
    pixels = arrays["student_pixels"]
    if pixels.dtype != np.uint8 and pixels.dtype != np.float32:
        raise ValueError(f"unexpected pixel dtype {pixels.dtype}")
    if pixels.shape[1:] != (64, 64, 3):
        raise ValueError(f"unexpected pixel shape {pixels.shape}")
    if float(pixels.min()) < 0.0 or float(pixels.max()) > 255.0:
        raise ValueError("pixels are outside [0,255]")
    teacher = arrays["teacher_geometry"].astype(np.float32)
    tcp = arrays["tcp_position_m"].astype(np.float32)
    box = arrays["sim_box_position_m"].astype(np.float32)
    goal = arrays["sim_goal_position_m"].astype(np.float32)
    np.testing.assert_allclose(tcp + teacher[:, :3] * 0.75, box, atol=1e-5)
    np.testing.assert_allclose(box + teacher[:, 3:] * 0.75, goal, atol=1e-5)
    arrays["student_pixels"] = pixels.astype(np.float32)
    arrays["student_state"] = arrays["student_state"].astype(np.float32)
    arrays["student_geometry"] = arrays["student_geometry"].astype(np.float32)
    arrays["teacher_geometry"] = teacher
    arrays["label_cube_to_tcp_m"] = teacher[:, :3] * 0.75
    arrays["label_cube_to_goal_m"] = teacher[:, 3:] * 0.75
    phase_counts = {
        str(int(phase)): int(np.sum(arrays["phase_index"] == phase))
        for phase in np.unique(arrays["phase_index"])
    }
    if int(phase_counts.get("1", 0)) < 100:
        raise ValueError("corpus has fewer than 100 DescendGrasp samples")
    metadata = {
        "schema": "warp.ros2-geometry-corpus",
        "version": 2,
        "geometry_scale_m": 0.75,
        "episodes": len(episodes),
        "samples": len(arrays["student_state"]),
        "phase_counts": phase_counts,
        "validation_seeds": sorted(validation_seeds),
        "max_box_skew_s": args.max_box_skew_s,
        "sources": sources,
    }
    arrays["metadata_json"] = np.asarray(
        json.dumps(metadata, sort_keys=True), dtype=np.str_
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
