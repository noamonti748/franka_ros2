#!/usr/bin/env python3
"""Summarize Panda policy telemetry CSV smoothness metrics.

The ROS policy node writes separate columns for raw actions, interpreted target
actions, governed/rate-limited commands, optional notch output, final command,
and measured joint velocity. This script reports compact metrics for comparing
RF3 deployment runs:

  - qvel_l2 RMS / p95 / max
  - per-stream command delta L2 RMS / p95 / max
  - per-stream command second-difference L2 RMS / p95 / max

By default it reports both the whole log and the near-goal subset
(tcp_distance_m <= 0.12 m), because hold-point vibration is usually the failure
mode that matters most.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable, Optional

JOINTS: tuple[str, ...] = tuple(f"joint{i}" for i in range(1, 8))
ACTION_STREAMS: tuple[str, ...] = (
    "raw_action",
    "target_action",
    "smoothed_action",
    "governed_action",
    "post_notch_command",
    "final_command",
)


def _float_or_none(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _vector(row: dict[str, str], prefix: str) -> Optional[list[float]]:
    values: list[float] = []
    for joint in JOINTS:
        value = _float_or_none(row.get(f"{prefix}_{joint}_rad"))
        if value is None:
            return None
        values.append(value)
    return values


def _qvel_vector(row: dict[str, str]) -> Optional[list[float]]:
    values: list[float] = []
    for joint in JOINTS:
        value = _float_or_none(row.get(f"qvel_{joint}_rad_s"))
        if value is None:
            return None
        values.append(value)
    return values


def _l2(values: Iterable[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _diff_l2(current: list[float], previous: list[float]) -> float:
    return _l2(c - p for c, p in zip(current, previous))


def _second_diff_l2(
    current: list[float],
    previous: list[float],
    previous_previous: list[float],
) -> float:
    return _l2(
        c - 2.0 * p + pp
        for c, p, pp in zip(current, previous, previous_previous)
    )


def _rms(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return math.sqrt(sum(value * value for value in values) / len(values))


def _percentile(values: list[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[int(rank)]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _summary(values: list[float]) -> dict[str, Optional[float] | int]:
    return {
        "count": len(values),
        "rms": _rms(values),
        "p95": _percentile(values, 95.0),
        "max": max(values) if values else None,
    }


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _add_derived_metrics(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    enriched: list[dict[str, object]] = []
    previous_vectors: dict[str, list[float]] = {}
    previous_previous_vectors: dict[str, list[float]] = {}

    for row in rows:
        item: dict[str, object] = {"row": row}
        qvel = _qvel_vector(row)
        item["qvel_l2"] = _l2(qvel) if qvel is not None else None

        for stream in ACTION_STREAMS:
            vector = _vector(row, stream)
            if vector is None:
                item[f"{stream}_delta_l2"] = None
                item[f"{stream}_second_diff_l2"] = None
                continue

            previous = previous_vectors.get(stream)
            previous_previous = previous_previous_vectors.get(stream)
            item[f"{stream}_delta_l2"] = (
                _diff_l2(vector, previous) if previous is not None else None
            )
            item[f"{stream}_second_diff_l2"] = (
                _second_diff_l2(vector, previous, previous_previous)
                if previous is not None and previous_previous is not None
                else None
            )
            if previous is not None:
                previous_previous_vectors[stream] = previous
            previous_vectors[stream] = vector

        enriched.append(item)

    return enriched


def _metric_values(rows: list[dict[str, object]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (float, int)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def _subset(
    rows: list[dict[str, object]],
    *,
    near_goal_m: Optional[float],
    start_time_s: Optional[float],
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for row in rows:
        raw = row["row"]
        assert isinstance(raw, dict)
        if start_time_s is not None:
            ros_time_s = _float_or_none(raw.get("ros_time_s"))
            if ros_time_s is None or ros_time_s < start_time_s:
                continue
        if near_goal_m is not None:
            tcp_distance = _float_or_none(raw.get("tcp_distance_m"))
            if tcp_distance is None or tcp_distance > near_goal_m:
                continue
        selected.append(row)
    return selected


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    metrics: dict[str, object] = {
        "rows": len(rows),
        "qvel_l2": _summary(_metric_values(rows, "qvel_l2")),
    }
    for stream in ACTION_STREAMS:
        metrics[f"{stream}_delta_l2"] = _summary(
            _metric_values(rows, f"{stream}_delta_l2")
        )
        metrics[f"{stream}_second_diff_l2"] = _summary(
            _metric_values(rows, f"{stream}_second_diff_l2")
        )
    return metrics


def _format_float(value: object) -> str:
    if isinstance(value, (float, int)) and math.isfinite(float(value)):
        return f"{float(value):.6g}"
    return "n/a"


def _print_table(title: str, metrics: dict[str, object]) -> None:
    print(f"\n{title}")
    print(f"rows: {metrics['rows']}")
    print("metric                              count        rms        p95        max")
    print("-" * 78)
    for key in (
        "qvel_l2",
        "raw_action_delta_l2",
        "target_action_delta_l2",
        "smoothed_action_delta_l2",
        "governed_action_delta_l2",
        "post_notch_command_delta_l2",
        "final_command_delta_l2",
        "raw_action_second_diff_l2",
        "target_action_second_diff_l2",
        "governed_action_second_diff_l2",
        "final_command_second_diff_l2",
    ):
        summary = metrics.get(key)
        if not isinstance(summary, dict):
            continue
        print(
            f"{key:<34} "
            f"{summary.get('count', 0):>5} "
            f"{_format_float(summary.get('rms')):>10} "
            f"{_format_float(summary.get('p95')):>10} "
            f"{_format_float(summary.get('max')):>10}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument(
        "--near-goal-m",
        type=float,
        default=0.12,
        help="Near-goal TCP-distance threshold. Use <=0 to skip near-goal summary.",
    )
    parser.add_argument(
        "--start-time-s",
        type=float,
        default=None,
        help="Ignore rows before this ROS time in seconds.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of text tables.",
    )
    args = parser.parse_args()

    rows = _add_derived_metrics(_load_rows(args.csv_path))
    near_goal_m = args.near_goal_m if args.near_goal_m > 0.0 else None
    result = {
        "path": str(args.csv_path),
        "all": _summarize(_subset(rows, near_goal_m=None, start_time_s=args.start_time_s)),
    }
    if near_goal_m is not None:
        result[f"near_goal_{near_goal_m:g}m"] = _summarize(
            _subset(rows, near_goal_m=near_goal_m, start_time_s=args.start_time_s)
        )

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    print(f"path: {args.csv_path}")
    _print_table("all rows", result["all"])
    near_key = f"near_goal_{near_goal_m:g}m" if near_goal_m is not None else None
    if near_key is not None:
        _print_table(f"near goal <= {near_goal_m:g} m", result[near_key])


if __name__ == "__main__":
    main()
