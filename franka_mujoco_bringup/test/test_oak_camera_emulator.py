"""ROS-free tests for OAK camera transport behavior."""

import importlib.util
from pathlib import Path
import sys


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "oak_camera_emulator.py"
)
SPEC = importlib.util.spec_from_file_location("oak_camera_emulator", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
LatestFrameTransport = MODULE.LatestFrameTransport


def test_fixed_delay_preserves_capture_stamp_and_releases_once():
    transport = LatestFrameTransport(delay_s=0.033, seed=1)
    transport.enqueue(sequence=7, capture_stamp_s=1.0, payload="frame")
    assert transport.release_latest(1.032) is None
    frame = transport.release_latest(1.034)
    assert frame.sequence == 7
    assert frame.capture_stamp_s == 1.0
    assert frame.payload == "frame"
    assert transport.release_latest(2.0) is None
    assert transport.diagnostics()["frames_out"] == 1


def test_release_uses_latest_due_frame_without_backlog_replay():
    transport = LatestFrameTransport(delay_s=0.0, seed=2)
    for sequence in range(3):
        transport.enqueue(
            sequence=sequence,
            capture_stamp_s=1.0 + sequence * 0.001,
            payload=sequence,
        )
    frame = transport.release_latest(2.0)
    assert frame.sequence == 2
    assert transport.diagnostics()["replacement_drops"] == 2
    assert transport.release_latest(2.0) is None


def test_in_flight_queue_is_bounded_and_replaces_oldest():
    transport = LatestFrameTransport(delay_s=1.0, max_in_flight=2, seed=3)
    for sequence in range(5):
        transport.enqueue(
            sequence=sequence,
            capture_stamp_s=float(sequence),
            payload=sequence,
        )
    diagnostics = transport.diagnostics()
    assert diagnostics["in_flight"] == 2
    assert diagnostics["replacement_drops"] == 3


def test_stochastic_drops_are_deterministic():
    first = LatestFrameTransport(drop_probability=0.25, seed=9)
    second = LatestFrameTransport(drop_probability=0.25, seed=9)
    accepted_first = [
        first.enqueue(sequence=i, capture_stamp_s=float(i), payload=i)
        for i in range(100)
    ]
    accepted_second = [
        second.enqueue(sequence=i, capture_stamp_s=float(i), payload=i)
        for i in range(100)
    ]
    assert accepted_first == accepted_second
    assert 10 <= first.diagnostics()["stochastic_drops"] <= 40
