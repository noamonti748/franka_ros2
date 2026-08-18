#!/usr/bin/env python3
"""Timestamp-preserving OAK-1 camera transport emulator for ROS2 MuJoCo."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
import math
import random
from typing import Any


@dataclass(frozen=True)
class TransportFrame:
    sequence: int
    capture_stamp_s: float
    release_stamp_s: float
    payload: Any


class LatestFrameTransport:
    """Delay frames in flight and release only the newest frame due per tick."""

    def __init__(
        self,
        *,
        delay_s: float = 0.0,
        jitter_s: float = 0.0,
        drop_probability: float = 0.0,
        max_in_flight: int = 16,
        seed: int = 0,
    ) -> None:
        if (
            delay_s < 0.0
            or jitter_s < 0.0
            or not 0.0 <= drop_probability <= 1.0
            or max_in_flight <= 0
        ):
            raise ValueError("invalid OAK transport settings")
        self.delay_s = float(delay_s)
        self.jitter_s = float(jitter_s)
        self.drop_probability = float(drop_probability)
        self.max_in_flight = int(max_in_flight)
        self._rng = random.Random(seed)
        self._frames: deque[TransportFrame] = deque()
        self.frames_in = 0
        self.frames_out = 0
        self.stochastic_drops = 0
        self.replacement_drops = 0
        self._delays: deque[float] = deque(maxlen=4096)

    def enqueue(
        self,
        *,
        sequence: int,
        capture_stamp_s: float,
        payload: Any,
    ) -> bool:
        if not math.isfinite(capture_stamp_s):
            raise ValueError("capture_stamp_s must be finite")
        self.frames_in += 1
        if self._rng.random() < self.drop_probability:
            self.stochastic_drops += 1
            return False
        jitter = self._rng.uniform(-self.jitter_s, self.jitter_s)
        release = capture_stamp_s + max(self.delay_s + jitter, 0.0)
        self._frames.append(
            TransportFrame(sequence, capture_stamp_s, release, payload)
        )
        while len(self._frames) > self.max_in_flight:
            self._frames.popleft()
            self.replacement_drops += 1
        return True

    def release_latest(self, now_s: float) -> TransportFrame | None:
        if not math.isfinite(now_s):
            raise ValueError("now_s must be finite")
        due = []
        while self._frames and self._frames[0].release_stamp_s <= now_s:
            due.append(self._frames.popleft())
        if not due:
            return None
        if len(due) > 1:
            self.replacement_drops += len(due) - 1
        frame = due[-1]
        self.frames_out += 1
        self._delays.append(max(now_s - frame.capture_stamp_s, 0.0))
        return frame

    def diagnostics(self) -> dict[str, float | int]:
        delays = sorted(self._delays)
        p95 = (
            delays[min(int(0.95 * (len(delays) - 1)), len(delays) - 1)]
            if delays
            else 0.0
        )
        return {
            "frames_in": self.frames_in,
            "frames_out": self.frames_out,
            "stochastic_drops": self.stochastic_drops,
            "replacement_drops": self.replacement_drops,
            "in_flight": len(self._frames),
            "mean_delivery_delay_s": (
                sum(delays) / len(delays) if delays else 0.0
            ),
            "delivery_delay_p95_s": p95,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ros-args", action="store_true", help=argparse.SUPPRESS)
    parser.parse_known_args()
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import String

    class OAKCameraEmulator(Node):
        def __init__(self) -> None:
            super().__init__("oak_camera_emulator")
            self.declare_parameter(
                "source_image_topic", "/policy_camera/sim/color/image_raw"
            )
            self.declare_parameter(
                "output_image_topic", "/policy_camera/color/image_raw"
            )
            self.declare_parameter(
                "source_info_topic", "/policy_camera/sim/camera_info"
            )
            self.declare_parameter(
                "output_info_topic", "/policy_camera/camera_info"
            )
            self.declare_parameter("enabled", False)
            self.declare_parameter("base_delay_ms", 0.0)
            self.declare_parameter("jitter_ms", 0.0)
            self.declare_parameter("drop_probability", 0.0)
            self.declare_parameter("max_in_flight", 16)
            self.declare_parameter("seed", 0)
            self.declare_parameter("release_poll_hz", 240.0)
            qos = QoSProfile(
                depth=1, reliability=ReliabilityPolicy.BEST_EFFORT
            )
            self._image_pub = self.create_publisher(
                Image, str(self.get_parameter("output_image_topic").value), qos
            )
            self._info_pub = self.create_publisher(
                CameraInfo,
                str(self.get_parameter("output_info_topic").value),
                qos,
            )
            self._diag_pub = self.create_publisher(
                String, "/policy_camera/oak_emulator/telemetry", 10
            )
            self._latest_info: CameraInfo | None = None
            self._sequence = 0
            self._last_diag_s = 0.0
            self._transport = LatestFrameTransport(
                delay_s=(
                    float(self.get_parameter("base_delay_ms").value) / 1000.0
                    if bool(self.get_parameter("enabled").value)
                    else 0.0
                ),
                jitter_s=(
                    float(self.get_parameter("jitter_ms").value) / 1000.0
                    if bool(self.get_parameter("enabled").value)
                    else 0.0
                ),
                drop_probability=(
                    float(self.get_parameter("drop_probability").value)
                    if bool(self.get_parameter("enabled").value)
                    else 0.0
                ),
                max_in_flight=int(
                    self.get_parameter("max_in_flight").value
                ),
                seed=int(self.get_parameter("seed").value),
            )
            self.create_subscription(
                Image,
                str(self.get_parameter("source_image_topic").value),
                self._on_image,
                qos,
            )
            self.create_subscription(
                CameraInfo,
                str(self.get_parameter("source_info_topic").value),
                self._on_info,
                qos,
            )
            poll_hz = float(self.get_parameter("release_poll_hz").value)
            self.create_timer(1.0 / poll_hz, self._release)

        @staticmethod
        def _stamp_s(message: Image) -> float:
            return float(message.header.stamp.sec) + (
                float(message.header.stamp.nanosec) * 1.0e-9
            )

        def _on_image(self, message: Image) -> None:
            self._transport.enqueue(
                sequence=self._sequence,
                capture_stamp_s=self._stamp_s(message),
                payload=message,
            )
            self._sequence += 1

        def _on_info(self, message: CameraInfo) -> None:
            self._latest_info = message

        def _release(self) -> None:
            now_s = self.get_clock().now().nanoseconds * 1.0e-9
            frame = self._transport.release_latest(now_s)
            if frame is not None:
                self._image_pub.publish(frame.payload)
                if self._latest_info is not None:
                    info = self._latest_info
                    info.header = frame.payload.header
                    self._info_pub.publish(info)
            if now_s - self._last_diag_s >= 1.0:
                payload = self._transport.diagnostics()
                payload["enabled"] = bool(
                    self.get_parameter("enabled").value
                )
                self._diag_pub.publish(
                    String(data=json.dumps(payload, sort_keys=True))
                )
                self._last_diag_s = now_s

    rclpy.init()
    node = OAKCameraEmulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
