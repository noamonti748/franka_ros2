"""Strict, lazily imported ONNX Runtime adapter."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .config import EXPECTED_ONNX_SHA256


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class OnnxPolicy:
    def __init__(
        self,
        model_path: str | Path,
        *,
        expected_sha256: str = EXPECTED_ONNX_SHA256,
        session: Any | None = None,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"student-v6 ONNX model not found: {self.model_path}")
        actual_sha = sha256_file(self.model_path)
        if actual_sha != expected_sha256:
            raise ValueError(
                f"student-v6 ONNX SHA256 mismatch: expected {expected_sha256}, got {actual_sha}"
            )
        if session is None:
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RuntimeError(
                    "onnxruntime is required only to execute the student-v6 policy"
                ) from exc
            session = ort.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"]
            )
        self._session = session
        self.autonomous = False
        self.has_advance = False
        self._validate_interface()

    def _validate_interface(self) -> None:
        inputs = {item.name: item for item in self._session.get_inputs()}
        outputs = {item.name: item for item in self._session.get_outputs()}
        if set(inputs) != {"pixels", "state"}:
            raise ValueError(f"unexpected ONNX inputs: {sorted(inputs)}")
        valid_outputs = (
            {"action", "geometry"},
            {"action", "geometry", "mode_probs"},
            {"action", "geometry", "mode_probs", "advance_prob"},
        )
        if set(outputs) not in valid_outputs:
            raise ValueError(f"unexpected ONNX outputs: {sorted(outputs)}")
        self.autonomous = "mode_probs" in outputs
        self.has_advance = "advance_prob" in outputs
        expected = {
            "pixels": (64, 64, 3),
            "state": (47,),
            "action": (7,),
            "geometry": (6,),
            "mode_probs": (8,),
        }
        for name, item in {**inputs, **outputs}.items():
            shape = tuple(item.shape)
            if name == "advance_prob":
                if len(shape) not in (1, 2):
                    raise ValueError(
                        f"advance_prob must be batched scalar, got {shape}"
                    )
                continue
            if len(shape) != 2 and name != "pixels":
                raise ValueError(f"{name} must be batched, got ONNX shape {shape}")
            trailing = tuple(shape[1:])
            if trailing != expected[name]:
                raise ValueError(
                    f"{name} ONNX shape must end in {expected[name]}, got {shape}"
                )

    def infer_all(
        self, pixels: np.ndarray, state: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, float | None]:
        pixels_array = np.asarray(pixels, dtype=np.float32)
        state_array = np.asarray(state, dtype=np.float32)
        if pixels_array.shape != (1, 64, 64, 3):
            raise ValueError(
                f"pixels must have shape (1,64,64,3), got {pixels_array.shape}"
            )
        if state_array.shape != (1, 47):
            raise ValueError(f"state must have shape (1,47), got {state_array.shape}")
        if not np.all(np.isfinite(pixels_array)) or not np.all(np.isfinite(state_array)):
            raise ValueError("policy input contains non-finite values")
        if float(np.min(pixels_array)) < 0.0 or float(np.max(pixels_array)) > 255.0:
            raise ValueError("policy pixels must be in [0, 255]")
        output_names = ["action", "geometry"]
        if self.autonomous:
            output_names.append("mode_probs")
        if self.has_advance:
            output_names.append("advance_prob")
        outputs = self._session.run(
            output_names, {"pixels": pixels_array, "state": state_array}
        )
        action, geometry = outputs[:2]
        mode = outputs[2] if self.autonomous else None
        advance = outputs[3] if self.has_advance else None
        action_array = np.asarray(action, dtype=np.float32)
        geometry_array = np.asarray(geometry, dtype=np.float32)
        if action_array.shape != (1, 7) or geometry_array.shape != (1, 6):
            raise ValueError(
                f"invalid ONNX output shapes action={action_array.shape}, "
                f"geometry={geometry_array.shape}"
            )
        if not np.all(np.isfinite(action_array)) or not np.all(np.isfinite(geometry_array)):
            raise ValueError("policy output contains non-finite values")
        mode_array = None
        if mode is not None:
            raw_mode = np.asarray(mode, dtype=np.float32)
            if raw_mode.shape != (1, 8) or not np.all(np.isfinite(raw_mode)):
                raise ValueError(f"invalid mode output shape/value {raw_mode.shape}")
            raw_mode = np.maximum(raw_mode[0], 0.0)
            total = float(np.sum(raw_mode))
            if total <= 0.0:
                raise ValueError("mode output has no probability mass")
            mode_array = raw_mode / total
        advance_value = None
        if advance is not None:
            raw_advance = np.asarray(advance, dtype=np.float32).reshape(-1)
            if raw_advance.size < 1 or not np.all(np.isfinite(raw_advance)):
                raise ValueError("invalid advance_prob output")
            advance_value = float(np.clip(raw_advance[0], 0.0, 1.0))
        return (
            np.clip(action_array[0], -1.0, 1.0),
            geometry_array[0].copy(),
            mode_array,
            advance_value,
        )

    def infer(
        self, pixels: np.ndarray, state: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Legacy two-output API retained for the v24+DLS rollback runtime."""
        action, geometry, _, _ = self.infer_all(pixels, state)
        return action, geometry
