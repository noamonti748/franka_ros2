"""Exact Panda hand-TCP forward kinematics and geometric Jacobian."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _rotation_x(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray(
        ((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine))
    )


def _rotation_y(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray(
        ((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine))
    )


def _rotation_z(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
    )


def _transform(
    xyz: Sequence[float] = (0.0, 0.0, 0.0),
    rpy: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in rpy)
    result = np.eye(4)
    result[:3, :3] = (
        _rotation_z(yaw) @ _rotation_y(pitch) @ _rotation_x(roll)
    )
    result[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return result


_JOINT_ORIGINS = (
    _transform((0.0, 0.0, 0.333)),
    _transform(rpy=(-np.pi / 2.0, 0.0, 0.0)),
    _transform((0.0, -0.316, 0.0), (np.pi / 2.0, 0.0, 0.0)),
    _transform((0.0825, 0.0, 0.0), (np.pi / 2.0, 0.0, 0.0)),
    _transform((-0.0825, 0.384, 0.0), (-np.pi / 2.0, 0.0, 0.0)),
    _transform(rpy=(np.pi / 2.0, 0.0, 0.0)),
    _transform((0.088, 0.0, 0.0), (np.pi / 2.0, 0.0, 0.0)),
)
_HAND_TCP_FIXED = (
    _transform((0.0, 0.0, 0.107))
    @ _transform(rpy=(0.0, 0.0, -np.pi / 4.0))
    @ _transform((0.0, 0.0, 0.1))
)


def _panda_hand_tcp_chain(
    joint_position: Sequence[float],
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Return the terminal transform plus joint origins and axes."""

    q = np.asarray(joint_position, dtype=np.float64)
    if q.shape != (7,) or not np.all(np.isfinite(q)):
        raise ValueError("Panda joint position must be a finite seven-vector")

    current = np.eye(4)
    origins: list[np.ndarray] = []
    axes: list[np.ndarray] = []
    local_z = np.asarray((0.0, 0.0, 1.0))
    for origin, angle in zip(_JOINT_ORIGINS, q):
        current = current @ origin
        origins.append(current[:3, 3].copy())
        axes.append(current[:3, :3] @ local_z)
        current = current @ _transform(rpy=(0.0, 0.0, float(angle)))

    current = current @ _HAND_TCP_FIXED
    return current, origins, axes


def panda_hand_tcp_pose(
    joint_position: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact base-frame TCP position and rotation."""

    current, _, _ = _panda_hand_tcp_chain(joint_position)
    return current[:3, 3].copy(), current[:3, :3].copy()


def panda_hand_tcp_kinematics(
    joint_position: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return base-frame TCP position and a 6x7 geometric Jacobian.

    The chain is the exact `panda_link0` to `panda_hand_tcp` chain shared by
    the training MJCF and the Humble MuJoCo URDF.
    """

    current, origins, axes = _panda_hand_tcp_chain(joint_position)
    tcp_position = current[:3, 3].copy()
    jacobian = np.zeros((6, 7), dtype=np.float64)
    for index, (origin, axis) in enumerate(zip(origins, axes)):
        jacobian[:3, index] = np.cross(axis, tcp_position - origin)
        jacobian[3:, index] = axis
    return tcp_position, jacobian


class PandaAnalyticJacobian:
    """Callable Jacobian provider for the canonical Panda hand TCP."""

    def __call__(self, joint_position: np.ndarray) -> np.ndarray:
        return panda_hand_tcp_kinematics(joint_position)[1]
