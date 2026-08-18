"""Promoted NumPy damped-least-squares approach controller."""

from __future__ import annotations

from collections.abc import Callable
from typing import Sequence

import numpy as np

from .config import Phase, StudentV6Config
from .integration import normalize_target


JacobianSource = np.ndarray | Callable[[np.ndarray], np.ndarray]
CUBE_ANCHOR_CENTER_M = (0.5, 0.15, 0.025)
CUBE_ANCHOR_Z_HALF_RANGE_M = 0.015
DLS_DESCEND_TARGETS = ("live", "sim_box", "hover_estimate")


def normalize_dls_descend_target(value: str) -> str:
    """Return a canonical DescendGrasp aiming mode."""

    target = str(value).strip().lower()
    if target not in DLS_DESCEND_TARGETS:
        raise ValueError(
            "dls_descend_target must be one of "
            f"{DLS_DESCEND_TARGETS}, got {value!r}"
        )
    return target


def lock_wrist_to_home(
    action: Sequence[float],
    config: StudentV6Config,
) -> np.ndarray:
    """Replace joint-7 with the home wrist so fingers stay cube-face aligned."""

    locked = np.asarray(action, dtype=np.float64).copy()
    if locked.shape != (7,):
        raise ValueError("action must have shape (7,)")
    if not np.all(np.isfinite(locked)):
        raise ValueError("action contains non-finite values")
    home = normalize_target(
        config.home_joint_position,
        config.joint_lower,
        config.joint_upper,
    )
    locked[6] = home[6]
    return locked.astype(np.float32)


def rewrite_cube_geometry_to_world_anchor(
    geometry: Sequence[float],
    tcp_position_m: Sequence[float],
    cube_anchor_m: Sequence[float],
    geometry_scale_m: float,
) -> np.ndarray:
    """Replace live cube residuals with the vector from TCP to a frozen world cube."""

    if not np.isfinite(geometry_scale_m) or geometry_scale_m <= 0.0:
        raise ValueError("geometry_scale_m must be finite and positive")
    rewritten = np.asarray(geometry, dtype=np.float64).copy()
    tcp = np.asarray(tcp_position_m, dtype=np.float64)
    anchor = np.asarray(cube_anchor_m, dtype=np.float64)
    if rewritten.shape != (6,) or tcp.shape != (3,) or anchor.shape != (3,):
        raise ValueError("geometry, tcp, and cube_anchor shapes must be (6,), (3,), (3,)")
    if not (np.all(np.isfinite(rewritten)) and np.all(np.isfinite(tcp)) and np.all(np.isfinite(anchor))):
        raise ValueError("geometry rewrite inputs contain non-finite values")
    rewritten[:3] = (anchor - tcp) / geometry_scale_m
    return rewritten


def update_hover_cube_anchor(
    current_anchor_m: np.ndarray | None,
    *,
    valid: bool,
    observed_cube_m: Sequence[float],
    alpha: float,
    clamp_radius_m: float,
    clamp_center_m: Sequence[float] = CUBE_ANCHOR_CENTER_M,
    clamp_z_m: float = CUBE_ANCHOR_Z_HALF_RANGE_M,
) -> np.ndarray:
    """EMA-update a world cube freeze from hover observations, clamped near spawn."""

    if not 0.0 < alpha <= 1.0:
        raise ValueError("dls_anchor_alpha must be in (0, 1]")
    if not np.isfinite(clamp_radius_m) or clamp_radius_m <= 0.0:
        raise ValueError("dls_anchor_clamp_radius_m must be positive")
    observed = np.asarray(observed_cube_m, dtype=np.float64)
    if observed.shape != (3,) or not np.all(np.isfinite(observed)):
        raise ValueError("observed cube must be a finite (3,) vector")
    if valid:
        current = np.asarray(current_anchor_m, dtype=np.float64)
        if current.shape != (3,) or not np.all(np.isfinite(current)):
            raise ValueError("current cube anchor must be a finite (3,) vector")
        proposed = (1.0 - alpha) * current + alpha * observed
    else:
        proposed = observed
    center = np.asarray(clamp_center_m, dtype=np.float64)
    bounds = np.asarray((clamp_radius_m, clamp_radius_m, clamp_z_m), dtype=np.float64)
    return np.clip(proposed, center - bounds, center + bounds)


class DLSController:
    """Translational DLS matching ``evaluate_observable_jax.py`` parameters."""

    def __init__(
        self,
        config: StudentV6Config,
        jacobian: JacobianSource,
    ) -> None:
        self.config = config
        self._jacobian = jacobian

    def should_engage(self, phase: int | Phase, phase_step: int) -> bool:
        phase_value = int(phase)
        if phase_value == int(Phase.HOVER_RED):
            return phase_step >= self.config.dls_warmup_steps_per_phase
        if phase_value == int(Phase.DESCEND_GRASP):
            return phase_step >= 0
        return False

    def _get_jacobian(self, q: np.ndarray) -> np.ndarray:
        value = self._jacobian(q) if callable(self._jacobian) else self._jacobian
        jacobian = np.asarray(value, dtype=np.float64)
        if jacobian.shape != (6, 7):
            raise ValueError(f"Jacobian must have shape (6, 7), got {jacobian.shape}")
        if not np.all(np.isfinite(jacobian)):
            raise ValueError("Jacobian contains non-finite values")
        return jacobian

    def compute_action(
        self,
        *,
        phase: int | Phase,
        phase_step: int,
        q: Sequence[float],
        qvel: Sequence[float],
        geometry: Sequence[float],
    ) -> np.ndarray:
        """Return normalized action; caller must use it only when engaged."""

        if not self.should_engage(phase, phase_step):
            raise ValueError(
                "DLS is restricted to HoverRed after the 30-step warmup "
                "and to DescendGrasp"
            )
        q_array = np.asarray(q, dtype=np.float64)
        qvel_array = np.asarray(qvel, dtype=np.float64)
        geometry_array = np.asarray(geometry, dtype=np.float64)
        if q_array.shape != (7,) or qvel_array.shape != (7,):
            raise ValueError("q and qvel must have shape (7,)")
        if geometry_array.shape != (6,):
            raise ValueError("geometry must have shape (6,)")
        if not (
            np.all(np.isfinite(q_array))
            and np.all(np.isfinite(qvel_array))
            and np.all(np.isfinite(geometry_array))
        ):
            raise ValueError("DLS inputs contain non-finite values")

        z_offset = (
            self.config.hover_offset_m
            if int(phase) == int(Phase.HOVER_RED)
            else self.config.grasp_tcp_offset_m
        )
        displacement = geometry_array[:3] * self.config.geometry_scale_m
        displacement = displacement + np.asarray((0.0, 0.0, z_offset))
        maximum_speed = (
            self.config.dls_hover_speed_m_s
            if int(phase) == int(Phase.HOVER_RED)
            else self.config.dls_descend_speed_m_s
        )
        nullspace_gain = 0.06 if int(phase) == int(Phase.HOVER_RED) else 0.0
        return self.compute_displacement_action(
            q=q_array,
            qvel=qvel_array,
            displacement_m=displacement,
            maximum_speed_m_s=maximum_speed,
            nullspace_gain=nullspace_gain,
            braking_distance_m=0.05,
        )

    def compute_displacement_action(
        self,
        *,
        q: Sequence[float],
        qvel: Sequence[float],
        displacement_m: Sequence[float],
        maximum_speed_m_s: float,
        nullspace_gain: float = 0.0,
        braking_distance_m: float = 0.05,
    ) -> np.ndarray:
        """Return a normalized DLS action for an explicit Cartesian displacement."""

        q_array = np.asarray(q, dtype=np.float64)
        qvel_array = np.asarray(qvel, dtype=np.float64)
        displacement = np.asarray(displacement_m, dtype=np.float64)
        if q_array.shape != (7,) or qvel_array.shape != (7,):
            raise ValueError("q and qvel must have shape (7,)")
        if displacement.shape != (3,):
            raise ValueError("displacement_m must have shape (3,)")
        if not (
            np.all(np.isfinite(q_array))
            and np.all(np.isfinite(qvel_array))
            and np.all(np.isfinite(displacement))
            and np.isfinite(maximum_speed_m_s)
            and maximum_speed_m_s > 0.0
            and np.isfinite(braking_distance_m)
            and braking_distance_m > 0.0
        ):
            raise ValueError("DLS displacement inputs are invalid")
        desired_velocity = self.config.dls_cartesian_gain * displacement
        speed = float(np.linalg.norm(desired_velocity))
        desired_velocity *= min(1.0, maximum_speed_m_s / max(speed, 1.0e-8))

        jacobian = self._get_jacobian(q_array)[:3, :]
        damping = self.config.dls_damping
        inverse = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping**2 * np.eye(3),
            np.eye(3),
        )
        nullspace = np.eye(7) - inverse @ jacobian
        home = np.asarray(self.config.home_joint_position)
        qdot = inverse @ desired_velocity + nullspace @ (
            nullspace_gain * (home - q_array)
        )
        braking = max(
            0.0,
            1.0 - float(np.linalg.norm(displacement)) / braking_distance_m,
        )
        qdot -= braking * 0.20 * qvel_array
        curriculum_limit = (
            np.asarray(self.config.rated_joint_velocity)
            * self.config.velocity_limit_fraction
        )
        dls_limit = self.config.dls_joint_velocity_limit_fraction * curriculum_limit
        qdot = np.clip(qdot, -dls_limit, dls_limit)
        raw_target = np.clip(
            q_array + self.config.response_time * qdot,
            self.config.joint_lower,
            self.config.joint_upper,
        )
        return normalize_target(
            raw_target, self.config.joint_lower, self.config.joint_upper
        ).astype(np.float32)
