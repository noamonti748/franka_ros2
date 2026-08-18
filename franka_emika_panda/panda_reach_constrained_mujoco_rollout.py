#!/usr/bin/env python3
"""Run the ppo_final_constrained "reach and hold" ONNX policy against native MuJoCo.

This is a deployment sanity check for panda_reach_constrained_ros2.py: it applies the
same tcp_pos/joint observation preprocessing as that ROS supervisor directly to the
MuJoCo position actuators from mjx_panda_reach.xml. Unlike real ROS joint_states
(no acceleration field), MuJoCo gives clean ground-truth data.qacc, making this the
best available validation path for the normalized_position_velocity_acceleration
joint observation mode before touching ROS/Gazebo.

This script is intentionally standalone (no rclpy/ROS dependency, mirroring
panda_mujoco_onnx_rollout.py's convention), so it can run in a plain MuJoCo + SB3/
onnxruntime venv.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

PANDA_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
PANDA_ACTUATORS = tuple(f"actuator{i}" for i in range(1, 8))

# Normalization constants for joint_observation_mode="normalized_position_velocity_acceleration"
# (ppo_final_constrained "reach and hold" checkpoint): obs = [qpos/pi, qvel/2.0, qacc/10.0].
JOINT_OBS_POS_SCALE = math.pi
JOINT_OBS_VEL_SCALE = 2.0
JOINT_OBS_ACC_SCALE = 10.0


def _package_file(name: str) -> str:
    try:
        from ament_index_python.packages import get_package_share_directory

        return str(Path(get_package_share_directory("franka_emika_panda")) / name)
    except Exception:
        return str(Path(__file__).resolve().parent / name)


def _vec3(text: str) -> np.ndarray:
    parts = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
    if len(parts) == 1 and " " in parts[0]:
        parts = [part for part in parts[0].split() if part]
    if len(parts) != 3:
        raise ValueError(f"expected 3 values, got {text!r}")
    return np.array([float(part) for part in parts], dtype=np.float32)


def _tcp_observation(
    mode: str,
    tcp_m: np.ndarray,
    target_m: np.ndarray,
    scale: float,
    signs: np.ndarray,
    unit_scale: float,
    clip: Optional[float] = None,
) -> np.ndarray:
    if scale <= 0.0:
        raise ValueError(f"scale must be positive, got {scale}")
    if mode in ("tcp_error_normalized", "tcp_minus_target_normalized"):
        obs = (tcp_m - target_m) * unit_scale / scale
    elif mode in ("target_error_normalized", "target_minus_tcp_normalized"):
        obs = (target_m - tcp_m) * unit_scale / scale
    elif mode == "tcp_normalized":
        obs = tcp_m * unit_scale / scale
    elif mode == "target_normalized":
        obs = target_m * unit_scale / scale
    elif mode == "tcp_minus_target":
        obs = tcp_m - target_m
    elif mode in ("target_delta", "target_minus_tcp"):
        obs = target_m - tcp_m
    elif mode == "tcp":
        obs = tcp_m
    elif mode == "target":
        obs = target_m
    else:
        raise ValueError(f"unsupported tcp observation mode: {mode}")
    obs = obs.astype(np.float32) * signs
    if clip is not None and clip > 0.0:
        obs = np.clip(obs, -clip, clip)
    return obs.astype(np.float32)


def _joint_observation(
    mode: str,
    position: float,
    velocity: float,
    command: float,
    acceleration: float = 0.0,
    clip: Optional[float] = None,
) -> np.ndarray:
    if mode == "position_velocity_command":
        obs = np.array([position, velocity, command], dtype=np.float32)
    elif mode == "position_velocity_zero":
        obs = np.array([position, velocity, 0.0], dtype=np.float32)
    elif mode == "position_velocity_error":
        obs = np.array([position, velocity, command - position], dtype=np.float32)
    elif mode == "normalized_position_velocity_acceleration":
        obs = np.array(
            [
                position / JOINT_OBS_POS_SCALE,
                velocity / JOINT_OBS_VEL_SCALE,
                acceleration / JOINT_OBS_ACC_SCALE,
            ],
            dtype=np.float32,
        )
    else:
        raise ValueError(f"unsupported joint observation mode: {mode}")
    if clip is not None and clip > 0.0:
        obs = np.clip(obs, -clip, clip)
    return obs


def _clip_ctrl(model, actuator_ids: Sequence[int], values: Sequence[float]) -> list[float]:
    clipped = []
    for actuator_id, value in zip(actuator_ids, values):
        low, high = model.actuator_ctrlrange[actuator_id]
        clipped.append(float(np.clip(value, low, high)))
    return clipped


def _normalized_to_ctrl(model, actuator_ids: Sequence[int], values: Sequence[float]) -> list[float]:
    controls = []
    for actuator_id, value in zip(actuator_ids, values):
        low, high = model.actuator_ctrlrange[actuator_id]
        normalized = float(np.clip(value, -1.0, 1.0))
        controls.append(float(low + 0.5 * (normalized + 1.0) * (high - low)))
    return controls


def _interpret_actions(
    model,
    actuator_ids: Sequence[int],
    raw_actions: Sequence[float],
    joint_positions: Sequence[float],
    last_command: Sequence[float],
    mode: str,
    delta_scale: float,
) -> list[float]:
    if mode == "absolute":
        return _clip_ctrl(model, actuator_ids, raw_actions)
    if mode == "normalized_absolute":
        return _normalized_to_ctrl(model, actuator_ids, raw_actions)
    if mode == "delta_position":
        return _clip_ctrl(
            model,
            actuator_ids,
            (
                float(position) + float(action) * delta_scale
                for position, action in zip(joint_positions, raw_actions)
            ),
        )
    if mode == "normalized_delta_position":
        return _clip_ctrl(
            model,
            actuator_ids,
            (
                float(position) + float(np.clip(action, -1.0, 1.0)) * delta_scale
                for position, action in zip(joint_positions, raw_actions)
            ),
        )
    if mode == "delta_command":
        return _clip_ctrl(
            model,
            actuator_ids,
            (
                float(command) + float(action) * delta_scale
                for command, action in zip(last_command, raw_actions)
            ),
        )
    if mode == "normalized_delta_command":
        return _clip_ctrl(
            model,
            actuator_ids,
            (
                float(command) + float(np.clip(action, -1.0, 1.0)) * delta_scale
                for command, action in zip(last_command, raw_actions)
            ),
        )
    raise ValueError(f"unsupported action output mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", default=_package_file("mjx_panda_reach.xml"))
    parser.add_argument("--model", default=_package_file("ppo_final_constrained.onnx"))
    parser.add_argument("--target", default="0.32,0,0.5")
    parser.add_argument("--tcp-observation-mode", default="tcp_error_normalized")
    parser.add_argument("--tcp-observation-scale", type=float, default=0.75)
    parser.add_argument("--tcp-observation-signs", default="1,1,1")
    parser.add_argument("--tcp-observation-unit-scale", type=float, default=1.0)
    parser.add_argument(
        "--joint-observation-mode", default="normalized_position_velocity_acceleration"
    )
    parser.add_argument(
        "--observation-clip",
        type=float,
        default=3.0,
        help="If > 0, clip every joint/tcp observation dim to +/-this value (matches the trained Box(-3,3) space).",
    )
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--action-output-mode", default="absolute")
    parser.add_argument("--action-delta-scale", type=float, default=0.05)
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=1.0,
        help="1.0 = no smoothing (matches training; this checkpoint was trained with no action filter).",
    )
    parser.add_argument("--control-hz", type=float, default=60.0)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--print-every", type=int, default=30)
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=0.075,
        help="Distance (m) below which the TCP is considered 'holding' the target.",
    )
    args = parser.parse_args()

    try:
        import mujoco
    except ImportError as exc:
        raise SystemExit(
            "Python package 'mujoco' is required. Install it in the active venv with: "
            "python -m pip install mujoco"
        ) from exc

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit(
            "Python package 'onnxruntime' is required. Install it in the active venv with: "
            "python -m pip install onnxruntime"
        ) from exc

    xml_path = Path(args.xml).expanduser()
    model_path = Path(args.model).expanduser()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        data.qpos[:] = model.key_qpos[key_id]
        data.ctrl[:] = model.key_ctrl[key_id]
        mujoco.mj_forward(model, data)

    joint_qpos_addr = []
    joint_qvel_addr = []
    for joint_name in PANDA_JOINTS:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise RuntimeError(f"Joint not found in MuJoCo model: {joint_name}")
        joint_qpos_addr.append(int(model.jnt_qposadr[joint_id]))
        joint_qvel_addr.append(int(model.jnt_dofadr[joint_id]))

    actuator_ids = []
    for actuator_name in PANDA_ACTUATORS:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if actuator_id < 0:
            raise RuntimeError(f"Actuator not found in MuJoCo model: {actuator_name}")
        actuator_ids.append(int(actuator_id))

    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "tcp_pos")
    if sensor_id < 0:
        raise RuntimeError("Sensor not found in MuJoCo model: tcp_pos")
    sensor_adr = int(model.sensor_adr[sensor_id])
    sensor_dim = int(model.sensor_dim[sensor_id])
    if sensor_dim != 3:
        raise RuntimeError(f"tcp_pos sensor has unexpected dim {sensor_dim}")

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    target = _vec3(args.target)
    signs = _vec3(args.tcp_observation_signs)
    observation_clip = args.observation_clip if args.observation_clip > 0.0 else None
    sim_steps_per_control = max(1, int(round((1.0 / args.control_hz) / model.opt.timestep)))
    last_command = [float(data.ctrl[actuator_id]) for actuator_id in actuator_ids]

    print(f"xml={xml_path}")
    print(f"onnx={model_path}")
    print(f"target={target.tolist()}")
    print(
        "tcp_observation="
        f"{args.tcp_observation_mode}, scale={args.tcp_observation_scale}, "
        f"unit_scale={args.tcp_observation_unit_scale}, signs={signs.tolist()}"
    )
    print(
        f"action_output_mode={args.action_output_mode}, "
        f"action_scale={args.action_scale}, action_delta_scale={args.action_delta_scale}"
    )
    print(f"mujoco_timestep={model.opt.timestep:.6f}, substeps/control={sim_steps_per_control}")
    print(
        f"joint_observation_mode={args.joint_observation_mode}, observation_clip={args.observation_clip}"
    )

    success_threshold_m = args.success_threshold
    initial_distance = None
    best_distance = float("inf")
    sustained_hold_steps = 0
    max_sustained_hold_steps = 0
    for step_idx in range(args.steps + 1):
        tcp = np.array(data.sensordata[sensor_adr : sensor_adr + 3], dtype=np.float32)
        distance = float(np.linalg.norm(tcp - target))
        if initial_distance is None:
            initial_distance = distance
        best_distance = min(best_distance, distance)
        if distance <= success_threshold_m:
            sustained_hold_steps += 1
            max_sustained_hold_steps = max(max_sustained_hold_steps, sustained_hold_steps)
        else:
            sustained_hold_steps = 0

        if step_idx % max(args.print_every, 1) == 0 or step_idx == args.steps:
            print(
                f"step={step_idx:04d} dist={distance:.4f} best={best_distance:.4f} "
                f"hold_steps={sustained_hold_steps} tcp={[round(float(v), 4) for v in tcp]}"
            )

        joint_positions = []
        feed = {}
        for i, joint_name in enumerate(PANDA_JOINTS):
            pos = float(data.qpos[joint_qpos_addr[i]])
            vel = float(data.qvel[joint_qvel_addr[i]])
            # True ground-truth MuJoCo acceleration (indexed like qvel, by dof) --
            # the whole point of this script as the qacc validation path, since
            # real joint_states has no acceleration field.
            acc = float(data.qacc[joint_qvel_addr[i]])
            cmd = float(last_command[i])
            joint_positions.append(pos)
            feed[joint_name] = _joint_observation(
                args.joint_observation_mode,
                pos,
                vel,
                cmd,
                acceleration=acc,
                clip=observation_clip,
            ).reshape(1, 3)

        feed["tcp_pos"] = _tcp_observation(
            args.tcp_observation_mode,
            tcp,
            target,
            args.tcp_observation_scale,
            signs,
            args.tcp_observation_unit_scale,
            clip=observation_clip,
        ).reshape(1, 3)

        outputs = session.run([f"actuator{i}" for i in range(1, 8)], feed)
        raw = [float(np.asarray(output).reshape(-1)[0]) * args.action_scale for output in outputs]
        command = _interpret_actions(
            model,
            actuator_ids,
            raw,
            joint_positions,
            last_command,
            args.action_output_mode,
            args.action_delta_scale,
        )
        alpha = float(np.clip(args.smooth_alpha, 0.0, 1.0))
        command = [
            previous + alpha * (current - previous)
            for previous, current in zip(last_command, command)
        ]
        for actuator_id, value in zip(actuator_ids, command):
            data.ctrl[actuator_id] = value
        last_command = command

        for _ in range(sim_steps_per_control):
            mujoco.mj_step(model, data)

    delta = float(best_distance - (initial_distance or 0.0))
    print(f"initial_dist={initial_distance:.4f} best_dist={best_distance:.4f} best_minus_initial={delta:.4f}")
    print(
        f"success_threshold={success_threshold_m:.4f} "
        f"max_sustained_hold_steps={max_sustained_hold_steps} "
        f"final_sustained_hold_steps={sustained_hold_steps}"
    )


if __name__ == "__main__":
    main()
