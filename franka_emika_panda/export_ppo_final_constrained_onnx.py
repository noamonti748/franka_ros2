#!/usr/bin/env python3
"""Export the ppo_final_constrained SB3 PPO checkpoint to ONNX for OnnxPolicyRunner.

This is the "reach and hold" checkpoint: 7 joints (joint1..joint7, each a
3-vector) + tcp_pos (3-vector) -> 7 absolute joint position targets
(actuator1..actuator7). Unlike ppo_track_franka.onnx, this checkpoint expects
its observations pre-normalized by the caller (see panda_track_reach_ros2.py
joint_observation_mode="normalized_position_velocity_acceleration" and
tcp_observation_mode="tcp_error_normalized" with scale=0.75). No
normalization is baked into this ONNX graph, matching the existing
ppo_track_franka.onnx deployment convention.

Run with the MuJoCo-SB3 venv (has stable_baselines3/torch/onnx installed):

    source /home/amd/Robotics/MuJoCo-SB3/.venv/bin/activate
    python export_ppo_final_constrained_onnx.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PANDA_ARM_JOINTS: tuple[str, ...] = tuple(f"joint{i}" for i in range(1, 8))

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PACKAGE_DIR / "ppo_final_constrained.zip"
DEFAULT_OUTPUT = PACKAGE_DIR / "ppo_final_constrained.onnx"


class OnnxPolicyWrapper(nn.Module):
    """Torch module matching OnnxPolicyRunner named I/O (see panda_track_reach_ros2.py)."""

    def __init__(self, policy) -> None:
        super().__init__()
        self.policy = policy
        self._joint_names = list(PANDA_ARM_JOINTS)
        action_space = policy.action_space
        self.register_buffer(
            "action_low",
            torch.as_tensor(action_space.low, dtype=torch.float32).reshape(1, -1),
        )
        self.register_buffer(
            "action_high",
            torch.as_tensor(action_space.high, dtype=torch.float32).reshape(1, -1),
        )

    def forward(
        self,
        joint1: torch.Tensor,
        joint2: torch.Tensor,
        joint3: torch.Tensor,
        joint4: torch.Tensor,
        joint5: torch.Tensor,
        joint6: torch.Tensor,
        joint7: torch.Tensor,
        tcp_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        obs_tensors = [joint1, joint2, joint3, joint4, joint5, joint6, joint7, tcp_pos]
        obs_dict = {
            name: obs_tensors[i] for i, name in enumerate([*self._joint_names, "tcp_pos"])
        }
        features = self.policy.extract_features(obs_dict)
        latent_pi = self.policy.mlp_extractor.forward_actor(features)
        actions = self.policy.action_net(latent_pi)
        actions = torch.clamp(actions, self.action_low, self.action_high)
        return tuple(actions[:, i : i + 1] for i in range(7))


def _sample_obs_batch(rng: np.random.Generator, count: int) -> list[dict[str, np.ndarray]]:
    """A handful of random-but-in-range obs dicts, each dim within [-3, 3]."""
    samples = []
    for _ in range(count):
        obs = {name: rng.uniform(-3.0, 3.0, size=(3,)).astype(np.float32) for name in PANDA_ARM_JOINTS}
        obs["tcp_pos"] = rng.uniform(-3.0, 3.0, size=(3,)).astype(np.float32)
        samples.append(obs)
    return samples


def _validate_onnx(onnx_path: Path, model, samples: list[dict[str, np.ndarray]]) -> float:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    max_diff = 0.0
    for sample_obs in samples:
        feed = {name: sample_obs[name].reshape(1, 3).astype(np.float32) for name in PANDA_ARM_JOINTS}
        feed["tcp_pos"] = sample_obs["tcp_pos"].reshape(1, 3).astype(np.float32)

        onnx_out = session.run([f"actuator{i}" for i in range(1, 8)], feed)
        onnx_actions = np.array([float(np.asarray(o).reshape(-1)[0]) for o in onnx_out])

        sb3_actions = model.predict(sample_obs, deterministic=True)[0].reshape(-1)

        diff = float(np.max(np.abs(onnx_actions - sb3_actions)))
        max_diff = max(max_diff, diff)

    print(f"ONNX vs SB3 max abs diff over {len(samples)} samples: {max_diff:.2e}")
    if max_diff > 1e-4:
        raise RuntimeError(f"ONNX export validation failed: max diff {max_diff}")
    return max_diff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-validation-samples",
        type=int,
        default=8,
        help="Number of random in-range obs dicts to compare ONNX vs SB3 against.",
    )
    args = parser.parse_args()

    from stable_baselines3 import PPO

    model = PPO.load(str(args.checkpoint))
    policy = model.policy
    policy.eval()

    wrapper = OnnxPolicyWrapper(policy)
    wrapper.eval()

    dummy = torch.zeros(1, 3, dtype=torch.float32)
    input_names = [*PANDA_ARM_JOINTS, "tcp_pos"]
    output_names = [f"actuator{i}" for i in range(1, 8)]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        (dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy),
        str(args.output),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={name: {0: "batch"} for name in input_names},
        opset_version=args.opset,
        dynamo=False,
    )
    print(f"Exported ONNX to {args.output}")

    rng = np.random.default_rng(args.seed)
    samples = _sample_obs_batch(rng, args.num_validation_samples)
    _validate_onnx(args.output, model, samples)


if __name__ == "__main__":
    main()
