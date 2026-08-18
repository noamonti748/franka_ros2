#!/usr/bin/env python3
"""Export the ppo_final_lower_vib SB3 PPO checkpoint to ONNX.

The lower-vibration reach/hold checkpoint uses the 38-dim dictionary observation
layout documented in PandaReachPolicyDeploymentNotes.md:

  - joint1..joint7: [qpos/pi, qvel/2.0, prev_qpos/pi, prev_qvel/2.0]
  - prev_action: previous applied 7-joint command divided by pi
  - tcp_pos: (tcp_m - target_m) / 0.75

The ONNX graph expects those values already normalized and clipped by the caller,
matching the package's other Panda deployment exports.

Run with the MuJoCo-SB3 venv:

    /home/amd/Robotics/MuJoCo-SB3/.venv/bin/python export_ppo_final_lower_vib_onnx.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PANDA_ARM_JOINTS: tuple[str, ...] = tuple(f"joint{i}" for i in range(1, 8))

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PACKAGE_DIR / "ppo_final_lower_vib.zip"
DEFAULT_OUTPUT = PACKAGE_DIR / "ppo_final_lower_vib.onnx"


class LowerVibOnnxPolicyWrapper(nn.Module):
    """Torch module with stable, named ONNX I/O for the lower-vib policy."""

    def __init__(self, policy) -> None:
        super().__init__()
        self.policy = policy
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
        prev_action: torch.Tensor,
        tcp_pos: torch.Tensor,
    ) -> torch.Tensor:
        obs_tensors = [joint1, joint2, joint3, joint4, joint5, joint6, joint7]
        obs_dict = {
            name: obs_tensors[i] for i, name in enumerate(PANDA_ARM_JOINTS)
        }
        obs_dict["prev_action"] = prev_action
        obs_dict["tcp_pos"] = tcp_pos

        features = self.policy.extract_features(obs_dict)
        latent_pi = self.policy.mlp_extractor.forward_actor(features)
        actions = self.policy.action_net(latent_pi)
        return torch.clamp(actions, self.action_low, self.action_high)


def _sample_obs_batch(rng: np.random.Generator, count: int) -> list[dict[str, np.ndarray]]:
    samples = []
    for _ in range(count):
        obs = {
            name: rng.uniform(-3.0, 3.0, size=(4,)).astype(np.float32)
            for name in PANDA_ARM_JOINTS
        }
        obs["prev_action"] = rng.uniform(-1.0, 1.0, size=(7,)).astype(np.float32)
        obs["tcp_pos"] = rng.uniform(-3.0, 3.0, size=(3,)).astype(np.float32)
        samples.append(obs)
    return samples


def _validate_onnx(onnx_path: Path, model, samples: list[dict[str, np.ndarray]]) -> float:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    max_diff = 0.0
    for sample_obs in samples:
        feed = {
            name: sample_obs[name].reshape(1, 4).astype(np.float32)
            for name in PANDA_ARM_JOINTS
        }
        feed["prev_action"] = sample_obs["prev_action"].reshape(1, 7).astype(np.float32)
        feed["tcp_pos"] = sample_obs["tcp_pos"].reshape(1, 3).astype(np.float32)

        onnx_action = np.asarray(session.run(["action"], feed)[0], dtype=np.float32).reshape(-1)
        sb3_action = model.predict(sample_obs, deterministic=True)[0].reshape(-1)

        diff = float(np.max(np.abs(onnx_action - sb3_action)))
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

    model = PPO.load(str(args.checkpoint), device="cpu")
    policy = model.policy
    policy.eval()

    wrapper = LowerVibOnnxPolicyWrapper(policy)
    wrapper.eval()

    dummy_joint = torch.zeros(1, 4, dtype=torch.float32)
    dummy_prev_action = torch.zeros(1, 7, dtype=torch.float32)
    dummy_tcp = torch.zeros(1, 3, dtype=torch.float32)
    input_names = [*PANDA_ARM_JOINTS, "prev_action", "tcp_pos"]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        (
            dummy_joint,
            dummy_joint,
            dummy_joint,
            dummy_joint,
            dummy_joint,
            dummy_joint,
            dummy_joint,
            dummy_prev_action,
            dummy_tcp,
        ),
        str(args.output),
        input_names=input_names,
        output_names=["action"],
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
