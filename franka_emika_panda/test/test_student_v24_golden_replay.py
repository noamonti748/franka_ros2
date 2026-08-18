"""ROS-free replay of a complete direct-JAX student-v24 episode."""

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from franka_emika_panda_policy.config import StudentV6Config
from franka_emika_panda_policy.dls import DLSController
from franka_emika_panda_policy.integration import IntegratedActionController
from franka_emika_panda_policy.kinematics import PandaAnalyticJacobian
from franka_emika_panda_policy.observation import ObservationBuilder


TRACE_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "student_v24_direct_seed1_controller_trace.npz"
)


def _trace():
    return np.load(TRACE_PATH, allow_pickle=False)


def _config_and_metadata(trace):
    metadata = json.loads(str(trace["metadata_json"]))
    config = replace(
        StudentV6Config(),
        response_time=float(metadata["response_time_s"]),
    )
    return config, metadata


def test_direct_observation_state_replays_to_float32_precision():
    with _trace() as trace:
        builder = ObservationBuilder()
        builder.reset(
            trace["pre_measured_q"][0],
            trace["pre_measured_qvel"][0],
        )
        replayed = []
        for index in range(len(trace["phase"])):
            openness = float(
                np.clip(
                    np.mean(trace["pre_gripper_joint_positions"][index])
                    / 0.04,
                    0.0,
                    1.0,
                )
            )
            replayed.append(
                builder.build_state(
                    trace["pre_measured_q"][index],
                    trace["pre_measured_qvel"][index],
                    trace["pre_smoothed_arm_target"][index],
                    trace["pre_measured_tcp_position"][index],
                    int(trace["phase"][index]),
                    int(trace["phase_step"][index]),
                    openness,
                )[0]
            )
        np.testing.assert_allclose(
            np.asarray(replayed),
            trace["state"],
            rtol=0.0,
            atol=6.0e-8,
        )


def test_direct_latency_integration_and_smoothing_replay():
    with _trace() as trace:
        config, metadata = _config_and_metadata(trace)
        controller = IntegratedActionController(
            config,
            action_latency_steps=int(metadata["action_latency_steps"]),
        )
        controller.reset(trace["pre_measured_q"][0])
        integrated = []
        smoothed = []
        for index in range(len(trace["phase"])):
            smoothed.append(
                controller.step(
                    trace["selected_action"][index],
                    trace["pre_measured_q"][index],
                )
            )
            integrated.append(controller.integrated_target.copy())
        np.testing.assert_allclose(
            integrated,
            trace["integrated_arm_target"],
            rtol=0.0,
            atol=2.2e-6,
        )
        np.testing.assert_allclose(
            smoothed,
            trace["smoothed_arm_target"],
            rtol=0.0,
            atol=2.2e-6,
        )


def test_direct_dls_actions_replay_with_analytic_panda_jacobian():
    with _trace() as trace:
        config, _ = _config_and_metadata(trace)
        controller = DLSController(config, PandaAnalyticJacobian())
        replayed = []
        expected = []
        for index in np.flatnonzero(trace["selected_controller"]):
            replayed.append(
                controller.compute_action(
                    phase=int(trace["phase"][index]),
                    phase_step=int(trace["phase_step"][index]),
                    q=trace["pre_measured_q"][index],
                    qvel=trace["pre_measured_qvel"][index],
                    geometry=trace["policy_geometry"][index],
                )
            )
            expected.append(trace["dls_action"][index])
        np.testing.assert_allclose(
            replayed,
            expected,
            rtol=0.0,
            atol=2.7e-4,
        )
