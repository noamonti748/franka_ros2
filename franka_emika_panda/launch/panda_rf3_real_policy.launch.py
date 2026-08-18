# Copyright (c) 2026 Franka Robotics GmbH
#
# Licensed under the Apache License, Version 2.0
#
# Runs the RF3 constrained reach-and-hold ONNX policy against an already-running
# real Panda controller stack. This launch intentionally does not start Gazebo or
# robot hardware bringup.

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    model = LaunchConfiguration("model")
    command_topic = LaunchConfiguration("command_topic")
    command_message_type = LaunchConfiguration("command_message_type")
    joint_states_topic = LaunchConfiguration("joint_states_topic")
    tcp_transform_topic = LaunchConfiguration("tcp_transform_topic")
    frame_id = LaunchConfiguration("frame_id")
    target_location_m = LaunchConfiguration("target_location_m")
    target_random_offset_max_m = LaunchConfiguration("target_random_offset_max_m")
    random_seed = LaunchConfiguration("random_seed")
    telemetry_enabled = LaunchConfiguration("telemetry_enabled")
    telemetry_csv_path = LaunchConfiguration("telemetry_csv_path")

    policy_node = Node(
        package="franka_emika_panda",
        executable="panda_reach_reduce_shake_ros2",
        name="panda_rf3_real_policy",
        output="screen",
        arguments=["--ros", "--policy"],
        parameters=[
            {
                "use_sim_time": False,
                "robot_type": "panda",
                "onnx_model_path": model,
                "frame_id": frame_id,
                "joint_states_topic": joint_states_topic,
                "tcp_source": "transform_topic",
                "tcp_transform_topic": tcp_transform_topic,
                "tcp_pose_topic": "ee_pose",
                "tcp_frame_id": "panda_hand_tcp",
                "command_topic": command_topic,
                "command_message_type": command_message_type,
                "tcp_observation_mode": "tcp_error_normalized",
                "tcp_observation_scale": 0.75,
                "tcp_observation_signs": "1,1,1",
                "tcp_observation_unit_scale": 1.0,
                "joint_observation_mode": (
                    "normalized_position_velocity_previous_position_velocity"
                ),
                "observation_clip": 3.0,
                "action_scale": 1.0,
                "action_output_mode": "absolute",
                "action_delta_scale": 0.05,
                "prev_action_scale": 3.141592653589793,
                "control_hz": 60.0,
                "trajectory_duration_s": 0.015,
                "target_location_m": target_location_m,
                "target_random_offset_max_m": target_random_offset_max_m,
                "random_seed": ParameterValue(random_seed, value_type=int),
                "resample_target_on_success": False,
                "symmetric_z_offset": False,
                "smooth_actions": True,
                "action_smoothing_alpha": 0.20,
                "limit_action_delta": True,
                "action_rate_limit_hz": 60.0,
                "action_velocity_limits_rad_s": (
                    "0.783,0.783,0.783,0.783,0.9396,0.9396,0.9396"
                ),
                "use_measured_velocity_governor": True,
                "measured_velocity_governor_start_ratio": 0.75,
                "startup_home_enabled": True,
                "startup_home_joints": "0,-0.943,0,-2.514,0,1.611,0",
                "startup_home_tolerance_rad": 0.03,
                "startup_home_qvel_l2": 0.15,
                "startup_home_hold_ticks": 12,
                "hold_enabled": True,
                "hold_enter_tcp_m": 0.07,
                "hold_exit_tcp_m": 0.10,
                "hold_enter_qvel_l2": 2.0,
                "hold_enter_ticks": 5,
                "telemetry_enabled": ParameterValue(
                    telemetry_enabled,
                    value_type=bool,
                ),
                "telemetry_csv_path": telemetry_csv_path,
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "model",
                default_value=[
                    PathJoinSubstitution(
                        [
                            FindPackageShare("franka_emika_panda"),
                            "ppo_final_rf3.onnx",
                        ]
                    )
                ],
                description="RF3 ONNX policy path.",
            ),
            DeclareLaunchArgument(
                "command_topic",
                default_value="/medrct_franka/desired_js",
                description="Command topic for the real arm.",
            ),
            DeclareLaunchArgument(
                "command_message_type",
                default_value="joint_state",
                description="Command message type: joint_state or joint_trajectory.",
            ),
            DeclareLaunchArgument(
                "joint_states_topic",
                default_value="/medrct_franka/measured_js",
                description="Measured JointState topic.",
            ),
            DeclareLaunchArgument(
                "tcp_transform_topic",
                default_value="/medrct_franka/measured_tf",
                description="Measured TCP TransformStamped topic.",
            ),
            DeclareLaunchArgument(
                "frame_id",
                default_value="panda_link0",
                description="Base/task frame for targets.",
            ),
            DeclareLaunchArgument(
                "target_location_m",
                default_value="0.6,0,0.5",
                description="RF3 target center in the base/task frame.",
            ),
            DeclareLaunchArgument(
                "target_random_offset_max_m",
                default_value="0.06,0.06,0.06",
                description="Target random offset magnitude. X/Y symmetric, Z positive-only.",
            ),
            DeclareLaunchArgument(
                "random_seed",
                default_value="0",
                description="Target sampling seed. 0 leaves sampling nondeterministic.",
            ),
            DeclareLaunchArgument(
                "telemetry_enabled",
                default_value="false",
                description="Enable CSV telemetry from the policy node.",
            ),
            DeclareLaunchArgument(
                "telemetry_csv_path",
                default_value="",
                description="Optional telemetry CSV path.",
            ),
            policy_node,
        ]
    )
