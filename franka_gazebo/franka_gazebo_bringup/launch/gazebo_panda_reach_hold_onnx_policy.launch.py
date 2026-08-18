# Copyright (c) 2026 Franka Robotics GmbH
#
# Licensed under the Apache License, Version 2.0

# Runs the ppo_final_constrained "reach and hold" ONNX policy on a Panda in Gazebo.
# Unlike gazebo_panda_onnx_policy.launch.py (ppo_track_franka.onnx, multi-target
# "track" behaviour with target resampling on success), this checkpoint was trained
# to reach a single target and hold station there -- no resampling after success.
#
# Observation preprocessing must match training exactly:
#   - joint1..joint7: [qpos/pi, qvel/2.0, qacc/10.0], each dim clipped to +/-3
#     (joint_observation_mode=normalized_position_velocity_acceleration,
#     observation_clip=3.0). qacc is approximated on real/Gazebo joint_states via
#     backward finite difference of velocity (no acceleration field on hardware).
#   - tcp_pos: (tcp_m - target_m) / 0.75, clipped to +/-3
#     (tcp_observation_mode=tcp_error_normalized, tcp_observation_scale=0.75,
#     tcp_observation_unit_scale=1.0, tcp_observation_signs=1,1,1).
#   - Actions are absolute joint position targets in radians (action_output_mode=absolute).
#
# Target defaults reproduce training: center (0.32, 0, 0.5) m with up to 0.06 m
# random offset per axis, sampled symmetrically on X/Y/Z (symmetric_z_offset=true;
# the legacy track checkpoint samples Z as U[0, +max] instead -- see
# panda_reach_constrained_ros2.py's _sample_target_offset_reach_hold for details/
# assumptions). This node (panda_reach_constrained_ros2) is a dedicated module,
# separate from the legacy panda_track_reach_ros2 node used by
# gazebo_panda_onnx_policy.launch.py / gazebo_fr3_onnx_policy.launch.py.

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    load_gripper = LaunchConfiguration('load_gripper')
    rviz = LaunchConfiguration('rviz')
    gz_args = LaunchConfiguration('gz_args')
    model = LaunchConfiguration('model')
    tcp_observation_mode = LaunchConfiguration('tcp_observation_mode')
    tcp_observation_scale = LaunchConfiguration('tcp_observation_scale')
    tcp_observation_signs = LaunchConfiguration('tcp_observation_signs')
    tcp_observation_unit_scale = LaunchConfiguration('tcp_observation_unit_scale')
    joint_observation_mode = LaunchConfiguration('joint_observation_mode')
    observation_clip = LaunchConfiguration('observation_clip')
    action_scale = LaunchConfiguration('action_scale')
    action_output_mode = LaunchConfiguration('action_output_mode')
    action_delta_scale = LaunchConfiguration('action_delta_scale')
    control_hz = LaunchConfiguration('control_hz')
    trajectory_duration_s = LaunchConfiguration('trajectory_duration_s')
    target_location_m = LaunchConfiguration('target_location_m')
    target_random_offset_max_m = LaunchConfiguration('target_random_offset_max_m')
    resample_target_on_success = LaunchConfiguration('resample_target_on_success')
    symmetric_z_offset = LaunchConfiguration('symmetric_z_offset')
    smooth_actions = LaunchConfiguration('smooth_actions')
    limit_action_delta = LaunchConfiguration('limit_action_delta')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('franka_gazebo_bringup'),
                'launch',
                'gazebo_panda_example_controller.launch.py',
            ])
        ]),
        launch_arguments={
            'load_gripper': load_gripper,
            'rviz': rviz,
            'gz_args': gz_args,
            'controller': 'panda_joint_trajectory_controller',
        }.items(),
    )

    policy_node = Node(
        package='franka_emika_panda',
        executable='panda_reach_constrained_ros2',
        name='panda_reach_constrained',
        output='screen',
        arguments=['--ros', '--policy'],
        parameters=[{
            'use_sim_time': True,
            'onnx_model_path': model,
            'tcp_source': 'tf',
            'tcp_frame_id': 'panda_hand_tcp',
            'frame_id': 'panda_link0',
            'command_topic': 'panda_joint_trajectory_controller/joint_trajectory',
            'tcp_observation_mode': tcp_observation_mode,
            'tcp_observation_scale': ParameterValue(tcp_observation_scale, value_type=float),
            'tcp_observation_signs': tcp_observation_signs,
            'tcp_observation_unit_scale': ParameterValue(
                tcp_observation_unit_scale,
                value_type=float,
            ),
            'joint_observation_mode': joint_observation_mode,
            'observation_clip': ParameterValue(observation_clip, value_type=float),
            'action_scale': ParameterValue(action_scale, value_type=float),
            'action_output_mode': action_output_mode,
            'action_delta_scale': ParameterValue(action_delta_scale, value_type=float),
            'control_hz': ParameterValue(control_hz, value_type=float),
            'trajectory_duration_s': ParameterValue(trajectory_duration_s, value_type=float),
            'target_location_m': target_location_m,
            'target_random_offset_max_m': target_random_offset_max_m,
            'resample_target_on_success': resample_target_on_success,
            'symmetric_z_offset': symmetric_z_offset,
            'smooth_actions': smooth_actions,
            'limit_action_delta': limit_action_delta,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument('load_gripper', default_value='true'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('gz_args', default_value='-r empty.sdf'),
        DeclareLaunchArgument('model', default_value=[
            PathJoinSubstitution([
                FindPackageShare('franka_emika_panda'),
                'ppo_final_constrained.onnx',
            ])
        ]),
        DeclareLaunchArgument(
            'tcp_observation_mode',
            default_value='tcp_error_normalized',
            description='Policy tcp_pos input: tcp_error_normalized, target_error_normalized, tcp_minus_target, target_minus_tcp, target_delta, tcp, or target.',
        ),
        DeclareLaunchArgument(
            'tcp_observation_scale',
            default_value='0.75',
            description='ObsTcpScale used for tcp_error_normalized: (tcp_position - target_position) / scale.',
        ),
        DeclareLaunchArgument(
            'tcp_observation_signs',
            default_value='1,1,1',
            description='Comma-separated XYZ signs applied after tcp_pos preprocessing, e.g. 1,-1,1.',
        ),
        DeclareLaunchArgument(
            'tcp_observation_unit_scale',
            default_value='1.0',
            description='Unit multiplier before ObsTcpScale. This checkpoint was trained on meters, so 1.0 (not the 100.0 used by the ppo_track_franka.onnx UE/cm checkpoint).',
        ),
        DeclareLaunchArgument(
            'joint_observation_mode',
            default_value='normalized_position_velocity_acceleration',
            description='Joint input packing for each jointN ONNX input: [qpos/pi, qvel/2.0, qacc/10.0] for this checkpoint.',
        ),
        DeclareLaunchArgument(
            'observation_clip',
            default_value='3.0',
            description='Clip every joint/tcp observation dim to +/-this value after scaling (matches the trained Box(-3, 3) observation space). 0 disables clipping.',
        ),
        DeclareLaunchArgument(
            'action_scale',
            default_value='1.0',
            description='Multiplier applied to the raw ONNX actuator outputs before filtering.',
        ),
        DeclareLaunchArgument(
            'action_output_mode',
            default_value='absolute',
            description='Interpret ONNX outputs as absolute, normalized_absolute, delta_position, normalized_delta_position, delta_command, or normalized_delta_command. This checkpoint outputs absolute joint position targets directly.',
        ),
        DeclareLaunchArgument(
            'action_delta_scale',
            default_value='0.05',
            description='Joint-radian step size for delta action output modes (unused for action_output_mode=absolute).',
        ),
        DeclareLaunchArgument(
            'control_hz',
            default_value='60.0',
            description='Policy/control loop rate in Hz. 60.0 matches the training control rate '
                        'for this checkpoint family: task/track_reach_session.py in the '
                        'MuJoCo-SB3 training repo defines action_rate_limit_hz=60.0, and every '
                        'related env/rollout/eval script defaults control_hz to 60.0 too. The '
                        'exact training script for ppo_final_constrained itself was not found '
                        '(the checkpoint is not present in the training repo), so this is strong '
                        'but not 100%-certain corroborating evidence -- override if you have a '
                        'more authoritative source. If you do change this, also rescale '
                        'trajectory_duration_s below (roughly 0.8-0.95 / control_hz).',
        ),
        DeclareLaunchArgument(
            'trajectory_duration_s',
            default_value='0.015',
            description='Time-from-start for each published JointTrajectory point (~0.9/control_hz '
                        'at the default 60 Hz). A new single-point trajectory is published every '
                        '1/control_hz seconds, replacing the in-flight one; if this duration is '
                        'longer than that period (the previous default of 0.05s was ~3x longer), '
                        'joint_trajectory_controller repeatedly interrupts its spline before it '
                        'ever decelerates into the target, so the arm is always caught in the '
                        'high-velocity ramp-up portion of the curve -- this, not control_hz, was '
                        'the root cause of the "too fast/snappy" deployment feel. Keeping this '
                        'just under 1/control_hz lets each spline (almost) finish before being '
                        'replaced, approximating the paced zero-order-hold position updates the '
                        'policy actually saw from MuJoCo\'s position-servo actuators during '
                        'training.',
        ),
        DeclareLaunchArgument(
            'target_location_m',
            default_value='0.32,0,0.5',
            description='Reach-and-hold target center in the arm base frame (meters), matching training.',
        ),
        DeclareLaunchArgument(
            'target_random_offset_max_m',
            default_value='0.06,0.06,0.06',
            description='Per-axis max random offset (meters) applied to target_location_m at episode reset, matching training.',
        ),
        DeclareLaunchArgument(
            'resample_target_on_success',
            default_value='false',
            description='If false (reach-and-hold), keep holding the same target after success instead of resampling a new one.',
        ),
        DeclareLaunchArgument(
            'symmetric_z_offset',
            default_value='true',
            description='If true, sample the Z target offset as U[-max, +max] like X/Y instead of the legacy U[0, +max]. ASSUMPTION: the training spec ("random radius 0.06 on X, Y, Z") was ambiguous about whether Z stayed positive-only; this defaults to symmetric sampling on all three axes -- adjust if training actually kept Z positive-only.',
        ),
        DeclareLaunchArgument(
            'smooth_actions',
            default_value='false',
            description='Low-pass filter raw policy actions before commanding. This checkpoint was trained with no action smoothing, so disabled by default to match training in sim; consider enabling for real hardware deployment.',
        ),
        DeclareLaunchArgument(
            'limit_action_delta',
            default_value='false',
            description='Per-joint velocity-based rate limiting on commanded actions. This checkpoint was trained with no rate limiting, so disabled by default to match training in sim; consider enabling for real hardware deployment.',
        ),
        gazebo,
        policy_node,
    ])
