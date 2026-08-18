# Copyright (c) 2026 Franka Robotics GmbH
#
# Licensed under the Apache License, Version 2.0

# Runs the ppo_final_lower_vib reach-and-hold ONNX policy on a Panda in Gazebo.
# This wraps gazebo_panda_reduce_shake_onnx_policy.launch.py because both policies
# share the same single-action ONNX interface; the lower-vib policy switches joint
# observations to [qpos/pi, qvel/2, prev_qpos/pi, prev_qvel/2] and normalizes
# prev_action by pi.
#
# Deployment damping defaults mirror the hard action channel documented for the
# lower-vib checkpoint: first-order smoothing alpha=0.20, 60 Hz command rate
# limiting, measured-velocity governor at 75% of the hard limit, and conservative
# per-joint velocity limits.

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    load_gripper = LaunchConfiguration('load_gripper')
    rviz = LaunchConfiguration('rviz')
    gz_args = LaunchConfiguration('gz_args')
    model = LaunchConfiguration('model')
    action_smoothing_alpha = LaunchConfiguration('action_smoothing_alpha')
    action_velocity_limits_rad_s = LaunchConfiguration('action_velocity_limits_rad_s')
    measured_velocity_governor_start_ratio = LaunchConfiguration(
        'measured_velocity_governor_start_ratio'
    )
    action_notch_filter_enabled = LaunchConfiguration('action_notch_filter_enabled')
    action_notch_filter_frequencies_hz = LaunchConfiguration(
        'action_notch_filter_frequencies_hz'
    )
    action_notch_filter_q = LaunchConfiguration('action_notch_filter_q')
    action_notch_filter_secondary_frequencies_hz = LaunchConfiguration(
        'action_notch_filter_secondary_frequencies_hz'
    )
    action_notch_filter_secondary_q = LaunchConfiguration(
        'action_notch_filter_secondary_q'
    )
    action_notch_filter_stage = LaunchConfiguration('action_notch_filter_stage')
    hold_enabled = LaunchConfiguration('hold_enabled')
    hold_enter_tcp_m = LaunchConfiguration('hold_enter_tcp_m')
    hold_exit_tcp_m = LaunchConfiguration('hold_exit_tcp_m')
    hold_enter_qvel_l2 = LaunchConfiguration('hold_enter_qvel_l2')
    hold_enter_ticks = LaunchConfiguration('hold_enter_ticks')
    trajectory_duration_s = LaunchConfiguration('trajectory_duration_s')
    target_location_m = LaunchConfiguration('target_location_m')
    target_random_offset_max_m = LaunchConfiguration('target_random_offset_max_m')
    random_seed = LaunchConfiguration('random_seed')
    telemetry_enabled = LaunchConfiguration('telemetry_enabled')
    telemetry_csv_path = LaunchConfiguration('telemetry_csv_path')
    telemetry_decimation = LaunchConfiguration('telemetry_decimation')
    telemetry_flush_every_n_rows = LaunchConfiguration('telemetry_flush_every_n_rows')

    lower_vib_policy = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('franka_gazebo_bringup'),
                'launch',
                'gazebo_panda_reduce_shake_onnx_policy.launch.py',
            ])
        ]),
        launch_arguments={
            'load_gripper': load_gripper,
            'rviz': rviz,
            'gz_args': gz_args,
            'model': model,
            'joint_observation_mode': (
                'normalized_position_velocity_previous_position_velocity'
            ),
            'prev_action_scale': '3.141592653589793',
            'smooth_actions': 'true',
            'action_smoothing_alpha': action_smoothing_alpha,
            'limit_action_delta': 'true',
            'action_rate_limit_hz': '60.0',
            'action_velocity_limits_rad_s': action_velocity_limits_rad_s,
            'use_measured_velocity_governor': 'true',
            'measured_velocity_governor_start_ratio': (
                measured_velocity_governor_start_ratio
            ),
            'action_notch_filter_enabled': action_notch_filter_enabled,
            'action_notch_filter_frequencies_hz': action_notch_filter_frequencies_hz,
            'action_notch_filter_q': action_notch_filter_q,
            'action_notch_filter_secondary_frequencies_hz': (
                action_notch_filter_secondary_frequencies_hz
            ),
            'action_notch_filter_secondary_q': action_notch_filter_secondary_q,
            'action_notch_filter_stage': action_notch_filter_stage,
            'hold_enabled': hold_enabled,
            'hold_enter_tcp_m': hold_enter_tcp_m,
            'hold_exit_tcp_m': hold_exit_tcp_m,
            'hold_enter_qvel_l2': hold_enter_qvel_l2,
            'hold_enter_ticks': hold_enter_ticks,
            'trajectory_duration_s': trajectory_duration_s,
            'target_location_m': target_location_m,
            'target_random_offset_max_m': target_random_offset_max_m,
            'random_seed': random_seed,
            'telemetry_enabled': telemetry_enabled,
            'telemetry_csv_path': telemetry_csv_path,
            'telemetry_decimation': telemetry_decimation,
            'telemetry_flush_every_n_rows': telemetry_flush_every_n_rows,
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument('load_gripper', default_value='true'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('gz_args', default_value='-r empty.sdf'),
        DeclareLaunchArgument(
            'model',
            default_value=[
                PathJoinSubstitution([
                    FindPackageShare('franka_emika_panda'),
                    'ppo_final_lower_vib.onnx',
                ])
            ],
            description='Lower-vib ONNX policy exported from ppo_final_lower_vib.zip.',
        ),
        DeclareLaunchArgument(
            'action_smoothing_alpha',
            default_value='0.20',
            description='Low-pass command alpha. 0.20 matches the lower-vib training/deployment hard action channel.',
        ),
        DeclareLaunchArgument(
            'action_velocity_limits_rad_s',
            default_value='0.744,0.744,0.744,0.744,0.893,0.893,0.893',
            description='Per-joint command slew limits in rad/s. Start here for vibration reduction, then tune in Gazebo/robot logs.',
        ),
        DeclareLaunchArgument(
            'measured_velocity_governor_start_ratio',
            default_value='0.75',
            description='Measured-velocity governor starts at this fraction of each velocity limit.',
        ),
        DeclareLaunchArgument(
            'action_notch_filter_enabled',
            default_value='false',
            description='Enable per-joint IIR notch filtering on target actions before smoothing/rate limiting.',
        ),
        DeclareLaunchArgument(
            'action_notch_filter_frequencies_hz',
            default_value='0,0,0,0,0,0,0',
            description='Primary per-joint notch frequencies in Hz. For current lower-vib logs, start with 10 Hz on joints 1-6 and 0 on joint 7.',
        ),
        DeclareLaunchArgument(
            'action_notch_filter_q',
            default_value='8.0',
            description='Primary notch Q. Higher values are narrower and attenuate less off-frequency.',
        ),
        DeclareLaunchArgument(
            'action_notch_filter_secondary_frequencies_hz',
            default_value='0,0,0,0,0,0,0',
            description='Optional secondary per-joint notch frequencies in Hz. Use 0 to disable.',
        ),
        DeclareLaunchArgument(
            'action_notch_filter_secondary_q',
            default_value='8.0',
            description='Secondary notch Q. Higher values are narrower and attenuate less off-frequency.',
        ),
        DeclareLaunchArgument(
            'action_notch_filter_stage',
            default_value='target_action',
            description='Where to apply notch filtering: target_action, final_command, or both.',
        ),
        DeclareLaunchArgument(
            'hold_enabled',
            default_value='false',
            description='Enable the near-goal command hold (deadband with hysteresis) that latches a fixed setpoint once the TCP is near the goal and nearly still, to kill the hold-point limit cycle. Off by default.',
        ),
        DeclareLaunchArgument(
            'hold_enter_tcp_m',
            default_value='0.02',
            description='Enter hold when measured TCP distance is below this many meters (and qvel L2 below hold_enter_qvel_l2) for hold_enter_ticks consecutive ticks.',
        ),
        DeclareLaunchArgument(
            'hold_exit_tcp_m',
            default_value='0.04',
            description='Exit hold when measured TCP distance exceeds this many meters. Must be greater than hold_enter_tcp_m for hysteresis.',
        ),
        DeclareLaunchArgument(
            'hold_enter_qvel_l2',
            default_value='0.5',
            description='Enter hold only while measured joint velocity L2 (rad/s) is below this value.',
        ),
        DeclareLaunchArgument(
            'hold_enter_ticks',
            default_value='10',
            description='Number of consecutive qualifying ticks required before latching the near-goal hold setpoint.',
        ),
        DeclareLaunchArgument(
            'trajectory_duration_s',
            default_value='0.015',
            description='Single-point JointTrajectory duration, approximately 0.9 / 60 Hz.',
        ),
        DeclareLaunchArgument(
            'target_location_m',
            default_value='0.32,0,0.5',
            description='Reach-and-hold target center in the arm base frame (meters).',
        ),
        DeclareLaunchArgument(
            'target_random_offset_max_m',
            default_value='0.06,0.06,0.06',
            description='Per-axis random target offset in meters.',
        ),
        DeclareLaunchArgument(
            'random_seed',
            default_value='0',
            description='Target sampling seed. 0 leaves target sampling nondeterministic.',
        ),
        DeclareLaunchArgument(
            'telemetry_enabled',
            default_value='false',
            description='Enable CSV tuning telemetry from panda_reach_reduce_shake_ros2.',
        ),
        DeclareLaunchArgument(
            'telemetry_csv_path',
            default_value='',
            description='Telemetry CSV output path. If telemetry is enabled and this is empty, the node writes a timestamped file under ~/.ros.',
        ),
        DeclareLaunchArgument(
            'telemetry_decimation',
            default_value='1',
            description='Write one telemetry row every N policy/control ticks.',
        ),
        DeclareLaunchArgument(
            'telemetry_flush_every_n_rows',
            default_value='60',
            description='Flush telemetry CSV after this many written rows.',
        ),
        lower_vib_policy,
    ])
