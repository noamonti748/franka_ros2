# Copyright (c) 2026 Franka Robotics GmbH
#
# Licensed under the Apache License, Version 2.0

# RF3 evaluation launch:
#   - run the RF3 ONNX policy for a fixed number of sampled reach targets
#   - after each target attempt, command the Panda back to the RF3 home pose
#   - only sample the next target after the home pose is reached and settled
#
# This deliberately uses commanded homing through joint_trajectory_controller rather
# than teleporting Gazebo state, so the evaluation stays on the same controller path
# as policy deployment.

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
    random_seed = LaunchConfiguration('random_seed')
    target_location_m = LaunchConfiguration('target_location_m')
    target_random_offset_max_m = LaunchConfiguration('target_random_offset_max_m')
    eval_enabled = LaunchConfiguration('eval_enabled')
    eval_target_count = LaunchConfiguration('eval_target_count')
    eval_max_steps_per_target = LaunchConfiguration('eval_max_steps_per_target')
    eval_shutdown_on_complete = LaunchConfiguration('eval_shutdown_on_complete')
    eval_home_tolerance_rad = LaunchConfiguration('eval_home_tolerance_rad')
    eval_home_qvel_l2 = LaunchConfiguration('eval_home_qvel_l2')
    eval_home_hold_ticks = LaunchConfiguration('eval_home_hold_ticks')
    eval_summary_csv_path = LaunchConfiguration('eval_summary_csv_path')
    action_smoothing_alpha = LaunchConfiguration('action_smoothing_alpha')
    action_velocity_limits_rad_s = LaunchConfiguration('action_velocity_limits_rad_s')
    measured_velocity_governor_start_ratio = LaunchConfiguration(
        'measured_velocity_governor_start_ratio'
    )
    trajectory_duration_s = LaunchConfiguration('trajectory_duration_s')
    hold_enabled = LaunchConfiguration('hold_enabled')
    hold_enter_tcp_m = LaunchConfiguration('hold_enter_tcp_m')
    hold_exit_tcp_m = LaunchConfiguration('hold_exit_tcp_m')
    hold_enter_qvel_l2 = LaunchConfiguration('hold_enter_qvel_l2')
    hold_enter_ticks = LaunchConfiguration('hold_enter_ticks')
    telemetry_enabled = LaunchConfiguration('telemetry_enabled')
    telemetry_csv_path = LaunchConfiguration('telemetry_csv_path')
    telemetry_decimation = LaunchConfiguration('telemetry_decimation')
    telemetry_flush_every_n_rows = LaunchConfiguration('telemetry_flush_every_n_rows')
    initial_joint1 = LaunchConfiguration('initial_joint1')
    initial_joint2 = LaunchConfiguration('initial_joint2')
    initial_joint3 = LaunchConfiguration('initial_joint3')
    initial_joint4 = LaunchConfiguration('initial_joint4')
    initial_joint5 = LaunchConfiguration('initial_joint5')
    initial_joint6 = LaunchConfiguration('initial_joint6')
    initial_joint7 = LaunchConfiguration('initial_joint7')

    rf3_eval = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('franka_gazebo_bringup'),
                'launch',
                'gazebo_panda_rf3_onnx_policy.launch.py',
            ])
        ]),
        launch_arguments={
            'load_gripper': load_gripper,
            'rviz': rviz,
            'gz_args': gz_args,
            'model': model,
            'random_seed': random_seed,
            'target_location_m': target_location_m,
            'target_random_offset_max_m': target_random_offset_max_m,
            'eval_enabled': eval_enabled,
            'eval_target_count': eval_target_count,
            'eval_reset_home_between_targets': 'true',
            'eval_home_joints': '0,-0.943,0,-2.514,0,1.611,0',
            'eval_home_tolerance_rad': eval_home_tolerance_rad,
            'eval_home_qvel_l2': eval_home_qvel_l2,
            'eval_home_hold_ticks': eval_home_hold_ticks,
            'eval_max_steps_per_target': eval_max_steps_per_target,
            'eval_shutdown_on_complete': eval_shutdown_on_complete,
            'eval_summary_csv_path': eval_summary_csv_path,
            'action_smoothing_alpha': action_smoothing_alpha,
            'action_velocity_limits_rad_s': action_velocity_limits_rad_s,
            'measured_velocity_governor_start_ratio': (
                measured_velocity_governor_start_ratio
            ),
            'trajectory_duration_s': trajectory_duration_s,
            'hold_enabled': hold_enabled,
            'hold_enter_tcp_m': hold_enter_tcp_m,
            'hold_exit_tcp_m': hold_exit_tcp_m,
            'hold_enter_qvel_l2': hold_enter_qvel_l2,
            'hold_enter_ticks': hold_enter_ticks,
            'telemetry_enabled': telemetry_enabled,
            'telemetry_csv_path': telemetry_csv_path,
            'telemetry_decimation': telemetry_decimation,
            'telemetry_flush_every_n_rows': telemetry_flush_every_n_rows,
            'initial_joint1': initial_joint1,
            'initial_joint2': initial_joint2,
            'initial_joint3': initial_joint3,
            'initial_joint4': initial_joint4,
            'initial_joint5': initial_joint5,
            'initial_joint6': initial_joint6,
            'initial_joint7': initial_joint7,
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument('load_gripper', default_value='true'),
        DeclareLaunchArgument('rviz', default_value='false'),
        DeclareLaunchArgument('gz_args', default_value='-r empty.sdf'),
        DeclareLaunchArgument(
            'model',
            default_value=[
                PathJoinSubstitution([
                    FindPackageShare('franka_emika_panda'),
                    'ppo_final_rf3.onnx',
                ])
            ],
        ),
        DeclareLaunchArgument(
            'random_seed',
            default_value='1',
            description='Target sampling seed for reproducible 20-target RF3 eval runs.',
        ),
        DeclareLaunchArgument(
            'target_location_m',
            default_value='0.6,0,0.5',
            description='RF3 eval target center in the arm base frame.',
        ),
        DeclareLaunchArgument(
            'target_random_offset_max_m',
            default_value='0.06,0.06,0.06',
            description='RF3 eval target offset magnitude. X/Y are symmetric; Z is positive-only.',
        ),
        DeclareLaunchArgument(
            'eval_enabled',
            default_value='true',
            description='Keep enabled for RF3 eval. Set false to use this launch as a normal RF3 run with eval defaults exposed.',
        ),
        DeclareLaunchArgument(
            'eval_target_count',
            default_value='20',
            description='Number of target attempts in one RF3 eval run.',
        ),
        DeclareLaunchArgument(
            'eval_max_steps_per_target',
            default_value='1800',
            description='Max control ticks per target attempt before counting a failure and homing. 1800 ticks is 30 seconds at 60 Hz. Use 0 to disable.',
        ),
        DeclareLaunchArgument(
            'eval_shutdown_on_complete',
            default_value='false',
            description='If true, shut down the policy node after eval_target_count attempts.',
        ),
        DeclareLaunchArgument(
            'eval_home_tolerance_rad',
            default_value='0.03',
            description='Max absolute joint error allowed before home reset is accepted.',
        ),
        DeclareLaunchArgument(
            'eval_home_qvel_l2',
            default_value='0.15',
            description='Joint velocity L2 threshold for accepting home reset.',
        ),
        DeclareLaunchArgument(
            'eval_home_hold_ticks',
            default_value='12',
            description='Consecutive settled home ticks required before sampling the next target.',
        ),
        DeclareLaunchArgument(
            'eval_summary_csv_path',
            default_value='/tmp/panda_tuning_logs/rf3_eval_summary.csv',
            description='One-row-per-target RF3 eval summary CSV path.',
        ),
        DeclareLaunchArgument(
            'action_smoothing_alpha',
            default_value='0.20',
            description='RF3 command low-pass alpha.',
        ),
        DeclareLaunchArgument(
            'action_velocity_limits_rad_s',
            default_value='0.783,0.783,0.783,0.783,0.9396,0.9396,0.9396',
            description='RF3 per-joint command slew limits in rad/s.',
        ),
        DeclareLaunchArgument(
            'measured_velocity_governor_start_ratio',
            default_value='0.75',
            description='RF3 measured-velocity governor threshold fraction.',
        ),
        DeclareLaunchArgument(
            'trajectory_duration_s',
            default_value='0.015',
            description='Single-point JointTrajectory duration.',
        ),
        DeclareLaunchArgument('hold_enabled', default_value='true'),
        DeclareLaunchArgument('hold_enter_tcp_m', default_value='0.07'),
        DeclareLaunchArgument('hold_exit_tcp_m', default_value='0.10'),
        DeclareLaunchArgument('hold_enter_qvel_l2', default_value='2.0'),
        DeclareLaunchArgument('hold_enter_ticks', default_value='5'),
        DeclareLaunchArgument(
            'telemetry_enabled',
            default_value='true',
            description='Enable CSV telemetry by default for eval runs.',
        ),
        DeclareLaunchArgument(
            'telemetry_csv_path',
            default_value='/tmp/panda_tuning_logs/rf3_eval.csv',
            description='RF3 eval telemetry CSV path.',
        ),
        DeclareLaunchArgument('telemetry_decimation', default_value='1'),
        DeclareLaunchArgument('telemetry_flush_every_n_rows', default_value='1'),
        DeclareLaunchArgument('initial_joint1', default_value='0.0'),
        DeclareLaunchArgument('initial_joint2', default_value='-0.943'),
        DeclareLaunchArgument('initial_joint3', default_value='0.0'),
        DeclareLaunchArgument('initial_joint4', default_value='-2.514'),
        DeclareLaunchArgument('initial_joint5', default_value='0.0'),
        DeclareLaunchArgument('initial_joint6', default_value='1.611'),
        DeclareLaunchArgument('initial_joint7', default_value='0.0'),
        rf3_eval,
    ])
