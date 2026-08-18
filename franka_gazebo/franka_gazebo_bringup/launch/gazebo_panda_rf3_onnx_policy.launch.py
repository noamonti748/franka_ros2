# Copyright (c) 2026 Franka Robotics GmbH
#
# Licensed under the Apache License, Version 2.0

# Runs the RF3 constrained reach-and-hold ONNX policy on a Panda in Gazebo.
#
# RF3 is exported from franka_emika_panda/ppo_final_rf3_balanced.zip and uses the
# lower-vib style ONNX interface:
#   - joint1..joint7:
#     [qpos/pi, qvel/2, prev_qpos/pi, prev_qvel/2], clipped to +/-3
#   - prev_action: previous applied/governed command divided by pi
#   - tcp_pos: (tcp_m - target_m) / 0.75, clipped to +/-3
#   - action: absolute 7-joint position target vector in radians
#
# Defaults intentionally match RF3's training hard-action channel, then add a
# near-goal setpoint hold because RF3's raw action stream still limit-cycles near
# the target in Gazebo:
#   - startup command to the RF3 constrained home pose before policy inference
#   - first-order smoothing alpha=0.20
#   - 60 Hz target-rate limiter
#   - measured-velocity governor at 75% of the per-joint RF3 limit
#   - RF3 velocity limits:
#     [0.783, 0.783, 0.783, 0.783, 0.9396, 0.9396, 0.9396] rad/s
#   - target Z offset is positive-only, matching RF3 training:
#     X/Y in +/-0.06 m, Z in [0, +0.06] m
#   - near-goal hold enabled:
#     enter at 0.07 m, exit at 0.10 m, qvel L2 gate 2.0 rad/s, 5 ticks

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
    use_sim_time = LaunchConfiguration('use_sim_time')
    frame_id = LaunchConfiguration('frame_id')
    tcp_source = LaunchConfiguration('tcp_source')
    tcp_frame_id = LaunchConfiguration('tcp_frame_id')
    tcp_pose_topic = LaunchConfiguration('tcp_pose_topic')
    tcp_transform_topic = LaunchConfiguration('tcp_transform_topic')
    joint_states_topic = LaunchConfiguration('joint_states_topic')
    command_topic = LaunchConfiguration('command_topic')
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
    eval_enabled = LaunchConfiguration('eval_enabled')
    eval_target_count = LaunchConfiguration('eval_target_count')
    eval_reset_home_between_targets = LaunchConfiguration(
        'eval_reset_home_between_targets'
    )
    eval_home_joints = LaunchConfiguration('eval_home_joints')
    eval_home_tolerance_rad = LaunchConfiguration('eval_home_tolerance_rad')
    eval_home_qvel_l2 = LaunchConfiguration('eval_home_qvel_l2')
    eval_home_hold_ticks = LaunchConfiguration('eval_home_hold_ticks')
    eval_max_steps_per_target = LaunchConfiguration('eval_max_steps_per_target')
    eval_shutdown_on_complete = LaunchConfiguration('eval_shutdown_on_complete')
    eval_summary_csv_path = LaunchConfiguration('eval_summary_csv_path')
    startup_home_enabled = LaunchConfiguration('startup_home_enabled')
    startup_home_joints = LaunchConfiguration('startup_home_joints')
    startup_home_tolerance_rad = LaunchConfiguration('startup_home_tolerance_rad')
    startup_home_qvel_l2 = LaunchConfiguration('startup_home_qvel_l2')
    startup_home_hold_ticks = LaunchConfiguration('startup_home_hold_ticks')
    target_location_m = LaunchConfiguration('target_location_m')
    target_random_offset_max_m = LaunchConfiguration('target_random_offset_max_m')
    random_seed = LaunchConfiguration('random_seed')
    initial_joint1 = LaunchConfiguration('initial_joint1')
    initial_joint2 = LaunchConfiguration('initial_joint2')
    initial_joint3 = LaunchConfiguration('initial_joint3')
    initial_joint4 = LaunchConfiguration('initial_joint4')
    initial_joint5 = LaunchConfiguration('initial_joint5')
    initial_joint6 = LaunchConfiguration('initial_joint6')
    initial_joint7 = LaunchConfiguration('initial_joint7')
    telemetry_enabled = LaunchConfiguration('telemetry_enabled')
    telemetry_csv_path = LaunchConfiguration('telemetry_csv_path')
    telemetry_decimation = LaunchConfiguration('telemetry_decimation')
    telemetry_flush_every_n_rows = LaunchConfiguration('telemetry_flush_every_n_rows')

    rf3_policy = IncludeLaunchDescription(
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
            'use_sim_time': use_sim_time,
            'frame_id': frame_id,
            'tcp_source': tcp_source,
            'tcp_frame_id': tcp_frame_id,
            'tcp_pose_topic': tcp_pose_topic,
            'tcp_transform_topic': tcp_transform_topic,
            'joint_states_topic': joint_states_topic,
            'command_topic': command_topic,
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
            'symmetric_z_offset': 'false',
            'resample_target_on_success': 'false',
            'initial_joint1': initial_joint1,
            'initial_joint2': initial_joint2,
            'initial_joint3': initial_joint3,
            'initial_joint4': initial_joint4,
            'initial_joint5': initial_joint5,
            'initial_joint6': initial_joint6,
            'initial_joint7': initial_joint7,
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
            'eval_enabled': eval_enabled,
            'eval_target_count': eval_target_count,
            'eval_reset_home_between_targets': eval_reset_home_between_targets,
            'eval_home_joints': eval_home_joints,
            'eval_home_tolerance_rad': eval_home_tolerance_rad,
            'eval_home_qvel_l2': eval_home_qvel_l2,
            'eval_home_hold_ticks': eval_home_hold_ticks,
            'eval_max_steps_per_target': eval_max_steps_per_target,
            'eval_shutdown_on_complete': eval_shutdown_on_complete,
            'eval_summary_csv_path': eval_summary_csv_path,
            'startup_home_enabled': startup_home_enabled,
            'startup_home_joints': startup_home_joints,
            'startup_home_tolerance_rad': startup_home_tolerance_rad,
            'startup_home_qvel_l2': startup_home_qvel_l2,
            'startup_home_hold_ticks': startup_home_hold_ticks,
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
            description='RF3 ONNX policy exported from ppo_final_rf3_balanced.zip.',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use false for /medrct_franka real robot topics unless /clock is available.',
        ),
        DeclareLaunchArgument(
            'frame_id',
            default_value='panda_link0',
            description='Base/task frame for targets and tf lookup.',
        ),
        DeclareLaunchArgument(
            'tcp_source',
            default_value='transform_topic',
            description='TCP source: transform_topic subscribes to tcp_transform_topic.',
        ),
        DeclareLaunchArgument(
            'tcp_frame_id',
            default_value='panda_hand_tcp',
            description='TCP frame used only when tcp_source=tf.',
        ),
        DeclareLaunchArgument(
            'tcp_pose_topic',
            default_value='ee_pose',
            description='PoseStamped topic used only when tcp_source=pose_topic.',
        ),
        DeclareLaunchArgument(
            'tcp_transform_topic',
            default_value='/medrct_franka/measured_tf',
            description='TransformStamped TCP measurement topic.',
        ),
        DeclareLaunchArgument(
            'joint_states_topic',
            default_value='/medrct_franka/measured_js',
            description='JointState measurement topic.',
        ),
        DeclareLaunchArgument(
            'command_topic',
            default_value='panda_joint_trajectory_controller/joint_trajectory',
            description='JointTrajectory command topic for the arm controller.',
        ),
        DeclareLaunchArgument(
            'action_smoothing_alpha',
            default_value='0.20',
            description='Low-pass command alpha. 0.20 matches RF3 training.',
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
            'action_notch_filter_enabled',
            default_value='false',
            description='Enable per-joint IIR notch filtering. Keep disabled for baseline RF3 validation.',
        ),
        DeclareLaunchArgument(
            'action_notch_filter_frequencies_hz',
            default_value='0,0,0,0,0,0,0',
            description='Primary per-joint notch frequencies in Hz. Use 0 to disable a joint.',
        ),
        DeclareLaunchArgument(
            'action_notch_filter_q',
            default_value='8.0',
            description='Primary notch Q. Higher values are narrower.',
        ),
        DeclareLaunchArgument(
            'action_notch_filter_secondary_frequencies_hz',
            default_value='0,0,0,0,0,0,0',
            description='Optional secondary per-joint notch frequencies in Hz.',
        ),
        DeclareLaunchArgument(
            'action_notch_filter_secondary_q',
            default_value='8.0',
            description='Secondary notch Q. Higher values are narrower.',
        ),
        DeclareLaunchArgument(
            'action_notch_filter_stage',
            default_value='final_command',
            description='Where to apply notch filtering if enabled: target_action, final_command, or both.',
        ),
        DeclareLaunchArgument(
            'hold_enabled',
            default_value='true',
            description='Enable near-goal setpoint hold. Enabled by default for RF3 because the raw policy limit-cycles near the target in Gazebo.',
        ),
        DeclareLaunchArgument(
            'hold_enter_tcp_m',
            default_value='0.07',
            description='Enter hold below this TCP distance, subject to qvel/tick gates.',
        ),
        DeclareLaunchArgument(
            'hold_exit_tcp_m',
            default_value='0.10',
            description='Exit hold above this TCP distance.',
        ),
        DeclareLaunchArgument(
            'hold_enter_qvel_l2',
            default_value='2.0',
            description='Enter hold only below this joint-velocity L2.',
        ),
        DeclareLaunchArgument(
            'hold_enter_ticks',
            default_value='5',
            description='Consecutive qualifying ticks before latching near-goal hold.',
        ),
        DeclareLaunchArgument(
            'trajectory_duration_s',
            default_value='0.015',
            description='Single-point JointTrajectory duration, approximately 0.9 / 60 Hz.',
        ),
        DeclareLaunchArgument(
            'eval_enabled',
            default_value='false',
            description='Enable RF3 multi-target eval sequencing.',
        ),
        DeclareLaunchArgument(
            'eval_target_count',
            default_value='20',
            description='Number of RF3 eval target attempts to run. 0 means unlimited.',
        ),
        DeclareLaunchArgument(
            'eval_reset_home_between_targets',
            default_value='true',
            description='Return to the RF3 home pose before sampling each next eval target.',
        ),
        DeclareLaunchArgument(
            'eval_home_joints',
            default_value='0,-0.943,0,-2.514,0,1.611,0',
            description='RF3 eval home pose as comma-separated joint radians.',
        ),
        DeclareLaunchArgument(
            'eval_home_tolerance_rad',
            default_value='0.03',
            description='Max absolute joint error allowed before eval home reset is considered reached.',
        ),
        DeclareLaunchArgument(
            'eval_home_qvel_l2',
            default_value='0.15',
            description='Joint velocity L2 threshold for accepting eval home reset.',
        ),
        DeclareLaunchArgument(
            'eval_home_hold_ticks',
            default_value='12',
            description='Consecutive home-settled ticks required before sampling the next RF3 eval target.',
        ),
        DeclareLaunchArgument(
            'eval_max_steps_per_target',
            default_value='0',
            description='Maximum control ticks per RF3 eval target attempt. 0 disables this timeout.',
        ),
        DeclareLaunchArgument(
            'eval_shutdown_on_complete',
            default_value='false',
            description='If true, shut down the policy node after eval_target_count attempts.',
        ),
        DeclareLaunchArgument(
            'eval_summary_csv_path',
            default_value='',
            description='Optional one-row-per-target RF3 eval summary CSV path.',
        ),
        DeclareLaunchArgument(
            'startup_home_enabled',
            default_value='true',
            description='Command the RF3 home pose before enabling policy inference.',
        ),
        DeclareLaunchArgument(
            'startup_home_joints',
            default_value='0,-0.943,0,-2.514,0,1.611,0',
            description='RF3 startup home pose as comma-separated joint radians.',
        ),
        DeclareLaunchArgument(
            'startup_home_tolerance_rad',
            default_value='0.03',
            description='Max absolute joint error allowed before RF3 startup home is accepted.',
        ),
        DeclareLaunchArgument(
            'startup_home_qvel_l2',
            default_value='0.15',
            description='Joint velocity L2 threshold for accepting RF3 startup home.',
        ),
        DeclareLaunchArgument(
            'startup_home_hold_ticks',
            default_value='12',
            description='Consecutive settled startup-home ticks required before RF3 policy inference starts.',
        ),
        DeclareLaunchArgument(
            'target_location_m',
            default_value='0.6,0,0.5',
            description='RF3 target center in the arm base frame (meters).',
        ),
        DeclareLaunchArgument(
            'target_random_offset_max_m',
            default_value='0.06,0.06,0.06',
            description='RF3 target offset magnitude. X/Y are symmetric; Z is positive-only.',
        ),
        DeclareLaunchArgument(
            'random_seed',
            default_value='0',
            description='Target sampling seed. Use fixed nonzero seeds for comparable Gazebo runs.',
        ),
        DeclareLaunchArgument(
            'initial_joint1',
            default_value='0.0',
            description='Initial Gazebo Panda joint1 position in radians. RF3 default is the constrained home pose.',
        ),
        DeclareLaunchArgument(
            'initial_joint2',
            default_value='-0.943',
            description='Initial Gazebo Panda joint2 position in radians. RF3 constrained home pose.',
        ),
        DeclareLaunchArgument(
            'initial_joint3',
            default_value='0.0',
            description='Initial Gazebo Panda joint3 position in radians. RF3 constrained home pose.',
        ),
        DeclareLaunchArgument(
            'initial_joint4',
            default_value='-2.514',
            description='Initial Gazebo Panda joint4 position in radians. RF3 constrained home pose.',
        ),
        DeclareLaunchArgument(
            'initial_joint5',
            default_value='0.0',
            description='Initial Gazebo Panda joint5 position in radians. RF3 constrained home pose.',
        ),
        DeclareLaunchArgument(
            'initial_joint6',
            default_value='1.611',
            description='Initial Gazebo Panda joint6 position in radians. RF3 constrained home pose.',
        ),
        DeclareLaunchArgument(
            'initial_joint7',
            default_value='0.0',
            description='Initial Gazebo Panda joint7 position in radians. RF3 constrained home pose.',
        ),
        DeclareLaunchArgument(
            'telemetry_enabled',
            default_value='false',
            description='Enable CSV tuning telemetry from panda_reach_reduce_shake_ros2.',
        ),
        DeclareLaunchArgument(
            'telemetry_csv_path',
            default_value='',
            description='Telemetry CSV path. Supplying a path enables telemetry in the node.',
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
        rf3_policy,
    ])
