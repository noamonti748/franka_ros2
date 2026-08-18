# Copyright (c) 2026 Franka Robotics GmbH
#
# Licensed under the Apache License, Version 2.0

# Runs the ppo_final_reduce_shake "reach and hold" ONNX policy on a Panda in Gazebo.
# Unlike gazebo_panda_onnx_policy.launch.py (ppo_track_franka.onnx, multi-target
# "track" behaviour with target resampling on success), this checkpoint was trained
# to reach a single target and hold station there -- no resampling after success.
#
# Observation preprocessing must match training exactly:
#   - joint1..joint7: [qpos/pi, qvel/2.0], each dim clipped to +/-3
#     (joint_observation_mode=normalized_position_velocity, observation_clip=3.0).
#   - prev_action: previous 7-joint command vector, optionally divided by
#     prev_action_scale for compatible lower-vib policies.
#   - tcp_pos: (tcp_m - target_m) / 0.75, clipped to +/-3
#     (tcp_observation_mode=tcp_error_normalized, tcp_observation_scale=0.75,
#     tcp_observation_unit_scale=1.0, tcp_observation_signs=1,1,1).
#   - action: 7-value absolute joint position target vector in radians
#     (action_output_mode=absolute).
#
# Target defaults reproduce training: center (0.32, 0, 0.5) m with up to 0.06 m
# random offset per axis, sampled symmetrically on X/Y/Z (symmetric_z_offset=true;
# the legacy track checkpoint samples Z as U[0, +max] instead -- see
# panda_reach_reduce_shake_ros2.py's _sample_target_offset_reach_hold for details/
# assumptions). This node (panda_reach_reduce_shake_ros2) is a dedicated module,
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
    use_sim_time = LaunchConfiguration('use_sim_time')
    frame_id = LaunchConfiguration('frame_id')
    tcp_source = LaunchConfiguration('tcp_source')
    tcp_frame_id = LaunchConfiguration('tcp_frame_id')
    tcp_pose_topic = LaunchConfiguration('tcp_pose_topic')
    tcp_transform_topic = LaunchConfiguration('tcp_transform_topic')
    joint_states_topic = LaunchConfiguration('joint_states_topic')
    command_topic = LaunchConfiguration('command_topic')
    tcp_observation_mode = LaunchConfiguration('tcp_observation_mode')
    tcp_observation_scale = LaunchConfiguration('tcp_observation_scale')
    tcp_observation_signs = LaunchConfiguration('tcp_observation_signs')
    tcp_observation_unit_scale = LaunchConfiguration('tcp_observation_unit_scale')
    joint_observation_mode = LaunchConfiguration('joint_observation_mode')
    observation_clip = LaunchConfiguration('observation_clip')
    action_scale = LaunchConfiguration('action_scale')
    action_output_mode = LaunchConfiguration('action_output_mode')
    action_delta_scale = LaunchConfiguration('action_delta_scale')
    prev_action_scale = LaunchConfiguration('prev_action_scale')
    control_hz = LaunchConfiguration('control_hz')
    trajectory_duration_s = LaunchConfiguration('trajectory_duration_s')
    target_location_m = LaunchConfiguration('target_location_m')
    target_random_offset_max_m = LaunchConfiguration('target_random_offset_max_m')
    random_seed = LaunchConfiguration('random_seed')
    resample_target_on_success = LaunchConfiguration('resample_target_on_success')
    symmetric_z_offset = LaunchConfiguration('symmetric_z_offset')
    initial_joint1 = LaunchConfiguration('initial_joint1')
    initial_joint2 = LaunchConfiguration('initial_joint2')
    initial_joint3 = LaunchConfiguration('initial_joint3')
    initial_joint4 = LaunchConfiguration('initial_joint4')
    initial_joint5 = LaunchConfiguration('initial_joint5')
    initial_joint6 = LaunchConfiguration('initial_joint6')
    initial_joint7 = LaunchConfiguration('initial_joint7')
    smooth_actions = LaunchConfiguration('smooth_actions')
    limit_action_delta = LaunchConfiguration('limit_action_delta')
    action_smoothing_alpha = LaunchConfiguration('action_smoothing_alpha')
    action_rate_limit_hz = LaunchConfiguration('action_rate_limit_hz')
    action_velocity_limits_rad_s = LaunchConfiguration('action_velocity_limits_rad_s')
    use_measured_velocity_governor = LaunchConfiguration('use_measured_velocity_governor')
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
    telemetry_enabled = LaunchConfiguration('telemetry_enabled')
    telemetry_csv_path = LaunchConfiguration('telemetry_csv_path')
    telemetry_decimation = LaunchConfiguration('telemetry_decimation')
    telemetry_flush_every_n_rows = LaunchConfiguration('telemetry_flush_every_n_rows')

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
            'initial_joint1': initial_joint1,
            'initial_joint2': initial_joint2,
            'initial_joint3': initial_joint3,
            'initial_joint4': initial_joint4,
            'initial_joint5': initial_joint5,
            'initial_joint6': initial_joint6,
            'initial_joint7': initial_joint7,
        }.items(),
    )

    policy_node = Node(
        package='franka_emika_panda',
        executable='panda_reach_reduce_shake_ros2',
        name='panda_reach_reduce_shake',
        output='screen',
        arguments=['--ros', '--policy'],
        parameters=[{
            'use_sim_time': ParameterValue(use_sim_time, value_type=bool),
            'onnx_model_path': model,
            'tcp_source': tcp_source,
            'tcp_frame_id': tcp_frame_id,
            'tcp_pose_topic': tcp_pose_topic,
            'tcp_transform_topic': tcp_transform_topic,
            'joint_states_topic': joint_states_topic,
            'frame_id': frame_id,
            'command_topic': command_topic,
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
            'prev_action_scale': ParameterValue(prev_action_scale, value_type=float),
            'control_hz': ParameterValue(control_hz, value_type=float),
            'trajectory_duration_s': ParameterValue(trajectory_duration_s, value_type=float),
            'target_location_m': target_location_m,
            'target_random_offset_max_m': target_random_offset_max_m,
            'random_seed': ParameterValue(random_seed, value_type=int),
            'resample_target_on_success': ParameterValue(
                resample_target_on_success,
                value_type=bool,
            ),
            'symmetric_z_offset': ParameterValue(symmetric_z_offset, value_type=bool),
            'smooth_actions': ParameterValue(smooth_actions, value_type=bool),
            'limit_action_delta': ParameterValue(limit_action_delta, value_type=bool),
            'action_smoothing_alpha': ParameterValue(
                action_smoothing_alpha,
                value_type=float,
            ),
            'action_rate_limit_hz': ParameterValue(action_rate_limit_hz, value_type=float),
            'action_velocity_limits_rad_s': action_velocity_limits_rad_s,
            'use_measured_velocity_governor': ParameterValue(
                use_measured_velocity_governor,
                value_type=bool,
            ),
            'measured_velocity_governor_start_ratio': ParameterValue(
                measured_velocity_governor_start_ratio,
                value_type=float,
            ),
            'action_notch_filter_enabled': ParameterValue(
                action_notch_filter_enabled,
                value_type=bool,
            ),
            'action_notch_filter_frequencies_hz': action_notch_filter_frequencies_hz,
            'action_notch_filter_q': ParameterValue(
                action_notch_filter_q,
                value_type=float,
            ),
            'action_notch_filter_secondary_frequencies_hz': (
                action_notch_filter_secondary_frequencies_hz
            ),
            'action_notch_filter_secondary_q': ParameterValue(
                action_notch_filter_secondary_q,
                value_type=float,
            ),
            'action_notch_filter_stage': action_notch_filter_stage,
            'hold_enabled': ParameterValue(hold_enabled, value_type=bool),
            'hold_enter_tcp_m': ParameterValue(hold_enter_tcp_m, value_type=float),
            'hold_exit_tcp_m': ParameterValue(hold_exit_tcp_m, value_type=float),
            'hold_enter_qvel_l2': ParameterValue(hold_enter_qvel_l2, value_type=float),
            'hold_enter_ticks': ParameterValue(hold_enter_ticks, value_type=int),
            'eval_enabled': ParameterValue(eval_enabled, value_type=bool),
            'eval_target_count': ParameterValue(eval_target_count, value_type=int),
            'eval_reset_home_between_targets': ParameterValue(
                eval_reset_home_between_targets,
                value_type=bool,
            ),
            'eval_home_joints': eval_home_joints,
            'eval_home_tolerance_rad': ParameterValue(
                eval_home_tolerance_rad,
                value_type=float,
            ),
            'eval_home_qvel_l2': ParameterValue(eval_home_qvel_l2, value_type=float),
            'eval_home_hold_ticks': ParameterValue(
                eval_home_hold_ticks,
                value_type=int,
            ),
            'eval_max_steps_per_target': ParameterValue(
                eval_max_steps_per_target,
                value_type=int,
            ),
            'eval_shutdown_on_complete': ParameterValue(
                eval_shutdown_on_complete,
                value_type=bool,
            ),
            'eval_summary_csv_path': eval_summary_csv_path,
            'startup_home_enabled': ParameterValue(
                startup_home_enabled,
                value_type=bool,
            ),
            'startup_home_joints': startup_home_joints,
            'startup_home_tolerance_rad': ParameterValue(
                startup_home_tolerance_rad,
                value_type=float,
            ),
            'startup_home_qvel_l2': ParameterValue(
                startup_home_qvel_l2,
                value_type=float,
            ),
            'startup_home_hold_ticks': ParameterValue(
                startup_home_hold_ticks,
                value_type=int,
            ),
            'telemetry_enabled': ParameterValue(telemetry_enabled, value_type=bool),
            'telemetry_csv_path': telemetry_csv_path,
            'telemetry_decimation': ParameterValue(telemetry_decimation, value_type=int),
            'telemetry_flush_every_n_rows': ParameterValue(
                telemetry_flush_every_n_rows,
                value_type=int,
            ),
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument('load_gripper', default_value='true'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('gz_args', default_value='-r empty.sdf'),
        DeclareLaunchArgument('model', default_value=[
            PathJoinSubstitution([
                FindPackageShare('franka_emika_panda'),
                'ppo_final_reduce_shake.onnx',
            ])
        ]),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation time for the policy node. Set false for real robot topics unless /clock is available.',
        ),
        DeclareLaunchArgument(
            'frame_id',
            default_value='panda_link0',
            description='Base/task frame for targets and tf lookup.',
        ),
        DeclareLaunchArgument(
            'tcp_source',
            default_value='tf',
            description='TCP source: tf, pose_topic, or transform_topic.',
        ),
        DeclareLaunchArgument(
            'tcp_frame_id',
            default_value='panda_hand_tcp',
            description='TCP frame used when tcp_source=tf.',
        ),
        DeclareLaunchArgument(
            'tcp_pose_topic',
            default_value='ee_pose',
            description='PoseStamped topic used when tcp_source=pose_topic.',
        ),
        DeclareLaunchArgument(
            'tcp_transform_topic',
            default_value='measured_tf',
            description='TransformStamped topic used when tcp_source=transform_topic.',
        ),
        DeclareLaunchArgument(
            'joint_states_topic',
            default_value='joint_states',
            description='JointState topic used for measured joint positions and velocities.',
        ),
        DeclareLaunchArgument(
            'command_topic',
            default_value='panda_joint_trajectory_controller/joint_trajectory',
            description='JointTrajectory command topic for the arm controller.',
        ),
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
            default_value='normalized_position_velocity',
            description='Joint input packing for each jointN ONNX input: [qpos/pi, qvel/2.0] for this checkpoint.',
        ),
        DeclareLaunchArgument(
            'observation_clip',
            default_value='3.0',
            description='Clip every joint/tcp observation dim to +/-this value after scaling (matches the trained Box(-3, 3) observation space). 0 disables clipping.',
        ),
        DeclareLaunchArgument(
            'action_scale',
            default_value='1.0',
            description='Multiplier applied to the raw ONNX action vector before filtering.',
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
            'prev_action_scale',
            default_value='1.0',
            description='Divide previous-command prev_action input by this value before ONNX inference. 1.0 preserves ppo_final_reduce_shake behaviour; use pi for ppo_final_lower_vib.',
        ),
        DeclareLaunchArgument(
            'control_hz',
            default_value='60.0',
            description='Policy/control loop rate in Hz. 60.0 matches the training control rate '
                        'for this checkpoint family: task/track_reach_session.py in the '
                        'MuJoCo-SB3 training repo defines action_rate_limit_hz=60.0, and every '
                        'related env/rollout/eval script defaults control_hz to 60.0 too. The '
                        'exact training script for ppo_final_reduce_shake itself was not found '
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
            'random_seed',
            default_value='0',
            description='Target sampling seed. 0 leaves target sampling nondeterministic.',
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
            'initial_joint1',
            default_value='0.0',
            description='Initial Gazebo Panda joint1 position in radians.',
        ),
        DeclareLaunchArgument(
            'initial_joint2',
            default_value='0.0',
            description='Initial Gazebo Panda joint2 position in radians.',
        ),
        DeclareLaunchArgument(
            'initial_joint3',
            default_value='0.0',
            description='Initial Gazebo Panda joint3 position in radians.',
        ),
        DeclareLaunchArgument(
            'initial_joint4',
            default_value='-1.5707963267948966',
            description='Initial Gazebo Panda joint4 position in radians.',
        ),
        DeclareLaunchArgument(
            'initial_joint5',
            default_value='0.0',
            description='Initial Gazebo Panda joint5 position in radians.',
        ),
        DeclareLaunchArgument(
            'initial_joint6',
            default_value='1.5707963267948966',
            description='Initial Gazebo Panda joint6 position in radians.',
        ),
        DeclareLaunchArgument(
            'initial_joint7',
            default_value='-0.7853981633974483',
            description='Initial Gazebo Panda joint7 position in radians.',
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
        DeclareLaunchArgument(
            'action_smoothing_alpha',
            default_value='0.05',
            description='Low-pass update alpha when smooth_actions=true: command = previous + alpha * (target - previous). Lower is smoother but adds lag.',
        ),
        DeclareLaunchArgument(
            'action_rate_limit_hz',
            default_value='60.0',
            description='Rate-limit denominator in Hz when limit_action_delta=true. Keep this aligned with the policy/control tick unless deliberately testing a different command slew.',
        ),
        DeclareLaunchArgument(
            'action_velocity_limits_rad_s',
            default_value='1.2,1.45,1.45,1.45,1.2,1.55,1.55',
            description='Comma-separated per-joint command velocity limits used by the action delta limiter.',
        ),
        DeclareLaunchArgument(
            'use_measured_velocity_governor',
            default_value='true',
            description='If true, stop moving a command farther in the same direction when measured joint velocity is already near the limiter.',
        ),
        DeclareLaunchArgument(
            'measured_velocity_governor_start_ratio',
            default_value='0.70',
            description='Measured-velocity governor threshold as a fraction of each action_velocity_limits_rad_s value.',
        ),
        DeclareLaunchArgument(
            'action_notch_filter_enabled',
            default_value='false',
            description='Enable per-joint IIR notch filtering on target actions before smoothing/rate limiting.',
        ),
        DeclareLaunchArgument(
            'action_notch_filter_frequencies_hz',
            default_value='0,0,0,0,0,0,0',
            description='Primary per-joint notch frequencies in Hz. Use 0 to disable a joint.',
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
            description='Enable the near-goal command hold (deadband with hysteresis) that latches a fixed setpoint once the TCP is near the goal and nearly still, to kill the hold-point limit cycle.',
        ),
        DeclareLaunchArgument(
            'hold_enter_tcp_m',
            default_value='0.02',
            description='Enter hold when measured TCP distance is below this many meters (and qvel L2 is below hold_enter_qvel_l2) for hold_enter_ticks consecutive ticks.',
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
            'eval_enabled',
            default_value='false',
            description='Enable multi-target evaluation sequencing instead of normal single-target reach-and-hold.',
        ),
        DeclareLaunchArgument(
            'eval_target_count',
            default_value='20',
            description='Number of target attempts to run in eval mode. 0 means unlimited.',
        ),
        DeclareLaunchArgument(
            'eval_reset_home_between_targets',
            default_value='true',
            description='When eval mode completes a target attempt, command the robot back to eval_home_joints before sampling the next target.',
        ),
        DeclareLaunchArgument(
            'eval_home_joints',
            default_value='0,0,0,-1.5707963267948966,0,1.5707963267948966,-0.7853981633974483',
            description='Comma-separated 7-joint home pose used for eval resets.',
        ),
        DeclareLaunchArgument(
            'eval_home_tolerance_rad',
            default_value='0.03',
            description='Eval home reset is considered reached when max absolute joint error is below this many radians.',
        ),
        DeclareLaunchArgument(
            'eval_home_qvel_l2',
            default_value='0.15',
            description='Eval home reset is considered settled when joint velocity L2 is below this value.',
        ),
        DeclareLaunchArgument(
            'eval_home_hold_ticks',
            default_value='12',
            description='Consecutive home-settled ticks required before the next eval target is sampled.',
        ),
        DeclareLaunchArgument(
            'eval_max_steps_per_target',
            default_value='0',
            description='Maximum control ticks per eval target attempt. 0 disables this timeout.',
        ),
        DeclareLaunchArgument(
            'eval_shutdown_on_complete',
            default_value='false',
            description='If true, shut down the policy node when the requested eval target count is complete.',
        ),
        DeclareLaunchArgument(
            'eval_summary_csv_path',
            default_value='',
            description='Optional one-row-per-target eval summary CSV path.',
        ),
        DeclareLaunchArgument(
            'startup_home_enabled',
            default_value='false',
            description='If true, command startup_home_joints first and hold there before enabling policy inference.',
        ),
        DeclareLaunchArgument(
            'startup_home_joints',
            default_value='0,0,0,-1.5707963267948966,0,1.5707963267948966,-0.7853981633974483',
            description='Comma-separated 7-joint startup home pose. Used to seed prev_action before the first policy tick.',
        ),
        DeclareLaunchArgument(
            'startup_home_tolerance_rad',
            default_value='0.03',
            description='Max absolute joint error allowed before startup home is accepted.',
        ),
        DeclareLaunchArgument(
            'startup_home_qvel_l2',
            default_value='0.15',
            description='Joint velocity L2 threshold for accepting startup home.',
        ),
        DeclareLaunchArgument(
            'startup_home_hold_ticks',
            default_value='12',
            description='Consecutive settled startup-home ticks required before policy inference starts.',
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
        gazebo,
        policy_node,
    ])
