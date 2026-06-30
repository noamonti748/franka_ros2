# Copyright (c) 2026 Franka Robotics GmbH
#
# Licensed under the Apache License, Version 2.0

# Runs the ONNX reach/track policy (ppo_track_franka.onnx) on an FR3 in Gazebo,
# reusing the same supervisor node as the Panda setup (panda_track_reach_ros2)
# with robot_type:=fr3. The policy was trained on the classic Panda; since the
# FR3 shares the Panda kinematics this is a sim validation before hardware.

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    load_gripper = LaunchConfiguration('load_gripper')
    # Named 'use_rviz' (not 'rviz') on purpose: IncludeLaunchDescription leaks the
    # 'rviz' launch_argument we pass to the inner launch into this scope, which
    # would otherwise disable our own RViz node below.
    use_rviz = LaunchConfiguration('use_rviz')
    gz_args = LaunchConfiguration('gz_args')
    model = LaunchConfiguration('model')
    tcp_observation_mode = LaunchConfiguration('tcp_observation_mode')
    tcp_observation_scale = LaunchConfiguration('tcp_observation_scale')
    tcp_observation_signs = LaunchConfiguration('tcp_observation_signs')
    tcp_observation_unit_scale = LaunchConfiguration('tcp_observation_unit_scale')
    joint_observation_mode = LaunchConfiguration('joint_observation_mode')
    action_scale = LaunchConfiguration('action_scale')
    action_output_mode = LaunchConfiguration('action_output_mode')
    action_delta_scale = LaunchConfiguration('action_delta_scale')
    control_hz = LaunchConfiguration('control_hz')
    trajectory_duration_s = LaunchConfiguration('trajectory_duration_s')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('franka_gazebo_bringup'),
                'launch',
                'gazebo_franka_arm_example_controller.launch.py',
            ])
        ]),
        launch_arguments={
            'robot_type': 'fr3',
            'load_gripper': load_gripper,
            # Disable the inner launch's RViz; we run our own (below) with the
            # reach-target marker config. (This 'rviz' value leaks to our scope,
            # which is why our toggle is named 'use_rviz'.)
            'rviz': 'false',
            'gz_args': gz_args,
            'controller': 'fr3_joint_trajectory_controller',
        }.items(),
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=[
            '--display-config',
            PathJoinSubstitution([
                FindPackageShare('franka_emika_panda'),
                'fr3_track_reach.rviz',
            ]),
        ],
        parameters=[{'use_sim_time': True}],
        output='screen',
        condition=IfCondition(use_rviz),
    )

    policy_node = Node(
        package='franka_emika_panda',
        executable='panda_track_reach_ros2',
        name='panda_track_reach',
        output='screen',
        arguments=['--ros', '--policy'],
        parameters=[{
            'use_sim_time': True,
            'robot_type': 'fr3',
            'onnx_model_path': model,
            'tcp_source': 'tf',
            'tcp_frame_id': 'fr3_hand_tcp',
            'frame_id': 'fr3_link0',
            'command_topic': 'fr3_joint_trajectory_controller/joint_trajectory',
            'tcp_observation_mode': tcp_observation_mode,
            'tcp_observation_scale': ParameterValue(tcp_observation_scale, value_type=float),
            'tcp_observation_signs': tcp_observation_signs,
            'tcp_observation_unit_scale': ParameterValue(
                tcp_observation_unit_scale,
                value_type=float,
            ),
            'joint_observation_mode': joint_observation_mode,
            'action_scale': ParameterValue(action_scale, value_type=float),
            'action_output_mode': action_output_mode,
            'action_delta_scale': ParameterValue(action_delta_scale, value_type=float),
            'control_hz': ParameterValue(control_hz, value_type=float),
            'trajectory_duration_s': ParameterValue(trajectory_duration_s, value_type=float),
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument('load_gripper', default_value='true'),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('gz_args', default_value='-r empty.sdf'),
        DeclareLaunchArgument('model', default_value=[
            PathJoinSubstitution([
                FindPackageShare('franka_emika_panda'),
                'ppo_track_franka.onnx',
            ])
        ]),
        DeclareLaunchArgument(
            'tcp_observation_mode',
            default_value='tcp_error_normalized',
            description='Policy tcp_pos input: tcp_error_normalized, target_error_normalized, tcp_minus_target, target_minus_tcp, target_delta, tcp, or target.',
        ),
        DeclareLaunchArgument(
            'tcp_observation_scale',
            default_value='1.0',
            description='ObsTcpScale used for tcp_error_normalized: (tcp_position - target_position) / scale.',
        ),
        DeclareLaunchArgument(
            'tcp_observation_signs',
            default_value='1,1,1',
            description='Comma-separated XYZ signs applied after tcp_pos preprocessing, e.g. 1,-1,1.',
        ),
        DeclareLaunchArgument(
            'tcp_observation_unit_scale',
            default_value='100.0',
            description='Unit multiplier before ObsTcpScale; use 100.0 if the policy saw centimeters.',
        ),
        DeclareLaunchArgument(
            'joint_observation_mode',
            default_value='position_velocity_command',
            description='Joint input packing for each jointN ONNX input.',
        ),
        DeclareLaunchArgument(
            'action_scale',
            default_value='1.0',
            description='Multiplier applied to the raw ONNX actuator outputs before filtering.',
        ),
        DeclareLaunchArgument(
            'action_output_mode',
            default_value='normalized_delta_position',
            description='Interpret ONNX outputs as absolute, normalized_absolute, delta_position, normalized_delta_position, delta_command, or normalized_delta_command.',
        ),
        DeclareLaunchArgument(
            'action_delta_scale',
            default_value='0.05',
            description='Joint-radian step size for delta action output modes.',
        ),
        DeclareLaunchArgument(
            'control_hz',
            default_value='60.0',
            description='Policy/control loop rate in Hz.',
        ),
        DeclareLaunchArgument(
            'trajectory_duration_s',
            default_value='0.05',
            description='Time-from-start for each published JointTrajectory point.',
        ),
        gazebo,
        rviz_node,
        policy_node,
    ])
