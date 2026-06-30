# Copyright (c) 2026 Franka Robotics GmbH
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

############################################################################
# Deploys the ONNX reach/track policy (ppo_track_franka.onnx) on a *real* FR3.
#
# It brings up the standard FR3 hardware stack via franka.launch.py, spawns a
# joint_trajectory_controller (fr3_joint_trajectory_controller) on the position
# command interface, and runs the same supervisor node used in Gazebo
# (panda_track_reach_ros2) with robot_type:=fr3.
#
# The policy was trained on the classic Panda. The FR3 shares the Panda
# kinematics, so it transfers, but joint limits differ (notably joint6 >= 0.5445
# rad on FR3). The supervisor clips to FR3 limits, but you MUST still validate
# carefully:
#   - Keep the workspace clear and an e-stop within reach.
#   - First runs: lower control_hz / action_delta_scale and watch joint6.
#   - Disable the policy (policy_enabled:=false) to just bring up the stack/JTC.
#
# Example:
#   ros2 launch franka_bringup fr3_onnx_policy.launch.py robot_ip:=172.16.0.2
############################################################################

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _spawn_policy(context):
    policy_enabled = LaunchConfiguration('policy_enabled').perform(context).lower() == 'true'
    node_args = ['--ros']
    if policy_enabled:
        node_args.append('--policy')

    policy_node = Node(
        package='franka_emika_panda',
        executable='panda_track_reach_ros2',
        name='panda_track_reach',
        output='screen',
        arguments=node_args,
        parameters=[{
            'use_sim_time': False,
            'robot_type': 'fr3',
            'onnx_model_path': LaunchConfiguration('model'),
            'tcp_source': 'tf',
            'tcp_frame_id': 'fr3_hand_tcp',
            'frame_id': 'fr3_link0',
            'command_topic': 'fr3_joint_trajectory_controller/joint_trajectory',
            'tcp_observation_mode': LaunchConfiguration('tcp_observation_mode'),
            'tcp_observation_scale': ParameterValue(
                LaunchConfiguration('tcp_observation_scale'), value_type=float),
            'tcp_observation_signs': LaunchConfiguration('tcp_observation_signs'),
            'tcp_observation_unit_scale': ParameterValue(
                LaunchConfiguration('tcp_observation_unit_scale'), value_type=float),
            'joint_observation_mode': LaunchConfiguration('joint_observation_mode'),
            'action_scale': ParameterValue(
                LaunchConfiguration('action_scale'), value_type=float),
            'action_output_mode': LaunchConfiguration('action_output_mode'),
            'action_delta_scale': ParameterValue(
                LaunchConfiguration('action_delta_scale'), value_type=float),
            'control_hz': ParameterValue(
                LaunchConfiguration('control_hz'), value_type=float),
            'trajectory_duration_s': ParameterValue(
                LaunchConfiguration('trajectory_duration_s'), value_type=float),
        }],
    )
    return [policy_node]


def generate_launch_description():
    franka = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare('franka_bringup'), 'launch', 'franka.launch.py']
            )
        ),
        launch_arguments={
            'robot_type': 'fr3',
            'robot_ip': LaunchConfiguration('robot_ip'),
            'load_gripper': LaunchConfiguration('load_gripper'),
            'use_fake_hardware': LaunchConfiguration('use_fake_hardware'),
            'joint_state_rate': LaunchConfiguration('joint_state_rate'),
        }.items(),
    )

    jtc_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'fr3_joint_trajectory_controller',
            '--controller-manager-timeout', '30',
        ],
        parameters=[
            PathJoinSubstitution(
                [FindPackageShare('franka_bringup'), 'config', 'controllers.yaml']
            )
        ],
        output='screen',
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=[
            '--display-config',
            PathJoinSubstitution(
                [FindPackageShare('franka_emika_panda'), 'fr3_track_reach.rviz']
            ),
        ],
        output='screen',
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'robot_ip', default_value='172.16.0.2',
            description='Hostname or IP address of the FR3.'),
        DeclareLaunchArgument(
            'load_gripper', default_value='true',
            description='Load the Franka hand so the fr3_hand_tcp frame is published.'),
        DeclareLaunchArgument(
            'use_fake_hardware', default_value='false',
            description='Use mock hardware instead of the real FCI connection.'),
        DeclareLaunchArgument(
            'joint_state_rate', default_value='100',
            description='Joint state publish rate (Hz). Raise above 30 for smoother feedback.'),
        DeclareLaunchArgument(
            'rviz', default_value='false',
            description='Launch RViz with the franka_description config.'),
        DeclareLaunchArgument(
            'policy_enabled', default_value='true',
            description='If false, brings up the stack + JTC but does not command the robot.'),
        DeclareLaunchArgument('model', default_value=[
            PathJoinSubstitution(
                [FindPackageShare('franka_emika_panda'), 'ppo_track_franka.onnx']
            )
        ]),
        DeclareLaunchArgument(
            'tcp_observation_mode', default_value='tcp_error_normalized',
            description='Policy tcp_pos input mode.'),
        DeclareLaunchArgument(
            'tcp_observation_scale', default_value='1.0',
            description='ObsTcpScale: (tcp_position - target_position) / scale.'),
        DeclareLaunchArgument(
            'tcp_observation_signs', default_value='1,1,1',
            description='Comma-separated XYZ signs applied after tcp_pos preprocessing.'),
        DeclareLaunchArgument(
            'tcp_observation_unit_scale', default_value='100.0',
            description='Unit multiplier before ObsTcpScale; 100.0 if the policy saw centimeters.'),
        DeclareLaunchArgument(
            'joint_observation_mode', default_value='position_velocity_command',
            description='Joint input packing for each jointN ONNX input.'),
        DeclareLaunchArgument(
            'action_scale', default_value='1.0',
            description='Multiplier applied to the raw ONNX outputs before filtering.'),
        DeclareLaunchArgument(
            'action_output_mode', default_value='normalized_delta_position',
            description='How to interpret ONNX outputs.'),
        DeclareLaunchArgument(
            'action_delta_scale', default_value='0.05',
            description='Joint-radian step size for delta action output modes.'),
        DeclareLaunchArgument(
            'control_hz', default_value='60.0',
            description='Policy/control loop rate in Hz.'),
        DeclareLaunchArgument(
            'trajectory_duration_s', default_value='0.05',
            description='Time-from-start for each published JointTrajectory point.'),
        franka,
        jtc_spawner,
        rviz_node,
        OpaqueFunction(function=_spawn_policy),
    ])
