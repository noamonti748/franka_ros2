# Copyright (c) 2026 Franka Robotics GmbH
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import xml.dom.minidom

import xacro

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.actions import RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def load_controller(context: LaunchContext, controller_name):
    controller_name_str = context.perform_substitution(controller_name)
    return [Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster',
            controller_name_str,
            '--controller-manager-timeout', '30',
        ],
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('franka_gazebo_bringup'),
                'config',
                'franka_gazebo_controllers.yaml',
            ]),
            PathJoinSubstitution([
                FindPackageShare('franka_gazebo_bringup'),
                'config',
                'franka_gazebo_panda_controllers.yaml',
            ]),
        ],
        output='screen',
    )]


def get_robot_description(context: LaunchContext, load_gripper):
    load_gripper_str = context.perform_substitution(load_gripper)

    panda_xacro_file = os.path.join(
        get_package_share_directory('franka_emika_panda'),
        'panda.urdf.xacro'
    )

    robot_description_config = xacro.process_file(
        panda_xacro_file,
        mappings={
            'arm_id': 'panda',
            'hand': load_gripper_str,
            'gazebo': 'true',
            'include_ros2_control': 'true',
            'gazebo_effort': 'true',
        }
    )

    if not isinstance(robot_description_config, xml.dom.minidom.Document):
        raise RuntimeError(
            f'The given xacro file {panda_xacro_file} is not a valid xml format.')

    robot_description = {'robot_description': robot_description_config.toxml()}

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='both',
        parameters=[robot_description],
    )

    return [robot_state_publisher]


def generate_launch_description():
    load_gripper_name = 'load_gripper'
    namespace_name = 'namespace'
    controller_name = 'controller'
    rviz_name = 'rviz'
    gz_args_name = 'gz_args'

    load_gripper = LaunchConfiguration(load_gripper_name)
    namespace = LaunchConfiguration(namespace_name)
    controller = LaunchConfiguration(controller_name)
    rviz = LaunchConfiguration(rviz_name)
    gz_args = LaunchConfiguration(gz_args_name)

    load_gripper_launch_argument = DeclareLaunchArgument(
        load_gripper_name,
        default_value='false',
        description='true/false for activating the gripper')
    namespace_launch_argument = DeclareLaunchArgument(
        namespace_name,
        default_value='',
        description='Namespace for the robot. If not set, the robot will be launched in the root namespace.')
    controller_launch_argument = DeclareLaunchArgument(
        controller_name,
        default_value='gravity_compensation_example_controller',
        description='The controller name to be used. You can choose one from the franka_example_controllers.')
    gz_args_launch_argument = DeclareLaunchArgument(
        gz_args_name,
        default_value='-r empty.sdf',
        description='Extra args to be forwarded to Gazebo')
    rviz_launch_argument = DeclareLaunchArgument(
        rviz_name,
        default_value='true',
        description='true/false for visualizing the robot in rviz')

    robot_state_publisher = OpaqueFunction(
        function=get_robot_description,
        args=[load_gripper])

    os.environ['GZ_SIM_RESOURCE_PATH'] = os.pathsep.join([
        os.path.dirname(get_package_share_directory('franka_emika_panda')),
    ])

    gazebo_empty_world = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py',
            ])
        ]),
        launch_arguments={'gz_args': gz_args}.items(),
    )

    spawn = Node(
        package='ros_gz_sim',
        executable='create',
        namespace=namespace,
        arguments=['-topic', '/robot_description'],
        output='screen',
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
        output='screen',
    )

    rviz_node = Node(package='rviz2',
                     executable='rviz2',
                     name='rviz2',
                     namespace=namespace,
                     arguments=[
                         '--display-config',
                         PathJoinSubstitution([
                             FindPackageShare('franka_gazebo_bringup'),
                             'rviz',
                             'panda_track_reach.rviz',
                         ]),
                         '-f',
                         'world',
                     ],
                     condition=IfCondition(rviz))

    launch_controller = OpaqueFunction(
        function=load_controller,
        args=[controller]
    )

    return LaunchDescription([
        load_gripper_launch_argument,
        namespace_launch_argument,
        controller_launch_argument,
        gz_args_launch_argument,
        rviz_launch_argument,
        gazebo_empty_world,
        robot_state_publisher,
        rviz_node,
        spawn,
        bridge,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawn,
                on_exit=[launch_controller],
            )
        ),
    ])
