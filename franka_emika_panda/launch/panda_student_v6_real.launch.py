# Copyright (c) 2026 Franka Robotics GmbH
#
# Licensed under the Apache License, Version 2.0
#
# HARDWARE WARNING:
# This launch file starts only the student-v6 supervisor. It deliberately does
# not include franka_bringup, spawn a controller, or connect to robot hardware.
# Start and validate the Humble Franka stack and its safety configuration
# separately. With the defaults below the node is monitor-only and publishes no
# arm or gripper command.

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


_ENABLE_CONFIRMATION = "I_UNDERSTAND_THIS_WILL_COMMAND_A_REAL_PANDA"


def _is_true(value):
    return value.strip().lower() in ("1", "true", "yes", "on")


def _start_student_v6(context):
    """Validate the hardware interlock, then construct the policy node."""
    policy_enabled = _is_true(LaunchConfiguration("policy_enabled").perform(context))
    confirmation = LaunchConfiguration("enable_confirmation").perform(context)

    if policy_enabled and confirmation != _ENABLE_CONFIRMATION:
        raise RuntimeError(
            "Refusing to enable student-v6 real-hardware commands. Re-run with "
            "policy_enabled:=true and enable_confirmation:="
            f"{_ENABLE_CONFIRMATION} only after validating the workcell, "
            "controller, camera calibration, TF, gripper, reflex limits, and E-stop."
        )

    status = (
        "REAL-HARDWARE COMMANDING REQUESTED; the node must still pass every "
        "configured freshness/readiness gate before publishing."
        if policy_enabled
        else "Student-v6 is monitor-only (policy_enabled=false); no commands are published."
    )

    parameters = [LaunchConfiguration("config")]
    profile_config = LaunchConfiguration("profile_config").perform(context).strip()
    if profile_config:
        parameters.append(profile_config)
    parameters.append(
        {
            "use_sim_time": False,
            "policy_enabled": ParameterValue(
                LaunchConfiguration("policy_enabled"), value_type=bool
            ),
            "startup_home_enabled": ParameterValue(
                LaunchConfiguration("startup_home_enabled"), value_type=bool
            ),
            "onnx_model_path": LaunchConfiguration("onnx_model_path"),
            "robot_id": LaunchConfiguration("arm_id"),
            "base_frame": LaunchConfiguration("base_frame"),
            "tcp_frame": LaunchConfiguration("tcp_frame"),
            "camera_frame": LaunchConfiguration("camera_frame"),
            "require_camera_tf": ParameterValue(
                LaunchConfiguration("require_camera_tf"), value_type=bool
            ),
            "robot_description": LaunchConfiguration("robot_description"),
            "robot_description_topic": LaunchConfiguration(
                "robot_description_topic"
            ),
            "joint_states_topic": LaunchConfiguration("joint_states_topic"),
            "image_topic": LaunchConfiguration("image_topic"),
            "camera_info_topic": LaunchConfiguration("camera_info_topic"),
            "command_topic": LaunchConfiguration("arm_command_topic"),
            "gripper_state_topic": LaunchConfiguration("gripper_state_topic"),
            "grasp_action_name": LaunchConfiguration("grasp_action_name"),
            "move_action_name": LaunchConfiguration("move_action_name"),
            "telemetry_topic": LaunchConfiguration("telemetry_topic"),
            "seated_offset_m": ParameterValue(
                LaunchConfiguration("seated_offset_m"), value_type=float
            ),
        }
    )
    node = Node(
        package="franka_emika_panda",
        executable="panda_student_v6_ros2",
        name="panda_student_v6_real",
        output="screen",
        emulate_tty=True,
        parameters=parameters,
    )
    return [LogInfo(msg=status), node]


def generate_launch_description():
    package_share = FindPackageShare("franka_emika_panda")
    model_dir = PathJoinSubstitution(
        [package_share, "models", "student_v6"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config",
                default_value=PathJoinSubstitution(
                    [package_share, "config", "student_v6_real.yaml"]
                ),
                description="Installed real-hardware safety/calibration parameter file.",
            ),
            DeclareLaunchArgument(
                "profile_config",
                default_value="",
                description=(
                    "Optional parameter overlay loaded after the base safety "
                    "configuration, e.g. student_autonomous_v1.yaml."
                ),
            ),
            DeclareLaunchArgument(
                "onnx_model_path",
                default_value=PathJoinSubstitution(
                    [model_dir, "student_v6_final.onnx"]
                ),
                description="Promoted student-v6 ONNX model.",
            ),
            DeclareLaunchArgument(
                "policy_enabled",
                default_value="false",
                description=(
                    "Permit arm/gripper publication after all readiness gates pass. "
                    "False is monitor-only."
                ),
            ),
            DeclareLaunchArgument(
                "enable_confirmation",
                default_value="NOT_CONFIRMED",
                description=(
                    "When policy_enabled=true this must exactly equal "
                    f"'{_ENABLE_CONFIRMATION}'."
                ),
            ),
            DeclareLaunchArgument(
                "startup_home_enabled",
                default_value="true",
                description=(
                    "Require the measured arm to remain at the trained home pose "
                    "before policy commands; this node does not move the arm home."
                ),
            ),
            DeclareLaunchArgument("arm_id", default_value="panda"),
            DeclareLaunchArgument("base_frame", default_value="panda_link0"),
            DeclareLaunchArgument("tcp_frame", default_value="panda_hand_tcp"),
            DeclareLaunchArgument(
                "camera_frame",
                default_value="policy_camera_optical_frame",
                description="Calibrated RGB camera optical frame; must exist in TF.",
            ),
            DeclareLaunchArgument(
                "require_camera_tf",
                default_value="true",
                description="Require a valid base-to-camera transform before commands.",
            ),
            DeclareLaunchArgument(
                "robot_description",
                default_value="",
                description=(
                    "Optional URDF XML string for KDL. Leave empty to consume the "
                    "transient-local robot_description_topic."
                ),
            ),
            DeclareLaunchArgument(
                "robot_description_topic",
                default_value="/robot_description",
                description=(
                    "Existing transient-local std_msgs/String robot description used "
                    "to initialize the Panda KDL chain."
                ),
            ),
            DeclareLaunchArgument(
                "joint_states_topic", default_value="/joint_states"
            ),
            DeclareLaunchArgument(
                "image_topic",
                default_value="/policy_camera/image_raw",
                description="RGB8 64x64 image topic matching the trained camera contract.",
            ),
            DeclareLaunchArgument(
                "camera_info_topic",
                default_value="/policy_camera/camera_info",
                description=(
                    "CameraInfo topic used by oak_4_by_3 preprocessing. Override "
                    "this and image_topic for an existing site camera namespace."
                ),
            ),
            DeclareLaunchArgument(
                "arm_command_topic",
                default_value="/panda_joint_trajectory_controller/joint_trajectory",
                description=(
                    "Command topic of an already loaded, explicitly activated "
                    "seven-joint position JointTrajectoryController."
                ),
            ),
            DeclareLaunchArgument(
                "gripper_state_topic",
                default_value="/franka_gripper/joint_states",
            ),
            DeclareLaunchArgument(
                "grasp_action_name",
                default_value="/franka_gripper/grasp",
                description=(
                    "franka_msgs/action/Grasp server used only to close; only a "
                    "successful result may latch CloseGripper."
                ),
            ),
            DeclareLaunchArgument(
                "move_action_name",
                default_value="/franka_gripper/move",
                description="franka_msgs/action/Move server used to release/open.",
            ),
            DeclareLaunchArgument(
                "telemetry_topic",
                default_value="/panda_student_v6/telemetry",
            ),
            DeclareLaunchArgument(
                "seated_offset_m",
                default_value="0.0229",
                description=(
                    "Measured TCP-to-seated-object vertical offset. The default is "
                    "the training-range mean and must be calibrated."
                ),
            ),
            OpaqueFunction(function=_start_student_v6),
        ]
    )
