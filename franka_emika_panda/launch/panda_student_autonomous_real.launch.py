"""Monitor-first real-hardware launch for the autonomous distilled student."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare("franka_emika_panda")
    base_launch = PathJoinSubstitution(
        [package_share, "launch", "panda_student_v6_real.launch.py"]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("policy_enabled", default_value="false"),
            DeclareLaunchArgument(
                "enable_confirmation", default_value="NOT_CONFIRMED"
            ),
            DeclareLaunchArgument(
                "image_topic",
                default_value="/policy_camera/image_raw",
                description=(
                    "Physical RGB image topic. Override for the site camera "
                    "namespace, including legacy /medcvr/... topics."
                ),
            ),
            DeclareLaunchArgument(
                "camera_info_topic",
                default_value="/policy_camera/camera_info",
                description=(
                    "Physical CameraInfo topic paired with image_topic."
                ),
            ),
            DeclareLaunchArgument(
                "camera_frame",
                default_value="policy_camera_optical_frame",
                description="Calibrated optical frame published in TF.",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(base_launch),
                launch_arguments={
                    "config": PathJoinSubstitution(
                        [package_share, "config", "student_v6_real.yaml"]
                    ),
                    "profile_config": PathJoinSubstitution(
                        [package_share, "config", "student_autonomous_v1.yaml"]
                    ),
                    "onnx_model_path": PathJoinSubstitution(
                        [
                            package_share,
                            "models",
                            "student_autonomous_v1",
                            "student_autonomous_v1.onnx",
                        ]
                    ),
                    "policy_enabled": LaunchConfiguration("policy_enabled"),
                    "enable_confirmation": LaunchConfiguration(
                        "enable_confirmation"
                    ),
                    "image_topic": LaunchConfiguration("image_topic"),
                    "camera_info_topic": LaunchConfiguration(
                        "camera_info_topic"
                    ),
                    "camera_frame": LaunchConfiguration("camera_frame"),
                }.items(),
            ),
        ]
    )
