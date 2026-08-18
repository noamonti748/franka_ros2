"""Launch Panda pick-place with MuJoCo and ros2_control on ROS 2 Humble."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
    RegisterEventHandler,
    Shutdown,
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile, ParameterValue
from launch_ros.substitutions import FindPackageShare


def _launch_setup(context, *args, **kwargs):
    del args, kwargs
    package_share = FindPackageShare("franka_mujoco_bringup")
    controllers_file = PathJoinSubstitution(
        [package_share, "config", "controllers.yaml"]
    )
    plugins_file = PathJoinSubstitution(
        [package_share, "config", "mujoco_plugins.yaml"]
    )

    robot_description_command = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [package_share, "urdf", "panda_mujoco.urdf.xacro"]
            ),
            " headless:=",
            LaunchConfiguration("headless"),
            " sim_speed_factor:=",
            LaunchConfiguration("sim_speed_factor"),
            " mujoco_model:=",
            LaunchConfiguration("mujoco_model"),
        ]
    )
    robot_description_xml = robot_description_command.perform(context)
    robot_description_value = ParameterValue(
        robot_description_xml, value_type=str
    )
    robot_description = {
        "robot_description": robot_description_value
    }

    state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description, {"use_sim_time": True}],
    )

    control_node = Node(
        package="mujoco_ros2_control",
        executable="ros2_control_node",
        output="both",
        emulate_tty=True,
        parameters=[
            robot_description,
            {"use_sim_time": True},
            ParameterFile(controllers_file),
            ParameterFile(plugins_file),
        ],
        # controller_manager on Humble subscribes to this private name.
        remappings=[("~/robot_description", "/robot_description")],
        on_exit=Shutdown(reason="MuJoCo ros2_control node exited"),
    )

    joint_state_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager",
            "--param-file",
            controllers_file,
        ],
        output="both",
    )
    arm_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "panda_arm_controller",
            "--controller-manager",
            "/controller_manager",
            "--param-file",
            controllers_file,
        ],
        output="both",
    )
    gripper_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "panda_gripper_controller",
            "--controller-manager",
            "/controller_manager",
            "--param-file",
            controllers_file,
        ],
        output="both",
    )
    oak_emulator = Node(
        package="franka_mujoco_bringup",
        executable="oak_camera_emulator.py",
        name="oak_camera_emulator",
        output="both",
        parameters=[
            {
                "use_sim_time": True,
                "enabled": ParameterValue(
                    LaunchConfiguration("oak_emulator_enabled"),
                    value_type=bool,
                ),
                "base_delay_ms": ParameterValue(
                    LaunchConfiguration("oak_delay_ms"), value_type=float
                ),
                "jitter_ms": ParameterValue(
                    LaunchConfiguration("oak_jitter_ms"), value_type=float
                ),
                "drop_probability": ParameterValue(
                    LaunchConfiguration("oak_drop_probability"),
                    value_type=float,
                ),
                "seed": ParameterValue(
                    LaunchConfiguration("oak_seed"), value_type=int
                ),
            }
        ],
    )

    policy_node = Node(
        package=LaunchConfiguration("policy_package"),
        executable="panda_student_v6_ros2",
        name="student_v6_policy",
        output="both",
        parameters=[
            {
                "use_sim_time": True,
                "policy_enabled": ParameterValue(
                    LaunchConfiguration("policy_enabled"), value_type=bool
                ),
                "autonomous_controller": ParameterValue(
                    LaunchConfiguration("autonomous_controller"),
                    value_type=bool,
                ),
                "dls_enabled": ParameterValue(
                    LaunchConfiguration("dls_enabled"), value_type=bool
                ),
                "advance_threshold": ParameterValue(
                    LaunchConfiguration("advance_threshold"), value_type=float
                ),
                "mode_min_dwell_steps": ParameterValue(
                    LaunchConfiguration("mode_min_dwell_steps"), value_type=int
                ),
                "approach_lock_wrist_home": ParameterValue(
                    LaunchConfiguration("approach_lock_wrist_home"),
                    value_type=bool,
                ),
                "onnx_model_path": LaunchConfiguration("policy_model_path"),
                "robot_id": "panda",
                "base_frame": "panda_link0",
                "tcp_frame": "panda_hand_tcp",
                "camera_frame": "policy_camera_optical_frame",
                "joint_states_topic": "/joint_states",
                "gripper_state_topic": "/joint_states",
                "image_topic": "/policy_camera/color/image_raw",
                "image_subscription_depth": 1,
                "image_qos_reliability": "best_effort",
                "image_stale_timeout_s": 0.10,
                "command_topic": "/panda_arm_controller/commands",
                "command_message_type": "float64_multi_array",
                "telemetry_topic": "/panda_student_v6/telemetry",
                "control_hz": 60.0,
                "action_latency_steps": 0,
                "response_time_s": 0.5,
                "snapshot_poll_hz": 240.0,
                "trajectory_duration_s": 0.05,
                # The trained integrator already enforces rate/lead limits.
                # Bypass the second command-history limiter in simulation so
                # the 60 Hz setpoint matches direct MuJoCo exactly.
                "bypass_command_limiter": True,
                "dls_anchor_estimate": False,
                "dls_anchor_alpha": 0.1,
                "dls_anchor_clamp_radius_m": 0.025,
                "dls_descend_target": "sim_box",
                "box_state_topic": "/box/free_joint_state",
                "gripper_backend": "sim",
                "sim_grasp_width_tolerance_m": 0.01,
                "gripper_action_name": (
                    "/panda_gripper_controller/gripper_cmd"
                ),
                "use_pykdl": True,
                "robot_description": robot_description_value,
                # The seeded evaluator pauses, resets, and seeds the controller
                # command atomically before enabling policy execution.
                "startup_home_enabled": False,
                "startup_home_command_enabled": False,
                "startup_home_trajectory_duration_s": 2.0,
                "startup_home_tolerance_rad": 0.005,
                "startup_home_velocity_tolerance_rad_s": 0.02,
                "readiness_consecutive_ticks": 1,
            }
        ],
    )

    # Spawn in dependency order.  The policy process always starts after both
    # action servers; policy_enabled=false leaves it running monitor-only.
    start_arm = RegisterEventHandler(
        OnProcessExit(target_action=joint_state_spawner, on_exit=[arm_spawner])
    )
    start_gripper = RegisterEventHandler(
        OnProcessExit(target_action=arm_spawner, on_exit=[gripper_spawner])
    )
    start_policy = RegisterEventHandler(
        OnProcessExit(target_action=gripper_spawner, on_exit=[policy_node])
    )

    return [
        state_publisher,
        control_node,
        oak_emulator,
        joint_state_spawner,
        start_arm,
        start_gripper,
        start_policy,
    ]


def generate_launch_description():
    default_model = PathJoinSubstitution(
        [
            FindPackageShare("franka_emika_panda"),
            "models",
            "student_v6",
            "student_v6_final.onnx",
        ]
    )
    default_mujoco_model = PathJoinSubstitution(
        [
            FindPackageShare("franka_mujoco_bringup"),
            "mjcf",
            "panda_pick_place.xml",
        ]
    )
    arguments = [
        DeclareLaunchArgument(
            "headless",
            default_value="true",
            description="Run MuJoCo without the Simulate GUI",
        ),
        DeclareLaunchArgument(
            "sim_speed_factor",
            default_value="1.0",
            description="Wall-clock speed multiplier for the simulation",
        ),
        DeclareLaunchArgument(
            "policy_enabled",
            default_value="false",
            description=(
                "Enable student-v6 commands; false starts it monitor-only"
            ),
        ),
        DeclareLaunchArgument(
            "autonomous_controller",
            default_value="false",
            description="Use the ONNX mode head and disable task-gate ownership",
        ),
        DeclareLaunchArgument(
            "dls_enabled",
            default_value="true",
            description="Enable the legacy phase-selective DLS controller",
        ),
        DeclareLaunchArgument(
            "advance_threshold",
            default_value="0.55",
            description="Learned advance-head threshold (autonomous_controller)",
        ),
        DeclareLaunchArgument(
            "mode_min_dwell_steps",
            default_value="20",
            description="Min phase dwell before advance may fire (autonomous)",
        ),
        DeclareLaunchArgument(
            "approach_lock_wrist_home",
            default_value="true",
            description="Apply the legacy approach wrist lock",
        ),
        DeclareLaunchArgument(
            "policy_package",
            default_value="franka_emika_panda",
            description="Package providing the student-v6 ROS node",
        ),
        DeclareLaunchArgument(
            "policy_model_path",
            default_value=default_model,
            description="Installed student-v6 ONNX model",
        ),
        DeclareLaunchArgument(
            "mujoco_model",
            default_value=default_mujoco_model,
            description="Compiled MJCF path, including camera variants",
        ),
        DeclareLaunchArgument(
            "oak_emulator_enabled",
            default_value="false",
            description="Enable OAK-1 delay/jitter/drop emulation",
        ),
        DeclareLaunchArgument(
            "oak_delay_ms",
            default_value="0.0",
            description="Fixed camera transport delay in milliseconds",
        ),
        DeclareLaunchArgument(
            "oak_jitter_ms",
            default_value="0.0",
            description="Uniform camera transport jitter half-width",
        ),
        DeclareLaunchArgument(
            "oak_drop_probability",
            default_value="0.0",
            description="Independent camera frame drop probability",
        ),
        DeclareLaunchArgument(
            "oak_seed",
            default_value="0",
            description="Deterministic camera fault seed",
        ),
    ]
    return LaunchDescription(arguments + [OpaqueFunction(function=_launch_setup)])
