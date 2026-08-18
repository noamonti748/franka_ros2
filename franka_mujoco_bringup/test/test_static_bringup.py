"""ROS-free structural checks for the Panda MuJoCo bringup package."""

import ast
import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
ARM_JOINTS = [f"panda_joint{index}" for index in range(1, 8)]
FINGER_JOINTS = ["panda_finger_joint1", "panda_finger_joint2"]


def _parse_xml(relative_path):
    path = ROOT / relative_path
    if path.name == "panda_pick_place.xml.in":
        source = path.read_text(encoding="utf-8")
        source = (
            source.replace("@POLICY_CAMERA_POS@", "1.0 0 0.5")
            .replace(
                "@POLICY_CAMERA_XYAXES@",
                "0 1 0 -0.7660444431 0 0.6427876097",
            )
            .replace("@POLICY_CAMERA_FOVY@", "55")
            .replace("@FRANKA_PANDA_ASSETS@", "/tmp/panda-assets")
        )
        return ET.fromstring(source)
    return ET.parse(path).getroot()


def test_all_xml_sources_are_well_formed():
    for relative_path in (
        "package.xml",
        "urdf/panda_mujoco.urdf.xacro",
        "mjcf/panda_pick_place.xml.in",
    ):
        _parse_xml(relative_path)


def test_mjcf_scene_and_camera_match_training_contract():
    model = _parse_xml("mjcf/panda_pick_place.xml.in")
    option = model.find("option")
    assert option is not None
    assert float(option.attrib["timestep"]) == 1.0 / 240.0
    assert option.attrib["integrator"] == "RK4"

    camera = model.find("./worldbody/camera[@name='policy_camera']")
    assert camera is not None
    assert camera.attrib["resolution"] == "64 64"
    assert float(camera.attrib["fovy"]) == 55.0
    assert camera.attrib["pos"] == "1.0 0 0.5"
    assert camera.attrib["xyaxes"] == (
        "0 1 0 -0.7660444431 0 0.6427876097"
    )
    scene = model.find("./worldbody/camera[@name='scene_camera']")
    assert scene is not None
    assert scene.attrib["resolution"] == "640 480"

    assert model.find("./worldbody/geom[@name='floor']") is not None
    assert model.find(".//body[@name='box']/geom[@material='cube_red']") is not None
    assert (
        model.find(
            ".//body[@name='mocap_target']/geom[@material='paper_white']"
        )
        is not None
    )
    collision_default = model.find(
        "./default/default/default[@class='collision']/geom"
    )
    assert collision_default is not None
    assert collision_default.attrib["rgba"] == "0 0 0 0"
    assert all(
        light.attrib.get("castshadow") == "false"
        for light in model.findall("./worldbody/light")
    )

    home = model.find("./keyframe/key[@name='home']")
    assert home is not None
    assert len(home.attrib["qpos"].split()) == 16
    assert len(home.attrib["ctrl"].split()) == 8
    seeded = [
        model.find(f"./keyframe/key[@name='eval_seed_{seed}']")
        for seed in range(1, 17)
    ]
    assert all(item is not None for item in seeded)
    assert all(len(item.attrib["qpos"].split()) == 16 for item in seeded)
    assert all(len(item.attrib["ctrl"].split()) == 8 for item in seeded)
    assert all(len(item.attrib["mpos"].split()) == 3 for item in seeded)
    assert all(len(item.attrib["mquat"].split()) == 4 for item in seeded)
    seed_1_qpos = [float(value) for value in seeded[0].attrib["qpos"].split()]
    seed_1_mpos = [float(value) for value in seeded[0].attrib["mpos"].split()]
    np.testing.assert_allclose(seed_1_qpos[9:12], [0.49710938, 0.15146686, 0.025])
    np.testing.assert_allclose(seed_1_mpos, [0.50173355, -0.15827835, 0.0005])
    seed_16_qpos = [float(value) for value in seeded[-1].attrib["qpos"].split()]
    seed_16_mpos = [float(value) for value in seeded[-1].attrib["mpos"].split()]
    np.testing.assert_allclose(seed_16_qpos[9:12], [0.5072435, 0.13542804, 0.025])
    np.testing.assert_allclose(seed_16_mpos, [0.4988371, -0.16125461, 0.0005])


def test_mjcf_joint_and_actuator_names_map_directly():
    model = _parse_xml("mjcf/panda_pick_place.xml.in")
    joint_names = {
        element.attrib["name"]
        for element in model.findall(".//joint")
        if "name" in element.attrib
    }
    assert set(ARM_JOINTS + FINGER_JOINTS).issubset(joint_names)

    actuators = {
        element.attrib["name"]: element.attrib.get("joint")
        for element in model.findall("./actuator/*")
    }
    expected_controlled = ARM_JOINTS + ["panda_finger_joint1"]
    assert set(actuators) == set(expected_controlled)
    assert all(actuators[name] == name for name in expected_controlled)
    assert "panda_finger_joint2" not in actuators

    equality = model.find(
        "./equality/joint[@joint1='panda_finger_joint1']"
        "[@joint2='panda_finger_joint2']"
    )
    assert equality is not None


def test_urdf_hardware_contract_matches_mjcf():
    robot = _parse_xml("urdf/panda_mujoco.urdf.xacro")
    control = robot.find("ros2_control")
    assert control is not None
    assert control.findtext("./hardware/plugin") == (
        "mujoco_ros2_control/MujocoSystemInterface"
    )
    hardware_parameters = {
        item.attrib["name"]: (item.text or "").strip()
        for item in control.findall("./hardware/param")
    }
    assert hardware_parameters["initial_keyframe"] == "home"
    assert hardware_parameters["headless"] == "$(arg headless)"
    assert hardware_parameters["sim_speed_factor"] == "$(arg sim_speed_factor)"
    assert hardware_parameters["mujoco_model"] == "$(arg mujoco_model)"

    control_joints = {
        joint.attrib["name"]: joint for joint in control.findall("joint")
    }
    assert set(control_joints) == set(ARM_JOINTS + FINGER_JOINTS)
    for name in ARM_JOINTS + ["panda_finger_joint1"]:
        assert (
            control_joints[name].find("./command_interface[@name='position']")
            is not None
        )
    assert (
        control_joints["panda_finger_joint2"].find("command_interface") is None
    )


def test_controller_and_camera_yaml_contracts():
    controllers = yaml.safe_load(
        (ROOT / "config/controllers.yaml").read_text(encoding="utf-8")
    )
    manager = controllers["controller_manager"]["ros__parameters"]
    assert manager["update_rate"] == 240
    assert manager["panda_arm_controller"]["type"] == (
        "forward_command_controller/ForwardCommandController"
    )
    assert manager["panda_gripper_controller"]["type"] == (
        "position_controllers/GripperActionController"
    )
    arm = controllers["panda_arm_controller"]["ros__parameters"]
    assert arm["joints"] == ARM_JOINTS
    assert arm["interface_name"] == "position"
    gripper = controllers["panda_gripper_controller"]["ros__parameters"]
    assert gripper["joint"] == "panda_finger_joint1"
    assert gripper["action_monitor_rate"] == 120.0
    assert gripper["allow_stalling"] is True
    assert gripper["stall_timeout"] == 0.25

    plugins = yaml.safe_load(
        (ROOT / "config/mujoco_plugins.yaml").read_text(encoding="utf-8")
    )
    camera_plugin = plugins["/**"]["ros__parameters"]["mujoco_plugins"][
        "mujoco_camera_plugin"
    ]
    assert camera_plugin["type"] == (
        "mujoco_ros2_control_plugins/CameraPlugin"
    )
    assert camera_plugin["camera_publish_rate"] == 60.0
    camera = camera_plugin["policy_camera"]
    assert camera["policy"] == "streaming"
    assert camera["frame_name"] == "policy_camera_optical_frame"
    assert camera["image_topic"] == "/policy_camera/sim/color/image_raw"
    assert camera["depth_topic"] == "/policy_camera/sim/depth/image_raw"
    assert camera["info_topic"] == "/policy_camera/sim/camera_info"
    scene = camera_plugin["scene_camera"]
    assert scene["policy"] == "polled"
    assert scene["image_topic"] == "/scene_camera/color/image_raw"
    assert scene["trigger_service_name"] == "/scene_camera/trigger"
    box_plugin = plugins["/**"]["ros__parameters"]["mujoco_plugins"][
        "free_joint_state_publisher"
    ]
    assert box_plugin["type"] == (
        "mujoco_ros2_control_plugins/FreeJointStatePublisherPlugin"
    )
    assert box_plugin["body_names"] == ["box"]
    assert box_plugin["topic"] == "/box/free_joint_state"


def test_launch_is_valid_python_and_contains_humble_wiring():
    launch_path = ROOT / "launch/panda_pick_place.launch.py"
    source = launch_path.read_text(encoding="utf-8")
    ast.parse(source, filename=str(launch_path))

    assert 'package="mujoco_ros2_control"' in source
    assert 'executable="ros2_control_node"' in source
    assert '("~/robot_description", "/robot_description")' in source
    assert '{"use_sim_time": True}' in source
    for controller in (
        "joint_state_broadcaster",
        "panda_arm_controller",
        "panda_gripper_controller",
    ):
        assert f'"{controller}"' in source
    assert 'executable="panda_student_v6_ros2"' in source
    assert source.index("target_action=gripper_spawner") < source.index(
        "on_exit=[policy_node]"
    )
    assert '"policy_enabled": ParameterValue(' in source
    assert 'LaunchConfiguration("policy_enabled"), value_type=bool' in source
    assert 'default_value="false"' in source
    assert '"student_v6",' in source
    assert '"student_v6_final.onnx",' in source

    expected_policy_parameters = {
        '"onnx_model_path": LaunchConfiguration("policy_model_path")',
        '"robot_id": "panda"',
        '"base_frame": "panda_link0"',
        '"tcp_frame": "panda_hand_tcp"',
        '"camera_frame": "policy_camera_optical_frame"',
        '"joint_states_topic": "/joint_states"',
        '"gripper_state_topic": "/joint_states"',
        '"image_topic": "/policy_camera/color/image_raw"',
        '"image_subscription_depth": 1',
        '"image_qos_reliability": "best_effort"',
        '"image_stale_timeout_s": 0.10',
        '"command_topic": "/panda_arm_controller/commands"',
        '"command_message_type": "float64_multi_array"',
        '"telemetry_topic": "/panda_student_v6/telemetry"',
        '"control_hz": 60.0',
        '"action_latency_steps": 0',
        '"response_time_s": 0.5',
        '"snapshot_poll_hz": 240.0',
        '"trajectory_duration_s": 0.05',
        '"bypass_command_limiter": True',
        '"dls_anchor_estimate": False',
        '"dls_anchor_alpha": 0.1',
        '"dls_anchor_clamp_radius_m": 0.025',
        '"dls_descend_target": "sim_box"',
        '"box_state_topic": "/box/free_joint_state"',
        '"gripper_backend": "sim"',
        '"sim_grasp_width_tolerance_m": 0.01',
        '"/panda_gripper_controller/gripper_cmd"',
        '"use_pykdl": True',
        '"robot_description": robot_description_value',
        '"startup_home_enabled": False',
        '"startup_home_command_enabled": False',
        '"startup_home_trajectory_duration_s": 2.0',
        '"startup_home_tolerance_rad": 0.005',
        '"startup_home_velocity_tolerance_rad_s": 0.02',
        '"readiness_consecutive_ticks": 1',
    }
    for parameter in expected_policy_parameters:
        assert parameter in source

    assert 'executable="oak_camera_emulator.py"' in source
    assert 'LaunchConfiguration("oak_emulator_enabled")' in source
    assert '"model_path":' not in source
    assert '"policy_rate_hz":' not in source
    assert '"policy_executable"' not in source
    assert source.count("remappings=") == 1
    for wrong_endpoint in (
        '"camera/image_raw"',
        '"camera/camera_info"',
        '"arm_controller/joint_trajectory"',
        '"gripper_controller/gripper_cmd"',
    ):
        assert wrong_endpoint not in source

    evaluator = ROOT / "scripts/evaluate_student_v24.py"
    evaluator_source = evaluator.read_text(encoding="utf-8")
    ast.parse(evaluator_source, filename=str(evaluator))
    assert "physical_success" in evaluator_source
    assert "estimator_success" in evaluator_source
    corpus_builder = ROOT / "scripts/build_ros_geometry_corpus.py"
    ast.parse(
        corpus_builder.read_text(encoding="utf-8"),
        filename=str(corpus_builder),
    )
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "scripts/evaluate_student_v24.py" in cmake
    assert "scripts/build_ros_geometry_corpus.py" in cmake


def test_build_config_verifies_vendored_mujoco_and_panda_assets():
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    model = (ROOT / "mjcf/panda_pick_place.xml.in").read_text(
        encoding="utf-8"
    )
    assert "find_package(franka_emika_panda REQUIRED)" in cmake
    assert "find_package(mujoco_vendor REQUIRED)" in cmake
    assert "find_package(mujoco " not in cmake
    assert "lib/libmujoco.so.3.10.0" in cmake
    assert "include/mujoco/mujoco.h" in cmake
    assert "include/simulate/simulate.h" in cmake
    assert "configure_file(" in cmake
    assert "@FRANKA_PANDA_ASSETS@" in model
    assert not (ROOT / "mjcf/assets").exists()
    assert "ff9e648e1af418c555c37cb6b4fcc42885057421" in readme
    assert "bootstrap_humble_mujoco.sh --backend pixi --all" in readme
    assert "AMENT_VENDOR_POLICY=NEVER_VENDOR" in readme
    assert "No upstream checkout is" in readme


def test_mjcf_compiles_with_mujoco_3_10_when_available():
    if importlib.util.find_spec("mujoco") is None:
        return

    import mujoco

    assert mujoco.__version__ == "3.10.0"
    assets = ROOT.parent / "franka_emika_panda" / "assets"
    assert (assets / "link1.obj").is_file()
    template = (ROOT / "mjcf/panda_pick_place.xml.in").read_text(
        encoding="utf-8"
    )
    template = (
        template.replace("@FRANKA_PANDA_ASSETS@", str(assets))
        .replace("@POLICY_CAMERA_POS@", "1.0 0 0.5")
        .replace(
            "@POLICY_CAMERA_XYAXES@",
            "0 1 0 -0.7660444431 0 0.6427876097",
        )
        .replace("@POLICY_CAMERA_FOVY@", "55")
    )
    model = mujoco.MjModel.from_xml_string(template)
    assert model.nq == 16
    assert model.nu == 8
    assert model.ncam == 2
    assert model.opt.timestep == 1.0 / 240.0
    assert int(model.opt.integrator) == int(mujoco.mjtIntegrator.mjINT_RK4)
    assert model.dof_damping[:9].tolist() == [
        40.0,
        40.0,
        40.0,
        40.0,
        2.0,
        2.0,
        2.0,
        10.0,
        10.0,
    ]
    np.testing.assert_allclose(
        model.actuator_gainprm[:8, :3],
        [
            [1000.0, 0.0, 0.0],
            [1000.0, 0.0, 0.0],
            [750.0, 0.0, 0.0],
            [750.0, 0.0, 0.0],
            [300.0, 0.0, 0.0],
            [300.0, 0.0, 0.0],
            [300.0, 0.0, 0.0],
            [350.0, 0.0, 0.0],
        ],
    )
    np.testing.assert_allclose(
        model.actuator_biasprm[:8, :3],
        [
            [0.0, -1000.0, -20.0],
            [0.0, -1000.0, -20.0],
            [0.0, -750.0, -4.0],
            [0.0, -750.0, -4.0],
            [0.0, -300.0, -2.0],
            [0.0, -300.0, -2.0],
            [0.0, -300.0, -2.0],
            [0.0, -350.0, -10.0],
        ],
    )
    np.testing.assert_allclose(
        model.actuator_forcerange[:8],
        [
            [-100.0, 100.0],
            [-100.0, 100.0],
            [-100.0, 100.0],
            [-100.0, 100.0],
            [-12.0, 12.0],
            [-12.0, 12.0],
            [-12.0, 12.0],
            [-200.0, 200.0],
        ],
    )
    for actuator_id, expected_name in enumerate(
        ARM_JOINTS + ["panda_finger_joint1"]
    ):
        assert (
            mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id
            )
            == expected_name
        )
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        assert (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            == expected_name
        )
    camera_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, "policy_camera"
    )
    assert model.cam_resolution[camera_id].tolist() == [64, 64]
    assert model.cam_fovy[camera_id] == 55.0
    scene_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, "scene_camera"
    )
    assert scene_id >= 0
    assert model.cam_resolution[scene_id].tolist() == [640, 480]
    home_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_KEY, "home"
    )
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    assert data.qpos[:9].tolist() == [
        0.0,
        0.3,
        0.0,
        -1.5707963268,
        0.0,
        2.0,
        -0.7853981634,
        0.04,
        0.04,
    ]
