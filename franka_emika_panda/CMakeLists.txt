cmake_minimum_required(VERSION 3.8)

project(franka_emika_panda NONE)

find_package(ament_cmake REQUIRED)

set(_menagerie_assets "${CMAKE_CURRENT_SOURCE_DIR}/mujoco_menagerie/franka_emika_panda/assets")
if(NOT EXISTS "${_menagerie_assets}/link1.obj")
  message(FATAL_ERROR
    "MuJoCo Menagerie meshes are missing. From the repo root run:\n"
    "  ./franka_emika_panda/scripts/init_menagerie_assets.sh")
endif()

install(
  DIRECTORY "${_menagerie_assets}/"
  DESTINATION share/${PROJECT_NAME}/assets
  USE_SOURCE_PERMISSIONS
)

install(
  FILES
    panda.urdf.xacro
    ppo_track_franka.onnx
    joint_limits.yaml
    panda.xml
    panda_nohand.xml
    hand.xml
    scene.xml
    mjx_panda.xml
    mjx_panda_nohand.xml
    mjx_hand.xml
    mjx_scene.xml
    mjx_single_cube.xml
    mjx_panda_reach.xml
    mjx_panda_reach_ue.xml
    panda_ue.xml
    panda.png
    README.md
    CHANGELOG.md
    LICENSE
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    panda_mujoco_onnx_rollout.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS
    panda_track_reach_ros2.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME panda_track_reach_ros2
)

ament_package()
