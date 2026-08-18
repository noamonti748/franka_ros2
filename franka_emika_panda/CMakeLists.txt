cmake_minimum_required(VERSION 3.10)

project(franka_emika_panda NONE)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(
  franka_emika_panda_policy
  PACKAGE_DIR python/franka_emika_panda_policy
)

set(_menagerie_assets_candidates
  "${CMAKE_CURRENT_SOURCE_DIR}/assets"
  "${CMAKE_CURRENT_SOURCE_DIR}/mujoco_menagerie/franka_emika_panda/assets"
)

set(_menagerie_assets "")
foreach(_candidate ${_menagerie_assets_candidates})
  if(EXISTS "${_candidate}/link1.obj")
    set(_menagerie_assets "${_candidate}")
    break()
  endif()
endforeach()

if(NOT _menagerie_assets)
  message(FATAL_ERROR
    "MuJoCo Menagerie meshes are missing. Expected link1.obj in one of:\n"
    "  ${CMAKE_CURRENT_SOURCE_DIR}/assets\n"
    "  ${CMAKE_CURRENT_SOURCE_DIR}/mujoco_menagerie/franka_emika_panda/assets\n"
    "From the repo root, run:\n"
    "  ./franka_emika_panda/scripts/init_menagerie_assets.sh")
endif()

get_filename_component(_menagerie_assets "${_menagerie_assets}" REALPATH)

install(
  DIRECTORY "${_menagerie_assets}/"
  DESTINATION share/${PROJECT_NAME}/assets
  USE_SOURCE_PERMISSIONS
)

install(
  DIRECTORY launch config models
  DESTINATION share/${PROJECT_NAME}
)

install(
  FILES
    panda.urdf.xacro
    ppo_track_franka.onnx
    ppo_final_constrained.onnx
    ppo_final_reduce_shake.onnx
    ppo_final_lower_vib.onnx
    ppo_final_rf3.onnx
    fr3_track_reach.rviz
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
    panda_reach_constrained_mujoco_rollout.py
    analyze_policy_telemetry.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS
    panda_track_reach_ros2.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME panda_track_reach_ros2
)

# panda_reach_constrained_ros2.py imports shared helpers (PANDA_ARM_JOINTS,
# PandaTrackReachSession, OnnxPolicyRunner, etc.) from panda_track_reach_ros2.py at
# runtime, so also install the latter under its plain importable .py name alongside
# the ros2-run-friendly renamed copy above.
install(
  FILES
    panda_track_reach_ros2.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS
    panda_reach_constrained_ros2.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME panda_reach_constrained_ros2
)

install(
  PROGRAMS
    panda_reach_reduce_shake_ros2.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME panda_reach_reduce_shake_ros2
)

install(
  PROGRAMS
    panda_student_v6_ros2.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME panda_student_v6_ros2
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(
    student_v6_runtime_tests
    test
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python
    TIMEOUT 120
  )
endif()

ament_package()
