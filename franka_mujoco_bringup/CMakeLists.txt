cmake_minimum_required(VERSION 3.10)
if(POLICY CMP0072)
  cmake_policy(SET CMP0072 NEW)
endif()
project(franka_mujoco_bringup NONE)

find_package(ament_cmake REQUIRED)
find_package(franka_emika_panda REQUIRED)
find_package(mujoco_vendor REQUIRED)

# The bootstrap builds mujoco_vendor in NEVER_VENDOR mode and creates an
# overlay-local forwarding link from opt/mujoco_vendor to the isolated,
# locked MuJoCo 3.10.0 Pixi prefix. Verify that exact ABI through the
# forwarding package instead of relying on whichever MuJoCo is on PATH.
get_filename_component(
  _mujoco_vendor_cmake_dir "${mujoco_vendor_DIR}" REALPATH
)
get_filename_component(
  _mujoco_vendor_prefix "${_mujoco_vendor_cmake_dir}/../../.." REALPATH
)
set(_mujoco_vendor_root "${_mujoco_vendor_prefix}/opt/mujoco_vendor")
set(_mujoco_3_10_library
  "${_mujoco_vendor_root}/lib/libmujoco.so.3.10.0"
)

if(NOT EXISTS "${_mujoco_3_10_library}")
  message(FATAL_ERROR
    "MuJoCo 3.10.0 forwarding library not found at ${_mujoco_3_10_library}. "
    "Run scripts/bootstrap_humble_mujoco.sh --backend pixi --all.")
endif()
if(NOT EXISTS "${_mujoco_vendor_root}/include/mujoco/mujoco.h")
  message(FATAL_ERROR "Vendored MuJoCo headers are missing")
endif()
if(NOT EXISTS "${_mujoco_vendor_root}/include/simulate/simulate.h")
  message(FATAL_ERROR "Vendored MuJoCo Simulate headers are missing")
endif()
message(STATUS "Using vendored MuJoCo 3.10.0 at ${_mujoco_vendor_root}")

# Resolve the installed Panda mesh directory once at configure time.  MuJoCo
# does not expand package:// URIs in MJCF mesh paths, so the small MJCF source
# stays in this package while the large meshes remain owned by
# franka_emika_panda.
get_filename_component(_franka_panda_cmake_dir "${franka_emika_panda_DIR}" REALPATH)
get_filename_component(FRANKA_PANDA_SHARE "${_franka_panda_cmake_dir}/.." REALPATH)
set(FRANKA_PANDA_ASSETS "${FRANKA_PANDA_SHARE}/assets")

if(NOT EXISTS "${FRANKA_PANDA_ASSETS}/link1.obj")
  message(FATAL_ERROR
    "franka_emika_panda assets were not found at ${FRANKA_PANDA_ASSETS}")
endif()

function(configure_pick_place_variant name position xyaxes fovy)
  set(POLICY_CAMERA_POS "${position}")
  set(POLICY_CAMERA_XYAXES "${xyaxes}")
  set(POLICY_CAMERA_FOVY "${fovy}")
  configure_file(
    mjcf/panda_pick_place.xml.in
    "${CMAKE_CURRENT_BINARY_DIR}/${name}.xml"
    @ONLY
  )
endfunction()

set(_camera_xy_nominal "0 1 0 -0.7660444431 0 0.6427876097")
configure_pick_place_variant(
  panda_pick_place "1.0 0 0.5" "${_camera_xy_nominal}" "55"
)
configure_pick_place_variant(
  panda_pick_place_cam_pos_minus "0.9984 0 0.5" "${_camera_xy_nominal}" "55"
)
configure_pick_place_variant(
  panda_pick_place_cam_pos_plus "1.0016 0 0.5" "${_camera_xy_nominal}" "55"
)
configure_pick_place_variant(
  panda_pick_place_cam_pitch_minus
  "1.0 0 0.5" "0 1 0 -0.7626683297 0 0.6467897795" "55"
)
configure_pick_place_variant(
  panda_pick_place_cam_pitch_plus
  "1.0 0 0.5" "0 1 0 -0.7693995550 0 0.6387678175" "55"
)
configure_pick_place_variant(
  panda_pick_place_cam_fovy_minus "1.0 0 0.5" "${_camera_xy_nominal}" "54.7"
)
configure_pick_place_variant(
  panda_pick_place_cam_fovy_plus "1.0 0 0.5" "${_camera_xy_nominal}" "55.3"
)

install(
  DIRECTORY launch config urdf
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/build_ros_geometry_corpus.py
    scripts/compare_policy_traces.py
    scripts/evaluate_student_v24.py
    scripts/oak_camera_emulator.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/panda_pick_place.xml"
    "${CMAKE_CURRENT_BINARY_DIR}/panda_pick_place_cam_pos_minus.xml"
    "${CMAKE_CURRENT_BINARY_DIR}/panda_pick_place_cam_pos_plus.xml"
    "${CMAKE_CURRENT_BINARY_DIR}/panda_pick_place_cam_pitch_minus.xml"
    "${CMAKE_CURRENT_BINARY_DIR}/panda_pick_place_cam_pitch_plus.xml"
    "${CMAKE_CURRENT_BINARY_DIR}/panda_pick_place_cam_fovy_minus.xml"
    "${CMAKE_CURRENT_BINARY_DIR}/panda_pick_place_cam_fovy_plus.xml"
  DESTINATION share/${PROJECT_NAME}/mjcf
)

install(
  FILES README.md
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest QUIET)
  if(ament_cmake_pytest_FOUND)
    ament_add_pytest_test(
      static_bringup_test
      test/test_static_bringup.py
      ENV "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
  endif()
endif()

ament_package()
