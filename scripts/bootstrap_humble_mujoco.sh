#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

readonly ROS_DISTRO_TARGET="humble"
readonly JAMMY_CODENAME="jammy"
readonly DEFAULT_WORKSPACE="${HOME}/franka_humble_mujoco_ws"
readonly DEFAULT_SOURCE_ROOT="/home/noamonti/franka_ros2"
readonly DEFAULT_PIXI_PROJECT="/home/noamonti/franka_humble_mujoco_pixi"
readonly MUJOCO_ROS2_CONTROL_COMMIT="7ee969776fe097f0aedbb3ae6edcf9f85fb31244"
readonly MUJOCO_VENDOR_COMMIT="ff9e648e1af418c555c37cb6b4fcc42885057421"
readonly MUJOCO_VERSION="3.10.0"
readonly PIXI_VERSION="0.76.2"
readonly ONNXRUNTIME_VERSION="1.23.2"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPOS_FILE="${REPOSITORY_ROOT}/dependency-humble-mujoco.repos"
WORKSPACE="${FRANKA_HUMBLE_MUJOCO_WS:-${DEFAULT_WORKSPACE}}"
SOURCE_ROOT="${FRANKA_ROS2_SOURCE_ROOT:-${DEFAULT_SOURCE_ROOT}}"
PIXI_PROJECT="${FRANKA_MUJOCO_PIXI_PROJECT:-${DEFAULT_PIXI_PROJECT}}"
BACKEND="pixi"

DO_INSTALL=false
DO_IMPORT=false
DO_BUILD=false

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Prepare an isolated Ubuntu 22.04 / ROS 2 Humble MuJoCo overlay. With no action
option, this help is printed and nothing is changed.

Actions (may be combined):
  --install              Install the selected environment. The default Pixi
                         backend is rootless; apt is an explicit alternative.
  --import               Import pinned external repositories and create only
                         the two approved local-package symlinks.
  --build                Resolve dependencies for the selected backend and
                         build only the MuJoCo/Panda overlay packages.
  --all                  Equivalent to --install --import --build.

Configuration:
  --workspace PATH       Overlay workspace (default: ${DEFAULT_WORKSPACE}).
  --source-root PATH     Source checkout containing the two local packages
                         (default: ${DEFAULT_SOURCE_ROOT}).
  --pixi-project PATH    Separate Pixi project containing MuJoCo 3.10.0
                         (default: ${DEFAULT_PIXI_PROJECT}).
  --backend pixi|apt     Environment backend (default: pixi). The apt backend
                         still uses the separate Pixi MuJoCo 3.10.0 prefix.
  -h, --help             Show this help.

Environment equivalents:
  FRANKA_HUMBLE_MUJOCO_WS
  FRANKA_ROS2_SOURCE_ROOT
  FRANKA_MUJOCO_PIXI_PROJECT

The apt backend uses sudo for repository setup. Pixi installs under the user
account. Neither backend reads or writes /home/noamonti/warp/.venv.
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

while (($# > 0)); do
  case "$1" in
    --install)
      DO_INSTALL=true
      ;;
    --import)
      DO_IMPORT=true
      ;;
    --build)
      DO_BUILD=true
      ;;
    --all)
      DO_INSTALL=true
      DO_IMPORT=true
      DO_BUILD=true
      ;;
    --workspace)
      (($# >= 2)) || die "--workspace requires a path"
      WORKSPACE="$2"
      shift
      ;;
    --source-root)
      (($# >= 2)) || die "--source-root requires a path"
      SOURCE_ROOT="$2"
      shift
      ;;
    --pixi-project)
      (($# >= 2)) || die "--pixi-project requires a path"
      PIXI_PROJECT="$2"
      shift
      ;;
    --backend)
      (($# >= 2)) || die "--backend requires pixi or apt"
      BACKEND="$2"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1 (use --help)"
      ;;
  esac
  shift
done

[[ "${BACKEND}" == "pixi" || "${BACKEND}" == "apt" ]] ||
  die "--backend must be pixi or apt (found ${BACKEND})"

if [[ "${DO_INSTALL}" == false && "${DO_IMPORT}" == false && "${DO_BUILD}" == false ]]; then
  usage
  exit 0
fi

[[ -r /etc/os-release ]] || die "cannot identify the operating system"
# shellcheck disable=SC1091
source /etc/os-release
[[ "${ID:-}" == "ubuntu" && "${VERSION_CODENAME:-}" == "${JAMMY_CODENAME}" ]] ||
  die "this setup is restricted to Ubuntu 22.04 Jammy (found ${ID:-unknown} ${VERSION_CODENAME:-unknown})"

WORKSPACE="$(realpath -m -- "${WORKSPACE}")"
SOURCE_ROOT="$(realpath -m -- "${SOURCE_ROOT}")"
PIXI_PROJECT="$(realpath -m -- "${PIXI_PROJECT}")"
[[ "${WORKSPACE}" != "${REPOSITORY_ROOT}" ]] ||
  die "workspace must be separate from the source checkout"
[[ "${WORKSPACE}" != "${SOURCE_ROOT}" ]] ||
  die "workspace must be separate from the source checkout"
[[ "${PIXI_PROJECT}" != "${REPOSITORY_ROOT}" && "${PIXI_PROJECT}" != "${SOURCE_ROOT}" ]] ||
  die "Pixi project must be separate from the source checkout"
[[ -f "${REPOS_FILE}" ]] || die "missing pinned repository manifest: ${REPOS_FILE}"

PIXI_BIN=""
MUJOCO_PREFIX="${PIXI_PROJECT}/.pixi/envs/default"

run_privileged() {
  if ((EUID == 0)); then
    "$@"
  else
    command -v sudo >/dev/null 2>&1 || die "sudo is required for --install/rosdep apt operations"
    sudo "$@"
  fi
}

assert_no_jazzy_environment() {
  local variable value
  if [[ -n "${ROS_DISTRO:-}" && "${ROS_DISTRO}" != "${ROS_DISTRO_TARGET}" ]]; then
    die "ROS_DISTRO=${ROS_DISTRO}; start a clean shell instead of overlaying Humble on another ROS distribution"
  fi
  for variable in AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH; do
    value="${!variable:-}"
    if [[ "${value,,}" == *jazzy* ]]; then
      die "${variable} contains a Jazzy path; start a clean shell before importing/building Humble"
    fi
  done
}

find_or_install_pixi() {
  local candidate
  for candidate in \
    "${PIXI_HOME:-${HOME}/.pixi}/bin/pixi" \
    "${HOME}/.pixi/bin/pixi" \
    "$(command -v pixi 2>/dev/null || true)"; do
    if [[ -n "${candidate}" && -x "${candidate}" ]]; then
      PIXI_BIN="${candidate}"
      break
    fi
  done

  if [[ -z "${PIXI_BIN}" ]]; then
    printf 'Installing rootless Pixi %s under %s.\n' \
      "${PIXI_VERSION}" "${PIXI_HOME:-${HOME}/.pixi}"
    curl --fail --location --silent --show-error https://pixi.sh/install.sh |
      PIXI_VERSION="${PIXI_VERSION}" \
      PIXI_HOME="${PIXI_HOME:-${HOME}/.pixi}" \
      PIXI_NO_PATH_UPDATE=1 \
      sh
    PIXI_BIN="${PIXI_HOME:-${HOME}/.pixi}/bin/pixi"
  fi

  [[ -x "${PIXI_BIN}" ]] || die "Pixi installation did not produce ${PIXI_BIN}"
  [[ "$("${PIXI_BIN}" --version)" == "pixi ${PIXI_VERSION}" ]] ||
    die "Pixi must be version ${PIXI_VERSION}; found $("${PIXI_BIN}" --version)"
}

write_pixi_manifest_if_missing() {
  local manifest="${PIXI_PROJECT}/pixi.toml"
  if [[ -e "${manifest}" ]]; then
    return 0
  fi
  mkdir -p -- "${PIXI_PROJECT}"
  cat >"${manifest}" <<'EOF'
[workspace]
channels = ["https://prefix.dev/robostack-humble", "https://prefix.dev/conda-forge"]
name = "franka_humble_mujoco_pixi"
platforms = ["linux-64"]
version = "0.1.0"

[dependencies]
ros-humble-ros-base = ">=0.10.0,<0.11"
ros-humble-ros2-control = ">=2.54.0,<3"
ros-humble-ros2-controllers = ">=2.53.1,<3"
ros-humble-ros2-control-cmake = ">=0.2.1,<0.3"
ros-humble-ament-cmake-python = "*"
ros-humble-ament-cmake-pytest = "*"
ros-humble-backward-ros = "*"
ros-humble-realtime-tools = "*"
ros-humble-pluginlib = "*"
ros-humble-xacro = ">=2.1.1,<3"
ros-humble-robot-state-publisher = ">=3.0.3,<4"
ros-humble-cv-bridge = ">=3.2.1,<4"
ros-humble-image-transport = ">=3.1.12,<4"
ros-humble-message-filters = ">=4.3.16,<5"
ros-humble-kdl-parser = ">=2.6.4,<3"
ros-humble-tf2-ros = ">=0.25.20,<0.26"
ros-humble-launch-ros = ">=0.19.13,<0.20"
ros-humble-control-msgs = ">=4.9.0,<5"
ros-humble-franka-msgs = ">=1.0.0,<2"
colcon-common-extensions = ">=0.3.0,<0.4"
compilers = ">=2.0.0,<3"
cmake = ">=4.2.3,<5"
make = ">=4.4.1,<5"
ninja = ">=1.13.2,<2"
patchelf = ">=0.18.0,<0.19"
pkg-config = ">=0.29.2,<0.30"
rosdep = ">=0.26.0,<0.27"
vcstool = ">=1.1.7,<2"
python = ">=3.12.13,<3.13"
numpy = ">=2.5.2,<3"
pytest = ">=9.1.1,<10"
onnxruntime = ">=1.28.0,<2"
pyyaml = ">=6.0.3,<7"
libgl-devel = ">=1.7.0,<2"
mujoco = "3.10.0.*"
EOF
}

validate_mujoco_prefix() {
  [[ -f "${PIXI_PROJECT}/pixi.toml" ]] ||
    die "missing Pixi manifest: ${PIXI_PROJECT}/pixi.toml"
  grep -Eq '^[[:space:]]*mujoco[[:space:]]*=[[:space:]]*"3\.10\.0\.\*"[[:space:]]*$' \
    "${PIXI_PROJECT}/pixi.toml" ||
    die "Pixi manifest must pin mujoco = \"3.10.0.*\""
  [[ -f "${MUJOCO_PREFIX}/lib/cmake/mujoco/mujocoConfig.cmake" ]] ||
    die "MuJoCo CMake package is missing from ${MUJOCO_PREFIX}"
  [[ -f "${MUJOCO_PREFIX}/lib/libmujoco.so.${MUJOCO_VERSION}" ]] ||
    die "MuJoCo ${MUJOCO_VERSION} library is missing from ${MUJOCO_PREFIX}"
  [[ -f "${MUJOCO_PREFIX}/include/simulate/simulate.h" ]] ||
    die "MuJoCo simulate headers are missing from ${MUJOCO_PREFIX}"
}

install_pixi_environment() {
  find_or_install_pixi
  write_pixi_manifest_if_missing
  if [[ -f "${PIXI_PROJECT}/pixi.lock" ]]; then
    "${PIXI_BIN}" install --manifest-path "${PIXI_PROJECT}/pixi.toml" --locked
  else
    "${PIXI_BIN}" install --manifest-path "${PIXI_PROJECT}/pixi.toml"
  fi
  validate_mujoco_prefix
}

create_python_environment() {
  local python_version
  python_version="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  [[ "${python_version}" == "3.10" ]] ||
    die "Jammy ONNX Runtime environment requires Python 3.10 (python3 is ${python_version})"

  if [[ ! -x "${WORKSPACE}/.venv/bin/python" ]]; then
    python3 -m venv --system-site-packages "${WORKSPACE}/.venv"
  fi

  "${WORKSPACE}/.venv/bin/python" -m pip install --upgrade \
    "pip==25.2" \
    "coloredlogs==15.0.1" \
    "flatbuffers==25.2.10" \
    "humanfriendly==10.0" \
    "mpmath==1.3.0" \
    "numpy==1.26.4" \
    "onnxruntime==${ONNXRUNTIME_VERSION}" \
    "packaging==25.0" \
    "protobuf==6.32.0" \
    "sympy==1.14.0"

  "${WORKSPACE}/.venv/bin/python" -c \
    'import onnxruntime as ort; assert ort.__version__ == "1.23.2"; print("ONNX Runtime", ort.__version__)'
}

install_dependencies() {
  local key_tmp
  readonly -a bootstrap_packages=(
    ca-certificates
    curl
    git
    gnupg
    lsb-release
    locales
    software-properties-common
  )
  readonly -a development_packages=(
    build-essential
    cmake
    libglfw3-dev
    ninja-build
    patchelf
    pkg-config
    python3-colcon-common-extensions
    python3-numpy
    python3-opencv
    python3-pip
    python3-pykdl
    python3-pytest
    python3-rosdep
    python3-vcstool
    python3-venv
    python3-yaml
  )
  readonly -a ros_packages=(
    ros-humble-ament-cmake-vendor-package
    ros-humble-control-msgs
    ros-humble-controller-manager
    ros-humble-cv-bridge
    ros-humble-gripper-controllers
    ros-humble-image-transport
    ros-humble-joint-state-broadcaster
    ros-humble-joint-trajectory-controller
    ros-humble-kdl-parser
    ros-humble-message-filters
    ros-humble-robot-state-publisher
    ros-humble-ros-base
    ros-humble-ros2-control
    ros-humble-ros2-controllers
    ros-humble-xacro
  )

  printf 'Installing apt dependencies (sudo may prompt for your password).\n'
  run_privileged apt-get update
  run_privileged apt-get install -y "${bootstrap_packages[@]}"
  run_privileged locale-gen en_US en_US.UTF-8
  run_privileged update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
  run_privileged add-apt-repository -y universe

  key_tmp="$(mktemp)"
  trap 'rm -f -- "${key_tmp:-}"' RETURN
  curl --fail --location --silent --show-error \
    "https://raw.githubusercontent.com/ros/rosdistro/master/ros.key" \
    --output "${key_tmp}"
  run_privileged install -m 0644 "${key_tmp}" /usr/share/keyrings/ros-archive-keyring.gpg
  printf 'deb [arch=%s signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu %s main\n' \
    "$(dpkg --print-architecture)" "${JAMMY_CODENAME}" |
    run_privileged tee /etc/apt/sources.list.d/ros2.list >/dev/null
  rm -f -- "${key_tmp}"
  trap - RETURN

  run_privileged apt-get update
  run_privileged apt-get install -y "${development_packages[@]}" "${ros_packages[@]}"

  if [[ ! -e /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
    run_privileged rosdep init
  fi
  rosdep update --rosdistro "${ROS_DISTRO_TARGET}"

  mkdir -p -- "${WORKSPACE}"
  create_python_environment
}

link_local_package() {
  local package_name="$1"
  local source_path="${SOURCE_ROOT}/${package_name}"
  local link_path="${WORKSPACE}/src/${package_name}"
  local current_target

  [[ -d "${source_path}" && -f "${source_path}/package.xml" ]] ||
    die "required local package is missing: ${source_path} (override with --source-root)"

  if [[ -L "${link_path}" ]]; then
    current_target="$(readlink -f -- "${link_path}")"
    [[ "${current_target}" == "$(readlink -f -- "${source_path}")" ]] ||
      die "${link_path} points to ${current_target}, not ${source_path}"
    return
  fi
  [[ ! -e "${link_path}" ]] || die "refusing to replace existing path: ${link_path}"
  ln -s -- "${source_path}" "${link_path}"
}

validate_external_checkout() {
  local directory="$1"
  local expected_commit="$2"
  local actual_commit
  [[ -d "${directory}/.git" ]] || die "expected git checkout is missing: ${directory}"
  actual_commit="$(git -C "${directory}" rev-parse HEAD)"
  [[ "${actual_commit}" == "${expected_commit}" ]] ||
    die "${directory} is at ${actual_commit}; expected pinned commit ${expected_commit}"
}

import_sources() {
  local mujoco_control_dir="${WORKSPACE}/src/mujoco_ros2_control"
  local mujoco_vendor_dir="${WORKSPACE}/src/mujoco_vendor"

  assert_no_jazzy_environment
  activate_selected_environment
  command -v vcs >/dev/null 2>&1 || die "vcstool is missing; run --install first"
  mkdir -p -- "${WORKSPACE}/src"

  if [[ ! -e "${mujoco_control_dir}" && ! -e "${mujoco_vendor_dir}" ]]; then
    vcs import --input "${REPOS_FILE}" "${WORKSPACE}/src"
  elif [[ ! -e "${mujoco_control_dir}" || ! -e "${mujoco_vendor_dir}" ]]; then
    die "partial external import detected; remove the incomplete external checkout(s) and rerun --import"
  fi

  validate_external_checkout "${mujoco_control_dir}" "${MUJOCO_ROS2_CONTROL_COMMIT}"
  validate_external_checkout "${mujoco_vendor_dir}" "${MUJOCO_VENDOR_COMMIT}"
  link_local_package "franka_emika_panda"
  link_local_package "franka_mujoco_bringup"
}

source_humble_environment() {
  [[ -r /opt/ros/humble/setup.bash ]] || die "ROS 2 Humble is not installed; run --install first"
  assert_no_jazzy_environment
  # ROS and venv setup scripts are not written for nounset.
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
  if [[ -r "${WORKSPACE}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${WORKSPACE}/.venv/bin/activate"
  fi
  set -u
  validate_mujoco_prefix
  export CMAKE_PREFIX_PATH="${MUJOCO_PREFIX}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
  export LD_LIBRARY_PATH="${MUJOCO_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export PKG_CONFIG_PATH="${MUJOCO_PREFIX}/lib/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"
  export CMAKE_GENERATOR="Ninja"
  [[ "${ROS_DISTRO:-}" == "${ROS_DISTRO_TARGET}" ]] || die "failed to activate ROS 2 Humble"
}

source_pixi_environment() {
  assert_no_jazzy_environment
  validate_mujoco_prefix
  export CONDA_PREFIX="${MUJOCO_PREFIX}"
  export PATH="${MUJOCO_PREFIX}/bin:${PATH}"
  export CMAKE_PREFIX_PATH="${MUJOCO_PREFIX}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
  export LD_LIBRARY_PATH="${MUJOCO_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export PKG_CONFIG_PATH="${MUJOCO_PREFIX}/lib/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"
  export CMAKE_GENERATOR="Ninja"
  set +u
  # shellcheck disable=SC1091
  source "${MUJOCO_PREFIX}/setup.bash"
  set -u
  [[ "${ROS_DISTRO:-}" == "${ROS_DISTRO_TARGET}" ]] ||
    die "Pixi project does not provide ROS 2 Humble"
}

activate_selected_environment() {
  if [[ "${BACKEND}" == "pixi" ]]; then
    source_pixi_environment
  else
    source_humble_environment
  fi
}

build_overlay() {
  readonly -a selected_source_paths=(
    "${WORKSPACE}/src/mujoco_vendor"
    "${WORKSPACE}/src/mujoco_ros2_control"
    "${WORKSPACE}/src/franka_emika_panda"
    "${WORKSPACE}/src/franka_mujoco_bringup"
  )
  readonly -a selected_packages=(
    mujoco_ros2_control_msgs
    mujoco_3d_lidar
    mujoco_ros2_control_plugins
    mujoco_ros2_control
    franka_emika_panda
    franka_mujoco_bringup
  )
  local path
  local compatibility_include="${WORKSPACE}/compat/mujoco-${MUJOCO_VERSION}/include"

  for path in "${selected_source_paths[@]}"; do
    [[ -e "${path}" ]] || die "source is missing: ${path}; run --import first"
  done
  validate_external_checkout \
    "${WORKSPACE}/src/mujoco_ros2_control" "${MUJOCO_ROS2_CONTROL_COMMIT}"
  validate_external_checkout \
    "${WORKSPACE}/src/mujoco_vendor" "${MUJOCO_VENDOR_COMMIT}"
  activate_selected_environment
  command -v rosdep >/dev/null 2>&1 || die "rosdep is missing; run --install first"
  command -v colcon >/dev/null 2>&1 || die "colcon is missing; run --install first"

  if [[ "${BACKEND}" == "apt" ]]; then
    rosdep install \
      --from-paths "${selected_source_paths[@]}" \
      --ignore-src \
      --rosdistro "${ROS_DISTRO_TARGET}" \
      --as-root apt:true \
      -y
  else
    printf 'Pixi backend: dependencies come from the locked rootless environment; skipping apt rosdep installation.\n'
  fi

  # MuJoCo 3.10 folded the legacy mjtnum.h declarations into mjtype.h, while
  # the pinned optional lidar extension still includes the old public header.
  # Keep the source checkouts immutable and provide the forwarding header only
  # in this generated build workspace.
  mkdir -p -- "${compatibility_include}/mujoco"
  cat >"${compatibility_include}/mujoco/mjtnum.h" <<'EOF'
#ifndef FRANKA_HUMBLE_MUJOCO_COMPAT_MJTNUM_H_
#define FRANKA_HUMBLE_MUJOCO_COMPAT_MJTNUM_H_
#include <mujoco/mjtype.h>
#endif
EOF

  # NEVER_VENDOR installs only a forwarding ament package. The pinned
  # mujoco_ros2_control CMake derives MuJoCo's include root from that package's
  # install prefix, so expose the external Pixi prefix there without copying it.
  colcon --log-base "${WORKSPACE}/log" build \
    --base-paths "${WORKSPACE}/src/mujoco_vendor" \
    --build-base "${WORKSPACE}/build" \
    --install-base "${WORKSPACE}/install" \
    --symlink-install \
    --cmake-clean-cache \
    --packages-select mujoco_vendor \
    --cmake-args \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTING=OFF \
      -DAMENT_VENDOR_POLICY=NEVER_VENDOR \
      "-Dmujoco_DIR=${MUJOCO_PREFIX}/lib/cmake/mujoco"

  mkdir -p -- "${WORKSPACE}/install/mujoco_vendor/opt"
  if [[ -L "${WORKSPACE}/install/mujoco_vendor/opt/mujoco_vendor" ]]; then
    [[ "$(readlink -f -- "${WORKSPACE}/install/mujoco_vendor/opt/mujoco_vendor")" == "${MUJOCO_PREFIX}" ]] ||
      die "existing MuJoCo forwarding link points to the wrong prefix"
  elif [[ -e "${WORKSPACE}/install/mujoco_vendor/opt/mujoco_vendor" ]]; then
    die "refusing to replace existing MuJoCo vendor payload"
  else
    ln -s -- "${MUJOCO_PREFIX}" "${WORKSPACE}/install/mujoco_vendor/opt/mujoco_vendor"
  fi

  set +u
  # shellcheck disable=SC1091
  source "${WORKSPACE}/install/mujoco_vendor/share/mujoco_vendor/local_setup.bash"
  set -u

  colcon --log-base "${WORKSPACE}/log" build \
    --base-paths "${selected_source_paths[@]}" \
    --build-base "${WORKSPACE}/build" \
    --install-base "${WORKSPACE}/install" \
    --symlink-install \
    --cmake-clean-cache \
    --packages-select "${selected_packages[@]}" \
    --cmake-args \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTING=OFF \
      -DAMENT_VENDOR_POLICY=NEVER_VENDOR \
      "-Dmujoco_DIR=${MUJOCO_PREFIX}/lib/cmake/mujoco" \
      "-DCMAKE_C_FLAGS=-I${compatibility_include} -I${MUJOCO_PREFIX}/include" \
      "-DCMAKE_CXX_FLAGS=-I${compatibility_include} -I${MUJOCO_PREFIX}/include"
}

mkdir -p -- "${WORKSPACE}"

if [[ "${DO_INSTALL}" == true ]]; then
  if [[ "${BACKEND}" == "apt" ]]; then
    install_dependencies
  fi
  install_pixi_environment
fi
if [[ "${DO_IMPORT}" == true ]]; then
  import_sources
fi
if [[ "${DO_BUILD}" == true ]]; then
  build_overlay
fi

printf '\nOverlay preparation complete.\n'
printf 'Activate it in each clean shell with:\n'
if [[ "${BACKEND}" == "pixi" ]]; then
  printf '  export CONDA_PREFIX=%q\n' "${MUJOCO_PREFIX}"
  printf '  export PATH=%q/bin:"$PATH"\n' "${MUJOCO_PREFIX}"
  printf '  export LD_LIBRARY_PATH=%q/lib:"${LD_LIBRARY_PATH:-}"\n' "${MUJOCO_PREFIX}"
  printf '  source %q/setup.bash\n' "${MUJOCO_PREFIX}"
else
  printf '  source /opt/ros/humble/setup.bash\n'
  printf '  source %q/.venv/bin/activate\n' "${WORKSPACE}"
  printf '  export CMAKE_PREFIX_PATH=%q:"${CMAKE_PREFIX_PATH:-}"\n' "${MUJOCO_PREFIX}"
  printf '  export LD_LIBRARY_PATH=%q/lib:"${LD_LIBRARY_PATH:-}"\n' "${MUJOCO_PREFIX}"
fi
printf '  source %q/install/setup.bash\n' "${WORKSPACE}"
