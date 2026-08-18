#!/usr/bin/env bash
# Sparse partial clone: only mujoco_menagerie/franka_emika_panda/assets on disk.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pkg_dir="$(dirname "${script_dir}")"
repo_root="$(git -C "${pkg_dir}" rev-parse --show-toplevel)"

submodule_rel="franka_emika_panda/mujoco_menagerie"
submodule_path="${repo_root}/${submodule_rel}"
assets_link="${repo_root}/franka_emika_panda/assets"
sparse_path="franka_emika_panda/assets"
url="https://github.com/google-deepmind/mujoco_menagerie.git"

pinned_sha="$(git -C "${repo_root}" ls-tree HEAD "${submodule_rel}" 2>/dev/null | awk '{print $3}')"

is_submodule_ready() {
  [[ -f "${submodule_path}/.git" || -d "${submodule_path}/.git" ]] &&
    [[ -f "${submodule_path}/${sparse_path}/link1.obj" ]]
}

sparse_checkout_assets() {
  git -C "${submodule_path}" sparse-checkout init --no-cone
  git -C "${submodule_path}" sparse-checkout set "${sparse_path}/"
  git -C "${submodule_path}" read-tree -mu HEAD
}

clone_sparse_submodule() {
  rm -rf "${submodule_path}"
  git -C "${repo_root}" submodule init "${submodule_rel}"

  git -C "${repo_root}" clone \
    --filter=blob:none \
    --sparse \
    --depth 1 \
    --no-checkout \
    "${url}" "${submodule_rel}"

  git -C "${submodule_path}" sparse-checkout init --no-cone
  git -C "${submodule_path}" sparse-checkout set "${sparse_path}/"

  if [[ -n "${pinned_sha}" ]]; then
    git -C "${submodule_path}" fetch --depth 1 origin "${pinned_sha}"
    git -C "${submodule_path}" checkout "${pinned_sha}"
  else
    git -C "${submodule_path}" checkout main
  fi

  sparse_checkout_assets
  git -C "${repo_root}" submodule absorbgitdirs "${submodule_rel}"
}

if is_submodule_ready; then
  sparse_checkout_assets
else
  clone_sparse_submodule
fi

ln -sfn mujoco_menagerie/franka_emika_panda/assets "${assets_link}"

echo "Menagerie sparse checkout ready: ${submodule_rel}/${sparse_path}/"
