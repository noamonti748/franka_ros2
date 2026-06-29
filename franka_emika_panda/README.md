# Franka Emika Panda Description (MJCF)

> [!IMPORTANT]
> Requires MuJoCo 2.3.3 or later.

## Changelog

See [CHANGELOG.md](./CHANGELOG.md) for a full history of changes.

## Overview

This package contains a simplified robot description (MJCF) of the [Franka Emika
Panda](https://www.franka.de/) developed by [Franka
Emika](https://www.franka.de/company). It is derived from the [publicly
available URDF
description](https://github.com/frankaemika/franka_ros/tree/develop/franka_description).

Mesh files under [`assets/`](./assets) come from the
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda/assets)
via a **sparse partial clone** of that repository. Only
`mujoco_menagerie/franka_emika_panda/assets` is checked out — not the rest of
Menagerie.

After cloning this repo, fetch the meshes with:

```bash
./franka_emika_panda/scripts/init_menagerie_assets.sh
```

Do **not** use plain `git submodule update --init` for this dependency; the
script uses `--filter=blob:none --sparse` so clones stay small. `assets/` is a
symlink to `mujoco_menagerie/franka_emika_panda/assets` so MJCF and URDF paths
stay unchanged.

The Franka Research gripper is authored in [`hand.xml`](./hand.xml) (standalone MJCF: meshes, finger joints, tendon, equality, and gripper actuator). [`panda.xml`](./panda.xml) inlines the same kinematic tree under `link7` so that body, joint, and actuator names stay unchanged for existing scenes and scripts. The [`franka_fr3_v2_1`](../franka_fr3_v2_1/README.md) model ships its **own copy** of that MJCF and meshes and attaches it with [`attach`](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-attach) under the FR3 flange.

<p float="left">
  <img src="panda.png" width="400">
</p>

## URDF → MJCF derivation steps

1. Converted the DAE [mesh
   files](https://github.com/frankaemika/franka_ros/tree/develop/franka_description/meshes/visual)
   to OBJ format using [Blender](https://www.blender.org/).
2. Processed `.obj` files with [`obj2mjcf`](https://github.com/kevinzakka/obj2mjcf).
3. Eliminated the perfectly flat `link0_6` from the resulting submeshes created for `link0`.
4. Created a convex decomposition of the STL collision [mesh
   file](https://github.com/frankaemika/franka_ros/tree/develop/franka_description/meshes/collision)
   for `link5` using [V-HACD](https://github.com/kmammou/v-hacd).
5. Added `<mujoco> <compiler discardvisual="false"/> </mujoco>` to the
   [URDF](https://github.com/frankaemika/franka_ros/tree/develop/franka_description/robots)'s
   `<robot>` clause in order to preserve visual geometries.
6. Loaded the URDF into MuJoCo and saved a corresponding MJCF.
7. Matched inertial parameters with [inertial.yaml](
   https://github.com/frankaemika/franka_ros/blob/develop/franka_description/robots/common/inertial.yaml).
8. Added a tracking light to the base.
9. Manually edited the MJCF to extract common properties into the `<default>` section.
10. Added `<exclude>` clauses to prevent collisions between `link7` and `link8`.
11. Manually designed collision geoms for the fingertips.
12. Added position-controlled actuators for the arm.
13. Added an equality constraint so that the left finger mimics the position of the right finger.
14. Added a tendon to split the force equally between both fingers and a
    position actuator acting on this tendon.
15. Added `scene.xml` which includes the robot, with a textured groundplane, skybox, and haze.

### MJX

A version of the Franka Emika Panda environment was created for MJX. Steps:

1. Added `mjx_panda.xml`, forked from `panda.xml`.
2. Added `mjx_scene.xml` and `mjx_single_cube.xml`, forked from `scene.xml`.
3. Gripper collision geometries were modified to contain less geoms. A capsule collision geom was added to the hand.
4. Solver parameters were tuned for performance.
5. Actuator `kp` and `kv` were reduced for more stable simulation.
6. Added a `site` to the gripper.
7. Removed tendon and added position actuator for the gripper. Changed gripper `ctrlrange`.

## License

This model is released under an [Apache-2.0 License](LICENSE).
