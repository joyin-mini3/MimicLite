# Mini3 dual-gripper pick scene

Robot: [`mini3_gripper.xml`](../any4hdmi/assets/robots/mini3_mjlab/mini3_gripper.xml).
Complete scene: [`scene_pick_place.xml`](../any4hdmi/assets/robots/mini3_mjlab/scene_pick_place.xml).
The original `mini3.xml` and training asset remain unchanged.

## Robot changes

Each forearm receives a **100 mm rigid extension**, followed by a parallel
two-finger gripper. The extension starts at the original elbow-tip collision
sphere center: `(0.107, -0.018, 0.0085)` on the left and
`(0.107, 0.018, 0.0085)` on the right, along the elbow body's local **+X**.
The 100 mm excludes the gripper; its TCP lies another 68 mm beyond the mount.

| Item | Setting |
| --- | --- |
| Maximum inner opening | 70 mm |
| Finger length | 70 mm |
| Added mass per arm | 0.20 kg: extension 0.06, palm 0.08, two fingers 0.03 each |
| Original joints/actuators | Original 21 joint parameters and 21 motor names/order retained |
| Added degrees of freedom | Two slide joints per hand, synchronized by a joint equality |
| Added inputs | One position actuator per hand; 23 total actuator inputs |
| Collision and inertia | Colliding extension/palm/fingers; added inertia computed from geometry and mass |

The gripper is a simplified simulation model. Its added masses, inertias, and
servo parameters are initial debugging values, not identified hardware parameters.

Control interface:

```python
# Resolve controls by name; gripper joints are not simply appended to qpos.
data.ctrl[model.actuator("left_gripper_ctrl").id] = 0.035  # Left: 70 mm gap
data.ctrl[model.actuator("right_gripper_ctrl").id] = 0.0   # Right: close
```

Controls are in metres, range `[0, 0.035]`; inner opening is `2 * ctrl`.
TCP sites are `left_gripper_tcp` and `right_gripper_tcp`. Finger joints are
`{side}_gripper_finger_joint` and `{side}_gripper_follower_joint`.
The robot has `nq=32, nv=31, nu=23`; the scene has `nq=53, nv=49, nu=23`.

## RGB cameras

Three RGB cameras are fixed to the head and gripper palms and move with those links.

| Name | Parent link | Optical center in parent coordinates (m) | Direction |
| --- | --- | --- | --- |
| `head_rgb` | `head_link` | `(0.073, 0, 0.048)` | **45° down** from the head's forward +X direction |
| `left_gripper_rgb` | `left_gripper_palm` | `(0.020, 0, 0.040)` | Toward the left TCP/between the fingers, about 39.81° down |
| `right_gripper_rgb` | `right_gripper_palm` | `(0.020, 0, 0.040)` | Toward the right TCP/between the fingers, about 39.81° down |

All cameras use **640 × 480 RGB and an 80° vertical field of view**. The head
camera's optical axis in head coordinates is `(√0.5, 0, -√0.5)`. Its downward
angle is relative to the head, not fixed in the world as the robot moves.
Housings, lenses, and gripper camera brackets are visual only, with zero mass
and no collision. The extensions and grippers still add 200 g per side,
400 g total; camera hardware mass and inertia have not been added.

```bash
# Open the head RGB view directly
python preview_mini3_pick_scene.py --camera head_rgb
# Save 640×480 RGB frames
python preview_mini3_pick_scene.py --camera head_rgb \
  --screenshot outputs/mini3_pick_scene/head_rgb.png
python preview_mini3_pick_scene.py --camera left_gripper_rgb \
  --screenshot outputs/mini3_pick_scene/left_gripper_rgb.png
python preview_mini3_pick_scene.py --camera right_gripper_rgb \
  --screenshot outputs/mini3_pick_scene/right_gripper_rgb.png
```

Read an RGB array (`uint8`, shape `[480, 640, 3]`) in your own simulation loop:

```python
option = mujoco.MjvOption()
option.geomgroup[1] = 0  # Hide the original robot's collision proxies
with mujoco.Renderer(model, height=480, width=640) as renderer:
    renderer.update_scene(data, camera="head_rgb", scene_option=option)
    rgb = renderer.render()
```

## Object layout

The default pelvis position is `(0, 0, 0.46305)` and forward is world **+X**.
Props are placed relative to the initial horizontal pelvis position and yaw.
After initialization, their positions remain in the world rather than moving
with the robot.

| Object | Planar offset from initial pelvis (m) | Dimensions/physics |
| --- | --- | --- |
| Red `pick_cube_1` | `(0.28, 0.12)` | 4 cm sides, 40 g, free rigid body |
| Green `pick_cube_2` | `(0.33, 0)` | Same |
| Blue `pick_cube_3` | `(0.28, -0.12)` | Same |
| Basket `pick_basket` | `(0.52, 0)` | Interior 24 × 22 cm, outer height 12 cm, 1 cm floor/walls |

Cube centers start 2.05 cm above the ground and settle onto it. The basket is
fixed to the ground with a colliding bottom and four walls, open at the top.
Its target site is `pick_basket_target`.

## Preview and manual adjustment

From the repository root, use the configured MJLab Python environment:

```bash
source active-adaptation/venv/mjlab/.venv/bin/activate
python preview_mini3_pick_scene.py
```

| Key | Action |
| --- | --- |
| `[` / `]` | Select the previous/next original joint; name prints in the terminal |
| Up / Down | Change selected joint target by ±0.05 rad |
| `Z` / `X` | Open/close left gripper |
| `C` / `V` | Open/close right gripper |
| `B` | Toggle pelvis support |
| `W` / `S`, `A` / `D` | Move support target along world X/Y, 1 cm per press |
| `U` / `J` | Raise/lower support target, 1 cm per press |
| `I` / `K` | Pitch support target around its local Y axis, 3° per press |
| `F8` / `F9` / `F10` / `F11` | Overview / head RGB / left gripper RGB / right gripper RGB |
| `P` | Pause/resume physics |
| `R` | Reset robot, cubes, joint targets, and motor state |
| `Esc` | Close window |

The preview enables a **pelvis support fixture** for manual adjustments;
`--no-support` disables it. The XML itself has `base_support_weld` **inactive**,
so programs loading the scene directly get a free-floating robot. The original
21 joints use the Mini3 Real Motor model with joint position PD; the grippers
use the added ideal position servos. Original collision proxies are hidden in
the preview, while their physical contacts remain active.

Standing upright still does not allow ground reach. Coordinate leg, torso, and
arm posture; lowering only the pelvis presses the feet into the floor. This
entry point provides joint/scene debugging, not autonomous balance or picking.
The existing 21-joint MimicLite checkpoint runner cannot directly load this
expanded model. Policy integration must separate the original actions from
gripper controls by name and evaluate the changed reach and mass.

```bash
# Physics check without a window
python preview_mini3_pick_scene.py --headless --duration 5
# Initial-state image; needs an available OpenGL context
python preview_mini3_pick_scene.py --screenshot outputs/mini3_pick_scene/scene.png
```

## Regenerate and change initial heading

```bash
python any4hdmi/scripts/generate_mini3_gripper.py
python any4hdmi/scripts/generate_mini3_pick_scene.py
# A separate scene with props placed relative to this initial pose
python any4hdmi/scripts/generate_mini3_pick_scene.py \
  --x 1 --y 2 --yaw-deg 90 --output outputs/mini3_pick_scene/rotated_scene.xml
```

The scene is a complete copy of the generated robot with props added. Regenerate
the scene after changing the robot XML. Generated files reference the original
meshes through relative paths; include those meshes when copying to another machine.

Validation:

```bash
python -m unittest discover -s any4hdmi/tests -p 'test_mini3_gripper.py' -v
python -m unittest discover -s any4hdmi/tests -p 'test_mini3_pick_scene.py' -v
python -m unittest discover -s any4hdmi/tests -p 'test_mini3_rgb_cameras.py' -v
```

Physics tests cover each gripper holding a free cube under gravity and releasing
it, stable cubes on the ground, a free drop into the basket, containment by all
four walls, and layout consistency under changes in initial position and yaw.
Camera tests check orientation, motion with parent links, a clear optical ray,
and preservation of mass and inertia.
