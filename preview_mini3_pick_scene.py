"""Preview and manually position Mini3's extended arms and grippers in MuJoCo.

This is a joint/scene debugging tool, with an optional pelvis support fixture.
It does not run a learned balancing or autonomous pick-and-place policy.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from queue import SimpleQueue
import sys
import time


ROOT = Path(__file__).resolve().parent
DEFAULT_SCENE = ROOT / "any4hdmi/assets/robots/mini3_mjlab/scene_pick_place.xml"
CAMERA_VIEWS = ("overview", "head_rgb", "left_gripper_rgb", "right_gripper_rgb")
# GLFW F8--F11. Digits toggle MuJoCo geom groups; F7 toggles labels.
CAMERA_SHORTCUTS = dict(zip((297, 298, 299, 300), CAMERA_VIEWS))


def setup_paths() -> None:
    for directory in ("sim2real", "mimic-lite", "active-adaptation", ".cache/mimiclite_sim2sim/deps"):
        sys.path.insert(0, str(ROOT / directory))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/mimiclite_sim2sim/matplotlib"))


class SceneControls:
    """Map the original 21 motor joints by name, separately from gripper joints."""

    def __init__(self, model, data, *, support: bool):
        import numpy as np
        import active_adaptation as aa
        try:
            aa.get_backend()
        except RuntimeError:
            aa.set_backend("mjlab")
        from mimic_lite.assets.mini3 import MINI3_STIFFNESS, MINI3_DAMPING
        from sim2real.config.robots.mini3 import MINI3_CFG
        from any4hdmi.utils.mini3_real_motor import Mini3RealMotorModel

        self.model, self.data = model, data
        self.names = list(MINI3_CFG.joint_names)
        self.joint_ids = np.array([model.joint(name).id for name in self.names])
        self.qpos_ids = model.jnt_qposadr[self.joint_ids]
        self.qvel_ids = model.jnt_dofadr[self.joint_ids]
        self.motor_ids = np.array([model.actuator(f"{name}_ctrl").id for name in self.names])
        self.kp = np.array([MINI3_STIFFNESS[name] for name in self.names])
        self.kd = np.array([MINI3_DAMPING[name] for name in self.names])
        self.limits = np.array([MINI3_CFG.joint_effort_limit[name] for name in self.names])
        for index, limit in zip(self.motor_ids, self.limits):
            model.actuator_ctrllimited[index] = True
            model.actuator_forcelimited[index] = True
            model.actuator_ctrlrange[index] = [-limit, limit]
            model.actuator_forcerange[index] = [-limit, limit]
        config = MINI3_CFG.real_motor
        self.motor = Mini3RealMotorModel(
            tuple(self.names), self.kp, self.kd, self.limits, dt=model.opt.timestep,
            response_enabled=config.torque_response_enabled,
            tn_enabled=config.tn_torque_limit_enabled,
            tn_limit_after_response=config.tn_limit_after_response,
            kt_enabled=config.kt_output_model_enabled,
            response_kp=config.torque_response_kp, response_ki=config.torque_response_ki,
            response_plant_tau_s=config.torque_response_plant_tau_s,
            response_delay_steps=config.torque_response_delay_steps,
            ankle_motor_torque_limit=config.ankle_motor_torque_limit,
        )
        self.support_id = model.equality("base_support_weld").id
        self.mocap_id = int(model.body("base_support").mocapid[0])
        self.support = support
        self.selected = self.names.index("left_shoulder_pitch_joint")
        self.reset()

    def reset(self) -> None:
        import mujoco
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        self.data.eq_active[self.support_id] = self.support
        self.targets = self.data.qpos[self.qpos_ids].copy()
        self.motor.reset()
        mujoco.mj_forward(self.model, self.data)

    def step(self) -> None:
        import mujoco
        import numpy as np
        torque = self.motor.compute(self.targets, self.data.qpos[self.qpos_ids],
                                    self.data.qvel[self.qvel_ids])
        if not np.isfinite(torque).all():
            raise RuntimeError("Motor torque is not finite")
        self.data.ctrl[self.motor_ids] = np.clip(torque, -self.limits, self.limits)
        mujoco.mj_step(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("MuJoCo state is not finite")

    def key(self, key: int) -> None:
        import mujoco
        import numpy as np
        if key in (ord("["), ord("]"), 265, 264):
            if key in (ord("["), ord("]")):
                self.selected = (self.selected + (1 if key == ord("]") else -1)) % len(self.names)
            else:
                limits = self.model.jnt_range[self.joint_ids[self.selected]]
                self.targets[self.selected] = np.clip(
                    self.targets[self.selected] + (0.05 if key == 265 else -0.05), *limits
                )
            print(f"Joint {self.selected + 1}/21: {self.names[self.selected]} = {self.targets[self.selected]:+.3f} rad")
        elif key in (ord("Z"), ord("X"), ord("C"), ord("V")):
            side = "left" if key in (ord("Z"), ord("X")) else "right"
            opening = 0.035 if key in (ord("Z"), ord("C")) else 0.0
            self.data.ctrl[self.model.actuator(f"{side}_gripper_ctrl").id] = opening
            print(f"{side} gripper: {'open' if opening else 'close'}")
        elif key == ord("B"):
            self.support = not self.support
            if self.support:
                root = self.model.body("base_link").id
                self.data.mocap_pos[self.mocap_id] = self.data.xpos[root]
                self.data.mocap_quat[self.mocap_id] = self.data.xquat[root]
            self.data.eq_active[self.support_id] = self.support
            print(f"Pelvis support: {'ON' if self.support else 'OFF'}")
        elif key in map(ord, "WASDUJIK"):
            if not self.support:
                print("Press B to enable the pelvis support before moving it")
                return
            position = self.data.mocap_pos[self.mocap_id]
            if key in map(ord, "WASDUJ"):
                axis, delta = {
                    ord("W"): (0, 0.01), ord("S"): (0, -0.01),
                    ord("A"): (1, 0.01), ord("D"): (1, -0.01),
                    ord("U"): (2, 0.01), ord("J"): (2, -0.01),
                }[key]
                position[axis] += delta
                position[2] = np.clip(position[2], 0.1, 0.9)
            else:
                rotation, result = np.empty(4), np.empty(4)
                angle = math.radians(3 if key == ord("I") else -3)
                mujoco.mju_axisAngle2Quat(rotation, np.array([0., 1., 0.]), angle)
                mujoco.mju_mulQuat(result, self.data.mocap_quat[self.mocap_id], rotation)
                self.data.mocap_quat[self.mocap_id] = result
            print(f"Support target: {position.round(3)} m")


def close_viewer(viewer) -> None:
    viewer.close()
    simulate_ref = getattr(viewer, "_sim", None)
    deadline = time.monotonic() + 3
    if callable(simulate_ref):
        while simulate_ref() is not None and time.monotonic() < deadline:
            time.sleep(0.01)


def configure_camera(camera, model, data, name: str) -> None:
    """Select a body-mounted camera or restore the free scene overview."""
    import mujoco

    if name not in CAMERA_VIEWS:
        raise ValueError(f"Unknown camera view: {name}")
    if name != "overview":
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if camera_id < 0:
            raise ValueError(
                f"Camera {name!r} is missing. Regenerate mini3_gripper.xml and scene_pick_place.xml."
            )
        camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera.fixedcamid = camera_id
        return
    base_id = model.body("base_link").id
    center = data.xpos[base_id] + 0.25 * data.xmat[base_id].reshape(3, 3)[:, 0]
    center[2] = 0.3
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.fixedcamid = -1
    camera.lookat[:] = center
    camera.distance = 1.6
    camera.azimuth = 135 + math.degrees(math.atan2(data.xmat[base_id][3], data.xmat[base_id][0]))
    camera.elevation = -40


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--duration", type=float, default=0, help="Simulated seconds; 0 = until window closes")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-support", action="store_true", help="Disable the preview's pelvis support")
    parser.add_argument("--start-paused", action="store_true")
    parser.add_argument("--camera", choices=CAMERA_VIEWS, default="overview", help="Initial viewer/screenshot camera")
    parser.add_argument("--screenshot", type=Path, help="Save an image of the initialized scene and exit")
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration < 0:
        parser.error("duration must be finite and non-negative")
    if args.headless and not args.screenshot and (not args.duration or args.start_paused):
        parser.error("headless playback requires --duration > 0 and cannot start paused")
    setup_paths()
    import torch  # Import before graphics libraries on platforms with limited static TLS.
    import mujoco
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    controls = SceneControls(model, data, support=not args.no_support)
    if args.screenshot:
        from PIL import Image
        camera = mujoco.MjvCamera()
        configure_camera(camera, model, data, args.camera)
        option = mujoco.MjvOption()
        option.geomgroup[1] = 0  # Hide collision proxies; retain their physical contacts.
        width, height = (1280, 960) if args.camera == "overview" else (640, 480)
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            renderer.update_scene(data, camera=camera, scene_option=option)
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(renderer.render()).save(args.screenshot)
        print(f"Saved {args.screenshot}")
        return

    events: SimpleQueue[int] = SimpleQueue()
    viewer = None
    paused, total_steps = args.start_paused, 0
    limit = math.ceil(args.duration / model.opt.timestep) if args.duration else None
    decimation = max(1, round(0.02 / model.opt.timestep))
    print("Scene: 3 ground cubes, open basket, two 100 mm extensions and actuated grippers.")
    print("Original joints: Mini3 real motor model. Grippers: generic position servos.")
    print(f"Pelvis support: {'ON (manipulation fixture)' if controls.support else 'OFF'}")
    print("Keys: [/] select joint, Up/Down change target; Z/X left open/close; C/V right open/close.")
    print("B toggle support; W/S X, A/D Y, U/J height, I/K pitch; P pause, R reset, Esc exit.")
    print("Views: F8 overview, F9 head RGB (45 deg down), F10 left gripper RGB, F11 right gripper RGB.")
    print("Lowering the pelvis requires matching leg joint adjustments to keep the feet above the floor.")
    try:
        if not args.headless:
            viewer = mujoco.viewer.launch_passive(model, data, key_callback=events.put)
            with viewer.lock():
                configure_camera(viewer.cam, model, data, args.camera)
                viewer.opt.geomgroup[1] = 0
        while viewer is None or viewer.is_running():
            started = time.perf_counter()
            quit_requested = False
            while not events.empty():
                key = events.get()
                if key == ord("P"):
                    paused = not paused
                elif key == ord("R"):
                    controls.reset()
                elif key == 256:
                    quit_requested = True
                elif key in CAMERA_SHORTCUTS and viewer is not None:
                    name = CAMERA_SHORTCUTS[key]
                    with viewer.lock():
                        configure_camera(viewer.cam, model, data, name)
                    print(f"Camera: {name}")
                else:
                    controls.key(key)
            if quit_requested or (limit is not None and total_steps >= limit):
                break
            if not paused:
                for _ in range(min(decimation, limit - total_steps) if limit else decimation):
                    controls.step()
                    total_steps += 1
            if viewer is not None:
                viewer.sync()
                time.sleep(max(0, 0.02 - (time.perf_counter() - started)))
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            close_viewer(viewer)
        print(f"Finished: {total_steps} physics steps, {total_steps * model.opt.timestep:.3f} simulated seconds")


if __name__ == "__main__":
    main()
