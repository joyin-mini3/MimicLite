#!/usr/bin/env python3
"""Render a recorded Mini3 trajectory; no policy inference or physics stepping.

Example:
    python render_mini3_pick_carry.py --trajectory outputs/mini3_pick_carry_trial4/trajectory.npz \
        --output outputs/mini3_pick_carry_trial4/demo.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
DEFAULT_SCENE = ROOT / "any4hdmi/assets/robots/mini3_mjlab/scene_pick_carry.xml"
CAMERAS = ("overview", "head_rgb", "left_gripper_rgb", "right_gripper_rgb")


def load_trajectory(path: Path, model: mujoco.MjModel) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        result = {name: source[name].copy() for name in ("time", "phase", "qpos", "qvel")}
    count = len(result["time"])
    if count == 0 or result["qpos"].shape != (count, model.nq) or result["qvel"].shape != (count, model.nv):
        raise ValueError("Trajectory dimensions do not match this scene")
    if result["phase"].shape != (count,) or np.any(np.diff(result["time"]) <= 0):
        raise ValueError("Trajectory requires one phase per frame and strictly increasing time")
    if not all(np.isfinite(result[name]).all() for name in ("time", "qpos", "qvel")):
        raise ValueError("Trajectory contains non-finite values")
    return result


def stage_indices(phases: np.ndarray, *, final_success: bool = False) -> dict[str, int]:
    result = {"init": 0}
    for name, phase in (("close", "CLOSE"), ("lift", "LIFT"), ("carry", "CARRY"), ("success", "SUCCESS")):
        indices = np.flatnonzero(phases == phase)
        if indices.size:
            result[name] = int(indices[len(indices) // 2] if name == "carry" else indices[-1])
    if final_success and "success" not in result:
        # The controller may record the last physics sample immediately before
        # declaring success; retain that exact state and consult its report.
        result["success"] = len(phases) - 1
    return result


def configure_camera(camera: mujoco.MjvCamera, model: mujoco.MjModel, data: mujoco.MjData, name: str) -> None:
    if name != "overview":
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if camera_id < 0:
            raise ValueError(f"Camera {name!r} is missing from the selected scene")
        camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera.fixedcamid = camera_id
        return
    base = data.xpos[model.body("base_link").id]
    targets = np.array([data.xpos[model.body(name).id] for name in ("pick_table", "basket_table")])
    left = min(float(base[0]) - .3, float(targets[:, 0].min()) - .3)
    right = max(float(base[0]) + .4, float(targets[:, 0].max()) + .3)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.fixedcamid = -1
    # Follow the moving base while widening the view to retain both tables.
    camera.lookat[:] = [(left + right) / 2, -.12, .40]
    camera.distance = max(2.15, (right - left) * 1.25)
    camera.azimuth = -110
    camera.elevation = -40


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--output", required=True, type=Path, help="MP4 output, or PNG for one final frame")
    parser.add_argument("--camera", choices=CAMERAS, default="overview")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--snapshots-only", action="store_true", help="Write stage PNGs without encoding the video")
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.output.suffix.lower() not in (".mp4", ".png"):
        parser.error("--output must end in .mp4 or .png")
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    recorded = load_trajectory(args.trajectory, model)
    report_path = args.trajectory.with_name("report.json")
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    final_success = report.get("success") is True
    width, height = (1280, 720) if args.camera == "overview" else (640, 480)
    if args.camera != "overview":
        # Scene extent spans several metres, but wrist objects are centimetres
        # from the lens. Keep the virtual near plane at most 2 mm away.
        model.vis.map.znear = min(model.vis.map.znear, .002 / model.stat.extent)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    camera = mujoco.MjvCamera()
    option = mujoco.MjvOption()
    option.geomgroup[1] = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), 18 if width > 640 else 13) if font_path.exists() else ImageFont.load_default()

    with mujoco.Renderer(model, height=height, width=width) as renderer:
        def render(frame: int) -> np.ndarray:
            data.qpos[:] = recorded["qpos"][frame]
            data.qvel[:] = recorded["qvel"][frame]
            data.time = float(recorded["time"][frame])
            mujoco.mj_forward(model, data)
            configure_camera(camera, model, data, args.camera)
            renderer.update_scene(data, camera=camera, scene_option=option)
            # Original STL meshes exhibit shadow-map acne; disable those purely
            # visual passes for a clean replay without changing any contacts.
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
            output = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(output)
            draw.rectangle((0, 0, width, 62 if width > 640 else 47), fill=(18, 24, 32))
            draw.text((14, 7), "RECORDED TRAJECTORY REPLAY | no policy inference", font=font, fill=(224, 234, 245))
            result_label = "   result=SUCCESS" if final_success and frame == len(recorded["time"]) - 1 else ""
            draw.text((14, 33 if width > 640 else 26),
                      f"t={data.time:5.2f}s   phase={recorded['phase'][frame]}   camera={args.camera}{result_label}",
                      font=font, fill=(120, 205, 250))
            return np.asarray(output)

        for stage, frame in stage_indices(recorded["phase"], final_success=final_success).items():
            path = args.output.parent / f"{args.camera}_{stage}.png"
            Image.fromarray(render(frame)).save(path)
            print(f"Saved recorded frame: {path}", flush=True)
        if args.snapshots_only:
            return
        if args.output.suffix.lower() == ".png":
            Image.fromarray(render(len(recorded["time"]) - 1)).save(args.output)
            return
        times = np.arange(recorded["time"][0], recorded["time"][-1], 1 / args.fps)
        frame_ids = np.searchsorted(recorded["time"], times, side="left")
        with imageio.get_writer(args.output, fps=args.fps, codec="libx264", quality=8,
                                macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"]) as writer:
            for count, frame in enumerate(frame_ids):
                writer.append_data(render(int(frame)))
                if count % (args.fps * 5) == 0:
                    print(f"Rendered {count}/{len(frame_ids)} recorded frames", flush=True)
            # Retain the final success/failure record even when sampling skips it.
            writer.append_data(render(len(recorded["time"]) - 1))
        print(f"Saved recorded trajectory replay: {args.output}", flush=True)


if __name__ == "__main__":
    main()
