"""Seeded tabletop episodes and spatial retargeting of the policy reference."""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


COLORS = {1: "red", 2: "green", 3: "blue"}


def generate_episode(scene: Path, output: Path, *, seed: int | None = None,
                     target: str = "random") -> tuple[Path, dict]:
    """Save a compiled, replayable scene; sample only before physics starts.

    Every identity retains its own sampling region. Target selection is uniform
    and independent of layout, so selecting red really changes the pickup point.
    """
    if target not in (*COLORS.values(), "random"):
        raise ValueError("target must be random, red, green, or blue")
    seed = int(np.random.SeedSequence().entropy) if seed is None else seed
    rng = np.random.default_rng(seed)
    scene, output = Path(scene).resolve(), Path(output).resolve()
    tree = ET.parse(scene)
    root = tree.getroot()
    compiler = root.find("compiler")
    for attribute in ("meshdir", "texturedir"):
        directory = compiler.get(attribute, ".")
        compiler.set(attribute, str((scene.parent / directory).resolve()))
    table = root.find(".//body[@name='pick_table']")
    # Widen away from the robot; keep both approach-facing edges unchanged.
    table.set("pos", "0.43 -0.27 0")
    table.find("geom[@name='pick_table_top']").set("size", "0.20 0.08 0.02")
    for index, (x, y) in enumerate(((-.175, -.055), (-.175, .055),
                                    (.175, -.055), (.175, .055)), 1):
        table.find(f"geom[@name='pick_table_leg_{index}']").set("pos", f"{x} {y} 0.23")
    cubes = []
    for index in COLORS:
        position = np.array([.64 - .12 * index, -.24, .5205])
        position[:2] += rng.uniform(-.012, .012, 2)
        yaw = float(rng.uniform(-np.deg2rad(12), np.deg2rad(12)))
        quaternion = [float(np.cos(yaw / 2)), 0., 0., float(np.sin(yaw / 2))]
        body = root.find(f".//body[@name='pick_cube_{index}']")
        body.set("pos", " ".join(map(str, position)))
        body.set("quat", " ".join(map(str, quaternion)))
        cubes.append({"body": f"pick_cube_{index}", "color": COLORS[index],
                      "position": position.tolist(), "quaternion_wxyz": quaternion,
                      "yaw_degrees": float(np.rad2deg(yaw))})
    target_index = (int(rng.integers(1, 4)) if target == "random"
                    else next(index for index, color in COLORS.items() if color == target))
    # Explicit keyframe coordinates override body defaults in MuJoCo.
    original = mujoco.MjModel.from_xml_path(str(scene))
    home = root.find(".//key[@name='home']")
    qpos = np.fromstring(home.get("qpos"), sep=" ")
    for cube in cubes:
        address = int(original.joint(cube["body"] + "_free").qposadr[0])
        qpos[address:address + 7] = cube["position"] + cube["quaternion_wxyz"]
    delta = np.asarray(cubes[target_index - 1]["position"]) - [.28, -.24, .5205]
    qpos[:2] += delta[:2]
    home.set("qpos", " ".join(map(str, qpos)))
    output.mkdir(parents=True, exist_ok=True)
    generated = output / "episode_scene.xml"
    tree.write(generated, encoding="utf-8", xml_declaration=True)
    mujoco.MjModel.from_xml_path(str(generated))
    metadata = {"seed": seed, "target_index": target_index,
                "target_body": f"pick_cube_{target_index}", "target_color": COLORS[target_index],
                "cubes": cubes, "pickup_translation_m": delta.tolist(),
                "table_size_m": [.40, .16, .04], "table_height_m": .50,
                "position_jitter_m": .012, "yaw_jitter_degrees": 12.,
                "source_scene": str(scene), "episode_scene": str(generated)}
    (output / "episode.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return generated, metadata


def retarget_reference(plan, episode: dict) -> None:
    """Translate pickup references, returning smoothly to the fixed basket.

    Original joint references and outputs remain unchanged. Only world-space
    root positions change; the trained policy tracks the adapted route.
    """
    delta = np.asarray(episode["pickup_translation_m"])
    events = {event["phase"]: event for event in plan.events}

    def weight(times, start, end):
        fraction = np.clip((times - start) / (end - start), 0., 1.)
        return 1. - fraction * fraction * (3. - 2. * fraction)

    # The policy observes future displacement relative to its reference anchor,
    # not absolute world XY. Keep its pickup origin numerically identical to
    # the recording: translating float32 FK positions causes cancellation noise
    # that the tracking policy can amplify. Physics is initialized at +delta.
    plan.qpos[:, :3] += (weight(plan.times, events["CARRY"]["time"],
                                events["PLACE"]["time"]) - 1.)[:, None] * delta
    plan.source_scene_qpos[:, :3] += weight(
        plan.source_times, events["CARRY"]["source_time"],
        events["PLACE"]["source_time"])[:, None] * delta
    plan.metadata["episode"] = episode
