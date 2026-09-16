"""Build ground or tabletop pick-and-place scenes from the extended Mini3 XML."""

from __future__ import annotations

import argparse
import copy
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET


ASSET_DIR = Path(__file__).resolve().parents[1] / "assets/robots/mini3_mjlab"
ROBOT_XML = ASSET_DIR / "mini3_gripper.xml"
SCENE_XML = ASSET_DIR / "scene_pick_place.xml"
CUBE_LOCATIONS = ((0.28, 0.12), (0.33, 0.0), (0.28, -0.12))
LONGITUDINAL_CUBE_LOCATIONS = ((0.52, -0.24), (0.40, -0.24), (0.28, -0.24))
CUBE_COLORS = ("0.9 0.16 0.12 1", "0.12 0.65 0.25 1", "0.12 0.32 0.9 1")


def vector(values) -> str:
    return " ".join(f"{float(value):.10g}" for value in values)


def scene_tree(robot_xml: Path, output: Path, *, x: float, y: float, yaw: float,
               start_distance: float = 0.0, surface: str = "ground",
               table_height: float = 0.32, pickup_x: float = 0.0,
               pickup_y: float = 0.0, basket_x: float = 0.52,
               basket_y: float = 0.0, cube_layout: str = "original") -> ET.Element:
    """Place props in an approach frame, optionally starting the robot farther back.

    ``x, y, yaw`` define the original near-pickup approach pose. Positive
    ``start_distance`` moves only the robot backwards along that frame's +X.
    Cube locations are offset by ``pickup_x, pickup_y`` in the same frame;
    ``basket_x, basket_y`` locate the basket independently.
    """
    if not all(math.isfinite(value) for value in (
        x, y, yaw, start_distance, table_height, pickup_x, pickup_y, basket_x, basket_y,
    )):
        raise ValueError("Spawn, station positions and table dimensions must be finite")
    if surface not in ("ground", "table"):
        raise ValueError("Surface must be ground or table")
    if cube_layout not in ("original", "longitudinal"):
        raise ValueError("Cube layout must be original or longitudinal")
    if start_distance < 0:
        raise ValueError("Start distance must be nonnegative")
    if table_height <= 0.04:
        raise ValueError("Table height must exceed the 4 cm tabletop thickness")
    root = ET.parse(robot_xml).getroot()
    root.set("model", "mini3_gripper_pick_place")
    for tag in ("keyframe", "option", "visual", "statistic"):
        for element in root.findall(tag):
            root.remove(element)
    compiler = root.find("compiler")
    meshes = (robot_xml.parent / compiler.get("meshdir", "meshes")).resolve()
    compiler.set("meshdir", os.path.relpath(meshes, output.parent.resolve()))
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast",
                  gravity="0 0 -9.81", iterations="80", ls_iterations="30")
    scene_span = max(1.2, start_distance + 0.7, abs(basket_x) + 0.5, abs(basket_y) + 0.5)
    ET.SubElement(root, "statistic", center=vector((x + (0.2 - start_distance / 2) * math.cos(yaw),
                                                  y + (0.2 - start_distance / 2) * math.sin(yaw), 0.35)),
                  extent=vector((scene_span,)))
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "headlight", ambient="0.15 0.15 0.15", diffuse="0.35 0.35 0.35", specular="0.1 0.1 0.1")
    ET.SubElement(visual, "global", azimuth="135", elevation="-40", offwidth="1280", offheight="960")
    ET.SubElement(visual, "rgba", haze="0.85 0.89 0.93 1")
    assets = root.find("asset")
    ET.SubElement(assets, "texture", name="pick_floor_texture", type="2d", builtin="checker",
                  rgb1="0.68 0.72 0.76", rgb2="0.78 0.81 0.84", width="256", height="256")
    ET.SubElement(assets, "material", name="pick_floor_material", texture="pick_floor_texture",
                  texrepeat="8 8", texuniform="true", reflectance="0.05")

    cosine, sine = math.cos(yaw), math.sin(yaw)

    def world(local_x: float, local_y: float, z: float) -> str:
        return vector((x + cosine * local_x - sine * local_y,
                       y + sine * local_x + cosine * local_y, z))

    quat = vector((math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)))
    worldbody = root.find("worldbody")
    base = worldbody.find("body[@name='base_link']")
    height = float(base.get("pos").split()[2])
    base.set("pos", world(-start_distance, 0.0, height))
    base.set("quat", quat)
    ET.SubElement(worldbody, "light", name="pick_key_light", pos=world(1, -1, 2.5),
                  dir="-0.3 0.2 -1", directional="true", diffuse="0.55 0.55 0.55")
    ET.SubElement(worldbody, "light", name="pick_fill_light", pos=world(-1, 1, 1.5),
                  dir="0.3 -0.2 -1", directional="true", diffuse="0.15 0.15 0.15")
    ET.SubElement(worldbody, "geom", name="pick_floor", type="plane", size="0 0 0.05",
                  material="pick_floor_material", friction="1.5 0.01 0.001")

    surface_height = table_height if surface == "table" else 0.0
    locations = CUBE_LOCATIONS if cube_layout == "original" else LONGITUDINAL_CUBE_LOCATIONS
    cube_locations = tuple((local_x + pickup_x, local_y + pickup_y)
                           for local_x, local_y in locations)
    if surface == "table":
        # Close stations share a tabletop; separated stations get separate tables.
        # Every top and leg is a static collision geom, never an object constraint.
        cube_x, cube_y = zip(*cube_locations)
        tables = [
            ("pick_table", (min(cube_x) - 0.05, max(cube_x) + 0.05,
                            min(cube_y) - 0.05, max(cube_y) + 0.05)),
            ("basket_table", (basket_x - 0.16, basket_x + 0.16,
                              basket_y - 0.15, basket_y + 0.15)),
        ]
        first, second = tables[0][1], tables[1][1]
        if (first[0] < second[1] and second[0] < first[1]
                and first[2] < second[3] and second[2] < first[3]):
            tables = [("pick_table", (min(first[0], second[0]), max(first[1], second[1]),
                                      min(first[2], second[2]), max(first[3], second[3])))]
        for name, (low_x, high_x, low_y, high_y) in tables:
            half_x, half_y = (high_x - low_x) / 2, (high_y - low_y) / 2
            table = ET.SubElement(worldbody, "body", name=name,
                                  pos=world((low_x + high_x) / 2, (low_y + high_y) / 2, 0),
                                  quat=quat)
            ET.SubElement(table, "geom", name=f"{name}_top", type="box",
                          pos=vector((0, 0, table_height - 0.02)),
                          size=vector((half_x, half_y, 0.02)), rgba="0.58 0.45 0.32 1",
                          friction="1.2 0.01 0.001", solref="0.005 1")
            leg_half_height = (table_height - 0.04) / 2
            for index, (sign_x, sign_y) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1)), 1):
                ET.SubElement(table, "geom", name=f"{name}_leg_{index}", type="box",
                              pos=vector((sign_x * (half_x - 0.025), sign_y * (half_y - 0.025),
                                          leg_half_height)),
                              size=vector((0.015, 0.015, leg_half_height)),
                              rgba="0.25 0.27 0.29 1", friction="1.2 0.01 0.001")

    for index, ((local_x, local_y), color) in enumerate(zip(cube_locations, CUBE_COLORS), 1):
        body = ET.SubElement(worldbody, "body", name=f"pick_cube_{index}",
                            pos=world(local_x, local_y, surface_height + 0.0205), quat=quat)
        ET.SubElement(body, "freejoint", name=f"pick_cube_{index}_free")
        ET.SubElement(body, "geom", name=f"pick_cube_{index}_geom", type="box",
                      size="0.02 0.02 0.02", mass="0.04", rgba=color,
                      friction="1.2 0.01 0.001", condim="6", solref="0.005 1")
        ET.SubElement(body, "site", name=f"pick_cube_{index}_center", size="0.003",
                      rgba="1 1 1 0", group="5")

    basket = ET.SubElement(worldbody, "body", name="pick_basket",
                           pos=world(basket_x, basket_y, surface_height), quat=quat)
    # Interior: 0.24 x 0.22 m; 1 cm bottom/walls, 12 cm outer height.
    panels = {
        "bottom": ((0, 0, 0.005), (0.13, 0.12, 0.005)),
        "front": ((0.125, 0, 0.065), (0.005, 0.12, 0.055)),
        "back": ((-0.125, 0, 0.065), (0.005, 0.12, 0.055)),
        "left": ((0, 0.115, 0.065), (0.12, 0.005, 0.055)),
        "right": ((0, -0.115, 0.065), (0.12, 0.005, 0.055)),
    }
    for name, (position, size) in panels.items():
        ET.SubElement(basket, "geom", name=f"pick_basket_{name}", type="box",
                      pos=vector(position), size=vector(size), rgba="0.67 0.47 0.24 1",
                      friction="1.0 0.01 0.001", solref="0.005 1")
    ET.SubElement(basket, "site", name="pick_basket_target", pos="0 0 0.07",
                  type="sphere", size="0.008", rgba="0.2 1 0.5 0", group="5")

    # Optional manipulation fixture. The XML is free-floating by default;
    # the preview program explicitly activates this support for joint tuning.
    support = ET.SubElement(worldbody, "body", name="base_support", mocap="true",
                            pos=world(-start_distance, 0.0, height), quat=quat)
    ET.SubElement(support, "site", name="base_support_marker", size="0.01",
                  rgba="1 0.7 0.1 0", group="5")
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    ET.SubElement(equality, "weld", name="base_support_weld", body1="base_link",
                  body2="base_support", active="false", solref="0.01 1")
    return root


def build_scene(robot_xml: Path = ROBOT_XML, output: Path = SCENE_XML, *,
                x: float = 0.0, y: float = 0.0, yaw: float = 0.0,
                start_distance: float = 0.0, surface: str = "ground",
                table_height: float = 0.32, pickup_x: float = 0.0,
                pickup_y: float = 0.0, basket_x: float = 0.52,
                basket_y: float = 0.0, cube_layout: str = "original") -> Path:
    import mujoco

    robot_xml, output = Path(robot_xml).resolve(), Path(output).resolve()
    if robot_xml == output:
        raise ValueError("Scene output must differ from the source robot XML")
    root = scene_tree(robot_xml, output, x=x, y=y, yaw=yaw,
                      start_distance=start_distance, surface=surface, table_height=table_height,
                      pickup_x=pickup_x, pickup_y=pickup_y, basket_x=basket_x, basket_y=basket_y,
                      cube_layout=cube_layout)
    validation_root = copy.deepcopy(root)
    compiler = validation_root.find("compiler")
    compiler.set("meshdir", str((output.parent / compiler.get("meshdir")).resolve()))
    model = mujoco.MjModel.from_xml_string(ET.tostring(validation_root, encoding="unicode"))
    qpos, ctrl = model.qpos0.copy(), [0.0] * model.nu
    for side in ("left", "right"):
        for suffix in ("finger_joint", "follower_joint"):
            qpos[model.joint(f"{side}_gripper_{suffix}").qposadr[0]] = 0.035
        ctrl[model.actuator(f"{side}_gripper_ctrl").id] = 0.035
    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(keyframe, "key", name="home", qpos=vector(qpos), ctrl=vector(ctrl))
    root.insert(0, ET.Comment(
        " Generated by any4hdmi/scripts/generate_mini3_pick_scene.py. "
        "Props are fixed in the world relative to the initial approach frame; "
        "only the three cubes have free joints. base_support_weld is disabled by default. "
    ))
    ET.indent(root, space="  ")
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    verified = mujoco.MjModel.from_xml_path(str(output))
    print(f"Wrote {output} (nq={verified.nq}, nv={verified.nv}, nu={verified.nu})")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-xml", type=Path, default=ROBOT_XML)
    parser.add_argument("--output", type=Path, default=SCENE_XML)
    parser.add_argument("--x", type=float, default=0.0)
    parser.add_argument("--y", type=float, default=0.0)
    parser.add_argument("--yaw-deg", type=float, default=0.0, help="Initial yaw; props rotate with the spawn heading")
    parser.add_argument("--start-distance", type=float, default=0.0,
                        help="Move the robot this many metres behind the original approach pose")
    parser.add_argument("--surface", choices=("ground", "table"), default="ground")
    parser.add_argument("--cube-layout", choices=("original", "longitudinal"), default="original",
                        help="Longitudinal places blue, green and red along +X on the right of the approach path")
    parser.add_argument("--table-height", type=float, default=0.32, help="Tabletop upper surface height in metres")
    parser.add_argument("--pickup-x", type=float, default=0.0, help="X offset of all cubes in the approach frame")
    parser.add_argument("--pickup-y", type=float, default=0.0, help="Y offset of all cubes in the approach frame")
    parser.add_argument("--basket-x", type=float, default=0.52, help="Basket X position in the approach frame")
    parser.add_argument("--basket-y", type=float, default=0.0, help="Basket Y position in the approach frame")
    args = parser.parse_args()
    build_scene(args.robot_xml, args.output, x=args.x, y=args.y, yaw=math.radians(args.yaw_deg),
                start_distance=args.start_distance, surface=args.surface, table_height=args.table_height,
                pickup_x=args.pickup_x, pickup_y=args.pickup_y,
                basket_x=args.basket_x, basket_y=args.basket_y, cube_layout=args.cube_layout)


if __name__ == "__main__":
    main()
