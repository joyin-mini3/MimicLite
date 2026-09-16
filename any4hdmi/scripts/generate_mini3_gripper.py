"""Generate Mini3 with two forearm extensions, grippers and RGB cameras.

The original ``mini3.xml`` is the source of truth for the robot.  The additions
are deliberately simple estimates, not an identified hardware model. A 100 mm
extension and gripper add 0.20 kg per hand; shorter extensions preserve the same
capsule material density. The two gripper controls are in metres of finger travel: 0 closes
the jaws, 0.035 opens them to an inner gap of 70 mm.  A joint equality moves the
second finger symmetrically.  Existing joint and motor attributes are retained.
One head camera looks forward and down by 45 degrees.  A camera on each palm
looks at the finger gap.  Cameras and their visual mounts add no mass or joints;
their hardware mass has not been specified.

With ``--articulated-arms``, each arm gains a forearm axial-twist hinge and
distal wrist roll/pitch hinges using the original elbow-pitch joint's physical
parameters. The root hinge keeps the name ``elbow_yaw`` for compatibility but
rotates about local X, along the straight forearm, rather than bending it.
No motor housings or additional link mass are added. Articulated arms default
to 50 mm extensions; the original rigid arms retain 100 mm extensions. The
default articulated output is a separate ``mini3_gripper_7dof.xml``.

Run with the project's MuJoCo Python environment.  Mesh references in the output
are relative to the output file, so the generated model remains portable.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET


ASSET_DIR = Path(__file__).resolve().parents[1] / "assets/robots/mini3_mjlab"
DEFAULT_SOURCE = ASSET_DIR / "mini3.xml"
DEFAULT_OUTPUT = ASSET_DIR / "mini3_gripper.xml"
DEFAULT_ARTICULATED_OUTPUT = ASSET_DIR / "mini3_gripper_7dof.xml"
EXTENSION_LENGTH = 0.100
ARTICULATED_EXTENSION_LENGTH = 0.050
EXTENSION_RADIUS = 0.011
REFERENCE_EXTENSION_MASS = 0.060
FINGER_TRAVEL = 0.035
TCP_OFFSET = 0.068
RGB_CAMERA_FOVY = 80.0
HEAD_CAMERA_POS = (0.073, 0.0, 0.048)
GRIPPER_CAMERA_POS = (0.020, 0.0, 0.040)


def _numbers(values: list[float] | tuple[float, ...]) -> str:
    return " ".join(f"{value:.10g}" for value in values)


def _resolve_extension_length(articulated_arms: bool, extension_length: float | None) -> float:
    default = ARTICULATED_EXTENSION_LENGTH if articulated_arms else EXTENSION_LENGTH
    length = default if extension_length is None else extension_length
    if not math.isfinite(length) or length <= 0:
        raise ValueError("Extension length must be finite and positive")
    return length


def _extension_mass(length: float) -> float:
    """Scale the full capsule volume, including its two hemispherical ends."""
    end_volume = 4.0 / 3.0 * math.pi * EXTENSION_RADIUS ** 3
    cross_section = math.pi * EXTENSION_RADIUS ** 2
    reference_volume = cross_section * EXTENSION_LENGTH + end_volume
    return REFERENCE_EXTENSION_MASS * (cross_section * length + end_volume) / reference_volume


def _add_rgb_camera(parent: ET.Element, name: str, position: tuple[float, float, float],
                    downward_angle: float) -> None:
    """Attach a massless RGB camera with image-right toward the body's -Y axis.

    MuJoCo cameras look along local -Z; xyaxes specifies image-right and image-up
    in the parent body frame.  The lens and enclosure stay behind the image plane
    so the camera can see through its own mount without hiding any robot geometry.
    """
    cosine, sine = math.cos(downward_angle), math.sin(downward_angle)
    forward = (cosine, 0.0, -sine)
    xyaxes = _numbers((0.0, -1.0, 0.0, sine, 0.0, cosine))
    parent.append(ET.Comment(
        " Virtual RGB camera and visual-only mount: no added hardware mass or collision. "
    ))
    ET.SubElement(parent, "camera", name=name, mode="fixed", pos=_numbers(position),
                  xyaxes=xyaxes, fovy=f"{RGB_CAMERA_FOVY:g}", resolution="640 480")
    for suffix, distance, shape, size, material in (
        ("housing", 0.014, "box", "0.012 0.009 0.01", "rgb_camera_housing"),
        ("lens", 0.003, "cylinder", "0.005 0.001", "rgb_camera_lens"),
    ):
        center = tuple(value - distance * axis for value, axis in zip(position, forward))
        ET.SubElement(parent, "geom", name=f"{name}_{suffix}", type=shape,
                      pos=_numbers(center), xyaxes=xyaxes, size=size, mass="0",
                      contype="0", conaffinity="0", group="2", material=material)


def _joint_layout(root: ET.Element, velocity: bool = False) -> dict[str, slice]:
    layout: dict[str, slice] = {}
    offset = 0
    worldbody = root.find("worldbody")
    assert worldbody is not None
    for element in worldbody.iter():
        if element.tag not in ("joint", "freejoint"):
            continue
        kind = "free" if element.tag == "freejoint" else element.get("type", "hinge")
        width = {"free": 6 if velocity else 7, "ball": 3 if velocity else 4}.get(kind, 1)
        layout[element.attrib["name"]] = slice(offset, offset + width)
        offset += width
    return layout


def _extend_keyframes(root: ET.Element, source: ET.Element) -> None:
    for key in root.findall("keyframe/key"):
        for attribute in ("qpos", "qvel"):
            if attribute not in key.attrib:
                continue
            old_layout = _joint_layout(source, velocity=attribute == "qvel")
            new_layout = _joint_layout(root, velocity=attribute == "qvel")
            original = [float(value) for value in key.attrib[attribute].split()]
            expected = max(item.stop for item in old_layout.values())
            if len(original) != expected:
                raise ValueError(f"Keyframe {key.get('name')!r} has an invalid {attribute} length")
            expanded = [0.0] * max(item.stop for item in new_layout.values())
            for name, indices in new_layout.items():
                if name in old_layout:
                    expanded[indices] = original[old_layout[name]]
                elif attribute == "qpos" and name.endswith(("_gripper_finger_joint", "_gripper_follower_joint")):
                    expanded[indices] = [FINGER_TRAVEL]
            key.set(attribute, _numbers(expanded))
        old_controls = [float(value) for value in key.get("ctrl", "").split()]
        old_actuators = source.find("actuator")
        assert old_actuators is not None
        if not old_controls:
            old_controls = [0.0] * len(old_actuators)
        if len(old_controls) != len(old_actuators):
            raise ValueError(f"Keyframe {key.get('name')!r} has an invalid ctrl length")
        old_by_name = {
            actuator.attrib["name"]: value
            for actuator, value in zip(old_actuators, old_controls, strict=True)
        }
        new_actuators = root.find("actuator")
        assert new_actuators is not None
        key.set("ctrl", _numbers([
            old_by_name.get(actuator.attrib["name"],
                            FINGER_TRAVEL if actuator.get("joint", "").endswith("_gripper_finger_joint") else 0.0)
            for actuator in new_actuators
        ]))


def build_model(source: Path = DEFAULT_SOURCE, *, articulated_arms: bool = False,
                extension_length: float | None = None) -> ET.ElementTree:
    """Return a new MJCF tree without changing the source model."""
    length = _resolve_extension_length(articulated_arms, extension_length)
    extension_mass = _extension_mass(length)
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.parse(source, parser=parser)
    root = tree.getroot()
    original = ET.fromstring(ET.tostring(root))
    root.set("model", "mini3_gripper_7dof" if articulated_arms else "mini3_gripper")
    root.insert(0, ET.Comment(
        " Generated by any4hdmi/scripts/generate_mini3_gripper.py. "
        f"Each arm has a {length * 1000:g} mm extension plus a 0-70 mm parallel gripper; "
        f"added mass is {extension_mass + 0.14:.10g} kg per arm. Gripper ctrl is half the inner gap in metres. "
        "Head and palm RGB cameras are massless virtual sensors with visual-only mounts. "
    ))
    asset = root.find("asset")
    actuators = root.find("actuator")
    contact = root.find("contact")
    assert asset is not None and actuators is not None and contact is not None
    ET.SubElement(asset, "material", name="gripper_metal", rgba="0.23 0.42 0.53 1")
    ET.SubElement(asset, "material", name="gripper_finger_material", rgba="0.12 0.14 0.17 1")
    ET.SubElement(asset, "material", name="rgb_camera_housing", rgba="0.08 0.1 0.13 1")
    ET.SubElement(asset, "material", name="rgb_camera_lens", rgba="0.08 0.38 0.65 1",
                  specular="0.9", shininess="0.8")
    head = root.find(".//body[@name='head_link']")
    assert head is not None
    _add_rgb_camera(head, "head_rgb", HEAD_CAMERA_POS, math.radians(45.0))
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    arm_motors: list[ET.Element] = []

    def add_arm_joint(parent: ET.Element, side: str, name: str, axis: str, limit: float) -> None:
        source_joint = original.find(f".//body/joint[@name='{side}_elbow_pitch_joint']")
        source_motor = original.find(f"actuator/motor[@joint='{side}_elbow_pitch_joint']")
        assert source_joint is not None and source_motor is not None
        joint_name = f"{side}_{name}_joint"
        joint_attributes = dict(source_joint.attrib)
        joint_attributes.update(name=joint_name, type="hinge", pos="0 0 0", ref="0",
                                axis=axis, range=_numbers((-limit, limit)), limited="true")
        ET.SubElement(parent, "joint", **joint_attributes)
        motor_attributes = dict(source_motor.attrib)
        motor_attributes.update(name=f"{joint_name}_ctrl", joint=joint_name,
                                ctrllimited="true", ctrlrange="-12.5 12.5",
                                forcelimited="true", forcerange="-12.5 12.5")
        arm_motors.append(ET.Element("motor", motor_attributes))

    for side in ("left", "right"):
        elbow = root.find(f".//body[@name='{side}_elbow_pitch_link']")
        assert elbow is not None
        tip = elbow.find(f"geom[@name='{side}_elbow_pitch_link_collision']")
        assert tip is not None
        extension = ET.SubElement(elbow, "body", name=f"{side}_forearm_extension", pos=tip.attrib["pos"])
        if articulated_arms:
            extension.append(ET.Comment(
                " Forearm axial twist about local X keeps the extension straight; "
                "the elbow_yaw name is retained for compatibility. No added motor body or mass. "
            ))
            add_arm_joint(extension, side, "elbow_yaw", "1 0 0", math.pi / 2)
            ET.SubElement(contact, "exclude", body1=f"{side}_elbow_pitch_link",
                          body2=f"{side}_forearm_extension")
        ET.SubElement(extension, "site", name=f"{side}_forearm_original_tip", pos="0 0 0",
                      size="0.003", rgba="1 0.7 0.1 1", group="3")
        ET.SubElement(extension, "geom", name=f"{side}_forearm_extension_geom", type="capsule",
                      fromto=f"0 0 0 {length:.10g} 0 0", size=f"{EXTENSION_RADIUS:g}", mass=f"{extension_mass:.10g}",
                      **{"class": "collision", "material": "gripper_metal", "group": "0"})
        palm = ET.SubElement(extension, "body", name=f"{side}_gripper_palm", pos=f"{length:.10g} 0 0")
        if articulated_arms:
            palm.append(ET.Comment(" Two co-located wrist hinges use the existing palm inertia. "))
            add_arm_joint(palm, side, "wrist_roll", "1 0 0", math.pi)
            add_arm_joint(palm, side, "wrist_pitch", "0 1 0", math.pi / 2)
            ET.SubElement(contact, "exclude", body1=f"{side}_forearm_extension",
                          body2=f"{side}_gripper_palm")
        ET.SubElement(palm, "site", name=f"{side}_gripper_mount", pos="0 0 0",
                      size="0.003", rgba="1 0.7 0.1 1", group="3")
        ET.SubElement(palm, "geom", name=f"{side}_gripper_palm_geom", type="box", pos="0.009 0 0",
                      size="0.009 0.048 0.014", mass="0.08",
                      **{"class": "collision", "material": "gripper_metal", "group": "0"})
        ET.SubElement(palm, "site", name=f"{side}_gripper_tcp", pos=f"{TCP_OFFSET:g} 0 0",
                      type="sphere", size="0.004", rgba="0.1 1 0.5 0.8", group="3")
        _add_rgb_camera(palm, f"{side}_gripper_rgb", GRIPPER_CAMERA_POS,
                        math.atan2(GRIPPER_CAMERA_POS[2], TCP_OFFSET - GRIPPER_CAMERA_POS[0]))
        ET.SubElement(palm, "geom", name=f"{side}_gripper_rgb_mount", type="cylinder",
                      fromto="0.009 0 0.014 0.009 0 0.04", size="0.003", mass="0",
                      contype="0", conaffinity="0", group="2", material="rgb_camera_housing")
        driver = f"{side}_gripper_finger_joint"
        follower = f"{side}_gripper_follower_joint"
        for direction, label, joint_name in ((1, "positive", driver), (-1, "negative", follower)):
            finger_name = f"{side}_gripper_finger_{label}"
            finger = ET.SubElement(palm, "body", name=finger_name, pos=f"0.018 {direction * 0.005:g} 0")
            ET.SubElement(finger, "joint", name=joint_name, type="slide", axis=f"0 {direction} 0",
                          range=f"0 {FINGER_TRAVEL:g}", damping="2", armature="0.0001", frictionloss="0.05")
            ET.SubElement(finger, "geom", name=f"{finger_name}_geom", type="box", pos="0.035 0 0",
                          size="0.035 0.005 0.014", mass="0.03", condim="4", friction="2 0.02 0.001",
                          **{"class": "collision", "material": "gripper_finger_material", "group": "0"})
            ET.SubElement(contact, "exclude", body1=f"{side}_gripper_palm", body2=finger_name)
        ET.SubElement(equality, "joint", name=f"{side}_gripper_coupling", joint1=follower, joint2=driver,
                      polycoef="0 1 0 0 0", solref="0.004 1", solimp="0.99 0.999 0.00001")
        ET.SubElement(actuators, "position", name=f"{side}_gripper_ctrl", joint=driver, kp="800", kv="8",
                      ctrllimited="true", ctrlrange=f"0 {FINGER_TRAVEL:g}", forcelimited="true", forcerange="-20 20")

    # Keep the original 21 motors and the two gripper controls in their existing
    # order. The six additional torque motors are appended and mapped by name.
    actuators.extend(arm_motors)
    _extend_keyframes(root, original)
    ET.indent(tree, space="  ")
    return tree


def generate_model(source: Path = DEFAULT_SOURCE, output: Path | None = None,
                   *, articulated_arms: bool = False, extension_length: float | None = None) -> Path:
    """Generate the MJCF and compile it with MuJoCo before returning its path."""
    import mujoco

    if output is None:
        output = DEFAULT_ARTICULATED_OUTPUT if articulated_arms else DEFAULT_OUTPUT
    source, output = source.resolve(), output.resolve()
    if source == output:
        raise ValueError("Output must differ from the source mini3.xml")
    tree = build_model(source, articulated_arms=articulated_arms, extension_length=extension_length)
    compiler = tree.getroot().find("compiler")
    assert compiler is not None
    mesh_dir = (source.parent / compiler.get("meshdir", ".")).resolve()
    compiler.set("meshdir", Path(os.path.relpath(mesh_dir, output.parent)).as_posix())
    output.parent.mkdir(parents=True, exist_ok=True)
    # Compile the complete candidate first; a failed validation never replaces
    # an existing generated file. Use an absolute mesh path only in memory.
    compiler.set("meshdir", str(mesh_dir))
    mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(), encoding="unicode"))
    compiler.set("meshdir", Path(os.path.relpath(mesh_dir, output.parent)).as_posix())
    tree.write(output, encoding="utf-8", xml_declaration=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path,
                        help="Output MJCF; default depends on --articulated-arms")
    parser.add_argument("--articulated-arms", action="store_true",
                        help="Add forearm axial twist (named elbow_yaw), wrist roll and wrist pitch to both arms")
    parser.add_argument("--extension-length", type=float,
                        help="Extension length in metres; default 0.05 for articulated arms, otherwise 0.1")
    args = parser.parse_args()
    output = generate_model(args.source, args.output, articulated_arms=args.articulated_arms,
                            extension_length=args.extension_length)
    length = _resolve_extension_length(args.articulated_arms, args.extension_length)
    extension_mass = _extension_mass(length)
    print(f"Generated and compiled {output}")
    print(f"Each gripper: ctrl=0 closes, ctrl=0.035 opens to 70 mm; added mass={extension_mass + 0.14:.10g} kg/arm.")
    print(f"Extension: {length * 1000:g} mm, {extension_mass:.10g} kg; original capsule density retained.")
    print("RGB cameras: head_rgb (45 degrees down), left_gripper_rgb, right_gripper_rgb; no added mass.")
    if args.articulated_arms:
        print("Each arm: elbow_yaw axial twist X +/-90 deg, wrist_roll X +/-180 deg, "
              f"wrist_pitch Y +/-90 deg; straight {length * 1000:g} mm extension; 12.5 Nm motors.")


if __name__ == "__main__":
    main()
