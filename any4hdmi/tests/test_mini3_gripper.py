from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


ANY4HDMI_DIR = Path(__file__).resolve().parents[1]
ASSET_DIR = ANY4HDMI_DIR / "assets/robots/mini3_mjlab"
SOURCE = ASSET_DIR / "mini3.xml"
GENERATED = ASSET_DIR / "mini3_gripper.xml"
spec = importlib.util.spec_from_file_location(
    "generate_mini3_gripper", ANY4HDMI_DIR / "scripts/generate_mini3_gripper.py"
)
assert spec is not None and spec.loader is not None
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


def isolated_gripper(side: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Fixed palm, freely moving fingers and a free 40 mm / 60 g cube."""
    source = ET.parse(GENERATED).getroot()
    root = ET.Element("mujoco")
    ET.SubElement(root, "option", timestep="0.001", integrator="implicitfast", gravity="0 0 0")
    root.append(copy.deepcopy(source.find("default")))
    asset = ET.SubElement(root, "asset")
    for material in source.findall("asset/material"):
        asset.append(copy.deepcopy(material))
    world = ET.SubElement(root, "worldbody")
    extension = copy.deepcopy(source.find(f".//body[@name='{side}_forearm_extension']"))
    assert extension is not None
    extension.set("pos", "0 0 0.3")
    world.append(extension)
    cube = ET.SubElement(world, "body", name="test_cube", pos="0.168 0 0.3")
    ET.SubElement(cube, "freejoint", name="test_cube_joint")
    ET.SubElement(cube, "geom", name="test_cube_geom", type="box", size="0.02 0.02 0.02",
                  mass="0.06", friction="2 0.02 0.001", condim="4")
    actuator = ET.SubElement(root, "actuator")
    actuator.append(copy.deepcopy(source.find(f"actuator/position[@name='{side}_gripper_ctrl']")))
    equality = ET.SubElement(root, "equality")
    equality.append(copy.deepcopy(source.find(f"equality/joint[@name='{side}_gripper_coupling']")))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    for name in (f"{side}_gripper_finger_joint", f"{side}_gripper_follower_joint"):
        data.qpos[model.joint(name).qposadr[0]] = 0.035
    mujoco.mj_forward(model, data)
    return model, data


class Mini3GripperTest(unittest.TestCase):
    def test_existing_joints_and_motors_keep_all_attributes_and_motor_order(self) -> None:
        original = ET.parse(SOURCE).getroot()
        generated = ET.parse(GENERATED).getroot()
        joints = original.findall(".//body/joint")
        self.assertEqual(len(joints), 21)
        for joint in joints:
            actual = generated.find(f".//body/joint[@name='{joint.attrib['name']}']")
            self.assertIsNotNone(actual)
            self.assertEqual(joint.attrib, actual.attrib)
        old_motors = original.findall("actuator/motor")
        new_actuators = list(generated.find("actuator"))
        self.assertEqual([item.attrib for item in old_motors], [item.attrib for item in new_actuators[:21]])
        self.assertEqual([item.tag for item in new_actuators[21:]], ["position", "position"])

    def test_compiled_model_mass_and_stand_keyframe(self) -> None:
        original = mujoco.MjModel.from_xml_path(str(SOURCE))
        model = mujoco.MjModel.from_xml_path(str(GENERATED))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
        self.assertEqual((model.nq, model.nv, model.nu), (32, 31, 23))
        self.assertAlmostEqual(float(model.body_mass.sum() - original.body_mass.sum()), 0.4, places=10)
        for index in range(original.njnt):
            name = original.joint(index).name
            old_address = original.joint(name).qposadr[0]
            new_address = model.joint(name).qposadr[0]
            count = 7 if name == "floating_base" else 1
            np.testing.assert_array_equal(
                data.qpos[new_address:new_address + count],
                original.key("stand").qpos[old_address:old_address + count],
            )
        for side in ("left", "right"):
            for suffix in ("finger", "follower"):
                joint = model.joint(f"{side}_gripper_{suffix}_joint")
                self.assertAlmostEqual(data.qpos[joint.qposadr[0]], 0.035)
                self.assertGreater(model.body_mass[joint.bodyid[0]], 0.0)
                self.assertTrue(np.all(model.body_inertia[joint.bodyid[0]] > 0.0))
            self.assertAlmostEqual(data.ctrl[model.actuator(f"{side}_gripper_ctrl").id], 0.035)

    def test_rigid_extension_is_100_mm_in_both_arm_poses(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(GENERATED))
        data = mujoco.MjData(model)
        for angle in (0.0, 1.2):
            for side in ("left", "right"):
                data.qpos[model.joint(f"{side}_elbow_pitch_joint").qposadr[0]] = angle
            mujoco.mj_forward(model, data)
            for side in ("left", "right"):
                tip = data.site(f"{side}_forearm_original_tip").xpos
                mount = data.site(f"{side}_gripper_mount").xpos
                tcp = data.site(f"{side}_gripper_tcp").xpos
                self.assertAlmostEqual(float(np.linalg.norm(mount - tip)), 0.1, places=10)
                self.assertAlmostEqual(float(np.linalg.norm(tcp - mount)), 0.068, places=10)

    def test_both_grippers_hold_a_free_cube_by_contact_and_release_it(self) -> None:
        for side in ("left", "right"):
            with self.subTest(side=side):
                model, data = isolated_gripper(side)
                driver = model.joint(f"{side}_gripper_finger_joint").qposadr[0]
                follower = model.joint(f"{side}_gripper_follower_joint").qposadr[0]
                for _ in range(500):
                    mujoco.mj_step(model, data)
                # The 40 mm cube must stop closure at approximately 20 mm per finger.
                self.assertAlmostEqual(data.qpos[driver], 0.02, delta=0.0002)
                self.assertAlmostEqual(data.qpos[driver], data.qpos[follower], delta=0.0001)
                cube_geom = model.geom("test_cube_geom").id
                contacting = {
                    int(contact.geom1 if contact.geom2 == cube_geom else contact.geom2)
                    for contact in data.contact
                    if cube_geom in (contact.geom1, contact.geom2)
                }
                for label in ("positive", "negative"):
                    self.assertIn(model.geom(f"{side}_gripper_finger_{label}_geom").id, contacting)
                self.assertGreater(abs(data.actuator_force[0]), 1.0)
                self.assertLessEqual(abs(data.actuator_force[0]), 20.0)
                self.assertGreater(min(contact.dist for contact in data.contact), -0.0002)
                model.opt.gravity[2] = -9.81
                for _ in range(1000):
                    mujoco.mj_step(model, data)
                self.assertAlmostEqual(data.body("test_cube").xpos[2], 0.3, delta=0.001)
                data.ctrl[0] = 0.035
                for _ in range(200):
                    mujoco.mj_step(model, data)
                self.assertGreater(data.qpos[driver], 0.034)
                self.assertAlmostEqual(data.qpos[driver], data.qpos[follower], delta=0.0001)
                self.assertLess(data.body("test_cube").xpos[2], 0.2)

    def test_generator_is_repeatable_and_keeps_source_unchanged(self) -> None:
        before = SOURCE.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "mini3_gripper.xml"
            generator.generate_model(SOURCE, output)
            first = output.read_bytes()
            generator.generate_model(SOURCE, output)
            self.assertEqual(first, output.read_bytes())
            mujoco.MjModel.from_xml_path(str(output))
        self.assertEqual(before, SOURCE.read_bytes())
        with self.assertRaises(ValueError):
            generator.generate_model(SOURCE, SOURCE)


if __name__ == "__main__":
    unittest.main()
