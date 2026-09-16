from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


ANY4HDMI_DIR = Path(__file__).resolve().parents[1]
ASSET_DIR = ANY4HDMI_DIR / "assets/robots/mini3_mjlab"
SOURCE = ASSET_DIR / "mini3.xml"
RIGID = ASSET_DIR / "mini3_gripper.xml"
ARTICULATED = ASSET_DIR / "mini3_gripper_7dof.xml"
SCENE = ASSET_DIR / "scene_pick_carry_7dof.xml"
SHORT_EXTENSION_LENGTH = 0.050
JOINT_SPECS = (
    ("elbow_yaw", (1.0, 0.0, 0.0), math.pi / 2),
    ("wrist_roll", (1.0, 0.0, 0.0), math.pi),
    ("wrist_pitch", (0.0, 1.0, 0.0), math.pi / 2),
)
spec = importlib.util.spec_from_file_location(
    "generate_mini3_articulated_gripper", ANY4HDMI_DIR / "scripts/generate_mini3_gripper.py"
)
assert spec is not None and spec.loader is not None
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


def rotation(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    quat, matrix = np.zeros(4), np.zeros(9)
    mujoco.mju_axisAngle2Quat(quat, np.asarray(axis, dtype=float), angle)
    mujoco.mju_quat2Mat(matrix, quat)
    return matrix.reshape(3, 3)


def generated_model(*, articulated_arms: bool, extension_length: float | None = None) -> mujoco.MjModel:
    tree = generator.build_model(SOURCE, articulated_arms=articulated_arms,
                                  extension_length=extension_length)
    tree.getroot().find("compiler").set("meshdir", str(ASSET_DIR / "meshes"))
    return mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(), encoding="unicode"))


class Mini3ArticulatedGripperTest(unittest.TestCase):
    def test_six_hinges_copy_elbow_physics_and_have_limited_torque_motors(self) -> None:
        old = mujoco.MjModel.from_xml_path(str(RIGID))
        model = mujoco.MjModel.from_xml_path(str(ARTICULATED))
        self.assertEqual((model.nq, model.nv, model.nu), (38, 37, 29))
        self.assertEqual([model.actuator(i).name for i in range(23)],
                         [old.actuator(i).name for i in range(23)])
        data = mujoco.MjData(model)
        for side in ("left", "right"):
            original = model.joint(f"{side}_elbow_pitch_joint")
            old_dof = original.dofadr[0]
            for suffix, axis, limit in JOINT_SPECS:
                joint = model.joint(f"{side}_{suffix}_joint")
                self.assertEqual(model.jnt_type[joint.id], mujoco.mjtJoint.mjJNT_HINGE)
                np.testing.assert_array_equal(model.jnt_axis[joint.id], axis)
                np.testing.assert_allclose(model.jnt_range[joint.id], (-limit, limit), atol=1e-9)
                self.assertTrue(model.jnt_limited[joint.id])
                for values in (model.dof_armature, model.dof_damping, model.dof_frictionloss):
                    self.assertEqual(values[joint.dofadr[0]], values[old_dof])
                actuator = model.actuator(f"{joint.name}_ctrl")
                self.assertEqual(model.actuator_trnid[actuator.id, 0], joint.id)
                self.assertTrue(model.actuator_ctrllimited[actuator.id])
                self.assertTrue(model.actuator_forcelimited[actuator.id])
                np.testing.assert_array_equal(model.actuator_ctrlrange[actuator.id], (-12.5, 12.5))
                np.testing.assert_array_equal(model.actuator_forcerange[actuator.id], (-12.5, 12.5))
                data.ctrl[actuator.id] = 100
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.actuator_force[23:], 12.5)

    def test_joint_origins_axes_and_serial_wrist_rotation(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(ARTICULATED))
        data = mujoco.MjData(model)
        yaw, roll, pitch = 0.45, -0.65, 0.35
        for side in ("left", "right"):
            data.qpos[model.joint(f"{side}_elbow_pitch_joint").qposadr] = 0.5
            for suffix, angle in (("elbow_yaw", yaw), ("wrist_roll", roll), ("wrist_pitch", pitch)):
                joint = model.joint(f"{side}_{suffix}_joint")
                data.qpos[joint.qposadr] = angle
                np.testing.assert_array_equal(model.jnt_pos[joint.id], (0, 0, 0))
        mujoco.mj_forward(model, data)
        for side in ("left", "right"):
            elbow = model.body(f"{side}_elbow_pitch_link").id
            extension = model.body(f"{side}_forearm_extension").id
            palm = model.body(f"{side}_gripper_palm").id
            wrist_roll = model.joint(f"{side}_wrist_roll_joint")
            wrist_pitch = model.joint(f"{side}_wrist_pitch_joint")
            self.assertEqual(wrist_roll.bodyid[0], palm)
            self.assertEqual(wrist_pitch.bodyid[0], palm)
            expected_extension = data.xmat[elbow].reshape(3, 3) @ rotation((1, 0, 0), yaw)
            expected_palm = expected_extension @ rotation((1, 0, 0), roll) @ rotation((0, 1, 0), pitch)
            np.testing.assert_allclose(data.xmat[extension].reshape(3, 3), expected_extension, atol=1e-12)
            np.testing.assert_allclose(data.xmat[palm].reshape(3, 3), expected_palm, atol=1e-12)
            np.testing.assert_allclose(data.xpos[palm] - data.xpos[extension],
                                       expected_extension @ (SHORT_EXTENSION_LENGTH, 0, 0), atol=1e-12)
            np.testing.assert_allclose(data.site(f"{side}_gripper_tcp").xpos,
                                       data.xpos[palm] + expected_palm @ (0.068, 0, 0), atol=1e-12)
            self.assertEqual(model.cam_bodyid[model.camera(f"{side}_gripper_rgb").id], palm)

    def test_root_twist_never_bends_extension_or_moves_distal_wrist_anchor(self) -> None:
        upstream_poses = ((0.0, 0.0, 0.0, 0.0),
                          (0.4, 0.5, -0.6, 0.7),
                          (-0.6, 0.25, 0.8, 1.4))
        for path in (ARTICULATED, SCENE):
            model = mujoco.MjModel.from_xml_path(str(path))
            data = mujoco.MjData(model)
            for side in ("left", "right"):
                elbow = model.body(f"{side}_elbow_pitch_link").id
                extension = model.body(f"{side}_forearm_extension").id
                palm = model.body(f"{side}_gripper_palm").id
                twist = model.joint(f"{side}_elbow_yaw_joint").id
                roll = model.joint(f"{side}_wrist_roll_joint").id
                pitch = model.joint(f"{side}_wrist_pitch_joint").id
                self.assertEqual(model.body_parentid[extension], elbow)
                self.assertEqual(model.body_parentid[palm], extension)
                self.assertEqual(model.body_jntnum[extension], 1)
                self.assertEqual(model.body_jntadr[extension], twist)
                self.assertEqual(model.body_jntnum[palm], 2)
                np.testing.assert_array_equal(model.jnt_bodyid[[twist, roll, pitch]],
                                              [extension, palm, palm])
                np.testing.assert_array_equal(model.jnt_pos[[twist, roll, pitch]], np.zeros((3, 3)))
                np.testing.assert_array_equal(model.jnt_axis[twist], [1, 0, 0])
                np.testing.assert_array_equal(model.jnt_axis[[roll, pitch]], [[1, 0, 0], [0, 1, 0]])
                np.testing.assert_array_equal(model.body_pos[palm], [SHORT_EXTENSION_LENGTH, 0, 0])
                expected_wrist_in_elbow = model.body_pos[extension] + [SHORT_EXTENSION_LENGTH, 0, 0]
                for upstream in upstream_poses:
                    mujoco.mj_resetData(model, data)
                    for suffix, angle in zip(("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch"),
                                             upstream, strict=True):
                        if side == "right" and suffix in ("shoulder_roll", "shoulder_yaw"):
                            angle = -angle
                        data.qpos[model.joint(f"{side}_{suffix}_joint").qposadr] = angle
                    data.qpos[model.jnt_qposadr[[roll, pitch]]] = [0.45, -0.3]
                    for angle in np.linspace(*model.jnt_range[twist], 9):
                        with self.subTest(model=path.name, side=side, upstream=upstream, twist=angle):
                            data.qpos[model.jnt_qposadr[twist]] = angle
                            mujoco.mj_kinematics(model, data)
                            elbow_rotation = data.xmat[elbow].reshape(3, 3)
                            extension_rotation = data.xmat[extension].reshape(3, 3)
                            # The longitudinal direction and wrist pivot stay fixed;
                            # only the extension's Y/Z cross-section rotates.
                            np.testing.assert_allclose(extension_rotation[:, 0], elbow_rotation[:, 0], atol=1e-12)
                            np.testing.assert_allclose(elbow_rotation.T @ extension_rotation,
                                                       rotation((1, 0, 0), angle), atol=1e-12)
                            wrist_in_elbow = elbow_rotation.T @ (data.xpos[palm] - data.xpos[elbow])
                            np.testing.assert_allclose(wrist_in_elbow, expected_wrist_in_elbow, atol=1e-12)
                            np.testing.assert_allclose(data.xaxis[twist], elbow_rotation[:, 0], atol=1e-12)
                            np.testing.assert_allclose(data.xanchor[roll], data.xanchor[pitch], atol=1e-12)
                            for wrist in (roll, pitch):
                                np.testing.assert_allclose(data.xanchor[wrist] - data.xanchor[twist],
                                                           SHORT_EXTENSION_LENGTH * elbow_rotation[:, 0], atol=1e-12)

    def test_explicit_100mm_zero_extra_angles_preserve_link_poses_inertia_and_mass(self) -> None:
        old = mujoco.MjModel.from_xml_path(str(RIGID))
        model = generated_model(articulated_arms=True, extension_length=0.1)
        original = mujoco.MjModel.from_xml_path(str(SOURCE))
        old_data, data = mujoco.MjData(old), mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(old, old_data, old.key("stand").id)
        mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
        for name in ("left_elbow_pitch_joint", "right_elbow_pitch_joint"):
            old_data.qpos[old.joint(name).qposadr] = 0.6
            data.qpos[model.joint(name).qposadr] = 0.6
        mujoco.mj_forward(old, old_data)
        mujoco.mj_forward(model, data)
        self.assertEqual(old.nbody, model.nbody)
        for i in range(old.nbody):
            j = model.body(old.body(i).name).id
            np.testing.assert_allclose(data.xpos[j], old_data.xpos[i], atol=1e-12)
            np.testing.assert_allclose(data.xmat[j], old_data.xmat[i], atol=1e-12)
            for values in ("body_mass", "body_inertia", "body_ipos", "body_iquat"):
                np.testing.assert_array_equal(getattr(model, values)[j], getattr(old, values)[i])
        self.assertAlmostEqual(float(model.body_mass.sum() - original.body_mass.sum()), 0.4, places=10)
        self.assertEqual(model.ncam, old.ncam)
        for name in ("head_rgb", "left_gripper_rgb", "right_gripper_rgb"):
            np.testing.assert_array_equal(model.cam_resolution[model.camera(name).id], (640, 480))

    def test_short_extension_retains_capsule_density_and_updates_mass_com_and_inertia(self) -> None:
        old = mujoco.MjModel.from_xml_path(str(RIGID))
        model = mujoco.MjModel.from_xml_path(str(ARTICULATED))
        original = mujoco.MjModel.from_xml_path(str(SOURCE))
        radius, length = 0.011, SHORT_EXTENSION_LENGTH
        density = 0.06 / (math.pi * radius ** 2 * 0.1 + 4 * math.pi * radius ** 3 / 3)
        cylinder_mass = density * math.pi * radius ** 2 * length
        end_mass = density * 4 * math.pi * radius ** 3 / 3
        mass = cylinder_mass + end_mass
        axial_inertia = 0.5 * cylinder_mass * radius ** 2 + 0.4 * end_mass * radius ** 2
        transverse_inertia = cylinder_mass * (length ** 2 / 12 + radius ** 2 / 4) + end_mass * (
            0.4 * radius ** 2 + 3 * length * radius / 8 + length ** 2 / 4)
        for side in ("left", "right"):
            body = model.body(f"{side}_forearm_extension").id
            geom = model.geom(f"{side}_forearm_extension_geom").id
            np.testing.assert_allclose(model.geom_size[geom, :2], [radius, length / 2], atol=1e-12)
            self.assertAlmostEqual(model.body_mass[body], mass, places=10)
            np.testing.assert_allclose(model.body_ipos[body], [length / 2, 0, 0], atol=1e-12)
            matrix = np.zeros(9)
            mujoco.mju_quat2Mat(matrix, model.body_iquat[body])
            matrix = matrix.reshape(3, 3)
            actual_inertia = matrix @ np.diag(model.body_inertia[body]) @ matrix.T
            np.testing.assert_allclose(actual_inertia,
                                       np.diag([axial_inertia, transverse_inertia, transverse_inertia]),
                                       rtol=1e-9, atol=1e-14)
        # The gripper, cameras, and original robot retain their local inertia;
        # only the two shortened capsules change mass distribution.
        for body in range(old.nbody):
            name = old.body(body).name
            if name.endswith("forearm_extension"):
                continue
            new_body = model.body(name).id
            for attribute in ("body_mass", "body_inertia", "body_ipos", "body_iquat"):
                np.testing.assert_array_equal(getattr(model, attribute)[new_body], getattr(old, attribute)[body])
        self.assertAlmostEqual(float(model.body_mass.sum() - original.body_mass.sum()),
                               2 * (0.14 + mass), places=10)

    def test_extension_length_defaults_override_and_validation(self) -> None:
        for articulated, requested, expected in ((False, None, 0.1), (True, None, 0.05),
                                                   (True, 0.075, 0.075)):
            model = generated_model(articulated_arms=articulated, extension_length=requested)
            for side in ("left", "right"):
                np.testing.assert_allclose(model.body(f"{side}_gripper_palm").pos,
                                           [expected, 0, 0], atol=1e-12)
        for invalid in (0.0, -0.05, math.inf, math.nan):
            with self.subTest(length=invalid), self.assertRaisesRegex(ValueError, "finite and positive"):
                generator.build_model(SOURCE, articulated_arms=True, extension_length=invalid)

    def test_original_contract_and_nonzero_keyframes_are_remapped_by_name(self) -> None:
        protected = (SOURCE, RIGID, ASSET_DIR / "scene_pick_carry.xml")
        before = {path: path.read_bytes() for path in protected}
        source_tree = ET.parse(SOURCE)
        root = source_tree.getroot()
        root.find("compiler").set("meshdir", str(ASSET_DIR / "meshes"))
        original = mujoco.MjModel.from_xml_path(str(SOURCE))
        key = root.find("keyframe/key")
        key.set("qvel", " ".join(str(value) for value in np.linspace(-0.2, 0.2, original.nv)))
        key.set("ctrl", " ".join(str(i * 0.1) for i in range(original.nu)))
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "source.xml", Path(directory) / "articulated.xml"
            source_tree.write(source)
            generated = generator.generate_model(source, target, articulated_arms=True)
            first = generated.read_bytes()
            generator.generate_model(source, target, articulated_arms=True)
            self.assertEqual(first, generated.read_bytes())
            source_model = mujoco.MjModel.from_xml_path(str(source))
            model = mujoco.MjModel.from_xml_path(str(generated))
            data = mujoco.MjData(model)
            mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
            source_xml, new_xml = ET.parse(source).getroot(), ET.parse(generated).getroot()
            for joint in source_xml.findall(".//body/joint"):
                self.assertEqual(joint.attrib, new_xml.find(f".//body/joint[@name='{joint.get('name')}']").attrib)
            self.assertEqual([motor.attrib for motor in source_xml.findall("actuator/motor")],
                             [motor.attrib for motor in list(new_xml.find("actuator"))[:21]])
            for i in range(source_model.njnt):
                old_joint, joint = source_model.joint(i), model.joint(source_model.joint(i).name)
                for values, address, count in (
                    ("qpos", "qposadr", 7 if i == 0 else 1),
                    ("qvel", "dofadr", 6 if i == 0 else 1),
                ):
                    a, b = getattr(old_joint, address)[0], getattr(joint, address)[0]
                    np.testing.assert_allclose(getattr(data, values)[b:b + count],
                                               getattr(source_model.key("stand"), values)[a:a + count], atol=1e-9)
            np.testing.assert_allclose(data.ctrl[:21], source_model.key("stand").ctrl)
            np.testing.assert_array_equal(data.ctrl[21:23], (0.035, 0.035))
            np.testing.assert_array_equal(data.ctrl[23:], np.zeros(6))
            for side in ("left", "right"):
                for suffix, _, _ in JOINT_SPECS:
                    self.assertEqual(data.qpos[model.joint(f"{side}_{suffix}_joint").qposadr[0]], 0)
                for suffix in ("finger", "follower"):
                    self.assertEqual(data.qpos[model.joint(f"{side}_gripper_{suffix}_joint").qposadr[0]], 0.035)
        for path, contents in before.items():
            self.assertEqual(contents, path.read_bytes())

    def test_new_solids_collide_with_environment_nonadjacent_base_and_upper_arm(self) -> None:
        old_root, root = ET.parse(RIGID).getroot(), ET.parse(ARTICULATED).getroot()
        exclusions = lambda element: {
            frozenset((item.get("body1"), item.get("body2"))) for item in element.findall("contact/exclude")
        }
        added = exclusions(root) - exclusions(old_root)
        expected = {
            frozenset((f"{side}_{a}", f"{side}_{b}")) for side in ("left", "right")
            for a, b in (("elbow_pitch_link", "forearm_extension"), ("forearm_extension", "gripper_palm"))
        }
        self.assertEqual(added, expected)
        self.assertTrue(exclusions(old_root).issubset(exclusions(root)))
        model = mujoco.MjModel.from_xml_path(str(ARTICULATED))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
        mujoco.mj_forward(model, data)
        names = [f"{side}_{part}_geom" for side in ("left", "right")
                 for part in ("forearm_extension", "gripper_palm", "gripper_finger_positive", "gripper_finger_negative")]
        root.find("compiler").set("meshdir", str(ASSET_DIR / "meshes"))
        base = root.find(".//body[@name='base_link']")
        base_id = model.body("base_link").id
        pairs = []
        for i, name in enumerate(names):
            geom = model.geom(name).id
            self.assertEqual(model.geom_contype[geom], 1)
            self.assertEqual(model.geom_conaffinity[geom], 1)
            position = data.geom_xpos[geom]
            side = name.split("_", 1)[0]
            upper_name = f"{side}_shoulder_yaw_link"
            upper_id = model.body(upper_name).id
            for label, parent, point in (
                ("environment", root.find("worldbody"), position),
                ("base", base, data.xmat[base_id].reshape(3, 3).T @ (position - data.xpos[base_id])),
                ("upper_arm", root.find(f".//body[@name='{upper_name}']"),
                 data.xmat[upper_id].reshape(3, 3).T @ (position - data.xpos[upper_id])),
            ):
                probe = f"audit_{label}_{i}"
                ET.SubElement(parent, "geom", name=probe, type="sphere", size="0.005",
                              pos=" ".join(map(str, point)), mass="0", contype="1", conaffinity="1")
                pairs.append((name, probe))
        test_model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        test_data = mujoco.MjData(test_model)
        mujoco.mj_resetDataKeyframe(test_model, test_data, test_model.key("stand").id)
        mujoco.mj_forward(test_model, test_data)
        contacts = {frozenset((test_model.geom(int(c.geom1)).name, test_model.geom(int(c.geom2)).name))
                    for c in test_data.contact}
        for pair in pairs:
            self.assertIn(frozenset(pair), contacts)

    def test_scene_home_keeps_initial_distance_tables_and_gripper_controls(self) -> None:
        old = mujoco.MjModel.from_xml_path(str(ASSET_DIR / "scene_pick_carry.xml"))
        model = mujoco.MjModel.from_xml_path(str(SCENE))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        mujoco.mj_forward(model, data)
        self.assertEqual((model.nq, model.nv, model.nu), (59, 55, 29))
        self.assertFalse(data.eq_active[model.equality("base_support_weld").id])
        for name in ("pick_cube_1", "pick_cube_2", "pick_cube_3", "pick_basket", "pick_table", "basket_table"):
            np.testing.assert_array_equal(model.body(name).pos, old.body(name).pos)
        self.assertAlmostEqual(data.xpos[model.body("pick_cube_3").id, 0] - data.qpos[0], 3.0)
        np.testing.assert_array_equal(data.ctrl[23:], np.zeros(6))


if __name__ == "__main__":
    unittest.main()
