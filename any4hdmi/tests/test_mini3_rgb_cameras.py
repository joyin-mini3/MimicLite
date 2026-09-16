from __future__ import annotations

import math
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


ASSET_DIR = Path(__file__).resolve().parents[1] / "assets/robots/mini3_mjlab"
ROBOT_XML = ASSET_DIR / "mini3_gripper.xml"
CAMERA_BODIES = {
    "head_rgb": "head_link",
    "left_gripper_rgb": "left_gripper_palm",
    "right_gripper_rgb": "right_gripper_palm",
}


class Mini3RgbCamerasTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))

    def test_three_fixed_rgb_cameras_have_the_requested_mounts(self) -> None:
        for name, parent in CAMERA_BODIES.items():
            with self.subTest(camera=name):
                camera = self.model.camera(name)
                self.assertEqual(int(camera.bodyid[0]), self.model.body(parent).id)
                self.assertEqual(int(camera.mode[0]), mujoco.mjtCamLight.mjCAMLIGHT_FIXED)
                self.assertAlmostEqual(float(camera.fovy[0]), 80.0)
                np.testing.assert_array_equal(self.model.cam_resolution[camera.id], [640, 480])

    def test_head_is_45_degrees_down_and_all_cameras_follow_the_robot(self) -> None:
        data = mujoco.MjData(self.model)
        for rotated in (False, True):
            mujoco.mj_resetDataKeyframe(self.model, data, self.model.key("stand").id)
            if rotated:
                # Include roll and pitch: checking only yaw would miss a world-fixed camera.
                quaternion = np.array([0.8, 0.2, -0.3, 0.4])
                data.qpos[3:7] = quaternion / np.linalg.norm(quaternion)
                data.qpos[:3] += [0.7, -0.2, 0.5]
                for side in ("left", "right"):
                    data.qpos[self.model.joint(f"{side}_elbow_pitch_joint").qposadr[0]] = 1.1
                    data.qpos[self.model.joint(f"{side}_shoulder_pitch_joint").qposadr[0]] = -0.4
            mujoco.mj_forward(self.model, data)
            for name, parent in CAMERA_BODIES.items():
                with self.subTest(camera=name, rotated=rotated):
                    camera = self.model.camera(name)
                    body_rotation = data.body(parent).xmat.reshape(3, 3)
                    camera_rotation = data.camera(name).xmat.reshape(3, 3)
                    local_position = body_rotation.T @ (data.camera(name).xpos - data.body(parent).xpos)
                    np.testing.assert_allclose(local_position, camera.pos, atol=1e-10)
                    # Camera image-right must stay along parent -Y, preventing a rolled image.
                    np.testing.assert_allclose(body_rotation.T @ camera_rotation[:, 0], [0, -1, 0], atol=1e-10)
                    if name == "head_rgb":
                        forward = body_rotation.T @ -camera_rotation[:, 2]
                        np.testing.assert_allclose(forward, [math.sqrt(0.5), 0, -math.sqrt(0.5)], atol=1e-10)
                    else:
                        side = name.split("_", 1)[0]
                        to_tcp = data.site(f"{side}_gripper_tcp").xpos - data.camera(name).xpos
                        np.testing.assert_allclose(-camera_rotation[:, 2], to_tcp / np.linalg.norm(to_tcp), atol=1e-10)

    def test_sensors_and_visual_mounts_preserve_all_body_mass_and_inertia(self) -> None:
        root = ET.parse(ROBOT_XML).getroot()
        for body in root.findall(".//body"):
            for child in list(body):
                name = child.get("name", "")
                if any(name == camera or name.startswith(camera + "_") for camera in CAMERA_BODIES):
                    body.remove(child)
        compiler = root.find("compiler")
        assert compiler is not None
        compiler.set("meshdir", str(ASSET_DIR / compiler.get("meshdir", ".")))
        without_cameras = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.assertEqual((self.model.nq, self.model.nv, self.model.nu), (32, 31, 23))
        for attribute in ("nq", "nv", "nu", "nbody"):
            self.assertEqual(getattr(self.model, attribute), getattr(without_cameras, attribute))
        for attribute in ("body_mass", "body_inertia", "body_ipos", "body_iquat"):
            np.testing.assert_array_equal(getattr(self.model, attribute), getattr(without_cameras, attribute))
        original = mujoco.MjModel.from_xml_path(str(ASSET_DIR / "mini3.xml"))
        self.assertAlmostEqual(float(self.model.body_mass.sum() - original.body_mass.sum()), 0.4, places=10)

    def test_camera_housings_are_visual_only_and_leave_the_optical_ray_clear(self) -> None:
        data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, data, self.model.key("stand").id)
        mujoco.mj_forward(self.model, data)
        for index in range(self.model.ngeom):
            geom = self.model.geom(index)
            if any(geom.name.startswith(camera + "_") for camera in CAMERA_BODIES):
                self.assertEqual(int(geom.contype[0]), 0)
                self.assertEqual(int(geom.conaffinity[0]), 0)
        for name in CAMERA_BODIES:
            origin = data.camera(name).xpos
            forward = -data.camera(name).xmat.reshape(3, 3)[:, 2]
            hit = np.array([-1], dtype=np.int32)
            distance = mujoco.mj_ray(self.model, data, origin, forward, None, True, -1, hit)
            # The lens plane is ahead of its housing; no shell or mount blocks a nearby target.
            self.assertTrue(distance < 0 or distance > 0.025, (name, distance, hit))


if __name__ == "__main__":
    unittest.main()
