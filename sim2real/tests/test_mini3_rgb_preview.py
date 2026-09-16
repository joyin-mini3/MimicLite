from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "preview_mini3_pick_scene_rgb", ROOT / "preview_mini3_pick_scene.py"
)
assert spec is not None and spec.loader is not None
preview = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preview)


class Mini3RgbPreviewTest(unittest.TestCase):
    def setUp(self) -> None:
        # Deliberately put cameras in a different order from the viewer shortcuts.
        self.model = mujoco.MjModel.from_xml_string("""
            <mujoco>
              <worldbody>
                <camera name="unrelated" pos="0 0 3"/>
                <body name="base_link" pos="1 2 0.8">
                  <freejoint/>
                  <geom type="box" size="0.1 0.1 0.1" mass="1"/>
                  <camera name="right_gripper_rgb" pos="0 -0.2 0"/>
                  <camera name="head_rgb" pos="0 0 0.3"/>
                  <camera name="left_gripper_rgb" pos="0 0.2 0"/>
                </body>
              </worldbody>
            </mujoco>
        """)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.camera = mujoco.MjvCamera()

    def test_rgb_views_select_named_fixed_camera(self) -> None:
        for name in ("head_rgb", "left_gripper_rgb", "right_gripper_rgb"):
            with self.subTest(camera=name):
                preview.configure_camera(self.camera, self.model, self.data, name)
                self.assertEqual(self.camera.type, mujoco.mjtCamera.mjCAMERA_FIXED)
                self.assertEqual(self.camera.fixedcamid, self.model.camera(name).id)

    def test_overview_restores_free_camera_facing_robot_forward(self) -> None:
        preview.configure_camera(self.camera, self.model, self.data, "head_rgb")
        # Turn the robot 90 degrees and move it before returning to the overview.
        self.data.qpos[:3] = [2.0, -1.0, 0.8]
        self.data.qpos[3:7] = [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]
        mujoco.mj_forward(self.model, self.data)
        position_before = self.data.qpos.copy()
        preview.configure_camera(self.camera, self.model, self.data, "overview")
        self.assertEqual(self.camera.type, mujoco.mjtCamera.mjCAMERA_FREE)
        self.assertEqual(self.camera.fixedcamid, -1)
        np.testing.assert_allclose(self.camera.lookat, [2.0, -0.75, 0.3])
        self.assertAlmostEqual(self.camera.azimuth, 225.0)
        self.assertLess(self.camera.elevation, 0.0)
        self.assertGreater(self.camera.distance, 0.0)
        np.testing.assert_array_equal(self.data.qpos, position_before)

    def test_missing_rgb_camera_explains_how_to_regenerate_assets(self) -> None:
        legacy_model = mujoco.MjModel.from_xml_string("""
            <mujoco><worldbody><body name="base_link"/></worldbody></mujoco>
        """)
        legacy_data = mujoco.MjData(legacy_model)
        mujoco.mj_forward(legacy_model, legacy_data)
        for name in ("head_rgb", "left_gripper_rgb", "right_gripper_rgb"):
            with self.subTest(camera=name):
                with self.assertRaisesRegex(
                    ValueError, f"{name}.*Regenerate.*mini3_gripper.xml.*scene_pick_place.xml"
                ):
                    preview.configure_camera(self.camera, legacy_model, legacy_data, name)
        # Existing custom scenes can still use the ordinary free camera.
        preview.configure_camera(self.camera, legacy_model, legacy_data, "overview")
        self.assertEqual(self.camera.type, mujoco.mjtCamera.mjCAMERA_FREE)

    def test_unknown_view_preserves_previous_camera(self) -> None:
        preview.configure_camera(self.camera, self.model, self.data, "left_gripper_rgb")
        selected = self.camera.fixedcamid
        with self.assertRaisesRegex(ValueError, "Unknown camera view: wrist_rgb"):
            preview.configure_camera(self.camera, self.model, self.data, "wrist_rgb")
        self.assertEqual(self.camera.type, mujoco.mjtCamera.mjCAMERA_FIXED)
        self.assertEqual(self.camera.fixedcamid, selected)


if __name__ == "__main__":
    unittest.main()
