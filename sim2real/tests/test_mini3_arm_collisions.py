from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("arm_collision_audit", ROOT / "audit_mini3_arm_collisions.py")
assert spec is not None and spec.loader is not None
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

XML = """
<mujoco><worldbody>
  <body name="base_link" pos="0 0 .5"><freejoint/><geom name="base_link_collision" size=".03"/>
    <body name="right_shoulder_yaw_link"><joint axis="0 1 0"/>
      <geom name="right_shoulder_yaw_link_collision" size=".05"/>
      <body name="right_elbow_pitch_link" pos=".02 0 0"><joint axis="0 1 0"/>
        <geom name="right_elbow_pitch_link_collision" size=".03"/>
        <body name="right_forearm_extension" pos=".02 0 0">
          <geom name="right_forearm_extension_geom" size=".03"/>
          <body name="right_gripper_palm" pos=".1 0 0">
            <geom name="right_gripper_palm_geom" type="box" size=".01 .03 .01"/>
            <body name="right_gripper_finger_positive" pos=".04 .04 0"><joint type="slide" axis="0 1 0"/>
              <geom name="right_gripper_finger_positive_geom" type="box" size=".02 .005 .014"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </body>
  <body name="pick_cube_3" pos=".18 0 .5"><freejoint/><geom name="pick_cube_3_geom" type="box" size=".02 .02 .02"/></body>
  <body name="pick_table" pos=".4 0 .2"><geom name="pick_table_top" type="box" size=".1 .1 .02"/></body>
  <geom name="right_gripper_rgb_housing" size=".01" contype="0" conaffinity="0"/>
</worldbody></mujoco>
"""


class Mini3ArmCollisionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = mujoco.MjModel.from_xml_string(XML)

    def ids(self, first: str, second: str) -> tuple[int, int]:
        return self.model.geom(first).id, self.model.geom(second).id

    def test_only_direct_mechanical_neighbours_are_allowed(self) -> None:
        direct = self.ids("right_elbow_pitch_link_collision", "right_forearm_extension_geom")
        distal = self.ids("right_shoulder_yaw_link_collision", "right_forearm_extension_geom")
        self.assertEqual(audit.pair_classification(self.model, *direct), "allowed_mechanical_interface")
        self.assertEqual(audit.pair_classification(self.model, *distal), "unexpected_arm_self")
        self.assertIn("welded_parent_child", audit.collision_filter_reasons(self.model, *distal))
        self.assertIn(tuple(sorted(distal)), audit.arm_collision_pairs(self.model, "right"))

    def test_grasp_permission_is_fingers_only_and_visuals_are_ignored(self) -> None:
        finger = self.ids("right_gripper_finger_positive_geom", "pick_cube_3_geom")
        palm = self.ids("right_gripper_palm_geom", "pick_cube_3_geom")
        pairs = audit.arm_collision_pairs(self.model, "right")
        self.assertNotIn(tuple(sorted(finger)), pairs)
        self.assertIn(tuple(sorted(palm)), pairs)
        camera = self.model.geom("right_gripper_rgb_housing").id
        self.assertFalse(any(camera in pair for pair in pairs))

    def test_sat_returns_positive_box_gap_without_changing_physics(self) -> None:
        data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, data)
        original_flags = int(self.model.opt.disableflags)
        original_qpos = data.qpos.copy()
        first, second = self.ids("right_gripper_finger_positive_geom", "pick_cube_3_geom")
        distance = audit.collision_distance(self.model, data, first, second, distmax=.1)
        self.assertAlmostEqual(distance, .015, places=7)
        self.assertEqual(int(self.model.opt.disableflags), original_flags)
        np.testing.assert_array_equal(data.qpos, original_qpos)

    def test_sat_and_capsule_queries_match_actual_penetrating_contacts(self) -> None:
        for shape in ('type="box" size=".03 .01 .03"',
                      'type="capsule" fromto="-.03 0 0 .03 0 0" size=".03"'):
            with self.subTest(shape=shape):
                model = mujoco.MjModel.from_xml_string(f"""
                <mujoco><worldbody>
                  <geom name="table" type="box" size=".1 .1 .02"/>
                  <body pos="0 0 .045"><freejoint/><geom name="finger" {shape}/></body>
                </worldbody></mujoco>""")
                data = mujoco.MjData(model)
                mujoco.mj_forward(model, data)
                actual = min(float(contact.dist) for contact in data.contact)
                measured = audit.collision_distance(model, data, 0, 1, distmax=.03)
                self.assertAlmostEqual(measured, actual, places=7)
                self.assertAlmostEqual(measured, -.005, places=7)

    def test_offline_audit_detects_filtered_penetration_and_duration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            scene, trajectory = directory / "scene.xml", directory / "trajectory.npz"
            scene.write_text(XML)
            np.savez(trajectory, time=np.array([0., .02, .04]), phase=np.array(["REACH"] * 3),
                     qpos=np.tile(self.model.qpos0, (3, 1)), qvel=np.zeros((3, self.model.nv)))
            original = trajectory.read_bytes()
            report = audit.audit(trajectory, scene)
            target = next(item for item in report["unexpected_collisions"]
                          if set(item["geoms"]) == {"right_shoulder_yaw_link_collision", "right_forearm_extension_geom"})
            self.assertIn("welded_parent_child", target["filter_reasons"])
            self.assertAlmostEqual(target["maximum_penetration_m"], .04, places=7)
            self.assertEqual(target["first_penetration_time_s"], 0)
            self.assertEqual(target["maximum_consecutive_penetrating_frames"], 3)
            self.assertEqual(target["actual_contact_frames"], 0)
            self.assertEqual(trajectory.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
