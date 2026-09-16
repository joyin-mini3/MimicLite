from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from audit_mini3_arm_collisions import arm_collision_pairs
from mini3_pick_carry_7dof import ArticulatedPolicyScene


XML = """
<mujoco><worldbody>
  <geom name="pick_table_top" type="box" size=".15 .15 .01"/>
  <body name="right_forearm_extension" pos="0 0 .019"><freejoint/>
    <geom name="right_forearm_extension_geom" type="box" size=".01 .01 .01"/>
  </body>
  <body name="right_gripper_finger_positive" pos="1 0 .02"><freejoint/>
    <geom name="right_gripper_finger_positive_geom" type="box" size=".02 .005 .014"/>
  </body>
  <body name="pick_cube_3" pos="1 .02 .02"><freejoint/>
    <geom name="pick_cube_3_geom" type="box" size=".02 .02 .02"/>
  </body>
</worldbody></mujoco>
"""


def monitor_fixture() -> ArticulatedPolicyScene:
    # Exercise the real monitor and actual MuJoCo contact manifold, independent
    # of private checkpoints, inference, motor initialization and the full scene.
    runtime = ArticulatedPolicyScene.__new__(ArticulatedPolicyScene)
    runtime.model = mujoco.MjModel.from_xml_string(XML)
    runtime.data = mujoco.MjData(runtime.model)
    runtime.data.time = .4
    mujoco.mj_forward(runtime.model, runtime.data)
    runtime.unexpected_pairs = set(arm_collision_pairs(runtime.model))
    runtime.pregrasp_pairs = {tuple(sorted((runtime.model.geom("pick_cube_3_geom").id,
                                          runtime.model.geom("right_gripper_finger_positive_geom").id)))}
    runtime.collision_stop_depth = .002
    runtime.collision_phase = "CLOSE"
    runtime.contact_checks = 0
    runtime.unexpected_contacts = {}
    return runtime


def physical_snapshot(data: mujoco.MjData) -> dict[str, np.ndarray | float]:
    result = {name: getattr(data, name).copy() for name in
              ("qpos", "qvel", "qacc", "qacc_warmstart", "qfrc_bias", "ctrl", "xpos")}
    result["time"] = float(data.time)
    return result


class Mini3ArmContactMonitorTest(unittest.TestCase):
    def test_same_target_contact_stops_early_approach_but_is_allowed_during_grasp(self) -> None:
        for phase in ("APPROACH", "CLEAR_ARM", "REACH", "LOWER", "CLOSE", "LIFT", "CARRY", "PLACE", "RELEASE"):
            with self.subTest(phase=phase):
                runtime = monitor_fixture()
                runtime.data.qpos[2] = .05  # Isolate the real finger/blue-cube manifold.
                runtime.collision_phase = phase
                mujoco.mj_forward(runtime.model, runtime.data)
                pair = next(iter(runtime.pregrasp_pairs))
                contacts = [contact for contact in runtime.data.contact
                            if tuple(sorted((int(contact.geom1), int(contact.geom2)))) == pair]
                self.assertGreater(len(contacts), 1)
                self.assertLess(min(contact.dist for contact in contacts), -runtime.collision_stop_depth)
                before = physical_snapshot(runtime.data)
                if phase in ("APPROACH", "CLEAR_ARM", "REACH"):
                    with self.assertRaisesRegex(RuntimeError, "Unexpected arm collision.*finger"):
                        runtime.check_arm_contacts()
                    self.assertEqual(set(runtime.unexpected_contacts), {pair})
                    self.assertEqual(runtime.unexpected_contacts[pair]["first_phase"], phase)
                    self.assertEqual(runtime.unexpected_contacts[pair]["physics_steps"], 1)
                else:
                    runtime.check_arm_contacts()
                    self.assertEqual(runtime.unexpected_contacts, {})
                self.assertEqual(runtime.contact_checks, 1)
                for name, expected in before.items():
                    np.testing.assert_array_equal(getattr(runtime.data, name), expected)

    def test_contact_manifold_counts_once_without_advancing_or_solving_physics(self) -> None:
        runtime = monitor_fixture()
        pair = tuple(sorted((runtime.model.geom("right_forearm_extension_geom").id,
                             runtime.model.geom("pick_table_top").id)))
        manifold = [contact for contact in runtime.data.contact
                    if tuple(sorted((int(contact.geom1), int(contact.geom2)))) == pair]
        self.assertGreater(len(manifold), 1)
        before = physical_snapshot(runtime.data)
        with patch("mujoco.mj_step", side_effect=AssertionError("monitor advanced physics")), \
             patch("mujoco.mj_forward", side_effect=AssertionError("monitor reran dynamics")):
            runtime.check_arm_contacts()
        self.assertEqual(runtime.contact_checks, 1)
        self.assertEqual(set(runtime.unexpected_contacts), {pair})
        entry = runtime.unexpected_contacts[pair]
        self.assertEqual(entry["physics_steps"], 1)
        self.assertAlmostEqual(entry["maximum_penetration_m"], .001, places=7)
        self.assertEqual(entry["first_phase"], "CLOSE")
        self.assertEqual(entry["first_time_s"], .4)
        for name, expected in before.items():
            np.testing.assert_array_equal(getattr(runtime.data, name), expected)

    def test_legal_grasp_is_ignored_but_deep_forearm_table_contact_stops(self) -> None:
        runtime = monitor_fixture()
        runtime.data.qpos[2] = .05  # Clear the forearm while preserving finger/cube contact.
        mujoco.mj_forward(runtime.model, runtime.data)
        grasp_pair = {runtime.model.geom("right_gripper_finger_positive_geom").id,
                      runtime.model.geom("pick_cube_3_geom").id}
        grasp_contacts = [contact for contact in runtime.data.contact
                          if {int(contact.geom1), int(contact.geom2)} == grasp_pair]
        self.assertTrue(grasp_contacts)
        self.assertLess(min(contact.dist for contact in grasp_contacts), -.002)
        runtime.check_arm_contacts()
        self.assertEqual(runtime.unexpected_contacts, {})

        runtime.data.qpos[2] = .015  # Actual 5 mm forearm/table penetration.
        mujoco.mj_forward(runtime.model, runtime.data)
        before = physical_snapshot(runtime.data)
        with self.assertRaisesRegex(RuntimeError, "Unexpected arm collision.*forearm"):
            runtime.check_arm_contacts()
        self.assertEqual(runtime.contact_checks, 2)
        entry = next(iter(runtime.unexpected_contacts.values()))
        self.assertGreater(entry["maximum_penetration_m"], runtime.collision_stop_depth)
        self.assertEqual(entry["physics_steps"], 1)
        for name, expected in before.items():
            np.testing.assert_array_equal(getattr(runtime.data, name), expected)


if __name__ == "__main__":
    unittest.main()
