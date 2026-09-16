from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("preview_mini3_pick_scene", ROOT / "preview_mini3_pick_scene.py")
assert spec is not None and spec.loader is not None
preview = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preview)
preview.setup_paths()


class Mini3PickPreviewTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(preview.DEFAULT_SCENE))
        self.data = mujoco.MjData(self.model)
        self.controls = preview.SceneControls(self.model, self.data, support=True)

    def press(self, *keys: int) -> None:
        with redirect_stdout(io.StringIO()):
            for key in keys:
                self.controls.key(key)

    def test_original_motor_mapping_skips_fingers_and_free_objects(self) -> None:
        original = mujoco.MjModel.from_xml_path(str(preview.DEFAULT_SCENE.with_name("mini3.xml")))
        expected_names = [original.joint(index).name for index in range(1, original.njnt)]
        self.assertEqual(self.controls.names, expected_names)
        self.assertEqual(len(set(self.controls.qpos_ids)), 21)
        unrelated_addresses = set(range(self.model.nq)) - set(self.controls.qpos_ids)
        for side in ("left", "right"):
            for finger in ("finger", "follower"):
                address = self.model.joint(f"{side}_gripper_{finger}_joint").qposadr[0]
                self.assertIn(address, unrelated_addresses)
        for index in range(1, 4):
            address = self.model.joint(f"pick_cube_{index}_free").qposadr[0]
            self.assertTrue(set(range(address, address + 7)).issubset(unrelated_addresses))
        before = self.data.qpos.copy()
        values = np.arange(21, dtype=float) * 0.01
        self.data.qpos[self.controls.qpos_ids] = values
        for name, value in zip(expected_names, values):
            self.assertAlmostEqual(self.data.qpos[self.model.joint(name).qposadr[0]], value)
        unrelated = sorted(unrelated_addresses)
        np.testing.assert_array_equal(self.data.qpos[unrelated], before[unrelated])
        actuator_joints = self.model.actuator_trnid[self.controls.motor_ids, 0]
        np.testing.assert_array_equal(actuator_joints, self.controls.joint_ids)

    def test_grippers_are_independent_and_not_overwritten_by_motor_step(self) -> None:
        left = self.model.actuator("left_gripper_ctrl").id
        right = self.model.actuator("right_gripper_ctrl").id
        self.press(ord("X"))
        self.assertEqual(self.data.ctrl[left], 0.0)
        self.assertEqual(self.data.ctrl[right], 0.035)
        self.press(ord("V"), ord("Z"))
        self.assertEqual(self.data.ctrl[left], 0.035)
        self.assertEqual(self.data.ctrl[right], 0.0)
        torque = np.linspace(-0.5, 0.5, 21)
        with patch.object(self.controls.motor, "compute", return_value=torque) as compute:
            before_position = self.data.qpos[self.controls.qpos_ids].copy()
            before_velocity = self.data.qvel[self.controls.qvel_ids].copy()
            self.controls.step()
            np.testing.assert_array_equal(compute.call_args.args[1], before_position)
            np.testing.assert_array_equal(compute.call_args.args[2], before_velocity)
        np.testing.assert_allclose(self.data.ctrl[self.controls.motor_ids], torque)
        self.assertEqual(self.data.ctrl[left], 0.035)
        self.assertEqual(self.data.ctrl[right], 0.0)
        self.press(ord("C"))
        self.assertEqual(self.data.ctrl[right], 0.035)

    def test_reset_restores_objects_targets_grippers_and_motor_state(self) -> None:
        self.controls.targets[self.controls.selected] += 0.2
        for _ in range(10):
            self.controls.step()
        self.assertGreater(float(np.linalg.norm(self.controls.motor.applied_torque)), 0.0)
        cube_address = self.model.joint("pick_cube_1_free").qposadr[0]
        self.data.qpos[cube_address:cube_address + 3] += [0.3, 0.1, 0.2]
        self.press(ord("X"), ord("V"), ord("W"), ord("I"), ord("B"))
        self.controls.reset()
        home = self.model.key("home")
        np.testing.assert_array_equal(self.data.qpos, home.qpos)
        np.testing.assert_array_equal(self.data.qvel, home.qvel)
        np.testing.assert_array_equal(self.data.ctrl, home.ctrl)
        np.testing.assert_array_equal(self.controls.targets, home.qpos[self.controls.qpos_ids])
        np.testing.assert_array_equal(self.controls.motor.applied_torque, np.zeros(21))
        np.testing.assert_array_equal(self.controls.motor.response_torque, np.zeros(21))
        np.testing.assert_array_equal(self.data.mocap_pos.ravel(), home.mpos)
        np.testing.assert_array_equal(self.data.mocap_quat.ravel(), home.mquat)
        self.assertFalse(self.data.eq_active[self.controls.support_id])
        self.assertEqual(self.data.time, 0.0)

    def test_support_toggle_matches_current_root_without_teleporting(self) -> None:
        self.press(ord("B"))
        self.data.qpos[:3] += [0.12, -0.05, 0.08]
        self.data.qpos[3:7] = [np.cos(0.2), 0.0, 0.0, np.sin(0.2)]
        mujoco.mj_forward(self.model, self.data)
        before = self.data.qpos.copy()
        root_position = self.data.body("base_link").xpos.copy()
        root_quaternion = self.data.body("base_link").xquat.copy()
        self.press(ord("B"))
        self.assertTrue(self.data.eq_active[self.controls.support_id])
        np.testing.assert_array_equal(self.data.qpos, before)
        np.testing.assert_allclose(self.data.mocap_pos[self.controls.mocap_id], root_position)
        np.testing.assert_allclose(self.data.mocap_quat[self.controls.mocap_id], root_quaternion)
        mujoco.mj_forward(self.model, self.data)
        weld_rows = ((self.data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY)
                     & (self.data.efc_id == self.controls.support_id))
        self.assertEqual(int(weld_rows.sum()), 6)
        np.testing.assert_allclose(self.data.efc_pos[weld_rows], 0.0, atol=1.0e-10)
        self.press(ord("B"))
        np.testing.assert_array_equal(self.data.qpos, before)
        support_target = self.data.mocap_pos.copy()
        self.press(ord("W"), ord("J"))
        np.testing.assert_array_equal(self.data.mocap_pos, support_target)

    def test_joint_target_controls_respect_limits(self) -> None:
        selected = self.controls.selected
        lower, upper = self.model.jnt_range[self.controls.joint_ids[selected]]
        self.controls.targets[selected] = upper - 0.001
        self.press(265)
        self.assertEqual(self.controls.targets[selected], upper)
        self.controls.targets[selected] = lower + 0.001
        self.press(264)
        self.assertEqual(self.controls.targets[selected], lower)
        self.controls.selected = 0
        self.press(ord("["))
        self.assertEqual(self.controls.selected, 20)
        self.press(ord("]"))
        self.assertEqual(self.controls.selected, 0)


if __name__ == "__main__":
    unittest.main()
