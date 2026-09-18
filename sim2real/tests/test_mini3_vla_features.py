from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mini3_vla_features import FEATURE_NAMES, FEATURE_UNITS, extract_features  # noqa: E402


def quaternion(roll: float = 0, pitch: float = 0, yaw: float = 0) -> np.ndarray:
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


class Mini3VlaFeaturesTest(unittest.TestCase):
    def setUp(self) -> None:
        joints = [name for name in FEATURE_NAMES[:29] if name.endswith("_joint")]
        added = [name for name in joints if any(part in name for part in ("elbow_yaw", "wrist_"))]
        originals = [name for name in joints if name not in added]
        # Deliberately use different orders and nonzero root addresses. Export
        # must use names/metadata, not assume a motor/model array layout.
        self.metadata = {
            "policy_joint_names": originals[::-1], "extra_joint_names": added[::-1],
            "gripper_joint_names": ["right_gripper_finger_joint", "left_gripper_finger_joint"],
            "policy_qpos_indices": list(range(9, 30)), "extra_qpos_indices": list(range(30, 36)),
            "gripper_qpos_indices": [37, 36], "root_qpos_start": 2, "root_qvel_start": 3,
            "reference_joint_names": originals[5:] + originals[:5],
            "reference_body_names": ["world", "other_body", "base_link"],
            "reference_step_offsets": [-2, 0, 1], "control_hz": 50,
        }
        self.samples = {
            "qpos": np.zeros((3, 40)), "qvel": np.zeros((3, 38)),
            "reference_joint_pos": np.full((3, 3, 21), -999.), "extra_q": np.zeros((3, 6)),
            "gripper_opening": np.tile([.007, .035], (3, 1)),
            "reference_body_pos_w": np.zeros((3, 3, 3, 3)),
            "reference_body_quat_w": np.zeros((3, 3, 3, 4)),
            "reference_body_lin_vel_w": np.zeros((3, 3, 3, 3)),
            "reference_step_offsets": np.array([-2, 0, 1]),
            "policy_q": np.full((3, 21), 999.),
        }
        self.samples["qpos"][:, 4] = .51
        self.samples["qpos"][:, 5:9] = quaternion()
        self.samples["qpos"][:, 36:38] = [.0175, .035]
        self.samples["reference_body_quat_w"][:] = quaternion()
        self.samples["reference_body_pos_w"][:, :, 2, 2] = .49
        for kind in ("policy", "extra"):
            for name, address in zip(self.metadata[kind + "_joint_names"], self.metadata[kind + "_qpos_indices"]):
                self.samples["qpos"][:, address] = joints.index(name) * .01
        for i, name in enumerate(self.metadata["reference_joint_names"]):
            self.samples["reference_joint_pos"][:, 1, i] = joints.index(name) * .01 + .1
        for i, name in enumerate(self.metadata["extra_joint_names"]):
            self.samples["extra_q"][:, i] = joints.index(name) * .01 + .2
        self.joints = joints

    def test_schema_and_name_mapping_use_reference_for_original_and_motor_plan_for_added(self) -> None:
        state, action = extract_features(self.samples, self.metadata)
        self.assertEqual(state.shape, (3, 36))
        self.assertEqual(action.shape, (3, 36))
        self.assertEqual(len(FEATURE_UNITS), 36)
        self.assertEqual(FEATURE_NAMES[32:35], ("root_linear_velocity_heading_x",
                                               "root_linear_velocity_heading_y",
                                               "root_linear_velocity_heading_z"))
        self.assertEqual(state.dtype, np.float32)
        self.assertEqual(action.dtype, np.float32)
        for index, name in enumerate(FEATURE_NAMES[:29]):
            if name.endswith("_joint"):
                with self.subTest(joint=name):
                    actual = self.joints.index(name) * .01
                    planned = actual + (.2 if name in self.metadata["extra_joint_names"] else .1)
                    np.testing.assert_allclose(state[:, index], actual)
                    np.testing.assert_allclose(action[:, index], planned)
        np.testing.assert_allclose(state[:, 35], .51)
        np.testing.assert_allclose(action[:, 35], .49)
        self.assertTrue(np.isfinite(state).all())
        self.assertTrue(np.isfinite(action).all())

    def test_gripper_closedness_matches_measured_and_commanded_openings_and_clamps(self) -> None:
        state, action = extract_features(self.samples, self.metadata)
        np.testing.assert_allclose(state[:, [7, 15]], [[.5, 0]] * 3)
        np.testing.assert_allclose(action[:, [7, 15]], [[0, .8]] * 3)
        self.samples["qpos"][:, 36:38] = [-.001, .04]
        self.samples["gripper_opening"][:] = [-.002, .04]
        state, action = extract_features(self.samples, self.metadata)
        np.testing.assert_array_equal(state[:, [7, 15]], [[1, 0]] * 3)
        np.testing.assert_array_equal(action[:, [7, 15]], [[0, 1]] * 3)

    def test_yaw_reference_crossing_pi_has_small_signed_velocity_and_sign_invariance(self) -> None:
        self.samples["reference_body_quat_w"][:, 1, 2] = quaternion(yaw=np.pi - .01)
        self.samples["reference_body_quat_w"][:, 2, 2] = -quaternion(yaw=-np.pi + .03)
        _, action = extract_features(self.samples, self.metadata)
        np.testing.assert_allclose(action[:, 31], 2.)
        self.samples["reference_body_quat_w"][:, 2, 2] = quaternion(yaw=np.pi - .01)
        _, held = extract_features(self.samples, self.metadata)
        np.testing.assert_allclose(held[:, 31], 0, atol=1e-7)

    def test_state_yaw_rate_uses_body_angular_velocity_and_root_euler_angles(self) -> None:
        self.samples["qpos"][:, 5:9] = quaternion(roll=.3, pitch=.2, yaw=.5)
        self.samples["qvel"][:, 6:9] = [.4, .6, .8]
        state, _ = extract_features(self.samples, self.metadata)
        expected = (np.sin(.3) * .6 + np.cos(.3) * .8) / np.cos(.2)
        np.testing.assert_allclose(state[:, 29:32], [[.3, .2, expected]] * 3)

    def test_both_linear_velocities_use_measured_heading_not_reference_heading_or_tilt(self) -> None:
        self.samples["qpos"][:, 5:9] = quaternion(roll=.3, pitch=.2, yaw=np.pi / 2)
        self.samples["qvel"][:, 3:6] = [1, 2, 3]
        self.samples["reference_body_quat_w"][:, 1, 2] = quaternion(yaw=-np.pi / 2)
        self.samples["reference_body_lin_vel_w"][:, 1, 2] = [4, 5, 6]
        state, action = extract_features(self.samples, self.metadata)
        np.testing.assert_allclose(state[:, 32:35], [[2, -1, 3]] * 3)
        np.testing.assert_allclose(action[:, 32:35], [[5, -4, 6]] * 3)

    def test_extraction_owns_outputs_and_does_not_modify_source(self) -> None:
        before = {key: value.copy() for key, value in self.samples.items()}
        metadata = copy.deepcopy(self.metadata)
        state, action = extract_features(self.samples, self.metadata)
        state[:] = 0
        action[:] = 0
        self.assertEqual(metadata, self.metadata)
        for key, value in before.items():
            np.testing.assert_array_equal(self.samples[key], value)

    def test_empty_but_well_formed_recording_is_supported(self) -> None:
        samples = {key: value if key == "reference_step_offsets" else value[:0]
                   for key, value in self.samples.items()}
        state, action = extract_features(samples, self.metadata)
        self.assertEqual(state.shape, (0, 36))
        self.assertEqual(action.shape, (0, 36))

    def test_nonfinite_zero_quaternion_and_singular_root_are_rejected(self) -> None:
        for key in ("qpos", "qvel", "reference_joint_pos", "extra_q", "gripper_opening",
                    "reference_body_pos_w", "reference_body_quat_w", "reference_body_lin_vel_w"):
            with self.subTest(array=key):
                samples = {name: value.copy() for name, value in self.samples.items()}
                samples[key].flat[0] = np.nan
                with self.assertRaisesRegex(ValueError, "finite"):
                    extract_features(samples, self.metadata)
        self.samples["qpos"][:, 5:9] = 0
        with self.assertRaisesRegex(ValueError, "zero quaternion"):
            extract_features(self.samples, self.metadata)
        self.samples["qpos"][:, 5:9] = quaternion(pitch=np.pi / 2)
        with self.assertRaisesRegex(ValueError, "singularity"):
            extract_features(self.samples, self.metadata)

    def test_ambiguous_or_misaligned_metadata_is_rejected(self) -> None:
        for patch in ({"reference_step_offsets": [-2, 0, 2]},
                      {"reference_body_names": ["world", "other_body", "pelvis"]},
                      {"root_qpos_start": 39}, {"control_hz": 0},
                      {"gripper_qpos_indices": [10, 36]}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                extract_features(self.samples, self.metadata | patch)
        samples = dict(self.samples, reference_step_offsets=np.array([0, 1, 2]))
        with self.assertRaisesRegex(ValueError, "disagree"):
            extract_features(samples, self.metadata)


if __name__ == "__main__":
    unittest.main()
