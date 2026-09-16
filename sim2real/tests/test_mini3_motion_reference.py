from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sim2real"))
sys.path.insert(0, str(ROOT / ".cache/mimiclite_sim2sim/deps"))
spec = importlib.util.spec_from_file_location("mini3_motion_reference", ROOT / "mini3_motion_reference.py")
assert spec is not None and spec.loader is not None
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


class Mini3MotionReferenceTest(unittest.TestCase):
    def test_world_transform_preserves_shape_and_aligns_travel(self) -> None:
        poses = np.zeros((3, 28))
        poses[:, :3] = [[1, 2, .45], [1, 3, .46], [1, 4, .45]]
        poses[:, 3] = 1
        poses[:, 7:] = np.arange(21)
        placed = reference.place_qpos(poses, origin_xy=(-3, .2), yaw=0, align_travel=True)
        np.testing.assert_allclose(placed[:, :3], [[-3, .2, .45], [-2, .2, .46], [-1, .2, .45]])
        np.testing.assert_array_equal(placed[:, 7:], poses[:, 7:])
        self.assertAlmostEqual(reference.quaternion_yaw(placed[0, 3:7]), -np.pi / 2)
        np.testing.assert_array_equal(poses[0, :3], [1, 2, .45])

    def test_pose_blend_takes_short_quaternion_path(self) -> None:
        start = np.zeros(28)
        start[3] = 1
        end = start.copy()
        end[0] = 1
        end[3] = -1  # Same orientation with opposite quaternion sign.
        blended = reference.blend_poses(start, end, 1.0)
        np.testing.assert_allclose(blended[:, 3:7], np.tile([1, 0, 0, 0], (51, 1)))
        self.assertLess(blended[1, 0], .002)
        self.assertAlmostEqual(blended[-1, 0], 1)

    def test_reference_fk_uses_original_names_and_zero_hold_velocity(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(reference.MINI3_XML))
        poses = np.tile(model.qpos0, (3, 1))
        poses[:, 0] = 2
        motion = reference.MotionReference(poses, model=model)
        root = motion.body_names.index("base_link")
        np.testing.assert_allclose(motion._storage["body_pos_w"][:, root, 0], 2)
        np.testing.assert_array_equal(motion._storage["joint_vel"], np.zeros((3, 21)))
        self.assertEqual(len(motion.joint_names), 21)
        self.assertNotIn("left_gripper_finger_joint", motion.joint_names)

    def test_reference_velocity_and_negative_future_frames_match_fk(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(reference.MINI3_XML))
        poses = np.tile(model.qpos0, (3, 1))
        poses[:, 0] = [0, .02, .04]
        motion = reference.MotionReference(poses, model=model)
        data = mujoco.MjData(model)
        data.qpos[:] = poses[0]
        data.qvel[0] = 1.0
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(motion._storage["body_lin_vel_w"][0], data.cvel[:, 3:])
        sampled = motion.get_slice(np.array([0]), np.array([0]), np.array([-8, 0, 1, 10]))
        root = motion.body_names.index("base_link")
        np.testing.assert_allclose(sampled.body_pos_w[0, :, root, 0], [0, 0, .02, .04])
        np.testing.assert_array_equal(sampled.step, [[0, 0, 1, 2]])

    def test_shortening_retains_start_stop_and_removes_matching_gait_cycles(self) -> None:
        poses = np.zeros((500, 28))
        poses[:, 0] = np.arange(500) * .01
        poses[:, 2] = .45
        poses[:, 3] = 1
        poses[:, 7:] = np.sin(np.arange(500)[:, None] * 2 * np.pi / 50) * .2
        shortened, metadata = reference.shorten_walk(poses, 2.99)
        np.testing.assert_array_equal(shortened[0], poses[0])
        np.testing.assert_array_equal(shortened[-1, 2:], poses[-1, 2:])
        self.assertLess(abs(metadata["distance"] - 2.99), .02)
        self.assertLess(metadata["leg_pose_mismatch"], .01)
        self.assertGreater(metadata["removed_frames"], 100)

    def test_install_preserves_policy_action_and_history(self) -> None:
        action = np.arange(21)
        history = np.ones((7, 21))
        state = SimpleNamespace(joint_names=["joint"], motion_ids=np.array([0]), motion_t=np.array([50]))
        updates = []
        state._update_motion_data = lambda: updates.append(int(state.motion_t[0]))
        policy = SimpleNamespace(state_processor=state, state_dict={"action": action, "paused": True}, history=history)
        motion = SimpleNamespace(joint_names=["joint"], body_names=["base"], num_steps=100)
        reference.install_reference(policy, motion)
        self.assertIs(policy.state_dict["action"], action)
        self.assertIs(policy.history, history)
        self.assertFalse(policy.state_dict["paused"])
        self.assertEqual(updates, [-1])


if __name__ == "__main__":
    unittest.main()
