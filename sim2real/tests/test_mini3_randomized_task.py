from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

from mini3_randomized_task import generate_episode, retarget_reference


SCENE = Path(__file__).resolve().parents[2] / "any4hdmi/assets/robots/mini3_mjlab/scene_pick_carry_7dof.xml"


class RandomizedEpisodeTest(unittest.TestCase):
    def test_seed_replays_layout_and_target_and_home_matches_body_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path, first = generate_episode(SCENE, Path(directory), seed=42)
            _, second = generate_episode(SCENE, Path(directory), seed=42)
            self.assertEqual(first, second)
            model = mujoco.MjModel.from_xml_path(str(path))
            data = mujoco.MjData(model)
            mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
            mujoco.mj_forward(model, data)
            for cube in first["cubes"]:
                body = model.body(cube["body"]).id
                np.testing.assert_allclose(data.xpos[body], cube["position"])
                np.testing.assert_allclose(data.xquat[body], cube["quaternion_wxyz"])
                np.testing.assert_allclose(model.body_pos[body], cube["position"])
            target = data.xpos[model.body(first["target_body"]).id]
            self.assertAlmostEqual(target[0] - data.qpos[0], 3.)

    def test_sampling_keeps_rotated_cubes_on_table_separated_and_selects_all_colors(self):
        targets, positions = set(), set()
        with tempfile.TemporaryDirectory() as directory:
            for seed in range(24):
                path, episode = generate_episode(SCENE, Path(directory), seed=seed)
                targets.add(episode["target_color"])
                positions.add(tuple(episode["cubes"][0]["position"]))
                model = mujoco.MjModel.from_xml_path(str(path))
                data = mujoco.MjData(model)
                mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
                mujoco.mj_forward(model, data)
                top = model.geom("pick_table_top").id
                bounds = []
                for cube in episode["cubes"]:
                    geom = model.geom(cube["body"] + "_geom").id
                    half = np.abs(data.geom_xmat[geom].reshape(3, 3)) @ model.geom_size[geom]
                    center = data.geom_xpos[geom]
                    self.assertTrue(np.all(np.abs(center[:2] - data.geom_xpos[top, :2])
                                           + half[:2] + .003 < model.geom_size[top, :2]))
                    self.assertAlmostEqual(center[2] - half[2], .5005)
                    bounds.append((center, half))
                for index, (center, half) in enumerate(bounds):
                    for other, extent in bounds[index + 1:]:
                        self.assertTrue(np.any(np.abs(center[:2] - other[:2]) > half[:2] + extent[:2] + .02))
            self.assertEqual(targets, {"red", "green", "blue"})
            self.assertEqual(len(positions), 24)

    def test_target_override_does_not_change_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            episodes = [generate_episode(SCENE, Path(directory), seed=6, target=color)[1]
                        for color in ("red", "green", "blue")]
            self.assertEqual([item["target_index"] for item in episodes], [1, 2, 3])
            self.assertEqual(episodes[0]["cubes"], episodes[1]["cubes"])
            self.assertEqual(episodes[0]["cubes"], episodes[2]["cubes"])

    def test_retarget_preserves_original_joint_references_and_fixed_basket_endpoint(self):
        times = np.arange(11, dtype=float)
        poses = np.tile(np.arange(28, dtype=float), (11, 1))
        source = np.tile(np.arange(59, dtype=float), (11, 1))
        plan = SimpleNamespace(times=times, qpos=poses.copy(), source_times=times,
                               source_scene_qpos=source.copy(), metadata={},
                               events=[{"phase": "CARRY", "time": 3., "source_time": 3.},
                                       {"phase": "PLACE", "time": 8., "source_time": 8.}])
        episode = {"pickup_translation_m": [.24, -.01, 0.]}
        retarget_reference(plan, episode)
        np.testing.assert_array_equal(plan.qpos[:, 3:], poses[:, 3:])
        np.testing.assert_array_equal(plan.source_scene_qpos[:, 3:], source[:, 3:])
        np.testing.assert_allclose(plan.qpos[8:, :3], poses[8:, :3] - episode["pickup_translation_m"])
        np.testing.assert_array_equal(plan.source_scene_qpos[8:], source[8:])
        np.testing.assert_array_equal(plan.qpos[:4], poses[:4])
        self.assertTrue(np.all(np.diff(plan.qpos[:, 0]) <= 0))


if __name__ == "__main__":
    unittest.main()
