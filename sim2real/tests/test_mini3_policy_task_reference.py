from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mini3_policy_task_reference import (  # noqa: E402
    DEFAULT_SCENE, EXTRA_NAMES, MINI3_XML, PHASES, RIGHT_ARM_NAMES,
    build_task_reference,
)
from mini3_motion_reference import place_qpos, shorten_walk  # noqa: E402


class PolicyTaskReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.TemporaryDirectory(prefix="mini3_reference_test_")
        cls.folder = Path(cls.directory.name)
        cls.scene = mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE))
        cls.canonical = mujoco.MjModel.from_xml_path(str(MINI3_XML))
        cls.arm_ids = np.array([cls.canonical.joint(name).qposadr[0] for name in RIGHT_ARM_NAMES])
        cls.actual_arm_ids = np.array([cls.scene.joint(name).qposadr[0] for name in RIGHT_ARM_NAMES])
        cls.extra_ids = np.array([cls.scene.joint(name).qposadr[0] for name in EXTRA_NAMES])
        cls.source_times = np.arange(1, 1644) * .02
        cls.source_qpos = np.tile(cls.scene.qpos0, (1643, 1))
        cls.source_qpos[:, 0] = 4 + cls.source_times * .1  # Deliberate measured tracking offset.
        cls.source_qpos[:, 3:7] = [1, 0, 0, 0]
        cls.arm_actual = (np.array([.1, -.3, -.2, .4])
                          + cls.source_times[:, None] * [.002, .001, .002, .001])
        cls.extra_actual = (np.array([0, .6, .4, 0, -.6, .4])
                            + cls.source_times[:, None] * [0, 0, 0, .001, .001, .001])
        cls.source_qpos[:, cls.actual_arm_ids] = cls.arm_actual
        cls.source_qpos[:, cls.extra_ids] = cls.extra_actual
        cls.trajectory = cls.folder / "trajectory.npz"
        np.savez(cls.trajectory, time=cls.source_times, qpos=cls.source_qpos,
                 arm_command=np.column_stack((cls.arm_actual + .05, cls.extra_actual[:, 3:])),
                 extra_command=cls.extra_actual + .002)
        event_times = [0, 8.48, 13.48, 15.98, 18.5, 19.72, 23.24, 28.82, 31.84, 32.86]
        cls.report = {"success": True, "initial_root_xyz": [-2.72, 0, .46305],
                      "wait_times_s": {"approach": 2, "lift": .5, "basket": 1},
                      "events": [{"phase": name, "time": time,
                                  "base_xyz": [1.05 if time >= 28 else .077, .004, .46]}
                                 for name, time in zip(PHASES, event_times)]}
        (cls.folder / "report.json").write_text(json.dumps(cls.report))
        cls.walk = np.tile(cls.canonical.qpos0, (500, 1))
        cls.walk[:, 0] = np.arange(500) * .01
        cls.walk[:, 3:7] = [1, 0, 0, 0]
        cls.walk[:, 7:19] += np.sin(np.arange(500)[:, None] * 2 * np.pi / 50) * .05
        cls.motion = cls.folder / "walk.npz"
        np.savez(cls.motion, qpos=cls.walk)
        cls.plan = cls.build()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    @classmethod
    def build(cls, **kwargs):
        return build_task_reference(cls.trajectory, original_motion=cls.motion, **kwargs)

    def test_name_mapping_keeps_added_axes_out_of_original_policy_schema(self) -> None:
        self.assertEqual(self.plan.qpos.shape, (1314, 28))
        self.assertEqual(self.plan.extra_command.shape, (1314, 6))
        self.assertEqual(self.plan.metadata["original_policy_joint_count"], 21)
        sample = self.plan.sample(10)
        source = sample["arm_source_time"]
        expected = [.1 + source * .002, -.3 + source * .001,
                    -.2 + source * .002, .4 + source * .001]
        np.testing.assert_allclose(sample["qpos"][self.arm_ids], expected)
        self.assertNotIn("right_elbow_yaw_joint", RIGHT_ARM_NAMES)

    def test_locomotion_uses_original_reference_instead_of_recorded_root_error(self) -> None:
        shortened, _ = shorten_walk(self.walk, 2.65)
        placed = place_qpos(shortened, origin_xy=(-2.72, 0), align_travel=True)
        np.testing.assert_allclose(self.plan.sample(1)["qpos"], placed[50])
        self.assertGreater(abs(self.plan.sample(1)["qpos"][0] - self.source_qpos[49, 0]), 4)
        hold = self.plan.sample(10)["qpos"]
        np.testing.assert_allclose(hold[:2], [.077, .004])
        np.testing.assert_allclose(hold[7:19], placed[-1, 7:19])

    def test_arm_raises_during_approach_and_keeps_recorded_movement_speed(self) -> None:
        self.assertEqual(self.plan.metadata["arm_raise_start"], 3.48)
        self.assertEqual(self.plan.metadata["approach_end"], 6.48)
        self.assertLess(self.plan.metadata["place_start"], self.plan.metadata["carry_walk_end_s"])
        for start, end in [(4.1, 8.8), (9, 11.4), (11.6, 13.8),
                           (14.1, 15.1), (15.4, 18), (18.4, 22), (22.4, 25), (25.4, 26)]:
            with self.subTest(interval=(start, end)):
                delta = self.plan.sample(end)["arm_source_time"] - self.plan.sample(start)["arm_source_time"]
                self.assertAlmostEqual(delta, end - start, places=8)
        self.assertAlmostEqual(self.plan.duration, 26.26)

    def test_zero_overlap_with_source_waits_preserves_every_phase_time(self) -> None:
        baseline = self.build(approach_overlap=0, basket_overlap=0,
                              approach_wait=2, lift_wait=.5, basket_wait=1)
        for event, old in zip(baseline.events, self.report["events"]):
            self.assertEqual(event["phase"], old["phase"])
            self.assertAlmostEqual(event["time"], old["time"])
        self.assertEqual(baseline.metadata["arm_blend_s"], 0)
        shortened, _ = shorten_walk(self.walk, 2.65)
        placed = place_qpos(shortened, origin_xy=(-2.72, 0), align_travel=True)
        np.testing.assert_allclose(baseline.sample(7.0)["qpos"][:19], placed[-1, :19])

    def test_reference_bias_only_affects_right_original_four_after_smooth_blend(self) -> None:
        bias = np.array([.01, -.02, .03, .04])
        changed = self.build(arm_reference_bias=tuple(bias))
        keep = np.setdiff1d(np.arange(28), self.arm_ids)
        np.testing.assert_array_equal(changed.qpos[:, keep], self.plan.qpos[:, keep])
        np.testing.assert_array_equal(changed.extra_command, self.plan.extra_command)
        np.testing.assert_array_equal(changed.sample(3.48)["qpos"], self.plan.sample(3.48)["qpos"])
        for time, weight in [(3.48, 0), (3.74, .529984), (4, 1), (20, 1)]:
            delta = changed.sample(time)["qpos"] - self.plan.sample(time)["qpos"]
            np.testing.assert_allclose(delta[self.arm_ids], weight * bias, atol=1e-10)

    def test_command_and_actual_channels_are_independently_selectable(self) -> None:
        changed = self.build(arm_source="command", extra_source="actual")
        actual = self.plan.sample(10)
        command = changed.sample(10)
        np.testing.assert_allclose(command["qpos"][self.arm_ids] - actual["qpos"][self.arm_ids], .05)
        np.testing.assert_allclose(command["extra_command"] - actual["extra_command"], -.002)

    def test_sampling_clamps_normalizes_and_returns_owned_arrays(self) -> None:
        sample = self.plan.sample(-1)
        np.testing.assert_allclose(sample["qpos"], self.plan.qpos[0])
        sample["qpos"][:] = 0
        self.assertAlmostEqual(np.linalg.norm(self.plan.sample(.137)["qpos"][3:7]), 1)
        np.testing.assert_array_equal(self.plan.sample(100)["qpos"], self.plan.qpos[-1])
        source = self.plan.sample_source_qpos(1.0)
        source[:] = 0
        self.assertGreater(np.linalg.norm(self.plan.source_scene_qpos), 0)
        self.assertEqual(self.plan.sample(14)["phase"], "CLOSE")

    def test_bad_configuration_is_rejected_before_reference_creation(self) -> None:
        for kwargs in ({"dt": 0}, {"dt": .01}, {"lift_wait": -1},
                       {"basket_overlap": float("nan")}, {"arm_source": "torque"},
                       {"arm_reference_bias": (0, 0)},
                       {"arm_reference_bias": (0, 0, 0, float("inf"))}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.build(**kwargs)

    def test_saved_plan_contains_timeline_and_reference_provenance(self) -> None:
        output = self.folder / "saved"
        self.plan.save(output)
        info = json.loads((output / "task_reference.json").read_text())
        self.assertEqual(info["close_start"], 14)
        self.assertTrue(info["gates_are_advisory"])
        self.assertEqual(info["events"][1]["phase"], "CLEAR_ARM")
        with np.load(output / "task_reference.npz") as stored:
            np.testing.assert_array_equal(stored["qpos"], self.plan.qpos)
            np.testing.assert_array_equal(stored["arm_source_time"], self.plan.arm_source_time)


if __name__ == "__main__":
    unittest.main()
