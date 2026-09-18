"""Dataset admission, reproducibility, metadata, and safe export resumption."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import record_mini3_openhlm as recording
from mini3_vla_features import FEATURE_NAMES, FEATURE_UNITS


class RecordingFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.attempt = self.root / "attempts/attempt_000000"
        self.attempt.mkdir(parents=True)

    def write_attempt(self, *, frames=100, report_updates=None, sample_updates=None):
        report = {
            "success": True, "failure": None, "cube_settled_inside_basket": True,
            "policy_output_override_max": 0.,
            "arm_collision_monitor": {"unexpected_contacts": []},
        }
        report.update(report_updates or {})
        recording.write_json(self.attempt / "report.json", report)
        times = np.arange(frames, dtype=np.float64) / recording.FPS
        samples = {
            "time": times, "next_time": times + 1 / recording.FPS,
            "action_executed": np.ones(frames, dtype=bool),
            "executed_substeps": np.full(frames, 10, dtype=np.int64),
            "qpos": np.zeros((frames, 4)),
            "tool_target": np.full((frames, 3), np.nan),
            "tool_target_active": np.zeros(frames, dtype=bool),
            "phase": np.full(frames, "APPROACH"),
        }
        samples.update(sample_updates or {})
        np.savez_compressed(self.attempt / "control_samples.npz", **samples)
        return report, samples

    def quality(self):
        return recording.quality_check(self.attempt, .0005)


class EpisodeAdmissionTest(RecordingFixture):
    def test_accepts_complete_success_and_inactive_nan_tool_targets(self):
        self.write_attempt()
        accepted, details = self.quality()
        self.assertTrue(accepted)
        self.assertEqual(details["frames"], 100)
        self.assertEqual(details["reason"], "accepted")
        self.assertEqual(details["maximum_unexpected_penetration_m"], 0)

    def test_requires_success_settled_cube_and_unmodified_original_policy(self):
        for updates in ({"success": False}, {"cube_settled_inside_basket": False},
                        {"policy_output_override_max": 1e-8}):
            with self.subTest(updates=updates):
                self.write_attempt(report_updates=updates)
                self.assertFalse(self.quality()[0])

    def test_contact_limit_accepts_boundary_but_rejects_deeper_contact(self):
        for penetration, expected in ((.000499, True), (.0005, True), (.000501, False)):
            with self.subTest(penetration=penetration):
                contacts = [{"maximum_penetration_m": .000001},
                            {"maximum_penetration_m": penetration}]
                self.write_attempt(report_updates={
                    "arm_collision_monitor": {"unexpected_contacts": contacts}})
                accepted, details = self.quality()
                self.assertEqual(accepted, expected)
                self.assertEqual(details["unexpected_contact_pairs"], 2)
                self.assertEqual(details["maximum_unexpected_penetration_m"], penetration)

    def test_missing_collector_report_and_samples_are_rejected(self):
        self.assertEqual(self.quality(), (False, {"reason": "collector_error"}))
        self.write_attempt()
        (self.attempt / "control_samples.npz").unlink()
        accepted, details = self.quality()
        self.assertFalse(accepted)
        self.assertEqual(details["reason"], "missing_synchronized_samples")

    def test_active_tool_nan_and_nonfinite_measured_state_are_rejected(self):
        for updates, reason in (({"tool_target_active": np.ones(100, bool)}, "tool_target"),
                                ({"qpos": np.full((100, 4), np.inf)}, "qpos")):
            with self.subTest(reason=reason):
                self.write_attempt(sample_updates=updates)
                accepted, details = self.quality()
                self.assertFalse(accepted)
                self.assertEqual(details["reason"], "nonfinite_samples:" + reason)

    def test_partial_final_action_or_wrong_duration_is_rejected(self):
        for name, value in (("action_executed", False), ("executed_substeps", 9),
                            ("next_time", 1.999)):
            with self.subTest(field=name):
                _, samples = self.write_attempt()
                samples[name][-1] = value
                np.savez_compressed(self.attempt / "control_samples.npz", **samples)
                accepted, details = self.quality()
                self.assertFalse(accepted)
                self.assertEqual(details["reason"], "incomplete_control_action")

    def test_timestamps_must_begin_at_zero_and_use_native_fifty_hz(self):
        for times in (np.arange(100) / 50 + .02, np.arange(100) / 30,
                      np.r_[np.arange(99) / 50, 98 / 50]):
            with self.subTest(times=times[-2:].tolist()):
                self.write_attempt(sample_updates={"time": times, "next_time": times + .02})
                accepted, details = self.quality()
                self.assertFalse(accepted)
                self.assertEqual(details["reason"], "invalid_control_timestamps")
        self.write_attempt(frames=99)
        self.assertFalse(self.quality()[0])


class ReproducibilityTest(RecordingFixture):
    def test_exact_language_and_stable_color_indices(self):
        self.assertEqual(recording.COLORS, ("red", "green", "blue"))
        for color in recording.COLORS:
            self.assertEqual(recording.task_text(color),
                             f"Please put the {color} square from the tabletop into the basket.")
        with self.assertRaises(ValueError):
            recording.task_text("Blue")

    def test_seed_mapping_does_not_depend_on_global_rng_or_completion_order(self):
        expected = [recording.attempt_seed(20260918, index) for index in range(20)]
        np.random.seed(991)
        np.random.random(1000)
        shuffled = {index: recording.attempt_seed(20260918, index) for index in reversed(range(20))}
        self.assertEqual(expected, [shuffled[index] for index in range(20)])
        self.assertEqual(len(set(expected)), 20)
        self.assertNotEqual(expected, [recording.attempt_seed(20260919, index) for index in range(20)])
        self.assertTrue(all(isinstance(seed, int) and 0 <= seed < 2**64 for seed in expected))

    def test_interrupted_attempt_is_kept_without_relaunching_or_overwriting(self):
        scene = self.attempt / "episode_scene.xml"
        scene.write_text("partial attempt evidence")
        with patch.object(recording.subprocess, "run") as run:
            result = recording.collect_attempt(self.root, 0, 123)
        run.assert_not_called()
        self.assertEqual(result, self.attempt)
        self.assertEqual(scene.read_text(), "partial attempt evidence")
        self.assertFalse(self.quality()[0])

    def test_resume_refuses_changed_collection_configuration(self):
        args = argparse.Namespace(output=self.root, episodes=200, seed=5,
                                  max_penetration=.0005, resume=True)
        recording.write_json(self.root / "collection_config.json", {"base_seed": 4})
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            recording.run_collection(args)

    def test_resume_refuses_changed_source_hash_before_starting_workers(self):
        args = argparse.Namespace(output=self.root, episodes=200, seed=5,
                                  max_penetration=.0005, resume=True)
        recording.write_json(self.root / "collection_config.json", {
            "episodes": 200, "base_seed": 5, "maximum_unexpected_penetration_m": .0005,
            "dataset": str(self.root / "lerobot" / recording.REPO_ID), "fps": 50})
        recording.write_json(self.root / "source_hashes.json", {"record_mini3_openhlm.py": "changed"})
        with patch.object(recording, "ThreadPoolExecutor") as workers:
            with self.assertRaisesRegex(ValueError, "Collection code changed"):
                recording.run_collection(args)
        workers.assert_not_called()


class ExportReuseTest(RecordingFixture):
    def setUp(self):
        super().setUp()
        self.dataset = self.root / "dataset"
        self.parquet = self.dataset / "data/chunk-000/episode_000003.parquet"
        self.parquet.parent.mkdir(parents=True)
        self.parquet.write_bytes(b"existing episode payload; no rendering required for reuse")
        self.metadata_path = self.dataset / ".episode_meta/episode_000003.json"
        self.metadata = {
            "episode_index": 3, "start_index": 3600,
            "source_attempt": str(self.attempt.resolve()),
            "sha256": hashlib.sha256(self.parquet.read_bytes()).hexdigest(),
        }
        recording.write_json(self.metadata_path, self.metadata)

    def test_verified_existing_export_is_reused_without_source_arrays(self):
        self.assertEqual(recording.export_episode(self.attempt, self.dataset, 3, 3600), self.metadata)

    def test_reuse_requires_matching_source_episode_and_global_frame(self):
        for change in ({"episode_index": 4}, {"start_index": 3601},
                       {"source_attempt": str(self.root / "different_attempt")}):
            with self.subTest(change=change):
                recording.write_json(self.metadata_path, self.metadata | change)
                with self.assertRaisesRegex(ValueError, "different source or global index"):
                    recording.export_episode(self.attempt, self.dataset, 3, 3600)

    def test_corrupt_parquet_is_not_silently_reused(self):
        self.parquet.write_bytes(self.parquet.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "checksum does not match"):
            recording.export_episode(self.attempt, self.dataset, 3, 3600)


class DatasetMetadataTest(RecordingFixture):
    def test_finalize_records_all_channels_tasks_statistics_and_episode_split(self):
        dataset = self.root / "dataset"
        start = 0
        entries = []
        for index in range(12):
            length = 100 + index
            color = recording.COLORS[index % 3]
            stats = {"state": recording.vector_stats(np.zeros((length, 36)))}
            entry = {
                "episode_index": index, "start_index": start, "length": length,
                "source_attempt": str(self.attempt), "seed": index,
                "target_color": color, "tasks": [recording.task_text(color)],
                "stats": stats, "sha256": f"hash-{index}",
            }
            recording.write_json(dataset / ".episode_meta" / f"episode_{index:06d}.json", entry)
            entries.append(entry)
            start += length
        info = recording.finalize(dataset, len(entries))
        self.assertEqual(recording.read_json(dataset / "meta/info.json"), info)
        self.assertEqual(info["codebase_version"], "v2.1")
        self.assertEqual((info["total_episodes"], info["total_frames"], info["fps"]), (12, start, 50))
        self.assertEqual(info["total_videos"], 0)
        for key in ("state", "actions"):
            self.assertEqual(info["features"][key]["shape"], [36])
            self.assertEqual(info["features"][key]["names"], list(FEATURE_NAMES))
        for key in recording.CAMERAS:
            self.assertEqual(info["features"][key]["dtype"], "image")
            self.assertEqual(info["features"][key]["shape"], [224, 224, 3])
        schema = recording.read_json(dataset / "meta/mini3_schema.json")
        self.assertEqual(schema["units"], list(FEATURE_UNITS))
        self.assertIn("pre-action", schema["observation_action_alignment"])
        tasks = [json.loads(line) for line in (dataset / "meta/tasks.jsonl").read_text().splitlines()]
        self.assertEqual(tasks, [{"task_index": i, "task": recording.task_text(color)}
                                 for i, color in enumerate(recording.COLORS)])
        episodes = [json.loads(line) for line in (dataset / "meta/episodes.jsonl").read_text().splitlines()]
        self.assertEqual([item["length"] for item in episodes], [entry["length"] for entry in entries])
        stats = [json.loads(line) for line in (dataset / "meta/episodes_stats.jsonl").read_text().splitlines()]
        self.assertEqual(stats[-1], {"episode_index": 11, "stats": entries[-1]["stats"]})
        manifest = recording.read_json(dataset / "meta/collection_manifest.json")
        self.assertEqual(manifest[-1]["sha256"], "hash-11")
        self.assertNotIn("stats", manifest[-1])
        split = recording.read_json(dataset / "meta/recommended_split.json")
        self.assertEqual(split["unit"], "whole_episode")
        self.assertEqual(split["train_episodes"], list(range(11)))
        self.assertEqual(split["validation_episodes"], [11])

    def test_vector_statistics_keep_feature_dimensions_and_population_std(self):
        values = np.array([[1., 10.], [3., 20.], [5., 30.]])
        stats = recording.vector_stats(values)
        self.assertEqual(stats["count"], [3])
        self.assertEqual(stats["min"], [1., 10.])
        self.assertEqual(stats["max"], [5., 30.])
        self.assertEqual(stats["mean"], [3., 20.])
        np.testing.assert_allclose(stats["std"], values.std(axis=0))
        self.assertEqual(recording.vector_stats(np.arange(3))["mean"], [1.])


if __name__ == "__main__":
    unittest.main()
