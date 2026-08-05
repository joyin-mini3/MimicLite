from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

import active_adaptation as aa


try:
    _backend = aa.get_backend()
except RuntimeError:
    aa.set_backend("mjlab")
else:
    if _backend != "mjlab":
        raise RuntimeError(f"Reference filter tests require mjlab, got {_backend}")

from mimic_lite.reference_motion_filter import (  # noqa: E402
    DIAGNOSTIC_COLUMNS,
    MotionCatalog,
    apply_nominal_profile,
    classify_termination_flags,
    finalize_results,
    load_result_rows,
    write_result_shard,
)
from mimic_lite.tasks.observations.track import (  # noqa: E402
    tracking_filter_diagnostics,
)


class ReferenceMotionFilterTest(unittest.TestCase):
    def test_diagnostic_column_contract_matches_observation(self) -> None:
        self.assertEqual(
            DIAGNOSTIC_COLUMNS,
            tracking_filter_diagnostics.columns,
        )

    def test_finished_row_reads_flat_observation_groups(self) -> None:
        script_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "filter_reference_motions.py"
        )
        spec = importlib.util.spec_from_file_location(
            "_filter_reference_motions_test", script_path
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        diagnostics = torch.zeros(1, len(DIAGNOSTIC_COLUMNS))
        diagnostics[0, DIAGNOSTIC_COLUMNS.index("motion_t")] = 7
        diagnostics[0, DIAGNOSTIC_COLUMNS.index("applied_action_abs_max")] = 0.4
        diagnostics[0, DIAGNOSTIC_COLUMNS.index("applied_torque_abs_max")] = 3.2
        termination = TensorDict(
            {
                "motion_timeout": torch.ones(1, 1),
                "root_pos_error": torch.zeros(1, 1),
                "root_ori_error": torch.zeros(1, 1),
                "body_pos_error": torch.zeros(1, 1),
                "body_ori_error": torch.zeros(1, 1),
            },
            batch_size=[1],
        )
        next_td = TensorDict(
            {
                "_filter_diagnostics": diagnostics,
                "stats": TensorDict(
                    {"termination": termination}, batch_size=[1]
                ),
            },
            batch_size=[1],
        )
        td = TensorDict(
            {
                "action": torch.tensor([[0.1, -0.2]]),
                "next": next_td,
            },
            batch_size=[1],
        )
        catalog = MotionCatalog(
            dataset_root=Path("/dataset"),
            motions_root=Path("/dataset/motions"),
            relative_paths=("sample.npz",),
            lengths=(10,),
        )

        rows = module._record_finished_rows(
            td=td,
            finished_env_ids=torch.tensor([0]),
            source_ids_before=torch.tensor([0]),
            catalog=catalog,
            tracking_body_names=["base_link"],
            tracking_joint_names=["joint"],
        )

        self.assertEqual(rows[0]["status"], "passed")
        self.assertEqual(rows[0]["termination_t"], 7)
        self.assertAlmostEqual(rows[0]["max_abs_policy_action"], 0.2)
        self.assertAlmostEqual(rows[0]["max_abs_applied_action"], 0.4)
        self.assertAlmostEqual(rows[0]["max_abs_applied_torque"], 3.2)

    def test_tracking_failure_wins_over_motion_timeout(self) -> None:
        status, reason = classify_termination_flags(
            {
                "motion_timeout": True,
                "root_pos_error": True,
                "root_ori_error": False,
                "body_pos_error": False,
                "body_ori_error": False,
            }
        )
        self.assertEqual(status, "failed")
        self.assertEqual(reason, "root_pos_error")

    def test_timeout_only_is_passed_and_unknown_done_is_runtime_error(self) -> None:
        self.assertEqual(
            classify_termination_flags({"motion_timeout": True}),
            ("passed", "motion_timeout"),
        )
        self.assertEqual(
            classify_termination_flags({}),
            ("runtime_error", "unknown_done"),
        )

    def test_nominal_profile_uses_saved_shape_but_removes_stochasticity(self) -> None:
        cfg = OmegaConf.create(
            {
                "backend": "mjlab",
                "device": "cuda:0",
                "headless": True,
                "app": {"headless": True, "enable_cameras": False},
                "seed": 5,
                "checkpoint_path": None,
                "task": {
                    "num_envs": 8,
                    "max_episode_length": 1000,
                    "shared": {"termination_root_body_name": "base_link"},
                    "randomization": {"push": {"enabled": True}},
                    "input": {
                        "action": {
                            "min_delay": 1,
                            "max_delay": 3,
                            "alpha_range": [0.8, 1.0],
                        }
                    },
                    "observation": {
                        "command": {
                            "ref": {"_target_": "unused", "noise_std": 0.1}
                        }
                    },
                    "command": {
                        "motion_cfgs": {
                            "sonic": {
                                "path": "old",
                                "weight": 1.0,
                                "full_motion": False,
                            }
                        },
                        "rewind_prob": 0.8,
                        "pose_range": {"x": [-0.1, 0.1]},
                        "velocity_range": {"x": [-0.2, 0.2]},
                        "init_joint_pos_noise": 0.1,
                        "init_joint_vel_noise": 0.2,
                    },
                },
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            checkpoint = temp / "checkpoint.pt"
            checkpoint.touch()
            dataset = temp / "dataset"
            dataset.mkdir()
            result = apply_nominal_profile(
                cfg,
                checkpoint_path=checkpoint,
                dataset_root=dataset,
                num_envs=4,
                window_frames=128,
                max_motion_length=2000,
                headless=True,
            )

        self.assertEqual(result.task.num_envs, 4)
        self.assertEqual(result.task.max_episode_length, 2002)
        self.assertEqual(result.task.randomization, {})
        self.assertEqual(result.task.terrain, "plane")
        self.assertEqual(result.task.command.rewind_prob, 0.0)
        self.assertEqual(result.task.command.init_joint_pos_noise, 0.0)
        self.assertEqual(result.task.command.init_joint_vel_noise, 0.0)
        self.assertTrue(
            all(
                bounds == [0.0, 0.0]
                for bounds in result.task.command.pose_range.values()
            )
        )
        self.assertTrue(
            all(
                bounds == [0.0, 0.0]
                for bounds in result.task.command.velocity_range.values()
            )
        )
        self.assertTrue(result.task.command.sequential_eval)
        self.assertEqual(result.task.command.sequential_window_frames, 128)
        self.assertEqual(result.task.input.action.min_delay, 2)
        self.assertEqual(result.task.input.action.max_delay, 2)
        self.assertEqual(result.task.input.action.alpha, 0.9)
        self.assertEqual(
            result.task.observation.command.ref.noise_std,
            0.0,
        )
        self.assertIn("_filter_diagnostics", result.task.observation)

    def test_result_shards_resume_and_finalize_lists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            catalog = MotionCatalog(
                dataset_root=output,
                motions_root=output / "motions",
                relative_paths=("a.npz", "nested/b.npz", "c.npz"),
                lengths=(10, 20, 30),
            )
            rows = [
                {
                    "dataset_motion_id": 1,
                    "relative_path": "nested/b.npz",
                    "status": "failed",
                    "termination_reason": "body_pos_error",
                    "termination_flags": {"body_pos_error": True},
                },
                {
                    "dataset_motion_id": 0,
                    "relative_path": "a.npz",
                    "status": "passed",
                    "termination_reason": "motion_timeout",
                    "termination_flags": {"motion_timeout": True},
                },
                {
                    "dataset_motion_id": 2,
                    "relative_path": "c.npz",
                    "status": "runtime_error",
                    "termination_reason": "unknown_done",
                    "termination_flags": {},
                },
            ]
            write_result_shard(output, rows[:2], worker_index=0)
            write_result_shard(output, rows[2:], worker_index=0)

            loaded = load_result_rows(output)
            self.assertEqual(set(loaded), {0, 1, 2})
            summary = finalize_results(
                output,
                catalog=catalog,
                expected_motion_ids=[0, 1, 2],
            )

            self.assertEqual(summary["passed"], 1)
            self.assertEqual(summary["failed"], 1)
            self.assertEqual(summary["runtime_error"], 1)
            self.assertEqual(
                (output / "failed_motions.txt").read_text(),
                "nested/b.npz\n",
            )
            result_rows = [
                json.loads(line)
                for line in (output / "results.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [row["dataset_motion_id"] for row in result_rows], [0, 1, 2]
            )


if __name__ == "__main__":
    unittest.main()
