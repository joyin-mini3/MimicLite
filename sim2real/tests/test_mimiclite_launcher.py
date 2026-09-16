from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "mimiclite_desktop_launcher", ROOT / "sim2sim_mini3_mimiclite.py"
)
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)

CPU_SPEC = importlib.util.spec_from_file_location(
    "mimiclite_cpu_export_tested", ROOT / "mimic_lite_cpu_export.py"
)
cpu_export = importlib.util.module_from_spec(CPU_SPEC)
sys.modules[CPU_SPEC.name] = cpu_export
CPU_SPEC.loader.exec_module(cpu_export)


class MimicLiteLauncherTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="mimiclite launcher ")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.dataset = self.root / "dataset with spaces"
        self.motion = self.dataset / "motion clips" / "walk and stop" / "clip one.npz"
        self.motion.parent.mkdir(parents=True)
        self.motion.touch()
        (self.motion.parent / "another clip.npz").touch()
        (self.dataset / "manifest.json").write_text(
            json.dumps({"motions_subdir": "motion clips"}), encoding="utf-8"
        )

    def test_motion_dataset_selects_only_named_clip_with_spaces(self):
        directory, filename = launcher.motion_dataset(self.motion)
        self.assertEqual(directory, self.dataset)
        self.assertEqual(filename, "walk and stop/clip one.npz")

    def test_motion_outside_manifest_motion_directory_is_rejected(self):
        outside = self.dataset / "outside.npz"
        outside.touch()
        with self.assertRaisesRegex(ValueError, "Motion must be inside"):
            launcher.motion_dataset(outside)

    def test_motion_without_manifest_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "No any4hdmi manifest"):
            launcher.motion_dataset(self.root / "orphan.npz")

    def test_training_config_replaces_dataset_with_one_selected_clip(self):
        from omegaconf import OmegaConf

        config = self.root / "training config.yaml"
        original = {
            "task": {
                "robot": {"name": "mini3-mesh"},
                "command": {
                    "motion_cfgs": {
                        "original": {
                            "path": "/old/server/dataset",
                            "filenames": ["unavailable.npz"],
                            "full_motion": False,
                        }
                    }
                },
                "num_envs": 4096,
            },
            "algo": {},
            "wandb": {"mode": "online"},
        }
        OmegaConf.save(OmegaConf.create(original), config)
        cfg = launcher.training_config(config, self.root / "model.pt", self.motion, 7)
        motions = OmegaConf.to_container(cfg.task.command.motion_cfgs, resolve=True)
        self.assertEqual(list(motions), ["local_test"])
        self.assertEqual(motions["local_test"]["filenames"], ["walk and stop/clip one.npz"])
        self.assertEqual(motions["local_test"]["path"], str(self.dataset))
        self.assertFalse(motions["local_test"]["full_motion"])
        self.assertEqual(cfg.task.num_envs, 1)
        self.assertEqual(cfg.seed, 7)
        self.assertEqual(OmegaConf.to_container(OmegaConf.load(config)), original)

    def test_headless_pause_and_unbounded_loop_are_rejected_before_setup(self):
        cases = (
            ["--headless", "--start-paused"],
            ["--headless", "--duration", "0"],
        )
        for argv in cases:
            with self.subTest(argv=argv), mock.patch.object(launcher, "setup_paths") as setup:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                    launcher.main(argv)
                self.assertEqual(caught.exception.code, 2)
                setup.assert_not_called()

    def test_nonfinite_and_negative_durations_are_rejected(self):
        for value in ("nan", "inf", "-1"):
            with self.subTest(value=value):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                    launcher.main(["--duration", value])
                self.assertEqual(caught.exception.code, 2)

    def test_checkpoint_without_training_config_has_actionable_error(self):
        checkpoint = self.root / "checkpoint.pt"
        checkpoint.touch()
        error = io.StringIO()
        with mock.patch.object(launcher, "setup_paths"), mock.patch.object(
            launcher, "prepare_export"
        ) as export:
            with contextlib.redirect_stderr(error), self.assertRaises(SystemExit) as caught:
                launcher.main([
                    "--load_model", str(checkpoint), "--motion", str(self.motion),
                    "--headless", "--duration", "1",
                ])
            self.assertEqual(caught.exception.code, 1)
            self.assertIn("supply --config", error.getvalue())
            export.assert_not_called()


class CpuExportValidationTest(unittest.TestCase):
    def setUp(self):
        from omegaconf import OmegaConf

        self.saved = {
            "seed": 0,
            "algo": {
                "_target_": "mimic_lite_learning.ppo.PPOPolicy",
                "in_keys": ["policy"],
                "vecnorm": True,
                "layer_norm": "before",
            },
            "task": {
                "robot": {"name": "mini3-mesh"},
                "command": {
                    "future_steps": [-1, 0, 1],
                    "motion_cfgs": {"train": {"path": "/server/motions"}},
                },
                "observation": {
                    "policy": {
                        "joint_pos": {"_target_": "mimic_lite.joint_pos_history"},
                        "joint_vel": {"_target_": "mimic_lite.joint_vel_history"},
                    }
                },
                "input": {"action": {"action_scaling": {"left": 0.25, "right": 0.25}}},
            },
        }
        self.cfg = OmegaConf.create(self.saved)

    def test_checkpoint_config_allows_motion_path_and_seed_changes(self):
        self.cfg.seed = 42
        self.cfg.task.command.motion_cfgs = {
            "local_test": {"path": "/local/motion with spaces", "filenames": ["clip.npz"]}
        }
        self.assertTrue(cpu_export._validate_checkpoint_config(self.cfg, self.saved))

    def test_checkpoint_config_rejects_same_shape_observation_reordering(self):
        terms = self.cfg.task.observation.policy
        self.cfg.task.observation.policy = {
            "joint_vel": terms.joint_vel,
            "joint_pos": terms.joint_pos,
        }
        with self.assertRaisesRegex(ValueError, "task.observation.policy"):
            cpu_export._validate_checkpoint_config(self.cfg, self.saved)

    def test_checkpoint_config_rejects_future_step_order_change(self):
        self.cfg.task.command.future_steps = [0, -1, 1]
        with self.assertRaisesRegex(ValueError, "task.command.future_steps"):
            cpu_export._validate_checkpoint_config(self.cfg, self.saved)

    def test_checkpoint_config_rejects_action_scaling_order_change(self):
        self.cfg.task.input.action.action_scaling = {"right": 0.25, "left": 0.25}
        with self.assertRaisesRegex(ValueError, "task.input.action.action_scaling"):
            cpu_export._validate_checkpoint_config(self.cfg, self.saved)

    def test_observation_spec_rejects_disabled_vecnorm(self):
        self.cfg.algo.vecnorm = False
        with self.assertRaisesRegex(ValueError, "requires saved VecNorm"):
            cpu_export._observation_spec(self.cfg, {})

    def test_observation_spec_rejects_inconsistent_moment_shapes(self):
        import torch

        state = {"vecnorms": {"policy.sum": torch.zeros(2), "policy.ssq": torch.zeros(3)}}
        with self.assertRaisesRegex(ValueError, "Inconsistent VecNorm shape"):
            cpu_export._observation_spec(self.cfg, state)

    def test_observation_spec_rejects_missing_moments(self):
        with self.assertRaisesRegex(ValueError, "Missing one-dimensional VecNorm"):
            cpu_export._observation_spec(self.cfg, {"vecnorms": {}})


if __name__ == "__main__":
    unittest.main()
