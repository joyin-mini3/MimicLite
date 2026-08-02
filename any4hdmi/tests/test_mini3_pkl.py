from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import joblib
import mujoco
import numpy as np

from any4hdmi.scripts.preprocess.mini3_pkl import (
    DEFAULT_MJCF,
    MINI3_JOINT_NAMES,
    convert_dataset,
    load_mini3_pkl,
    target_frame_count,
    validate_mini3_model,
)


class Mini3PklConversionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = mujoco.MjModel.from_xml_path(str(DEFAULT_MJCF))
        cls.layout = validate_mini3_model(cls.model)
        cls.default_dof = np.asarray(cls.model.qpos0[cls.layout.joint_qpos_adrs]).copy()

    def _write_motion(
        self,
        path: Path,
        *,
        frames: int = 5,
        fps: float = 100.0,
        joint_names: list[str] | None = None,
    ) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        root_pos = np.zeros((frames, 3), dtype=np.float32)
        root_pos[:, 0] = np.arange(frames, dtype=np.float32)
        root_rot = np.zeros((frames, 4), dtype=np.float32)
        root_rot[:, 3] = 1.0  # Source order is xyzw.
        payload: dict[str, object] = {
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": np.repeat(self.default_dof[None, :], frames, axis=0),
            "fps": fps,
        }
        if joint_names is not None:
            payload["joint_names"] = joint_names
        joblib.dump(payload, path)
        return path

    def test_real_asset_has_strict_mini3_layout(self) -> None:
        self.assertEqual(self.model.nq, 28)
        self.assertEqual(tuple(self.layout.qpos_names[7:]), MINI3_JOINT_NAMES)
        self.assertEqual(set(self.layout.actuator_target_joints), set(MINI3_JOINT_NAMES))

    def test_target_frame_count_preserves_last_sample_duration(self) -> None:
        self.assertEqual(target_frame_count(5, 100.0, 50.0), 3)
        self.assertEqual(target_frame_count(1, 120.0, 50.0), 1)

    def test_single_conversion_writes_qpos_only_and_preserves_relative_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_root = root / "pkl"
            source = self._write_motion(source_root / "230210" / "clip.pkl")
            output_root = root / "output" / "mini3" / "sonic"

            summary = convert_dataset(
                source,
                source_root=source_root,
                output_root=output_root,
                viewer=False,
            )

            output = output_root / "motions" / "230210" / "clip.npz"
            self.assertEqual(summary.converted, 1)
            self.assertEqual(summary.output_frames, 3)
            self.assertTrue(output.is_file())
            with np.load(output, allow_pickle=False) as payload:
                self.assertEqual(payload.files, ["qpos"])
                qpos = payload["qpos"]
            self.assertEqual(qpos.shape, (3, 28))
            np.testing.assert_allclose(qpos[:, self.layout.root_qpos_adr], [0.0, 2.0, 4.0])
            np.testing.assert_allclose(
                qpos[:, self.layout.root_qpos_adr + 3 : self.layout.root_qpos_adr + 7],
                np.asarray([[1.0, 0.0, 0.0, 0.0]] * 3),
            )

            manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["qpos_dim"], 28)
            self.assertEqual(manifest["num_motions"], 1)
            self.assertEqual(manifest["source"]["robot"], "mini3")
            self.assertFalse(manifest["source"]["joint_values_clipped"])

    def test_single_defaults_to_viewer_and_directory_defaults_to_headless(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            single_source = self._write_motion(root / "single" / "one.pkl")
            with mock.patch("any4hdmi.scripts.viewer.view_motion") as view_motion:
                convert_dataset(single_source, output_root=root / "single-output")
            view_motion.assert_called_once()

            batch_source = root / "batch"
            self._write_motion(batch_source / "one.pkl")
            self._write_motion(batch_source / "nested" / "two.pkl")
            with mock.patch("any4hdmi.scripts.viewer.view_motion") as view_motion:
                summary = convert_dataset(batch_source, output_root=root / "batch-output")
            view_motion.assert_not_called()
            self.assertEqual(summary.converted, 2)

    def test_joint_names_must_exactly_match_canonical_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = self._write_motion(
                Path(temporary_directory) / "bad.pkl",
                joint_names=[MINI3_JOINT_NAMES[0]] * len(MINI3_JOINT_NAMES),
            )
            with self.assertRaisesRegex(ValueError, "duplicates"):
                load_mini3_pkl(source)


if __name__ == "__main__":
    unittest.main()
