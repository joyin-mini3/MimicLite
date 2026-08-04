from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from any4hdmi.scripts.preprocess.mini3_pico import (
    PICO_BODY_NAMES,
    convert_dataset,
    discover_pico_clips,
    load_pico_clip,
)
from any4hdmi.scripts.preprocess.mini3_pkl import (
    DEFAULT_MJCF,
    MINI3_JOINT_NAMES,
    validate_mini3_model,
)


class Mini3PicoConversionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = mujoco.MjModel.from_xml_path(str(DEFAULT_MJCF))
        cls.layout = validate_mini3_model(cls.model)

    def _write_unpacked_clip(self, path: Path, *, axes_version: int = 3) -> Path:
        path.mkdir(parents=True)
        model = self.model
        data = mujoco.MjData(model)
        source_targets = {
            "pelvis": ("base_link", np.zeros(3)),
            "left_ankle_roll_link": ("left_ankle_roll_link", np.zeros(3)),
            "right_ankle_roll_link": ("right_ankle_roll_link", np.zeros(3)),
            "left_wrist_yaw_link": (
                "left_elbow_pitch_link",
                np.asarray(model.geom_pos[model.geom("left_elbow_pitch_link_collision").id]),
            ),
            "right_wrist_yaw_link": (
                "right_elbow_pitch_link",
                np.asarray(model.geom_pos[model.geom("right_elbow_pitch_link_collision").id]),
            ),
        }
        joint_frames = [
            {
                "left_hip_pitch_joint": -0.075,
                "left_knee_pitch_joint": 0.15,
                "left_ankle_pitch_joint": -0.075,
                "right_hip_pitch_joint": -0.075,
                "right_knee_pitch_joint": 0.15,
                "right_ankle_pitch_joint": -0.075,
                "left_shoulder_pitch_joint": -0.3,
                "left_elbow_pitch_joint": 0.8,
                "right_shoulder_pitch_joint": -0.25,
                "right_elbow_pitch_joint": 0.7,
            },
            {
                "left_hip_pitch_joint": -0.25,
                "left_knee_pitch_joint": 0.5,
                "left_ankle_pitch_joint": -0.25,
                "right_hip_pitch_joint": -0.2,
                "right_knee_pitch_joint": 0.4,
                "right_ankle_pitch_joint": -0.2,
                "left_shoulder_pitch_joint": -0.45,
                "left_elbow_pitch_joint": 1.0,
                "right_shoulder_pitch_joint": -0.35,
                "right_elbow_pitch_joint": 0.9,
            },
            {
                "left_hip_pitch_joint": -0.15,
                "left_knee_pitch_joint": 0.3,
                "left_ankle_pitch_joint": -0.15,
                "right_hip_pitch_joint": -0.18,
                "right_knee_pitch_joint": 0.36,
                "right_ankle_pitch_joint": -0.18,
                "left_shoulder_pitch_joint": -0.2,
                "left_elbow_pitch_joint": 0.85,
                "right_shoulder_pitch_joint": -0.15,
                "right_elbow_pitch_joint": 0.8,
            },
        ]
        positions = np.empty((len(joint_frames), len(PICO_BODY_NAMES), 3), dtype=np.float32)
        quaternions = np.empty((len(joint_frames), len(PICO_BODY_NAMES), 4), dtype=np.float32)
        anchors = np.empty((len(joint_frames), 6), dtype=np.float32)
        root_yaw_quat = np.empty(4, dtype=np.float64)
        mujoco.mju_axisAngle2Quat(root_yaw_quat, np.asarray([0.0, 0.0, 1.0]), 1.1)
        link_frame_offset = np.empty(4, dtype=np.float64)
        link_frame_offset_inverse = np.empty(4, dtype=np.float64)
        mujoco.mju_axisAngle2Quat(
            link_frame_offset,
            np.asarray([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8]),
            0.9,
        )
        mujoco.mju_negQuat(link_frame_offset_inverse, link_frame_offset)
        for frame_idx, joint_values in enumerate(joint_frames):
            data.qpos[:] = model.qpos0
            data.qpos[0] += 0.02 * frame_idx
            data.qpos[3:7] = root_yaw_quat
            for joint_name, value in joint_values.items():
                joint_id = model.joint(joint_name).id
                data.qpos[model.jnt_qposadr[joint_id]] = value
            mujoco.mj_forward(model, data)
            base_id = model.body("base_link").id
            anchors[frame_idx] = data.xmat[base_id].reshape(3, 3)[:, :2].reshape(6)
            for body_idx, source_name in enumerate(PICO_BODY_NAMES):
                body_name, local_point = source_targets[source_name]
                body_id = model.body(body_name).id
                rotation = data.xmat[body_id].reshape(3, 3)
                positions[frame_idx, body_idx] = data.xpos[body_id] + rotation @ local_point
                if source_name == "pelvis":
                    quaternions[frame_idx, body_idx] = data.xquat[body_id]
                else:
                    source_quat = np.empty(4, dtype=np.float64)
                    mujoco.mju_mulQuat(
                        source_quat,
                        data.xquat[body_id],
                        link_frame_offset_inverse,
                    )
                    quaternions[frame_idx, body_idx] = source_quat

        arrays = {
            "body_names": np.asarray(PICO_BODY_NAMES),
            "body_pos_w": positions,
            "body_quat_w": quaternions,
            "sonic_smpl_anchor_orientation": anchors,
            "fps": np.asarray([50.0], dtype=np.float64),
            "dt": np.asarray([0.02], dtype=np.float64),
            "source": np.asarray("pico_motion_clip"),
            "pico_position_axes_version": np.asarray(axes_version, dtype=np.int32),
            "body_state_frame": np.asarray("g1_robotics_zup_v1"),
        }
        for name, value in arrays.items():
            np.save(path / f"{name}.npy", value)
        return path

    def test_loads_unpacked_clip_and_prefers_sonic_root_orientation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = self._write_unpacked_clip(Path(temporary_directory) / "clip")
            paths, source_root = discover_pico_clips(source)
            self.assertEqual(paths, [source])
            self.assertEqual(source_root, source.parent)
            clip = load_pico_clip(source)
            self.assertEqual(clip.body_pos_w.shape, (3, 5, 3))
            self.assertEqual(clip.root_orientation_source, "sonic_smpl_anchor_orientation")
            np.testing.assert_allclose(
                clip.root_quat_wxyz[0],
                [np.cos(0.55), 0.0, 0.0, np.sin(0.55)],
                atol=1.0e-7,
            )

    def test_conversion_writes_qpos_only_dataset_and_ik_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = self._write_unpacked_clip(root / "clip")
            output_root = root / "output"
            summary = convert_dataset(
                source,
                output_root=output_root,
                scale=1.0,
                max_iterations=60,
                viewer=False,
                show_progress=False,
            )

            output = output_root / "motions" / "clip.npz"
            self.assertEqual(summary.converted, 1)
            self.assertEqual(summary.output_frames, 3)
            with np.load(output, allow_pickle=False) as payload:
                self.assertEqual(payload.files, ["qpos"])
                qpos = payload["qpos"]
            self.assertEqual(qpos.shape, (3, 28))
            np.testing.assert_allclose(qpos[:, 0], [0.0, 0.02, 0.04], atol=1.0e-6)
            self.assertLessEqual(
                float(np.abs(np.diff(qpos[:, self.layout.joint_qpos_adrs], axis=0)).max()),
                0.120001,
            )
            joint_index = {
                name: idx for idx, name in enumerate(MINI3_JOINT_NAMES)
            }
            hip_yaw = qpos[
                0,
                [
                    self.layout.joint_qpos_adrs[joint_index["left_hip_yaw_joint"]],
                    self.layout.joint_qpos_adrs[joint_index["right_hip_yaw_joint"]],
                ],
            ]
            self.assertLess(float(np.abs(hip_yaw).max()), 0.2)
            ankle_roll_indices = [
                self.layout.joint_qpos_adrs[
                    joint_index["left_ankle_roll_joint"]
                ],
                self.layout.joint_qpos_adrs[
                    joint_index["right_ankle_roll_joint"]
                ],
            ]
            self.assertLess(
                float(np.abs(qpos[:, ankle_roll_indices]).max()),
                0.2,
            )

            report = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(report["root_orientation_source"], "sonic_smpl_anchor_orientation")
            self.assertLess(max(report["ik_mean_position_error_m"].values()), 0.04)
            manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["qpos_dim"], 28)
            self.assertEqual(manifest["num_motions"], 1)
            self.assertEqual(manifest["source"]["robot"], "mini3")

    def test_rejects_old_pico_axes_version(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = self._write_unpacked_clip(
                Path(temporary_directory) / "clip", axes_version=2
            )
            with self.assertRaisesRegex(ValueError, "pico_position_axes_version=2"):
                load_pico_clip(source)


if __name__ == "__main__":
    unittest.main()
