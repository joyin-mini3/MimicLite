from __future__ import annotations

import os
import unittest
from pathlib import Path

import mujoco

if "ACTIVE_ADAPTATION_BACKEND" not in os.environ:
    import active_adaptation as aa

    aa.set_backend("mjlab")

from mimic_lite.assets.mini3 import MINI3_JOINT_NAMES
from mimic_lite.assets.mini3_v031 import MINI3_V031_CFG, MINI3_V031_MJCF_PATH


class Mini3V031TrainingContractTest(unittest.TestCase):
    def test_asset_contract_and_neutral_height(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(MINI3_V031_MJCF_PATH))
        hinge_joint_names = [
            model.joint(joint_id).name
            for joint_id in range(model.njnt)
            if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE
        ]
        self.assertEqual(hinge_joint_names, list(MINI3_JOINT_NAMES))
        self.assertEqual((model.nq, model.nv, model.nu), (28, 27, 21))
        self.assertEqual(MINI3_V031_CFG.init_state.pos, (0.0, 0.0, 0.43625))

    def test_task_uses_the_versioned_asset_and_motion_root(self) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        task = (repository_root / "cfg/task/tracking-base-mini3-v031.yaml").read_text(
            encoding="utf-8"
        )
        motion = (repository_root / "cfg/task/motion/mini3_v031/sonic.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn("name: mini3-v031-mesh", task)
        self.assertIn("motion: mini3_v031/sonic", task)
        self.assertIn("output/mini3_v031/sonic", motion)
