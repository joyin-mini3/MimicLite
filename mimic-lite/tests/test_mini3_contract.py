from __future__ import annotations

import json
import unittest
from pathlib import Path

import mujoco
import yaml

import active_adaptation as aa


try:
    _backend = aa.get_backend()
except RuntimeError:
    aa.set_backend("mjlab")
else:
    if _backend != "mjlab":
        raise RuntimeError(f"Mini3 contract tests require mjlab, got {_backend}")

from mimic_lite.assets.mini3 import (  # noqa: E402
    MINI3_CFG,
    MINI3_DAMPING,
    MINI3_JOINT_NAMES,
    MINI3_MJCF_PATH,
    MINI3_STIFFNESS,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class Mini3TrainingContractTest(unittest.TestCase):
    def test_asset_joint_and_actuator_order(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(MINI3_MJCF_PATH))
        hinge_joint_names = [
            model.joint(joint_id).name
            for joint_id in range(model.njnt)
            if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE
        ]
        actuator_targets = [
            model.joint(int(model.actuator_trnid[actuator_id, 0])).name
            for actuator_id in range(model.nu)
        ]
        self.assertEqual(hinge_joint_names, list(MINI3_JOINT_NAMES))
        self.assertEqual(actuator_targets, list(MINI3_JOINT_NAMES))
        self.assertEqual(list(MINI3_CFG.init_state.joint_pos), list(MINI3_JOINT_NAMES))
        self.assertEqual(list(MINI3_STIFFNESS), list(MINI3_JOINT_NAMES))
        self.assertEqual(list(MINI3_DAMPING), list(MINI3_JOINT_NAMES))

        entity_cfg = MINI3_CFG.mjlab()
        self.assertTrue(entity_cfg.strict_joint_contract)
        self.assertEqual(list(entity_cfg.init_state.joint_pos), list(MINI3_JOINT_NAMES))
        self.assertEqual(
            [
                name
                for actuator in entity_cfg.articulation.actuators
                for name in actuator.target_names_expr
            ],
            list(MINI3_JOINT_NAMES),
        )
        actuator_types = {
            type(actuator).__name__
            for actuator in entity_cfg.articulation.actuators
        }
        self.assertEqual(
            actuator_types,
            {
                "Mini3RealMotorActuatorCfg",
                "Mini3ParallelAnkleRealMotorActuatorCfg",
            },
        )
        training_model = entity_cfg.build().spec.compile()
        self.assertEqual(training_model.nu, 21)
        training_targets = [
            training_model.joint(
                int(training_model.actuator_trnid[actuator_id, 0])
            ).name
            for actuator_id in range(training_model.nu)
        ]
        self.assertEqual(training_targets, list(MINI3_JOINT_NAMES))

    def test_training_config_has_strict_action_and_timing(self) -> None:
        task_path = REPOSITORY_ROOT / "mimic-lite" / "cfg" / "task" / "tracking-base-mini3.yaml"
        task = yaml.safe_load(task_path.read_text(encoding="utf-8"))
        self.assertEqual(task["robot"]["name"], "mini3-mesh")
        self.assertEqual(task["sim"]["step_dt"], 0.02)
        self.assertEqual(task["sim"]["mujoco_physics_dt"], 0.002)
        self.assertEqual(
            list(task["input"]["action"]["action_scaling"]),
            list(MINI3_JOINT_NAMES),
        )
        self.assertEqual(
            task["reward"]["loco"]["feet_air_time"]["body_names"],
            ["left_ankle_roll_link", "right_ankle_roll_link"],
        )
        feet_air_time = task["reward"]["loco"]["feet_air_time"]
        self.assertNotIn("body2_names", feet_air_time)
        self.assertEqual(feet_air_time["height_range"], [0.03, 0.10])
        self.assertEqual(task["reward"]["tracking"]["root_pos"]["sigma"], 0.2)
        self.assertEqual(task["reward"]["tracking"]["body_pos"]["sigma"], 0.2)

    def test_motion_manifest_matches_asset_contract(self) -> None:
        manifest_path = REPOSITORY_ROOT / "any4hdmi" / "output" / "mini3" / "sonic" / "manifest.json"
        if not manifest_path.is_file():
            self.skipTest("Converted Mini3 dataset is uploaded separately")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(float(manifest["timestep"]), 0.02)
        self.assertEqual(int(manifest["qpos_dim"]), 28)
        self.assertEqual(manifest["qpos_names"][7:], list(MINI3_JOINT_NAMES))


if __name__ == "__main__":
    unittest.main()
