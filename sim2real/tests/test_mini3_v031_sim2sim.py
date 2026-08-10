from __future__ import annotations

import unittest

import mujoco

from sim2real.config.robots import get_robot_cfg
from sim2real.sim_env.utils.mjcf import load_sim_model


class Mini3V031Sim2SimTest(unittest.TestCase):
    def test_versioned_model_uses_new_limits_and_neutral_height(self) -> None:
        cfg = get_robot_cfg("mini3-v031")
        self.assertEqual(cfg.name, "mini3-v031")
        self.assertEqual(cfg.default_qpos[:3], (0.0, 0.0, 0.43625))
        self.assertEqual(cfg.joint_pos_lower_limit["left_hip_pitch_joint"], -2.7925)
        self.assertEqual(cfg.joint_pos_upper_limit["left_knee_pitch_joint"], 2.3)
        self.assertEqual(cfg.joint_pos_lower_limit["left_shoulder_roll_joint"], 0.0)

        model = load_sim_model(cfg)
        self.assertEqual((model.nq, model.nv, model.nu), (28, 27, 21))
        actuator_targets = [
            model.joint(int(model.actuator_trnid[actuator_id, 0])).name
            for actuator_id in range(model.nu)
        ]
        self.assertEqual(actuator_targets, list(cfg.joint_names))

    def test_legacy_variant_remains_distinct(self) -> None:
        legacy = get_robot_cfg("mini3")
        versioned = get_robot_cfg("mini3-v031")
        self.assertNotEqual(legacy.resolve_mjcf_path(), versioned.resolve_mjcf_path())
        self.assertNotEqual(legacy.default_qpos, versioned.default_qpos)
        self.assertNotEqual(
            legacy.joint_pos_lower_limit["left_hip_pitch_joint"],
            versioned.joint_pos_lower_limit["left_hip_pitch_joint"],
        )
