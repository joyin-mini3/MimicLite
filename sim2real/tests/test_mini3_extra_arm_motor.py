from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from sim2sim_mini3_mimiclite import setup_paths

setup_paths()
from any4hdmi.utils.mini3_real_motor import Mini3RealMotorModel
from mini3_extra_arm_motor import EXTRA_ARM_JOINT_NAMES, ExtraArmMotorModel
from sim2real.config.robots.mini3 import MINI3_CFG


def original_motor() -> Mini3RealMotorModel:
    cfg = MINI3_CFG.real_motor
    count = len(MINI3_CFG.joint_names)
    return Mini3RealMotorModel(
        tuple(MINI3_CFG.joint_names), np.full(count, 70.0), np.full(count, 3.0),
        np.array([MINI3_CFG.joint_effort_limit[name] for name in MINI3_CFG.joint_names]),
        dt=0.002, response_enabled=cfg.torque_response_enabled,
        tn_enabled=cfg.tn_torque_limit_enabled,
        tn_limit_after_response=cfg.tn_limit_after_response,
        kt_enabled=cfg.kt_output_model_enabled, response_kp=cfg.torque_response_kp,
        response_ki=cfg.torque_response_ki,
        response_plant_tau_s=cfg.torque_response_plant_tau_s,
        response_delay_steps=cfg.torque_response_delay_steps,
        ankle_motor_torque_limit=cfg.ankle_motor_torque_limit,
    )


class ExtraArmMotorTest(unittest.TestCase):
    def test_six_independent_channels_match_original_elbows_at_every_tick(self) -> None:
        extra = ExtraArmMotorModel()
        originals = [original_motor() for _ in range(3)]
        elbow_indices = [MINI3_CFG.joint_names.index(f"{side}_elbow_pitch_joint")
                         for side in ("left", "right")]
        self.assertEqual(EXTRA_ARM_JOINT_NAMES, (
            "left_elbow_yaw_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
            "right_elbow_yaw_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
        ))
        rng = np.random.default_rng(10)
        for tick in range(80):
            # Start from zero, apply distinct steps to all six motors, then
            # exercise both torque signs and the velocity-dependent envelope.
            target = np.zeros(6) if tick < 3 else np.linspace(-0.2, 0.3, 6)
            position = rng.uniform(-0.05, 0.05, 6) if tick >= 30 else np.zeros(6)
            velocity = rng.uniform(-50, 50, 6) if tick >= 30 else np.zeros(6)
            target_velocity = rng.uniform(-1, 1, 6) if tick >= 30 else np.zeros(6)
            effort = rng.uniform(-12, 12, 6) if tick >= 30 else np.zeros(6)
            kp, kd = np.linspace(60, 75, 6), np.linspace(2, 4, 6)
            actual = extra.compute(target, position, velocity, target_velocity, effort, kp, kd)
            expected = np.zeros(6)
            for family, motor in enumerate(originals):
                pair = [family, family + 3]
                vectors = []
                for source in (target, position, velocity, target_velocity, effort, kp, kd):
                    values = np.zeros(len(MINI3_CFG.joint_names))
                    values[elbow_indices] = source[pair]
                    vectors.append(values)
                q_target, q, dq, dq_target, feedforward, active_kp, active_kd = vectors
                original = motor.compute(q_target, q, dq, target_vel=dq_target,
                                         effort=feedforward, kp=active_kp, kd=active_kd)
                expected[pair] = original[elbow_indices]
            np.testing.assert_array_equal(actual, expected)

    def test_current_response_tn_kt_and_effort_limit_remain_active(self) -> None:
        motor = ExtraArmMotorModel()
        zero = np.zeros(6)
        motor.compute(zero, zero, zero)
        initial = motor.compute(zero, zero, zero, effort=np.full(6, 100.0))
        for _ in range(250):
            steady = motor.compute(zero, zero, zero, effort=np.full(6, 100.0))
        self.assertTrue(np.all(initial < steady))
        self.assertTrue(np.all(steady > 8))
        self.assertTrue(np.all(steady < 12.5))  # KT calibration follows TN limiting.
        np.testing.assert_array_equal(motor.pre_response_torque, np.full(6, 12.5))
        no_load = motor.tn_model.no_load_speed_rpm * 2 * np.pi / 60
        at_no_load = motor.compute(zero, zero, np.full(6, no_load),
                                   effort=np.full(6, 100.0), kd=zero)
        np.testing.assert_allclose(at_no_load, zero, atol=1e-12)
        limited = ExtraArmMotorModel(effort_limit=2.0)
        for _ in range(50):
            output = limited.compute(np.ones(6), zero, zero)
        self.assertTrue(np.all(np.abs(output) <= 2.0))

    def test_reset_clears_current_loop_history_without_coupling_channels(self) -> None:
        motor = ExtraArmMotorModel()
        zero = np.zeros(6)
        target = np.array([0.2, 0, 0, 0, 0, 0])
        for _ in range(20):
            output = motor.compute(target, zero, zero)
        self.assertGreater(output[0], 0)
        np.testing.assert_array_equal(output[1:], zero[1:])
        motor.reset()
        for state in (motor.raw_pd_torque, motor.pre_response_torque,
                      motor.response_torque, motor.applied_torque):
            np.testing.assert_array_equal(state, zero)
        fresh = ExtraArmMotorModel()
        for _ in range(5):
            np.testing.assert_array_equal(motor.compute(target, zero, zero),
                                          fresh.compute(target, zero, zero))

    def test_rejects_invalid_command_before_advancing_motor_history(self) -> None:
        motor = ExtraArmMotorModel()
        zero = np.zeros(6)
        for malformed in (np.zeros(5), np.full(6, np.nan)):
            with self.subTest(command=malformed):
                with self.assertRaises(ValueError):
                    motor.compute(malformed, zero, zero)
        np.testing.assert_array_equal(motor.compute(np.ones(6), zero, zero),
                                      ExtraArmMotorModel().compute(np.ones(6), zero, zero))


if __name__ == "__main__":
    unittest.main()
