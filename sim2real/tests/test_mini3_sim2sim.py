from __future__ import annotations

import unittest

import mujoco
import numpy as np

from sim2real.config.robots import get_robot_cfg
from sim2real.rl_policy.observations.common import (
    joint_pos_history,
    joint_vel_history,
    prev_actions,
    projected_gravity_history,
    root_ang_vel_history,
)
from sim2real.sim_env.utils.mjcf import load_sim_model
from sim2real.sim_env.integrated_sim2sim import IntegratedSimRuntime


class _State:
    def __init__(self) -> None:
        self.joint_names = ["joint_a", "joint_b"]
        self.root_ang_vel_b = np.asarray([1.0, 2.0, 3.0])
        self.root_quat_w = np.asarray([1.0, 0.0, 0.0, 0.0])
        self.joint_pos = np.asarray([0.25, -0.5])
        self.joint_vel = np.asarray([0.75, -1.0])


class _Env:
    def __init__(self) -> None:
        self.state_processor = _State()
        self.joint_names_simulation = ["joint_a", "joint_b"]
        self.num_actions = 2


class Mini3Sim2SimTest(unittest.TestCase):
    def test_model_and_actuator_transmission_contract(self) -> None:
        cfg = get_robot_cfg("mini3")
        model = load_sim_model(cfg)

        self.assertEqual((model.nq, model.nv, model.nu), (28, 27, 21))
        scalar_joint_names = [
            model.joint(joint_id).name
            for joint_id in range(model.njnt)
            if model.jnt_type[joint_id]
            in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        ]
        actuator_targets = [
            model.joint(int(model.actuator_trnid[actuator_id, 0])).name
            for actuator_id in range(model.nu)
        ]
        body_names = [model.body(body_id).name for body_id in range(1, model.nbody)]
        self.assertEqual(scalar_joint_names, list(cfg.joint_names))
        self.assertEqual(actuator_targets, list(cfg.joint_names))
        self.assertEqual(body_names, list(cfg.body_names))
        self.assertTrue(all(model.actuator(i).name.endswith("_ctrl") for i in range(model.nu)))
        self.assertIsNotNone(cfg.real_motor)
        self.assertTrue(cfg.real_motor.enabled)

    def test_integrated_runtime_enables_stateful_real_motor(self) -> None:
        cfg = get_robot_cfg("mini3")
        runtime = IntegratedSimRuntime(
            cfg,
            sim_dt=0.002,
            headless=True,
            key_callback=None,
        )
        self.assertIsNotNone(runtime.real_motor)
        target = np.zeros(len(cfg.joint_names), dtype=np.float32)
        target[0] = 1.0
        kp = np.full(len(cfg.joint_names), 20.0, dtype=np.float32)
        kd = np.ones(len(cfg.joint_names), dtype=np.float32)
        runtime.apply_command(
            target,
            np.zeros_like(target),
            np.zeros_like(target),
            kp,
            kd,
        )
        runtime.compute_torques()
        applied = runtime.torques[runtime.joint_idx_in_ctrl]
        self.assertGreater(float(applied[0]), 0.0)
        self.assertLess(float(applied[0]), 20.0)
        np.testing.assert_allclose(applied[4:6], 0.0, atol=1e-7)
        runtime.reset_motor_state()
        np.testing.assert_allclose(runtime.real_motor.applied_torque, 0.0)

    def test_self_collision_keeps_adjacent_excludes(self) -> None:
        cfg = get_robot_cfg("mini3")
        spec = mujoco.MjSpec.from_file(str(cfg.resolve_mjcf_path()))
        excluded_pairs = {
            frozenset((exclude.bodyname1, exclude.bodyname2))
            for exclude in spec.excludes
        }
        self.assertIn(
            frozenset(("left_knee_pitch_link", "left_ankle_pitch_link")),
            excluded_pairs,
        )
        self.assertNotIn(
            frozenset(("left_knee_pitch_link", "right_knee_pitch_link")),
            excluded_pairs,
        )

        model = mujoco.MjModel.from_xml_path(str(cfg.resolve_mjcf_path()))
        for geom_name in (
            "left_knee_pitch_link_collision",
            "right_knee_pitch_link_collision",
        ):
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            self.assertGreaterEqual(geom_id, 0)
            self.assertEqual(int(model.geom_contype[geom_id]), 1)
            self.assertEqual(int(model.geom_conaffinity[geom_id]), 1)

    def test_common_history_reset_fills_current_state(self) -> None:
        env = _Env()
        observations = (
            root_ang_vel_history(env=env, history_steps=[0, 1, 4]),
            projected_gravity_history(env=env, history_steps=[0, 1, 4]),
            joint_pos_history(env=env, history_steps=[0, 1, 4]),
            joint_vel_history(env=env, history_steps=[0, 1, 4]),
        )
        for observation in observations:
            observation.reset()

        np.testing.assert_allclose(
            observations[0].root_ang_vel_history,
            np.broadcast_to(env.state_processor.root_ang_vel_b, (5, 3)),
        )
        np.testing.assert_allclose(
            observations[1].projected_gravity_history,
            np.broadcast_to(np.asarray([0.0, 0.0, -1.0]), (5, 3)),
        )
        np.testing.assert_allclose(
            observations[2].joint_pos_multistep,
            np.broadcast_to(env.state_processor.joint_pos, (5, 2)),
        )
        np.testing.assert_allclose(
            observations[3].joint_vel_multistep,
            np.broadcast_to(env.state_processor.joint_vel, (5, 2)),
        )

        action_history = prev_actions(env=env, steps=3)
        action_history.update({"action": np.asarray([1.0, -1.0])})
        action_history.reset()
        np.testing.assert_array_equal(action_history.compute(), np.zeros(6))


if __name__ == "__main__":
    unittest.main()
