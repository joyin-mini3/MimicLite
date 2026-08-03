from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from any4hdmi.utils.mini3_real_motor import (
    ANKLE_PARAMS,
    KT_OUTPUT_TABLES,
    MOTOR_SPECS,
    MotorKtOutputModel,
    MotorTnLimit,
    Mini3RealMotorModel,
    ankle_ik,
)
from mjlab.sim import Simulation, SimulationCfg

import active_adaptation as aa


try:
    _backend = aa.get_backend()
except RuntimeError:
    aa.set_backend("mjlab")
else:
    if _backend != "mjlab":
        raise RuntimeError(f"Mini3 actuator tests require mjlab, got {_backend}")

from mimic_lite.assets.mini3_real_motor import (  # noqa: E402
    Mini3ParallelAnkleRealMotorActuator,
    Mini3RealMotorActuator,
    _PhysicsStepTorqueDelay,
)
from mimic_lite.assets.mini3 import (  # noqa: E402
    MINI3_CFG,
    MINI3_DAMPING,
    MINI3_EFFORT_LIMIT,
    MINI3_JOINT_NAMES,
    MINI3_STIFFNESS,
)
from active_adaptation.envs.mdp.randomizations.common import (  # noqa: E402
    motor_params_implicit,
)


class Mini3TrainingRealMotorTest(unittest.TestCase):
    def test_full_training_actuator_matches_shared_sim2sim_model(self) -> None:
        entity = MINI3_CFG.mjlab().build()
        model = entity.spec.compile()
        model.opt.timestep = 0.002
        simulation = Simulation(1, SimulationCfg(), model, "cpu")
        entity.initialize(
            simulation.mj_model,
            simulation.model,
            simulation.data,
            "cpu",
        )
        position = np.linspace(-0.03, 0.03, 21).astype(np.float32)
        velocity = np.linspace(0.1, -0.1, 21).astype(np.float32)
        target = position + np.linspace(-0.02, 0.02, 21).astype(np.float32)
        entity.data.write_joint_state(
            torch.from_numpy(position).unsqueeze(0),
            torch.from_numpy(velocity).unsqueeze(0),
        )
        entity.set_joint_position_target(torch.from_numpy(target).unsqueeze(0))
        entity._apply_actuator_controls()
        training_torque = simulation.data.ctrl[0].numpy().copy()

        shared_model = Mini3RealMotorModel(
            MINI3_JOINT_NAMES,
            np.asarray([MINI3_STIFFNESS[name] for name in MINI3_JOINT_NAMES]),
            np.asarray([MINI3_DAMPING[name] for name in MINI3_JOINT_NAMES]),
            np.asarray([MINI3_EFFORT_LIMIT[name] for name in MINI3_JOINT_NAMES]),
            dt=0.002,
        )
        sim2sim_torque = shared_model.compute(target, position, velocity)
        np.testing.assert_allclose(
            training_torque,
            sim2sim_torque,
            rtol=2.0e-4,
            atol=2.0e-5,
        )

    def test_explicit_pd_randomization_updates_actuator_tensors(self) -> None:
        class _Actuator:
            target_names = ["joint_a", "joint_b"]
            default_stiffness = torch.tensor([[10.0, 20.0], [10.0, 20.0]])
            default_damping = torch.tensor([[1.0, 2.0], [1.0, 2.0]])

            def set_gains(self, env_ids, kp=None, kd=None) -> None:
                self.kp = kp
                self.kd = kd

        actuator = _Actuator()
        randomization = object.__new__(motor_params_implicit)
        randomization.env = SimpleNamespace(device=torch.device("cpu"))
        randomization.explicit_pd_actuators = [actuator]
        randomization.kp_factor_range = {
            "joint_a": (0.5, 0.5),
            "joint_b": (0.5, 0.5),
        }
        randomization.kd_factor_range = {
            "joint_a": (2.0, 2.0),
            "joint_b": (2.0, 2.0),
        }
        randomization._reset_explicit_pd(torch.tensor([0, 1]))
        torch.testing.assert_close(
            actuator.kp, torch.tensor([[5.0, 10.0], [5.0, 10.0]])
        )
        torch.testing.assert_close(
            actuator.kd, torch.tensor([[2.0, 4.0], [2.0, 4.0]])
        )

    def test_response_delay_resets_environments_independently(self) -> None:
        delay = _PhysicsStepTorqueDelay(2, 1, 1.0, "cpu")
        torch.testing.assert_close(
            delay.append_and_read(torch.tensor([[1.0], [10.0]])),
            torch.tensor([[1.0], [10.0]]),
        )
        torch.testing.assert_close(
            delay.append_and_read(torch.tensor([[2.0], [20.0]])),
            torch.tensor([[1.0], [10.0]]),
        )
        delay.reset(torch.tensor([0]))
        torch.testing.assert_close(
            delay.append_and_read(torch.tensor([[3.0], [30.0]])),
            torch.tensor([[3.0], [20.0]]),
        )

    def test_training_tn_and_kt_match_shared_numpy_model(self) -> None:
        actuator = object.__new__(Mini3RealMotorActuator)
        actuator.motor_type = "4340p"
        actuator.motor_strength = torch.ones(1, 3)
        table = KT_OUTPUT_TABLES["4340p"]
        actuator.kt_feedback = torch.tensor((0.0, *table["feedback_tau_nm"]))
        actuator.kt_actual = torch.tensor((0.0, *table["actual_tau_nm"]))
        torque = torch.tensor([[30.0, 20.0, -20.0]])
        velocity = torch.tensor([[0.0, math.pi, -math.pi]])
        training_tn = actuator._tn_clip(torque, velocity).numpy()
        shared_tn = MotorTnLimit(
            name="4340p", **MOTOR_SPECS["4340p"]
        ).clip(torque.numpy(), velocity.numpy())
        np.testing.assert_allclose(training_tn, shared_tn, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            actuator._kt_map(torch.from_numpy(shared_tn)).numpy(),
            MotorKtOutputModel(name="4340p", **table).map(shared_tn),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_parallel_ankle_ik_matches_shared_numpy_model(self) -> None:
        params = ANKLE_PARAMS
        args = (
            params["d"],
            params["df"],
            params["zl"] + params["z0"],
            params["zr"] + params["z0"],
            params["l"],
            params["lm"],
            params["hl"],
            params["hr"],
            params["z0"],
        )
        roll = torch.tensor([0.0, 0.1], dtype=torch.float64)
        pitch = torch.tensor([0.0, -0.08], dtype=torch.float64)
        training_tmr, training_tml, valid = (
            Mini3ParallelAnkleRealMotorActuator._ankle_ik(args, roll, pitch)
        )
        self.assertTrue(torch.all(valid))
        expected = np.asarray(
            [ankle_ik(*args, float(r), float(q)) for r, q in zip(roll, pitch)]
        )
        np.testing.assert_allclose(training_tmr.numpy(), expected[:, 0], atol=1e-10)
        np.testing.assert_allclose(training_tml.numpy(), expected[:, 1], atol=1e-10)


if __name__ == "__main__":
    unittest.main()
