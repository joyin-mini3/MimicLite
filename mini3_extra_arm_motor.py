"""Run six added Mini3 arm joints through the calibrated elbow motor model."""

from __future__ import annotations

import math

import numpy as np


EXTRA_ARM_JOINT_NAMES = tuple(
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in ("elbow_yaw", "wrist_roll", "wrist_pitch")
)


class ExtraArmMotorModel:
    """Independent 4310P elbow dynamics for left-three, then right-three joints.

    Reuses the shared elbow current-loop, torque-speed limiter and KT classes
    directly. Six serial channels avoid the unrelated ankle Jacobian work
    required by the full-robot wrapper. This applies the same serial chain:
    PD/feedforward -> pre-TN -> current-loop -> post-TN -> KT -> effort clamp.
    """

    joint_names = EXTRA_ARM_JOINT_NAMES

    def __init__(
        self,
        *,
        dt: float = 0.002,
        kp: float | np.ndarray = 70.0,
        kd: float | np.ndarray = 3.0,
        effort_limit: float | np.ndarray = 12.5,
    ) -> None:
        from any4hdmi.utils.mini3_real_motor import (
            KT_OUTPUT_TABLES, MOTOR_SPECS, MotorKtOutputModel, MotorTnLimit,
            TorqueCurrentLoopResponse, mini3_motor_type,
        )
        from sim2real.config.robots.mini3 import MINI3_CFG

        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Motor timestep must be finite and positive")
        self.dt = float(dt)
        self.kp = self._parameter(kp, "kp")
        self.kd = self._parameter(kd, "kd")
        self.effort_limit = self._parameter(effort_limit, "effort_limit")
        if np.any(self.kp < 0) or np.any(self.kd < 0) or np.any(self.effort_limit <= 0):
            raise ValueError("Motor gains must be non-negative and effort limits positive")
        cfg = MINI3_CFG.real_motor
        self.response_enabled = cfg.torque_response_enabled
        self.tn_enabled = cfg.tn_torque_limit_enabled
        self.kt_enabled = cfg.kt_output_model_enabled
        self.tn_limit_after_response = cfg.tn_limit_after_response
        # Wrist names are intentionally absent from the original mapping:
        # assign these new motors the actual elbow family rather than infer it.
        self.motor_type = mini3_motor_type("left_elbow_pitch_joint")
        self.tn_model = MotorTnLimit(name=self.motor_type, **MOTOR_SPECS[self.motor_type])
        self.kt_model = MotorKtOutputModel(name=self.motor_type, **KT_OUTPUT_TABLES[self.motor_type])
        self.response = TorqueCurrentLoopResponse(
            6, self.dt, kp=cfg.torque_response_kp, ki=cfg.torque_response_ki,
            plant_tau_s=cfg.torque_response_plant_tau_s,
            delay_steps=cfg.torque_response_delay_steps,
        )
        self.raw_pd_torque = np.zeros(6)
        self.pre_response_torque = np.zeros(6)
        self.response_torque = np.zeros(6)
        self.applied_torque = np.zeros(6)

    @staticmethod
    def _parameter(value: float | np.ndarray, name: str) -> np.ndarray:
        result = np.asarray(value, dtype=np.float64)
        if result.ndim == 0:
            result = np.full(6, float(result))
        if result.shape != (6,) or not np.isfinite(result).all():
            raise ValueError(f"{name} must be finite with shape (6,), or a scalar parameter")
        return result.copy()

    @staticmethod
    def _command(value: np.ndarray | None, name: str, default: np.ndarray) -> np.ndarray:
        result = default if value is None else np.asarray(value, dtype=np.float64)
        if result.shape != (6,) or not np.isfinite(result).all():
            raise ValueError(f"{name} must be a finite array with shape (6,)")
        return result

    def reset(self) -> None:
        self.response.reset()
        for state in (self.raw_pd_torque, self.pre_response_torque,
                      self.response_torque, self.applied_torque):
            state.fill(0)

    def compute(
        self,
        target_pos: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        target_vel: np.ndarray | None = None,
        effort: np.ndarray | None = None,
        kp: np.ndarray | None = None,
        kd: np.ndarray | None = None,
    ) -> np.ndarray:
        zero = np.zeros(6)
        commands = tuple(self._command(value, name, default) for value, name, default in (
            (target_pos, "target_pos", zero), (joint_pos, "joint_pos", zero),
            (joint_vel, "joint_vel", zero), (target_vel, "target_vel", zero),
            (effort, "effort", zero), (kp, "kp", self.kp), (kd, "kd", self.kd),
        ))
        if np.any(commands[5] < 0) or np.any(commands[6] < 0):
            raise ValueError("Motor gains must be non-negative")
        q_target, q, dq, dq_target, feedforward, active_kp, active_kd = commands
        self.raw_pd_torque[:] = active_kp * (q_target - q) + active_kd * (dq_target - dq) + feedforward
        self.pre_response_torque[:] = (
            self.tn_model.clip(self.raw_pd_torque, dq) if self.tn_enabled else
            np.clip(self.raw_pd_torque, -self.effort_limit, self.effort_limit)
        )
        self.response_torque[:] = (
            self.response.compute(self.pre_response_torque) if self.response_enabled else
            self.pre_response_torque
        )
        output = self.response_torque
        if self.tn_enabled and self.tn_limit_after_response:
            output = self.tn_model.clip(output, dq)
        self.applied_torque[:] = self.kt_model.map(output) if self.kt_enabled else output
        np.clip(self.applied_torque, -self.effort_limit, self.effort_limit,
                out=self.applied_torque)
        return self.applied_torque.copy()
