#!/usr/bin/env python3
"""Run a trained Mini3 tracking policy with an independent contact gripper controller."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any, TYPE_CHECKING

from sim2sim_mini3_mimiclite import ROOT, setup_paths

if TYPE_CHECKING:
    import mujoco
    import numpy as np


def validate_policy_schema(config: dict[str, Any]) -> None:
    """Reject references that would consume stale arm-body FK after streaming IK."""
    expected = (
        ("ref_root_pos_future_local", "ref_body_pos_future_local"),
        ("ref_root_ori_future_b", "ref_root_ori_future_b"),
        ("ref_joint_pos_future", "ref_joint_pos_future"),
        ("ref_root_lin_vel_future_local", "ref_root_lin_vel_future_local"),
    )
    observations = config.get("observation", {})
    commands = observations.get("command", {})
    if list(commands) != [name for name, _ in expected]:
        raise ValueError("This controller requires the Mini3 root/joint tracking command schema")
    for name, target in expected:
        if commands[name].get("_target_") != f"mimic_lite.{target}":
            raise ValueError(f"Unsupported tracking command target: {name}")
    if commands["ref_root_pos_future_local"].get("body_names") not in ("base_link", ["base_link"]):
        raise ValueError("Reference body positions must select base_link only")
    motion = config.get("motion", {})
    if motion.get("root_body_name") != "base_link" or motion.get("anchor_body_name") != "base_link":
        raise ValueError("Reference root and anchor must both be base_link")
    allowed = {f"mimic_lite.{name}" for name in (
        "root_ang_vel_history", "projected_gravity_history", "joint_pos_history",
        "joint_vel_history", "prev_actions")}
    for group, items in observations.items():
        if group != "command" and any(item.get("_target_") not in allowed for item in items.values()):
            raise ValueError("Unsupported policy observations for streamed arm references")


class ArmIK:
    """Four-joint bounded IK; computes targets in a separate kinematic state."""

    def __init__(self, model: mujoco.MjModel, side: str = "right", tool_offset: float = 0.0) -> None:
        import mujoco
        import numpy as np
        self.model = model
        self.scratch = mujoco.MjData(model)
        self.names = [f"{side}_{part}_joint" for part in
                      ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch")]
        self.jids = np.array([model.joint(name).id for name in self.names])
        self.qids = model.jnt_qposadr[self.jids]
        self.vids = model.jnt_dofadr[self.jids]
        self.site = model.site(f"{side}_gripper_tcp").id
        self.tool_offset = float(tool_offset)
        self.last_error = math.inf

    def position(self, data: mujoco.MjData) -> np.ndarray:
        return data.site_xpos[self.site] + self.tool_offset * data.site_xmat[self.site].reshape(3, 3)[:, 0]

    def solve(self, data: mujoco.MjData, target: np.ndarray) -> np.ndarray:
        import mujoco
        import numpy as np
        from scipy.optimize import least_squares
        self.scratch.qpos[:] = data.qpos
        seed = data.qpos[self.qids].copy()
        low, high = self.model.jnt_range[self.jids].T
        seed = np.clip(seed, low + 1e-5, high - 1e-5)

        def residual(q):
            self.scratch.qpos[self.qids] = q
            mujoco.mj_kinematics(self.model, self.scratch)
            rotation = self.scratch.site_xmat[self.site].reshape(3, 3)
            return np.r_[10 * (self.position(self.scratch) - target),
                         0.4 * rotation[2, 1], 0.003 * (q - seed)]

        solution = least_squares(residual, seed, bounds=(low + 1e-5, high - 1e-5),
                                 max_nfev=25, ftol=1e-5, xtol=1e-5, gtol=1e-5)
        residual(solution.x)
        self.last_error = float(np.linalg.norm(self.position(self.scratch) - target))
        return solution.x


class PolicyScene:
    """Keep the trained 21-joint policy contract separate from all scene DOFs."""

    def __init__(self, scene: Path, policy_config: Path, motion: Path) -> None:
        import mujoco
        import numpy as np
        from any4hdmi.utils.mini3_real_motor import Mini3RealMotorModel
        from sim2real.config.robots.mini3 import MINI3_CFG
        from sim2real.sim_env.integrated_sim2sim import IntegratedPolicyRuntime, IntegratedSim2SimArgs
        import yaml
        with open(policy_config) as stream:
            validate_policy_schema(yaml.safe_load(stream))
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.scene_path = str(Path(scene).resolve())
        # Camera clipping is relative to scene extent; preserve a 2 mm near
        # plane for wrist cameras even in the longer three-metre scene.
        self.model.vis.map.znear = min(self.model.vis.map.znear, 0.002 / self.model.stat.extent)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 0.002
        args = IntegratedSim2SimArgs(policy_config=str(policy_config), motion_path=str(motion),
                                    robot="mini3", headless=True, initial_pause_s=0,
                                    inference_backend="onnx-cpu", sim_dt=0.002, env_dt=0.02)
        self.policy = IntegratedPolicyRuntime(args=args, robot_cfg=MINI3_CFG)
        self.state = self.policy.state_processor
        self.names = list(MINI3_CFG.joint_names)
        joint_ids = np.array([self.model.joint(name).id for name in self.names])
        self.qids, self.vids = self.model.jnt_qposadr[joint_ids], self.model.jnt_dofadr[joint_ids]
        self.aids = np.array([self.model.actuator(f"{name}_ctrl").id for name in self.names])
        if not np.array_equal(self.model.actuator_trnid[self.aids, 0], joint_ids):
            raise ValueError("A named motor targets a different joint than the trained policy expects")
        self.rootq = int(self.model.joint("floating_base").qposadr[0])
        self.rootv = int(self.model.joint("floating_base").dofadr[0])
        self.root = self.model.body("base_link").id
        self.grippers = np.array([self.model.actuator(f"{s}_gripper_ctrl").id for s in ("left", "right")])
        self.limits = np.array([MINI3_CFG.joint_effort_limit[n] for n in self.names])
        cfg = MINI3_CFG.real_motor
        self.motor = Mini3RealMotorModel(
            tuple(self.names), self.policy.joint_kp_unitree, self.policy.joint_kd_unitree,
            self.limits, dt=0.002, response_enabled=cfg.torque_response_enabled,
            tn_enabled=cfg.tn_torque_limit_enabled, tn_limit_after_response=cfg.tn_limit_after_response,
            kt_enabled=cfg.kt_output_model_enabled, response_kp=cfg.torque_response_kp,
            response_ki=cfg.torque_response_ki, response_plant_tau_s=cfg.torque_response_plant_tau_s,
            response_delay_steps=cfg.torque_response_delay_steps,
            ankle_motor_torque_limit=cfg.ankle_motor_torque_limit)
        # The generated TCP is 15 mm ahead of the finger pad midpoint. Grasp
        # around the pad midpoint so the cube cannot escape over the fingertips.
        self.ik = ArmIK(self.model, tool_offset=-0.015)
        self.arm_indices = np.array([self.names.index(name) for name in self.ik.names])
        self.arm_integral = np.zeros(4)
        self.arm_target = None
        self.arm_command = None
        self.openings = np.array([0.035, 0.035])
        self.reset()

    def reset(self) -> None:
        import mujoco
        import numpy as np
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        self.data.eq_active[self.model.equality("base_support_weld").id] = False
        self.motor.reset()
        self.arm_integral[:] = 0
        self.openings[:] = 0.035
        self.arm_target = None
        self.arm_command = None
        self.state.restart_motion()
        self.policy.state_dict = {"action": np.zeros(21, dtype=np.float32),
                                  "paused": True, "control_mode": "policy"}
        mujoco.mj_forward(self.model, self.data)
        self.sync()
        self.policy.reset()

    def sync(self) -> None:
        self.state.qpos[:7] = self.data.qpos[self.rootq:self.rootq + 7]
        self.state.qvel[:6] = self.data.qvel[self.rootv:self.rootv + 6]
        self.state.joint_pos[:] = self.data.qpos[self.qids]
        self.state.joint_vel[:] = self.data.qvel[self.vids]
        self.state.joint_torque[:] = self.data.actuator_force[self.aids]
        self.state.low_state_tick = int(self.data.time * 1000)

    def initialize_robot(self, qpos: np.ndarray) -> None:
        """Only used at episode initialization, never during phase transitions."""
        import mujoco
        if self.data.time != 0:
            raise RuntimeError("Robot initialization is only allowed before the first physics step")
        self.data.qpos[self.rootq:self.rootq + 7] = qpos[:7]
        self.data.qpos[self.qids] = qpos[7:]
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self.sync()
        self.policy.reset()

    def step(self) -> None:
        import mujoco
        import numpy as np
        self.sync()
        target = None
        if self.arm_target is not None:
            goal = self.ik.solve(self.data, self.arm_target)
            previous = self.data.qpos[self.ik.qids] if self.arm_command is None else self.arm_command
            target = previous + np.clip(goal - previous, -0.03, 0.03)
            self.arm_command = target
            reference = getattr(self.state, "motion_dataset", None)
            if reference is not None:
                first = min(max(0, int(self.state.motion_t[0]) + 1), reference.num_steps - 1)
                reference._storage["joint_pos"][first:, self.arm_indices] = target
                reference._storage["joint_vel"][first:, self.arm_indices] = 0
        result = self.policy.step()
        if result is None:
            raise RuntimeError("Policy inference failed")
        q, dq, effort, kp, kd = [np.asarray(value).copy() for value in result]
        if self.arm_target is not None:
            idx = self.arm_indices
            self.arm_integral = np.clip(self.arm_integral +
                                       0.02 * (target - self.data.qpos[self.ik.qids]), -0.12, 0.12)
            q[idx] = np.clip(target + 2.0 * self.arm_integral,
                            self.model.jnt_range[self.ik.jids, 0],
                            self.model.jnt_range[self.ik.jids, 1])
            kp[idx] = 70
            kd[idx] = 3
            effort[idx] = self.data.qfrc_bias[self.ik.vids]
        else:
            self.arm_integral[:] = 0
            self.arm_command = None
        for _ in range(10):
            torque = self.motor.compute(q, self.data.qpos[self.qids], self.data.qvel[self.vids],
                                        target_vel=dq, effort=effort, kp=kp, kd=kd)
            if not np.isfinite(torque).all():
                raise FloatingPointError("Non-finite motor torque")
            self.data.ctrl[self.aids] = np.clip(torque, -self.limits, self.limits)
            self.data.ctrl[self.grippers] = self.openings
            if self.data.eq_active[self.model.equality("base_support_weld").id]:
                raise RuntimeError("Pelvis support must stay disabled during the policy task")
            mujoco.mj_step(self.model, self.data)
        # Keep the project's normal mj_step sequence. Calling mj_forward here
        # would perform an extra constraint solve between motor-control ticks.
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError("Non-finite simulation state")


def resolve_policy(model: str, motion: Path, supplied: Path | None = None) -> Path:
    from types import SimpleNamespace
    from sim2sim_mini3_mimiclite import BUNDLE, PRESETS, prepare_export
    if supplied:
        path = supplied.resolve()
        path = path.with_suffix(".yaml")
        if not path.is_file() or not path.with_suffix(".onnx").is_file():
            raise FileNotFoundError("Supply an ONNX policy with its companion YAML")
        return path
    checkpoint = BUNDLE / "models" / model / PRESETS[model][0]
    config = checkpoint.parent / "play_local.yaml"
    if not config.is_file():
        config = checkpoint.parent / "config.yaml"
    policy, _, _ = prepare_export(checkpoint, config, motion, SimpleNamespace(seed=0, reexport=False))
    return policy.with_suffix(".yaml")


def smooth_fraction(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3 - 2 * value)


class PickCarryTask:
    """Measured phase transitions around policy walking and contact manipulation."""

    def __init__(self, runtime: PolicyScene, motion: Path, *,
                 approach_distance: float = 2.65, carry_distance: float = 1.0,
                 approach_wait: float = 2.0, lift_wait: float = 0.5,
                 basket_wait: float = 1.0) -> None:
        import numpy as np
        from mini3_motion_reference import MotionReference, install_reference, place_qpos, shorten_walk
        from sim2sim_mini3_mimiclite import motion_dataset
        self.wait_times = {"approach": approach_wait, "lift": lift_wait, "basket": basket_wait}
        for name, seconds in self.wait_times.items():
            if not math.isfinite(seconds) or seconds < 0:
                raise ValueError(f"{name} wait must be finite and non-negative")
        self.r = runtime
        self.motion_path = str(Path(motion).resolve())
        dataset_root, _ = motion_dataset(Path(motion).resolve())
        manifest = json.loads((dataset_root / "manifest.json").read_text())
        if not math.isclose(float(manifest["timestep"]), 0.02, abs_tol=1e-8):
            raise ValueError("This motion editor requires a 50 Hz source clip; resample the dataset first")
        with np.load(motion) as archive:
            self.walk = archive["qpos"].astype(np.float64)
        self.carry_distance = carry_distance
        self.cube = self.r.model.body("pick_cube_3").id
        self.cube_geom = self.r.model.geom("pick_cube_3_geom").id
        self.basket = self.r.model.body("pick_basket").id
        self.finger_geoms = [self.r.model.geom(f"right_gripper_finger_{part}_geom").id
                            for part in ("positive", "negative")]
        self.phase, self.phase_started = "APPROACH", 0.0
        self.events, self.records = [], []
        self.done, self.success, self.failure = False, False, None
        self.contact_lost_s = 0.0
        self.settled_s = 0.0
        self.last_print = -1
        self.initial_cube = self.r.data.xpos[self.cube].copy()
        self.initial_root = self.r.data.qpos[:3].copy()
        walk, self.approach_metadata = shorten_walk(self.walk, approach_distance)
        walk = place_qpos(walk, origin_xy=self.r.data.qpos[:2], align_travel=True)
        hold_frames = math.ceil(approach_wait / 0.02)
        self.reference = MotionReference(np.concatenate((walk, np.tile(walk[-1], (hold_frames, 1)))))
        install_reference(self.r.policy, self.reference, paused=True)
        self.r.initialize_robot(walk[0])
        self.r.policy.state_dict["paused"] = False
        self.r.state.motion_t[:] = -1
        self.events.append({"time": 0.0, "phase": self.phase})

    def enter(self, phase: str) -> None:
        self.phase, self.phase_started = phase, self.r.data.time
        self.events.append({"time": float(self.phase_started), "phase": phase,
                            "base_xyz": self.r.data.qpos[:3].tolist(),
                            "cube_xyz": self.r.data.xpos[self.cube].tolist()})
        print(f"PHASE t={self.phase_started:.2f}s {phase}", flush=True)

    def fail(self, reason: str) -> None:
        self.failure, self.done = reason, True
        self.enter("FAILED")
        print(f"FAILED: {reason}", flush=True)

    def contacts(self) -> np.ndarray:
        import mujoco
        import numpy as np
        forces = np.zeros(2)
        force = np.empty(6)
        for index, contact in enumerate(self.r.data.contact):
            if self.cube_geom not in (contact.geom1, contact.geom2):
                continue
            other = int(contact.geom2 if contact.geom1 == self.cube_geom else contact.geom1)
            if other in self.finger_geoms:
                mujoco.mj_contactForce(self.r.model, self.r.data, index, force)
                forces[self.finger_geoms.index(other)] += max(0, force[0])
        return forces

    def closing_command(self, elapsed: float) -> float:
        return 0.0

    def release_target_position(self) -> np.ndarray:
        """Default release waypoint above the basket's centre."""
        return self.r.data.xpos[self.basket] + [0, 0, 0.18]

    def update_clear_arm(self, elapsed: float, tcp: np.ndarray) -> None:
        self.r.arm_target = self.start_tcp + smooth_fraction(elapsed / 3) * (self.clear_target - self.start_tcp)
        if elapsed >= 3.5:
            self.start_tcp = tcp
            self.enter("REACH")

    def hold_reference(self) -> None:
        import numpy as np
        from mini3_motion_reference import MotionReference, install_reference
        pose = self.reference.qpos[-1].copy()
        pose[:2] = self.r.data.qpos[:2]
        self.reference = MotionReference(np.tile(pose, (2000, 1)))
        install_reference(self.r.policy, self.reference)

    def start_carry(self) -> None:
        import numpy as np
        from mini3_motion_reference import MotionReference, install_reference, place_qpos, shorten_walk, blend_poses
        walk, self.carry_metadata = shorten_walk(self.walk, self.carry_distance)
        walk = place_qpos(walk, origin_xy=self.r.data.qpos[:2], align_travel=True)
        pose = self.reference.qpos[-1].copy()
        pose[:2] = self.r.data.qpos[:2]
        hold_frames = math.ceil(self.wait_times["basket"] / 0.02)
        frames = np.concatenate((blend_poses(pose, walk[0], 1.0)[:-1], walk,
                                 np.tile(walk[-1], (hold_frames, 1))))
        self.reference = MotionReference(frames)
        install_reference(self.r.policy, self.reference)
        self.carry_offset = self.r.ik.position(self.r.data) - self.r.data.qpos[:3]
        self.carry_height = float(self.r.ik.position(self.r.data)[2])
        self.enter("CARRY")

    def cube_in_basket(self) -> bool:
        import numpy as np
        data, model = self.r.data, self.r.model
        rotation = data.xmat[self.basket].reshape(3, 3)
        relative = rotation.T @ (data.xpos[self.cube] - data.xpos[self.basket])
        cube_rotation = rotation.T @ data.xmat[self.cube].reshape(3, 3)
        half_extents = np.abs(cube_rotation) @ np.full(3, 0.02)
        contained = (np.all(np.abs(relative[:2]) + half_extents[:2] < [0.12, 0.11])
                     and 0.008 < relative[2] - half_extents[2] < 0.018
                     and relative[2] + half_extents[2] < 0.12)
        cube_v = int(model.jnt_dofadr[model.body_jntadr[self.cube]])
        bottom = model.geom("pick_basket_bottom").id
        cube_geom = self.cube_geom
        resting = any({int(contact.geom1), int(contact.geom2)} == {bottom, cube_geom}
                      and contact.dist < 0.0005 for contact in data.contact)
        return bool(contained and resting and np.linalg.norm(data.qvel[cube_v:cube_v + 6]) < 0.03)

    def step(self) -> None:
        import numpy as np
        if self.done:
            return
        r, data = self.r, self.r.data
        elapsed = data.time - self.phase_started
        tcp = r.ik.position(data).copy()
        cube = data.xpos[self.cube].copy()
        speed = float(np.linalg.norm(data.qvel[r.rootv:r.rootv + 2]))
        ended = int(r.state.motion_t[0]) >= r.state.motion_length - 1
        if data.qpos[r.rootq + 2] < 0.28 or data.xmat[r.root].reshape(3, 3)[2, 2] < 0.65:
            self.fail("Robot lost its upright stance")
            return

        if self.phase == "APPROACH" and ended:
            if speed > 0.06:
                self.fail("Approach ended without a stable stop")
                return
            self.hold_reference()
            self.start_tcp = tcp
            self.pick_high = cube + [0, 0, 0.065]
            self.pick_low = cube + [0, 0, 0.008]
            self.clear_target = self.start_tcp.copy()
            self.clear_target[2] = self.pick_high[2] + 0.045
            r.ik.solve(data, self.pick_low)
            if r.ik.last_error > 0.015:
                self.fail(f"Stopped outside grasp reach: IK residual {r.ik.last_error:.3f}m")
                return
            self.enter("CLEAR_ARM")
        elif self.phase == "CLEAR_ARM":
            self.update_clear_arm(elapsed, tcp)
        elif self.phase == "REACH":
            r.arm_target = self.start_tcp + smooth_fraction(elapsed / 2) * (self.pick_high - self.start_tcp)
            if elapsed >= 2.5:
                self.enter("LOWER")
        elif self.phase == "LOWER":
            r.arm_target = self.pick_high + smooth_fraction(elapsed / 2) * (self.pick_low - self.pick_high)
            if elapsed >= 2.5:
                if np.linalg.norm(tcp - self.pick_low) > 0.018:
                    self.fail("Gripper did not reach the cube before closing")
                    return
                self.enter("CLOSE")
        elif self.phase == "CLOSE":
            r.openings[1] = self.closing_command(elapsed)
            if elapsed >= 1.2:
                if np.any(self.contacts() < 0.1):
                    self.fail("Cube has no opposing finger contacts")
                    return
                self.lift_start = r.arm_target.copy()
                self.lift_goal = self.lift_start + [0, 0, 0.17]
                self.enter("LIFT")
        elif self.phase == "LIFT":
            r.arm_target = self.lift_start + smooth_fraction(elapsed / 3) * (self.lift_goal - self.lift_start)
            if elapsed >= 3.0 + self.wait_times["lift"]:
                if cube[2] < self.initial_cube[2] + 0.1 or np.any(self.contacts() < 0.1):
                    self.fail("Contact grasp failed the lift check")
                    return
                self.start_carry()
        elif self.phase == "CARRY":
            r.arm_target = data.qpos[:3] + self.carry_offset
            r.arm_target[2] = self.carry_height
            if ended:
                if speed > 0.06:
                    self.fail("Carry ended without a stable stop")
                    return
                self.hold_reference()
                self.start_tcp = tcp
                self.release_target = self.release_target_position()
                r.ik.solve(data, self.release_target)
                if r.ik.last_error > 0.02:
                    self.fail(f"Basket outside arm reach: {r.ik.last_error:.3f}m")
                    return
                self.enter("PLACE")
        elif self.phase == "PLACE":
            r.arm_target = self.start_tcp + smooth_fraction(elapsed / 2.5) * (self.release_target - self.start_tcp)
            if elapsed >= 3.0:
                if np.linalg.norm(tcp - self.release_target) > 0.02:
                    self.fail("Gripper did not reach the release position")
                    return
                self.enter("RELEASE")
        elif self.phase == "RELEASE":
            r.openings[1] = 0.035
            self.settled_s = self.settled_s + 0.02 if self.cube_in_basket() else 0
            if self.settled_s >= 0.8:
                self.success, self.done = True, True
                self.enter("SUCCESS")
                return
            elif elapsed > 5:
                self.fail("Released cube did not settle inside the basket")
                return

        if self.phase in ("LIFT", "CARRY", "PLACE"):
            self.contact_lost_s = self.contact_lost_s + 0.02 if np.any(self.contacts() < 0.03) else 0
            if self.contact_lost_s > 0.3 or np.linalg.norm(tcp - cube) > 0.11:
                self.fail("Cube slipped out of the gripper")
                return
        r.step()
        self.records.append({"time": float(data.time), "phase": self.phase,
                             "qpos": data.qpos.copy(), "qvel": data.qvel.copy(),
                             "tcp": data.site_xpos[r.ik.site].copy(),
                             "grasp_center": r.ik.position(data).copy(),
                             "cube": data.xpos[self.cube].copy(), "finger_forces": self.contacts()})
        if int(data.time) > self.last_print:
            self.last_print = int(data.time)
            print(f"t={data.time:.2f} {self.phase} base={data.qpos[:3].round(3)} "
                  f"cube={data.xpos[self.cube].round(3)} contacts={self.contacts().round(2)}", flush=True)

    def save(self, output: Path) -> None:
        import numpy as np
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        if self.records:
            np.savez_compressed(output / "trajectory.npz",
                                **{key: np.asarray([row[key] for row in self.records])
                                   for key in self.records[0]})
        report = {"success": self.success, "failure": self.failure, "phase": self.phase,
                  "simulated_seconds": float(self.r.data.time), "events": self.events,
                  "initial_root_xyz": self.initial_root.tolist(),
                  "initial_cube_xyz": self.initial_cube.tolist(),
                  "final_root_xyz": self.r.data.qpos[:3].tolist(),
                  "final_cube_xyz": self.r.data.xpos[self.cube].tolist(),
                  "base_support": False, "object_welds": False, "real_motor": True,
                  "visualization": getattr(self, "visualization", "headless"),
                  "policy": str(self.r.policy.model_path), "approach_reference": self.approach_metadata,
                  "wait_times_s": self.wait_times,
                  "scene": self.r.scene_path, "motion": self.motion_path,
                  "carry_reference": getattr(self, "carry_metadata", None)}
        if self.records:
            report["minimum_base_height_m"] = float(min(row["qpos"][self.r.rootq + 2] for row in self.records))
            report["minimum_base_up_z"] = float(min(
                1 - 2 * (row["qpos"][self.r.rootq + 4] ** 2 + row["qpos"][self.r.rootq + 5] ** 2)
                for row in self.records))
            report["maximum_cube_height_m"] = float(max(row["cube"][2] for row in self.records))
            report["initial_forward_distance_m"] = float(self.initial_cube[0] - self.initial_root[0])
            report["initial_horizontal_distance_m"] = float(np.linalg.norm(self.initial_cube[:2] - self.initial_root[:2]))
            report["cube_settled_inside_basket"] = self.cube_in_basket()
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Saved {output / 'report.json'}", flush=True)


def main(*, runtime_class=PolicyScene, task_class=PickCarryTask,
         default_scene: Path | None = None, default_output: Path | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=default_scene or ROOT / "any4hdmi/assets/robots/mini3_mjlab/scene_pick_carry.xml")
    parser.add_argument("--motion", type=Path, default=ROOT / "any4hdmi/output/mini3/sonic/motions/211117/Neutral_walk_forward_005__A057.npz")
    parser.add_argument("--policy", type=Path, help="Exported ONNX or companion YAML; default: trained Sonic checkpoint")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--start-paused", action="store_true")
    parser.add_argument("--duration", type=float, default=60, help="Maximum simulated seconds")
    parser.add_argument("--approach-distance", type=float, default=2.65, help="Reference walk distance in metres")
    parser.add_argument("--carry-distance", type=float, default=1.0, help="Reference carry walk distance in metres")
    parser.add_argument("--approach-wait", type=float, default=2.0, help="Standing wait after approaching the cubes, in seconds")
    parser.add_argument("--lift-wait", type=float, default=0.5, help="Wait after the unchanged 3-second lift, in seconds")
    parser.add_argument("--basket-wait", type=float, default=1.0, help="Standing wait after walking to the basket, in seconds")
    parser.add_argument("--output", type=Path, default=default_output or ROOT / "outputs/mini3_pick_carry")
    parser.add_argument("--camera", choices=("overview", "head_rgb", "left_gripper_rgb", "right_gripper_rgb"), default="overview")
    args = parser.parse_args()
    if args.headless and args.start_paused:
        parser.error("Headless runs cannot start paused")
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("Duration must be finite and positive")
    for name in ("approach_wait", "lift_wait", "basket_wait"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    setup_paths()
    policy = resolve_policy("sonic", args.motion, args.policy)
    runtime = runtime_class(args.scene, policy, args.motion)
    task = task_class(runtime, args.motion, approach_distance=args.approach_distance,
                         carry_distance=args.carry_distance, approach_wait=args.approach_wait,
                         lift_wait=args.lift_wait, basket_wait=args.basket_wait)
    task.visualization = "headless" if args.headless else "mujoco_viewer"
    viewer = None
    print("Trained Sonic policy + right-arm IK + independent contact gripper; pelvis support OFF.")
    print(f"Phase waits: approach={args.approach_wait:g}s, lift={args.lift_wait:g}s, basket={args.basket_wait:g}s.")
    print("P pause, F follow; F8 overview, F9 head, F10 left hand, F11 right hand; Esc exit.")
    try:
        if not args.headless:
            from mini3_task_viewer import Mini3TaskViewer
            viewer = Mini3TaskViewer(runtime.model, runtime.data, camera=args.camera,
                                      start_paused=args.start_paused)
            print("Viewer: mujoco_viewer (independent display state).")
        while not task.done and runtime.data.time < args.duration:
            started = time.perf_counter()
            if viewer is not None:
                viewer.poll_events()
                if not viewer.is_running():
                    break
            if viewer is None or not viewer.paused:
                task.step()
            if viewer is not None:
                viewer.render(runtime.data)
                time.sleep(max(0.0, 0.02 - (time.perf_counter() - started)))
        if not task.done:
            task.failure = "Stopped by viewer/user" if viewer is not None and runtime.data.time < args.duration else "Time limit reached"
    except KeyboardInterrupt:
        task.failure = "Interrupted by user"
    except (ValueError, RuntimeError, FloatingPointError, TypeError) as exc:
        task.fail(str(exc))
    finally:
        task.save(args.output)
        if viewer is not None:
            viewer.close()
    return 0 if task.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
