#!/usr/bin/env python3
"""Run the trained Mini3 policy with articulated forearms and seven-joint arm IK."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, TYPE_CHECKING

from mini3_pick_carry import ArmIK, PickCarryTask, PolicyScene, main, smooth_fraction
from sim2sim_mini3_mimiclite import ROOT, setup_paths

if TYPE_CHECKING:
    import mujoco
    import numpy as np


class SevenJointArmIK(ArmIK):
    """Solve position and palm orientation without changing the physical state."""

    def __init__(self, model: mujoco.MjModel, side: str = "right", tool_offset: float = -0.015) -> None:
        import mujoco
        import numpy as np
        from audit_mini3_arm_collisions import arm_collision_pairs
        super().__init__(model, side, tool_offset)
        self.names += [f"{side}_{part}_joint" for part in
                       ("elbow_yaw", "wrist_roll", "wrist_pitch")]
        self.jids = np.array([model.joint(name).id for name in self.names])
        self.qids = model.jnt_qposadr[self.jids]
        self.vids = model.jnt_dofadr[self.jids]
        self.lower, self.upper = model.jnt_range[self.jids].T.copy()
        # Use the outward-elbow posture family for this side-grasp task. The
        # XML retains its full joint ranges; this avoids a redundant IK flip
        # that can reach the high waypoint but not the subsequent low grasp.
        self.lower[1] = max(self.lower[1], -1.4 if side == "right" else -0.15)
        self.upper[1] = min(self.upper[1], 0.15 if side == "right" else 1.4)
        self.lower[0] = max(self.lower[0], -2.0)
        self.upper[3] = min(self.upper[3], 2.4)
        self.lower[5], self.upper[5] = -1.5, 1.5
        self.orientation = np.eye(3)
        self.preferred_posture = np.array([-0.1, -0.3, -0.3, 0.45, 0.0, 0.2, -0.25])
        if side == "left":
            self.preferred_posture[[1, 2, 4, 5]] *= -1
        self.last_solution = None
        self.allow_posture_reseed = True
        self.held_body_id = None
        self.last_orientation_error = math.inf
        pairs = arm_collision_pairs(model, side)
        self.target_pair_start = len(pairs)
        cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pick_cube_3_geom")
        if cube >= 0:
            pairs.extend(tuple(sorted((cube, model.geom(f"{side}_gripper_finger_{part}_geom").id)))
                         for part in ("positive", "negative"))
        self.collision_pairs = np.asarray(pairs, dtype=int)
        self.allow_target_contact = False
        self.pair_radii = model.geom_rbound[self.collision_pairs].sum(axis=1)
        self.plane_pairs = np.any(model.geom_type[self.collision_pairs] == mujoco.mjtGeom.mjGEOM_PLANE, axis=1)
        self.last_clearance = math.inf

    def clearances(self) -> np.ndarray:
        import numpy as np
        from audit_mini3_arm_collisions import collision_distance
        first, second = self.collision_pairs.T
        bounds = np.linalg.norm(self.scratch.geom_xpos[first] - self.scratch.geom_xpos[second], axis=1) - self.pair_radii
        distances = np.full(len(first), 0.03)
        for index in np.flatnonzero((bounds < 0.03) | self.plane_pairs):
            if self.allow_target_contact and index >= self.target_pair_start:
                continue
            a, b = self.collision_pairs[index]
            distances[index] = collision_distance(self.model, self.scratch, int(a), int(b), 0.03)
        return distances

    def solve(self, data: mujoco.MjData, target: np.ndarray) -> np.ndarray:
        import mujoco
        import numpy as np
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation
        self.scratch.qpos[:] = data.qpos
        low, high = self.lower, self.upper
        seed = np.clip(data.qpos[self.qids], low + 1e-5, high - 1e-5)
        posture = self.preferred_posture if self.allow_posture_reseed else seed
        held_transform = None
        if self.held_body_id is not None:
            body = self.held_body_id
            hand_rotation = data.site_xmat[self.site].reshape(3, 3)
            offset = hand_rotation.T @ (data.xpos[body] - self.position(data))
            orientation = hand_rotation.T @ data.xmat[body].reshape(3, 3)
            joint = self.model.body_jntadr[body]
            if joint < 0 or self.model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
                raise ValueError("The held object must have a free joint")
            held_transform = (int(self.model.jnt_qposadr[joint]), offset, orientation)

        def residual(q):
            self.scratch.qpos[self.qids] = q
            mujoco.mj_kinematics(self.model, self.scratch)
            rotation = self.scratch.site_xmat[self.site].reshape(3, 3)
            if held_transform is not None:
                address, offset, object_rotation = held_transform
                self.scratch.qpos[address:address + 3] = self.position(self.scratch) + rotation @ offset
                mujoco.mju_mat2Quat(self.scratch.qpos[address + 3:address + 7],
                                    (rotation @ object_rotation).ravel())
                # Only the IK scratch state follows the measured grasp. The
                # physical free object remains driven by contact and friction.
                mujoco.mj_kinematics(self.model, self.scratch)
            orientation = Rotation.from_matrix(self.orientation.T @ rotation).as_rotvec()
            distances = self.clearances()
            return np.r_[20 * (self.position(self.scratch) - target),
                         0.8 * orientation, 0.015 * (q - posture),
                         0.002 * (q - seed),
                         20 * np.maximum(0, 0.004 - distances)]

        actual_rotation = data.site_xmat[self.site].reshape(3, 3)
        if (np.linalg.norm(self.position(data) - target) < 1e-8
                and np.linalg.norm(actual_rotation - self.orientation) < 1e-8
                and np.all(data.qpos[self.qids] > low + 1e-5)
                and np.all(data.qpos[self.qids] < high - 1e-5)):
            residual(seed)
            if self.clearances().min() >= 0.004:
                self.last_error = self.last_orientation_error = 0.0
                self.last_clearance = float(self.clearances().min())
                self.last_solution = seed.copy()
                return seed
        warm = seed if self.last_solution is None else np.clip(self.last_solution, low + 1e-5, high - 1e-5)
        guesses = [warm]
        if self.allow_posture_reseed:
            guesses.append(np.clip(self.preferred_posture, low + 1e-5, high - 1e-5))
        solutions = [least_squares(residual, guess, bounds=(low + 1e-5, high - 1e-5),
                                   max_nfev=35, ftol=1e-5, xtol=1e-5, gtol=1e-5)
                     for guess in guesses]
        solution = min(solutions, key=lambda candidate: float(candidate.fun @ candidate.fun))
        residual(solution.x)
        self.last_error = float(np.linalg.norm(self.position(self.scratch) - target))
        rotation = self.scratch.site_xmat[self.site].reshape(3, 3)
        self.last_orientation_error = float(np.linalg.norm(
            Rotation.from_matrix(self.orientation.T @ rotation).as_rotvec()))
        self.last_clearance = float(self.clearances().min())
        self.last_solution = solution.x.copy()
        return solution.x


class ArticulatedPolicyScene(PolicyScene):
    """Original 21 policy motors plus six independent elbow-equivalent motors."""

    def __init__(self, scene: Path, policy_config: Path, motion: Path) -> None:
        import numpy as np
        from mini3_extra_arm_motor import ExtraArmMotorModel
        super().__init__(scene, policy_config, motion)
        self.extra_names = [f"{side}_{part}_joint" for side in ("left", "right")
                            for part in ("elbow_yaw", "wrist_roll", "wrist_pitch")]
        self.extra_jids = np.array([self.model.joint(name).id for name in self.extra_names])
        self.extra_qids = self.model.jnt_qposadr[self.extra_jids]
        self.extra_vids = self.model.jnt_dofadr[self.extra_jids]
        self.extra_aids = np.array([self.model.actuator(f"{name}_ctrl").id for name in self.extra_names])
        if not np.array_equal(self.model.actuator_trnid[self.extra_aids, 0], self.extra_jids):
            raise ValueError("Additional motor names must target their matching joints")
        # The added axes have much less load inertia than the original elbow.
        # Keep the same calibrated motor plant and tune its position controller.
        self.extra_motor = ExtraArmMotorModel(kp=20.0, kd=0.4)
        self.stow_angles = np.array([0.0, 0.6, 0.4, 0.0, -0.6, 0.4])
        self.extra_target = self.stow_angles.copy()
        self.ik = SevenJointArmIK(self.model)
        from audit_mini3_arm_collisions import arm_collision_pairs
        self.unexpected_pairs = set(arm_collision_pairs(self.model))
        cube_geom = self.model.geom("pick_cube_3_geom").id
        self.pregrasp_pairs = {
            tuple(sorted((cube_geom, self.model.geom(f"{side}_gripper_finger_{part}_geom").id)))
            for side in ("left", "right") for part in ("positive", "negative")
        }
        self.collision_stop_depth = 0.002
        self.collision_phase = "APPROACH"
        self.contact_checks = 0
        self.unexpected_contacts: dict[tuple[int, int], dict[str, Any]] = {}
        self.arm_indices = np.array([self.names.index(name) for name in self.ik.names[:4]])
        self.arm_kp, self.arm_kd = 30.0, 0.8
        self.arm_integral = np.zeros(7)
        self.freeze_arm_command = False
        self.reset()

    def reset(self) -> None:
        super().reset()
        if hasattr(self, "extra_motor"):
            import mujoco
            self.extra_motor.reset()
            self.extra_target[:] = self.stow_angles
            self.data.qpos[self.extra_qids] = self.stow_angles
            mujoco.mj_forward(self.model, self.data)
            self.sync()
            self.freeze_arm_command = False
            self.ik.last_solution = None
            self.ik.allow_posture_reseed = True
            self.ik.held_body_id = None
            self.ik.allow_target_contact = False
            self.ik.orientation[:] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            self.collision_phase = "APPROACH"
            self.contact_checks = 0
            self.unexpected_contacts.clear()

    def check_arm_contacts(self) -> None:
        """Check solved contacts at every 2 ms physics step, including both arms."""
        self.contact_checks += 1
        observed_pairs = set()
        for contact in self.data.contact:
            pair = tuple(sorted((int(contact.geom1), int(contact.geom2))))
            early_target_contact = (self.collision_phase in ("APPROACH", "CLEAR_ARM", "REACH")
                                    and pair in getattr(self, "pregrasp_pairs", ()))
            if (pair not in self.unexpected_pairs and not early_target_contact) or contact.dist >= -1e-7:
                continue
            depth = float(-contact.dist)
            entry = self.unexpected_contacts.setdefault(pair, {
                "geoms": [self.model.geom(i).name for i in pair],
                "first_time_s": float(self.data.time), "first_phase": self.collision_phase,
                "physics_steps": 0, "maximum_penetration_m": 0.0,
            })
            if pair not in observed_pairs:
                entry["physics_steps"] += 1
                observed_pairs.add(pair)
            entry["maximum_penetration_m"] = max(entry["maximum_penetration_m"], depth)
            if depth > self.collision_stop_depth:
                raise RuntimeError(f"Unexpected arm collision: {entry['geoms']}, depth={depth:.4f}m")

    def step(self) -> None:
        import mujoco
        import numpy as np
        self.sync()
        target = None
        if self.arm_target is not None:
            previous = self.data.qpos[self.ik.qids] if self.arm_command is None else self.arm_command
            goal = previous if self.freeze_arm_command else self.ik.solve(self.data, self.arm_target)
            target = previous + np.clip(goal - previous, -0.03, 0.03)
            self.arm_command = target
            reference = getattr(self.state, "motion_dataset", None)
            if reference is not None:
                first = min(max(0, int(self.state.motion_t[0]) + 1), reference.num_steps - 1)
                reference._storage["joint_pos"][first:, self.arm_indices] = target[:4]
                reference._storage["joint_vel"][first:, self.arm_indices] = 0
        result = self.policy.step()
        if result is None:
            raise RuntimeError("Policy inference failed")
        q, dq, effort, kp, kd = [np.asarray(value).copy() for value in result]
        extra_effort = np.zeros(6)
        if target is not None:
            self.arm_integral = np.clip(self.arm_integral +
                                       0.02 * (target - self.data.qpos[self.ik.qids]), -0.12, 0.12)
            command = np.clip(target + 2.0 * self.arm_integral,
                              self.model.jnt_range[self.ik.jids, 0],
                              self.model.jnt_range[self.ik.jids, 1])
            q[self.arm_indices] = command[:4]
            kp[self.arm_indices] = self.arm_kp
            kd[self.arm_indices] = self.arm_kd
            effort[self.arm_indices] = self.data.qfrc_bias[self.ik.vids[:4]]
            self.extra_target[3:] = command[4:]
            extra_effort[3:] = self.data.qfrc_bias[self.ik.vids[4:]]
        else:
            self.arm_integral[:] = 0
            self.arm_command = None
        for _ in range(10):
            torque = self.motor.compute(q, self.data.qpos[self.qids], self.data.qvel[self.vids],
                                        target_vel=dq, effort=effort, kp=kp, kd=kd)
            extra = self.extra_motor.compute(self.extra_target, self.data.qpos[self.extra_qids],
                                             self.data.qvel[self.extra_vids], effort=extra_effort)
            if not np.isfinite(torque).all() or not np.isfinite(extra).all():
                raise FloatingPointError("Non-finite motor torque")
            self.data.ctrl[self.aids] = np.clip(torque, -self.limits, self.limits)
            self.data.ctrl[self.extra_aids] = np.clip(extra, -12.5, 12.5)
            self.data.ctrl[self.grippers] = self.openings
            if self.data.eq_active[self.model.equality("base_support_weld").id]:
                raise RuntimeError("Pelvis support must stay disabled during the policy task")
            mujoco.mj_step(self.model, self.data)
            self.check_arm_contacts()
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError("Non-finite simulation state")


class ArticulatedPickCarryTask(PickCarryTask):
    def __init__(self, runtime: ArticulatedPolicyScene, motion: Path, **kwargs: Any) -> None:
        from scipy.spatial.transform import Rotation
        super().__init__(runtime, motion, **kwargs)
        # A horizontal diagonal approach suits the shorter, straight forearm.
        self.grasp_rotation = Rotation.from_euler("z", -0.5).as_matrix()
        self.r.ik.orientation = self.grasp_rotation.copy()

    def release_target_position(self) -> np.ndarray:
        """Use the nearest basket interior point with 30 mm object clearance."""
        import numpy as np
        model, data = self.r.model, self.r.data
        rotation = data.xmat[self.basket].reshape(3, 3)
        cube_rotation = data.xmat[self.cube].reshape(3, 3)
        half_extents = np.abs(rotation.T @ cube_rotation) @ model.geom_size[self.cube_geom]
        front, back = (model.geom(f"pick_basket_{side}") for side in ("front", "back"))
        left, right = (model.geom(f"pick_basket_{side}") for side in ("left", "right"))
        lower = np.array([back.pos[0] + back.size[0], right.pos[1] + right.size[1]])
        upper = np.array([front.pos[0] - front.size[0], left.pos[1] - left.size[1]])
        lower += half_extents[:2] + 0.03
        upper -= half_extents[:2] + 0.03
        if np.any(lower > upper):
            raise ValueError("Basket interior is too small for the cube and release clearance")
        local_cube = rotation.T @ (data.xpos[self.cube] - data.xpos[self.basket])
        local_cube[:2] = np.clip(local_cube[:2], lower, upper)
        local_cube[2] = max(0.18, *(wall.pos[2] + wall.size[2] + half_extents[2] + 0.02
                                  for wall in (front, back, left, right)))
        self.release_cube_target = data.xpos[self.basket] + rotation @ local_cube
        # Preserve the measured grasp offset: the cube, rather than only the
        # tool centre, must be over the safe interior before opening.
        return self.release_cube_target - (data.xpos[self.cube] - self.r.ik.position(data))

    def update_clear_arm(self, elapsed: float, tcp: np.ndarray) -> None:
        from scipy.spatial.transform import Rotation
        if elapsed < 1.5:
            self.r.arm_target = self.start_tcp + smooth_fraction(elapsed / 1.5) * (self.outward_target - self.start_tcp)
            self.r.ik.orientation = self.initial_palm_rotation
        else:
            alpha = smooth_fraction((elapsed - 1.5) / 3.0)
            self.r.arm_target = self.outward_target + alpha * (self.clear_target - self.outward_target)
            self.r.ik.orientation = (self.grasp_rotation @
                                     Rotation.from_rotvec((1 - alpha) * self.initial_palm_rotvec).as_matrix())
        if elapsed >= 5.0:
            self.start_tcp = tcp
            self.enter("REACH")

    def step(self) -> None:
        if self.phase in ("LIFT", "CARRY", "PLACE"):
            self.r.openings[1] = self.closing_command(1.0)
        previous = len(self.records)
        super().step()
        if len(self.records) > previous:
            r = self.r
            self.records[-1].update({
                "arm_command": (r.data.qpos[r.ik.qids] if r.arm_command is None else r.arm_command).copy(),
                "arm_integral": r.arm_integral.copy(),
                "extra_command": r.extra_target.copy(),
            })

    def closing_command(self, elapsed: float) -> float:
        import numpy as np
        # A diagonal grasp sees the cube's projected width. Maintain 2 mm of
        # servo preload per finger as the cube aligns, including during carry.
        data = self.r.data
        hand_rotation = data.site_xmat[self.r.ik.site].reshape(3, 3)
        cube_rotation = hand_rotation.T @ data.xmat[self.cube].reshape(3, 3)
        half_extents = np.abs(cube_rotation) @ self.r.model.geom_size[self.cube_geom]
        target = float(np.clip(half_extents[1] - 0.002, 0.0, 0.035))
        return 0.035 + smooth_fraction(elapsed / 1.0) * (target - 0.035)

    def enter(self, phase: str) -> None:
        if phase == "CLOSE":
            import numpy as np
            from scipy.spatial.transform import Rotation
            rotation = self.r.data.site_xmat[self.r.ik.site].reshape(3, 3)
            relative = rotation.T @ (self.r.data.xpos[self.cube] - self.r.ik.position(self.r.data))
            cube_rotation = rotation.T @ self.r.data.xmat[self.cube].reshape(3, 3)
            half_extents = np.abs(cube_rotation) @ self.r.model.geom_size[self.cube_geom]
            angle_error = float(np.linalg.norm(Rotation.from_matrix(self.r.ik.orientation.T @ rotation).as_rotvec()))
            self.grasp_alignment = {"cube_in_pad_frame_m": relative.tolist(),
                                    "orientation_error_rad": angle_error}
            if (abs(relative[0]) + half_extents[0] > 0.033
                    or abs(relative[1]) + half_extents[1] > 0.033
                    or abs(relative[2]) > 0.010 or angle_error > 0.12):
                self.fail("Cube is not safely centred between the open horizontal finger pads")
                return
        self.r.freeze_arm_command = phase == "CLOSE"
        self.r.collision_phase = phase
        # Keep the local posture continuous while leaving the hip/table area;
        # redundant posture changes are permitted once the hand is above it.
        self.r.ik.allow_posture_reseed = phase in ("REACH", "LOWER")
        self.r.ik.allow_target_contact = phase in ("LOWER", "CLOSE", "LIFT", "CARRY", "PLACE", "RELEASE", "SUCCESS")
        self.r.ik.held_body_id = self.cube if phase in ("LIFT", "CARRY", "PLACE") else None
        if phase == "LIFT":
            self.lift_start = self.r.ik.position(self.r.data).copy()
            self.lift_goal = self.lift_start + [0, 0, 0.17]
            self.r.ik.orientation = self.r.data.site_xmat[self.r.ik.site].reshape(3, 3).copy()
        if phase == "CLEAR_ARM":
            from scipy.spatial.transform import Rotation
            # The wrist now keeps the pads horizontal. Centre them vertically
            # on the cube instead of retaining the old tilted-hand offset.
            self.pick_low = self.r.data.xpos[self.cube].copy()
            self.pick_low[2] += 0.002
            self.outward_target = self.start_tcp.copy()
            self.outward_target[0] -= 0.05
            self.outward_target[1] = self.pick_low[1] - 0.03
            self.clear_target[1] = self.outward_target[1]
            self.initial_palm_rotation = self.r.data.site_xmat[self.r.ik.site].reshape(3, 3).copy()
            self.initial_palm_rotvec = Rotation.from_matrix(
                self.grasp_rotation.T @ self.initial_palm_rotation).as_rotvec()
            self.r.ik.last_solution = self.r.data.qpos[self.r.ik.qids].copy()
        super().enter(phase)

    def save(self, output: Path) -> None:
        super().save(output)
        path = Path(output) / "report.json"
        report = json.loads(path.read_text())
        report["extra_arm_joints"] = self.r.extra_names
        report["extra_arm_joint_axes_local"] = {
            name: self.r.model.jnt_axis[joint_id].tolist()
            for name, joint_id in zip(self.r.extra_names, self.r.extra_jids, strict=True)
        }
        report["forearm_extensions"] = {
            side: {
                "length_m": float(self.r.model.body(f"{side}_gripper_palm").pos[0]),
                "mass_kg": float(self.r.model.body(f"{side}_forearm_extension").mass[0]),
            }
            for side in ("left", "right")
        }
        report["extra_arm_motor"] = "elbow_pitch / 4310p; current-loop, torque-speed limit and KT"
        report["arm_ik_dofs"] = 7
        report["grasp_alignment"] = getattr(self, "grasp_alignment", None)
        report["grasp_yaw_rad"] = -0.5
        report["gripper_preload_m"] = 0.002
        report["release_cube_target_xyz"] = (self.release_cube_target.tolist()
                                               if hasattr(self, "release_cube_target") else None)
        report["extra_arm_control_gains"] = {"kp": self.r.extra_motor.kp.tolist(),
                                              "kd": self.r.extra_motor.kd.tolist()}
        report["original_right_arm_control_gains"] = {"kp": self.r.arm_kp, "kd": self.r.arm_kd}
        report["walking_stow_targets_rad"] = dict(zip(self.r.extra_names, self.r.stow_angles.tolist(), strict=True))
        report["arm_collision_monitor"] = {
            "physics_step_s": 0.002, "checks": self.r.contact_checks,
            "stop_penetration_m": self.r.collision_stop_depth,
            "unexpected_contacts": list(self.r.unexpected_contacts.values()),
        }
        path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    setup_paths()
    raise SystemExit(main(runtime_class=ArticulatedPolicyScene, task_class=ArticulatedPickCarryTask,
                         default_scene=ROOT / "any4hdmi/assets/robots/mini3_mjlab/scene_pick_carry_7dof.xml",
                         default_output=ROOT / "outputs/mini3_pick_carry_7dof"))
