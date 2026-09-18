#!/usr/bin/env python3
"""Whole-body MimicLite task: policy commands every original joint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from mini3_motion_reference import MotionReference, install_reference
from mini3_pick_carry import PickCarryTask, resolve_policy, smooth_fraction
from mini3_pick_carry_7dof import ArticulatedPolicyScene, SevenJointArmIK
from mini3_policy_task_reference import TaskReferencePlan, build_task_reference, DEFAULT_MOTION, DEFAULT_SCENE
from sim2sim_mini3_mimiclite import ROOT, setup_paths


SAVED_V1 = ROOT / "outputs/task_versions/mini3_pick_carry_v1_50mm/validated_run/trajectory.npz"


class AddedJointPlanner(SevenJointArmIK):
    """Optimize only the three added right-arm coordinates in scratch data."""

    def __init__(self, model: mujoco.MjModel) -> None:
        super().__init__(model)
        self.names = self.names[4:]
        self.jids, self.qids, self.vids = self.jids[4:], self.qids[4:], self.vids[4:]
        self.lower, self.upper = model.jnt_range[self.jids].T.copy()
        self.preferred_posture = np.zeros(3)
        self.clearance_margin = .008
        self.orientation_weight = 3.0

    def solve(self, data: mujoco.MjData, target: np.ndarray) -> np.ndarray:
        self.scratch.qpos[:] = data.qpos
        seed = np.clip(data.qpos[self.qids], self.lower + 1e-5, self.upper - 1e-5)
        held = None
        if self.held_body_id is not None:
            body = self.held_body_id
            hand = data.site_xmat[self.site].reshape(3, 3)
            joint = self.model.body_jntadr[body]
            held = (int(self.model.jnt_qposadr[joint]),
                    hand.T @ (data.xpos[body] - self.position(data)),
                    hand.T @ data.xmat[body].reshape(3, 3))

        def residual(q: np.ndarray) -> np.ndarray:
            # No original joint is an optimizer variable, even in scratch FK.
            self.scratch.qpos[self.qids] = q
            mujoco.mj_kinematics(self.model, self.scratch)
            rotation = self.scratch.site_xmat[self.site].reshape(3, 3)
            if held is not None:
                address, offset, local_rotation = held
                self.scratch.qpos[address:address + 3] = self.position(self.scratch) + rotation @ offset
                mujoco.mju_mat2Quat(self.scratch.qpos[address + 3:address + 7],
                                    (rotation @ local_rotation).ravel())
                mujoco.mj_kinematics(self.model, self.scratch)
            return np.r_[20 * (self.position(self.scratch) - target),
                         self.orientation_weight * Rotation.from_matrix(self.orientation.T @ rotation).as_rotvec(),
                         .025 * (q - self.preferred_posture), .01 * (q - seed),
                         30 * np.maximum(0, self.clearance_margin - self.clearances())]

        guess = seed if self.last_solution is None else self.last_solution
        result = least_squares(residual, np.clip(guess, self.lower + 1e-5, self.upper - 1e-5),
                               bounds=(self.lower + 1e-5, self.upper - 1e-5), max_nfev=30,
                               ftol=1e-5, xtol=1e-5, gtol=1e-5)
        residual(result.x)
        self.last_solution = result.x.copy()
        self.last_error = float(np.linalg.norm(self.position(self.scratch) - target))
        self.last_orientation_error = float(np.linalg.norm(Rotation.from_matrix(
            self.orientation.T @ self.scratch.site_xmat[self.site].reshape(3, 3)).as_rotvec()))
        self.last_clearance = float(self.clearances().min())
        return result.x


class PolicyOnlyScene(ArticulatedPolicyScene):
    """Pass all five policy command arrays directly to the original motor plant."""

    def __init__(self, scene: Path, policy_config: Path, motion: Path) -> None:
        super().__init__(scene, policy_config, motion)
        self.ik = AddedJointPlanner(self.model)
        self.extra_integral = np.zeros(6)
        self.extra_reference = self.stow_angles.copy()
        self.tool_target = None
        self.extra_planning_enabled = True
        self.last_policy_command = np.zeros(21)
        self.last_motor_command = np.zeros(21)
        self.policy_output_override_max = 0.0
        self.policy_output_checks = 0

    def reset(self) -> None:
        super().reset()
        if hasattr(self, "extra_reference"):
            self.extra_integral[:] = 0
            self.extra_reference[:] = self.stow_angles
            self.tool_target = None
            self.last_policy_command[:] = 0
            self.last_motor_command[:] = 0
            self.policy_output_override_max = 0.0
            self.policy_output_checks = 0

    def step(self) -> None:
        self.sync()
        desired_extra = self.extra_reference.copy()
        if self.tool_target is not None and self.extra_planning_enabled:
            self.ik.preferred_posture = desired_extra[3:].copy()
            desired_extra[3:] = self.ik.solve(self.data, self.tool_target)
        self.extra_target += np.clip(desired_extra - self.extra_target, -.03, .03)
        self.extra_integral = np.clip(self.extra_integral + .02 * (
            self.extra_target - self.data.qpos[self.extra_qids]), -.12, .12)
        self.extra_integral[:3] = 0
        if not self.extra_planning_enabled:
            self.extra_integral[:] = 0
        extra_command = np.clip(self.extra_target + 2 * self.extra_integral,
                                self.model.jnt_range[self.extra_jids, 0],
                                self.model.jnt_range[self.extra_jids, 1])
        result = self.policy.step()
        if result is None:
            raise RuntimeError("Policy inference failed")
        q, dq, effort, kp, kd = [np.asarray(array).copy() for array in result]
        self.last_policy_command = q.copy()
        self.last_motor_command = q.copy()
        extra_effort = np.zeros(6)
        if self.collision_phase != "APPROACH":
            extra_effort[3:] = self.data.qfrc_bias[self.extra_vids[3:]]
        for _ in range(10):
            # The original 21 joints, including both original four-joint arms,
            # use exactly the policy's targets, gains, and feedforward effort.
            torque = self.motor.compute(q, self.data.qpos[self.qids], self.data.qvel[self.vids],
                                        target_vel=dq, effort=effort, kp=kp, kd=kd)
            self.policy_output_checks += 1
            self.policy_output_override_max = max(self.policy_output_override_max,
                *(float(np.max(np.abs(value - original))) for value, original in
                  zip((q, dq, effort, kp, kd), result)))
            extra = self.extra_motor.compute(extra_command, self.data.qpos[self.extra_qids],
                                             self.data.qvel[self.extra_vids], effort=extra_effort)
            if not np.isfinite(torque).all() or not np.isfinite(extra).all():
                raise FloatingPointError("Non-finite motor torque")
            if self.data.eq_active[self.model.equality("base_support_weld").id]:
                raise RuntimeError("Pelvis support must stay disabled")
            self.data.ctrl[self.aids] = np.clip(torque, -self.limits, self.limits)
            self.data.ctrl[self.extra_aids] = np.clip(extra, -12.5, 12.5)
            self.data.ctrl[self.grippers] = self.openings
            mujoco.mj_step(self.model, self.data)
            self.check_arm_contacts()
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError("Non-finite simulation state")


class GatedReference(MotionReference):
    """Hold future samples at a contact gate until the actual grasp is ready."""

    def __init__(self, qpos: np.ndarray) -> None:
        super().__init__(qpos)
        self.allowed_frame = self.num_steps - 1

    def get_slice(self, motion_ids: np.ndarray, starts: np.ndarray, steps: np.ndarray) -> Any:
        from sim2real.rl_policy.utils.motion import MotionData
        if np.any(np.asarray(motion_ids) != 0):
            raise ValueError("Only motion zero is available")
        indices = np.clip(np.asarray(starts, dtype=np.int64).reshape(-1, 1)
                          + np.asarray(steps, dtype=np.int64).reshape(1, -1),
                          0, self.allowed_frame)
        return MotionData(**{name: values[indices] for name, values in self._storage.items()})


class PolicyPickCarryTask(PickCarryTask):
    """Overlapping reference playback with measured contact and release gates."""

    def __init__(self, runtime: PolicyOnlyScene, plan: TaskReferencePlan, *, gate_timeout: float = 3.0) -> None:
        if not np.isfinite(gate_timeout) or gate_timeout <= 0:
            raise ValueError("Gate timeout must be finite and positive")
        self.r, self.plan = runtime, plan
        self.cube = runtime.model.body("pick_cube_3").id
        self.cube_geom = runtime.model.geom("pick_cube_3_geom").id
        self.basket = runtime.model.body("pick_basket").id
        self.finger_geoms = [runtime.model.geom(f"right_gripper_finger_{part}_geom").id
                            for part in ("positive", "negative")]
        self.initial_root = runtime.data.qpos[:3].copy()
        self.initial_cube = runtime.data.xpos[self.cube].copy()
        self.phase, self.phase_started = "APPROACH", 0.0
        self.events, self.records = [], []
        self.done = self.success = False
        self.failure = None
        self.reference_time = 0.0
        self.phase_times = {event["phase"]: float(event["time"]) for event in plan.events}
        self.reference = GatedReference(plan.qpos)
        self.reference.allowed_frame = round(self.phase_times["LIFT"] / .02) - 1
        install_reference(runtime.policy, self.reference, paused=True)
        runtime.initialize_robot(plan.qpos[0])
        runtime.policy.state_dict["paused"] = False
        self.scratch = mujoco.MjData(runtime.model)
        self.gate_timeout = gate_timeout
        self.gate_wait = 0.0
        self.gate_waits = {name: 0.0 for name in ("CLOSE", "LIFT", "CARRY", "RELEASE")}
        self.settled_s = self.contact_lost_s = 0.0
        self.close_started = None
        self.grasp_verified = False
        self.lift_verified = False
        self.last_print = -1
        self.tool_height_offset = .005
        self.enter("APPROACH")

    def enter(self, phase: str) -> None:
        self.r.collision_phase = phase
        self.r.ik.allow_target_contact = phase in ("LOWER", "CLOSE", "LIFT", "CARRY", "PLACE", "RELEASE", "SUCCESS")
        self.r.ik.held_body_id = self.cube if phase in ("LIFT", "CARRY", "PLACE") else None
        super().enter(phase)

    def grasp_alignment(self) -> tuple[bool, np.ndarray]:
        data, ik = self.r.data, self.r.ik
        rotation = data.site_xmat[ik.site].reshape(3, 3)
        relative = rotation.T @ (data.xpos[self.cube] - ik.position(data))
        half = np.abs(rotation.T @ data.xmat[self.cube].reshape(3, 3)) @ self.r.model.geom_size[self.cube_geom]
        upright = abs(float(rotation[2, 1])) < .15 and abs(float(rotation[2, 0])) < .15
        ready = bool(np.all(np.abs(relative[:2]) + half[:2] < .033)
                     and abs(relative[2]) < .010 and upright)
        return ready, relative

    def closing_command(self, elapsed: float) -> float:
        rotation = self.r.data.site_xmat[self.r.ik.site].reshape(3, 3)
        half = np.abs(rotation.T @ self.r.data.xmat[self.cube].reshape(3, 3)) @ self.r.model.geom_size[self.cube_geom]
        closed = float(np.clip(half[1] - .002, 0, .035))
        return .035 + smooth_fraction(elapsed) * (closed - .035)

    def release_ready(self) -> bool:
        data = self.r.data
        rotation = data.xmat[self.basket].reshape(3, 3)
        local = rotation.T @ (data.xpos[self.cube] - data.xpos[self.basket])
        half = np.abs(rotation.T @ data.xmat[self.cube].reshape(3, 3)) @ self.r.model.geom_size[self.cube_geom]
        return bool(np.all(np.abs(local[:2]) + half[:2] < np.array([.12, .11]) - .012)
                    and local[2] - half[2] > .125)

    def set_extra_plan(self, sample: dict[str, Any]) -> None:
        r = self.r
        r.extra_reference[:] = sample["extra_command"]
        self.scratch.qpos[:] = self.plan.sample_source_qpos(sample["arm_source_time"])
        mujoco.mj_kinematics(r.model, self.scratch)
        source_rotation = self.scratch.site_xmat[r.ik.site].reshape(3, 3)
        source_position = r.ik.position(self.scratch).copy()
        source_base = self.scratch.xmat[r.root].reshape(3, 3)
        if self.phase in ("APPROACH", "CLEAR_ARM", "CARRY"):
            actual_base = r.data.xmat[r.root].reshape(3, 3)
            r.tool_target = r.data.xpos[r.root] + actual_base @ source_base.T @ (
                source_position - self.scratch.xpos[r.root])
            r.ik.orientation = actual_base @ source_base.T @ source_rotation
            if self.phase == "CARRY":
                r.tool_target[2] = source_position[2]
                r.ik.orientation = source_rotation.copy()
        else:
            r.tool_target = source_position
            r.ik.orientation = source_rotation.copy()
        if self.phase in ("REACH", "LOWER", "CLOSE", "LIFT", "CARRY", "PLACE", "RELEASE"):
            r.tool_target[2] += self.tool_height_offset
        if self.phase == "APPROACH":
            r.tool_target = None

    def step(self) -> None:
        if self.done:
            return
        r, data = self.r, self.r.data
        if data.qpos[2] < .28 or data.xmat[r.root].reshape(3, 3)[2, 2] < .65:
            self.fail("Robot lost its upright stance")
            return
        sample = self.plan.sample(self.reference_time)
        wanted_phase = sample["phase"]
        hold = False
        if wanted_phase == "CLOSE":
            if self.close_started is None:
                ready, offset = self.grasp_alignment()
                if ready:
                    self.close_started = float(data.time)
                else:
                    hold = True
            if self.close_started is not None:
                r.openings[1] = self.closing_command(float(data.time) - self.close_started)
        if self.reference_time + .02 >= self.phase_times["LIFT"] and not self.grasp_verified:
            if self.close_started is not None and data.time - self.close_started >= 1.0 and np.all(self.contacts() > .1):
                self.grasp_verified = True
                self.reference.allowed_frame = round(self.phase_times["CARRY"] / .02) - 1
            else:
                hold = True
                wanted_phase = "CLOSE"
        if (self.grasp_verified and not self.lift_verified
                and data.xpos[self.cube][2] >= self.initial_cube[2] + .08
                and np.all(self.contacts() >= .03)):
            self.lift_verified = True
            self.reference.allowed_frame = round(self.phase_times["RELEASE"] / .02) - 1
        if wanted_phase == "CARRY" and not self.lift_verified:
            hold = True
            wanted_phase = "LIFT"
        if wanted_phase in ("RELEASE", "SUCCESS") and self.phase not in ("RELEASE", "SUCCESS"):
            if not self.release_ready():
                hold = True
                wanted_phase = "PLACE"
            else:
                wanted_phase = "RELEASE"
                self.reference.allowed_frame = self.reference.num_steps - 1
        if self.phase == "RELEASE":
            wanted_phase = "RELEASE"
        if wanted_phase != self.phase:
            self.enter(wanted_phase)
        self.set_extra_plan(sample)
        if self.phase in ("LIFT", "CARRY", "PLACE"):
            r.openings[1] = self.closing_command(1.)
            self.contact_lost_s = self.contact_lost_s + .02 if np.any(self.contacts() < .03) else 0.
            if self.contact_lost_s > .3:
                self.fail("Cube slipped out of the gripper")
                return
        if self.phase == "RELEASE":
            r.openings[1] = .035
            self.settled_s = self.settled_s + .02 if self.cube_in_basket() else 0.
            if self.settled_s >= .8:
                self.success = self.done = True
                self.enter("SUCCESS")
                return
            if data.time - self.phase_started > 5.:
                self.fail("Released cube did not settle inside the basket")
                return
        self.gate_wait = self.gate_wait + .02 if hold else 0.
        if hold:
            gate = "CLOSE" if not self.grasp_verified else ("RELEASE" if self.phase == "PLACE" else "CARRY")
            self.gate_waits[gate] += .02
            if self.gate_wait > self.gate_timeout:
                _, offset = self.grasp_alignment()
                self.fail(f"Measured {gate} gate not reached; cube/pad offset={offset.round(4).tolist()}")
                return
        r.state.motion_t[:] = min(round(self.reference_time / .02), self.reference.allowed_frame) - 1
        r.policy.state_dict["paused"] = False
        r.step()
        self.records.append({
            "time": float(data.time), "reference_time": self.reference_time, "phase": self.phase,
            "qpos": data.qpos.copy(), "qvel": data.qvel.copy(),
            "tcp": data.site_xpos[r.ik.site].copy(), "grasp_center": r.ik.position(data).copy(),
            "cube": data.xpos[self.cube].copy(), "finger_forces": self.contacts(),
            "policy_command": r.last_policy_command.copy(), "motor_command": r.last_motor_command.copy(),
            "extra_command": r.extra_target.copy(), "extra_integral": r.extra_integral.copy(),
            "body_source_time": sample["body_source_time"], "arm_source_time": sample["arm_source_time"],
        })
        if not hold:
            self.reference_time = min(self.reference_time + .02, self.plan.duration)
        if int(data.time) > self.last_print:
            self.last_print = int(data.time)
            print(f"t={data.time:.2f} ref={self.reference_time:.2f} {self.phase} "
                  f"base={data.qpos[:3].round(3)} cube={data.xpos[self.cube].round(3)} "
                  f"pad={r.ik.position(data).round(3)} force={self.contacts().round(2)}", flush=True)

    def save(self, output: Path) -> None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        self.plan.save(output)
        if self.records:
            np.savez_compressed(output / "trajectory.npz", **{
                name: np.asarray([row[name] for row in self.records]) for name in self.records[0]})
        r = self.r
        report = {
            "success": self.success, "failure": self.failure, "phase": self.phase,
            "simulated_seconds": float(r.data.time), "reference_seconds": self.reference_time,
            "events": self.events, "visualization": self.visualization,
            "scene": r.scene_path, "policy": str(r.policy.model_path),
            "source_trajectory": self.plan.metadata["source_trajectory"],
            "source_sha256": hashlib.sha256(Path(self.plan.metadata["source_trajectory"]).read_bytes()).hexdigest(),
            "initial_root_xyz": self.initial_root.tolist(), "initial_cube_xyz": self.initial_cube.tolist(),
            "final_root_xyz": r.data.qpos[:3].tolist(), "final_cube_xyz": r.data.xpos[self.cube].tolist(),
            "initial_forward_distance_m": float(self.initial_cube[0] - self.initial_root[0]),
            "cube_settled_inside_basket": self.cube_in_basket(),
            "base_support": False, "object_welds": False, "real_motor": True,
            "original_joint_controller": "MimicLite policy, all 21 command channels passed through unchanged",
            "original_policy_joints": r.names, "planned_joints": r.extra_names,
            "online_ik_joints": r.ik.names if r.extra_planning_enabled else [],
            "extra_joint_controller": "three-joint IK" if r.extra_planning_enabled else "recorded joint trajectory",
            "policy_output_override_max": r.policy_output_override_max,
            "policy_output_checks": r.policy_output_checks, "gate_waits_s": self.gate_waits,
            "extra_planner_clearance_m": r.ik.clearance_margin,
            "extra_planner_orientation_weight": r.ik.orientation_weight,
            "tool_height_offset_m": self.tool_height_offset,
            "arm_collision_monitor": {"physics_step_s": .002, "checks": r.contact_checks,
                "stop_penetration_m": r.collision_stop_depth,
                "unexpected_contacts": list(r.unexpected_contacts.values())},
            "reference_plan": self.plan.metadata,
        }
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Saved {output / 'report.json'}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--reference", type=Path, default=SAVED_V1)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--start-paused", action="store_true")
    parser.add_argument("--duration", type=float, default=45.)
    parser.add_argument("--approach-overlap", type=float, default=3.)
    parser.add_argument("--basket-overlap", type=float, default=.6)
    parser.add_argument("--arm-source", choices=("actual", "command"), default="command")
    parser.add_argument("--arm-reference-bias", type=float, nargs=4, default=[0., 0., 0., 0.])
    parser.add_argument("--extra-mode", choices=("planned", "reference"), default="planned")
    parser.add_argument("--gate-timeout", type=float, default=3.)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/mini3_pick_carry_policy")
    args = parser.parse_args()
    if args.headless and args.start_paused:
        parser.error("Headless runs cannot start paused")
    if not np.isfinite(args.duration) or args.duration <= 0:
        parser.error("Duration must be finite and positive")
    if not np.isfinite(args.gate_timeout) or args.gate_timeout <= 0:
        parser.error("Gate timeout must be finite and positive")
    setup_paths()
    plan = build_task_reference(args.reference, original_motion=args.motion, scene_path=args.scene,
                               approach_overlap=args.approach_overlap, basket_overlap=args.basket_overlap,
                               arm_source=args.arm_source, arm_reference_bias=args.arm_reference_bias)
    runtime = PolicyOnlyScene(args.scene, resolve_policy("sonic", args.motion, args.policy), args.motion)
    runtime.extra_planning_enabled = args.extra_mode == "planned"
    task = PolicyPickCarryTask(runtime, plan, gate_timeout=args.gate_timeout)
    task.visualization = "headless" if args.headless else "mujoco_viewer"
    viewer = None
    try:
        if not args.headless:
            from mini3_task_viewer import Mini3TaskViewer
            viewer = Mini3TaskViewer(runtime.model, runtime.data, start_paused=args.start_paused)
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
                time.sleep(max(0, .02 - (time.perf_counter() - started)))
        if not task.done:
            task.fail("Time limit reached" if runtime.data.time >= args.duration else "Stopped by viewer/user")
    except KeyboardInterrupt:
        task.fail("Interrupted by user")
    except (ValueError, RuntimeError, FloatingPointError, TypeError) as exc:
        task.fail(str(exc))
    finally:
        task.save(args.output)
        if viewer is not None:
            viewer.close()
    return 0 if task.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
