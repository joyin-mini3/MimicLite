#!/usr/bin/env python3
"""Record one seeded Mini3 task with pre-action observations and exact commands.

This module collects raw arrays only. It deliberately leaves image rendering and
dataset-specific action mappings to a separate export step.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterator


def limit_compute_threads() -> None:
    """Call before importing the simulation stack in a collector worker."""
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)


@contextmanager
def onnx_session_threads(count: int) -> Iterator[None]:
    """Limit sessions created here without changing interactive runtime defaults."""
    if count < 1:
        raise ValueError("ONNX thread count must be positive")
    import onnxruntime as ort
    original = ort.InferenceSession

    def create_session(path, sess_options=None, *args, **kwargs):
        options = sess_options if sess_options is not None else ort.SessionOptions()
        options.intra_op_num_threads = count
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        return original(path, options, *args, **kwargs)

    ort.InferenceSession = create_session
    try:
        yield
    finally:
        ort.InferenceSession = original


class ControlRecorder:
    """Synchronized control samples independent of post-step task diagnostics."""

    def __init__(self, task) -> None:
        import numpy as np
        self.task = task
        self.rows: list[dict[str, Any]] = []
        steps = np.asarray(getattr(task.r.state, "motion_future_steps", []), dtype=np.int64)
        self.reference_step_offsets = steps.copy() if steps.size else np.array(
            [-8, -4, -2, 0, 1, 2, 3, 4], dtype=np.int64)
        self.policy_input_keys = sorted(task.r.policy.policy_config.get("observation", {}))

    def observe(self, runtime, commands: dict[str, Any]) -> None:
        """Called after inference and before the first motor/physics substep."""
        import mujoco
        import numpy as np
        task, data = self.task, runtime.data
        if self.rows and self.rows[-1]["next_time"] == self.rows[-1]["time"]:
            raise RuntimeError("Previous control sample was not completed")
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, data.qpos[runtime.rootq + 3:runtime.rootq + 7])
        rotation = rotation.reshape(3, 3)
        frame = int(np.clip(runtime.state.motion_t[0], 0, task.reference.allowed_frame))
        reference = task.reference.get_slice(np.array([0]), np.array([frame]), self.reference_step_offsets)
        row = {
            "time": float(data.time), "qpos": data.qpos.copy(), "qvel": data.qvel.copy(),
            "phase": task.phase, "reference_time": float(task.reference_time),
            "reference_frame": frame, "reference_qpos": task.reference.qpos[frame].copy(),
            "reference_allowed_frame": int(task.reference.allowed_frame),
            "root_velocity_correction": task.reference.root_velocity_correction.copy(),
            "root_rotation_w": rotation.copy(),
            "root_linear_velocity_body": rotation.T @ data.qvel[runtime.rootv:runtime.rootv + 3],
            "root_angular_velocity_body": data.qvel[runtime.rootv + 3:runtime.rootv + 6].copy(),
            "tool_target": (np.full(3, np.nan) if runtime.tool_target is None
                            else runtime.tool_target.copy()),
            "tool_target_active": runtime.tool_target is not None,
            "tool_orientation": runtime.ik.orientation.copy(),
            "extra_target": runtime.extra_target.copy(),
            "extra_integral": runtime.extra_integral.copy(),
            "next_time": float(data.time), "next_qpos": data.qpos.copy(),
            "next_qvel": data.qvel.copy(), "executed_substeps": 0,
            "action_executed": False,
        }
        row.update({name: value.copy() for name, value in commands.items()})
        for name in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w",
                     "joint_pos", "joint_vel", "step"):
            row["reference_" + name] = getattr(reference, name)[0].copy()
        for key in self.policy_input_keys:
            if key in runtime.policy.state_dict:
                row["policy_input_" + key] = np.asarray(runtime.policy.state_dict[key]).copy()
        self.rows.append(row)

    def complete_step(self, runtime) -> None:
        """Keep the actual successor, including partial failed physics steps."""
        if not self.rows:
            return
        row = self.rows[-1]
        elapsed = float(runtime.data.time) - row["time"]
        row["next_time"] = float(runtime.data.time)
        row["next_qpos"] = runtime.data.qpos.copy()
        row["next_qvel"] = runtime.data.qvel.copy()
        row["executed_substeps"] = int(round(elapsed / runtime.model.opt.timestep))
        row["action_executed"] = row["executed_substeps"] == 10

    def save(self, output: Path, *, wall_seconds: float, onnx_threads: int) -> dict[str, Any]:
        import numpy as np
        runtime, task = self.task.r, self.task
        self.complete_step(runtime)
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        arrays = ({name: np.asarray([row[name] for row in self.rows]) for name in self.rows[0]}
                  if self.rows else {"time": np.empty(0, dtype=float)})
        arrays.update(reference_step_offsets=self.reference_step_offsets,
                      terminal_qpos=runtime.data.qpos.copy(), terminal_qvel=runtime.data.qvel.copy(),
                      terminal_time=np.array(float(runtime.data.time)))
        np.savez_compressed(output / "control_samples.npz", **arrays)
        model = runtime.model
        metadata = {
            "schema_version": 1, "seed": None if task.episode is None else task.episode["seed"],
            "success": bool(task.success), "failure": task.failure,
            "samples": len(self.rows), "complete_actions": sum(row["action_executed"] for row in self.rows),
            "control_hz": 50, "physics_hz": 500, "physics_substeps_per_action": 10,
            "wall_seconds": wall_seconds, "onnx_threads": onnx_threads,
            "synchronization": "qpos/qvel/time and policy_input are pre-action; next_* are post-action",
            "high_level_action": "reference_qpos and gated reference_* samples; not policy_q",
            "low_level_action": "policy_q/dq/effort/kp/kd, extra_q/effort, gripper_opening",
            "extra_q": "Exact clipped extra_target + 2 * extra_integral sent to the motor model",
            "gripper_opening": "Left/right commanded half-opening in meters, ordered like gripper_actuators",
            "root_velocity_frames": "Linear velocity rotated world-to-body; angular freejoint velocity already body-frame",
            "policy_joint_names": list(runtime.names), "extra_joint_names": list(runtime.extra_names),
            "policy_qpos_indices": runtime.qids.tolist(), "policy_qvel_indices": runtime.vids.tolist(),
            "extra_qpos_indices": runtime.extra_qids.tolist(), "extra_qvel_indices": runtime.extra_vids.tolist(),
            "policy_actuator_indices": runtime.aids.tolist(), "extra_actuator_indices": runtime.extra_aids.tolist(),
            "gripper_actuator_indices": runtime.grippers.tolist(),
            "gripper_actuators": [model.actuator(int(index)).name for index in runtime.grippers],
            "gripper_joint_names": [model.joint(int(model.actuator_trnid[index, 0])).name
                                    for index in runtime.grippers],
            "gripper_qpos_indices": [int(model.jnt_qposadr[model.actuator_trnid[index, 0]])
                                     for index in runtime.grippers],
            "gripper_qvel_indices": [int(model.jnt_dofadr[model.actuator_trnid[index, 0]])
                                     for index in runtime.grippers],
            "extra_motor_kp": runtime.extra_motor.kp.tolist(), "extra_motor_kd": runtime.extra_motor.kd.tolist(),
            "reference_body_names": list(task.reference.body_names),
            "reference_joint_names": list(task.reference.joint_names),
            "reference_step_offsets": self.reference_step_offsets.tolist(),
            "policy_input_config": runtime.policy.policy_config.get("observation", {}),
            "root_qpos_start": int(runtime.rootq), "root_qvel_start": int(runtime.rootv),
            "scene": runtime.scene_path,
            "array_shapes": {name: list(array.shape) for name, array in arrays.items()},
            "array_dtypes": {name: str(array.dtype) for name, array in arrays.items()},
        }
        (output / "control_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return metadata


def collect_episode(*, seed: int, output: Path, duration: float = 45., onnx_threads: int = 1,
                    scene: Path | None = None, motion: Path | None = None,
                    reference: Path | None = None, policy: Path | None = None) -> dict[str, Any]:
    """Run the existing randomized controller and retain successful/failed attempts."""
    if seed < 0 or not math.isfinite(duration) or duration <= 0 or onnx_threads < 1:
        raise ValueError("Expected non-negative seed and positive finite duration/thread count")
    output = Path(output).resolve()
    for name in ("report.json", "control_samples.npz", "episode_scene.xml", "episode.json"):
        if (output / name).exists():
            raise FileExistsError(f"Refusing to overwrite an existing attempt: {output / name}")
    limit_compute_threads()
    from mini3_pick_carry_policy import (DEFAULT_MOTION, DEFAULT_SCENE, SAVED_V1,
                                         PolicyOnlyScene, PolicyPickCarryTask,
                                         build_task_reference, resolve_policy, setup_paths)
    from mini3_randomized_task import generate_episode, retarget_reference
    setup_paths()
    scene, motion = Path(scene or DEFAULT_SCENE), Path(motion or DEFAULT_MOTION)
    reference = Path(reference or SAVED_V1)
    started = time.perf_counter()
    generated, episode = generate_episode(scene, output, seed=seed, target="random")
    print(f"Collect seed={seed} target={episode['target_color']}", flush=True)
    plan = build_task_reference(reference, original_motion=motion, scene_path=scene,
                                approach_overlap=6., basket_overlap=.6, arm_source="command",
                                arm_reference_bias=[0., 0., 0., 0.])
    retarget_reference(plan, episode)
    with onnx_session_threads(onnx_threads):
        runtime = PolicyOnlyScene(generated, resolve_policy("sonic", motion, policy), motion)
    task = PolicyPickCarryTask(runtime, plan, target_body=episode["target_body"], episode=episode)
    task.visualization = "headless"
    recorder = ControlRecorder(task)
    runtime.control_observer = recorder.observe
    try:
        while not task.done and runtime.data.time < duration:
            task.step()
            recorder.complete_step(runtime)
        if not task.done:
            task.fail("Time limit reached")
    except KeyboardInterrupt:
        task.fail("Interrupted by user")
    except Exception as exc:
        task.fail(f"{type(exc).__name__}: {exc}")
    finally:
        recorder.complete_step(runtime)
        task.save(output)
        metadata = recorder.save(output, wall_seconds=time.perf_counter() - started,
                                 onnx_threads=onnx_threads)
        runtime.control_observer = None
    print(f"Collected success={task.success} samples={metadata['samples']} "
          f"wall={metadata['wall_seconds']:.2f}s output={output}", flush=True)
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=45.)
    parser.add_argument("--onnx-threads", type=int, default=1)
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--motion", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--policy", type=Path)
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("Seed must be non-negative")
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("Duration must be finite and positive")
    if args.onnx_threads < 1:
        parser.error("ONNX thread count must be positive")
    metadata = collect_episode(**vars(args))
    return 0 if metadata["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
