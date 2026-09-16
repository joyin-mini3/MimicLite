#!/usr/bin/env python3
"""Measure actual Mini3 tracking-policy ground reach with the extended grippers.

Run with the local MJLab environment. The robot is free standing: no support,
external forces, or reference-pose updates are applied after initialization.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path

import sim2sim_mini3_mimiclite as launcher

ROOT = Path(__file__).resolve().parent
DEFAULT_MOTIONS = tuple(
    ROOT / "any4hdmi/output/mini3/sonic/motions/211117" / f"{name}__A057.npz"
    for name in (
        "Neutral_stoop_down_001",
        "Neutral_stoop_down_003",
        "injured_torso_stoop_down_003",
    )
)


def find_sonic_export() -> tuple[Path, float, float]:
    """Discover a complete export by checkpoint identity, without cache hashes."""
    import yaml

    checkpoint = launcher.BUNDLE / "models/sonic/checkpoint_5200.pt"
    candidates = []
    for marker in launcher.CACHE.glob("*/ready.json"):
        training = marker.parent / "training.yaml"
        if not training.is_file():
            continue
        try:
            config = yaml.safe_load(training.read_text())
            source = Path(config.get("checkpoint_path", "")).expanduser().resolve()
            if source != checkpoint.resolve():
                continue
            metadata = json.loads(marker.read_text())
            policy = marker.parent / metadata["policy"]
            if policy.is_file() and policy.with_suffix(".yaml").is_file():
                candidates.append((marker.stat().st_mtime_ns, policy, metadata))
        except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError):
            continue
    if candidates:
        _, policy, metadata = max(candidates, key=lambda item: item[0])
        return policy.resolve(), float(metadata["env_dt"]), float(metadata["sim_dt"])

    config = checkpoint.parent / "play_local.yaml"
    if not config.is_file():
        config = checkpoint.parent / "config.yaml"
    return launcher.prepare_export(
        launcher.existing_file(checkpoint), launcher.existing_file(config),
        launcher.existing_file(DEFAULT_MOTIONS[0]),
        argparse.Namespace(seed=0, reexport=False),
    )


def evaluate(
    policy: Path, motion: Path, robot_xml: Path, output: Path,
    *, env_dt: float, sim_dt: float, max_duration: float,
) -> dict:
    import mujoco
    import numpy as np
    from sim2real.sim_env.integrated_sim2sim import (
        IntegratedSim2Sim, IntegratedSim2SimArgs, IntegratedSimRuntime,
    )
    from sim2real.sim_env.native_mimiclite import _Playback

    class GripperSimulation(IntegratedSimRuntime):
        """Keep the original policy's 21 motor commands separate from fingers."""

        def __init__(self, robot_cfg):
            super().__init__(
                replace(robot_cfg, mjcf_path=str(robot_xml), strict_joint_contract=False),
                sim_dt=sim_dt, headless=True, key_callback=None,
            )
            self.gripper_actuators = [
                self.mj_model.actuator(f"{side}_gripper_ctrl").id
                for side in ("left", "right")
            ]
            expected_joints = set(robot_cfg.joint_names) | {
                f"{side}_gripper_{suffix}"
                for side in ("left", "right")
                for suffix in ("finger_joint", "follower_joint")
            }
            actual_joints = {
                self.mj_model.joint(i).name for i in range(self.mj_model.njnt)
                if self.mj_model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
            }
            if actual_joints != expected_joints or self.mj_model.nu != 23:
                raise ValueError("Expected the original 21 joints plus four gripper slides and 23 actuators")
            if np.any(self.mj_model.eq_type == mujoco.mjtEq.mjEQ_WELD):
                raise ValueError("Reach audit requires a robot XML without body welds/supports")
            for side in ("left", "right"):
                for suffix in ("finger_joint", "follower_joint"):
                    address = self.mj_model.joint(f"{side}_gripper_{suffix}").qposadr
                    self.mj_data.qpos[address] = 0.035

        def sim_step(self):
            self.compute_torques()
            self.mj_data.ctrl[:] = self.torques
            self.mj_data.ctrl[self.gripper_actuators] = 0.035
            mujoco.mj_step(self.mj_model, self.mj_data)

    args = IntegratedSim2SimArgs(
        policy_config=str(policy.with_suffix(".yaml")), motion_path=str(motion),
        robot="mini3", sim_dt=sim_dt, env_dt=env_dt, initial_pause_s=0.0,
        headless=True, seed=0,
    )
    # Construction validates the complete original policy joint/observation
    # contract before attaching a physical model with additional finger joints.
    runtime = IntegratedSim2Sim(args)
    original_mass = float(runtime.sim.mj_model.body_mass.sum())
    runtime.sim.stop()
    runtime.sim = GripperSimulation(runtime.robot_cfg)
    runtime._reset_playback()
    sim = runtime.sim
    model, data = sim.mj_model, sim.mj_data
    site_ids = [model.site(f"{side}_gripper_tcp").id for side in ("left", "right")]
    finger_ids = [
        model.geom(f"{side}_gripper_finger_{sign}_geom").id
        for side in ("left", "right") for sign in ("positive", "negative")
    ]
    # Evaluate reference FK in a separate MjData; never overwrite live physics.
    reference_data = mujoco.MjData(model)
    reference_data.qpos[:] = data.qpos
    reference_tcp = []
    root_index = runtime.state_processor.motion_body_names.index("base_link")
    motion_joint_names = runtime.state_processor.motion_joint_names
    motion_indices = [motion_joint_names.index(name) for name in runtime.robot_cfg.joint_names]
    reference_root_min = float("inf")
    for frame in range(int(runtime.state_processor.motion_length)):
        reference = runtime._motion_frame(frame)
        reference_data.qpos[:3] = reference.body_pos_w[0, 0, root_index]
        reference_data.qpos[3:7] = reference.body_quat_w[0, 0, root_index]
        reference_data.qpos[sim.qpos_adrs] = reference.joint_pos[0, 0, motion_indices]
        mujoco.mj_forward(model, reference_data)
        reference_tcp.append(reference_data.site_xpos[site_ids].copy())
        reference_root_min = min(reference_root_min, float(reference_data.qpos[2]))

    playback = _Playback(runtime, loop=False, paused=False)
    samples, finger_bottoms = [], []
    stopped_reason = "motion_finished"
    try:
        while not playback.finished:
            playback.step()
            reference_root, _ = runtime._motion_root_state()
            # mj_step already computed body poses at the last dynamics step;
            # mj_forward makes geometric measurements match recorded qpos.
            mujoco.mj_forward(model, data)
            up = float(data.xmat[sim.pelvis_body_id].reshape(3, 3)[2, 2])
            samples.append(np.r_[
                data.time, int(runtime.state_processor.motion_t[0]), data.qpos[:3],
                reference_root, up, data.site_xpos[site_ids].ravel(), data.qpos,
            ])
            finger_bottoms.append([
                float(data.geom_xpos[g, 2] - np.abs(data.geom_xmat[g].reshape(3, 3)[2]) @ model.geom_size[g])
                for g in finger_ids
            ])
            if data.qpos[2] < 0.12 or up < 0.1:
                stopped_reason = "fall_threshold"
                break
            if data.time >= max_duration and not playback.finished:
                stopped_reason = "duration_limit"
                break
    finally:
        sim.stop()

    recorded = np.asarray(samples)
    tcp = recorded[:, 9:15].reshape(-1, 2, 3)
    finger_bottoms = np.asarray(finger_bottoms)
    reference_tcp = np.asarray(reference_tcp)
    errors = np.linalg.norm(recorded[:, 2:5] - recorded[:, 5:8], axis=1)
    stable = (recorded[:, 4] > 0.22) & (recorded[:, 8] > 0.4) & (errors < 0.2)
    stable_tcp = tcp[stable, :, 2].min(axis=0).tolist() if np.any(stable) else None
    result = {
        "policy": str(policy), "motion": str(motion), "robot_xml": str(robot_xml),
        "support": False, "external_stabilization": False,
        "real_motor": sim.real_motor is not None,
        "model_total_mass_kg": float(model.body_mass.sum()),
        "model_mass_increase_kg": float(model.body_mass.sum()) - original_mass,
        "policy_dt_s": env_dt, "physics_dt_s": sim_dt,
        "simulated_s": float(recorded[-1, 0]),
        "completed_motion": bool(playback.finished), "stopped_reason": stopped_reason,
        "root_z_min_m": float(recorded[:, 4].min()),
        "root_z_final_m": float(recorded[-1, 4]),
        "root_up_dot_min": float(recorded[:, 8].min()),
        "root_error_mean_m": float(errors.mean()), "root_error_max_m": float(errors.max()),
        "side_order": ["left", "right"],
        "actual_tcp_min_m": tcp[:, :, 2].min(axis=0).tolist(),
        "actual_stable_tcp_min_m": stable_tcp,
        "stable_definition": "root_z > 0.22 m and root_up_dot > 0.4 and root_error < 0.2 m",
        "stable_frames": int(stable.sum()), "recorded_frames": len(recorded),
        "finger_order": ["left_positive", "left_negative", "right_positive", "right_negative"],
        "actual_finger_surface_min_m": finger_bottoms.min(axis=0).tolist(),
        "reference_fk_tcp_min_m": reference_tcp[:, :, 2].min(axis=0).tolist(),
        "reference_root_z_min_m": reference_root_min,
        "cube_top_height_m": 0.04,
        "finger_surface_reached_cube_top_height": bool(np.min(finger_bottoms) <= 0.04),
        "caveat": "Reference FK is not policy performance. Height overlap alone does not establish a stable grasp.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{motion.stem}.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(
        output / f"{motion.stem}.npz", samples=recorded, qpos=recorded[:, 15:],
        tcp=tcp, finger_bottoms=finger_bottoms, stable=stable, reference_tcp=reference_tcp,
    )
    print("RESULT " + json.dumps(result), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, help="Trusted exported .onnx with sibling .yaml; default: cached Sonic 5200")
    parser.add_argument("--motion", type=Path, action="append", help="Reference .npz, repeat for multiple clips")
    parser.add_argument("--robot-xml", type=Path, default=ROOT / "any4hdmi/assets/robots/mini3_mjlab/mini3_gripper.xml")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/mini3_policy_reach_audit")
    parser.add_argument("--max-duration", type=float, default=60.0)
    options = parser.parse_args()
    if not math.isfinite(options.max_duration) or options.max_duration <= 0:
        parser.error("--max-duration must be finite and positive")
    launcher.setup_paths()
    if options.policy:
        policy = launcher.existing_file(options.policy)
        if policy.suffix != ".onnx":
            parser.error("--policy must be an exported .onnx")
        launcher.existing_file(policy.with_suffix(".yaml"))
        env_dt, sim_dt = 0.02, 0.002
    else:
        policy, env_dt, sim_dt = find_sonic_export()
    motions = [launcher.existing_file(path) for path in options.motion or DEFAULT_MOTIONS]
    for motion in motions:
        launcher.motion_dataset(motion)
    robot_xml = launcher.existing_file(options.robot_xml)
    results = [evaluate(
        policy, motion, robot_xml, options.output.resolve(), env_dt=env_dt,
        sim_dt=sim_dt, max_duration=options.max_duration,
    ) for motion in motions]
    (options.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
