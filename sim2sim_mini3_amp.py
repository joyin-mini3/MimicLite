# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mini3-Walk-AMP sim-to-sim deployment script.

Loads a trained Mini3-Walk-AMP (Daisy) TorchScript policy and runs it in MuJoCo
with keyboard-controlled velocity commands. Defaults match training on
``MINI3_REAL_MOTOR_DAISY_CFG`` (``mini3_daisy_shell`` MJCF + PD / friction).

Usage:
    python sim2sim_mini3_amp.py \\
        --load_model logs/rsl_rl/mini3_walk_amp_daisy/<run>/exported/policy.pt \\
        [--headless]
    python sim2sim_mini3_amp.py --load_model <policy.pt> --ros2

IMPORTANT — mini3_daisy_shell/mjcf/mini3.xml already enables the freejoint in scene.xml:
    <freejoint name="floating_base" />

Observation layout (72 dims, frame_stack=1):
    [ 0: 3]  base_ang_vel      = omega in body frame
    [ 3: 6]  projected_gravity = gvec in body frame
    [ 6: 9]  velocity_commands = [vx, vy, dyaw]
    [ 9:30]  joint_pos_rel     = q - default_pos in lab order
    [30:51]  joint_vel_rel     = dq * 0.1 in lab order
    [51:72]  last_action       = previous action in lab order

Keyboard controls:
    8 / 2  : increase / decrease forward speed (vx)
    4 / 6  : increase / decrease lateral speed (vy)
    7 / 9  : turn left / right (dyaw)
    0      : reset all commands and robot state
    F      : toggle camera follow
"""

import sys
from pathlib import Path

import numpy as np
import mujoco
import mujoco_viewer
import glfw
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import torch
import cv2
import time

from robolab.assets import ISAAC_DATA_DIR

# Reuse the exact MINI3_REAL_MOTOR_DAISY_CFG torque pipeline (T-N clip, action
# delay, parallel-ankle motor model) from the Mini3-Walk sim2sim so all three
# scripts share a single source of truth for the motor contract.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from sim2sim_mini3_walk import (  # noqa: E402
    Ros2MujocoPublisher,
    _build_real_motor_runtime,
    _compute_applied_torque,
)


# Joint friction: (dof_frictionloss, dof_damping) — MINI3_REAL_MOTOR_DAISY_CFG
JOINT_FRICTION = {
    "left_hip_pitch_joint":      (0.477, 0.25),
    "right_hip_pitch_joint":     (0.477, 0.25),
    "left_hip_roll_joint":       (0.53, 0.25),
    "right_hip_roll_joint":      (0.53, 0.25),
    "left_hip_yaw_joint":        (0.53, 0.25),
    "right_hip_yaw_joint":       (0.53, 0.25),
    "left_knee_pitch_joint":     (0.477, 0.25),
    "right_knee_pitch_joint":    (0.477, 0.25),
    "waist_yaw_joint":           (0.53, 0.25),
    "left_ankle_pitch_joint":    (0.7, 0.1),
    "right_ankle_pitch_joint":   (0.7, 0.1),
    "left_ankle_roll_joint":     (0.7, 0.1),
    "right_ankle_roll_joint":    (0.7, 0.1),
    "left_shoulder_pitch_joint": (0.35, 0.25),
    "right_shoulder_pitch_joint":(0.35, 0.25),
    "left_shoulder_roll_joint":  (0.35, 0.25),
    "right_shoulder_roll_joint": (0.35, 0.25),
    "left_shoulder_yaw_joint":   (0.35, 0.25),
    "right_shoulder_yaw_joint":  (0.35, 0.25),
    "left_elbow_pitch_joint":    (0.35, 0.25),
    "right_elbow_pitch_joint":   (0.35, 0.25),
}


# ── Keyboard command state ─────────────────────────────────────────────────────

class cmd:
    vx = 0.0
    vy = 0.0
    dyaw = 0.0
    vx_increment = 0.1
    vy_increment = 0.1
    dyaw_increment = 0.1

    # Match the command ranges saved with the Mini3-AMP checkpoint.
    min_vx = -0.5
    max_vx = 1.6
    min_vy = -0.5
    max_vy = 0.5
    min_dyaw = -1.5
    max_dyaw = 1.5
    camera_follow = True
    reset_requested = False

    @classmethod
    def print_velocity_command(cls):
        """Print the velocity command currently sent to the policy."""
        tqdm.write(
            "[velocity command] "
            f"vx={float(cls.vx):+.2f} m/s, "
            f"vy={float(cls.vy):+.2f} m/s, "
            f"wz(dyaw)={float(cls.dyaw):+.2f} rad/s"
        )

    @classmethod
    def update_vx(cls, delta):
        cls.vx = np.clip(cls.vx + delta, cls.min_vx, cls.max_vx)
        cls.print_velocity_command()

    @classmethod
    def update_vy(cls, delta):
        cls.vy = np.clip(cls.vy + delta, cls.min_vy, cls.max_vy)
        cls.print_velocity_command()

    @classmethod
    def update_dyaw(cls, delta):
        cls.dyaw = np.clip(cls.dyaw + delta, cls.min_dyaw, cls.max_dyaw)
        cls.print_velocity_command()

    @classmethod
    def toggle_camera_follow(cls):
        cls.camera_follow = not cls.camera_follow
        print(f"Camera follow: {cls.camera_follow}")

    @classmethod
    def reset(cls):
        cls.vx = 0.0
        cls.vy = 0.0
        cls.dyaw = 0.0
        cls.print_velocity_command()


def install_viewer_keyboard_controls(viewer):
    """Install velocity controls directly on the focused MuJoCo window.

    ``pynput`` relies on a global keyboard hook, which Wayland and many remote
    desktop sessions block. GLFW receives events from the MuJoCo window itself.
    Unhandled keys are forwarded to the viewer so its camera shortcuts continue
    to work.
    """
    default_callback = viewer._key_callback
    key_groups = {
        "vx_up": {glfw.KEY_8, glfw.KEY_KP_8},
        "vx_down": {glfw.KEY_2, glfw.KEY_KP_2},
        "vy_up": {glfw.KEY_4, glfw.KEY_KP_4},
        "vy_down": {glfw.KEY_6, glfw.KEY_KP_6},
        "dyaw_up": {glfw.KEY_7, glfw.KEY_KP_7},
        "dyaw_down": {glfw.KEY_9, glfw.KEY_KP_9},
        "reset": {glfw.KEY_0, glfw.KEY_KP_0},
        "camera_follow": {glfw.KEY_F},
    }
    controlled_keys = set().union(*key_groups.values())

    def key_callback(window, key, scancode, action, mods):
        if key in controlled_keys:
            if action in (glfw.PRESS, glfw.REPEAT):
                if key in key_groups["vx_up"]:
                    cmd.update_vx(cmd.vx_increment)
                elif key in key_groups["vx_down"]:
                    cmd.update_vx(-cmd.vx_increment)
                elif key in key_groups["vy_up"]:
                    cmd.update_vy(cmd.vy_increment)
                elif key in key_groups["vy_down"]:
                    cmd.update_vy(-cmd.vy_increment)
                elif key in key_groups["dyaw_up"]:
                    cmd.update_dyaw(cmd.dyaw_increment)
                elif key in key_groups["dyaw_down"]:
                    cmd.update_dyaw(-cmd.dyaw_increment)
                elif key in key_groups["camera_follow"]:
                    cmd.toggle_camera_follow()
                elif key in key_groups["reset"]:
                    cmd.reset_requested = True
                    print("Reset requested (0 key pressed)")
            # These keys overlap with viewer shortcuts (for example F changes
            # playback speed), so do not forward handled command keys.
            return

        default_callback(window, key, scancode, action, mods)

    # Keep the Python callback alive for as long as the GLFW window exists.
    viewer._sim2sim_key_callback = key_callback
    glfw.set_key_callback(viewer.window, key_callback)


# ── Observation extraction ─────────────────────────────────────────────────────

def get_obs(data):
    """Extract IMU + kinematic state from MuJoCo data.

    With freejoint enabled:
      data.qpos = [x, y, z, qw, qx, qy, qz, j0..j20]  (28 entries)
      data.qvel  = [vx, vy, vz, wx, wy, wz, dj0..dj20] (27 entries)

    framequat sensor returns [w, x, y, z]; scipy R.from_quat expects [x, y, z, w].
    Free-joint qvel[3:6] gives angular velocity directly in the body frame.
    """
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)

    # framequat sensor: [w, x, y, z] → scipy convention: [x, y, z, w]
    quat = data.sensor("base_link_site_quat").data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)

    # Linear velocity in body frame (for display only)
    v = r.apply(dq[:3], inverse=True).astype(np.double)

    # Angular velocity in body frame (free-joint convention: qvel[3:6] is body frame)
    omega = dq[3:6].astype(np.double)

    # Projected gravity: unit gravity (0,0,-1) expressed in body frame
    gvec = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.double)

    return q, dq, quat, v, omega, gvec


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """PD torque: (target_q - q) * kp + (target_dq - dq) * kd."""
    return (target_q - q) * kp + (target_dq - dq) * kd


# ── Main simulation loop ───────────────────────────────────────────────────────

def run_mujoco(policy, cfg, headless=False):
    """Run MuJoCo sim2sim with the Mini3 AMP policy.

    Args:
        policy:   Loaded TorchScript policy.
        cfg:      Sim2simCfg instance (sim_config + robot_config).
        headless: If True, render offscreen and save to simulation_mini3_amp.mp4.
    """
    print("=" * 60)
    print("Mini3-Walk-AMP (Daisy) sim2sim  |  keyboard controls:")
    print("  8 / 2  : increase / decrease forward speed (vx)")
    print("  4 / 6  : increase / decrease lateral speed (vy)")
    print("  7 / 9  : turn left / right (dyaw)")
    print("  0      : reset commands and robot state")
    print("  F      : toggle camera follow")
    print("=" * 60)
    cmd.print_velocity_command()

    # ── Build MuJoCo model ─────────────────────────────────────────────────────
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    data = mujoco.MjData(model)
    rng = np.random.default_rng(getattr(cfg.sim_config, "seed", 42))
    motor_rt = _build_real_motor_runtime(cfg, rng)

    velocity_print_hz = float(getattr(cfg.sim_config, "velocity_print_hz", 2.0))
    velocity_print_every_n_steps = None
    if velocity_print_hz > 0.0:
        velocity_print_every_n_steps = max(
            1,
            int(round(1.0 / (velocity_print_hz * cfg.sim_config.dt))),
        )
        actual_print_hz = 1.0 / (velocity_print_every_n_steps * cfg.sim_config.dt)
        print(f"[sim2sim] Root velocity printing enabled at {actual_print_hz:.2f} Hz (body frame).")

    # ── Joint friction (MINI3_REAL_MOTOR_DAISY_CFG) ────────────────────────────
    for jname, (fl, damp) in JOINT_FRICTION.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            print(f"[sim2sim] WARNING: joint '{jname}' not found")
            continue
        dof = model.jnt_dofadr[jid]
        model.dof_frictionloss[dof] = fl
        model.dof_damping[dof] = damp
    print(
        "[sim2sim] joint friction applied "
        "(MINI3_REAL_MOTOR_DAISY_CFG: 55D fl=0.477/d=0.25, 25D fl=0.53/d=0.25, "
        "ankles fl=0.7/d=0.1, arms fl=0.35/d=0.25)"
    )

    # Set default joint positions and step once to propagate
    data.qpos[2] = getattr(cfg.robot_config, "init_base_height", 0.494)
    data.qpos[-cfg.robot_config.num_actions:] = cfg.robot_config.default_pos
    mujoco.mj_step(model, data)

    initial_qpos = data.qpos.copy()
    initial_qvel = data.qvel.copy()

    # ── Renderer / viewer ──────────────────────────────────────────────────────
    if headless:
        renderer = mujoco.Renderer(model, width=1920, height=1080)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        cam = mujoco.MjvCamera()
        cam.distance = 3.0
        cam.azimuth = 45.0
        cam.elevation = -20.0
        cam.lookat = [0.0, 0.0, 0.5]
        out = cv2.VideoWriter(
            'simulation_mini3_amp.mp4',
            fourcc,
            1.0 / (cfg.sim_config.dt * cfg.sim_config.decimation),
            (1920, 1080),
        )
    else:
        viewer = mujoco_viewer.MujocoViewer(model, data, mode='window', width=1920, height=1080)
        install_viewer_keyboard_controls(viewer)
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 45.0
        viewer.cam.elevation = -20.0
        viewer.cam.lookat = [0.0, 0.0, 0.5]
        print("Keyboard controls active while the MuJoCo window is focused.")

    # ── Buffers ────────────────────────────────────────────────────────────────
    target_pos = np.zeros(cfg.robot_config.num_actions, dtype=np.double)
    action = np.zeros(cfg.robot_config.num_actions, dtype=np.double)

    hist_obs = np.zeros(
        (cfg.robot_config.frame_stack, cfg.robot_config.num_observations),
        dtype=np.double,
    )

    tau = np.zeros(cfg.robot_config.num_actions, dtype=np.double)
    count_lowlevel = 0
    is_first_frame = True
    start_time = time.time()

    # ── Optional ROS2 state publisher ─────────────────────────────────────────
    ros2_publisher = None
    ros2_publish_every_n_steps = int(cfg.sim_config.ros2_publish_every_n_steps)
    if cfg.sim_config.ros2_enable:
        if ros2_publish_every_n_steps <= 0:
            raise ValueError(
                "ros2_publish_every_n_steps must be > 0, "
                f"got {ros2_publish_every_n_steps}."
            )
        mj_actuator_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
            for actuator_id in range(model.nu)
        ]
        if len(mj_actuator_names) != cfg.robot_config.num_actions or any(
            name is None for name in mj_actuator_names
        ):
            raise ValueError(
                "ROS2 publishing requires one named MuJoCo actuator per policy action; "
                f"got {len(mj_actuator_names)} actuators for "
                f"{cfg.robot_config.num_actions} actions."
            )
        ankle_joint_names = [
            mj_actuator_names[mj_i] for mj_i in cfg.robot_config.ankle_delay_mj_indices
        ]
        ros2_publisher = Ros2MujocoPublisher(
            joint_names=mj_actuator_names,
            ankle_torque_names=cfg.robot_config.ankle_torque_names,
            ankle_joint_names=ankle_joint_names,
            topic_prefix=cfg.sim_config.ros2_topic_prefix,
            node_name=cfg.sim_config.ros2_node_name,
            ros2_python=cfg.sim_config.ros2_python,
        )
        publish_rate = 1.0 / (cfg.sim_config.dt * ros2_publish_every_n_steps)
        prefix = ros2_publisher.topic_prefix
        print(f"[sim2sim] ROS2 publishing enabled: prefix={prefix}, rate={publish_rate:.1f} Hz")
        print(f"  {prefix}/joint_states  sensor_msgs/JointState")
        print(f"  {prefix}/joint_states_feedback  sensor_msgs/JointState")
        print(f"  {prefix}/target_joint_states  sensor_msgs/JointState")
        print(f"  {prefix}/nominal_tau  sensor_msgs/JointState")
        print(f"  {prefix}/ankle_nominal_tau  sensor_msgs/JointState")
        print(f"  {prefix}/ankle_nominal_tau_clipped  sensor_msgs/JointState")
        print(f"  {prefix}/ankle_motor_tau_nominal  sensor_msgs/JointState")
        print(f"  {prefix}/ankle_motor_tau_actual  sensor_msgs/JointState")
        print(f"  {prefix}/ankle_joint_tau_nominal  sensor_msgs/JointState")
        print(f"  {prefix}/ankle_joint_tau_actual  sensor_msgs/JointState")
        print(f"  {prefix}/imu  sensor_msgs/Imu")
        print(f"  {prefix}/projected_gravity  geometry_msgs/Vector3Stamped")

    total_steps = int(cfg.sim_config.sim_duration / cfg.sim_config.dt)
    for step in tqdm(range(total_steps), desc="Simulating..."):

        # ── Reset on request ───────────────────────────────────────────────────
        if cmd.reset_requested:
            print('Performing reset: restoring qpos/qvel and zeroing commands')
            data.qpos[:] = initial_qpos
            data.qvel[:] = initial_qvel
            cmd.reset()
            data.ctrl[:] = 0.0
            tau[:] = 0.0
            action[:] = 0.0
            target_pos[:] = cfg.robot_config.default_pos
            motor_rt.action_delay.reset()
            motor_rt.action_delay.set_joint_delay_steps(motor_rt.sample_action_delay_steps())
            if motor_rt.joint_torque_response is not None:
                motor_rt.joint_torque_response.reset()
            if motor_rt.ankle_motor_response is not None:
                motor_rt.ankle_motor_response.reset()
            hist_obs.fill(0.0)
            is_first_frame = True
            mujoco.mj_forward(model, data)
            cmd.reset_requested = False

        # ── Extract observation ────────────────────────────────────────────────
        q, dq, quat, v, omega, gvec = get_obs(data)
        q_joints = q[-cfg.robot_config.num_actions:]    # MuJoCo joint order
        dq_joints = dq[-cfg.robot_config.num_actions:]  # MuJoCo joint order

        if velocity_print_every_n_steps is not None and step % velocity_print_every_n_steps == 0:
            horizontal_speed = float(np.linalg.norm(v[:2]))
            tqdm.write(
                "[root velocity] "
                f"t={data.time:8.3f} s | "
                f"cmd=({float(cmd.vx):+.3f}, {float(cmd.vy):+.3f}, {float(cmd.dyaw):+.3f}) | "
                f"actual_b=({v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f}) m/s | "
                f"speed_xy={horizontal_speed:.3f} m/s | "
                f"wz={omega[2]:+.3f} rad/s"
            )

        # ── Low-frequency policy update ────────────────────────────────────────
        if count_lowlevel % cfg.sim_config.decimation == 0:

            # Remap MuJoCo order → lab order, subtract default
            q_rel = q_joints - cfg.robot_config.default_pos
            q_obs = np.zeros(cfg.robot_config.num_actions, dtype=np.double)
            dq_obs = np.zeros(cfg.robot_config.num_actions, dtype=np.double)
            for lab_i, mj_i in enumerate(cfg.robot_config.usd2urdf):
                q_obs[lab_i] = q_rel[mj_i]
                dq_obs[lab_i] = dq_joints[mj_i]

            obs = np.zeros([1, cfg.robot_config.num_observations], dtype=np.float32)
            obs[0, 0:3]  = omega           # base angular velocity in body frame
            obs[0, 3:6]  = gvec            # projected gravity in body frame
            obs[0, 6]    = cmd.vx          # velocity command x
            obs[0, 7]    = cmd.vy          # velocity command y
            obs[0, 8]    = cmd.dyaw        # angular velocity command z
            obs[0, 9:30] = q_obs           # joint pos relative to default (lab order)
            obs[0, 30:51] = dq_obs * 0.1  # joint vel scaled (lab order)
            obs[0, 51:72] = action         # last action (lab order)

            # print(
            #     "cmd: vx={:.2f}  vy={:.2f}  dyaw={:.2f}  |  "
            #     "actual: vx={:.2f}  vy={:.2f}  wz={:.2f}".format(
            #         cmd.vx, cmd.vy, cmd.dyaw, v[0], v[1], omega[2]
            #     )
            # )

            if is_first_frame:
                hist_obs = np.tile(obs, (cfg.robot_config.frame_stack, 1))
                is_first_frame = False
            else:
                hist_obs = np.concatenate((hist_obs[1:], obs.reshape(1, -1)), axis=0)

            policy_input = hist_obs.reshape(1, -1).astype(np.float32)
            with torch.inference_mode():
                action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()

            # Remap action from lab order → MuJoCo order, add default_pos
            target_q = action * cfg.robot_config.action_scale
            target_pos[:] = cfg.robot_config.default_pos.copy()
            for lab_i, mj_i in enumerate(cfg.robot_config.usd2urdf):
                target_pos[mj_i] += target_q[lab_i]

            # ── Render ─────────────────────────────────────────────────────────
            if headless:
                renderer.update_scene(data, camera=cam)
                if cmd.camera_follow:
                    cam.lookat = data.qpos[0:3].tolist()
                img = renderer.render()
                out.write(img)
            else:
                if cmd.camera_follow:
                    viewer.cam.lookat = data.qpos[0:3].tolist()
                viewer.render()

        # ── Real-motor torque (T-N clip + action delay + parallel ankle) ───────
        tau, applied_target_pos, torque_diagnostics = _compute_applied_torque(
            cfg, motor_rt, target_pos, q_joints, dq_joints, return_diagnostics=True,
        )
        if ros2_publisher is not None and step % ros2_publish_every_n_steps == 0:
            # AMP has no separate encoder-feedback corruption, so feedback
            # positions intentionally match the physical MuJoCo joint state.
            ros2_publisher.publish(
                q_joints,
                q_joints,
                dq_joints,
                tau,
                applied_target_pos,
                quat,
                omega,
                gvec,
                torque_diagnostics.nominal_tau,
                torque_diagnostics.ankle_motor_tau_nominal,
                torque_diagnostics.ankle_motor_tau_cmd,
                torque_diagnostics.ankle_motor_tau_actual,
                torque_diagnostics.ankle_joint_tau_nominal,
                torque_diagnostics.ankle_joint_tau_actual,
            )
        data.ctrl = tau
        mujoco.mj_step(model, data)
        count_lowlevel += 1

        # Real-time pacing
        elapsed = time.time() - start_time
        target_t = (step + 1) * cfg.sim_config.dt
        if elapsed < target_t:
            time.sleep(target_t - elapsed)

    if headless:
        out.release()
    else:
        viewer.close()
    if ros2_publisher is not None:
        ros2_publisher.close()
    print("Simulation finished.")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Mini3 AMP sim-to-sim deployment.')
    parser.add_argument(
        '--load_model', type=str, required=True,
        help='Path to exported TorchScript policy (.pt file).',
    )
    parser.add_argument(
        '--headless', action='store_true',
        help='Render offscreen and save to simulation_mini3_amp.mp4.',
    )
    parser.add_argument(
        '--ros2', action='store_true',
        help='Publish MuJoCo joint state, torque, IMU, and projected gravity on ROS2 topics.',
    )
    parser.add_argument(
        '--ros2_node_name', type=str, default='mini3_amp_mujoco_state_publisher',
        help='ROS2 node name used when --ros2 is enabled.',
    )
    parser.add_argument(
        '--ros2_topic_prefix', type=str, default='/mini3/mujoco',
        help='ROS2 topic prefix used when --ros2 is enabled.',
    )
    parser.add_argument(
        '--ros2_python', type=str, default='/usr/bin/python3',
        help=(
            'Python executable for the fallback ROS2 bridge process. '
            'For ROS Humble on Ubuntu 22.04 this should usually be /usr/bin/python3.'
        ),
    )
    parser.add_argument(
        '--ros2_publish_every_n_steps', type=int, default=10,
        help=(
            'Publish once every N MuJoCo physics steps when --ros2 is enabled. '
            'With dt=0.002, N=10 publishes at 50 Hz.'
        ),
    )
    parser.add_argument(
        '--velocity_print_hz', type=float, default=2.0,
        help=(
            'Terminal print rate for env root velocity in the body frame. '
            'Set to 0 to disable. Defaults to 2 Hz.'
        ),
    )
    parser.add_argument(
        '--cmd_vx', type=float, default=0.0,
        help='Initial forward velocity command in m/s. Defaults to 0.0.',
    )
    parser.add_argument(
        '--cmd_vy', type=float, default=0.0,
        help='Initial lateral velocity command in m/s. Defaults to 0.0.',
    )
    parser.add_argument(
        '--cmd_wz', type=float, default=0.0,
        help='Initial yaw velocity command in rad/s. Defaults to 0.0.',
    )
    args = parser.parse_args()
    if args.ros2_publish_every_n_steps <= 0:
        parser.error('--ros2_publish_every_n_steps must be > 0')
    if not np.isfinite(args.velocity_print_hz) or args.velocity_print_hz < 0.0:
        parser.error('--velocity_print_hz must be a finite value >= 0')
    initial_commands = (
        ('--cmd_vx', args.cmd_vx, cmd.min_vx, cmd.max_vx),
        ('--cmd_vy', args.cmd_vy, cmd.min_vy, cmd.max_vy),
        ('--cmd_wz', args.cmd_wz, cmd.min_dyaw, cmd.max_dyaw),
    )
    for name, value, lower, upper in initial_commands:
        if not np.isfinite(value) or not lower <= value <= upper:
            parser.error(f'{name} must be finite and within [{lower}, {upper}]')

    class Sim2simCfg:

        class sim_config:
            # Daisy MJCF (same asset tree as Mini3-Walk-AMP training URDF)
            mujoco_model_path = f'{ISAAC_DATA_DIR}/robots/roboparty/mini3_daisy_shell/mjcf/scene.xml'

            # Match IsaacLab AMP training config:
            #   sim.dt = 0.002 s,  decimation = 10  →  control @ 50 Hz
            sim_duration = 1000000.0
            dt = 0.002
            decimation = 10
            seed = 42

            # Optional ROS2 publishing, matching sim2sim_mini3_bm.py. If this
            # Python environment cannot import rclpy, use ros2_python as a
            # newline-delimited JSON bridge process.
            ros2_enable = False
            ros2_node_name = 'mini3_amp_mujoco_state_publisher'
            ros2_topic_prefix = '/mini3/mujoco'
            ros2_python = '/usr/bin/python3'
            ros2_publish_every_n_steps = 10
            velocity_print_hz = 2.0

            # MINI3_REAL_MOTOR_DAISY_CFG actuator contract (matches training):
            #   T-N clip on; torque-response / KT off; all groups delay min/max=3/5.
            action_delay_enable = True
            pace_action_delay_range_steps = (3, 5)   # 55D/25D legs+waist
            ankle_action_delay_range_steps = (3, 5)  # parallel ankles
            arm_action_delay_range_steps = (3, 5)    # 10D arms
            torque_response_enable = False
            torque_response_kp = 0.0
            torque_response_ki = 90.6769527429
            torque_response_plant_tau_s = 0.00393417593548
            torque_response_delay_steps = 1.0
            tn_torque_limit_enable = True
            tn_limit_after_response = True
            kt_output_model_enable = False
            apply_clipped_ankle_motor_torque = True
            ankle_motor_tau_limit = 25.0             # parallel ankle 25D peak

        class robot_config:
            init_base_height = 0.494

            # ── Joint ordering ─────────────────────────────────────────────────
            # MuJoCo order (from mini3_daisy_shell/mjcf/mini3.xml <actuator> list):
            #   0:L_hip_pitch   1:L_hip_roll   2:L_hip_yaw
            #   3:L_knee_pitch  4:L_ankle_pitch 5:L_ankle_roll
            #   6:R_hip_pitch   7:R_hip_roll   8:R_hip_yaw
            #   9:R_knee_pitch 10:R_ankle_pitch 11:R_ankle_roll
            #  12:waist_yaw
            #  13:L_shoulder_pitch 14:L_shoulder_roll 15:L_shoulder_yaw 16:L_elbow_pitch
            #  17:R_shoulder_pitch 18:R_shoulder_roll 19:R_shoulder_yaw 20:R_elbow_pitch
            #
            # IsaacLab lab order (from scripts/tools/retarget/config/mini3.yaml):
            #   0:L_hip_pitch   1:R_hip_pitch   2:waist_yaw
            #   3:L_hip_roll    4:R_hip_roll
            #   5:L_shoulder_pitch  6:R_shoulder_pitch
            #   7:L_hip_yaw     8:R_hip_yaw
            #   9:L_shoulder_roll  10:R_shoulder_roll
            #  11:L_knee_pitch  12:R_knee_pitch
            #  13:L_shoulder_yaw  14:R_shoulder_yaw
            #  15:L_ankle_pitch  16:R_ankle_pitch
            #  17:L_elbow_pitch  18:R_elbow_pitch
            #  19:L_ankle_roll   20:R_ankle_roll
            #
            # usd2urdf[lab_idx] = mujoco_idx
            usd2urdf = [0, 6, 12, 1, 7, 13, 17, 2, 8, 14, 18, 3, 9, 15, 19, 4, 10, 16, 20, 5, 11]
            pace_delay_mj_indices = (0, 1, 2, 3, 6, 7, 8, 9, 12)
            ankle_delay_mj_indices = (4, 5, 10, 11)
            arm_delay_mj_indices = (13, 14, 15, 16, 17, 18, 19, 20)
            non_ankle_mj_indices = (0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20)
            left_ankle_motor_indices = (4, 5)
            right_ankle_motor_indices = (10, 11)

            # T-N specs — must match real_motor_actuator._MOTOR_SPECS (55d/25d/10d).
            # No peak_speed_rpm -> T-N corner falls back to rated_speed_rpm=120.
            motor_specs = {
                "55D": {
                    "rated_torque": 55.0,
                    "peak_torque": 55.1,
                    "rated_speed_rpm": 120.0,
                    "no_load_speed_rpm": 174.0,
                },
                "25D": {
                    "rated_torque": 25.0,
                    "peak_torque": 25.1,
                    "rated_speed_rpm": 120.0,
                    "no_load_speed_rpm": 174.0,
                },
                "10D": {
                    "rated_torque": 10.0,
                    "peak_torque": 10.1,
                    "rated_speed_rpm": 120.0,
                    "no_load_speed_rpm": 235.0,
                },
            }
            motor_type_by_mj_index = (
                "55D", "25D", "25D", "55D", "25D", "25D",
                "55D", "25D", "25D", "55D", "25D", "25D",
                "25D",
                "10D", "10D", "10D", "10D",
                "10D", "10D", "10D", "10D",
            )
            ankle_motor_type = "25D"
            kt_output_model_tables = {
                "25D": {
                    "feedback_tau_nm": (1.2, 2.35, 4.7, 7.12, 9.9, 13.5),
                    "actual_tau_nm": (1.2, 2.35, 4.7, 7.12, 9.9, 13.5),
                },
                "10D": {
                    "feedback_tau_nm": (1.2, 2.35, 4.7, 7.12, 9.9, 13.5),
                    "actual_tau_nm": (1.2, 2.35, 4.7, 7.12, 9.9, 13.5),
                },
                "55D": {
                    "feedback_tau_nm": (
                        1.558, 3.158, 5.477, 8.324, 10.55, 13.121,
                        15.733, 18.509, 21.34, 24.786, 27.576,
                    ),
                    "actual_tau_nm": (
                        1.558, 3.158, 5.477, 8.324, 10.55, 13.121,
                        15.733, 18.509, 21.34, 24.786, 27.576,
                    ),
                },
            }
            ankle_equivalent_joint_kd = {4: 1.2, 5: 1.2, 10: 1.2, 11: 1.2}
            ankle_torque_names = (
                "left_ankle_tMR_0x35",
                "left_ankle_tML_0x36",
                "right_ankle_tML_0x45",
                "right_ankle_tMR_0x46",
            )
            ankle_params = {
                # Daisy kinematics (MINI3_REAL_MOTOR_DAISY_CFG).
                "l": 26.0,
                "lm": 30.05,
                "hl": 89.451,
                "hr": 148.271,
                "z0": 0.0,
                "d": 22.0,
                "df": 14.0,
                "zl": 89.0,
                "zr": 148.0,
            }

            # ── PD gains (in MuJoCo order) — MINI3_REAL_MOTOR_DAISY_CFG ────────
            kps = np.array([
                60, 45, 45, 60, 50, 45,   # L leg  (pitch, roll, yaw, knee, ankle_pitch, ankle_roll)
                60, 45, 45, 60, 50, 45,   # R leg
                45,                       # waist
                30, 30, 30, 30,           # L arm  (pitch, roll, yaw, elbow)
                30, 30, 30, 30,           # R arm
            ], dtype=np.double)

            kds = np.array([
                4.5, 4.0, 4.0, 4.5, 1.2, 1.2,   # L leg
                4.5, 4.0, 4.0, 4.5, 1.2, 1.2,   # R leg
                4.0,                              # waist
                2.0, 2.0, 2.0, 2.0,              # L arm
                2.0, 2.0, 2.0, 2.0,              # R arm
            ], dtype=np.double)

            # ── Torque limits (datasheet peaks / effort_limit) ────────────────
            tau_limit = np.array([
                55, 25, 25, 55, 50, 50,   # L leg
                55, 25, 25, 55, 50, 50,   # R leg
                25,                        # waist
                10, 10, 10, 10,           # L arm
                10, 10, 10, 10,           # R arm
            ], dtype=np.double)

            # ── Default joint positions — roboparty._WALK_DEFAULT_POS ─────────
            default_pos = np.array([
                -0.20,  0.00,  0.00,  0.40, -0.20,  0.00,  # L leg
                -0.20,  0.00,  0.00,  0.40, -0.20,  0.00,  # R leg
                 0.00,                                       # waist
                 0.00,  0.25,  0.00,  1.00,                 # L arm
                 0.00, -0.25,  0.00,  1.00,                 # R arm
            ], dtype=np.double)

            # ── Policy dimensions ──────────────────────────────────────────────
            num_actions = 21
            action_scale = 0.25   # matches JointPositionActionCfg scale in env cfg

            # Per-frame obs: 3(ang_vel) + 3(gvec) + 3(cmd) + 21(q) + 21(dq) + 21(act) = 72
            # history_length = 1 (PolicyCfg default)  →  policy input = 72
            num_observations = 72
            frame_stack = 1

    cfg = Sim2simCfg()
    cfg.sim_config.ros2_enable = args.ros2
    cfg.sim_config.ros2_node_name = args.ros2_node_name
    cfg.sim_config.ros2_topic_prefix = args.ros2_topic_prefix
    cfg.sim_config.ros2_python = args.ros2_python
    cfg.sim_config.ros2_publish_every_n_steps = args.ros2_publish_every_n_steps
    cfg.sim_config.velocity_print_hz = args.velocity_print_hz

    # Apply the requested command before the first observation/policy update.
    cmd.vx = float(args.cmd_vx)
    cmd.vy = float(args.cmd_vy)
    cmd.dyaw = float(args.cmd_wz)

    policy = torch.jit.load(args.load_model)
    policy.eval()
    run_mujoco(policy, cfg, args.headless)
