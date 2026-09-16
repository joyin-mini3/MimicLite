"""Build reference motions for the existing Mini3 tracking policy.

These helpers edit reference arrays only. They never move the simulated robot,
reset policy histories, or attach the floating base to the world.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parent
MINI3_XML = ROOT / "any4hdmi/assets/robots/mini3_mjlab/mini3.xml"


def quaternion_yaw(quat: np.ndarray) -> float:
    w, x, y, z = np.asarray(quat, dtype=np.float64)
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def place_qpos(
    qpos: np.ndarray,
    *,
    origin_xy: tuple[float, float] | np.ndarray = (0.0, 0.0),
    yaw: float = 0.0,
    align_travel: bool = False,
) -> np.ndarray:
    """Rigidly place a canonical 28-coordinate clip in the world.

    ``yaw`` sets the initial base heading, or the net travel heading when
    ``align_travel`` is true. Angles are radians; root height is preserved.
    """
    result = np.asarray(qpos, dtype=np.float64).copy()
    if result.ndim != 2 or result.shape[1] != 28 or not len(result):
        raise ValueError("Expected nonempty canonical Mini3 qpos with shape (N, 28)")
    delta = result[-1, :2] - result[0, :2]
    original_yaw = (
        float(np.arctan2(delta[1], delta[0]))
        if align_travel and np.linalg.norm(delta) > 1e-8
        else quaternion_yaw(result[0, 3:7])
    )
    angle = float(yaw) - original_yaw
    c, s = np.cos(angle), np.sin(angle)
    result[:, :2] = (result[:, :2] - result[0, :2]) @ np.array(((c, s), (-s, c)))
    result[:, :2] += np.asarray(origin_xy)
    w, x, y, z = result[:, 3:7].copy().T
    half_c, half_s = np.cos(angle / 2), np.sin(angle / 2)
    result[:, 3:7] = np.column_stack(
        (half_c * w - half_s * z, half_c * x - half_s * y,
         half_c * y + half_s * x, half_c * z + half_s * w)
    )
    return result


def blend_poses(start: np.ndarray, end: np.ndarray, duration: float, dt: float = 0.02) -> np.ndarray:
    """Interpolate root position and joints with zero endpoint slope, plus SLERP."""
    start, end = np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64)
    if start.shape != (28,) or end.shape != (28,) or duration <= 0 or dt <= 0:
        raise ValueError("Expected two 28-coordinate poses and positive duration/dt")
    alpha = np.linspace(0, 1, max(2, int(round(duration / dt)) + 1))
    alpha = alpha * alpha * (3 - 2 * alpha)
    result = start[None, :] + alpha[:, None] * (end - start)[None, :]
    q0, q1 = start[3:7].copy(), end[3:7].copy()
    q0 /= np.linalg.norm(q0)
    q1 /= np.linalg.norm(q1)
    if np.dot(q0, q1) < 0:
        q1 *= -1
    dot = np.clip(np.dot(q0, q1), -1, 1)
    if dot > 0.9995:
        quats = q0 + alpha[:, None] * (q1 - q0)
    else:
        angle = np.arccos(dot)
        quats = (np.sin((1 - alpha) * angle)[:, None] * q0
                 + np.sin(alpha * angle)[:, None] * q1) / np.sin(angle)
    result[:, 3:7] = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    return result


def shorten_walk(qpos: np.ndarray, distance: float) -> tuple[np.ndarray, dict[str, float | int]]:
    """Remove a matched middle gait segment while retaining natural start/stop.

    Designed for a forward clip containing a steady middle walk and standing
    endpoints. Requested distance is reference displacement, not guaranteed
    physical distance: the policy and payload change the final tracking error.
    """
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != 28 or len(qpos) < 100:
        raise ValueError("Walk shortening needs at least 100 canonical Mini3 frames")
    travel = qpos[-1, :2] - qpos[0, :2]
    full_distance = float(np.linalg.norm(travel))
    if distance <= 0 or distance > full_distance:
        raise ValueError(f"Requested reference distance must be in (0, {full_distance:.3f}]")
    if full_distance - distance < 0.01:
        return qpos.copy(), {"distance": full_distance, "removed_frames": 0}
    direction = travel / full_distance
    best = None
    for first in range(int(len(qpos) * 0.14), int(len(qpos) * 0.4)):
        last = np.arange(first + 20, int(len(qpos) * 0.79))
        remaining = full_distance - (qpos[last, :2] - qpos[first, :2]) @ direction
        leg_error = np.linalg.norm(qpos[last, 7:19] - qpos[first, 7:19], axis=1)
        score = leg_error + 5 * np.abs(remaining - distance)
        k = int(np.argmin(score))
        candidate = (float(score[k]), first, int(last[k]), float(leg_error[k]))
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    _, first, last, leg_error = best
    tail = qpos[last + 1:].copy()
    tail[:, :2] -= qpos[last, :2] - qpos[first, :2]
    # Fade the small pose mismatch over 0.2 s without pausing forward motion.
    mismatch = qpos[first, 2:] - qpos[last, 2:]
    count = min(10, len(tail))
    weight = np.linspace(1, 0, count + 1)[1:]
    tail[:count, 2:] += weight[:, None] * mismatch[None, :]
    tail[:, 3:7] /= np.linalg.norm(tail[:, 3:7], axis=1, keepdims=True)
    result = np.concatenate((qpos[:first + 1], tail))
    return result, {
        "distance": float(np.linalg.norm(result[-1, :2] - result[0, :2])),
        "removed_frames": last - first,
        "first_frame": first,
        "last_frame": last,
        "leg_pose_mismatch": leg_error,
    }


class MotionReference:
    """In-memory MotionDataset adapter made with the original Mini3 FK model."""

    def __init__(
        self,
        qpos: np.ndarray,
        *,
        dt: float = 0.02,
        model: mujoco.MjModel | None = None,
    ):
        from any4hdmi.utils.dataset import compute_motion_qvel

        self.qpos = np.asarray(qpos, dtype=np.float64).copy()
        self.dt = float(dt)
        model = model if model is not None else mujoco.MjModel.from_xml_path(str(MINI3_XML))
        if self.qpos.ndim != 2 or self.qpos.shape != (len(self.qpos), model.nq) or not len(self.qpos):
            raise ValueError(f"Expected nonempty qpos with width {model.nq}")
        if dt <= 0 or not np.isfinite(self.qpos).all():
            raise ValueError("Reference must be finite with a positive timestep")
        data = mujoco.MjData(model)
        scalar_ids = np.flatnonzero(np.isin(model.jnt_type, (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)))
        self.joint_names = [model.joint(int(i)).name for i in scalar_ids]
        self.body_names = [model.body(i).name for i in range(model.nbody)]
        self.num_motions = 1
        self.num_steps = len(self.qpos)
        self.starts = np.array([0], dtype=np.int64)
        self.ends = np.array([self.num_steps], dtype=np.int64)
        self.lengths = self.ends.copy()
        velocities = compute_motion_qvel(model, self.qpos, 1 / self.dt)
        self._storage = {
            "motion_id": np.zeros(self.num_steps, dtype=np.int64),
            "step": np.arange(self.num_steps, dtype=np.int64),
            "body_pos_w": np.empty((self.num_steps, model.nbody, 3), dtype=np.float32),
            "body_quat_w": np.empty((self.num_steps, model.nbody, 4), dtype=np.float32),
            "body_lin_vel_w": np.empty((self.num_steps, model.nbody, 3), dtype=np.float32),
            "body_ang_vel_w": np.empty((self.num_steps, model.nbody, 3), dtype=np.float32),
            "joint_pos": self.qpos[:, model.jnt_qposadr[scalar_ids]].astype(np.float32),
            "joint_vel": velocities[:, model.jnt_dofadr[scalar_ids]].astype(np.float32),
        }
        for frame, (pose, velocity) in enumerate(zip(self.qpos, velocities, strict=True)):
            data.qpos[:] = pose
            data.qvel[:] = velocity
            mujoco.mj_forward(model, data)
            self._storage["body_pos_w"][frame] = data.xpos
            self._storage["body_quat_w"][frame] = data.xquat
            # Match the training any4hdmi FK convention, including cvel origin.
            self._storage["body_lin_vel_w"][frame] = data.cvel[:, 3:]
            self._storage["body_ang_vel_w"][frame] = data.cvel[:, :3]

    def get_slice(self, motion_ids: np.ndarray, starts: np.ndarray, steps: np.ndarray) -> Any:
        from sim2real.rl_policy.utils.motion import MotionData

        motion_ids = np.asarray(motion_ids).reshape(-1)
        starts = np.asarray(starts, dtype=np.int64).reshape(-1)
        if np.any(motion_ids != 0) or starts.size != motion_ids.size:
            raise ValueError("MotionReference supports motion ID zero with one start per ID")
        indices = np.clip(starts[:, None] + np.asarray(steps, dtype=np.int64).reshape(1, -1), 0, self.num_steps - 1)
        return MotionData(**{name: value[indices] for name, value in self._storage.items()})


def install_reference(policy: Any, reference: MotionReference, *, paused: bool = False) -> None:
    """Switch reference at the next policy tick without resetting robot/history."""
    state = policy.state_processor
    if list(reference.joint_names) != list(state.joint_names):
        raise ValueError("Reference joint order differs from the policy's canonical joints")
    state.motion_dataset = reference
    state.motion_joint_names = list(reference.joint_names)
    state.motion_body_names = list(reference.body_names)
    state.motion_length = reference.num_steps
    state.motion_ids[:] = 0
    state.motion_t[:] = 0 if paused else -1
    state._update_motion_data()
    policy.state_dict["paused"] = bool(paused)
