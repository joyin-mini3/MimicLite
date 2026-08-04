"""Retarget sparse PICO clips to qpos-only Mini3 any4hdmi motions."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from tqdm import tqdm

from any4hdmi.core.format import MOTIONS_SUBDIR, load_motion, repo_root, write_manifest
from any4hdmi.scripts.preprocess.mini3_pkl import (
    DEFAULT_MJCF,
    MINI3_JOINT_NAMES,
    Mini3Motion,
    motion_to_qpos,
    validate_mini3_model,
)


DEFAULT_OUTPUT_ROOT = repo_root() / "output" / "mini3" / "pico"
PICO_BODY_NAMES = (
    "pelvis",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)
REQUIRED_FIELDS = (
    "body_pos_w",
    "body_quat_w",
    "body_names",
    "fps",
    "dt",
    "source",
    "pico_position_axes_version",
    "body_state_frame",
)


@dataclass(frozen=True)
class PicoMotionClip:
    path: Path
    body_pos_w: np.ndarray
    body_quat_wxyz: np.ndarray
    root_quat_wxyz: np.ndarray
    fps: float
    position_axes_version: int
    body_state_frame: str
    root_orientation_source: str


@dataclass(frozen=True)
class TrackingTarget:
    source_name: str
    target_body_name: str
    target_body_id: int
    local_point: np.ndarray
    position_weight: float
    orientation_scale: float


@dataclass(frozen=True)
class RetargetResult:
    motion: Mini3Motion
    scale: float
    mean_position_error_m: dict[str, float]
    max_position_error_m: dict[str, float]
    mean_orientation_error_rad: dict[str, float]
    max_orientation_error_rad: dict[str, float]
    mean_iterations: float
    max_iterations: int
    target_mapping: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class ConversionSummary:
    selected: int
    converted: int
    skipped: int
    output_motions: int
    output_frames: int
    output_root: Path
    manifest_path: Path
    displayed_motion: Path | None


def _scalar(value: Any, key: str, path: Path) -> Any:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"PICO clip={path} field={key} must be scalar, got {array.shape}")
    return array.reshape(-1)[0]


def _load_payload(path: Path) -> dict[str, np.ndarray]:
    if path.is_dir():
        return {
            field_path.stem: np.load(field_path, allow_pickle=False)
            for field_path in sorted(path.glob("*.npy"))
        }
    if path.suffix.lower() != ".npz" or not path.is_file():
        raise FileNotFoundError(f"Expected a PICO .npz or unpacked clip directory: {path}")
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _normalize_quaternions_wxyz(values: Any, *, field: str, path: Path) -> np.ndarray:
    quaternions = np.asarray(values, dtype=np.float64)
    if quaternions.ndim < 2 or quaternions.shape[-1] != 4:
        raise ValueError(f"PICO clip={path} field={field} must end in 4, got {quaternions.shape}")
    if not np.all(np.isfinite(quaternions)):
        raise ValueError(f"PICO clip={path} field={field} contains non-finite values")
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if np.any(norms <= 1.0e-8):
        raise ValueError(f"PICO clip={path} field={field} contains a zero quaternion")
    return quaternions / norms


def _continuous_quaternions(quaternions: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternions, dtype=np.float64).copy()
    result /= np.linalg.norm(result, axis=-1, keepdims=True)
    for frame_idx in range(1, len(result)):
        if float(np.dot(result[frame_idx - 1], result[frame_idx])) < 0.0:
            result[frame_idx] *= -1.0
    return result


def _rotation_6d_columns_to_quaternions_wxyz(values: Any, path: Path) -> np.ndarray:
    rotations = np.asarray(values, dtype=np.float64)
    if rotations.ndim != 2 or rotations.shape[1] != 6:
        raise ValueError(
            f"PICO clip={path} sonic_smpl_anchor_orientation must have shape [T,6], "
            f"got {rotations.shape}"
        )
    quaternions = np.empty((len(rotations), 4), dtype=np.float64)
    for frame_idx, rotation in enumerate(rotations):
        columns = rotation.reshape(3, 2)
        first = columns[:, 0]
        first_norm = float(np.linalg.norm(first))
        if first_norm <= 1.0e-8:
            raise ValueError(f"PICO clip={path} has degenerate root 6D frame={frame_idx}")
        first = first / first_norm
        second = columns[:, 1] - float(np.dot(first, columns[:, 1])) * first
        second_norm = float(np.linalg.norm(second))
        if second_norm <= 1.0e-8:
            raise ValueError(f"PICO clip={path} has collinear root 6D frame={frame_idx}")
        second = second / second_norm
        matrix = np.stack((first, second, np.cross(first, second)), axis=-1)
        mujoco.mju_mat2Quat(quaternions[frame_idx], matrix.reshape(-1))
    return _continuous_quaternions(quaternions)


def load_pico_clip(path: str | Path) -> PicoMotionClip:
    clip_path = Path(path).expanduser().resolve()
    payload = _load_payload(clip_path)
    missing = sorted(set(REQUIRED_FIELDS).difference(payload))
    if missing:
        raise ValueError(f"PICO clip={clip_path} is missing fields: {missing}")

    source = str(_scalar(payload["source"], "source", clip_path))
    if source != "pico_motion_clip":
        raise ValueError(
            f"PICO clip={clip_path} has source={source!r}, expected 'pico_motion_clip'"
        )
    axes_version = int(
        _scalar(payload["pico_position_axes_version"], "pico_position_axes_version", clip_path)
    )
    if axes_version != 3:
        raise ValueError(
            f"PICO clip={clip_path} has unsupported pico_position_axes_version="
            f"{axes_version}; expected 3"
        )
    names = tuple(str(value) for value in np.asarray(payload["body_names"]).tolist())
    if names != PICO_BODY_NAMES:
        raise ValueError(
            f"PICO clip={clip_path} body_names must be {list(PICO_BODY_NAMES)}, "
            f"got {list(names)}"
        )

    positions = np.asarray(payload["body_pos_w"], dtype=np.float64)
    quaternions = _normalize_quaternions_wxyz(
        payload["body_quat_w"], field="body_quat_w", path=clip_path
    )
    if positions.ndim != 3 or positions.shape[1:] != (len(PICO_BODY_NAMES), 3):
        raise ValueError(
            f"PICO clip={clip_path} body_pos_w must have shape [T,5,3], got {positions.shape}"
        )
    if quaternions.shape != (positions.shape[0], len(PICO_BODY_NAMES), 4):
        raise ValueError(
            f"PICO clip={clip_path} body_quat_w must have shape "
            f"[{positions.shape[0]},5,4], got {quaternions.shape}"
        )
    if positions.shape[0] < 2 or not np.all(np.isfinite(positions)):
        raise ValueError(f"PICO clip={clip_path} must contain at least two finite frames")

    fps = float(_scalar(payload["fps"], "fps", clip_path))
    dt = float(_scalar(payload["dt"], "dt", clip_path))
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"PICO clip={clip_path} fps must be positive, got {fps}")
    if not np.isfinite(dt) or dt <= 0.0 or not np.isclose(
        dt, 1.0 / fps, rtol=1.0e-5, atol=1.0e-8
    ):
        raise ValueError(f"PICO clip={clip_path} dt={dt} does not match 1/fps={1.0 / fps}")

    if "sonic_smpl_anchor_orientation" in payload:
        root_quaternions = _rotation_6d_columns_to_quaternions_wxyz(
            payload["sonic_smpl_anchor_orientation"], clip_path
        )
        root_orientation_source = "sonic_smpl_anchor_orientation"
    else:
        source_root = quaternions[:, 0]
        alignment = _quat_conjugate_wxyz(source_root[0])
        root_quaternions = np.stack(
            [_quat_multiply_wxyz(alignment, value) for value in source_root]
        )
        root_quaternions = _continuous_quaternions(root_quaternions)
        root_orientation_source = "relative_pelvis_body_quat"
    if root_quaternions.shape != (positions.shape[0], 4):
        raise ValueError(f"PICO clip={clip_path} root orientation frame count mismatch")

    return PicoMotionClip(
        path=clip_path,
        body_pos_w=positions,
        body_quat_wxyz=quaternions,
        root_quat_wxyz=root_quaternions,
        fps=fps,
        position_axes_version=axes_version,
        body_state_frame=str(_scalar(payload["body_state_frame"], "body_state_frame", clip_path)),
        root_orientation_source=root_orientation_source,
    )


def _object_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"Mini3 model is missing {object_type.name} {name!r}")
    return int(object_id)


def _mini3_hand_point(model: mujoco.MjModel, side: str) -> tuple[str, np.ndarray]:
    body_name = f"{side}_elbow_pitch_link"
    geom_name = f"{side}_elbow_pitch_link_collision"
    body_id = _object_id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    geom_id = _object_id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if int(model.geom_bodyid[geom_id]) != body_id:
        raise ValueError(f"Mini3 hand fallback geom {geom_name!r} is not on {body_name!r}")
    return body_name, np.asarray(model.geom_pos[geom_id], dtype=np.float64).copy()


def _tracking_targets(model: mujoco.MjModel) -> list[TrackingTarget]:
    targets = [
        TrackingTarget(
            source_name=name,
            target_body_name=name,
            target_body_id=_object_id(model, mujoco.mjtObj.mjOBJ_BODY, name),
            local_point=np.zeros(3, dtype=np.float64),
            position_weight=2.0,
            orientation_scale=1.0,
        )
        for name in PICO_BODY_NAMES[1:3]
    ]
    for side, source_name in (("left", PICO_BODY_NAMES[3]), ("right", PICO_BODY_NAMES[4])):
        body_name, local_point = _mini3_hand_point(model, side)
        targets.append(
            TrackingTarget(
                source_name=source_name,
                target_body_name=body_name,
                target_body_id=_object_id(model, mujoco.mjtObj.mjOBJ_BODY, body_name),
                local_point=local_point,
                position_weight=1.0,
                orientation_scale=0.0,
            )
        )
    return targets


def _quat_conjugate_wxyz(quaternion: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternion, dtype=np.float64).copy()
    result[1:] *= -1.0
    return result


def _quat_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.asarray(left, dtype=np.float64)
    rw, rx, ry, rz = np.asarray(right, dtype=np.float64)
    return np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def _quat_error_vector_wxyz(desired: np.ndarray, actual: np.ndarray) -> np.ndarray:
    delta = _quat_multiply_wxyz(desired, _quat_conjugate_wxyz(actual))
    if delta[0] < 0.0:
        delta *= -1.0
    vector_norm = float(np.linalg.norm(delta[1:]))
    if vector_norm <= 1.0e-10:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * np.arctan2(vector_norm, float(np.clip(delta[0], -1.0, 1.0)))
    return delta[1:] * (angle / vector_norm)


def _world_point(data: mujoco.MjData, target: TrackingTarget) -> np.ndarray:
    rotation = data.xmat[target.target_body_id].reshape(3, 3)
    return data.xpos[target.target_body_id] + rotation @ target.local_point


def _clip_control_joints(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    joint_ids: np.ndarray,
    qpos_addresses: np.ndarray,
) -> None:
    for joint_id, qpos_address in zip(joint_ids, qpos_addresses):
        if not model.jnt_limited[joint_id]:
            continue
        lower, upper = (float(value) for value in model.jnt_range[joint_id])
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(joint_id)) or ""
        if "knee" in joint_name:
            lower = max(lower, 0.02)
        qpos[qpos_address] = np.clip(qpos[qpos_address], lower, upper)


def _weighted_residual(
    data: mujoco.MjData,
    targets: list[TrackingTarget],
    desired_positions: np.ndarray,
    desired_orientations: np.ndarray,
    orientation_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    residual: list[np.ndarray] = []
    position_errors: list[float] = []
    orientation_errors: list[float] = []
    for target_idx, target in enumerate(targets):
        position_error = desired_positions[target_idx] - _world_point(data, target)
        orientation_error = _quat_error_vector_wxyz(
            desired_orientations[target_idx], data.xquat[target.target_body_id]
        )
        residual.append(np.sqrt(target.position_weight) * position_error)
        weight = orientation_weight * target.orientation_scale
        if weight > 0.0:
            residual.append(np.sqrt(weight) * orientation_error)
        position_errors.append(float(np.linalg.norm(position_error)))
        orientation_errors.append(float(np.linalg.norm(orientation_error)))
    return np.concatenate(residual), np.asarray(position_errors), np.asarray(orientation_errors)


def _solve_ik_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    targets: list[TrackingTarget],
    desired_positions: np.ndarray,
    desired_orientations: np.ndarray,
    joint_ids: np.ndarray,
    qpos_addresses: np.ndarray,
    dof_addresses: np.ndarray,
    reference_dof_pos: np.ndarray,
    *,
    max_iterations: int,
    position_tolerance: float,
    orientation_weight: float,
    damping: float,
    posture_weight: float,
    max_joint_step: float,
    max_frame_joint_delta: float | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    identity = np.eye(len(qpos_addresses), dtype=np.float64)
    previous_cost = float("inf")
    stagnation_steps = 0
    iterations_used = 0
    for iteration in range(max_iterations):
        iterations_used = iteration + 1
        mujoco.mj_forward(model, data)
        residual, position_errors, orientation_errors = _weighted_residual(
            data, targets, desired_positions, desired_orientations, orientation_weight
        )
        posture_delta = reference_dof_pos - data.qpos[qpos_addresses]
        cost = float(residual @ residual + posture_weight * (posture_delta @ posture_delta))
        if float(np.max(position_errors)) <= position_tolerance:
            return position_errors, orientation_errors, iterations_used
        improvement = previous_cost - cost
        if np.isfinite(previous_cost) and improvement <= max(1.0e-12, previous_cost * 1.0e-7):
            stagnation_steps += 1
        else:
            stagnation_steps = 0
        if stagnation_steps >= 3:
            return position_errors, orientation_errors, iterations_used
        previous_cost = cost

        jacobian_rows: list[np.ndarray] = []
        for target in targets:
            jacobian_pos = np.zeros((3, model.nv), dtype=np.float64)
            jacobian_rot = np.zeros((3, model.nv), dtype=np.float64)
            mujoco.mj_jac(
                model,
                data,
                jacobian_pos,
                jacobian_rot,
                _world_point(data, target),
                target.target_body_id,
            )
            jacobian_rows.append(
                np.sqrt(target.position_weight) * jacobian_pos[:, dof_addresses]
            )
            weight = orientation_weight * target.orientation_scale
            if weight > 0.0:
                jacobian_rows.append(np.sqrt(weight) * jacobian_rot[:, dof_addresses])
        jacobian = np.vstack(jacobian_rows)
        lhs = jacobian.T @ jacobian + (damping + posture_weight) * identity
        rhs = jacobian.T @ residual + posture_weight * posture_delta
        try:
            joint_step = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            joint_step = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        joint_step = np.clip(joint_step, -max_joint_step, max_joint_step)

        current_dof_pos = data.qpos[qpos_addresses].copy()
        accepted = False
        for step_scale in (1.0, 0.5, 0.25, 0.125):
            data.qpos[qpos_addresses] = current_dof_pos + step_scale * joint_step
            _clip_control_joints(model, data.qpos, joint_ids, qpos_addresses)
            if max_frame_joint_delta is not None:
                data.qpos[qpos_addresses] = np.clip(
                    data.qpos[qpos_addresses],
                    reference_dof_pos - max_frame_joint_delta,
                    reference_dof_pos + max_frame_joint_delta,
                )
            mujoco.mj_forward(model, data)
            candidate_residual, _, _ = _weighted_residual(
                data, targets, desired_positions, desired_orientations, orientation_weight
            )
            candidate_posture = reference_dof_pos - data.qpos[qpos_addresses]
            candidate_cost = float(
                candidate_residual @ candidate_residual
                + posture_weight * (candidate_posture @ candidate_posture)
            )
            if candidate_cost < cost:
                accepted = True
                break
        if not accepted:
            data.qpos[qpos_addresses] = current_dof_pos
            mujoco.mj_forward(model, data)
            break

    _, position_errors, orientation_errors = _weighted_residual(
        data, targets, desired_positions, desired_orientations, orientation_weight
    )
    return position_errors, orientation_errors, iterations_used


def _nonsingular_leg_seed(default_dof_pos: np.ndarray) -> np.ndarray:
    seed = np.asarray(default_dof_pos, dtype=np.float64).copy()
    joint_index = {name: idx for idx, name in enumerate(MINI3_JOINT_NAMES)}
    for side in ("left", "right"):
        seed[joint_index[f"{side}_hip_pitch_joint"]] = -0.075
        seed[joint_index[f"{side}_knee_pitch_joint"]] = 0.15
        seed[joint_index[f"{side}_ankle_pitch_joint"]] = -0.075
    return seed


def _apply_geometric_leg_seed(
    dof_pos: np.ndarray,
    desired_positions: np.ndarray,
    root_position: np.ndarray,
    maximum_leg_lengths: np.ndarray,
) -> None:
    joint_index = {name: idx for idx, name in enumerate(MINI3_JOINT_NAMES)}
    for target_idx, side in enumerate(("left", "right")):
        leg_length = float(np.linalg.norm(desired_positions[target_idx] - root_position))
        ratio = np.clip(leg_length / float(maximum_leg_lengths[target_idx]), 0.0, 1.0)
        knee_angle = max(0.02, float(2.0 * np.arccos(ratio)))
        dof_pos[joint_index[f"{side}_hip_pitch_joint"]] = -0.5 * knee_angle
        dof_pos[joint_index[f"{side}_knee_pitch_joint"]] = knee_angle
        dof_pos[joint_index[f"{side}_ankle_pitch_joint"]] = -0.5 * knee_angle


def retarget_pico_clip(
    clip: PicoMotionClip,
    model: mujoco.MjModel,
    *,
    scale: float | None = None,
    max_iterations: int = 40,
    position_tolerance: float = 0.005,
    orientation_weight: float = 0.05,
    damping: float = 0.001,
    posture_weight: float = 0.0001,
    max_joint_step: float = 0.08,
    max_frame_joint_delta: float = 0.12,
    show_progress: bool = True,
) -> RetargetResult:
    for name, value in (
        ("position_tolerance", position_tolerance),
        ("damping", damping),
        ("max_joint_step", max_joint_step),
        ("max_frame_joint_delta", max_frame_joint_delta),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive, got {value}")
    if max_iterations <= 0:
        raise ValueError(f"max_iterations must be positive, got {max_iterations}")
    if not np.isfinite(orientation_weight) or orientation_weight < 0.0:
        raise ValueError(f"orientation_weight must be finite and non-negative, got {orientation_weight}")
    if not np.isfinite(posture_weight) or posture_weight < 0.0:
        raise ValueError(f"posture_weight must be finite and non-negative, got {posture_weight}")

    layout = validate_mini3_model(model)
    data = mujoco.MjData(model)
    joint_ids = np.asarray(
        [_object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in MINI3_JOINT_NAMES],
        dtype=np.int32,
    )
    qpos_addresses = layout.joint_qpos_adrs
    dof_addresses = np.asarray([model.jnt_dofadr[joint_id] for joint_id in joint_ids])
    targets = _tracking_targets(model)
    base_body_id = _object_id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    maximum_leg_lengths = np.asarray(
        [np.linalg.norm(_world_point(data, target) - data.xpos[base_body_id]) for target in targets[:2]],
        dtype=np.float64,
    )
    source_leg_lengths = np.linalg.norm(
        clip.body_pos_w[:, 1:3] - clip.body_pos_w[:, 0:1], axis=-1
    )
    source_extended_leg_length = float(np.percentile(source_leg_lengths, 95.0))
    if source_extended_leg_length <= 1.0e-8:
        raise ValueError(f"PICO clip={clip.path} has degenerate pelvis-to-foot distances")
    resolved_scale = (
        float(np.mean(maximum_leg_lengths) / source_extended_leg_length)
        if scale is None
        else float(scale)
    )
    if not np.isfinite(resolved_scale) or resolved_scale <= 0.0:
        raise ValueError(f"scale must be finite and positive, got {resolved_scale}")

    source_index = {name: idx for idx, name in enumerate(PICO_BODY_NAMES)}
    orientation_frame_offset_root = []
    source_root_quat_inverse_initial = _quat_conjugate_wxyz(
        clip.root_quat_wxyz[0]
    )
    target_root_quat_inverse_default = _quat_conjugate_wxyz(
        data.xquat[base_body_id]
    )
    for target in targets:
        source_quat = clip.body_quat_wxyz[0, source_index[target.source_name]]
        target_quat = data.xquat[target.target_body_id]
        source_quat_root_initial = _quat_multiply_wxyz(
            source_root_quat_inverse_initial, source_quat
        )
        target_quat_root_default = _quat_multiply_wxyz(
            target_root_quat_inverse_default, target_quat
        )
        orientation_frame_offset_root.append(
            _quat_multiply_wxyz(
                _quat_conjugate_wxyz(source_quat_root_initial),
                target_quat_root_default,
            )
        )

    frame_count = len(clip.body_pos_w)
    root_pos = np.empty((frame_count, 3), dtype=np.float64)
    dof_pos = np.empty((frame_count, len(qpos_addresses)), dtype=np.float64)
    position_errors = np.empty((frame_count, len(targets)), dtype=np.float64)
    orientation_errors = np.empty_like(position_errors)
    iterations = np.empty(frame_count, dtype=np.int32)
    target_root_origin = np.asarray(model.qpos0[:3], dtype=np.float64).copy()
    source_root_origin = clip.body_pos_w[0, 0].copy()
    previous_dof_pos = _nonsingular_leg_seed(model.qpos0[qpos_addresses])

    frame_indices: Any = range(frame_count)
    if show_progress:
        frame_indices = tqdm(frame_indices, desc="Retargeting PICO to Mini3", unit="frame")
    for frame_idx in frame_indices:
        current_root_pos = target_root_origin + resolved_scale * (
            clip.body_pos_w[frame_idx, 0] - source_root_origin
        )
        root_pos[frame_idx] = current_root_pos
        desired_positions = np.stack(
            [
                current_root_pos
                + resolved_scale
                * (
                    clip.body_pos_w[frame_idx, source_index[target.source_name]]
                    - clip.body_pos_w[frame_idx, source_index[PICO_BODY_NAMES[0]]]
                )
                for target in targets
            ]
        )
        frame_seed = previous_dof_pos.copy()
        _apply_geometric_leg_seed(
            frame_seed, desired_positions, current_root_pos, maximum_leg_lengths
        )
        if frame_idx > 0:
            frame_seed = np.clip(
                frame_seed,
                previous_dof_pos - max_frame_joint_delta,
                previous_dof_pos + max_frame_joint_delta,
            )
        data.qpos[:] = model.qpos0
        data.qpos[:3] = current_root_pos
        data.qpos[3:7] = clip.root_quat_wxyz[frame_idx]
        data.qpos[qpos_addresses] = frame_seed
        _clip_control_joints(model, data.qpos, joint_ids, qpos_addresses)
        source_root_quat_inverse = _quat_conjugate_wxyz(
            clip.root_quat_wxyz[frame_idx]
        )
        desired_orientations = np.stack(
            [
                _quat_multiply_wxyz(
                    clip.root_quat_wxyz[frame_idx],
                    _quat_multiply_wxyz(
                        _quat_multiply_wxyz(
                            source_root_quat_inverse,
                            clip.body_quat_wxyz[
                                frame_idx, source_index[target.source_name]
                            ],
                        ),
                        orientation_frame_offset_root[target_idx],
                    ),
                )
                for target_idx, target in enumerate(targets)
            ]
        )
        (
            position_errors[frame_idx],
            orientation_errors[frame_idx],
            iterations[frame_idx],
        ) = _solve_ik_frame(
            model,
            data,
            targets,
            desired_positions,
            desired_orientations,
            joint_ids,
            qpos_addresses,
            dof_addresses,
            previous_dof_pos,
            max_iterations=max_iterations,
            position_tolerance=position_tolerance,
            orientation_weight=orientation_weight,
            damping=damping,
            posture_weight=posture_weight,
            max_joint_step=max_joint_step,
            max_frame_joint_delta=max_frame_joint_delta if frame_idx > 0 else None,
        )
        previous_dof_pos = data.qpos[qpos_addresses].copy()
        dof_pos[frame_idx] = previous_dof_pos

    target_mapping = {
        target.source_name: {
            "target_body": target.target_body_name,
            "target_local_point": target.local_point.tolist(),
            "orientation_scale": target.orientation_scale,
        }
        for target in targets
    }
    mean_position_error = {
        target.source_name: float(np.mean(position_errors[:, idx]))
        for idx, target in enumerate(targets)
    }
    max_position_error = {
        target.source_name: float(np.max(position_errors[:, idx]))
        for idx, target in enumerate(targets)
    }
    mean_orientation_error = {
        target.source_name: float(np.mean(orientation_errors[:, idx]))
        for idx, target in enumerate(targets)
        if target.orientation_scale > 0.0
    }
    max_orientation_error = {
        target.source_name: float(np.max(orientation_errors[:, idx]))
        for idx, target in enumerate(targets)
        if target.orientation_scale > 0.0
    }
    return RetargetResult(
        motion=Mini3Motion(
            root_pos=root_pos,
            root_quat_wxyz=clip.root_quat_wxyz,
            dof_pos=dof_pos,
            fps=clip.fps,
            joint_names_present=True,
        ),
        scale=resolved_scale,
        mean_position_error_m=mean_position_error,
        max_position_error_m=max_position_error,
        mean_orientation_error_rad=mean_orientation_error,
        max_orientation_error_rad=max_orientation_error,
        mean_iterations=float(np.mean(iterations)),
        max_iterations=int(np.max(iterations)),
        target_mapping=target_mapping,
    )


def _write_npz_atomic(path: Path, qpos: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, qpos=np.asarray(qpos, dtype=np.float32))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _is_unpacked_clip(path: Path) -> bool:
    return path.is_dir() and all((path / f"{name}.npy").is_file() for name in REQUIRED_FIELDS)


def discover_pico_clips(input_path: str | Path) -> tuple[list[Path], Path]:
    source = Path(input_path).expanduser().resolve()
    if source.is_file():
        if source.suffix.lower() != ".npz":
            raise ValueError(f"PICO input file must be .npz: {source}")
        return [source], source.parent
    if not source.is_dir():
        raise FileNotFoundError(f"PICO input does not exist: {source}")
    if _is_unpacked_clip(source):
        return [source], source.parent
    clips = sorted(path for path in source.rglob("*.npz") if path.is_file())
    clips.extend(
        path
        for path in sorted(source.rglob("*"))
        if path.is_dir() and _is_unpacked_clip(path)
    )
    clips = sorted(clips)
    if not clips:
        raise FileNotFoundError(f"No PICO .npz or unpacked clip directories found under: {source}")
    return clips, source


def _relative_clip_key(path: Path, source_root: Path) -> Path:
    relative = path.relative_to(source_root)
    return relative.with_suffix("") if path.is_file() else relative


def _retarget_metadata(clip: PicoMotionClip, result: RetargetResult) -> dict[str, Any]:
    return {
        "source_file": str(clip.path),
        "source_schema": "pico_motion_clip_v3",
        "source_body_names": list(PICO_BODY_NAMES),
        "source_body_state_frame": clip.body_state_frame,
        "pico_position_axes_version": clip.position_axes_version,
        "root_orientation_source": clip.root_orientation_source,
        "endpoint_orientation_alignment": (
            "source-root-relative_with_right-multiplied_link-frame-offset"
        ),
        "position_scale": result.scale,
        "target_mapping": result.target_mapping,
        "ik_mean_position_error_m": result.mean_position_error_m,
        "ik_max_position_error_m": result.max_position_error_m,
        "ik_mean_orientation_error_rad": result.mean_orientation_error_rad,
        "ik_max_orientation_error_rad": result.max_orientation_error_rad,
        "ik_mean_iterations": result.mean_iterations,
        "ik_max_iterations": result.max_iterations,
    }


def convert_dataset(
    input_path: str | Path,
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    mjcf: str | Path = DEFAULT_MJCF,
    target_fps: float = 50.0,
    dataset_name: str = "pico",
    scale: float | None = None,
    max_iterations: int = 40,
    position_tolerance: float = 0.005,
    orientation_weight: float = 0.05,
    damping: float = 0.001,
    posture_weight: float = 0.0001,
    max_joint_step: float = 0.08,
    max_frame_joint_delta: float = 0.12,
    overwrite: bool = False,
    viewer: bool | None = None,
    viewer_port: int = 8080,
    viewer_loop: bool = False,
    show_progress: bool = True,
) -> ConversionSummary:
    paths, source_root = discover_pico_clips(input_path)
    output_root_path = Path(output_root).expanduser().resolve()
    mjcf_path = Path(mjcf).expanduser().resolve()
    if not mjcf_path.is_file():
        raise FileNotFoundError(f"Mini3 MJCF does not exist: {mjcf_path}")
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    layout = validate_mini3_model(model)
    output_root_path.mkdir(parents=True, exist_ok=True)
    resolved_viewer = len(paths) == 1 if viewer is None else viewer
    if resolved_viewer and len(paths) != 1:
        raise ValueError("Viewer mode supports exactly one PICO clip")

    converted = 0
    skipped = 0
    displayed_motion: Path | None = None
    path_iterator: Any = paths
    if show_progress and len(paths) > 1:
        path_iterator = tqdm(paths, desc="Converting PICO clips", unit="clip")
    for source_path in path_iterator:
        clip_key = _relative_clip_key(source_path, source_root)
        output_path = output_root_path / MOTIONS_SUBDIR / clip_key.with_suffix(".npz")
        if output_path.is_file() and not overwrite:
            qpos = load_motion(output_path)
            if qpos.shape[1] != model.nq:
                raise ValueError(
                    f"Existing output {output_path} has qpos width={qpos.shape[1]}, "
                    f"expected {model.nq}"
                )
            skipped += 1
        else:
            clip = load_pico_clip(source_path)
            result = retarget_pico_clip(
                clip,
                model,
                scale=scale,
                max_iterations=max_iterations,
                position_tolerance=position_tolerance,
                orientation_weight=orientation_weight,
                damping=damping,
                posture_weight=posture_weight,
                max_joint_step=max_joint_step,
                max_frame_joint_delta=max_frame_joint_delta,
                show_progress=show_progress,
            )
            qpos = motion_to_qpos(result.motion, model, layout, target_fps=target_fps)
            _write_npz_atomic(output_path, qpos)
            _write_json_atomic(output_path.with_suffix(".json"), _retarget_metadata(clip, result))
            converted += 1
        displayed_motion = output_path

    motion_paths = sorted((output_root_path / MOTIONS_SUBDIR).rglob("*.npz"))
    output_frames = sum(int(load_motion(path).shape[0]) for path in motion_paths)
    manifest_path = write_manifest(
        output_root_path,
        dataset_name=dataset_name,
        mjcf=mjcf_path,
        timestep=1.0 / target_fps,
        qpos_names=layout.qpos_names,
        num_motions=len(motion_paths),
        source={
            "format": "pico_motion_clip_v3_sparse_ik",
            "robot": "mini3",
            "source_root": str(source_root),
            "required_body_names": list(PICO_BODY_NAMES),
            "source_body_quaternion_order": "wxyz",
            "root_orientation_preference": "sonic_smpl_anchor_orientation",
            "target_joint_names": list(MINI3_JOINT_NAMES),
            "target_fps": float(target_fps),
            "retarget": "MuJoCo damped least-squares IK with joint-limit projection",
            "max_frame_joint_delta_rad": float(max_frame_joint_delta),
            "resampling": "linear root/joints; shortest-path quaternion SLERP",
        },
        total_hours=output_frames / target_fps / 3600.0,
    )

    if resolved_viewer and displayed_motion is not None:
        from any4hdmi.scripts.viewer import view_motion

        view_motion(displayed_motion, fps=target_fps, loop=viewer_loop, port=viewer_port)
    return ConversionSummary(
        selected=len(paths),
        converted=converted,
        skipped=skipped,
        output_motions=len(motion_paths),
        output_frames=output_frames,
        output_root=output_root_path,
        manifest_path=manifest_path,
        displayed_motion=displayed_motion if resolved_viewer else None,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retarget a PICO motion-clip .npz or unpacked .npy directory to "
            "qpos-only Mini3 any4hdmi data."
        )
    )
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--target-fps", type=float, default=50.0)
    parser.add_argument("--dataset-name", default="pico")
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--max-iterations", type=int, default=40)
    parser.add_argument("--position-tolerance", type=float, default=0.005)
    parser.add_argument("--orientation-weight", type=float, default=0.05)
    parser.add_argument("--damping", type=float, default=0.001)
    parser.add_argument("--posture-weight", type=float, default=0.0001)
    parser.add_argument("--max-joint-step", type=float, default=0.08)
    parser.add_argument("--max-frame-joint-delta", type=float, default=0.12)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--viewer",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Default: enabled for one clip and disabled for a batch root.",
    )
    parser.add_argument("--viewer-port", type=int, default=8080)
    parser.add_argument("--viewer-loop", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    summary = convert_dataset(
        args.input_path,
        output_root=args.output_path,
        mjcf=args.mjcf,
        target_fps=args.target_fps,
        dataset_name=args.dataset_name,
        scale=args.scale,
        max_iterations=args.max_iterations,
        position_tolerance=args.position_tolerance,
        orientation_weight=args.orientation_weight,
        damping=args.damping,
        posture_weight=args.posture_weight,
        max_joint_step=args.max_joint_step,
        max_frame_joint_delta=args.max_frame_joint_delta,
        overwrite=args.overwrite,
        viewer=args.viewer,
        viewer_port=args.viewer_port,
        viewer_loop=args.viewer_loop,
        show_progress=not args.no_progress,
    )
    payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(summary).items()
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
