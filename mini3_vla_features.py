"""Extract Mini3-specific 36D VLA observations and hierarchical actions.

The names follow this robot's kinematic chain, not the G1 joint convention.
The action is a compact learning target for a future Mini3-to-MimicLite adapter;
it is not a lossless replacement for the complete policy reference/history.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


ARM_PARTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch",
             "elbow_yaw", "wrist_roll", "wrist_pitch")
LEG_PARTS = ("hip_pitch", "hip_roll", "hip_yaw", "knee_pitch", "ankle_pitch", "ankle_roll")
FEATURE_NAMES = tuple(
    name
    for side in ("left", "right")
    for name in (*(f"{side}_{part}_joint" for part in ARM_PARTS), f"{side}_gripper_closedness")
) + tuple(f"{side}_{part}_joint" for side in ("left", "right") for part in LEG_PARTS) + (
    "waist_yaw_joint", "root_roll", "root_pitch", "root_yaw_rate",
    "root_linear_velocity_heading_x", "root_linear_velocity_heading_y",
    "root_linear_velocity_heading_z", "root_height",
)
FEATURE_UNITS = tuple("normalized" if "gripper" in name else "rad" for name in FEATURE_NAMES[:29]) + (
    "rad", "rad", "rad/s", "m/s", "m/s", "m/s", "m",
)
GRIPPER_MAX_HALF_OPENING_M = .035


def _names(metadata: Mapping[str, Any], key: str) -> list[str]:
    names = list(metadata[key])
    if not all(isinstance(name, str) for name in names) or len(names) != len(set(names)):
        raise ValueError(f"{key} must contain unique string names")
    return names


def _array(samples: Mapping[str, Any], name: str, shape: tuple[int | None, ...]) -> np.ndarray:
    array = np.asarray(samples[name], dtype=np.float64)
    if array.ndim != len(shape) or any(size is not None and array.shape[i] != size
                                      for i, size in enumerate(shape)):
        raise ValueError(f"{name} has shape {array.shape}; expected {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _indices(metadata: Mapping[str, Any], key: str, names: list[str], width: int) -> dict[str, int]:
    values = np.asarray(metadata[key])
    if (values.shape != (len(names),) or values.dtype.kind not in "iu"
            or np.any(values < 0) or np.any(values >= width)
            or len(set(values.tolist())) != len(values)):
        raise ValueError(f"{key} must contain one unique valid integer index per joint")
    return dict(zip(names, map(int, values), strict=True))


def _root_start(metadata: Mapping[str, Any], key: str, width: int, size: int) -> int:
    index = metadata[key]
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or not 0 <= index <= width - size:
        raise ValueError(f"{key} must index a complete root segment")
    return int(index)


def _euler_wxyz(quaternion: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    norm = np.linalg.norm(quaternion, axis=-1)
    if np.any(norm < 1e-8):
        raise ValueError(f"{name} contains a zero quaternion")
    w, x, y, z = (quaternion / norm[:, None]).T
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1., 1.))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def _heading_velocity(world_velocity: np.ndarray, measured_yaw: np.ndarray) -> np.ndarray:
    """World-to-heading rotation; heading shares world Z and measured base yaw."""
    c, s = np.cos(measured_yaw), np.sin(measured_yaw)
    x, y, z = world_velocity.T
    return np.column_stack((c * x + s * y, -s * x + c * y, z))


def extract_features(samples: Mapping[str, Any], metadata: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return owned finite ``(state, actions)`` float32 arrays of shape ``(N, 36)``.

    Input is ``control_samples.npz`` plus ``control_metadata.json`` written by
    ``ControlRecorder``. The 29 articulated channels are left arm (7), left
    gripper (1), right arm (7), right gripper (1), left leg (6), right leg (6),
    waist yaw (1), followed by the seven root channels in ``FEATURE_NAMES``.
    Joint positions use radians. Grippers use clipped closedness
    ``1 - half_opening / 0.035``: zero is fully open, one fully closed.

    State uses PRE-ACTION qpos/qvel, mapped by metadata joint names/addresses.
    MuJoCo freejoint translational velocity is world-aligned and angular
    velocity is body-local. State root yaw rate is the Euler yaw derivative
    ``(sin(roll) * omega_y + cos(roll) * omega_z) / cos(pitch)`` in rad/s, not
    just the third component of the body angular velocity. A near-vertical
    root pitch (Euler singularity) is rejected rather than silently clipped.

    Actions contain current (offset 0) ORIGINAL-21 joint reference positions
    from ``reference_joint_pos``, actual planned ``extra_q`` motor targets for
    the six added axes (including integral correction), and commanded gripper
    half-openings. Root roll/pitch/height come from offset 0 of the gated body
    reference for ``base_link``. Action yaw rate is wrapped Euler yaw change
    from reference offset 0 to offset 1, multiplied by ``control_hz``; at a
    clamped reference gate this is zero. Offsets are found by value, because
    the saved future-reference axis can begin with negative history offsets.

    Both state and action linear velocities are expressed in the SAME CURRENT
    MEASURED yaw-aligned heading frame: x forward, y left, z world-up. This frame
    retains world Z and does not tilt with measured root roll/pitch. Action
    velocity comes from
    ``reference_body_lin_vel_w`` (including online root-reference correction),
    rotated by measured yaw, not reference yaw. Height is absolute world Z in
    meters; neither world XY nor absolute yaw is encoded.

    Metadata requires policy/extra/gripper joint names and qpos indices,
    reference joint/body names and step offsets, root qpos/qvel starts, and
    control_hz. Source arrays/dictionaries are never modified. This function
    does not resample time or discard failed/partial actions; the dataset
    exporter must select valid complete rows and synchronize images itself.
    Original future references and low-level motor commands remain necessary
    sidecar records. A trained 36D policy still needs a Mini3-to-MimicLite
    deployment adapter; this vector alone does not reproduce its full inputs.
    """
    qpos = _array(samples, "qpos", (None, None))
    count = len(qpos)
    qvel = _array(samples, "qvel", (count, None))
    policy_names = _names(metadata, "policy_joint_names")
    extra_names = _names(metadata, "extra_joint_names")
    gripper_names = _names(metadata, "gripper_joint_names")
    reference_names = _names(metadata, "reference_joint_names")
    body_names = _names(metadata, "reference_body_names")
    if len(policy_names) != 21 or len(extra_names) != 6 or len(gripper_names) != 2:
        raise ValueError("Expected 21 original joints, six added joints and two grippers")
    if set(policy_names) & set(extra_names) or set(policy_names) != set(reference_names):
        raise ValueError("Original and added joints must be disjoint; reference joints must match original joints")
    actual_indices = _indices(metadata, "policy_qpos_indices", policy_names, qpos.shape[1])
    actual_indices.update(_indices(metadata, "extra_qpos_indices", extra_names, qpos.shape[1]))
    grip_indices = _indices(metadata, "gripper_qpos_indices", gripper_names, qpos.shape[1])
    if len(set((*actual_indices.values(), *grip_indices.values()))) != 29:
        raise ValueError("Original, added and gripper qpos addresses must not overlap")
    rootq = _root_start(metadata, "root_qpos_start", qpos.shape[1], 7)
    rootv = _root_start(metadata, "root_qvel_start", qvel.shape[1], 6)
    if any(rootq <= index < rootq + 7 for index in (*actual_indices.values(), *grip_indices.values())):
        raise ValueError("Joint addresses must not overlap the floating root")
    offsets = np.asarray(metadata["reference_step_offsets"])
    if (offsets.ndim != 1 or offsets.dtype.kind not in "iu"
            or len(set(offsets.tolist())) != len(offsets) or 0 not in offsets or 1 not in offsets):
        raise ValueError("reference_step_offsets must contain unique integers including 0 and 1")
    if "reference_step_offsets" in samples and not np.array_equal(samples["reference_step_offsets"], offsets):
        raise ValueError("Sample reference offsets disagree with metadata")
    current, future = int(np.flatnonzero(offsets == 0)[0]), int(np.flatnonzero(offsets == 1)[0])
    try:
        root_body = body_names.index("base_link")
    except ValueError as exc:
        raise ValueError("reference_body_names must include base_link") from exc
    hz = float(metadata["control_hz"])
    if not np.isfinite(hz) or hz <= 0:
        raise ValueError("control_hz must be finite and positive")
    reference = _array(samples, "reference_joint_pos", (count, len(offsets), len(reference_names)))
    extra = _array(samples, "extra_q", (count, len(extra_names)))
    opening = _array(samples, "gripper_opening", (count, len(gripper_names)))
    positions = _array(samples, "reference_body_pos_w", (count, len(offsets), len(body_names), 3))
    rotations = _array(samples, "reference_body_quat_w", (count, len(offsets), len(body_names), 4))
    velocities = _array(samples, "reference_body_lin_vel_w", (count, len(offsets), len(body_names), 3))
    ref_indices, extra_indices = {name: i for i, name in enumerate(reference_names)}, {
        name: i for i, name in enumerate(extra_names)}
    state = np.empty((count, len(FEATURE_NAMES)), dtype=np.float64)
    action = np.empty_like(state)
    for index, name in enumerate(FEATURE_NAMES[:29]):
        if name.endswith("_gripper_closedness"):
            joint = name.removesuffix("_closedness") + "_finger_joint"
            if joint not in grip_indices:
                raise ValueError(f"Missing required gripper joint: {joint}")
            state[:, index] = np.clip(1 - qpos[:, grip_indices[joint]] / GRIPPER_MAX_HALF_OPENING_M, 0, 1)
            action[:, index] = np.clip(1 - opening[:, gripper_names.index(joint)] / GRIPPER_MAX_HALF_OPENING_M, 0, 1)
        else:
            if name not in actual_indices:
                raise ValueError(f"Missing required Mini3 joint: {name}")
            state[:, index] = qpos[:, actual_indices[name]]
            action[:, index] = (extra[:, extra_indices[name]] if name in extra_indices
                                else reference[:, current, ref_indices[name]])
    roll, pitch, yaw = _euler_wxyz(qpos[:, rootq + 3:rootq + 7], "qpos root")
    denominator = np.cos(pitch)
    if np.any(np.abs(denominator) < 1e-6):
        raise ValueError("State root pitch is near the Euler yaw-rate singularity")
    omega = qvel[:, rootv + 3:rootv + 6]
    state[:, 29:32] = np.column_stack((roll, pitch,
        (np.sin(roll) * omega[:, 1] + np.cos(roll) * omega[:, 2]) / denominator))
    state[:, 32:35] = _heading_velocity(qvel[:, rootv:rootv + 3], yaw)
    state[:, 35] = qpos[:, rootq + 2]
    ref_roll, ref_pitch, ref_yaw = _euler_wxyz(rotations[:, current, root_body], "current reference root")
    _, _, next_yaw = _euler_wxyz(rotations[:, future, root_body], "next reference root")
    yaw_delta = (next_yaw - ref_yaw + np.pi) % (2 * np.pi) - np.pi
    action[:, 29:32] = np.column_stack((ref_roll, ref_pitch, yaw_delta * hz))
    action[:, 32:35] = _heading_velocity(velocities[:, current, root_body], yaw)
    action[:, 35] = positions[:, current, root_body, 2]
    with np.errstate(over="ignore", invalid="ignore"):
        state, action = state.astype(np.float32), action.astype(np.float32)
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("Features exceed finite float32 range")
    return state, action
