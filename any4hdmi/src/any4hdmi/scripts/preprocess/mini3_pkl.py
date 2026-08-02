"""Convert trusted, retargeted Mini3 joblib/PKL motions to any4hdmi NPZ.

PKL/joblib loading can execute arbitrary code. Only use this converter with
files from a trusted source.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import mujoco
import numpy as np
from tqdm import tqdm

from any4hdmi.core.format import MOTIONS_SUBDIR, load_motion, repo_root, write_manifest
from any4hdmi.utils.mjcf import qpos_names_from_model


DEFAULT_INPUT_ROOT = Path("/home/amax/Desktop/robot/UFO/humanoidverse/data/pkl")
DEFAULT_OUTPUT_ROOT = repo_root() / "output" / "mini3" / "sonic"
DEFAULT_MJCF = repo_root() / "assets" / "robots" / "mini3_mjlab" / "mini3.xml"
INDEX_NAME = "conversion_index.json"
FAILURES_NAME = "conversion_failures.json"
BASE_JOINT_NAME = "floating_base"

MINI3_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_pitch_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
)


@dataclass(frozen=True)
class Mini3Layout:
    root_qpos_adr: int
    joint_qpos_adrs: np.ndarray
    qpos_names: list[str]
    actuator_target_joints: tuple[str, ...]


@dataclass(frozen=True)
class Mini3Motion:
    root_pos: np.ndarray
    root_quat_wxyz: np.ndarray
    dof_pos: np.ndarray
    fps: float
    joint_names_present: bool


@dataclass(frozen=True)
class ConversionSummary:
    selected: int
    converted: int
    skipped: int
    failed: int
    output_motions: int
    output_frames: int
    output_root: Path
    manifest_path: Path
    displayed_motion: Path | None


def _joint_name(model: mujoco.MjModel, joint_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    if name is None:
        raise ValueError(f"MJCF joint id={joint_id} has no name")
    return name


def validate_mini3_model(model: mujoco.MjModel) -> Mini3Layout:
    """Validate strict Mini3 joint coverage and derive mappings from the MJCF."""

    if model.nq != 28:
        raise ValueError(f"Mini3 MJCF must have nq=28 (free base + 21 joints), got {model.nq}")

    free_joint_ids = [
        joint_id
        for joint_id in range(model.njnt)
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
    ]
    if len(free_joint_ids) != 1:
        raise ValueError(f"Mini3 MJCF must have exactly one free joint, got {len(free_joint_ids)}")
    root_joint_id = free_joint_ids[0]
    root_joint_name = _joint_name(model, root_joint_id)
    if root_joint_name != BASE_JOINT_NAME:
        raise ValueError(
            f"Mini3 free joint must be named {BASE_JOINT_NAME!r}, got {root_joint_name!r}"
        )

    motion_joint_ids = [joint_id for joint_id in range(model.njnt) if joint_id != root_joint_id]
    motion_joint_names = tuple(_joint_name(model, joint_id) for joint_id in motion_joint_ids)
    if motion_joint_names != MINI3_JOINT_NAMES:
        raise ValueError(
            "Mini3 MJCF motion joints must exactly match the canonical 21-joint order. "
            f"expected={list(MINI3_JOINT_NAMES)}, actual={list(motion_joint_names)}"
        )
    non_hinge = [
        name
        for joint_id, name in zip(motion_joint_ids, motion_joint_names)
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE
    ]
    if non_hinge:
        raise ValueError(f"Mini3 motion joints must all be hinge joints, got non-hinge={non_hinge}")

    actuator_targets: list[str] = []
    for actuator_id in range(model.nu):
        if model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_JOINT:
            raise ValueError(f"Mini3 actuator id={actuator_id} does not use a joint transmission")
        target_joint_id = int(model.actuator_trnid[actuator_id, 0])
        if target_joint_id < 0:
            raise ValueError(f"Mini3 actuator id={actuator_id} has no transmission target joint")
        actuator_targets.append(_joint_name(model, target_joint_id))
    if len(actuator_targets) != len(set(actuator_targets)):
        raise ValueError(f"Mini3 actuators target duplicate joints: {actuator_targets}")
    if set(actuator_targets) != set(MINI3_JOINT_NAMES):
        missing = sorted(set(MINI3_JOINT_NAMES) - set(actuator_targets))
        extra = sorted(set(actuator_targets) - set(MINI3_JOINT_NAMES))
        raise ValueError(
            "Mini3 actuator transmission targets must cover all 21 motion joints exactly once. "
            f"missing={missing}, extra={extra}"
        )

    return Mini3Layout(
        root_qpos_adr=int(model.jnt_qposadr[root_joint_id]),
        joint_qpos_adrs=np.asarray(
            [int(model.jnt_qposadr[joint_id]) for joint_id in motion_joint_ids], dtype=np.int32
        ),
        qpos_names=qpos_names_from_model(model, base_joint_name=BASE_JOINT_NAME),
        actuator_target_joints=tuple(actuator_targets),
    )


def _scalar_fps(value: Any, source_path: Path) -> float:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{source_path}: fps must contain one scalar, got shape={array.shape}")
    fps = float(array.reshape(-1)[0])
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"{source_path}: fps must be finite and positive, got {fps}")
    return fps


def _normalized_quaternions_wxyz(
    root_rot: Any, source_path: Path, root_quat_order: str
) -> np.ndarray:
    quaternions = np.asarray(root_rot, dtype=np.float64)
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError(f"{source_path}: root_rot must have shape [T,4], got {quaternions.shape}")
    if not np.all(np.isfinite(quaternions)):
        raise ValueError(f"{source_path}: root_rot contains non-finite values")
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(norms < 1e-8):
        frame = int(np.flatnonzero(norms < 1e-8)[0])
        raise ValueError(f"{source_path}: root_rot has a zero quaternion at frame {frame}")
    quaternions = quaternions / norms[:, None]
    if root_quat_order == "xyzw":
        quaternions = quaternions[:, [3, 0, 1, 2]]
    elif root_quat_order != "wxyz":
        raise ValueError(f"Unsupported root quaternion order: {root_quat_order}")

    # q and -q encode the same rotation; use one continuous hemisphere before SLERP.
    for frame in range(1, len(quaternions)):
        if np.dot(quaternions[frame - 1], quaternions[frame]) < 0.0:
            quaternions[frame] *= -1.0
    return quaternions


def load_mini3_pkl(source_path: str | Path, *, root_quat_order: str = "xyzw") -> Mini3Motion:
    """Load one trusted flat Mini3 PKL and validate its strict input schema."""

    source_path = Path(source_path).expanduser().resolve()
    if source_path.suffix.lower() != ".pkl" or not source_path.is_file():
        raise FileNotFoundError(f"Expected an existing .pkl file, got: {source_path}")

    payload = joblib.load(source_path)
    if not isinstance(payload, dict):
        raise ValueError(f"{source_path}: expected a flat dictionary, got {type(payload)!r}")
    required = ("root_pos", "root_rot", "dof_pos", "fps")
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"{source_path}: missing required fields {missing}")

    root_pos = np.asarray(payload["root_pos"], dtype=np.float64)
    dof_pos = np.asarray(payload["dof_pos"], dtype=np.float64)
    root_quat_wxyz = _normalized_quaternions_wxyz(
        payload["root_rot"], source_path, root_quat_order
    )
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"{source_path}: root_pos must have shape [T,3], got {root_pos.shape}")
    if dof_pos.ndim != 2 or dof_pos.shape[1] != len(MINI3_JOINT_NAMES):
        raise ValueError(
            f"{source_path}: dof_pos must have shape [T,{len(MINI3_JOINT_NAMES)}], "
            f"got {dof_pos.shape}"
        )
    frame_count = root_pos.shape[0]
    if frame_count < 1:
        raise ValueError(f"{source_path}: motion must contain at least one frame")
    if root_quat_wxyz.shape[0] != frame_count or dof_pos.shape[0] != frame_count:
        raise ValueError(
            f"{source_path}: root_pos/root_rot/dof_pos frame counts differ: "
            f"{root_pos.shape[0]}/{root_quat_wxyz.shape[0]}/{dof_pos.shape[0]}"
        )
    if not np.all(np.isfinite(root_pos)):
        raise ValueError(f"{source_path}: root_pos contains non-finite values")
    if not np.all(np.isfinite(dof_pos)):
        raise ValueError(f"{source_path}: dof_pos contains non-finite values")

    joint_names_present = "joint_names" in payload
    if joint_names_present:
        raw_joint_names = np.asarray(payload["joint_names"]).reshape(-1).tolist()
        joint_names = tuple(
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in raw_joint_names
        )
        if len(joint_names) != len(set(joint_names)):
            raise ValueError(f"{source_path}: joint_names contains duplicates: {joint_names}")
        if joint_names != MINI3_JOINT_NAMES:
            raise ValueError(
                f"{source_path}: joint_names must exactly match the canonical Mini3 order. "
                f"expected={list(MINI3_JOINT_NAMES)}, actual={list(joint_names)}"
            )

    return Mini3Motion(
        root_pos=root_pos,
        root_quat_wxyz=root_quat_wxyz,
        dof_pos=dof_pos,
        fps=_scalar_fps(payload["fps"], source_path),
        joint_names_present=joint_names_present,
    )


def _validate_joint_limits(
    model: mujoco.MjModel,
    layout: Mini3Layout,
    dof_pos: np.ndarray,
    source_path: Path,
    tolerance: float,
) -> None:
    for column, (joint_name, qpos_adr) in enumerate(
        zip(MINI3_JOINT_NAMES, layout.joint_qpos_adrs)
    ):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0 or int(model.jnt_qposadr[joint_id]) != int(qpos_adr):
            raise ValueError(f"Could not resolve qpos address for Mini3 joint {joint_name!r}")
        if not model.jnt_limited[joint_id]:
            raise ValueError(f"Mini3 joint {joint_name!r} must have an MJCF range")
        lower, upper = (float(value) for value in model.jnt_range[joint_id])
        values = dof_pos[:, column]
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        if minimum < lower - tolerance or maximum > upper + tolerance:
            raise ValueError(
                f"{source_path}: joint {joint_name!r} violates MJCF range [{lower:.6f}, "
                f"{upper:.6f}] (data=[{minimum:.6f}, {maximum:.6f}], "
                f"tolerance={tolerance:.6g}); values are not clipped"
            )


def target_frame_count(source_frames: int, source_fps: float, target_fps: float) -> int:
    return math.floor((source_frames - 1) / source_fps * target_fps) + 1


def _linear_resample(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    if len(values) == 1:
        return np.repeat(values, len(positions), axis=0)
    left = np.minimum(np.floor(positions).astype(np.int64), len(values) - 2)
    alpha = (positions - left)[:, None]
    return values[left] + alpha * (values[left + 1] - values[left])


def _slerp_resample(quaternions: np.ndarray, positions: np.ndarray) -> np.ndarray:
    if len(quaternions) == 1:
        return np.repeat(quaternions, len(positions), axis=0)
    left = np.minimum(np.floor(positions).astype(np.int64), len(quaternions) - 2)
    alpha = positions - left
    q0 = quaternions[left]
    q1 = quaternions[left + 1].copy()
    dots = np.sum(q0 * q1, axis=1)
    negative = dots < 0.0
    q1[negative] *= -1.0
    dots = np.clip(np.abs(dots), 0.0, 1.0)

    result = np.empty_like(q0)
    close = dots > 0.9995
    if np.any(close):
        close_alpha = alpha[close, None]
        result[close] = q0[close] + close_alpha * (q1[close] - q0[close])
    far = ~close
    if np.any(far):
        theta = np.arccos(dots[far])
        sin_theta = np.sin(theta)
        left_weight = np.sin((1.0 - alpha[far]) * theta) / sin_theta
        right_weight = np.sin(alpha[far] * theta) / sin_theta
        result[far] = left_weight[:, None] * q0[far] + right_weight[:, None] * q1[far]
    result /= np.linalg.norm(result, axis=1, keepdims=True)
    return result


def motion_to_qpos(
    motion: Mini3Motion,
    model: mujoco.MjModel,
    layout: Mini3Layout,
    *,
    target_fps: float,
) -> np.ndarray:
    if not np.isfinite(target_fps) or target_fps <= 0.0:
        raise ValueError(f"target_fps must be finite and positive, got {target_fps}")
    output_frames = target_frame_count(len(motion.root_pos), motion.fps, target_fps)
    target_times = np.arange(output_frames, dtype=np.float64) / target_fps
    positions = target_times * motion.fps

    root_pos = _linear_resample(motion.root_pos, positions)
    root_quat = _slerp_resample(motion.root_quat_wxyz, positions)
    dof_pos = _linear_resample(motion.dof_pos, positions)

    qpos = np.repeat(np.asarray(model.qpos0, dtype=np.float64)[None, :], output_frames, axis=0)
    root_adr = layout.root_qpos_adr
    qpos[:, root_adr : root_adr + 3] = root_pos
    qpos[:, root_adr + 3 : root_adr + 7] = root_quat
    qpos[:, layout.joint_qpos_adrs] = dof_pos
    if not np.all(np.isfinite(qpos)):
        raise ValueError("Converted qpos contains non-finite values")
    return qpos.astype(np.float32)


def _write_npz_atomic(path: Path, qpos: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as stream:
            np.savez_compressed(stream, qpos=np.asarray(qpos, dtype=np.float32))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def discover_pkl_files(input_path: str | Path, *, limit: int | None = None) -> list[Path]:
    input_path = Path(input_path).expanduser().resolve()
    if input_path.is_file():
        if input_path.suffix.lower() != ".pkl":
            raise ValueError(f"Single input must use the .pkl extension: {input_path}")
        paths = [input_path]
    elif input_path.is_dir():
        paths = sorted(path for path in input_path.rglob("*.pkl") if path.is_file())
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if not paths:
        raise FileNotFoundError(f"No .pkl files found under: {input_path}")
    if limit is not None:
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")
        paths = paths[:limit]
    return paths


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def resolve_source_root(input_path: Path, source_root: str | Path | None) -> Path:
    if source_root is not None:
        resolved = Path(source_root).expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"Source root is not a directory: {resolved}")
    elif _is_relative_to(input_path, DEFAULT_INPUT_ROOT) and DEFAULT_INPUT_ROOT.is_dir():
        resolved = DEFAULT_INPUT_ROOT.resolve()
    else:
        resolved = input_path if input_path.is_dir() else input_path.parent
    if not _is_relative_to(input_path, resolved):
        raise ValueError(f"Input path {input_path} is outside source root {resolved}")
    return resolved


def output_path_for(source_path: Path, source_root: Path, output_root: Path) -> Path:
    relative = source_path.relative_to(source_root).with_suffix(".npz")
    return output_root / MOTIONS_SUBDIR / relative


def _index_settings(
    *, target_fps: float, root_quat_order: str, qpos_names: list[str]
) -> dict[str, Any]:
    return {
        "target_fps": float(target_fps),
        "root_quat_order": root_quat_order,
        "qpos_names": qpos_names,
    }


def _load_index(output_root: Path, settings: dict[str, Any]) -> dict[str, int]:
    path = output_root / INDEX_NAME
    if not path.is_file():
        entries: dict[str, int] = {}
        motions_root = output_root / MOTIONS_SUBDIR
        if motions_root.is_dir():
            for motion_path in sorted(motions_root.rglob("*.npz")):
                qpos = load_motion(motion_path)
                entries[motion_path.relative_to(output_root).as_posix()] = int(qpos.shape[0])
        return entries
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "any4hdmi-mini3-pkl-index-v2":
        raise ValueError(f"Unsupported conversion index format in {path}")
    if payload.get("settings") != settings:
        raise ValueError(
            f"Conversion settings do not match the existing dataset index at {path}. "
            "Use the same target FPS/quaternion order/MJCF layout, or choose a new output path."
        )
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, dict):
        raise ValueError(f"Conversion index entries must be an object: {path}")
    entries = {str(key): int(value) for key, value in raw_entries.items()}
    return {
        relative: frames
        for relative, frames in entries.items()
        if frames > 0 and (output_root / relative).is_file()
    }


def _write_index(
    output_root: Path, entries: dict[str, int], settings: dict[str, Any]
) -> Path:
    path = output_root / INDEX_NAME
    _write_json_atomic(
        path,
        {
            "format": "any4hdmi-mini3-pkl-index-v2",
            "settings": settings,
            "entries": entries,
        },
    )
    return path


def _validate_existing_output(path: Path, expected_width: int) -> int:
    with np.load(path, allow_pickle=False) as payload:
        if payload.files != ["qpos"]:
            raise ValueError(f"Existing output {path} must contain only qpos, got {payload.files}")
        qpos = np.asarray(payload["qpos"])
    if qpos.ndim != 2 or qpos.shape[1] != expected_width or qpos.shape[0] < 1:
        raise ValueError(
            f"Existing output {path} has invalid qpos shape {qpos.shape}; "
            f"expected [T,{expected_width}]"
        )
    if not np.all(np.isfinite(qpos)):
        raise ValueError(f"Existing output {path} contains non-finite qpos values")
    return int(qpos.shape[0])


def convert_dataset(
    input_path: str | Path = DEFAULT_INPUT_ROOT,
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    mjcf: str | Path = DEFAULT_MJCF,
    source_root: str | Path | None = None,
    target_fps: float = 50.0,
    root_quat_order: str = "xyzw",
    dataset_name: str = "sonic",
    overwrite: bool = False,
    continue_on_error: bool = False,
    limit: int | None = None,
    viewer: bool | None = None,
    viewer_port: int = 8080,
    viewer_loop: bool = False,
    joint_limit_tolerance: float = 1e-4,
) -> ConversionSummary:
    input_path = Path(input_path).expanduser().resolve()
    input_is_file = input_path.is_file()
    if not np.isfinite(target_fps) or target_fps <= 0.0:
        raise ValueError(f"target_fps must be finite and positive, got {target_fps}")
    paths = discover_pkl_files(input_path, limit=limit)
    source_root_path = resolve_source_root(input_path, source_root)
    output_root_path = Path(output_root).expanduser().resolve()
    mjcf_path = Path(mjcf).expanduser().resolve()
    if not mjcf_path.is_file():
        raise FileNotFoundError(f"Mini3 MJCF does not exist: {mjcf_path}")
    if not np.isfinite(joint_limit_tolerance) or joint_limit_tolerance < 0.0:
        raise ValueError(
            f"joint_limit_tolerance must be finite and non-negative, got {joint_limit_tolerance}"
        )

    resolved_viewer = input_is_file if viewer is None else viewer
    if resolved_viewer and not input_is_file:
        raise ValueError("Viewer mode is only supported for a single PKL input, not a directory")

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    layout = validate_mini3_model(model)
    output_root_path.mkdir(parents=True, exist_ok=True)
    index_settings = _index_settings(
        target_fps=target_fps,
        root_quat_order=root_quat_order,
        qpos_names=layout.qpos_names,
    )
    index_entries = _load_index(output_root_path, index_settings)
    failures: list[dict[str, str]] = []
    converted = 0
    skipped = 0
    displayed_motion: Path | None = None
    fatal_error: Exception | None = None

    for item_index, source_path in enumerate(tqdm(paths, desc="Converting Mini3 PKL", unit="motion"), 1):
        output_path = output_path_for(source_path, source_root_path, output_root_path)
        output_relative = output_path.relative_to(output_root_path).as_posix()
        try:
            if output_path.is_file() and not overwrite:
                frames = index_entries.get(output_relative)
                if frames is None:
                    frames = _validate_existing_output(output_path, model.nq)
                    index_entries[output_relative] = frames
                skipped += 1
            else:
                motion = load_mini3_pkl(source_path, root_quat_order=root_quat_order)
                _validate_joint_limits(
                    model,
                    layout,
                    motion.dof_pos,
                    source_path,
                    joint_limit_tolerance,
                )
                qpos = motion_to_qpos(motion, model, layout, target_fps=target_fps)
                _validate_joint_limits(
                    model,
                    layout,
                    qpos[:, layout.joint_qpos_adrs],
                    source_path,
                    joint_limit_tolerance,
                )
                _write_npz_atomic(output_path, qpos)
                index_entries[output_relative] = int(qpos.shape[0])
                converted += 1
            if item_index % 5000 == 0:
                _write_index(output_root_path, index_entries, index_settings)
        except Exception as exc:  # Keep a machine-readable batch failure report.
            failures.append(
                {
                    "source": str(source_path),
                    "output": str(output_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            if not continue_on_error:
                fatal_error = exc
                break

    _write_index(output_root_path, index_entries, index_settings)
    _write_json_atomic(output_root_path / FAILURES_NAME, failures)
    output_motions = len(index_entries)
    output_frames = sum(index_entries.values())
    manifest_path = write_manifest(
        output_root_path,
        dataset_name=dataset_name,
        mjcf=mjcf_path,
        timestep=1.0 / target_fps,
        qpos_names=layout.qpos_names,
        num_motions=output_motions,
        source={
            "format": "trusted_joblib_pkl",
            "robot": "mini3",
            "source_root": str(source_root_path),
            "required_fields": ["root_pos", "root_rot", "dof_pos", "fps"],
            "root_quaternion_input_order": root_quat_order,
            "root_quaternion_output_order": "wxyz",
            "source_joint_names": list(MINI3_JOINT_NAMES),
            "source_joint_order_assumption": (
                "When joint_names is absent, dof_pos is assumed to use this canonical Mini3 order."
            ),
            "actuator_mapping": "Derived from each actuator transmission target joint.",
            "target_fps": float(target_fps),
            "resampling": "linear root/joints; shortest-path quaternion SLERP",
            "joint_values_clipped": False,
            "warning": "PKL/joblib loading can execute code; convert trusted inputs only.",
        },
        total_hours=output_frames / target_fps / 3600.0,
    )

    if fatal_error is not None:
        raise RuntimeError(
            f"Mini3 conversion stopped after {converted} converted and {len(failures)} failed file(s). "
            f"See {output_root_path / FAILURES_NAME}"
        ) from fatal_error

    if resolved_viewer:
        displayed_motion = output_path_for(paths[0], source_root_path, output_root_path)
        from any4hdmi.scripts.viewer import view_motion

        view_motion(
            displayed_motion,
            fps=target_fps,
            loop=viewer_loop,
            port=viewer_port,
        )

    return ConversionSummary(
        selected=len(paths),
        converted=converted,
        skipped=skipped,
        failed=len(failures),
        output_motions=output_motions,
        output_frames=output_frames,
        output_root=output_root_path,
        manifest_path=manifest_path,
        displayed_motion=displayed_motion,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert trusted Mini3 joblib/PKL motions to qpos-only any4hdmi NPZ files. "
            "A single-file input opens the MuJoCo/mjviser viewer by default; directory inputs do not."
        ),
        epilog=(
            "Security: joblib/PKL deserialization can execute arbitrary code. "
            "Never run this command on untrusted files."
        ),
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help=(
            "Root used to preserve relative directories. Inputs under the default UFO PKL root "
            "automatically keep their date directory."
        ),
    )
    parser.add_argument("--target-fps", type=float, default=50.0)
    parser.add_argument("--dataset-name", default="sonic")
    parser.add_argument(
        "--root-quat-order",
        choices=("xyzw", "wxyz"),
        default="xyzw",
        help="The provided UFO Mini3 dataset stores root_rot as xyzw.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Convert only the first N sorted files.")
    parser.add_argument(
        "--viewer",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Default: enabled for one file, disabled for a directory.",
    )
    parser.add_argument("--viewer-port", type=int, default=8080)
    parser.add_argument("--viewer-loop", action="store_true")
    parser.add_argument("--joint-limit-tolerance", type=float, default=1e-4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    summary = convert_dataset(
        args.input_path,
        output_root=args.output_path,
        mjcf=args.mjcf,
        source_root=args.source_root,
        target_fps=args.target_fps,
        root_quat_order=args.root_quat_order,
        dataset_name=args.dataset_name,
        overwrite=args.overwrite,
        continue_on_error=args.continue_on_error,
        limit=args.limit,
        viewer=args.viewer,
        viewer_port=args.viewer_port,
        viewer_loop=args.viewer_loop,
        joint_limit_tolerance=args.joint_limit_tolerance,
    )
    print(
        json.dumps(
            {
                "selected": summary.selected,
                "converted": summary.converted,
                "skipped": summary.skipped,
                "failed": summary.failed,
                "output_motions": summary.output_motions,
                "output_frames": summary.output_frames,
                "output_hours": summary.output_frames / args.target_fps / 3600.0,
                "output_root": str(summary.output_root),
                "manifest": str(summary.manifest_path),
                "displayed_motion": (
                    str(summary.displayed_motion) if summary.displayed_motion is not None else None
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
