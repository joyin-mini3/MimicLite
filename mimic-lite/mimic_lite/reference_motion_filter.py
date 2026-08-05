from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf


SCHEMA_VERSION = 1
PROFILE_NAME = "nominal-v1"
TERMINATION_NAMES = (
    "motion_timeout",
    "root_pos_error",
    "root_ori_error",
    "body_pos_error",
    "body_ori_error",
)
TRACKING_FAILURE_NAMES = TERMINATION_NAMES[1:]
DIAGNOSTIC_COLUMNS = (
    "motion_t",
    "root_pos_error",
    "root_ori_error",
    "body_pos_error_local_max",
    "body_pos_error_local_argmax",
    "body_ori_error_local_max",
    "body_ori_error_local_argmax",
    "joint_pos_error_max",
    "joint_pos_error_argmax",
    "applied_action_abs_max",
    "applied_torque_abs_max",
    "reference_root_pos_x",
    "reference_root_pos_y",
    "reference_root_pos_z",
    "reference_root_quat_w",
    "reference_root_quat_x",
    "reference_root_quat_y",
    "reference_root_quat_z",
    "robot_root_pos_x",
    "robot_root_pos_y",
    "robot_root_pos_z",
    "robot_root_quat_w",
    "robot_root_quat_x",
    "robot_root_quat_y",
    "robot_root_quat_z",
)


@dataclass(frozen=True)
class MotionCatalog:
    dataset_root: Path
    motions_root: Path
    relative_paths: tuple[str, ...]
    lengths: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.relative_paths)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def load_motion_catalog(dataset_root: Path) -> MotionCatalog:
    dataset_root = dataset_root.expanduser().resolve()
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Motion manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    motions_subdir = str(manifest.get("motions_subdir", "motions"))
    motions_root = dataset_root / motions_subdir
    motion_paths = sorted(motions_root.rglob("*.npz"))
    if not motion_paths:
        raise FileNotFoundError(f"No reference motions found under {motions_root}")
    relative_paths = tuple(
        path.relative_to(motions_root).as_posix() for path in motion_paths
    )

    conversion_index_path = dataset_root / "conversion_index.json"
    indexed_lengths: dict[str, int] = {}
    if conversion_index_path.is_file():
        payload = json.loads(conversion_index_path.read_text(encoding="utf-8"))
        entries = payload.get("entries", {})
        if isinstance(entries, dict):
            indexed_lengths = {str(key): int(value) for key, value in entries.items()}

    lengths: list[int] = []
    for path, relative_path in zip(motion_paths, relative_paths, strict=True):
        candidates = (relative_path, f"{motions_subdir}/{relative_path}")
        length = next(
            (indexed_lengths[key] for key in candidates if key in indexed_lengths),
            None,
        )
        if length is None:
            with np.load(path, allow_pickle=False) as motion:
                length = int(motion["qpos"].shape[0])
        if length <= 1:
            raise ValueError(f"Motion must contain at least two frames: {path}")
        lengths.append(length)

    return MotionCatalog(
        dataset_root=dataset_root,
        motions_root=motions_root,
        relative_paths=relative_paths,
        lengths=tuple(lengths),
    )


def classify_termination_flags(flags: Mapping[str, bool]) -> tuple[str, str]:
    for name in TRACKING_FAILURE_NAMES:
        if bool(flags.get(name, False)):
            return "failed", name
    if bool(flags.get("motion_timeout", False)):
        return "passed", "motion_timeout"
    return "runtime_error", "unknown_done"


def _zero_noise_std(node: Any) -> None:
    if isinstance(node, DictConfig):
        for key in list(node.keys()):
            if key == "noise_std":
                node[key] = 0.0
            else:
                _zero_noise_std(node[key])
    elif isinstance(node, (list, ListConfig)):
        for item in node:
            _zero_noise_std(item)


def apply_nominal_profile(
    cfg: DictConfig,
    *,
    checkpoint_path: Path,
    dataset_root: Path,
    num_envs: int,
    window_frames: int,
    max_motion_length: int,
    headless: bool,
) -> DictConfig:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.set_struct(cfg, False)
    cfg.checkpoint_path = str(checkpoint_path.resolve())
    cfg.backend = "mjlab"
    cfg.device = "cuda"
    cfg.headless = bool(headless)
    cfg.app.headless = bool(headless)
    cfg.app.enable_cameras = False
    cfg.seed = 0

    cfg.task.num_envs = int(num_envs)
    cfg.task.max_episode_length = int(max_motion_length) + 2
    cfg.task.terrain = "plane"
    cfg.task.randomization = {}
    cfg.task.command.rewind_prob = 0.0
    cfg.task.command.start_from_zero = True
    cfg.task.command.replay_motion = False
    cfg.task.command.pose_range = {
        name: [0.0, 0.0]
        for name in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    cfg.task.command.velocity_range = {
        name: [0.0, 0.0]
        for name in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    cfg.task.command.init_joint_pos_noise = 0.0
    cfg.task.command.init_joint_vel_noise = 0.0
    cfg.task.command.sequential_eval = True
    cfg.task.command.sequential_window_frames = int(window_frames)
    if len(cfg.task.command.motion_cfgs) != 1:
        raise ValueError("Checkpoint filtering requires exactly one motion dataset")
    dataset_name = next(iter(cfg.task.command.motion_cfgs))
    cfg.task.command.motion_cfgs[dataset_name].path = str(dataset_root.resolve())
    cfg.task.command.motion_cfgs[dataset_name].full_motion = False

    cfg.task.input.action.min_delay = 2
    cfg.task.input.action.max_delay = 2
    cfg.task.input.action.alpha = 0.9
    cfg.task.input.action.alpha_range = [0.9, 0.9]
    _zero_noise_std(cfg.task.observation)
    cfg.task.observation["_filter_diagnostics"] = {
        "tracking": {
            "_target_": "mimic_lite.tracking_filter_diagnostics",
            "root_body_name": str(cfg.task.shared.termination_root_body_name),
        }
    }
    return cfg


def load_result_rows(output_dir: Path) -> dict[int, dict[str, Any]]:
    paths = sorted((output_dir / "shards").glob("*.jsonl"))
    if not paths and (output_dir / "results.jsonl").is_file():
        paths = [output_dir / "results.jsonl"]
    rows: dict[int, dict[str, Any]] = {}
    for path in paths:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            motion_id = int(row["dataset_motion_id"])
            previous = rows.get(motion_id)
            if previous is not None and previous != row:
                raise ValueError(
                    f"Conflicting result for motion {motion_id} in {path}:{line_number}"
                )
            rows[motion_id] = row
    return rows


def write_result_shard(
    output_dir: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    worker_index: int,
) -> Path | None:
    serialized_rows = [
        json.dumps(dict(row), sort_keys=True, separators=(",", ":"))
        for row in rows
    ]
    if not serialized_rows:
        return None
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"worker_{worker_index:04d}_part_"
    existing = sorted(shards_dir.glob(prefix + "*.jsonl"))
    next_index = 0
    if existing:
        next_index = max(int(path.stem.rsplit("_", 1)[1]) for path in existing) + 1
    path = shards_dir / f"{prefix}{next_index:06d}.jsonl"
    _atomic_write_text(path, "\n".join(serialized_rows) + "\n")
    return path


def finalize_results(
    output_dir: Path,
    *,
    catalog: MotionCatalog,
    expected_motion_ids: Iterable[int],
) -> dict[str, Any]:
    expected = sorted(int(motion_id) for motion_id in expected_motion_ids)
    rows_by_id = load_result_rows(output_dir)
    missing = [motion_id for motion_id in expected if motion_id not in rows_by_id]
    if missing:
        raise RuntimeError(
            f"Cannot finalize filter output; {len(missing)} motions are missing, "
            f"first={missing[:8]}"
        )
    rows = [rows_by_id[motion_id] for motion_id in expected]
    for motion_id, row in zip(expected, rows, strict=True):
        expected_path = catalog.relative_paths[motion_id]
        if row["relative_path"] != expected_path:
            raise ValueError(
                f"Result path mismatch for motion {motion_id}: "
                f"expected={expected_path}, actual={row['relative_path']}"
            )

    _atomic_write_text(
        output_dir / "results.jsonl",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )
    statuses = ("passed", "failed", "runtime_error")
    grouped = {
        status: [row for row in rows if row["status"] == status]
        for status in statuses
    }
    for status, filename in (
        ("passed", "passed_motions.txt"),
        ("failed", "failed_motions.txt"),
        ("runtime_error", "runtime_errors.txt"),
    ):
        _atomic_write_text(
            output_dir / filename,
            "".join(f"{row['relative_path']}\n" for row in grouped[status]),
        )

    reason_counts: dict[str, int] = {}
    failure_dir = output_dir / "failure_by_reason"
    failure_dir.mkdir(parents=True, exist_ok=True)
    for reason in TRACKING_FAILURE_NAMES:
        reason_rows = [
            row
            for row in grouped["failed"]
            if reason in row.get("termination_flags", {})
            and bool(row["termination_flags"][reason])
        ]
        reason_counts[reason] = len(reason_rows)
        _atomic_write_text(
            failure_dir / f"{reason}.txt",
            "".join(f"{row['relative_path']}\n" for row in reason_rows),
        )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "evaluated_motions": len(rows),
        "passed": len(grouped["passed"]),
        "failed": len(grouped["failed"]),
        "runtime_error": len(grouped["runtime_error"]),
        "failure_reason_counts": reason_counts,
    }
    atomic_write_json(output_dir / "summary.json", summary)
    return summary
