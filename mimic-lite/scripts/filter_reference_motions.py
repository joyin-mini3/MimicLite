"""Filter complete reference motions with a trained MimicLite checkpoint."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from tensordict import TensorDictBase
from torchrl.envs.utils import ExplorationType, set_exploration_type
from tqdm import tqdm

import active_adaptation as aa
from active_adaptation.learning.modules.vecnorm import VecNorm


# Import the pure artifact/config helper without importing mimic_lite.__init__.
# The project package performs backend-dependent registrations and therefore can
# only be imported after aa.init(), while this helper is also needed beforehand.
_FILTER_LIB_PATH = (
    Path(__file__).resolve().parents[1]
    / "mimic_lite"
    / "reference_motion_filter.py"
)
_FILTER_LIB_SPEC = importlib.util.spec_from_file_location(
    "_mimic_lite_reference_motion_filter", _FILTER_LIB_PATH
)
if _FILTER_LIB_SPEC is None or _FILTER_LIB_SPEC.loader is None:
    raise RuntimeError(f"Failed to load filter helper from {_FILTER_LIB_PATH}")
_filter_lib = importlib.util.module_from_spec(_FILTER_LIB_SPEC)
sys.modules[_FILTER_LIB_SPEC.name] = _filter_lib
_FILTER_LIB_SPEC.loader.exec_module(_filter_lib)

DIAGNOSTIC_COLUMNS = _filter_lib.DIAGNOSTIC_COLUMNS
PROFILE_NAME = _filter_lib.PROFILE_NAME
SCHEMA_VERSION = _filter_lib.SCHEMA_VERSION
TERMINATION_NAMES = _filter_lib.TERMINATION_NAMES
apply_nominal_profile = _filter_lib.apply_nominal_profile
atomic_write_json = _filter_lib.atomic_write_json
classify_termination_flags = _filter_lib.classify_termination_flags
finalize_results = _filter_lib.finalize_results
load_motion_catalog = _filter_lib.load_motion_catalog
load_result_rows = _filter_lib.load_result_rows
sha256_file = _filter_lib.sha256_file
write_result_shard = _filter_lib.write_result_shard


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every complete reference motion with a MimicLite checkpoint "
            "and save motions that trigger training tracking terminations."
        )
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--run-cfg-path", required=True)
    parser.add_argument("--motions-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--window-frames", type=int, default=512)
    parser.add_argument("--flush-every", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Merge existing shards for the complete dataset without running simulation.",
    )
    return parser.parse_args()


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _optional_sha256(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def _git_commit() -> str | None:
    repository_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _scope_motion_ids(total: int, args: argparse.Namespace) -> list[int]:
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    start = total * args.shard_index // args.num_shards
    end = total * (args.shard_index + 1) // args.num_shards
    motion_ids = list(range(start, end))
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        motion_ids = motion_ids[: args.limit]
    return motion_ids


def _manifest_identity(
    *,
    checkpoint_path: Path,
    run_cfg_path: Path,
    motions_root: Path,
    window_frames: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "profile": PROFILE_NAME,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "run_cfg_path": str(run_cfg_path),
        "run_cfg_sha256": sha256_file(run_cfg_path),
        "motions_root": str(motions_root),
        "dataset_manifest_sha256": sha256_file(motions_root / "manifest.json"),
        "conversion_index_sha256": _optional_sha256(
            motions_root / "conversion_index.json"
        ),
        "code_commit": _git_commit(),
        "window_frames": int(window_frames),
        "nominal_overlay": {
            "deterministic_policy": True,
            "observation_noise": 0.0,
            "domain_randomization": False,
            "rewind_prob": 0.0,
            "start_t": 1,
            "action_delay": 2,
            "action_alpha": 0.9,
        },
    }


def _prepare_manifest(
    output_dir: Path,
    *,
    identity: dict[str, Any],
    args: argparse.Namespace,
    total_motions: int,
) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("identity") != identity:
            raise ValueError(
                "Existing filter output identity does not match this run; "
                "use a different --output-dir"
            )
        return manifest
    manifest = {
        "identity": identity,
        "status": "running",
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "command": list(sys.argv),
        "dataset_motions": int(total_motions),
        "sharding": {
            "num_shards": int(args.num_shards),
        },
        "workers": {
            str(args.shard_index): {
                "num_shards": int(args.num_shards),
                "num_envs": int(args.num_envs),
            }
        },
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def _tensor_at(td: TensorDictBase, key: Any) -> torch.Tensor | None:
    try:
        value = td.get(key)
    except (KeyError, ValueError):
        return None
    return value if isinstance(value, torch.Tensor) else None


def _max_abs_at(
    tensor: torch.Tensor | None, env_index: int
) -> float | None:
    if tensor is None:
        return None
    value = tensor[env_index]
    if value.numel() == 0:
        return None
    return float(value.detach().abs().max().cpu().item())


def _observation_width(group: Any) -> int:
    if isinstance(group, torch.Tensor):
        return int(group[0].numel())
    if isinstance(group, TensorDictBase):
        return sum(
            int(value[0].numel())
            for value in group.values(True, True)
            if isinstance(value, torch.Tensor)
        )
    raise TypeError(f"Unsupported observation group type: {type(group).__name__}")


def _record_finished_rows(
    *,
    td: TensorDictBase,
    finished_env_ids: torch.Tensor,
    source_ids_before: torch.Tensor,
    catalog,
    tracking_body_names: list[str],
    tracking_joint_names: list[str],
) -> list[dict[str, Any]]:
    next_td = td["next"]
    diagnostics = next_td["_filter_diagnostics"]
    policy_action = _tensor_at(td, "action")
    rows: list[dict[str, Any]] = []

    for env_index in finished_env_ids.detach().cpu().tolist():
        source_id = int(source_ids_before[env_index].detach().cpu().item())
        flags = {
            name: bool(
                next_td["stats", "termination", name][env_index]
                .reshape(-1)[0]
                .detach()
                .cpu()
                .item()
            )
            for name in TERMINATION_NAMES
        }
        status, reason = classify_termination_flags(flags)
        values = diagnostics[env_index].detach().cpu().tolist()
        diagnostic = dict(zip(DIAGNOSTIC_COLUMNS, values, strict=True))
        motion_t = int(round(float(diagnostic["motion_t"])))
        motion_len = int(catalog.lengths[source_id])
        body_pos_index = int(round(diagnostic["body_pos_error_local_argmax"]))
        body_ori_index = int(round(diagnostic["body_ori_error_local_argmax"]))
        joint_pos_index = int(round(diagnostic["joint_pos_error_argmax"]))
        progress = min(
            1.0,
            max(0.0, float(motion_t) / float(max(1, motion_len - 1))),
        )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset_motion_id": source_id,
                "relative_path": catalog.relative_paths[source_id],
                "motion_len": motion_len,
                "status": status,
                "termination_reason": reason,
                "termination_flags": flags,
                "termination_t": motion_t,
                "progress": progress,
                "max_errors": {
                    "root_pos_m": float(diagnostic["root_pos_error"]),
                    "root_ori_rad": float(diagnostic["root_ori_error"]),
                    "body_pos_local_m": float(
                        diagnostic["body_pos_error_local_max"]
                    ),
                    "body_ori_local_rad": float(
                        diagnostic["body_ori_error_local_max"]
                    ),
                    "joint_pos_rad": float(diagnostic["joint_pos_error_max"]),
                },
                "root_state": {
                    "reference_pos_w": [
                        float(diagnostic[f"reference_root_pos_{axis}"])
                        for axis in "xyz"
                    ],
                    "reference_quat_wxyz": [
                        float(diagnostic[f"reference_root_quat_{axis}"])
                        for axis in "wxyz"
                    ],
                    "robot_pos_w": [
                        float(diagnostic[f"robot_root_pos_{axis}"])
                        for axis in "xyz"
                    ],
                    "robot_quat_wxyz": [
                        float(diagnostic[f"robot_root_quat_{axis}"])
                        for axis in "wxyz"
                    ],
                },
                "max_body_pos_error_name": tracking_body_names[body_pos_index],
                "max_body_ori_error_name": tracking_body_names[body_ori_index],
                "max_joint_pos_error_name": tracking_joint_names[joint_pos_index],
                "max_abs_policy_action": _max_abs_at(policy_action, env_index),
                "max_abs_applied_action": float(
                    diagnostic["applied_action_abs_max"]
                ),
                "max_abs_applied_torque": float(
                    diagnostic["applied_torque_abs_max"]
                ),
            }
        )
    return rows


@VecNorm.freeze()
def _run_filter(
    *,
    cfg,
    catalog,
    pending_motion_ids: list[int],
    output_dir: Path,
    args: argparse.Namespace,
) -> int:
    from active_adaptation.helpers import make_env_policy
    from any4hdmi import SequentialWindowedMotionDataset

    env, policy = make_env_policy(cfg)
    base_env = env.base_env
    base_env.eval()
    command = base_env.command_manager
    dataset = command.dataset
    if not isinstance(dataset, SequentialWindowedMotionDataset):
        raise TypeError(
            "Filter expected SequentialWindowedMotionDataset, got "
            f"{type(dataset).__name__}"
        )
    actual_paths = tuple(
        path.resolve().relative_to(catalog.motions_root).as_posix()
        for path in dataset.motion_paths
    )
    if actual_paths != catalog.relative_paths:
        raise ValueError("Runtime dataset motion order differs from filter catalog")

    checkpoint = torch.load(cfg.checkpoint_path, weights_only=False)
    failed_modules = policy.load_state_dict(checkpoint["policy"])
    if failed_modules:
        raise RuntimeError(
            "Checkpoint did not load all policy modules: "
            f"{failed_modules}. Verify the saved run cfg is being used."
        )

    dataset.set_motion_queue(pending_motion_ids)
    rollout_policy = policy.get_rollout_policy("eval")
    carry = env.reset()
    observed_widths = {
        str(key): _observation_width(carry[key]) for key in cfg.algo.in_keys
    }
    print("Checkpoint observation widths:", observed_widths, flush=True)

    result_buffer: list[dict[str, Any]] = []
    completed = 0
    steps_since_completion = 0
    watchdog_steps = max(catalog.lengths[motion_id] for motion_id in pending_motion_ids) + 4
    progress = tqdm(
        total=len(pending_motion_ids),
        desc="Filtering reference motions",
        unit="motion",
    )
    try:
        with torch.inference_mode(), set_exploration_type(
            ExplorationType.DETERMINISTIC
        ):
            while completed < len(pending_motion_ids):
                source_ids_before = dataset.env_source_motion_ids.detach().clone()
                active_before = dataset.env_active.detach().clone()
                carry = rollout_policy(carry)
                td, carry = env.step_and_maybe_reset(carry)
                done = td["next", "done"].squeeze(-1)
                finished = done & active_before
                finished_env_ids = torch.nonzero(
                    finished, as_tuple=False
                ).squeeze(-1)
                if finished_env_ids.numel():
                    rows = _record_finished_rows(
                        td=td,
                        finished_env_ids=finished_env_ids,
                        source_ids_before=source_ids_before,
                        catalog=catalog,
                        tracking_body_names=list(command.tracking_body_names),
                        tracking_joint_names=list(command.tracking_joint_names),
                    )
                    result_buffer.extend(rows)
                    completed += len(rows)
                    progress.update(len(rows))
                    steps_since_completion = 0
                    if len(result_buffer) >= args.flush_every:
                        write_result_shard(
                            output_dir,
                            result_buffer,
                            worker_index=args.shard_index,
                        )
                        result_buffer.clear()
                else:
                    steps_since_completion += 1
                    if steps_since_completion > watchdog_steps:
                        raise RuntimeError(
                            "No active motion completed within the longest motion "
                            "watchdog; the termination or streaming runtime is stuck"
                        )
    finally:
        progress.close()
        if result_buffer:
            write_result_shard(
                output_dir,
                result_buffer,
                worker_index=args.shard_index,
            )
        env.close()
    return completed


def main() -> None:
    args = _parse_args()
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    run_cfg_path = Path(args.run_cfg_path).expanduser().resolve()
    motions_root = Path(args.motions_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not run_cfg_path.is_file():
        raise FileNotFoundError(run_cfg_path)
    if args.num_envs <= 0 or args.window_frames <= 1 or args.flush_every <= 0:
        raise ValueError("num-envs/flush-every must be positive and window-frames > 1")

    catalog = load_motion_catalog(motions_root)
    all_motion_ids = list(range(len(catalog)))
    if args.merge_only:
        manifest_path = output_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        identity = manifest.get("identity", {})
        expected_identity = _manifest_identity(
            checkpoint_path=checkpoint_path,
            run_cfg_path=run_cfg_path,
            motions_root=motions_root,
            window_frames=int(identity.get("window_frames", args.window_frames)),
        )
        if identity != expected_identity:
            raise ValueError(
                "Filter output identity does not match the merge inputs"
            )
        summary = finalize_results(
            output_dir,
            catalog=catalog,
            expected_motion_ids=all_motion_ids,
        )
        manifest["status"] = "complete"
        manifest["summary"] = summary
        manifest["updated_at"] = _utc_now()
        manifest["completed_at"] = manifest["updated_at"]
        atomic_write_json(manifest_path, manifest)
        print(json.dumps(summary, indent=2))
        return

    scope_motion_ids = _scope_motion_ids(len(catalog), args)
    if not scope_motion_ids:
        raise RuntimeError("Selected filter shard contains no motions")
    identity = _manifest_identity(
        checkpoint_path=checkpoint_path,
        run_cfg_path=run_cfg_path,
        motions_root=motions_root,
        window_frames=args.window_frames,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _prepare_manifest(
        output_dir,
        identity=identity,
        args=args,
        total_motions=len(catalog),
    )
    manifest["status"] = "running"
    manifest["updated_at"] = _utc_now()
    manifest.setdefault("workers", {})[str(args.shard_index)] = {
        "num_shards": int(args.num_shards),
        "num_envs": int(args.num_envs),
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    existing = load_result_rows(output_dir)
    existing_in_scope = {
        motion_id for motion_id in scope_motion_ids if motion_id in existing
    }
    if existing_in_scope and not args.resume:
        raise FileExistsError(
            f"Output already contains {len(existing_in_scope)} selected motions; "
            "pass --resume or use a new output directory"
        )
    pending = [
        motion_id for motion_id in scope_motion_ids if motion_id not in existing_in_scope
    ]
    if pending:
        effective_num_envs = min(args.num_envs, len(pending))
        cfg = OmegaConf.load(run_cfg_path)
        cfg = apply_nominal_profile(
            cfg,
            checkpoint_path=checkpoint_path,
            dataset_root=motions_root,
            num_envs=effective_num_envs,
            window_frames=args.window_frames,
            max_motion_length=max(catalog.lengths[motion_id] for motion_id in scope_motion_ids),
            headless=args.headless,
        )
        OmegaConf.save(cfg, output_dir / f"resolved_eval_cfg_worker_{args.shard_index:04d}.yaml")
        aa.init(cfg, auto_rank=False)
        completed = _run_filter(
            cfg=cfg,
            catalog=catalog,
            pending_motion_ids=pending,
            output_dir=output_dir,
            args=args,
        )
        if completed != len(pending):
            raise RuntimeError(
                f"Filter completed {completed} new motions, expected {len(pending)}"
            )

    if args.num_shards == 1:
        summary = finalize_results(
            output_dir,
            catalog=catalog,
            expected_motion_ids=scope_motion_ids,
        )
        manifest["status"] = "complete"
        manifest["summary"] = summary
        manifest["completed_at"] = _utc_now()
    else:
        worker_rows = load_result_rows(output_dir)
        complete = all(motion_id in worker_rows for motion_id in scope_motion_ids)
        summary = {
            "worker": args.shard_index,
            "selected": len(scope_motion_ids),
            "complete": complete,
        }
        atomic_write_json(
            output_dir / f"summary_worker_{args.shard_index:04d}.json", summary
        )
    manifest["updated_at"] = _utc_now()
    atomic_write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
