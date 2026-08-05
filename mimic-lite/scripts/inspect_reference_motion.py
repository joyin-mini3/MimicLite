"""Replay one reference motion from a checkpoint-filter result."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torchrl.envs.utils import ExplorationType, set_exploration_type

import active_adaptation as aa
from active_adaptation.learning.modules.vecnorm import VecNorm
from active_adaptation.utils.timerfd import Timer


# This helper must be available before aa.init(), whereas importing mimic_lite
# itself performs backend-dependent registrations. Load only the pure helper.
_FILTER_LIB_PATH = (
    Path(__file__).resolve().parents[1]
    / "mimic_lite"
    / "reference_motion_filter.py"
)
_FILTER_LIB_SPEC = importlib.util.spec_from_file_location(
    "_mimic_lite_reference_motion_inspector", _FILTER_LIB_PATH
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
classify_termination_flags = _filter_lib.classify_termination_flags
load_motion_catalog = _filter_lib.load_motion_catalog
load_result_rows = _filter_lib.load_result_rows
sha256_file = _filter_lib.sha256_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay one motion using the exact checkpoint and nominal profile "
            "recorded by filter_reference_motions.py."
        )
    )
    parser.add_argument("--filter-output-dir", required=True)
    parser.add_argument(
        "--motion",
        required=True,
        help="Dataset motion ID or relative .npz path from a filter result.",
    )
    parser.add_argument(
        "--realtime",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Throttle simulation to the policy step rate.",
    )
    parser.add_argument(
        "--hold-at-end",
        action="store_true",
        help="Keep the final pre-reset simulator state visible until Enter is pressed.",
    )
    return parser.parse_args()


def _resolve_motion_id(selector: str, relative_paths: tuple[str, ...]) -> int:
    try:
        motion_id = int(selector)
    except ValueError:
        normalized = Path(selector).as_posix()
        if normalized.startswith("motions/"):
            normalized = normalized[len("motions/") :]
        try:
            return relative_paths.index(normalized)
        except ValueError as error:
            raise ValueError(
                f"Motion path is not present in the filter dataset: {selector}"
            ) from error
    if not 0 <= motion_id < len(relative_paths):
        raise ValueError(
            f"Motion ID must be in [0, {len(relative_paths)}), got {motion_id}"
        )
    return motion_id


def _validate_identity(identity: dict[str, Any]) -> tuple[Path, Path, Path]:
    if identity.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            "Unsupported filter schema version: "
            f"{identity.get('schema_version')}"
        )
    if identity.get("profile") != PROFILE_NAME:
        raise ValueError(f"Unsupported filter profile: {identity.get('profile')}")

    checkpoint_path = Path(identity["checkpoint_path"]).expanduser().resolve()
    run_cfg_path = Path(identity["run_cfg_path"]).expanduser().resolve()
    motions_root = Path(identity["motions_root"]).expanduser().resolve()
    for path in (checkpoint_path, run_cfg_path, motions_root / "manifest.json"):
        if not path.exists():
            raise FileNotFoundError(path)
    for path, key in (
        (checkpoint_path, "checkpoint_sha256"),
        (run_cfg_path, "run_cfg_sha256"),
        (motions_root / "manifest.json", "dataset_manifest_sha256"),
    ):
        actual = sha256_file(path)
        if actual != identity[key]:
            raise ValueError(
                f"Artifact changed since filtering: {path}; "
                f"expected sha256={identity[key]}, actual={actual}"
            )
    conversion_index = motions_root / "conversion_index.json"
    expected_conversion_sha = identity.get("conversion_index_sha256")
    if expected_conversion_sha is not None:
        if not conversion_index.is_file():
            raise FileNotFoundError(conversion_index)
        actual = sha256_file(conversion_index)
        if actual != expected_conversion_sha:
            raise ValueError(
                f"Artifact changed since filtering: {conversion_index}; "
                f"expected sha256={expected_conversion_sha}, actual={actual}"
            )
    return checkpoint_path, run_cfg_path, motions_root


def _make_result(td, motion_id: int, relative_path: str) -> dict[str, Any]:
    next_td = td["next"]
    flags = {
        name: bool(next_td["stats", "termination", name][0].item())
        for name in TERMINATION_NAMES
    }
    status, reason = classify_termination_flags(flags)
    values = next_td["_filter_diagnostics"][0].cpu().tolist()
    diagnostics = dict(zip(DIAGNOSTIC_COLUMNS, values, strict=True))
    return {
        "dataset_motion_id": motion_id,
        "relative_path": relative_path,
        "status": status,
        "termination_reason": reason,
        "termination_flags": flags,
        "termination_t": int(round(diagnostics["motion_t"])),
        "max_errors": {
            "root_pos_m": diagnostics["root_pos_error"],
            "root_ori_rad": diagnostics["root_ori_error"],
            "body_pos_local_m": diagnostics["body_pos_error_local_max"],
            "body_ori_local_rad": diagnostics["body_ori_error_local_max"],
            "joint_pos_rad": diagnostics["joint_pos_error_max"],
        },
        "max_abs_applied_action": diagnostics["applied_action_abs_max"],
        "max_abs_applied_torque": diagnostics["applied_torque_abs_max"],
        "root_state": {
            "reference_pos_w": [
                diagnostics[f"reference_root_pos_{axis}"] for axis in "xyz"
            ],
            "reference_quat_wxyz": [
                diagnostics[f"reference_root_quat_{axis}"] for axis in "wxyz"
            ],
            "robot_pos_w": [
                diagnostics[f"robot_root_pos_{axis}"] for axis in "xyz"
            ],
            "robot_quat_wxyz": [
                diagnostics[f"robot_root_quat_{axis}"] for axis in "wxyz"
            ],
        },
    }


@VecNorm.freeze()
def _replay(cfg, motion_id: int, relative_path: str, *, realtime: bool):
    from active_adaptation.helpers import make_env_policy
    from any4hdmi import SequentialWindowedMotionDataset

    env, policy = make_env_policy(cfg)
    try:
        env.base_env.eval()
        dataset = env.base_env.command_manager.dataset
        if not isinstance(dataset, SequentialWindowedMotionDataset):
            raise TypeError(
                "Inspector expected SequentialWindowedMotionDataset, got "
                f"{type(dataset).__name__}"
            )
        checkpoint = torch.load(cfg.checkpoint_path, weights_only=False)
        failed_modules = policy.load_state_dict(checkpoint["policy"])
        if failed_modules:
            raise RuntimeError(
                "Checkpoint did not load all policy modules: "
                f"{failed_modules}"
            )

        dataset.set_motion_queue([motion_id])
        rollout_policy = policy.get_rollout_policy("eval")
        carry = env.reset()
        timer = Timer(env.step_dt)
        with torch.inference_mode(), set_exploration_type(
            ExplorationType.DETERMINISTIC
        ):
            while True:
                carry = rollout_policy(carry)
                td = env.step(carry)
                if bool(td["next", "done"][0].item()):
                    return _make_result(td, motion_id, relative_path), env

                # Match TorchRL's step_and_maybe_reset carry path, but stop at
                # done so the viewer retains the failure frame instead of the
                # automatically reset state.
                carry = env._step_mdp(td)
                if env._post_step_mdp_hooks is not None:
                    td, carry = env._post_step_mdp_hooks(td, carry)
                if realtime:
                    timer.sleep()
    except BaseException:
        env.close()
        raise


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.filter_output_dir).expanduser().resolve()
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = manifest["identity"]
    checkpoint_path, run_cfg_path, motions_root = _validate_identity(identity)
    catalog = load_motion_catalog(motions_root)
    motion_id = _resolve_motion_id(args.motion, catalog.relative_paths)
    relative_path = catalog.relative_paths[motion_id]

    cfg = apply_nominal_profile(
        OmegaConf.load(run_cfg_path),
        checkpoint_path=checkpoint_path,
        dataset_root=motions_root,
        num_envs=1,
        window_frames=int(identity["window_frames"]),
        max_motion_length=catalog.lengths[motion_id],
        headless=False,
    )
    aa.init(cfg, auto_rank=False)
    result, env = _replay(
        cfg,
        motion_id,
        relative_path,
        realtime=args.realtime,
    )
    try:
        recorded = load_result_rows(output_dir).get(motion_id)
        comparison = None
        if recorded is not None:
            comparison = {
                "recorded_status": recorded["status"],
                "recorded_reason": recorded["termination_reason"],
                "recorded_t": recorded["termination_t"],
                "replay_matches": (
                    recorded["status"] == result["status"]
                    and recorded["termination_reason"]
                    == result["termination_reason"]
                    and recorded["termination_t"] == result["termination_t"]
                ),
            }
        print(json.dumps({"replay": result, "comparison": comparison}, indent=2))
        if args.hold_at_end:
            input("Final pre-reset state is being held; press Enter to close... ")
    finally:
        env.close()


if __name__ == "__main__":
    main()
