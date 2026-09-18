"""Load Mini3 demonstrations with OpenHLM without changing the OpenHLM checkout.

The 36 channels are Mini3 joint/gripper values plus reference-root orientation,
heading-frame linear velocity and height. They are not the G1 34-channel contract.
This launcher registers a Mini3 configuration and can run OpenHLM's own data
validation, normalization-statistics entry point, or PyTorch training entry point.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import runpy
import sys
from typing import Any

import numpy as np

from mini3_vla_features import FEATURE_NAMES, FEATURE_UNITS


ROOT = Path(__file__).resolve().parent
DEFAULT_OPENHLM = ROOT.parent / "OpenHLM"
DEFAULT_LEROBOT_HOME = ROOT / "outputs/mini3_openhlm_recording/lerobot"
DEFAULT_ARTIFACTS_ROOT = ROOT / "outputs/mini3_openhlm_recording"
CONFIG_NAME = "mini3_pick_carry"
REPO_ID = "local/mini3_pick_carry"
CHANNELS = len(FEATURE_NAMES)
CAMERA_KEYS = ("head_image_left", "left_wrist_image", "right_wrist_image")
MODEL_CAMERA_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def prepare_imports(openhlm: Path, dataset_home: Path, *, validation: bool = False) -> Path:
    """Set paths before LeRobot resolves its module-level dataset directory."""
    component = openhlm.resolve() / "src/openpi4OpenHLM"
    if not (component / "src/openpi/training/data_loader.py").is_file():
        raise FileNotFoundError(f"OpenHLM training checkout not found: {component}")
    for path in reversed((component / "src", component / "packages/openpi-client/src")):
        sys.path.insert(0, str(path))
    overlay = ROOT / ".cache/mini3_vla_deps"
    if overlay.is_dir() and sys.executable.startswith(str(ROOT / "active-adaptation/venv")):
        # The optional verification overlay belongs to the local simulation
        # Python. A dedicated OpenHLM training environment keeps its own packages.
        sys.path.insert(0, str(overlay))
    if "LEROBOT_HOME" in os.environ:
        raise ValueError("Use HF_LEROBOT_HOME, not the deprecated LEROBOT_HOME")
    os.environ["HF_LEROBOT_HOME"] = str(dataset_home.resolve())
    os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / ".cache/mini3_vla_hf_datasets"))
    os.environ.setdefault("OPENPI_DATA_HOME", str(ROOT / ".cache/mini3_openpi"))
    if validation:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return component


def _image(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3:
        raise ValueError(f"Expected one RGB image, got {array.shape}")
    if array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] != 3 or not np.isfinite(array).all():
        raise ValueError(f"Expected finite RGB image, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        if array.min() < 0 or array.max() > 1:
            raise ValueError("Loader image floats must be in [0, 1]")
        array = np.rint(array * 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        raise ValueError(f"RGB images must be uint8 or float [0,1], got {array.dtype}")
    return array


@dataclasses.dataclass(frozen=True)
class Mini3Inputs:
    """Convert actual Mini3 fields without applying G1 joint or gripper rules."""

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data.get("state", data.get("observation/state")), dtype=np.float32)
        if state.shape != (CHANNELS,) or not np.isfinite(state).all():
            raise ValueError(f"Mini3 state must be finite with shape ({CHANNELS},), got {state.shape}")
        images = {}
        for source, target in zip(CAMERA_KEYS, MODEL_CAMERA_KEYS, strict=True):
            key = source if source in data else f"observation/{source}"
            images[target] = _image(data[key])
        result = {
            "state": state,
            "image": images,
            "image_mask": {key: np.bool_(True) for key in MODEL_CAMERA_KEYS},
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[-1] != CHANNELS or not np.isfinite(actions).all():
                raise ValueError(f"Mini3 action chunks must have shape (horizon, {CHANNELS}), got {actions.shape}")
            result["actions"] = actions
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class Mini3Outputs:
    """Keep all Mini3 action channels, including root velocity and height."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.ndim != 2 or actions.shape[-1] < CHANNELS:
            raise ValueError(f"Expected at least {CHANNELS} Mini3 output channels, got {actions.shape}")
        return {"actions": actions[:, :CHANNELS]}


@dataclasses.dataclass(frozen=True)
class Mini3Transforms:
    def __call__(self, model_config: Any) -> Any:
        from openpi import transforms

        if model_config.action_dim != CHANNELS:
            raise ValueError(f"Mini3 configuration requires action_dim={CHANNELS}")
        return transforms.Group(inputs=(Mini3Inputs(),), outputs=(Mini3Outputs(),))


@dataclasses.dataclass(frozen=True)
class PreTokenTransforms:
    """The real image/padding transforms; no tokenizer or model download."""

    def __call__(self, model_config: Any) -> Any:
        from openpi import transforms

        return transforms.Group(inputs=(transforms.ResizeImages(224, 224),
                                        transforms.PadStatesAndActions(model_config.action_dim)))


def register_config(*, repo_id: str = REPO_ID, mode: str = "train", batch_size: int = 32,
                    workers: int = 0, weights: Path | None = None,
                    artifacts_root: Path = DEFAULT_ARTIFACTS_ROOT) -> Any:
    from openpi.training import config

    base = config.get_config("openhlm_example")
    factory = config.SimpleDataConfig(
        repo_id=repo_id,
        base_config=config.DataConfig(prompt_from_task=True, action_sequence_keys=("actions",)),
        data_transforms=Mini3Transforms(),
        model_transforms=(config.ModelTransformFactory() if mode == "train" else PreTokenTransforms()),
    )
    model = dataclasses.replace(base.model, action_dim=CHANNELS, action_horizon=50)
    train_config = dataclasses.replace(
        base, name=CONFIG_NAME, model=model, data=factory,
        batch_size=batch_size, num_workers=workers, wandb_enabled=False,
        assets_base_dir=str(artifacts_root.resolve() / "openpi_assets"),
        checkpoint_base_dir=str(artifacts_root.resolve() / "checkpoints"),
        pytorch_weight_path=(str(weights.expanduser().resolve()) if weights is not None
                             else str(Path(base.pytorch_weight_path).expanduser())),
        policy_metadata={"robot_type": "mini3", "action_dim": CHANNELS, "fps": 50,
                         "action_semantics": "MimicLite reference plus added-joint and gripper commands",
                         "feature_names": list(FEATURE_NAMES), "feature_units": list(FEATURE_UNITS),
                         "absolute_actions": True},
    )
    config._CONFIGS_DICT[CONFIG_NAME] = train_config
    return train_config


def validate_dataset(train_config: Any, report_path: Path | None = None) -> dict:
    """Exercise the genuine OpenHLM loader and preprocessing for every episode."""
    import torch
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from openpi import transforms
    from openpi.training import data_loader

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    meta = LeRobotDatasetMetadata(data_config.repo_id)
    if meta.fps != 50:
        raise ValueError(f"Expected native 50 Hz Mini3 data, got {meta.fps}")
    for key in ("state", "actions"):
        if tuple(meta.features[key]["shape"]) != (CHANNELS,):
            raise ValueError(f"Unexpected {key} schema: {meta.features[key]}")
        if meta.features[key].get("names") != list(FEATURE_NAMES):
            raise ValueError(f"{key} must declare the actual Mini3 channel names and ordering")
    raw = data_loader.create_torch_dataset(data_config, 50, train_config.model)
    prepared = data_loader.transform_dataset(raw, data_config, skip_norm_stats=True)
    normalized = (data_loader.transform_dataset(raw, data_config)
                  if data_config.norm_stats is not None else None)
    unnormalize = transforms.Unnormalize(data_config.norm_stats,
                                        use_quantiles=data_config.use_quantile_norm)
    normalized_max_abs = {"state": 0.0, "actions": 0.0}
    allowed_prompts = {
        f"Please put the {color} square from the tabletop into the basket."
        for color in ("red", "green", "blue")
    }
    checked = 0
    offset = 0
    tasks = set()
    for episode_index, episode in meta.episodes.items():
        length = int(episode["length"])
        if length <= 0:
            raise ValueError(f"Episode {episode_index} is empty")
        for index in sorted({offset, offset + (length - 1) // 2, offset + length - 1}):
            source = raw[index]
            sample = prepared[index]
            if int(source["episode_index"]) != episode_index:
                raise ValueError("Episode indices do not match episode lengths")
            if sample["prompt"] not in allowed_prompts:
                raise ValueError(f"Unexpected task instruction: {sample['prompt']}")
            tasks.add(sample["prompt"])
            if sample["actions"].shape != (50, CHANNELS) or sample["state"].shape != (CHANNELS,):
                raise ValueError("OpenHLM transformed Mini3 dimensions changed")
            for key in MODEL_CAMERA_KEYS:
                if sample["image"][key].shape != (224, 224, 3):
                    raise ValueError(f"Invalid transformed camera {key}")
            if index == offset + length - 1:
                padding = source["actions_is_pad"]
                if bool(padding[0]) or not bool(padding[1:].all()):
                    raise ValueError("Action chunks cross an episode boundary")
                if not torch.equal(source["actions"], source["actions"][0].expand_as(source["actions"])):
                    raise ValueError("Final action padding does not repeat the final action")
            if not np.array_equal(Mini3Outputs()({"actions": sample["actions"]})["actions"],
                                  sample["actions"]):
                raise ValueError("Output adapter changed Mini3 action channels")
            if normalized is not None:
                normalized_sample = normalized[index]
                restored = unnormalize({key: normalized_sample[key] for key in ("state", "actions")})
                for key in ("state", "actions"):
                    values = normalized_sample[key]
                    if values.shape != sample[key].shape or not np.isfinite(values).all():
                        raise ValueError(f"Invalid normalized Mini3 {key}")
                    normalized_max_abs[key] = max(normalized_max_abs[key], float(np.abs(values).max()))
                    np.testing.assert_allclose(restored[key], sample[key], atol=1e-5, rtol=1e-5)
            checked += 1
        offset += length
    if len(raw) != offset or meta.total_frames != offset:
        raise ValueError("Episode lengths and stored frame count differ")
    result = {
        "success": True, "repo_id": data_config.repo_id, "episodes": meta.total_episodes,
        "frames": len(raw), "fps": meta.fps, "state_dim": CHANNELS, "action_dim": CHANNELS,
        "action_horizon": 50, "samples_checked": checked, "tasks": sorted(tasks),
        "loader": "OpenHLM create_torch_dataset + transform_dataset(skip_norm_stats=True)",
        "tokenizer_checked": False, "normalization_applied": normalized is not None,
        "normalization_samples_checked": checked if normalized is not None else 0,
        "normalization_roundtrip_checked": normalized is not None,
        "normalized_max_abs": normalized_max_abs if normalized is not None else None,
        "model_weights_loaded": False,
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("validate", "norm", "train"))
    parser.add_argument("--openhlm", type=Path, default=DEFAULT_OPENHLM)
    parser.add_argument("--dataset-home", type=Path, default=DEFAULT_LEROBOT_HOME)
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT,
                        help="Directory for this dataset's normalization assets and checkpoints")
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--report", type=Path)
    args, forwarded = parser.parse_known_args()
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    if args.batch_size <= 0 or args.workers < 0:
        parser.error("batch-size must be positive and workers nonnegative")
    if args.mode == "validate" and forwarded:
        parser.error(f"Unexpected validation arguments: {forwarded}")
    component = prepare_imports(args.openhlm, args.dataset_home, validation=args.mode == "validate")
    train_config = register_config(repo_id=args.repo_id, mode=args.mode, batch_size=args.batch_size,
                                   workers=args.workers, weights=args.weights,
                                   artifacts_root=args.artifacts_root)
    if args.mode == "validate":
        print(json.dumps(validate_dataset(train_config, args.report), indent=2))
        return 0
    if args.mode == "norm":
        script = component / "scripts/compute_norm_stats.py"
        sys.argv = [str(script), "--config-name", CONFIG_NAME, *forwarded]
    else:
        script = component / "scripts/train_pytorch.py"
        sys.argv = [str(script), CONFIG_NAME, *forwarded]
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    # OpenHLM replaces __main__ via runpy. Keep our transform classes in an
    # importable module so spawned DataLoader workers can unpickle them.
    from mini3_openhlm_training import main as module_main
    raise SystemExit(module_main())
