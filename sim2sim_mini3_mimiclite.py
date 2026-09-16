#!/usr/bin/env python3
r"""Run a Mini3 MimicLite tracking policy in the native MuJoCo desktop viewer.

Examples (run with the active-adaptation MJLab Python environment):
    python sim2sim_mini3_mimiclite.py --model beyondmimic
    python sim2sim_mini3_mimiclite.py --model sonic --headless --duration 10
    python sim2sim_mini3_mimiclite.py --load_model checkpoint_2400.pt \
        --config config.yaml --motion /path/to/dataset/motions/clip.npz
    python sim2sim_mini3_mimiclite.py --load_model policy.onnx --motion clip.npz

Unlike the AMP example, tracking commands come from a reference motion. The
existing MimicLite exporter and sim2real observation/motor implementations are
reused. Export, ONNX inference and MuJoCo physics run on CPU using the installed
MJLab Python dependencies; a functioning GPU is not required. Only load trusted
training checkpoints, as the project's exporter uses PyTorch pickle loading.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".cache" / "mimiclite_sim2sim"
BUNDLE = ROOT / "outputs" / "server_4090_4_review_20260910"
PRESETS = {
    "sonic": ("checkpoint_5200.pt", "sonic", "230210/jog_ff_stop_315_003__A179_M.npz"),
    "beyondmimic": ("checkpoint_2400.pt", "beyondmimic", "woody_chr01_zmp.npz"),
    "beyondmimic_fast": ("checkpoint_1000.pt", "beyondmimic", "woody_chr01_zmp.npz"),
}


def setup_paths() -> None:
    paths = [ROOT / "sim2real", ROOT / "mimic-lite", ROOT / "active-adaptation", CACHE / "deps"]
    for path in reversed(paths):
        sys.path.insert(0, str(path))
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(path) for path in paths] + [os.environ.get("PYTHONPATH", "")]
    )
    os.environ.setdefault("ANY4HDMI_CACHE_BUILD_DEVICE", "cpu")
    os.environ.setdefault("ANY4HDMI_CACHE_BUILD_NUM_WORKERS", "0")
    os.environ.setdefault("WARP_CACHE_PATH", str(ROOT / ".cache" / "warp" / "1.12.0"))
    os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
    # This entry point only uses CPU inference/export and native MuJoCo physics.
    # Avoid optional CUDA probing in imported training utilities.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def existing_file(path: str | Path) -> Path:
    result = Path(path).expanduser().resolve()
    if not result.is_file():
        raise ValueError(f"File does not exist: {result}")
    return result


def motion_dataset(motion: Path) -> tuple[Path, str]:
    """Resolve a selected clip without loading the rest of a large dataset."""
    for directory in motion.parents:
        manifest_path = directory / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            motions_dir = directory / manifest.get("motions_subdir", "motions")
            try:
                return directory, motion.relative_to(motions_dir).as_posix()
            except ValueError as exc:
                raise ValueError(f"Motion must be inside {motions_dir}") from exc
    raise ValueError(f"No any4hdmi manifest.json above {motion}")


def training_config(config: Path, checkpoint: Path, motion: Path, seed: int):
    # Import registers the 'eval'/'frac' OmegaConf resolvers used by training.
    import active_adaptation  # noqa: F401
    from omegaconf import DictConfig, OmegaConf

    cfg = OmegaConf.load(config)
    cfg.pop("hydra", None)
    if "task" not in cfg or "algo" not in cfg:
        raise ValueError("--config must be the full training config, not a deploy YAML")
    if OmegaConf.select(cfg, "task.robot.name") != "mini3-mesh":
        raise ValueError("This entry point supports the mini3-mesh tracking task")
    dataset_root, filename = motion_dataset(motion)
    motion_cfgs = cfg.task.command.motion_cfgs
    if not isinstance(motion_cfgs, DictConfig) or not motion_cfgs:
        raise ValueError("Expected a named task.command.motion_cfgs mapping")
    template = OmegaConf.to_container(next(iter(motion_cfgs.values())))
    cfg.task.command.motion_cfgs = {
        "local_test": {
            **template,
            "path": str(dataset_root),
            "filenames": [filename],
            "weight": 1.0,
            "full_motion": bool(template.get("full_motion", True)),
        }
    }
    cfg.task.command.start_from_zero = True
    cfg.task.num_envs = 1
    cfg.update(checkpoint_path=str(checkpoint), seed=seed, headless=True, backend="mjlab", device="cuda")
    cfg.wandb.mode = "disabled"
    OmegaConf.resolve(cfg)
    return cfg


def cache_key(checkpoint: Path, config_text: str) -> str:
    digest = hashlib.sha256(config_text.encode())
    with checkpoint.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    # Invalidate exports when local policy/normalization/export/asset code changes.
    source_roots = [ROOT / "mimic-lite", ROOT / "active-adaptation" / "active_adaptation"]
    for directory in source_roots:
        for path in sorted(directory.rglob("*.py")):
            if any(part.startswith(".") for part in path.relative_to(directory).parts):
                continue
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
    digest.update((ROOT / "mimic_lite_cpu_export.py").read_bytes())
    return digest.hexdigest()[:20]


def export_worker(directory: Path) -> None:
    """Keep checkpoint reconstruction/export isolated from the playback process."""
    from omegaconf import OmegaConf
    from mimic_lite_cpu_export import export_checkpoint_cpu

    cfg = OmegaConf.load(directory / "training.yaml")
    onnx_path = export_checkpoint_cpu(cfg, directory)
    if not onnx_path.with_suffix(".yaml").is_file():
        raise RuntimeError("Exporter did not produce the companion deploy YAML")
    (directory / "ready.json").write_text(json.dumps({
        "policy": str(onnx_path.relative_to(directory)),
        "env_dt": float(cfg.task.sim.step_dt),
        "sim_dt": float(cfg.task.sim.mujoco_physics_dt),
    }, indent=2) + "\n")


def prepare_export(checkpoint: Path, config: Path, motion: Path, args) -> tuple[Path, float, float]:
    from omegaconf import OmegaConf

    cfg = training_config(config, checkpoint, motion, args.seed)
    config_text = OmegaConf.to_yaml(cfg, resolve=True)
    directory = CACHE / cache_key(checkpoint, config_text)
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "ready.json"
    if args.reexport:
        marker.unlink(missing_ok=True)
    if not marker.is_file():
        (directory / "training.yaml").write_text(config_text)
        log_path = directory / "export.log"
        print(f"Exporting checkpoint on CPU. Log: {log_path}", flush=True)
        with log_path.open("w") as log:
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--export-worker", str(directory)],
                cwd=ROOT / "active-adaptation", stdout=log, stderr=subprocess.STDOUT,
            )
        if result.returncode:
            print("\n".join(log_path.read_text(errors="replace").splitlines()[-25:]), file=sys.stderr)
            raise RuntimeError(f"Checkpoint export failed; full log: {log_path}")
    metadata = json.loads(marker.read_text())
    model = existing_file(directory / metadata["policy"])
    existing_file(model.with_suffix(".yaml"))
    return model, metadata["env_dt"], metadata["sim_dt"]


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--model", choices=PRESETS, help="Downloaded model preset (default: beyondmimic)")
    source.add_argument("--load_model", "--checkpoint", dest="load_model", help="Training .pt or exported .onnx (with sibling .yaml)")
    parser.add_argument("--config", help="Full training YAML for .pt; auto-detect play_local.yaml/config.yaml beside checkpoint")
    parser.add_argument("--motion", help="One any4hdmi .npz reference clip, under a dataset with manifest.json")
    parser.add_argument("--duration", type=float, default=60.0, help="Total simulation seconds, including loops; 0 = unlimited GUI (default: 60)")
    parser.add_argument("--headless", action="store_true", help="Run without rendering or real-time throttling")
    parser.add_argument("--once", action="store_true", help="Exit after the motion finishes instead of looping")
    parser.add_argument("--start-paused", action="store_true", help="Wait for P in the viewer before simulating")
    parser.add_argument("--initial-pause", type=float, default=0.0, help="Seconds holding reference frame 0 with policy/physics active")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reexport", action="store_true", help="Regenerate cached ONNX from the checkpoint")
    parser.add_argument("--export-only", action="store_true", help="Print the export paths and exit")
    parser.add_argument("--trajectory", help="Optional .npz trajectory recording (use outputs/)")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = make_parser()
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration) or args.duration < 0:
        parser.error("--duration must be finite and >= 0")
    if not math.isfinite(args.initial_pause) or args.initial_pause < 0:
        parser.error("--initial-pause must be finite and >= 0")
    if args.headless and (args.start_paused or (args.duration == 0 and not args.once)):
        parser.error("Headless runs require a duration > 0 or --once, and cannot start paused")
    setup_paths()
    try:
        preset = None if args.load_model else (args.model or "beyondmimic")
        if preset:
            checkpoint_name, dataset, clip = PRESETS[preset]
            checkpoint = existing_file(BUNDLE / "models" / preset / checkpoint_name)
            motion = existing_file(args.motion or ROOT / "any4hdmi/output/mini3" / dataset / "motions" / clip)
        else:
            checkpoint = existing_file(args.load_model)
            if not args.motion:
                raise ValueError("Supply --motion with a custom --load_model/--checkpoint")
            motion = existing_file(args.motion)
        if motion.suffix != ".npz":
            raise ValueError("--motion must select a single .npz clip")
        motion_dataset(motion)

        if checkpoint.suffix == ".onnx":
            model, env_dt, sim_dt = checkpoint, 0.02, 0.002
            existing_file(model.with_suffix(".yaml"))
            if args.config or args.reexport:
                raise ValueError("--config/--reexport apply only to training checkpoints")
        elif checkpoint.suffix == ".pt":
            config = args.config
            if config is None:
                config = next((path for name in ("play_local.yaml", "config.yaml")
                               if (path := checkpoint.parent / name).is_file()), None)
            if config is None:
                raise ValueError("No full training YAML found beside checkpoint; supply --config")
            model, env_dt, sim_dt = prepare_export(checkpoint, existing_file(config), motion, args)
        else:
            raise ValueError("--load_model must be a training .pt or exported .onnx")
        # Deployment motion loading uses 50 Hz reference clips; do not silently
        # play a policy trained with another control rate at the wrong speed.
        if not math.isclose(env_dt, 0.02, abs_tol=1e-8):
            raise ValueError(f"Mini3 deployment requires a 50 Hz policy, got dt={env_dt}")
        print(f"Policy: {model}\nConfig: {model.with_suffix('.yaml')}\nMotion: {motion}", flush=True)
        if args.export_only:
            return
        from sim2real.sim_env.integrated_sim2sim import IntegratedSim2SimArgs
        from sim2real.sim_env.native_mimiclite import run_native

        runtime_args = IntegratedSim2SimArgs(
            policy_config=str(model.with_suffix(".yaml")), motion_path=str(motion),
            robot="mini3", env_dt=env_dt, sim_dt=sim_dt, headless=True,
            initial_pause_s=args.initial_pause, inference_backend="onnx-cpu", seed=args.seed,
            trajectory_output=args.trajectory, trajectory_policy_frames_only=True,
        )
        run_native(runtime_args, duration=args.duration, headless=args.headless,
                   loop=not args.once, start_paused=args.start_paused)
    except (ValueError, FileNotFoundError, RuntimeError, ImportError, FloatingPointError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--export-worker":
        setup_paths()
        export_worker(Path(sys.argv[2]))
    else:
        main()
