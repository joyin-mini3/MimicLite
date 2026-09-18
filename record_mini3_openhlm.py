#!/usr/bin/env python3
"""Collect successful Mini3 demonstrations and export an OpenHLM LeRobot v2.1 dataset."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
DEPS = ROOT / ".cache/mini3_vla_deps"
if DEPS.is_dir():
    sys.path.insert(0, str(DEPS))

import numpy as np

FPS = 50
CAMERAS = {"head_image_left": "head_rgb", "left_wrist_image": "left_gripper_rgb",
           "right_wrist_image": "right_gripper_rgb"}
COLORS = ("red", "green", "blue")
DEFAULT_OUTPUT = ROOT / "outputs/mini3_openhlm_recording"
REPO_ID = "local/mini3_pick_carry"


def task_text(color: str) -> str:
    if color not in COLORS:
        raise ValueError(f"Unknown target color: {color}")
    return f"Please put the {color} square from the tabletop into the basket."


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text())


def attempt_seed(base_seed: int, attempt: int) -> int:
    return int(np.random.SeedSequence([base_seed, attempt]).generate_state(1, dtype=np.uint64)[0])


def subprocess_env() -> dict:
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", LP_NUM_THREADS="2", MUJOCO_GL="egl",
               PYTHONUNBUFFERED="1")
    env["PYTHONPATH"] = str(DEPS) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def collect_attempt(output: Path, attempt: int, seed: int) -> Path:
    directory = output / "attempts" / f"attempt_{attempt:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    report = directory / "report.json"
    samples = directory / "control_samples.npz"
    if report.exists() and (not read_json(report).get("success") or samples.exists()):
        return directory
    if (directory / "episode_scene.xml").exists():
        # Keep a interrupted attempt for diagnosis; the next seed supplies a
        # replacement instead of overwriting partially recorded evidence.
        return directory
    command = [sys.executable, str(ROOT / "mini3_collect_episode.py"), "--seed", str(seed),
               "--output", str(directory), "--onnx-threads", "1"]
    with (directory / "collection.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, env=subprocess_env(), stdout=log,
                                stderr=subprocess.STDOUT, check=False)
    if not report.exists():
        write_json(directory / "collection_error.json", {"returncode": result.returncode,
                   "seed": seed, "log": str(directory / "collection.log")})
    return directory


def quality_check(directory: Path, maximum_penetration: float) -> tuple[bool, dict]:
    report_path = directory / "report.json"
    if not report_path.exists():
        return False, {"reason": "collector_error"}
    report = read_json(report_path)
    contacts = report["arm_collision_monitor"]["unexpected_contacts"]
    penetration = max((item["maximum_penetration_m"] for item in contacts), default=0.)
    details = {"reason": report.get("failure"), "success": report["success"],
               "maximum_unexpected_penetration_m": penetration,
               "unexpected_contact_pairs": len(contacts)}
    if not report["success"] or not report["cube_settled_inside_basket"]:
        return False, details
    if report["policy_output_override_max"] != 0 or penetration > maximum_penetration:
        details["reason"] = "policy_override_or_contact_quality_limit"
        return False, details
    path = directory / "control_samples.npz"
    if not path.exists():
        details["reason"] = "missing_synchronized_samples"
        return False, details
    with np.load(path) as samples:
        for name in samples.files:
            values = samples[name]
            if name == "tool_target":
                values = values[samples["tool_target_active"]]
            if np.issubdtype(values.dtype, np.number) and not np.isfinite(values).all():
                details["reason"] = "nonfinite_samples:" + name
                return False, details
        times = samples["time"]
        if (not np.all(samples["action_executed"]) or not np.all(samples["executed_substeps"] == 10)
                or not np.allclose(samples["next_time"] - times, 1 / FPS, atol=1e-8)):
            details["reason"] = "incomplete_control_action"
            return False, details
        if len(times) < 100 or abs(times[0]) > 1e-8 or not np.allclose(np.diff(times), 1 / FPS, atol=1e-8):
            details["reason"] = "invalid_control_timestamps"
            return False, details
        details["frames"] = len(times)
    details["reason"] = "accepted"
    return True, details


def vector_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    return {"min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist(),
            "mean": values.mean(axis=0).tolist(), "std": values.std(axis=0).tolist(),
            "count": [len(values)]}


def hf_features(dimension: int) -> dict:
    features = {key: {"_type": "Image"} for key in CAMERAS}
    features.update({key: {"feature": {"dtype": "float32", "_type": "Value"},
                           "length": dimension, "_type": "Sequence"} for key in ("state", "actions")})
    features["timestamp"] = {"dtype": "float32", "_type": "Value"}
    features.update({key: {"dtype": "int64", "_type": "Value"} for key in
                     ("frame_index", "episode_index", "index", "task_index")})
    return features


def export_episode(directory: Path, dataset: Path, episode: int, start_index: int) -> dict:
    """Render saved pre-action poses without advancing or perturbing physics."""
    import mujoco
    from OpenGL import GL
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image
    from mini3_vla_features import extract_features

    destination = dataset / "data" / f"chunk-{episode // 1000:03d}" / f"episode_{episode:06d}.parquet"
    metadata_path = dataset / ".episode_meta" / f"episode_{episode:06d}.json"
    if destination.exists() and metadata_path.exists():
        existing = read_json(metadata_path)
        if (existing["episode_index"] != episode or existing["start_index"] != start_index
                or existing["source_attempt"] != str(directory.resolve())):
            raise ValueError("Existing episode has a different source or global index")
        if existing["sha256"] != hashlib.sha256(destination.read_bytes()).hexdigest():
            raise ValueError("Existing episode checksum does not match; refusing silent reuse")
        return existing
    with np.load(directory / "control_samples.npz") as archive:
        samples = {name: archive[name] for name in archive.files}
    metadata = read_json(directory / "control_metadata.json")
    state, actions = extract_features(samples, metadata)
    count, dimension = state.shape
    if actions.shape != state.shape or not np.isfinite(state).all() or not np.isfinite(actions).all():
        raise ValueError("Invalid state/action feature shapes or values")
    scene = mujoco.MjModel.from_xml_path(str(directory / "episode_scene.xml"))
    scene.vis.map.znear = min(scene.vis.map.znear, .002 / scene.stat.extent)
    scene.vis.quality.offsamples = 0
    data = mujoco.MjData(scene)
    images = {key: [] for key in CAMERAS}
    moments = {key: {"sum": np.zeros(3), "sum_sq": np.zeros(3),
                     "min": np.full(3, np.inf), "max": np.full(3, -np.inf)} for key in CAMERAS}
    renderer = mujoco.Renderer(scene, height=240, width=320)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = False
    renderer_name = GL.glGetString(GL.GL_RENDERER).decode()
    try:
        for frame in range(count):
            data.qpos[:] = samples["qpos"][frame]
            data.qvel[:] = samples["qvel"][frame]
            data.time = float(samples["time"][frame])
            mujoco.mj_forward(scene, data)
            for key, camera in CAMERAS.items():
                renderer.update_scene(data, camera=camera)
                rgb = Image.fromarray(renderer.render()).resize((224, 224), Image.Resampling.LANCZOS)
                buffer = io.BytesIO()
                rgb.save(buffer, format="JPEG", quality=95, subsampling=0)
                encoded = buffer.getvalue()
                images[key].append({"bytes": encoded, "path": None})
                # Metadata statistics describe the decoded stored images.
                pixels = np.asarray(Image.open(io.BytesIO(encoded)), dtype=np.float64) / 255.
                stats = moments[key]
                stats["sum"] += pixels.sum(axis=(0, 1))
                stats["sum_sq"] += np.square(pixels).sum(axis=(0, 1))
                stats["min"] = np.minimum(stats["min"], pixels.min(axis=(0, 1)))
                stats["max"] = np.maximum(stats["max"], pixels.max(axis=(0, 1)))
            if frame % 250 == 0:
                print(f"episode={episode} rendered={frame}/{count} renderer={renderer_name}", flush=True)
    finally:
        renderer.close()
    episode_info = read_json(directory / "episode.json")
    color = episode_info["target_color"]
    arrays = {"state": state, "actions": actions,
              "timestamp": np.arange(count, dtype=np.float32) / FPS,
              "frame_index": np.arange(count, dtype=np.int64),
              "episode_index": np.full(count, episode, dtype=np.int64),
              "index": np.arange(start_index, start_index + count, dtype=np.int64),
              "task_index": np.full(count, COLORS.index(color), dtype=np.int64)}
    columns = {key: pa.array(values, type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]))
               for key, values in images.items()}
    columns.update({key: pa.array(values.tolist(), type=pa.list_(pa.float32(), dimension))
                    for key, values in arrays.items() if key in ("state", "actions")})
    columns.update({key: pa.array(values) for key, values in arrays.items() if key not in ("state", "actions")})
    table = pa.table(columns)
    table = table.replace_schema_metadata({b"huggingface": json.dumps(
        {"info": {"features": hf_features(dimension)}}).encode()})
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".parquet.tmp")
    pq.write_table(table, temporary, compression="zstd", row_group_size=100)
    temporary.replace(destination)
    stats = {key: vector_stats(values) for key, values in arrays.items()}
    pixel_count = count * 224 * 224
    for key, values in moments.items():
        mean = values["sum"] / pixel_count
        std = np.sqrt(np.maximum(0., values["sum_sq"] / pixel_count - mean**2))
        stats[key] = {"mean": mean[:, None, None].tolist(), "std": std[:, None, None].tolist(),
                      "min": values["min"][:, None, None].tolist(),
                      "max": values["max"][:, None, None].tolist(), "count": [count]}
    result = {"episode_index": episode, "start_index": start_index, "length": count,
              "source_attempt": str(directory.resolve()), "seed": episode_info["seed"],
              "target_color": color, "tasks": [task_text(color)], "stats": stats,
              "renderer": renderer_name, "parquet": str(destination.resolve()),
              "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}
    write_json(metadata_path, result)
    return result


def export_subprocess(directory: Path, dataset: Path, episode: int, start_index: int) -> dict:
    log_path = directory / "export.log"
    with log_path.open("w") as log:
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "export",
            "--attempt", str(directory), "--dataset", str(dataset), "--episode", str(episode),
            "--start-index", str(start_index)], cwd=ROOT, env=subprocess_env(), stdout=log,
            stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"Export failed for episode {episode}; see {log_path}")
    return read_json(dataset / ".episode_meta" / f"episode_{episode:06d}.json")


def finalize(dataset: Path, expected: int) -> dict:
    from mini3_vla_features import FEATURE_NAMES, FEATURE_UNITS
    episodes = [read_json(dataset / ".episode_meta" / f"episode_{index:06d}.json")
                for index in range(expected)]
    total = sum(item["length"] for item in episodes)
    features = {key: {"dtype": "image", "shape": [224, 224, 3],
                       "names": ["height", "width", "channels"]} for key in CAMERAS}
    features.update({key: {"dtype": "float32", "shape": [len(FEATURE_NAMES)],
                           "names": list(FEATURE_NAMES)} for key in ("state", "actions")})
    features["timestamp"] = {"dtype": "float32", "shape": [1], "names": None}
    features.update({key: {"dtype": "int64", "shape": [1], "names": None} for key in
                     ("frame_index", "episode_index", "index", "task_index")})
    info = {"codebase_version": "v2.1", "robot_type": "mini3_27dof_parallel_grippers",
            "total_episodes": expected, "total_frames": total, "total_tasks": 3,
            "total_videos": 0, "total_chunks": math.ceil(expected / 1000), "chunks_size": 1000,
            "fps": FPS, "splits": {"train": f"0:{expected}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": None, "features": features}
    write_json(dataset / "meta/info.json", info)
    for name, rows in (("tasks", [{"task_index": i, "task": task_text(color)} for i, color in enumerate(COLORS)]),
                       ("episodes", [{key: item[key] for key in ("episode_index", "tasks", "length")}
                                     for item in episodes]),
                       ("episodes_stats", [{"episode_index": item["episode_index"], "stats": item["stats"]}
                                           for item in episodes])):
        path = dataset / "meta" / (name + ".jsonl")
        temporary = path.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
        temporary.replace(path)
    write_json(dataset / "meta/mini3_schema.json", {
        "feature_names": list(FEATURE_NAMES), "units": list(FEATURE_UNITS),
        "fps": FPS, "observation_action_alignment": "pre-action observation at t; command applied over [t,t+0.02)",
        "action_interface": "Mini3 hierarchical reference, not G1 motor commands",
        "images": {"cameras": CAMERAS, "render_size": [320, 240], "stored_size": [224, 224],
                   "encoding": "JPEG quality=95 subsampling=0", "shadows": False, "reflections": False,
                   "method": "offline rendering of exact pre-action qpos, no physics stepping"},
        "teacher": "MimicLite plus added-joint planning and privileged object state; no object poses in state",
        "deployment": "Requires a Mini3 VLA-reference-to-MimicLite adapter; does not use G1 deployment unmodified",
    })
    write_json(dataset / "meta/collection_manifest.json", [{key: value for key, value in episode.items()
               if key != "stats"} for episode in episodes])
    # Explicit episode split, without claiming frame-wise splits are independent.
    validation = list(range(max(1, expected - max(1, expected // 10)), expected)) if expected > 1 else []
    write_json(dataset / "meta/recommended_split.json", {
        "train_episodes": [index for index in range(expected) if index not in validation],
        "validation_episodes": validation, "unit": "whole_episode"})
    return info


def run_collection(args) -> None:
    output = args.output.resolve()
    dataset = output / "lerobot" / REPO_ID
    configuration = {"episodes": args.episodes, "base_seed": args.seed,
                     "maximum_unexpected_penetration_m": args.max_penetration,
                     "dataset": str(dataset), "fps": FPS}
    config_path = output / "collection_config.json"
    if config_path.exists():
        if not args.resume:
            raise ValueError("Output already exists; use --resume to continue it")
        if read_json(config_path) != configuration:
            raise ValueError("Resume configuration differs from the original collection")
    else:
        write_json(config_path, configuration)
    source_files = [ROOT / name for name in ("mini3_collect_episode.py", "mini3_pick_carry_policy.py",
                    "mini3_policy_task_reference.py", "mini3_randomized_task.py", "mini3_vla_features.py",
                    "record_mini3_openhlm.py")]
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    hash_path = output / "source_hashes.json"
    if hash_path.exists() and read_json(hash_path) != hashes:
        raise ValueError("Collection code changed; use a new output to avoid silently mixing versions")
    write_json(hash_path, hashes)
    accepted, attempts, exported = [], [], []
    next_attempt = next_submit = total_frames = 0
    with ThreadPoolExecutor(max_workers=args.workers) as collectors, ThreadPoolExecutor(
            max_workers=args.render_workers) as exporters:
        pending = {}
        render_jobs = []
        while len(accepted) < args.episodes:
            while len(pending) < args.workers and next_submit < args.max_attempts:
                number = next_submit
                pending[number] = collectors.submit(collect_attempt, output, number, attempt_seed(args.seed, number))
                next_submit += 1
            if not pending:
                raise RuntimeError("Attempt limit reached before collecting enough accepted episodes")
            try:
                directory = pending[next_attempt].result(timeout=10)
            except TimeoutError:
                print(f"Collecting: accepted={len(accepted)}/{args.episodes}, next_attempt={next_attempt}", flush=True)
                continue
            del pending[next_attempt]
            ok, quality = quality_check(directory, args.max_penetration)
            record = {"attempt": next_attempt, "seed": attempt_seed(args.seed, next_attempt),
                      "directory": str(directory), "accepted": ok, **quality}
            attempts.append(record)
            next_attempt += 1
            if ok:
                episode = len(accepted)
                record["episode_index"] = episode
                accepted.append(record)
                render_jobs.append(exporters.submit(export_subprocess, directory, dataset, episode, total_frames))
                total_frames += quality["frames"]
            for job in render_jobs:
                if job.done() and job.exception() is not None:
                    raise job.exception()
            exported = [i for i, job in enumerate(render_jobs) if job.done()]
            write_json(output / "collection_status.json", {"accepted": len(accepted), "requested": args.episodes,
                "attempted": next_attempt, "exported": len(exported), "frames": total_frames,
                "attempts": attempts, "complete": False})
            print(f"accepted={len(accepted)}/{args.episodes} attempted={next_attempt} exported={len(exported)} "
                  f"last={record['reason']}", flush=True)
        for index, job in enumerate(render_jobs):
            while True:
                try:
                    job.result(timeout=10)
                    break
                except TimeoutError:
                    completed = sum(future.done() for future in render_jobs)
                    print(f"Rendering: {completed}/{args.episodes} episodes complete", flush=True)
            write_json(output / "collection_status.json", {"accepted": len(accepted), "requested": args.episodes,
                "attempted": next_attempt, "exported": sum(future.done() for future in render_jobs),
                "frames": total_frames, "attempts": attempts, "complete": False})
    info = finalize(dataset, args.episodes)
    write_json(output / "collection_status.json", {"accepted": len(accepted), "requested": args.episodes,
               "attempted": next_attempt, "exported": args.episodes, "frames": total_frames,
               "attempts": attempts, "complete": True, "dataset": str(dataset)})
    print(json.dumps({"complete": True, "dataset": str(dataset), "episodes": args.episodes,
                      "frames": info["total_frames"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect")
    collect.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    collect.add_argument("--episodes", type=int, default=200)
    collect.add_argument("--seed", type=int, default=20260918)
    collect.add_argument("--workers", type=int, default=6)
    collect.add_argument("--render-workers", type=int, default=2)
    collect.add_argument("--max-attempts", type=int, default=600)
    collect.add_argument("--max-penetration", type=float, default=.0005)
    collect.add_argument("--resume", action="store_true")
    export = commands.add_parser("export")
    export.add_argument("--attempt", type=Path, required=True)
    export.add_argument("--dataset", type=Path, required=True)
    export.add_argument("--episode", type=int, required=True)
    export.add_argument("--start-index", type=int, required=True)
    finish = commands.add_parser("finalize")
    finish.add_argument("--dataset", type=Path, required=True)
    finish.add_argument("--episodes", type=int, required=True)
    args = parser.parse_args()
    if args.command == "collect":
        if args.episodes <= 0 or args.workers <= 0 or args.render_workers <= 0 or args.max_attempts < args.episodes:
            parser.error("Counts must be positive and max-attempts must cover requested episodes")
        if args.seed < 0 or not math.isfinite(args.max_penetration) or args.max_penetration < 0:
            parser.error("Seed and finite penetration limit must be nonnegative")
        run_collection(args)
    elif args.command == "export":
        export_episode(args.attempt, args.dataset, args.episode, args.start_index)
    else:
        finalize(args.dataset, args.episodes)


if __name__ == "__main__":
    main()
