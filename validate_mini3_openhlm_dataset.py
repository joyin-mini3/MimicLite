#!/usr/bin/env python3
"""Audit every Mini3 dataset row and optionally compute full-data OpenHLM statistics."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / ".cache/mini3_vla_deps"))

import numpy as np
from PIL import Image
import pyarrow.parquet as pq

from mini3_vla_features import FEATURE_NAMES, extract_features
from record_mini3_openhlm import CAMERAS, COLORS, DEFAULT_OUTPUT, FPS, REPO_ID, quality_check, task_text, write_json


def action_windows(actions: np.ndarray, horizon: int = 50) -> np.ndarray:
    """Future chunks with exactly LeRobot's repeated-last-action boundary rule."""
    padded = np.concatenate((actions, np.repeat(actions[-1:], horizon - 1, axis=0)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, horizon, axis=0)
    return np.moveaxis(windows, -1, 1)


def validate(dataset: Path, report_path: Path, norm_output: Path | None = None) -> dict:
    dataset = dataset.resolve()
    info = json.loads((dataset / "meta/info.json").read_text())
    manifest = json.loads((dataset / "meta/collection_manifest.json").read_text())
    tasks = [json.loads(line) for line in (dataset / "meta/tasks.jsonl").read_text().splitlines()]
    assert tasks == [{"task_index": index, "task": task_text(color)} for index, color in enumerate(COLORS)]
    assert info["codebase_version"] == "v2.1" and info["fps"] == FPS
    assert len(manifest) == info["total_episodes"]
    dimension = len(FEATURE_NAMES)
    assert info["features"]["actions"]["names"] == list(FEATURE_NAMES)
    summaries = []
    colors = Counter()
    seeds = set()
    global_index = parquet_bytes = decoded_images = 0
    max_contact = 0.
    statistics = None
    if norm_output is not None:
        from mini3_openhlm_training import DEFAULT_OPENHLM, prepare_imports
        prepare_imports(DEFAULT_OPENHLM, dataset.parent.parent, validation=True)
        from openpi.shared import normalize
        statistics = {key: normalize.RunningStats() for key in ("state", "actions")}
    for episode_index, entry in enumerate(manifest):
        source = Path(entry["source_attempt"])
        accepted, quality = quality_check(source, .0005)
        assert accepted, (episode_index, quality)
        assert entry["episode_index"] == episode_index and entry["start_index"] == global_index
        assert entry["seed"] not in seeds, "Repeated seed"
        seeds.add(entry["seed"])
        colors[entry["target_color"]] += 1
        max_contact = max(max_contact, quality["maximum_unexpected_penetration_m"])
        parquet = dataset / "data" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.parquet"
        assert hashlib.sha256(parquet.read_bytes()).hexdigest() == entry["sha256"], parquet
        parquet_bytes += parquet.stat().st_size
        columns = ("state", "actions", "timestamp", "frame_index", "episode_index", "index", "task_index")
        table = pq.read_table(parquet, columns=list(columns))
        count = len(table)
        assert count == entry["length"] == quality["frames"]
        arrays = {key: table[key].combine_chunks().to_numpy(zero_copy_only=False)
                  for key in columns if key not in ("state", "actions")}
        for key in ("state", "actions"):
            array = table[key].combine_chunks()
            assert array.type.list_size == dimension
            arrays[key] = array.values.to_numpy().reshape(count, dimension)
            assert arrays[key].dtype == np.float32 and np.isfinite(arrays[key]).all()
        np.testing.assert_array_equal(arrays["frame_index"], np.arange(count))
        np.testing.assert_array_equal(arrays["index"], np.arange(global_index, global_index + count))
        assert np.all(arrays["episode_index"] == episode_index)
        assert np.all(arrays["task_index"] == COLORS.index(entry["target_color"]))
        np.testing.assert_allclose(arrays["timestamp"], np.arange(count) / FPS, atol=2e-6, rtol=0)
        with np.load(source / "control_samples.npz") as raw:
            state, actions = extract_features(raw, json.loads((source / "control_metadata.json").read_text()))
            np.testing.assert_array_equal(arrays["state"], state)
            np.testing.assert_array_equal(arrays["actions"], actions)
            np.testing.assert_array_equal(raw["qpos"][1:], raw["next_qpos"][:-1])
        image_table = pq.read_table(parquet, columns=list(CAMERAS))
        for index in (0, count // 2, count - 1):
            for key in CAMERAS:
                image = Image.open(io.BytesIO(image_table[key][index].as_py()["bytes"]))
                image.load()
                assert image.size == (224, 224) and image.mode == "RGB"
                assert np.asarray(image).max() > 0
                decoded_images += 1
        if statistics is not None:
            # Same numerical inputs and boundary padding as OpenHLM's loader;
            # image decoding is unnecessary to accumulate state/action moments.
            windows = action_windows(arrays["actions"])
            for start in range(0, count, 256):
                statistics["state"].update(arrays["state"][start:start + 256].astype(np.float64))
                statistics["actions"].update(windows[start:start + 256].astype(np.float64))
        summaries.append({"episode_index": episode_index, "color": entry["target_color"],
                          "frames": count, "maximum_unexpected_penetration_m": quality["maximum_unexpected_penetration_m"]})
        global_index += count
        print(f"validated={episode_index + 1}/{len(manifest)} frames={global_index}", flush=True)
    assert global_index == info["total_frames"]
    if statistics is not None:
        from openpi.shared import normalize
        result_stats = {key: value.get_statistics() for key, value in statistics.items()}
        normalize.save(norm_output, result_stats)
        # Read through the training project's actual serialization contract.
        loaded = normalize.load(norm_output)
        for key in ("state", "actions"):
            for value in (loaded[key].mean, loaded[key].std, loaded[key].q01, loaded[key].q99):
                assert value.shape == (dimension,) and np.isfinite(value).all()
    result = {"success": True, "episodes": len(manifest), "frames": global_index,
              "fps": FPS, "simulated_seconds": global_index / FPS, "state_dim": dimension,
              "action_dim": dimension, "color_counts": dict(colors), "unique_seeds": len(seeds),
              "parquet_bytes": parquet_bytes, "sample_images_decoded": decoded_images,
              "image_files_embedded": 3 * global_index,
              "every_row_matches_pre_action_raw": True, "every_parquet_sha256_verified": True,
              "maximum_unexpected_penetration_m": max_contact,
              "normalization_output": str(norm_output) if norm_output is not None else None,
              "normalization_method": ("OpenHLM RunningStats on every state and all 50-frame future action chunks, "
                                       "including repeated-final-action padding; RGB excluded from moment calculation")
                                       if statistics is not None else None,
              "episodes_detail": summaries}
    write_json(report_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_OUTPUT / "lerobot" / REPO_ID)
    parser.add_argument("--report", type=Path, default=DEFAULT_OUTPUT / "integrity_validation.json")
    parser.add_argument("--norm-output", type=Path)
    args = parser.parse_args()
    print(json.dumps({key: value for key, value in validate(args.dataset, args.report, args.norm_output).items()
                      if key != "episodes_detail"}, indent=2))


if __name__ == "__main__":
    main()
